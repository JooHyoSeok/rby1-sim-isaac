"""Robot-side building blocks for the control centre's executor service: seed pose, continuous visual-servo pick, place, home.

Runs in the simulator container (rby1_sdk is installed only there); servo_server.py wraps it in a UDP service. The control law of the
pick is the one of examples/vservo/hybrid_grasp.py (see its docstring for the reasoning); what is new here is that it can be aborted,
reports what it does, can be started from any hand pose, and that a held box can be put down at a chosen place.

NOT RUN YET: written while the simulator was off limits. The pure helpers (smoothstep, place_waypoints) are covered by test_offline.py,
everything that talks to the robot is not.
"""
import collections
import json
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/opt/rby1-sim-isaac/app")  # sim_gripper_bridge
sys.path.insert(0, "/tmp")                      # vision.py, servo.py, scene_config.py, gripper_example.py (docker cp'd here)
import scene_config as C  # noqa: E402

TOOL = 0.12          # fingertips lie this far beyond the ee frame along the tool axis (-z_ee)
RATE = 20.0          # streaming loop [Hz]
LATENCY = 0.12       # [s] from the instant of a camera image to the arrival of its measurement here
KP_LAT, KP_FWD = 1.2, 1.5
V_LAT_MAX, V_FWD_MAX = 0.08, 0.08
LAT_GATE = 0.02
LEAD_MAX = 0.06
DONE_REMAIN, DONE_LAT = 0.003, 0.002
SERVO_TIMEOUT = 40.0
LIFT = np.array([-0.06, 0.0, 0.12])
LIFT_TIME = 3.5
MOVE_SPEED = 0.10    # [m/s] position-controlled moves (place); duration = distance / speed, at least MOVE_MIN_TIME
MOVE_MIN_TIME = 1.0
HOVER = 0.06         # box bottom this far above the table while carrying it over the place position
R_FRONT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)  # tool axis forward, fingers open sideways
TORSO = np.array([0.0, 1.0, -2.0, 1.0, 0.0, 0.0])
READY = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 0.3, -1.57])
HOLDING_CLOSENESS = (0.2, 0.85)  # gripper closeness range of a closed hand with the 5 cm box between the fingers (about 0.5)


class TaskError(Exception):
    """A task cannot continue (explained in the message); the robot is left where it is."""


class Aborted(TaskError):
    pass


def smoothstep(s):
    """0 -> 1 with zero velocity at both ends."""
    s = min(1.0, max(0.0, s))
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def place_waypoints(tip_now, x, y, top):
    """Fingertip waypoints to put a held box down at (x, y) on a table of height `top`: carry it over the spot (box bottom
    HOVER above the table, or the current height if that is higher), lower it, and after the gripper opens back off."""
    cx = x + C.TIP_BEYOND_CENTRE
    hover_z = max(tip_now[2], top + C.BOX_SIZE[2] / 2 + HOVER)
    above = np.array([cx, y, hover_z])
    down = np.array([cx, y, top + C.BOX_SIZE[2] / 2 + 0.004])
    back = down + np.array([-0.10, 0.0, 0.0])
    return above, down, back


class Perception:
    """Latest message of perception_node.py (UDP 5102): stereo box pose and the wrist camera's red blobs."""

    def __init__(self, port=5102):
        self.n, self.t, self.data = 0, 0.0, None
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.bind(("127.0.0.1", port))
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while True:
            self.data = json.loads(self._rx.recv(65536))
            self.t = time.time()
            self.n += 1

    def wait_box(self, count=6, timeout=8.0, abort=None):
        """`count` fresh messages that contain a box pose from at least 3 landmarks."""
        n0, t0, out = self.n, time.time(), []
        while len(out) < count and time.time() - t0 < timeout:
            if abort is not None and abort.is_set():
                raise Aborted("aborted")
            if self.n != n0:
                n0 = self.n
                d = self.data
                if d and d.get("box") and d["box"]["n"] >= 3:
                    out.append(d)
            time.sleep(0.02)
        return out


class GraspRobot:
    def __init__(self, perception: Perception, event=lambda text: print(text, flush=True)):
        import rby1_sdk as rby
        from gripper_example import Gripper
        from sim_gripper_bridge import SimDynamixelBus
        import servo

        self.rby, self.servo, self.perception, self.event = rby, servo, perception, event
        self.abort_flag = threading.Event()
        self.phase, self.holding, self.seeded = "idle", False, False
        self.robot = rby.create_robot_a("127.0.0.1:50051")
        assert self.robot.connect(), "cannot reach app_main (is run_sdk.sh running?)"
        self.robot.power_on(".*")
        self.robot.servo_on(".*")
        self._recover()
        self.gripper = Gripper(SimDynamixelBus("sim"))
        assert self.gripper.initialize()
        self.gripper.homing()
        self.gripper.start()
        self.gripper.set_open()
        self.dyn = self.robot.get_dynamics()
        self.fk = self.dyn.make_state(["base", "ee_right"], rby.Model_A().robot_joint_names)
        self.left_hold = dict(zip([f"left_arm_{i}" for i in range(7)], np.array(self.robot.get_state().position)[15:22]))
        self.dbg = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stream = None
        self._fk_lock = threading.Lock()  # the status heartbeat and the running task both compute forward kinematics
        self.cmd_tip = self.tip()

    # ---- state -------------------------------------------------------------------------------------------------------
    def ee_pose(self):
        q = np.array(self.robot.get_state().position)
        with self._fk_lock:
            self.fk.set_q(q)
            self.dyn.compute_forward_kinematics(self.fk)
            return self.dyn.compute_transformation(self.fk, 0, 1)

    @staticmethod
    def tip_of(T_ee):
        return T_ee[:3, 3] + TOOL * (T_ee[:3, :3] @ np.array([0.0, 0.0, -1.0]))

    def tip(self):
        return self.tip_of(self.ee_pose())

    def closeness(self):
        """Right gripper: 0 open ... 1 closed (fully closed means nothing between the fingers)."""
        g = self.gripper
        enc = dict(g.bus.group_fast_sync_read_encoder(g.GRIPPER_IDS))
        return float(1.0 - (enc[1] - g.min_q[1]) / (g.max_q[1] - g.min_q[1]))

    def status(self):
        return {"phase": self.phase, "holding": self.holding, "seeded": self.seeded, "tip": self.tip().round(4).tolist(),
                "closeness": round(self.closeness(), 2)}

    def debug(self, **info):
        self.dbg.sendto(json.dumps(info).encode(), ("127.0.0.1", 5103))

    # ---- control plumbing --------------------------------------------------------------------------------------------
    def abort(self):
        self.abort_flag.set()

    def _check(self):
        if self.abort_flag.is_set():
            raise Aborted("aborted")
        state = str(self.robot.get_control_manager_state().state)
        if "Fault" in state:
            raise TaskError(f"the robot's control manager faulted ({state}); recovering")

    def _recover(self):
        self.robot.reset_fault_control_manager()
        assert self.robot.enable_control_manager()

    def _end_stream(self):
        if self.stream is not None:
            self.stream.cancel()
            self.stream = None

    def _stream_to(self, tip_xyz):
        """Replace the whole-body Cartesian target; the robot tracks it continuously (reached within 0.3 s, held for 1 s)."""
        rby = self.rby
        T = np.eye(4)
        T[:3, :3] = R_FRONT
        T[:3, 3] = np.asarray(tip_xyz) - TOOL * np.array([1.0, 0.0, 0.0])
        cart = (rby.CartesianCommandBuilder().set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1.0))
                .set_minimum_time(0.3).add_target("base", "ee_right", T, 0.3, 1.0, 1.0))
        for joint, q in self.left_hold.items():
            cart.add_joint_position_target(joint, float(q))
        rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyCommandBuilder().set_command(cart)))
        for attempt in (0, 1):
            if self.stream is None:
                self.stream = self.robot.create_command_stream(10)
            try:
                self.stream.send_command(rc, 200)
                return
            except RuntimeError as exc:
                if "expired" not in str(exc) or attempt:
                    raise
                self.stream = None   # a stream that sat idle (between two tasks) expires: open a new one and send again

    def _joint_move(self, body_builder, name):
        """Blocking joint-space move that can be aborted."""
        rby = self.rby
        self._end_stream()
        handler = self.robot.send_command(rby.RobotCommandBuilder().set_command(
            rby.ComponentBasedCommandBuilder().set_body_command(body_builder)), 10)
        while not handler.wait_for(100):
            if self.abort_flag.is_set():
                handler.cancel()
                raise Aborted("aborted")
        code = handler.get().finish_code
        if code != rby.RobotCommandFeedback.FinishCode.Ok:
            self._recover()
            raise TaskError(f"{name}: the robot reported {code}")

    def _tick(self, t_loop):
        time.sleep(max(0.0, 1.0 / RATE - (time.time() - t_loop)))

    def _move_tip(self, target, name, duration=None):
        """Position-controlled straight move of the fingertips along a smooth ramp, streamed."""
        start = self.cmd_tip.copy()
        target = np.asarray(target, float)
        if not C.tip_reachable(target):
            raise TaskError(f"{name}: target {np.round(target, 3).tolist()} is out of the arm's reach")
        duration = duration or max(MOVE_MIN_TIME, float(np.linalg.norm(target - start)) / MOVE_SPEED)
        t0 = time.time()
        while time.time() - t0 < duration:
            t_loop = time.time()
            self._check()
            self.cmd_tip = start + (target - start) * smoothstep((t_loop - t0) / duration)
            self._stream_to(self.cmd_tip)
            self._tick(t_loop)
        self.cmd_tip = target
        self._stream_to(target)

    def _hold(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            t_loop = time.time()
            self._check()
            self._stream_to(self.cmd_tip)
            self._tick(t_loop)

    def _run(self, name, fn, *args, **kw):
        """Run a task body: clears the abort flag, recovers after a fault or an abort, leaves the robot ready for the next task."""
        self.abort_flag.clear()
        self.phase = name
        try:
            return fn(*args, **kw)
        except TaskError:
            try:
                self._end_stream()
                self._recover()
                self.cmd_tip = self.tip()
            except Exception:  # noqa: BLE001 - recovery is best effort, the original error is what matters
                pass
            raise
        finally:
            self.phase = "idle"

    # ---- tasks -------------------------------------------------------------------------------------------------------
    def seed(self):
        """Torso squat and bent right arm: the start of every grasp (the straight arm is a Cartesian singularity)."""
        def body():
            rby = self.rby
            self.event("seed pose: torso squat")
            self._joint_move(rby.BodyComponentBasedCommandBuilder().set_torso_command(
                rby.TorsoCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(6.0).set_position(TORSO))), "torso")
            self.event("seed pose: right arm")
            self._joint_move(rby.BodyComponentBasedCommandBuilder().set_right_arm_command(
                rby.ArmCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(8.0).set_position(READY))), "arm")
            self.cmd_tip = self.tip()
            self.seeded = True
        self._run("seed", body)

    def home(self):
        def body():
            if self.holding:
                raise TaskError("the hand holds the box: place it first")
            self.gripper.set_open()
            self.seeded = False
            self.seed()
        self._run("home", body)

    def set_gripper(self, closeness):
        self.gripper.set_normalized(float(closeness))
        if closeness < 0.5:
            self.holding = False

    def pick(self, use_image=True, bias=(0.0, 0.0, 0.0)):
        """Visual-servo to the box, close the gripper, lift. Starts from the current hand pose (seeds first if needed)."""
        def body():
            if self.holding:
                raise TaskError("already holding the box")
            if not self.seeded:
                self.seed()
                self.abort_flag.clear()
                self.phase = "pick"
            self.gripper.set_open()
            est = self.perception.wait_box(6, abort=self.abort_flag)
            if not est:
                raise TaskError("no box pose from the stereo rig (perception node running? 3 landmarks visible?)")
            box_R = np.array(est[-1]["box"]["R"])
            box_t = np.mean([m["box"]["t"] for m in est], axis=0) + np.asarray(bias)
            if not C.tip_reachable(box_t + np.array([C.TIP_BEYOND_CENTRE, 0.0, 0.0])):
                raise TaskError(f"the box at {np.round(box_t, 3).tolist()} is out of the arm's reach")
            self.event(f"stereo box estimate {np.round(box_t, 3).tolist()} ({est[-1]['box']['n']} landmarks)")
            self._servo(box_R, box_t, use_image, np.asarray(bias, float))
            self.event("grasp")
            self.debug(stage="grasp", text=["grasp"])
            self.gripper.set_closed()
            self._hold(1.8)
            c = self.closeness()
            if not (HOLDING_CLOSENESS[0] < c < HOLDING_CLOSENESS[1]):
                self.gripper.set_open()
                raise TaskError(f"grasp missed (gripper closeness {c:.2f}: nothing between the fingers)")
            self.holding = True
            self.event("lift")
            self.debug(stage="lift", text=["lift"])
            base = self.cmd_tip.copy()
            t0 = time.time()
            while time.time() - t0 < LIFT_TIME:
                t_loop = time.time()
                self._check()
                self.cmd_tip = base + LIFT * smoothstep((t_loop - t0) / LIFT_TIME)
                self._stream_to(self.cmd_tip)
                self._tick(t_loop)
            self.cmd_tip = base + LIFT
            self._hold(0.3)
            self.debug(stage="done", text=["done"])
        self._run("pick", body)

    def place(self, x, y, top=C.TABLE_TOP):
        """Carry the held box over (x, y), lower it onto the table, let go, back off."""
        def body():
            if not self.holding:
                raise TaskError("nothing in the hand: pick first")
            above, down, back = place_waypoints(self.tip(), x, y, top)
            self.cmd_tip = self.tip()
            self.event(f"place at ({x:.3f}, {y:.3f}): carry")
            self.debug(stage="place: carry", text=[f"place at ({x:.3f}, {y:.3f})", "carry"])
            self._move_tip(above, "carry")
            self.event("place: lower")
            self.debug(stage="place: lower", text=["place", "lower"])
            self._move_tip(down, "lower", duration=3.0)
            self.event("place: release")
            self.gripper.set_open()
            self.holding = False
            self._hold(1.5)
            self.event("place: back off")
            self._move_tip(back, "back off", duration=2.0)
            self.debug(stage="done", text=["done"])
        self._run("place", body)

    # ---- the continuous servo (examples/vservo/hybrid_grasp.py) -----------------------------------------------------------
    def _servo(self, box_R, box_t, use_image, bias):
        servo, perception = self.servo, self.perception
        hist = collections.deque(maxlen=int(3 * RATE))

        def pose_at(t):
            return min(hist, key=lambda h: abs(h[0] - t))[1]

        T_ee = self.ee_pose()
        self.cmd_tip = self.tip_of(T_ee)
        c_world, src, view = None, "-", {}
        last_n, ok_ticks, t_start, tick, dt = perception.n, 0, time.time(), 0, 1.0 / RATE
        while time.time() - t_start < SERVO_TIMEOUT:
            t_loop = time.time()
            self._check()
            T_ee = self.ee_pose()
            hist.append((t_loop, T_ee))
            if perception.n != last_n:                       # a new perception message: update the (static) box measurement
                last_n, data, t_msg = perception.n, perception.data, perception.t
                if use_image:
                    T_then = pose_at(t_msg - LATENCY)
                    blobs = np.array(data["wrist"]) if data["wrist"] else np.zeros((0, 3))
                    centre, Z, n_seen = servo.box_in_ee(blobs, servo.cam_pose(T_then), box_R, box_t)
                    if centre is not None and n_seen >= 2:
                        c_world, src = (T_then @ np.r_[centre, 1])[:3], "image"
                        to_cam = np.linalg.inv(servo.CAM_IN_EE)
                        pred = servo.predicted_landmarks(servo.cam_pose(T_then), box_R, box_t)[0]
                        px_of = lambda p: (servo.K @ p)[:2] / p[2]  # noqa: E731
                        view = dict(pred=pred.tolist(),
                                    matched=[[k, float(p[0]), float(p[1])] for k, p in servo.match_landmarks(blobs, pred)],
                                    centre_px=px_of((to_cam @ np.r_[centre, 1])[:3]).tolist(),
                                    goal_px=px_of((to_cam @ np.r_[0.0, 0.0, centre[2], 1])[:3]).tolist(), depth_cm=Z * 100)
                if (c_world is None or not use_image) and data.get("box"):   # no usable image (yet): the stereo measurement
                    c_world, src = np.array(data["box"]["t"]) + bias, "stereo"
            if c_world is not None:
                c_now = (np.linalg.inv(T_ee) @ np.r_[c_world, 1])[:3]  # box centre in the hand frame now
                lat, remaining = c_now[:2], -c_now[2] - (TOOL - C.TIP_BEYOND_CENTRE)
                R = T_ee[:3, :3]
                v_lat = R @ np.r_[KP_LAT * lat, 0.0]
                if np.linalg.norm(v_lat) > V_LAT_MAX:
                    v_lat *= V_LAT_MAX / np.linalg.norm(v_lat)
                gate = float(np.clip(1.0 - np.hypot(*lat) / LAT_GATE, 0.0, 1.0))
                v_fwd = 0.0 if remaining < DONE_REMAIN else min(V_FWD_MAX, KP_FWD * remaining) * gate
                self.cmd_tip = self.cmd_tip + (v_lat + v_fwd * (R @ np.array([0.0, 0.0, -1.0]))) * dt
                lead = self.cmd_tip - self.tip_of(T_ee)
                if np.linalg.norm(lead) > LEAD_MAX:          # no wind-up when the robot cannot follow
                    self.cmd_tip = self.tip_of(T_ee) + lead * LEAD_MAX / np.linalg.norm(lead)
                ok_ticks = ok_ticks + 1 if (remaining < DONE_REMAIN and np.hypot(*lat) < DONE_LAT) else 0
                if tick % 2 == 0:
                    self.debug(stage=f"continuous servo ({src})", err_mm=[lat[0] * 1000, lat[1] * 1000], remaining_mm=remaining * 1000,
                               text=[f"continuous servo, measured by the {src}",
                                     f"lateral error {lat[0] * 1000:+.1f} / {lat[1] * 1000:+.1f} mm, {remaining * 1000:.0f} mm to go"], **view)
                if ok_ticks >= int(0.25 * RATE):
                    return
            self._stream_to(self.cmd_tip)
            tick += 1
            self._tick(t_loop)
        raise TaskError("the servo did not arrive within the time limit")
