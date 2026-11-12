"""Robot-side building blocks for the control centre's executor service: seed pose, continuous visual-servo pick, place, stack, home.

Runs in the simulator container (rby1_sdk is installed only there); servo_server.py wraps it in a UDP service. The control law of the
pick is the one of examples/vservo/hybrid_grasp.py (see its docstring for the reasoning); what is new here is that it can be aborted,
reports what it does, can be started from any hand pose, and that a held box can be put down at a chosen place or on another box
(stereo-relative alignment, see _align_over). Three boxes are told apart by the colour of their landmarks.

pick / place / pick_place / home / place_on / stack were run in the simulator. NOT RUN YET: unstack. The pure
helpers (smoothstep, place_waypoints, scene_config) are covered by test_offline.py, everything that talks to the robot is not.
"""
import collections
import contextlib
import json
import socket
import sys
import threading
import time
import types

import numpy as np

sys.path.insert(0, "/opt/rby1-sim-isaac/app")  # sim_gripper_bridge
sys.path.insert(0, "/tmp")                      # vision.py, servo.py, scene_config.py, gripper_example.py, kinematics.py, safety.py (docker cp'd here)
import base_motion as B  # noqa: E402
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
HOVER = 0.06         # box bottom this far above the surface (table or stack) while carrying it over the place position
ALIGN_KP = 1.5       # [1/s] stack alignment: velocity = KP * offset of the held box from the box below, seen by the stereo rig
ALIGN_V_MAX = 0.03   # [m/s]
ALIGN_TOL = 0.003    # [m] aligned when the offset stays under this for 0.5 s
ALIGN_TIMEOUT = 12.0 # [s]
STACK_Z_TOL = 0.015  # [m] a placed box counts as stacked when its centre is within this of one box height above the lower one
STACK_XY_TOL = 0.02  # [m] ... and this close above it (x and y)
R_FRONT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)  # tool axis forward, fingers open sideways
TORSO = np.array([0.0, 1.0, -2.0, 1.0, 0.0, 0.0])
READY = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 0.3, -1.57])      # the left arm takes the same values (checked with forward kinematics: same hand orientation,
PARK = np.zeros(7)                                                # camera on the back of the hand); the arm that is not working hangs at its side
GRIP_TORQUE = 20.0   # gripper current limit in the current-based position mode (the library's loop uses 5)
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
        self.base_pose = lambda: (0.0, 0.0, 0.0)   # set by GraspRobot: where the base stands in the world; the boxes are handed on in the BASE frame
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.bind(("127.0.0.1", port))
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while True:
            data = json.loads(self._rx.recv(65536))
            try:
                self._to_base(data)
            except Exception as exc:  # noqa: BLE001 - one bad message (or the robot not ready yet) must not stop the listener
                print(f"perception: could not convert to the base frame: {exc}", flush=True)
            self.data = data
            self.t = time.time()
            self.n += 1

    def _to_base(self, data):
        """The stereo rig sees the boxes in the world; everything downstream (servo, reach model, joint safety) works in the robot's base frame."""
        x, y, yaw = (float(v) for v in self.base_pose())
        if x == y == yaw == 0.0 or not data.get("boxes"):
            return
        c, s = np.cos(yaw), np.sin(yaw)
        Rt = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])         # world -> base rotation (the transpose of Rz(yaw))
        for b in data["boxes"].values():
            b["t"] = (Rt @ (np.array(b["t"]) - [x, y, 0.0])).tolist()
            b["R"] = (Rt @ np.array(b["R"])).tolist()

    def boxes(self, max_age=1.5):
        """{colour: {"R", "t", "n", "rms"}} of the boxes in the latest message (empty when it is older than max_age seconds)."""
        d = self.data
        return dict(d.get("boxes", {})) if d and time.time() - self.t < max_age else {}

    def wait_box(self, color, count=6, timeout=8.0, abort=None):
        """`count` fresh messages that contain a pose of the `color` box from at least 3 landmarks."""
        n0, t0, out = self.n, time.time(), []
        while len(out) < count and time.time() - t0 < timeout:
            if abort is not None and abort.is_set():
                raise Aborted("aborted")
            if self.n != n0:
                n0 = self.n
                d = self.data
                if d and d.get("boxes", {}).get(color, {}).get("n", 0) >= 3:
                    out.append(d)
            time.sleep(0.02)
        return out


class GraspRobot:
    _driving, _latched = False, None     # (class defaults: the offline tests build the object without __init__)
    meeting, stash = None, types.MappingProxyType({})     # (class defaults too; __init__ gives each robot its own dict for the stash)

    def __init__(self, perception: Perception, event=lambda text: print(text, flush=True)):
        import rby1_sdk as rby
        from gripper_example import Gripper
        from sim_gripper_bridge import SimDynamixelBus
        import servo

        self.rby, self.servo, self.perception, self.event = rby, servo, perception, event
        self.abort_flag = threading.Event()
        self.phase, self.holding, self.seeded = "idle", False, False
        self.held = None      # colour of the box in the hand
        self.side, self.held_side = "right", None     # the arm a task works with / the arm that holds the box
        self.stash = {}       # {arm: colour}: the box the OTHER arm holds while this one takes a second box (two-handed tasks)
        self.meeting = None   # {arm: ee pose before the meeting pose}: both hands are at the meeting pose (meet), come back with unmeet
        self.squatted, self.ready = False, {"right": False, "left": False}     # torso squat done / which arm stands in the READY pose
        self._depth = 0       # nesting of _run / job: only the outermost one clears the abort flag
        self.robot = rby.create_robot_a("127.0.0.1:50051")
        assert self.robot.connect(), "cannot reach app_main (is run_sdk.sh running?)"
        self.robot.power_on(".*")
        self.robot.servo_on(".*")
        self._recover()
        self.gripper = Gripper(SimDynamixelBus("sim"))
        assert self.gripper.initialize()
        self.gripper.homing()
        self.gripper.start()
        time.sleep(0.5)       # the library's loop writes its own torque limit (5) once at its start; a box with the hand turned pushed a finger open at that limit
        self.gripper.bus.group_sync_write_send_torque([(i, GRIP_TORQUE) for i in self.gripper.GRIPPER_IDS])
        self.gripper.set_open()
        self.dyn = self.robot.get_dynamics()
        self.fks = {s: self.dyn.make_state(["base", f"ee_{s}"], rby.Model_A().robot_joint_names) for s in C.ARMS}
        pos = np.array(self.robot.get_state().position)
        # the arm that is not working is held at these joint values by the whole-body Cartesian command (right arm: 8..14, left arm: 15..21)
        self.hold = {s: dict(zip([f"{s}_arm_{i}" for i in range(7)], pos[8:15] if s == "right" else pos[15:22])) for s in C.ARMS}
        self.dbg = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.stream, self._base_stream, self.base_mode = None, None, "se2"
        self.odo, self._odo_lock = B.Odometry(*self._wheels()), threading.Lock()   # the base starts at the world origin, facing +x
        self.footprint = self._read_footprint()
        self.perception.base_pose = self.base_pose      # from here on the perception hands the boxes on in the base frame
        self.gt_base = None    # (pose, wall time) of the simulator's base pose, trusted only while the base drives (see base_pose)
        self._driving, self._latched = False, None
        self._base_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._base_rx.bind(("127.0.0.1", C.PORT_BASE_POSE))
        threading.Thread(target=self._listen_base, daemon=True).start()
        self._fk_lock = threading.Lock()  # the status heartbeat and the running task both compute forward kinematics
        self.cmd_tip = self.tip()

    # ---- state -------------------------------------------------------------------------------------------------------
    def ee_pose(self, side=None):
        side = side or self.side
        q = np.array(self.robot.get_state().position)
        with self._fk_lock:
            fk = self.fks[side]
            fk.set_q(q)
            self.dyn.compute_forward_kinematics(fk)
            return self.dyn.compute_transformation(fk, 0, 1)

    @staticmethod
    def tip_of(T_ee):
        return T_ee[:3, 3] + TOOL * (T_ee[:3, :3] @ np.array([0.0, 0.0, -1.0]))

    def tip(self, side=None):
        return self.tip_of(self.ee_pose(side))

    @staticmethod
    def _grip_id(side):
        return 0 if side == "left" else 1          # Dynamixel ID 0 = left gripper, 1 = right (examples/gripper_example.py)

    def closeness(self, side=None):
        """Gripper of `side` (default: the arm at work): 0 open ... 1 closed (fully closed means nothing between the fingers)."""
        g, i = self.gripper, self._grip_id(side or self.side)
        enc = dict(g.bus.group_fast_sync_read_encoder(g.GRIPPER_IDS))
        return float(1.0 - (enc[i] - g.min_q[i]) / (g.max_q[i] - g.min_q[i]))

    def _grip(self, closeness, side=None):
        """Open (0) or close (1) one gripper and leave the other as it is (the library's set_normalized moves both)."""
        g, i = self.gripper, self._grip_id(side or self.side)
        q = np.array(g.target_q if g.target_q is not None else (1.0 - 0.0) * (g.max_q - g.min_q) + g.min_q, float)
        q[i] = (1.0 - float(np.clip(closeness, 0.0, 1.0))) * (g.max_q[i] - g.min_q[i]) + g.min_q[i]
        g.target_q = q

    def status(self):
        x, y, yaw = self.base_pose()
        return {"phase": self.phase, "holding": self.holding, "held": self.held, "seeded": self.seeded,
                "tip": self.tip("right").round(4).tolist(), "closeness": round(self.closeness("right"), 2), "base": [round(x, 3), round(y, 3), round(float(np.degrees(yaw)), 1)],
                "arm": self.held_side or self.side, "ready": [s for s in C.ARMS if self.ready[s]], "tip_left": self.tip("left").round(4).tolist(),
                "closeness_left": round(self.closeness("left"), 2), "stash": dict(self.stash), "meeting": bool(self.meeting)}

    def state(self):
        """Joint positions by name [rad] and the right hand's pose, for the agent tools (examples/agent/robot_tools.py)."""
        names = list(self.rby.Model_A().robot_joint_names)
        q = np.array(self.robot.get_state().position)
        return {"joints": {n: round(float(v), 5) for n, v in zip(names, q)}, "tip": self.tip("right").round(4).tolist(),
                "tip_left": self.tip("left").round(4).tolist(), "ee_right": self.ee_pose("right").round(5).tolist(),
                "ee_left": self.ee_pose("left").round(5).tolist(), "robot": self.status()}

    def scene(self):
        """What the robot sees: the boxes from the stereo rig in the BASE frame (centre, yaw, what each rests on) and its own state (robot.base = where
        the base stands in the world: x, y, yaw in degrees). While the base is at its start pose the two frames are the same."""
        boxes = self.perception.boxes()
        on = C.stacked_on({c: b["t"] for c, b in boxes.items()})
        out = {}
        for c, b in boxes.items():
            R = np.array(b["R"])
            yaw = (float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) + 45.0) % 90.0 - 45.0   # a box looks the same every 90 degrees
            pos = {n: p["t"] for n, p in boxes.items()}
            out[c] = {"pos": [round(v, 3) for v in b["t"]], "yaw_deg": round(yaw, 1), "landmarks": b["n"], "on": on[c],
                      "best_arm": C.best_arm(c, pos, prefer=self.side),
                      "arms": {a: o["problem"] is None and o["comfort"] for a, o in C.arm_options(c, pos).items()}}
        return {"boxes": out, "robot": self.status(),
                "table": {"center": list(C.TABLE_CENTER), "size": list(C.TABLE_SIZE), "top": C.TABLE_TOP, "assumed": True}}

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
            time.sleep(0.4)    # the robot releases the stream's control a moment later: a command sent at once was rejected ("priority not higher") and cancelled

    def _stream_to(self, tip_xyz):
        """Replace the whole-body Cartesian target; the robot tracks it continuously (reached within 0.3 s, held for 1 s)."""
        rby = self.rby
        T = np.eye(4)
        T[:3, :3] = R_FRONT
        T[:3, 3] = np.asarray(tip_xyz) - TOOL * np.array([1.0, 0.0, 0.0])
        cart = (rby.CartesianCommandBuilder().set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1.0))
                .set_minimum_time(0.3).add_target("base", f"ee_{self.side}", T, 0.3, 1.0, 1.0))
        for joint, q in self.hold["left" if self.side == "right" else "right"].items():      # the other arm stays where it is
            cart.add_joint_position_target(joint, float(q))
        self._send_stream(rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyCommandBuilder().set_command(cart)).set_mobility_command(self._mobility_hold())))

    def _send_stream(self, rc):
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

    def _mobility_hold(self):
        """Zero planar velocity for the wheels, sent along with every arm and torso command: the unbraked base rolls away otherwise (it drifted 17 cm and 6
        degrees during one left-arm pick and place, from the reaction of the arm and torso motions)."""
        rby = self.rby
        return rby.MobilityCommandBuilder().set_command(rby.SE2VelocityCommandBuilder().set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(1.0)).set_minimum_time(0.2).set_velocity(np.array([0.0, 0.0]), 0.0))

    def _joint_move(self, body_builder, name, head_builder=None):
        """Blocking joint-space move that can be aborted. `head_builder` (a HeadCommandBuilder) moves the head along with the body."""
        rby = self.rby
        self._end_stream()
        components = rby.ComponentBasedCommandBuilder().set_mobility_command(self._mobility_hold())
        if body_builder is not None:
            components.set_body_command(body_builder)
        if head_builder is not None:
            components.set_head_command(head_builder)
        handler = self.robot.send_command(rby.RobotCommandBuilder().set_command(components), 10)
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
        if not C.tip_reachable(target, self.side):
            raise TaskError(f"{name}: target {np.round(target, 3).tolist()} is out of the {self.side} arm's reach")
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

    @contextlib.contextmanager
    def job(self):
        """One executor job, which may run several tasks in a row (pick, then place): an abort requested between two of them is kept."""
        if self._depth == 0:
            self.abort_flag.clear()
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    def _driving_run(self, name, body):
        """A task that moves the base: the simulator's base pose is trusted while it runs, and its last value stays afterwards."""
        def wrapped():
            self._driving = True
            try:
                return body()
            finally:
                time.sleep(0.3)
                self.base_pose()                   # latch the final pose before the flag goes down
                self._driving = False
        self._run(name, wrapped)

    def _run(self, name, fn, *args, **kw):
        """Run a task body: clears the abort flag (outermost call only), recovers after a fault or an abort, leaves the robot ready for
        the next task. Tasks nest (stack = pick + place_on, pick = seed first)."""
        if self.meeting and name not in ("meet", "unmeet", "joints"):
            raise TaskError("the hands are at the meeting pose: unmeet first")
        with self.job():
            outer, self.phase = self.phase, name
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
                self.phase = outer

    # ---- tasks -------------------------------------------------------------------------------------------------------
    def _joint_arm(self, side, q, seconds, name):
        rby = self.rby
        body = rby.BodyComponentBasedCommandBuilder()
        command = rby.ArmCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(seconds).set_position(np.asarray(q, float)))
        (body.set_right_arm_command if side == "right" else body.set_left_arm_command)(command)
        other = "left" if side == "right" else "right"
        if other in self.stash:       # the other arm holds a box: command it to stay (a command without it was cancelled by the robot)
            pos = np.array(self.robot.get_state().position)
            keep = rby.ArmCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(seconds).set_position(pos[15:22] if other == "left" else pos[8:15]))
            (body.set_right_arm_command if other == "right" else body.set_left_arm_command)(keep)
        self._joint_move(body, name)

    def _activate(self, side, park_other=True):
        """Make `side` the arm at work: the other arm hangs at its side (it would be in the way near the middle), the torso is squatted (the straight
        arm is a Cartesian singularity), this arm stands in the READY pose. Does only what is not already so."""
        other = "left" if side == "right" else "right"
        if park_other and self.ready[other] and other not in self.stash:     # an arm that holds a box stays where it is
            self.event(f"park the {other} arm")
            self._joint_arm(other, PARK, 6.0, "park")
            self.ready[other] = False
        pos = np.array(self.robot.get_state().position)
        self.hold[other] = dict(zip([f"{other}_arm_{i}" for i in range(7)], pos[15:22] if other == "left" else pos[8:15]))
        if not self.squatted:
            self._require_table_clear_for_seed()
            self.event("seed pose: torso squat")
            rby = self.rby
            self._joint_move(rby.BodyComponentBasedCommandBuilder().set_torso_command(
                rby.TorsoCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(6.0).set_position(TORSO))), "torso")
            self.squatted = True
        if not self.ready[side]:
            self.event(f"seed pose: {side} arm")
            self._joint_arm(side, READY, 8.0, "arm")
            self.ready[side] = True
        self.side, self.seeded = side, True
        self.cmd_tip = self.tip()

    def seed(self, side="right"):
        """Torso squat and bent arm: the start of every grasp."""
        self._run("seed", self._activate, side)

    def home(self):
        def body():
            if self.holding or self.stash:
                raise TaskError("a hand holds a box: place it first")
            for s in C.ARMS:
                self._grip(0.0, s)
            self.ready["right"] = False          # stand the right arm in the READY pose again, whatever happened to it
            self._activate("right")
        self._run("home", body)

    def set_gripper(self, closeness, arm="right"):
        self._grip(float(closeness), arm)
        if closeness < 0.5 and arm in self.stash:
            self.stash.pop(arm)
        if closeness < 0.5 and arm == self.held_side:
            self.holding, self.held, self.held_side = False, None, None

    def _choose_arm(self, color, arm, dest=None):
        """The arm that picks box `color` (and puts it at dest = (x, y, surface height), if given): `arm` if the caller named one, else the one that can
        do it most comfortably (C.best_arm), keeping the arm already at work unless the other is clearly better. Raises with both arms' reasons."""
        scene = {c: b["t"] for c, b in self.perception.boxes().items()}
        if color not in scene:
            raise TaskError(f"cannot see the {color} box")
        spot, top = (None, C.TABLE_TOP) if dest is None else ((dest[0], dest[1]), dest[2])
        if arm in C.ARMS:
            problem = C.pick_problem(color, scene, arm) or (None if spot is None or C.tip_reachable(C.box_tip(*spot, top), arm)
                                                           else f"the place at ({spot[0]:.2f}, {spot[1]:.2f}) is out of reach for the {arm} arm")
            if problem:
                raise TaskError(problem)
            return arm
        side = C.best_arm(color, scene, spot=spot, top=top, prefer=self.side)
        if side is None:
            reasons = [p for a in C.ARMS if (p := C.pick_problem(color, scene, a))]
            raise TaskError("neither arm can do it: " + (" / ".join(reasons) if reasons else
                                                         f"each arm reaches the {color} box, but none reaches both it and the place at {spot}"))
        return side

    def pick(self, color="blue", use_image=True, bias=(0.0, 0.0, 0.0), arm="auto", dest=None):
        """Visual-servo to the `color` box, close the gripper, lift. The arm: `arm` ("right" / "left") or "auto" = the one that holds its hand most
        comfortably (examples/gui/scene_config.py best_arm; `dest` = (x, y, height) where the box is going next, so that the arm can reach both).
        Starts from the current hand pose (the arm is made ready first if needed, the other arm parks)."""
        def body():
            nonlocal arm
            if color in self.stash.values() or color == self.held:
                raise TaskError(f"already holding the {color} box")
            if self.holding:        # one arm holds a box: the other takes a second one (the first stays in its hand)
                other = "left" if self.held_side == "right" else "right"
                if arm not in ("auto", other):
                    raise TaskError(f"the {self.held_side} arm holds the {self.held} box: only the {other} arm can take another")
                arm = other
                self.stash[self.held_side] = self.held
                self.holding, self.held, self.held_side = False, None, None
            if color not in C.BOX_NAMES:
                raise TaskError(f"unknown box '{color}' (boxes: {', '.join(C.BOX_NAMES)})")
            if not self.perception.wait_box(color, 3, abort=self.abort_flag):
                raise TaskError(f"no pose of the {color} box from the stereo rig (perception node running? 3 landmarks visible?)")
            side = self._choose_arm(color, arm, dest)
            self.event(f"the {side} arm takes the {color} box")
            self._activate(side)
            self._grip(0.0)
            est = self.perception.wait_box(color, 6, abort=self.abort_flag)
            if not est:
                raise TaskError(f"no pose of the {color} box from the stereo rig (perception node running? 3 landmarks visible?)")
            box_R = np.array(est[-1]["boxes"][color]["R"])
            box_t = np.mean([m["boxes"][color]["t"] for m in est], axis=0) + np.asarray(bias)
            scene = {c: b["t"] for c, b in self.perception.boxes().items()}
            scene[color] = box_t.tolist()
            problem = C.pick_problem(color, scene, side)
            if problem:
                raise TaskError(problem)
            self.event(f"stereo estimate of the {color} box {np.round(box_t, 3).tolist()} ({est[-1]['boxes'][color]['n']} landmarks)")
            self._servo(color, box_R, box_t, use_image, np.asarray(bias, float))
            self.event("grasp")
            self.debug(stage="grasp", text=[f"grasp the {color} box"])
            self._grip(1.0)
            self._hold(1.8)
            c = self.closeness()
            if not (HOLDING_CLOSENESS[0] < c < HOLDING_CLOSENESS[1]):
                self._grip(0.0)
                raise TaskError(f"grasp missed (gripper closeness {c:.2f}: nothing between the fingers)")
            self.holding, self.held, self.held_side = True, color, side
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

    # ---- two hands -----------------------------------------------------------------------------------------------------
    MIN_ARM_GAP = 0.04      # [m] elbows and forearms closer than this stop a two-handed move (examples/assembly/meet_study.py: 9 cm "forward", 33 cm "inward")

    def _arm_gap(self):
        """Smallest distance between the elbows / forearms of the two arms from the joint state [m]."""
        import kinematics as K
        q = dict(zip(self.rby.Model_A().robot_joint_names, np.array(self.robot.get_state().position)))
        P = K.all_points(q)
        return min(float(np.linalg.norm(P[f"right_{a}"] - P[f"left_{b}"])) for a in ("elbow", "forearm") for b in ("elbow", "forearm"))

    def _sdk_fk(self, q7, side):
        """ee pose of `side` for arm joints q7 (the other joints as they are now), by the SDK's own forward kinematics: arm_ik.solve plans with it."""
        q = np.array(self.robot.get_state().position)
        i0 = 8 if side == "right" else 15
        q[i0:i0 + 7] = q7
        with self._fk_lock:
            fk = self.fks[side]
            fk.set_q(q)
            self.dyn.compute_forward_kinematics(fk)
            return self.dyn.compute_transformation(fk, 0, 1)

    def _arm_q(self):
        pos = np.array(self.robot.get_state().position)
        return {"right": pos[8:15], "left": pos[15:22]}

    def _joint_pair(self, q_to, seconds, name):
        """Both arms to the joint values q_to ({arm: 7 values}) in one straight joint-space move. Refused when the arms would come closer than MIN_ARM_GAP on the way."""
        import arm_ik
        q0 = self._arm_q()
        for a in np.linspace(0.0, 1.0, 13):
            gap = arm_ik.arm_gap(*(q0[s] + a * (np.asarray(q_to[s]) - q0[s]) for s in C.ARMS))
            if gap < self.MIN_ARM_GAP:
                raise TaskError(f"{name}: the arms would come {gap * 100:.1f} cm apart on the way (limit {self.MIN_ARM_GAP * 100:.0f} cm)")
        rby = self.rby
        body = rby.BodyComponentBasedCommandBuilder()
        for side, setter in (("right", body.set_right_arm_command), ("left", body.set_left_arm_command)):
            setter(rby.ArmCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(seconds).set_position(np.asarray(q_to[side], float))))
        self._joint_move(body, name)

    def meet(self, x=0.48, z=1.0, gap=C.MEET_GRASP_BACK, variant="forward", seconds=16.0):
        """Both hands to the mating plane in front of the chest (scene_config.meet_targets), whatever they hold. Both arms are made ready first (an arm that holds a
        box stays as it is, so pick the two parts first). The joint values come from inverse kinematics started at the joints the arms have now (arm_ik.py) and the
        arms go there in one straight joint-space move: the whole-body Cartesian IK of the robot turned the right wrist the long way round and the box fell. Come back with unmeet."""
        def body():
            import arm_ik
            if variant not in C.MEET_VARIANTS:
                raise TaskError(f"unknown hand orientation '{variant}' ({', '.join(C.MEET_VARIANTS)})")
            self._activate("right", park_other=False)
            self._activate("left", park_other=False)
            tg = C.meet_targets(x, z, gap, variant)
            q_now, q_to = self._arm_q(), {}
            for s_ in C.ARMS:
                q = arm_ik.solve(s_, np.array(tg[s_]["grasp"]), np.array(tg[s_]["R"]), seeds=8, q_start=q_now[s_], fk=self._sdk_fk)
                if q is None:
                    raise TaskError(f"the {s_} hand cannot reach the meeting pose x {x:.2f}, z {z:.2f} ({variant})")
                q_to[s_] = q
            change = {s_: float(np.degrees(np.abs(q_to[s_] - q_now[s_]).max())) for s_ in C.ARMS}
            if max(change.values()) > 150.0:
                raise TaskError(f"the meeting pose needs a joint to turn {max(change.values()):.0f} degrees: refused")
            if self.meeting is None:
                self.meeting = {s_: q_now[s_].copy() for s_ in C.ARMS}
            self.event(f"both hands to the meeting pose x {x:.2f}, z {z:.2f} ({variant}, grasp points {(2 * C.MEET_GRASP_BACK + gap) * 100:.0f} cm apart): "
                       f"largest joint change right {change['right']:.0f} deg, left {change['left']:.0f} deg")
            self._joint_pair(q_to, seconds, "meet")
            err = {}
            for s_ in C.ARMS:
                T = self.ee_pose(s_)
                ang = np.degrees(np.arccos(np.clip((np.trace(np.array(tg[s_]["R"]) @ T[:3, :3].T) - 1) / 2, -1.0, 1.0)))
                err[s_] = (float(np.linalg.norm(T[:3, 3] - np.array(tg[s_]["ee"]))) * 1000, float(ang))
            self.event("meeting pose reached: " + ", ".join(f"{s_} {e[0]:.1f} mm / {e[1]:.1f} deg" for s_, e in err.items()) + f"; arms {self._arm_gap() * 100:.0f} cm apart")
            self.cmd_tip = self.tip()
        self._run("meet", body)

    def unmeet(self, seconds=16.0):
        """Both arms back to the joint values they had before the first meet."""
        def body():
            if not self.meeting:
                raise TaskError("the hands are not at the meeting pose")
            self.event("both hands back from the meeting pose")
            self._joint_pair(self.meeting, seconds, "unmeet")
            self.meeting = None
            pos = np.array(self.robot.get_state().position)           # the arms are not where the holds of the last task say
            self.hold = {s: dict(zip([f"{s}_arm_{i}" for i in range(7)], pos[8:15] if s == "right" else pos[15:22])) for s in C.ARMS}
            self.cmd_tip = self.tip()
        self._run("unmeet", body)

    def _place_at(self, x, y, top, base=None):
        """Carry the held box over (x, y) (rise first, so it clears boxes on the way), lower it onto a surface of height `top`, let go,
        back off. With `base` (colour of the box below) the hand first aligns the held box over it with the stereo rig (_align_over)."""
        if not self.holding:
            raise TaskError("nothing in the hand: pick first")
        self.side = self.held_side or self.side              # the arm that holds the box carries it
        what = f"on the {base} box" if base else f"at ({x:.3f}, {y:.3f})"
        above, down, back = place_waypoints(self.tip(), x, y, top)
        self.cmd_tip = self.tip()
        if above[2] - self.cmd_tip[2] > 0.01:
            self.event(f"place {what}: rise")
            self._move_tip(np.array([self.cmd_tip[0], self.cmd_tip[1], above[2]]), "rise")
        self.event(f"place {what}: carry")
        self.debug(stage="place: carry", text=[f"place {what}", "carry"])
        self._move_tip(above, "carry")
        if base:
            self._align_over(base)
            down = np.array([self.cmd_tip[0], self.cmd_tip[1], down[2]])   # straight down from where the alignment ended
        self.event("place: lower")
        self.debug(stage="place: lower", text=["place", "lower"])
        self._move_tip(down, "lower", duration=3.0)
        self.event("place: release")
        self._grip(0.0)
        placed, self.holding, self.held, self.held_side = self.held, False, None, None
        self._hold(1.5)
        back = self._back_off(down)
        self.event(f"place: back off {np.linalg.norm(back - down) * 100:.0f} cm")
        self._move_tip(back, "back off", duration=2.0)
        self.debug(stage="done", text=["done"])
        return placed

    def _back_off(self, down):
        """Where the hand goes after the release: 10 cm back, less if the reach model stops it earlier (on a stack the wrist is closer
        to the shoulder). 4 cm is enough to clear the box with the fingers."""
        for d in (0.10, 0.08, 0.06, 0.05, 0.04):
            back = down + np.array([-d, 0.0, 0.0])
            if C.tip_reachable(back, self.side):
                return back
        raise TaskError(f"cannot back off from the box placed at {np.round(down, 3).tolist()}: out of reach")

    def place(self, x, y, top=C.TABLE_TOP):
        """Carry the held box over (x, y), lower it onto the table, let go, back off."""
        self._run("place", self._place_at, x, y, top)

    def place_on(self, base):
        """Put the held box on top of the `base` box: position and height from the stereo rig, alignment over it, check afterwards."""
        def body():
            if not self.holding:
                raise TaskError("nothing in the hand: pick first")
            est = self.perception.wait_box(base, 6, abort=self.abort_flag)
            if not est:
                raise TaskError(f"no pose of the {base} box from the stereo rig")
            bx, by, bz = np.mean([m["boxes"][base]["t"] for m in est], axis=0)
            top = bz + C.BOX_SIZE[2] / 2
            if not C.tip_reachable((bx + C.TIP_BEYOND_CENTRE, by, top + C.BOX_SIZE[2] / 2 + HOVER), self.held_side or self.side):
                raise TaskError(f"the top of the {base} box at ({bx:.2f}, {by:.2f}, {top:.2f}) is out of reach for the {self.held_side or self.side} arm")
            placed = self._place_at(bx, by, top, base=base)
            self._verify_stacked(placed, base)
        self._run("place_on", body)

    def stack(self, order):
        """Stack the boxes of `order` (colours, bottom first): the first stays where it is, every other one is picked and put on the one
        before it. Checks the whole plan against the scene first (reach, finger clearance), so nothing starts that cannot finish."""
        def body():
            problems = C.stack_problems(order, {c: b["t"] for c, b in self.perception.boxes().items()})
            if problems:
                raise TaskError("; ".join(problems))
            bx, by, _ = self.perception.boxes()[order[0]]["t"]
            for k, (below, c) in enumerate(zip(order, order[1:]), 1):
                self.pick(c, dest=(bx, by, C.TABLE_TOP + k * C.BOX_SIZE[2]))          # the arm that can pick it AND reach this level of the stack
                self.place_on(below)
        self._run("stack", body)

    def unstack(self):
        """Take a stack apart: the top box first, each one onto a free table spot (its own start spot when that is free), until no box rests on
        another one. The bottom box stays. The scene is read again before every box, so it works from whatever the cameras see."""
        def body():
            if self.holding:
                raise TaskError(f"the hand holds the {self.held} box: put it down first")
            done = 0
            while done < len(C.BOX_NAMES):
                time.sleep(0.5)                                  # let the cameras catch up with the last placement
                boxes = {c: tuple(b["t"]) for c, b in self.perception.boxes().items()}
                order = C.unstack_order(boxes)
                if not order:
                    break
                c = order[0]
                base = tuple(float(v) for v in self.base_pose())
                others = [a for a in C.ARMS if a != self.side]
                for arm in [self.side] + others:               # the arm already at work first; its own spot (reachable for it), else the other arm
                    spot = C.table_spot(c, boxes, base=base, arm=arm)
                    if spot is not None and C.best_arm(c, boxes, spot=spot, top=C.TABLE_TOP) is not None:
                        break
                else:
                    raise TaskError(f"no free table spot for the {c} box")
                arm = C.best_arm(c, boxes, spot=spot, top=C.TABLE_TOP, prefer=arm)
                self.event(f"unstack: the {c} box goes to ({spot[0]:.2f}, {spot[1]:.2f})")
                self.pick(c, arm=arm, dest=(spot[0], spot[1], C.TABLE_TOP))
                self.place(spot[0], spot[1])
                self._verify_on_table(c)
                done += 1
            if done == 0:
                raise TaskError("nothing is stacked")
        self._run("unstack", body)

    def _verify_on_table(self, color):
        """After a placement on the table: is the box standing at table height? Logs where; raises when it clearly is not."""
        est = self.perception.wait_box(color, 3, timeout=4.0, abort=self.abort_flag)
        if not est:
            self.event(f"could not check the {color} box (not visible)")
            return
        x, y, z = np.mean([m["boxes"][color]["t"] for m in est], axis=0)
        self.event(f"{color} box on the table at ({x:.3f}, {y:.3f}), height {z:.3f}")
        if abs(z - (C.TABLE_TOP + C.BOX_SIZE[2] / 2)) > 0.02:
            raise TaskError(f"the {color} box is not standing on the table (height {z:.3f} m)")

    def move_joints(self, traj):
        """Play a joint trajectory designed by the agent (examples/agent: {"unit", "waypoints": [{"t", joint: value}]}): right arm, head, torso
        yaw only. The safety check of examples/agent/safety.py runs here again, whatever the client did; waypoint by waypoint, each a blocking
        joint-space move of the given duration (it stops at every waypoint) that can be aborted. Afterwards the hand is wherever the trajectory
        left it, so the next pick starts with the seed pose again."""
        import safety
        def body():
            rby = self.rby
            cur = self.state()["joints"]
            boxes = {c: b["t"] for c, b in self.perception.boxes().items()}
            problems, summary = safety.check(cur, traj, boxes, base=self.base_pose())
            if problems:
                raise TaskError("the trajectory is not safe: " + "; ".join(problems))
            self.event(f"joints: {len(traj['waypoints'])} waypoints, {summary['duration_s']} s")
            self._end_stream()
            for i, (t, q, _, _) in enumerate(safety.expand(cur, traj), 1):
                arm, larm, torso, head = ([q.get(f"{g}_{k}", cur[f"{g}_{k}"]) for k in range(n)]
                                          for g, n in (("right_arm", 7), ("left_arm", 7), ("torso", 6), ("head", 2)))
                changed = lambda names: any(abs(q.get(n, cur[n]) - cur[n]) > 1e-4 for n in names)   # noqa: E731
                arm_c, larm_c = changed([f"right_arm_{k}" for k in range(7)]), changed([f"left_arm_{k}" for k in range(7)])
                torso_c, head_c = changed(["torso_5"]), changed(["head_0", "head_1"])
                self.event(f"joints: waypoint {i}/{len(traj['waypoints'])} ({t:.1f} s)")
                if not (arm_c or larm_c or torso_c or head_c):
                    t0 = time.time()
                    while time.time() - t0 < t:
                        self._check()
                        time.sleep(0.05)
                    continue
                body_b = rby.BodyComponentBasedCommandBuilder() if (arm_c or larm_c or torso_c) else None
                if arm_c:
                    body_b.set_right_arm_command(rby.ArmCommandBuilder().set_command(
                        rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(np.array(arm))))
                if larm_c:
                    body_b.set_left_arm_command(rby.ArmCommandBuilder().set_command(
                        rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(np.array(larm))))
                if torso_c:
                    body_b.set_torso_command(rby.TorsoCommandBuilder().set_command(
                        rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(np.array(torso))))
                head_b = rby.HeadCommandBuilder().set_command(
                    rby.JointPositionCommandBuilder().set_minimum_time(t).set_position(np.array(head))) if head_c else None
                self._joint_move(body_b, f"waypoint {i}", head_builder=head_b)
                cur.update(q)
            self.cmd_tip = self.tip()
            for s in C.ARMS:      # the Cartesian streams of the next pick hold the arm that is not at work HERE; both arms are wherever the trajectory left them
                self.hold[s] = {f"{s}_arm_{k}": cur[f"{s}_arm_{k}"] for k in range(7)}
            self.ready = {"right": False, "left": False}
            self.squatted = self.squatted and abs(cur.get("torso_5", 0.0)) < 1e-3     # a turned torso: squat again (that also straightens it)
            self.seeded = False       # the next pick makes its arm ready first
        self._run("joints", body)

    # ---- the base ----------------------------------------------------------------------------------------------------
    def _wheels(self):
        pos = self.robot.get_state().position      # Model_A.mobility_idx = [0, 1]: right_wheel, left_wheel
        return float(pos[0]), float(pos[1])

    def _listen_base(self):
        while True:
            try:
                msg = json.loads(self._base_rx.recv(4096))
                self.gt_base = (np.array(msg["base"], float), time.time())
            except (ValueError, KeyError):
                continue

    def base_pose(self):
        """(x, y, yaw) of the base relative to where the executor started. The simulator's true pose when it keeps sending it (the wheels slip
        when the base turns: odometry reported 30 degrees where 15 happened); else the wheel odometry. Called by the status heartbeat too, so
        the odometry never has more than a fraction of a second of wheel travel to integrate."""
        with self._odo_lock:
            odo = self.odo.update(*self._wheels()).copy()
            if self.gt_base is not None and time.time() - self.gt_base[1] < 0.5 and self._driving:
                self._latched = self.gt_base[0].copy()      # driving: the simulator's pose (the wheels slip when it turns, odometry is off by 2x)
                self.odo.pose = self._latched.copy()        # and the fallback carries on from here
                return self._latched.copy()
            if self._latched is not None:                   # not driving: the base stands where the last drive left it. The pose the simulator reports
                self.odo.pose = self._latched.copy()        # jumped by 10-20 cm during arm motions while the transform tree and the forward kinematics
                return self._latched.copy()                 # of the arms show the base standing still
            return odo

    def _require_table_clear_for_seed(self):
        """The squat of the seed pose needs room in front of the base (C.SEED_EDGE_X): the table is where it is in the world, the base may stand anywhere."""
        if self.squatted:
            return
        ahead = C.table_front_distance(self.base_pose())
        if ahead < C.SEED_EDGE_X:
            raise TaskError(f"the table is only {ahead * 100:.0f} cm ahead of the base: the seed pose needs {C.SEED_EDGE_X * 100:.0f} cm (drive back first)")

    def _read_footprint(self):
        """Base and wheel outline from the simulator's ground truth file (src/rby1_vision.py write_gt), else the generous default."""
        try:
            return tuple(json.load(open("/isaac-sim/.local/share/ov/data/rby1_gt.json"))["base_footprint"])
        except (OSError, KeyError, ValueError):
            return B.FOOTPRINT

    def _send_base(self, v, w):
        """One SE(2) velocity command (forward speed v [m/s], turn rate w [rad/s]), valid for 0.3 s: if this program dies the base stops by itself."""
        rby = self.rby
        header = rby.CommandHeaderBuilder().set_control_hold_time(0.3)
        if self.base_mode == "wheels":    # the wheel joints directly: (v, w) -> the two wheel speeds (rad/s) with the geometry of base_motion.py
            wheels = np.array([v + w * B.TRACK / 2, v - w * B.TRACK / 2]) / (B.WHEEL_R * B.FORWARD_SIGN)
            command = rby.JointVelocityCommandBuilder().set_command_header(header).set_minimum_time(0.2).set_velocity(wheels)
        else:                             # the SDK's planar velocity command
            command = rby.SE2VelocityCommandBuilder().set_command_header(header).set_minimum_time(0.2).set_velocity(np.array([v, 0.0]), w)
        rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_mobility_command(rby.MobilityCommandBuilder().set_command(command)))
        for attempt in (0, 1):
            if self._base_stream is None:
                self._base_stream = self.robot.create_command_stream(10)
            try:
                self._base_stream.send_command(rc, 200)
                return
            except RuntimeError as exc:
                if "expired" not in str(exc) or attempt:
                    raise
                self._base_stream = None

    def _stop_base(self):
        try:
            for _ in range(3):
                self._send_base(0.0, 0.0)
                time.sleep(0.05)
        finally:
            if self._base_stream is not None:
                self._base_stream.cancel()
                self._base_stream = None

    def _drive_segment(self, kind, target):
        """One closed-loop move of the base: a turn of `target` rad or a straight move of `target` m along the current heading."""
        start = self.base_pose()
        v_max, acc, v_min, tol = (B.W_MAX, B.A_ANG, B.W_MIN, B.TOL_ANG) if kind == "turn" else (B.V_MAX, B.A_LIN, B.V_MIN, B.TOL_LIN)
        limit, t0, sign = 5.0 + 3.0 * abs(target) / v_max, time.time(), 1.0 if target >= 0 else -1.0
        boost, last_progress, last_done = 1.0, time.time(), 0.0
        try:
            while True:
                t_loop = time.time()
                self._check()
                pose = self.base_pose()
                done = B.wrap(pose[2] - start[2]) if kind == "turn" else float(np.dot(pose[:2] - start[:2], [np.cos(start[2]), np.sin(start[2])]))
                remaining = target - done
                if remaining * sign <= tol:
                    return
                if done * sign < -(0.03 if kind == "forward" else 0.05):   # moving the wrong way: the wheel sign is off (or the base is blocked); do not run away
                    raise TaskError(f"the base moves the wrong way ({done:+.3f} while driving {target:+.3f}): wheel direction wrong? stopped")
                if time.time() - t0 > limit:
                    raise TaskError(f"the base did not finish the {kind} within {limit:.0f} s (done {done:.3f} of {target:.3f})")
                if abs(done - last_done) > (0.002 if kind == "forward" else 0.004):   # moving: fine; stuck for a second: push harder (stiction)
                    last_progress, last_done, boost = time.time(), done, 1.0
                elif time.time() - last_progress > B.STALL_TIME:
                    boost, last_progress = min(boost * B.STALL_BOOST, 3.0), time.time()
                s = sign * min(boost * B.speed(remaining, v_max, acc, v_min), 1.5 * v_max)
                self._send_base(0.0 if kind == "turn" else s, s if kind == "turn" else 0.0)
                self._tick(t_loop)
        finally:
            self._stop_base()
            time.sleep(0.3)

    def drive_raw(self, v, w, seconds, mode="se2"):
        """Command a constant (v, w) for `seconds` and log the base pose every half second: for finding out how the base really moves (hard limits:
        |v| <= 0.15 m/s, |w| <= 0.5 rad/s, 6 s; the table check of drive does not apply, so only use it with the arms hanging and the table far away)."""
        def body():
            import kinematics
            if abs(v) > 0.15 or abs(w) > 0.5 or not 0 < seconds <= 6.0:
                raise TaskError("drive_raw: |v| <= 0.15 m/s, |w| <= 0.5 rad/s, 0 < seconds <= 6")
            if B.hits_table(self.base_pose(), list(kinematics.all_points(self.state()["joints"]).values()), self.footprint):
                raise TaskError("drive_raw: the base or an arm is already at the table")
            self._end_stream()
            t0, nxt, q0, self.base_mode = time.time(), 0.0, self._wheels(), mode
            try:
                while time.time() - t0 < seconds:
                    t_loop = time.time()
                    self._check()
                    if t_loop - t0 >= nxt:
                        p, q = self.base_pose(), self._wheels()
                        self.event(f"raw t={t_loop - t0:.1f} s pose ({p[0]:.3f}, {p[1]:.3f}, {np.degrees(p[2]):.1f} deg) wheels turned right {q[0] - q0[0]:+.2f} left {q[1] - q0[1]:+.2f} rad")
                        nxt += 0.5
                    self._send_base(v, w)
                    self._tick(t_loop)
            finally:
                self._stop_base()
                self.base_mode = "se2"
                time.sleep(1.0)
            p = self.base_pose()
            self.event(f"raw end pose ({p[0]:.3f}, {p[1]:.3f}, {np.degrees(p[2]):.1f} deg)")
        self._driving_run("drive_raw", body)

    def drive(self, forward=0.0, turn_deg=0.0, home=False):
        """Roll the base: first turn by `turn_deg` (positive = left), then forward by `forward` m (negative = back); or `home`: back to the start
        pose. Small steps with hard limits (base_motion.py), checked against the table with the arm where it is, driven closed-loop on the wheel
        odometry. While the base is away from its start pose the manipulation skills refuse (their frames assume it is there)."""
        def body():
            import kinematics
            self._end_stream()
            arm = lambda: list(kinematics.all_points(self.state()["joints"]).values())   # noqa: E731 - the arms now, in the base frame
            if not home:   # one command is checked as a whole (size, distance from the start, table), then done: turn first, then forward
                turn = float(np.radians(turn_deg))
                problems = B.check_move(self.base_pose(), float(forward), turn, arm(), self.footprint)
                if problems:
                    raise TaskError("; ".join(problems))
                if abs(turn) > B.TOL_ANG:
                    self.event(f"drive: turn {turn_deg:+.0f} deg")
                    self._drive_segment("turn", turn)
                if abs(forward) > B.TOL_LIN:
                    self.event(f"drive: forward {forward * 100:+.0f} cm")
                    self._drive_segment("forward", float(forward))
            else:          # the way back: one step at a time, planned again from where the base really is (a turn moves the base origin)
                for _ in range(12):
                    if self.odo.at_home() and abs(self.base_pose()[2]) <= B.HOME_TOL_ANG / 2:
                        break
                    t1, d, t2 = B.way_home(self.base_pose())
                    kind, amount = next(((k, a) for k, a, tol in (("turn", t1, B.TOL_ANG), ("forward", d, B.TOL_LIN), ("turn", t2, B.TOL_ANG)) if abs(a) > tol), (None, 0.0))
                    if kind is None:
                        break
                    step = float(np.clip(amount, -(B.MAX_TURN if kind == "turn" else B.MAX_FORWARD), B.MAX_TURN if kind == "turn" else B.MAX_FORWARD))
                    problems = B.check_move(self.base_pose(), step if kind == "forward" else 0.0, step if kind == "turn" else 0.0, arm(), self.footprint,
                                            check_range=False)
                    if problems:
                        raise TaskError("; ".join(problems))
                    self.event(f"drive home: {kind} {np.degrees(step):+.0f} deg" if kind == "turn" else f"drive home: forward {step * 100:+.0f} cm")
                    self._drive_segment(kind, step)
                if not self.odo.at_home():
                    x, y, yaw = self.base_pose()
                    raise TaskError(f"could not get exactly back to the start pose: {np.hypot(x, y) * 100:.0f} cm and {np.degrees(yaw):.0f} degrees off "
                                    f"(the simulated base turns poorly: its turns swing the base about 20-60 cm); the manipulation skills stay refused")
            x, y, yaw = self.base_pose()
            self.event(f"base at ({x:.3f}, {y:.3f}), heading {np.degrees(yaw):.1f} deg")
        self._driving_run("drive", body)

    def _align_over(self, base):
        """Hover loop over the `base` box: drive the held box's centre onto the base box's centre. Both are measured by the stereo rig
        in the same image, so the (calibration) error of the rig and the tracking error of the position loop do not enter: only the
        relative position counts. Falls back to the hand's own pose when the held box is not seen (its fingers hide landmarks)."""
        dt, t0, ok, err_f, src = 1.0 / RATE, time.time(), 0, None, "-"
        while time.time() - t0 < ALIGN_TIMEOUT:
            t_loop = time.time()
            self._check()
            T_ee = self.ee_pose()
            boxes = self.perception.boxes()
            if base in boxes:
                if self.held in boxes:
                    held_xy, src = np.array(boxes[self.held]["t"][:2]), "stereo"
                else:
                    held_xy, src = self.tip_of(T_ee)[:2] - np.array([C.TIP_BEYOND_CENTRE, 0.0]), "hand"
                err = np.array(boxes[base]["t"][:2]) - held_xy
                err_f = err if err_f is None else 0.6 * err_f + 0.4 * err        # the stereo fit is a few mm noisy
                v = ALIGN_KP * err_f
                if np.linalg.norm(v) > ALIGN_V_MAX:
                    v *= ALIGN_V_MAX / np.linalg.norm(v)
                self.cmd_tip = self.cmd_tip + np.r_[v * dt, 0.0]
                lead = self.cmd_tip - self.tip_of(T_ee)
                if np.linalg.norm(lead) > LEAD_MAX:
                    self.cmd_tip = self.tip_of(T_ee) + lead * LEAD_MAX / np.linalg.norm(lead)
                ok = ok + 1 if np.linalg.norm(err_f) < ALIGN_TOL else 0
                self.debug(stage="align", text=[f"align the {self.held} box over the {base} box (measured by the {src})",
                                                f"offset {err_f[0] * 1000:+.1f} / {err_f[1] * 1000:+.1f} mm"])
                if ok >= int(0.5 * RATE):
                    self.event(f"aligned over the {base} box: offset {np.linalg.norm(err_f) * 1000:.1f} mm ({src})")
                    return
            self._stream_to(self.cmd_tip)
            self._tick(t_loop)
        if err_f is None:
            raise TaskError(f"the {base} box is not visible: cannot align over it")
        self.event(f"alignment over the {base} box did not settle within {ALIGN_TIMEOUT:.0f} s "
                   f"(offset still {np.linalg.norm(err_f) * 1000:.1f} mm): placing anyway")

    def _verify_stacked(self, color, base):
        """After the release: is the `color` box standing on the `base` box? Logs the offset; raises when it clearly is not."""
        est_c = self.perception.wait_box(color, 4, timeout=4.0, abort=self.abort_flag)
        est_b = self.perception.wait_box(base, 1, timeout=2.0, abort=self.abort_flag)
        if not est_c or not est_b:
            self.event(f"could not check the stack ({color} or {base} box not visible)")
            return
        tc, tb = np.array(est_c[-1]["boxes"][color]["t"]), np.array(est_b[-1]["boxes"][base]["t"])
        d = tc - tb
        self.event(f"{color} box on the {base} box: offset ({d[0] * 1000:+.1f}, {d[1] * 1000:+.1f}) mm, height above it {d[2] * 1000:.0f} mm")
        if abs(d[2] - C.BOX_SIZE[2]) > STACK_Z_TOL or abs(d[0]) > STACK_XY_TOL or abs(d[1]) > STACK_XY_TOL:
            raise TaskError(f"the {color} box did not stay on the {base} box (offset {d.round(3).tolist()} m)")

    # ---- the continuous servo (examples/vservo/hybrid_grasp.py) -----------------------------------------------------------
    def _servo(self, color, box_R, box_t, use_image, bias):
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
                    blobs = np.array(data["wrist" if self.side == "right" else "wrist_left"].get(color) or np.zeros((0, 3)))   # the camera on the hand at work
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
                if (c_world is None or not use_image) and color in data.get("boxes", {}):   # no usable image (yet): the stereo measurement
                    c_world, src = np.array(data["boxes"][color]["t"]) + bias, "stereo"
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
                    self.debug(stage=f"continuous servo ({src})", arm=self.side, err_mm=[lat[0] * 1000, lat[1] * 1000], remaining_mm=remaining * 1000,
                               text=[f"continuous servo to the {color} box, measured by the {src}",
                                     f"lateral error {lat[0] * 1000:+.1f} / {lat[1] * 1000:+.1f} mm, {remaining * 1000:.0f} mm to go"], **view)
                if ok_ticks >= int(0.25 * RATE):
                    self.event(f"servo arrived ({src}, {self.side} arm): fingertips {np.round(self.tip_of(T_ee), 3).tolist()}, the box centre measured at "
                               f"{np.round(c_world, 3).tolist()} (error in the hand frame {lat[0] * 1000:+.1f} / {lat[1] * 1000:+.1f} mm, {remaining * 1000:.0f} mm to go)")
                    return
            self._stream_to(self.cmd_tip)
            tick += 1
            self._tick(t_loop)
        raise TaskError("the servo did not arrive within the time limit")
