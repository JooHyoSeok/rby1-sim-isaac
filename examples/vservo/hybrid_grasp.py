"""Continuous hybrid visual-servoing grasp: the hand flows to the box while the cameras keep correcting it.

Runs inside the simulator container (rby1_sdk is installed only there) next to perception_node.py (ROS 2 container).
The sim must run with docker/run_sdk.sh ... --gripper --gripper-name rb_gripper --vision:
  docker cp examples/vservo/{vision.py,servo.py,hybrid_grasp.py} examples/gripper_example.py rby1-sim-isaac:/tmp/
  docker exec rby1-sim-isaac /isaac-sim/python.sh /tmp/hybrid_grasp.py [--no-servo]

  1. seed pose (torso squat, bent right arm), as in examples/grasp/grasp_object.py
  2. the stereo rig gives a first box pose (association and orientation of the landmarks, and the fallback measurement)
  3. one control loop at 20 Hz until the fingertips are around the box, with no stop-and-go:
       - the box centre is measured in the wrist camera image (3D, in the hand frame at the instant of the image, then fixed in the
         world, because the box does not move) and compared with the *current* hand pose, so latency does not make the loop
         correct the same measurement twice; without a usable image the stereo estimate is used
       - the commanded fingertip target moves with a lateral velocity proportional to the box centre's offset from the tool line and a
         forward velocity proportional to the distance left, reduced while the lateral error is large; the target is integrated, not
         re-based on the measured pose, so the position loop's steady-state offset is driven out (integral action)
       - the target goes to the robot as a command *stream* (whole-body Cartesian, torso + arms), so the motion is continuous
  4. close the gripper, then lift along a smooth ramp, also streamed.
--no-servo uses the stereo measurement only (the same controller), for comparison. --color picks another box (default blue).
Tip positions are in the base frame (= world).
"""
import argparse
import collections
import json
import socket
import sys
import threading
import time

import numpy as np
import rby1_sdk as rby

sys.path.insert(0, "/opt/rby1-sim-isaac/app")  # sim_gripper_bridge
sys.path.insert(0, "/tmp")                      # vision.py, servo.py, gripper_example.py (docker cp'd next to this script)
import servo                                    # noqa: E402
from gripper_example import Gripper             # noqa: E402
from sim_gripper_bridge import SimDynamixelBus  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--no-servo", action="store_true", help="stereo measurement only (position-based), for comparison")
parser.add_argument("--color", default="blue", choices=("blue", "green", "yellow"), help="the box to pick")
args = parser.parse_args()

TOOL = 0.12          # fingertips lie this far beyond the ee frame along the tool axis (-z_ee)
TIP_BEYOND = 0.005   # the fingertips end this far past the box centre
RATE = 20.0          # control loop [Hz]
LATENCY = 0.12       # [s] from the instant of a camera image to the arrival of its measurement here
KP_LAT = 1.2         # [1/s] lateral: velocity = KP * offset of the box centre from the tool line
KP_FWD = 1.5         # [1/s] forward: velocity = KP * distance left
V_LAT_MAX, V_FWD_MAX = 0.08, 0.08  # [m/s]
LAT_GATE = 0.02      # [m] forward speed falls to zero as the lateral error approaches this
LEAD_MAX = 0.06      # [m] the commanded target stays within this distance of the measured fingertips (no wind-up)
DONE_REMAIN, DONE_LAT = 0.003, 0.002  # [m] arrival criteria, held for 0.25 s
TIMEOUT = 40.0       # [s]
LIFT = np.array([-0.06, 0.0, 0.12])
LIFT_TIME = 3.5
R_FRONT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)  # tool axis forward, fingers open sideways
TORSO = np.array([0.0, 1.0, -2.0, 1.0, 0.0, 0.0])
READY = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 0.3, -1.57])

# servo state for the viewers: the perception node draws it over the camera images (see perception_node.py)
dbg_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def debug(**info):
    dbg_tx.sendto(json.dumps(info).encode(), ("127.0.0.1", 5103))


def px_of(p_cam):
    return (servo.K @ p_cam)[:2] / p_cam[2]


# latest perception message (UDP from perception_node.py)
msg = {"n": 0, "t": 0.0, "data": None}
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind(("127.0.0.1", 5102))


def listen():
    while True:
        data = json.loads(rx.recv(65536))
        msg["data"], msg["t"] = data, time.time()
        msg["n"] += 1


threading.Thread(target=listen, daemon=True).start()


def wait_messages(count, timeout=8.0):
    n0, t0, out = msg["n"], time.time(), []
    while len(out) < count and time.time() - t0 < timeout:
        if msg["n"] != n0:
            n0 = msg["n"]
            out.append(msg["data"])
        time.sleep(0.02)
    return out


robot = rby.create_robot_a("127.0.0.1:50051")
assert robot.connect(), "cannot reach app_main (is run_sdk.sh running?)"
robot.power_on(".*")
robot.servo_on(".*")
robot.reset_fault_control_manager()
assert robot.enable_control_manager()
gripper = Gripper(SimDynamixelBus("sim"))
assert gripper.initialize()
gripper.homing()
gripper.start()
gripper.set_open()

dyn = robot.get_dynamics()
fk = dyn.make_state(["base", "ee_right"], rby.Model_A().robot_joint_names)
left_hold = dict(zip([f"left_arm_{i}" for i in range(7)], np.array(robot.get_state().position)[15:22]))


def ee_pose():
    fk.set_q(np.array(robot.get_state().position))
    dyn.compute_forward_kinematics(fk)
    return dyn.compute_transformation(fk, 0, 1)


def tip_of(T_ee):
    return T_ee[:3, 3] + TOOL * (T_ee[:3, :3] @ np.array([0.0, 0.0, -1.0]))


def recover():
    robot.reset_fault_control_manager()
    assert robot.enable_control_manager()


def joint_move(body_builder):
    rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(body_builder))
    return robot.send_command(rc, 10).get().finish_code


stream = robot.create_command_stream(10)


def stream_to(tip_xyz):
    """Replace the whole-body Cartesian target; the robot tracks it continuously (target reached within 0.3 s, held 1 s)."""
    T = np.eye(4)
    T[:3, :3] = R_FRONT
    T[:3, 3] = np.asarray(tip_xyz) - TOOL * np.array([1.0, 0.0, 0.0])
    cart = (rby.CartesianCommandBuilder().set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(1.0))
            .set_minimum_time(0.3).add_target("base", "ee_right", T, 0.3, 1.0, 1.0))
    for joint, q in left_hold.items():
        cart.add_joint_position_target(joint, float(q))
    return stream.send_command(rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
        rby.BodyCommandBuilder().set_command(cart))), 200)


# 1. seed pose
recover()
body = rby.BodyComponentBasedCommandBuilder().set_torso_command(
    rby.TorsoCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(6.0).set_position(TORSO)))
print("torso:", joint_move(body), flush=True)
body = rby.BodyComponentBasedCommandBuilder().set_right_arm_command(
    rby.ArmCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(8.0).set_position(READY)))
print("ready:", joint_move(body), flush=True)

# 2. first stereo estimate of the box (landmark association, orientation, fallback measurement)
est = [m for m in wait_messages(6) if m["boxes"].get(args.color, {}).get("n", 0) >= 3]
assert est, f"no {args.color} box pose from the stereo rig (is perception_node.py running? are 3 landmarks visible?)"
box_R = np.array(est[-1]["boxes"][args.color]["R"])
box_t = np.mean([m["boxes"][args.color]["t"] for m in est], axis=0)
print(f"stereo {args.color} box estimate {np.round(box_t, 3)} ({est[-1]['boxes'][args.color]['n']} landmarks)", flush=True)

# 3. continuous servo
hist = collections.deque(maxlen=int(3 * RATE))  # (time, ee pose): the hand pose at the instant of a camera image


def pose_at(t):
    return min(hist, key=lambda h: abs(h[0] - t))[1]


T_ee = ee_pose()
cmd_tip = tip_of(T_ee)
c_world, src, n_seen, view = None, "-", 0, {}
last_n, ok_ticks, t_start, tick, log_t = msg["n"], 0, time.time(), 0, 0.0
dt = 1.0 / RATE
while time.time() - t_start < TIMEOUT:
    t_loop = time.time()
    T_ee = ee_pose()
    hist.append((t_loop, T_ee))
    if msg["n"] != last_n:                       # a new perception message: update the (static) box measurement
        last_n, data = msg["n"], msg["data"]
        if not args.no_servo:
            T_then = pose_at(msg["t"] - LATENCY)
            blobs = np.array(data["wrist"].get(args.color) or np.zeros((0, 3)))
            centre, Z, n_seen = servo.box_in_ee(blobs, servo.cam_pose(T_then), box_R, box_t)
            if centre is not None and n_seen >= 2:
                c_world, src = (T_then @ np.r_[centre, 1])[:3], "image"
                to_cam = np.linalg.inv(servo.CAM_IN_EE)
                pred = servo.predicted_landmarks(servo.cam_pose(T_then), box_R, box_t)[0]
                view = dict(pred=pred.tolist(),
                            matched=[[k, float(p[0]), float(p[1])] for k, p in servo.match_landmarks(blobs, pred)],
                            centre_px=px_of((to_cam @ np.r_[centre, 1])[:3]).tolist(),
                            goal_px=px_of((to_cam @ np.r_[0.0, 0.0, centre[2], 1])[:3]).tolist(), depth_cm=Z * 100)
        if (c_world is None or args.no_servo) and args.color in data["boxes"]:   # no usable image (yet): the stereo measurement
            c_world, src = np.array(data["boxes"][args.color]["t"]), "stereo"
    if c_world is not None:
        c_now = (np.linalg.inv(T_ee) @ np.r_[c_world, 1])[:3]  # box centre in the hand frame now
        lat, remaining = c_now[:2], -c_now[2] - (TOOL - TIP_BEYOND)
        R = T_ee[:3, :3]
        v_lat = R @ np.r_[KP_LAT * lat, 0.0]
        if np.linalg.norm(v_lat) > V_LAT_MAX:
            v_lat *= V_LAT_MAX / np.linalg.norm(v_lat)
        gate = float(np.clip(1.0 - np.hypot(*lat) / LAT_GATE, 0.0, 1.0))
        v_fwd = 0.0 if remaining < DONE_REMAIN else min(V_FWD_MAX, KP_FWD * remaining) * gate
        cmd_tip = cmd_tip + (v_lat + v_fwd * (R @ np.array([0.0, 0.0, -1.0]))) * dt
        lead = cmd_tip - tip_of(T_ee)
        if np.linalg.norm(lead) > LEAD_MAX:      # no wind-up when the robot cannot follow
            cmd_tip = tip_of(T_ee) + lead * LEAD_MAX / np.linalg.norm(lead)
        ok_ticks = ok_ticks + 1 if (remaining < DONE_REMAIN and np.hypot(*lat) < DONE_LAT) else 0
        if tick % 2 == 0:
            debug(stage=f"continuous servo ({src})", err_mm=[lat[0] * 1000, lat[1] * 1000], remaining_mm=remaining * 1000,
                  text=[f"continuous servo, measured by the {src}", f"lateral error {lat[0] * 1000:+.1f} / {lat[1] * 1000:+.1f} mm, "
                        f"{remaining * 1000:.0f} mm to go"], **view)
        if time.time() - log_t > 0.25:
            earlier = [h for h in hist if h[0] <= time.time() - 0.2]  # measured fingertip speed over the last ~0.2 s
            v_meas = np.linalg.norm(tip_of(T_ee) - tip_of(earlier[-1][1])) / (t_loop - earlier[-1][0]) if earlier else 0.0
            print(f"  t={time.time() - t_start:4.1f}s {src:6s} lateral ({lat[0] * 1000:+6.1f}, {lat[1] * 1000:+6.1f}) mm  "
                  f"to go {remaining * 1000:6.1f} mm  commanded {np.linalg.norm(v_lat + v_fwd * (R @ np.array([0, 0, -1.0]))) * 100:4.1f} cm/s"
                  f"  measured {v_meas * 100:4.1f} cm/s", flush=True)
            log_t = time.time()
        if ok_ticks >= int(0.25 * RATE):
            break
    stream_to(cmd_tip)
    tick += 1
    time.sleep(max(0.0, dt - (time.time() - t_loop)))
else:
    print("timed out before arriving", flush=True)
print(f"arrived in {time.time() - t_start:.1f} s", flush=True)

# 4. grasp, then lift along a smooth ramp (also streamed)
debug(stage="grasp", text=["grasp"])
gripper.set_closed()
t0 = time.time()
while time.time() - t0 < 2.0:
    stream_to(cmd_tip)
    time.sleep(dt)
debug(stage="lift", text=["lift"])
base, t0 = cmd_tip.copy(), time.time()
while time.time() - t0 < LIFT_TIME:
    s = min(1.0, (time.time() - t0) / LIFT_TIME)
    stream_to(base + LIFT * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))  # smoothstep: zero velocity at both ends
    time.sleep(dt)
stream_to(base + LIFT)
time.sleep(0.5)
debug(stage="done", text=["done"])
stream.cancel()
gripper.stop()
