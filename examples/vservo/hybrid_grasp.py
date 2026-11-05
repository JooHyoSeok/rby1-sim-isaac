"""Hybrid visual-servoing grasp: stereo position-based approach, palm-camera image-based fine alignment, then grasp.

Runs inside the simulator container (rby1_sdk is installed only there) next to perception_node.py (ROS 2 container).
The sim must run with docker/run_sdk.sh ... --gripper --gripper-name rb_gripper --vision:
  docker cp examples/vservo/{vision.py,servo.py,hybrid_grasp.py} examples/gripper_example.py rby1-sim-isaac:/tmp/
  docker exec rby1-sim-isaac /isaac-sim/python.sh /tmp/hybrid_grasp.py

  1. seed pose (torso squat, bent right arm), as in examples/grasp/grasp_object.py
  2. position-based: box pose from the stereo rig -> whole-body Cartesian move to a pre-grasp tip 12 cm in front of it
  3. image-based: the palm camera sees the box landmarks; the pixel error against "box centre on the optical axis" is turned
     into a lateral correction, repeated until it is below 2 px. This removes the steady-state error of the position loop and
     of the stereo calibration, which the position-based step cannot see. Orientation stays with the position loop.
  4. depth from the image (landmark spread) sets how far to advance, then close the gripper and lift.
Tip positions are in the robot base frame (= world frame in this scene). Untested in the simulator.
"""
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

TOOL = 0.12      # fingertips lie this far beyond the ee frame along -z_ee
BACK = 0.12      # pre-grasp: fingertip distance in front of the box centre
CAM_TO_TIP = 0.12 - 0.0789  # palm camera -> fingertip along the tool axis
GRASP_DEPTH = CAM_TO_TIP - 0.005   # camera-to-box-centre distance at the grasp (the tip ends 5 mm past the centre)
CONVERGED_PX = 2.0
LIFT = [-0.06, 0.0, 0.12]
R_FRONT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)  # tool axis forward, fingers open sideways
TOOL_DIR = np.array([1.0, 0.0, 0.0])
TORSO = np.array([0.0, 1.0, -2.0, 1.0, 0.0, 0.0])
READY = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 0.3, -1.57])

# latest perception message (UDP from perception_node.py)
msg = {"t": 0.0, "data": None}
rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind(("127.0.0.1", 5102))


def listen():
    while True:
        msg["data"], msg["t"] = json.loads(rx.recv(65536)), time.time()


threading.Thread(target=listen, daemon=True).start()


def fresh(max_age=0.6, timeout=5.0):
    """Wait for a perception message that arrived after the robot stopped moving."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if msg["data"] is not None and time.time() - msg["t"] < max_age and msg["t"] > t0:
            return msg["data"]
        time.sleep(0.05)
    return None


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


def tip_position():
    return ee_pose()[:3, 3] + TOOL * TOOL_DIR


def recover():
    robot.reset_fault_control_manager()
    assert robot.enable_control_manager()


def joint_move(body_builder):
    rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(body_builder))
    return robot.send_command(rc, 10).get().finish_code


def move_ee(tip_xyz, seconds):
    T = np.eye(4)
    T[:3, :3] = R_FRONT
    T[:3, 3] = np.asarray(tip_xyz) - TOOL * TOOL_DIR
    cart = rby.CartesianCommandBuilder().set_minimum_time(seconds).add_target("base", "ee_right", T, 0.3, 1.0, 1.0)
    for joint, q in left_hold.items():
        cart.add_joint_position_target(joint, float(q))
    handler = robot.send_command(rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
        rby.BodyCommandBuilder().set_command(cart))), 10)
    if not handler.wait_for(int((seconds + 8.0) * 1000)):
        handler.cancel()
        print("  (timed out)", flush=True)
    code = handler.get().finish_code
    if code != rby.RobotCommandFeedback.FinishCode.Ok:
        recover()
    return code


# 1. seed pose
recover()
body = rby.BodyComponentBasedCommandBuilder().set_torso_command(
    rby.TorsoCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(6.0).set_position(TORSO)))
print("torso:", joint_move(body), flush=True)
body = rby.BodyComponentBasedCommandBuilder().set_right_arm_command(
    rby.ArmCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(8.0).set_position(READY)))
print("ready:", joint_move(body), flush=True)

# 2. position-based: stereo estimate of the box pose, pre-grasp in front of it
est = [m for m in (fresh() for _ in range(5)) if m and m["box"] and m["box"]["n"] >= 3]
assert est, "no box pose from the stereo rig (is perception_node.py running? are 3 landmarks visible?)"
box_R = np.array(est[-1]["box"]["R"])
box_t = np.mean([m["box"]["t"] for m in est], axis=0)
print(f"stereo box estimate {np.round(box_t, 3)} ({est[-1]['box']['n']} landmarks)", flush=True)
print("pre-grasp:", move_ee(box_t - [BACK, 0, 0], 4.0), flush=True)

# 3. image-based fine alignment with the palm camera
Z = None
for i in range(10):
    m = fresh()
    blobs = np.array(m["wrist"]) if m and m["wrist"] else np.zeros((0, 3))
    T_ee = ee_pose()
    err, Z, n = servo.image_errors(blobs, servo.cam_pose(T_ee), box_R, box_t)
    if err is None:
        print("  no landmark in the palm camera; keeping the position-based pose", flush=True)
        break
    print(f"  iter {i}: pixel error ({err[0]:5.1f}, {err[1]:5.1f}), {n} landmarks, depth {Z * 100:.1f} cm", flush=True)
    if np.linalg.norm(err) < CONVERGED_PX:
        break
    move_ee(tip_position() + servo.lateral_step(err, Z, T_ee), 1.5)

# 4. advance by the depth seen in the image, grasp, lift
advance = (Z - GRASP_DEPTH) if Z else (box_t[0] + 0.005 - tip_position()[0])  # image depth, else the stereo estimate
advance = float(np.clip(advance, 0.0, BACK + 0.05))
print(f"advance {advance * 100:.1f} cm:", move_ee(tip_position() + advance * TOOL_DIR, 3.0), flush=True)
gripper.set_closed()
time.sleep(2.0)
print("lift:", move_ee(tip_position() + LIFT, 4.0), flush=True)
time.sleep(1.0)
gripper.stop()
