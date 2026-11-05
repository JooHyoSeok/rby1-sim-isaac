"""Pick up the box on the table with the right hand, approaching from the front.

Whole-body IK by the rby1 core: one Cartesian command at body level (torso + both arms), with soft joint-position
targets that keep the torso squatted, the wrist away from its singularity and the left arm where it is. The gripper
is driven over UDP.

Runs inside the simulator container (rby1_sdk is installed only there); the sim must run with
docker/run_sdk.sh ... --gripper --gripper-name rb_gripper --vision:
  docker cp examples/grasp/grasp_object.py examples/gripper_example.py rby1-sim-isaac:/tmp/
  docker exec rby1-sim-isaac /isaac-sim/python.sh /tmp/grasp_object.py
Tip positions are in the robot "base" frame (= world frame in this scene).
"""
import argparse
import sys
import time

import numpy as np
import rby1_sdk as rby

sys.path.insert(0, "/opt/rby1-sim-isaac/app")  # sim_gripper_bridge
sys.path.insert(0, "/tmp")                      # gripper_example (docker cp'd next to this script)
from gripper_example import Gripper             # noqa: E402
from sim_gripper_bridge import SimDynamixelBus  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--stop-after", choices=["pregrasp", "approach", "close", "lift"], default="lift",
                    help="stop after this stage (to look at the scene)")
parser.add_argument("--posture", choices=["none", "wrist", "torso", "full"], default="none",
                    help="soft joint targets for the whole-body IK: none, wrist only, torso only, or both")
args = parser.parse_args()

OBJECT = np.array([0.64, -0.20, 0.76])  # box centre (src/rby1_vision.py OBJECT_POS)
TOOL = 0.12      # fingertips lie this far beyond the ee frame along -z_ee (gripper extent in the wrist frame)
BACK = 0.12      # fingertip distance in front of the object at the pre-grasp pose
LIFT = [-0.06, 0.0, 0.12]  # lift: up and a little back toward the body
# Tool axis (-z_ee) points forward (+x) and the fingers open sideways (x_ee = -y, y_ee = up, z_ee = -x). Reaching the
# object from above needs the arm almost straight (a Cartesian singularity), so the hand comes in horizontally.
R_FRONT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)
TOOL_DIR = np.array([1.0, 0.0, 0.0])
# Seed configuration, reached with joint moves because the straight home pose is a Cartesian singularity: torso folded
# like a squat (shoulders drop from z=1.37 to 1.05), right arm bent with the tool axis forward.
TORSO = np.array([0.0, 1.0, -2.0, 1.0, 0.0, 0.0])
READY = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 0.3, -1.57])
# Soft posture targets for the whole-body IK (they resolve the redundancy; the hand target still wins).
POSTURES = {
    "none": {},
    "wrist": {"right_arm_5": 0.5},
    "torso": {f"torso_{i}": q for i, q in enumerate(TORSO)},
}
POSTURES["full"] = {**POSTURES["wrist"], **POSTURES["torso"]}
POSTURE = POSTURES[args.posture]

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


def recover():
    robot.reset_fault_control_manager()
    assert robot.enable_control_manager()


def move_torso(q, seconds):
    torso = rby.TorsoCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(seconds).set_position(q))
    rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
        rby.BodyComponentBasedCommandBuilder().set_torso_command(torso)))
    print(f"torso: {robot.send_command(rc, 10).get().finish_code}", flush=True)


def move_joints(q, seconds):
    arm = rby.ArmCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(seconds).set_position(q))
    rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
        rby.BodyComponentBasedCommandBuilder().set_right_arm_command(arm)))
    print(f"joint move: {robot.send_command(rc, 10).get().finish_code}", flush=True)


dyn = robot.get_dynamics()
fk = dyn.make_state(["base", "ee_right"], rby.Model_A().robot_joint_names)
LEFT_ARM = [f"left_arm_{i}" for i in range(7)]
left_hold = dict(zip(LEFT_ARM, np.array(robot.get_state().position)[15:22]))  # keep the idle arm where it is


def tip_position():
    fk.set_q(np.array(robot.get_state().position))
    dyn.compute_forward_kinematics(fk)
    ee = dyn.compute_transformation(fk, 0, 1)
    return ee[:3, 3] + TOOL * TOOL_DIR


def pose(tip_xyz):
    T = np.eye(4)
    T[:3, :3] = R_FRONT
    T[:3, 3] = np.asarray(tip_xyz) - TOOL * TOOL_DIR  # the ee frame is TOOL behind the fingertips
    return T


def move_ee(tip_xyz, seconds):
    cart = rby.CartesianCommandBuilder().set_minimum_time(seconds).add_target("base", "ee_right", pose(tip_xyz), 0.3, 1.0, 1.0)
    for joint, q in {**POSTURE, **left_hold}.items():
        cart.add_joint_position_target(joint, float(q))
    rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
        rby.BodyCommandBuilder().set_command(cart)))
    handler = robot.send_command(rc, 10)
    if not handler.wait_for(int((seconds + 8.0) * 1000)):  # the command never finishes when the IK stalls short of the target
        handler.cancel()
        print("  (timed out: IK did not reach the target)", flush=True)
    code = handler.get().finish_code
    err = np.linalg.norm(tip_position() - np.asarray(tip_xyz))
    print(f"whole-body move to tip {np.round(tip_xyz, 3)}: {code}, tip error {err * 1000:.1f} mm", flush=True)
    if code != rby.RobotCommandFeedback.FinishCode.Ok:
        recover()
    return code


recover()
move_torso(TORSO, 6.0)
move_joints(READY, 8.0)
stages = [
    ("pregrasp", lambda: move_ee(OBJECT - [BACK, 0, 0], 4.0)),
    ("approach", lambda: move_ee(OBJECT + [0.005, 0, 0], 3.0)),
    ("close", lambda: (gripper.set_closed(), time.sleep(2.0))),
    ("lift", lambda: move_ee(OBJECT + LIFT, 4.0)),
]
for name, run in stages:
    print(name, flush=True)
    run()
    if name == args.stop_after:
        break
time.sleep(1.0)
gripper.stop()
