"""Pick up the box on the table with the right hand, approaching from the front: IK by the rby1 core (Cartesian commands), gripper over UDP.

Runs inside the simulator container (rby1_sdk is installed only there); the sim must run with
docker/run_sdk.sh ... --gripper --gripper-name rb_gripper --vision:
  docker cp examples/grasp/grasp_object.py examples/gripper_example.py rby1-sim-isaac:/tmp/
  docker exec rby1-sim-isaac /isaac-sim/python.sh /tmp/grasp_object.py
Tip positions below are in the robot "base" frame (= world frame in this scene); targets are sent in link_torso_5.
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
args = parser.parse_args()

OBJECT = np.array([0.62, -0.20, 1.09])  # cube centre (src/rby1_vision.py OBJECT_POS)
TOOL = 0.12      # fingertips lie this far beyond the ee frame along -z_ee (gripper extent in the wrist frame)
BACK = 0.12      # fingertip distance in front of the object at the pre-grasp pose
LIFT = 0.12
# Tool axis (-z_ee) points forward (+x) and the fingers open sideways (x_ee = -y, y_ee = up, z_ee = -x). Reaching the
# object from above needs the arm almost straight (a Cartesian singularity), so the hand comes in horizontally.
R_FRONT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)
TOOL_DIR = np.array([1.0, 0.0, 0.0])
# Right-arm joints (right_arm_0..6) found by a forward-kinematics grid search: hand about (0.41, -0.22, 1.11) with the
# tool axis forward and the elbow well bent; the straight home pose is a Cartesian singularity.
READY = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 0.3, -1.57])

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


def move_joints(q, seconds):
    arm = rby.ArmCommandBuilder().set_command(rby.JointPositionCommandBuilder().set_minimum_time(seconds).set_position(q))
    rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
        rby.BodyComponentBasedCommandBuilder().set_right_arm_command(arm)))
    print(f"ready pose: {robot.send_command(rc, 10).get().finish_code}", flush=True)


dyn = robot.get_dynamics()
fk = dyn.make_state(["base", "link_torso_5", "ee_right"], rby.Model_A().robot_joint_names)


def base_to_torso5():
    """Pose of link_torso_5 in the base frame (the arm's own reference frame for Cartesian targets)."""
    fk.set_q(np.array(robot.get_state().position))
    dyn.compute_forward_kinematics(fk)
    return dyn.compute_transformation(fk, 0, 1)


def pose(tip_xyz):
    T = np.eye(4)
    T[:3, :3] = R_FRONT
    T[:3, 3] = np.asarray(tip_xyz) - TOOL * TOOL_DIR  # the ee frame is TOOL behind the fingertips
    return np.linalg.inv(base_to_torso5()) @ T  # expressed in link_torso_5


def move_ee(tip_xyz, seconds):
    cart = rby.CartesianCommandBuilder().set_minimum_time(seconds).add_target("link_torso_5", "ee_right", pose(tip_xyz), 0.3, 1.0, 1.0)
    rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(
        rby.BodyComponentBasedCommandBuilder().set_right_arm_command(rby.ArmCommandBuilder().set_command(cart))))
    code = robot.send_command(rc, 10).get().finish_code
    print(f"move to tip {np.round(tip_xyz, 3)}: {code}", flush=True)
    if code != rby.RobotCommandFeedback.FinishCode.Ok:
        recover()
    return code


recover()
move_joints(READY, 3.0)
stages = [
    ("pregrasp", lambda: move_ee(OBJECT - [BACK, 0, 0], 4.0)),
    ("approach", lambda: move_ee(OBJECT + [0.005, 0, 0], 3.0)),
    ("close", lambda: (gripper.set_closed(), time.sleep(2.0))),
    ("lift", lambda: move_ee(OBJECT + [0, 0, LIFT], 3.0)),
]
for name, run in stages:
    print(name, flush=True)
    run()
    if name == args.stop_after:
        break
time.sleep(1.0)
gripper.stop()
