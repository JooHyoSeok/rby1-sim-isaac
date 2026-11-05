"""Result of a grasp, from the simulator's ground truth (ROS 2 container): box height and finger gap.
  python3 check_result.py [blue|green|yellow]   prints one line and exits 0 when the box is lifted, 1 otherwise
"""
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

LIFTED_Z = 0.80  # the box centre rests at z = 0.76 on the table
BOX = sys.argv[1] if len(sys.argv) > 1 else "blue"
rclpy.init()
node = Node("check_result")
got = {}
node.create_subscription(TFMessage, "/rby1/gt/tf", lambda m: [got.__setitem__("box", (t.transform.translation.x, t.transform.translation.y,
                         t.transform.translation.z)) for t in m.transforms if t.child_frame_id == f"Box_{BOX}"], 10)
node.create_subscription(JointState, "/rby1/joint_states", lambda m: got.__setitem__(
    "gap", 2 * abs(dict(zip(m.name, m.position)).get("gripper_finger_r1", 0.0))), 10)
t0 = time.time()
while time.time() - t0 < 8 and not ("box" in got and "gap" in got):
    rclpy.spin_once(node, timeout_sec=0.2)
if "box" not in got:
    print("no ground truth received")
    sys.exit(2)
x, y, z = got["box"]
lifted = z > LIFTED_Z
print(f"box centre ({x:.3f}, {y:.3f}, {z:.3f}) m, right finger travel {got.get('gap', 0) / 2 * 1000:.0f} mm each: "
      + ("LIFTED" if lifted else "NOT lifted"))
sys.exit(0 if lifted else 1)
