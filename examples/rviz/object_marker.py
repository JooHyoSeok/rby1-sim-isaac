"""Show the table and the box in RViz (the URDF only has the robot): /rby1/scene_markers in frame "world".

The box follows /rby1/gt/tf (frame RBY1 -> Object, the simulator's ground truth); the table is fixed. Sizes and the table
pose duplicate src/rby1_vision.py (OBJECT_SIZE, TABLE_*). Run next to RViz with the sim started with --vision:
  /usr/bin/python3 object_marker.py   (ROS 2 python; a venv python3 lacks yaml)
"""
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

OBJECT_SIZE = (0.05, 0.05, 0.08)
TABLE_CENTER, TABLE_SIZE = (0.88, 0.0, 0.70), (0.6, 0.8, 0.04)


def cube(ns, size, rgba, pos):
    m = Marker()
    m.header.frame_id, m.ns, m.id, m.type, m.action = "world", ns, 0, Marker.CUBE, Marker.ADD
    m.scale.x, m.scale.y, m.scale.z = size
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
    m.pose.orientation.w = 1.0
    return m


rclpy.init()
node = Node("scene_markers")
pub = node.create_publisher(MarkerArray, "/rby1/scene_markers", 10)
box = cube("box", OBJECT_SIZE, (0.9, 0.5, 0.1, 1.0), (0.64, -0.2, 0.76))
table = cube("table", TABLE_SIZE, (0.7, 0.6, 0.5, 0.8), TABLE_CENTER)


def on_tf(msg):
    for t in msg.transforms:
        if t.child_frame_id == "Object":
            p, q = t.transform.translation, t.transform.rotation
            box.pose.position.x, box.pose.position.y, box.pose.position.z = p.x, p.y, p.z
            box.pose.orientation = q
    pub.publish(MarkerArray(markers=[box, table]))


node.create_subscription(TFMessage, "/rby1/gt/tf", on_tf, 10)
rclpy.spin(node)
