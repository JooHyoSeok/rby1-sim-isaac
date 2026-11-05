"""Republish /tf without the flat "RBY1 -> link" frames as /tf_chain (only the articulation chain stays, one parent per frame).

Run in the ROS 2 container next to the sim started with --vision, then view with
  rviz2 -d examples/rviz/rby1_tf.rviz --ros-args -r /tf:=/tf_chain
"""
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

rclpy.init()
n = Node("tf_chain_relay")
pub = n.create_publisher(TFMessage, "/tf_chain", 50)


def cb(msg):
    keep = [t for t in msg.transforms if t.header.frame_id != "RBY1"]
    if keep:
        pub.publish(TFMessage(transforms=keep))


n.create_subscription(TFMessage, "/tf", cb, 50)
rclpy.spin(n)
