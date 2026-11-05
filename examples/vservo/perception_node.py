"""Perception for the hybrid grasp, in the ROS 2 container: stereo box pose + palm-camera blobs, sent to the executor.

  docker cp examples/vservo/{vision.py,perception_node.py} intelligent-robotic-manipulation:/tmp/
  docker cp ~/docker/isaac-sim/data/rby1_gt.json intelligent-robotic-manipulation:/tmp/calib.json   # stereo extrinsics
  docker exec intelligent-robotic-manipulation bash -c 'source /opt/ros/jazzy/setup.bash;
      export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; cd /tmp; python3 perception_node.py [--bias DX DY DZ]'

Sends JSON datagrams to 127.0.0.1:5102 at ~10 Hz:
  {"stamp": sim time [s], "box": {"R": 3x3, "t": [x, y, z], "n": landmarks used, "rms": m} or null,
   "wrist": [[u, v, area], ...]}   red blobs of the palm camera (right hand)
The stereo extrinsics come from rby1_gt.json here (the simulator's ground truth); the real pipeline puts the result of
the eye-to-hand calibration (HW7) there. The servo stage does not depend on them being exact.
"""
import argparse
import json
import socket

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import vision as V

parser = argparse.ArgumentParser()
parser.add_argument("--calib", default="/tmp/calib.json")
parser.add_argument("--bias", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"),
                    help="add this to the stereo box position [m]: emulates a calibration error")
args = parser.parse_args()
calib = json.load(open(args.calib))
T_L, T_R = np.array(calib["stereo"]["stereo_left"]), np.array(calib["stereo"]["stereo_right"])
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

rclpy.init()
node = Node("hybrid_perception")
latest = {}


def to_array(m):
    return np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)[:, :, :3]


for cam in ("stereo_left", "stereo_right", "wrist_right"):
    node.create_subscription(Image, f"/rby1/{cam}/image_raw",
                             lambda m, cam=cam: latest.__setitem__(cam, (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, to_array(m))), 2)


def tick():
    if not all(k in latest for k in ("stereo_left", "stereo_right", "wrist_right")):
        return
    stamp = latest["stereo_left"][0]
    pts = V.stereo_points(latest["stereo_left"][1], latest["stereo_right"][1], T_L, T_R)
    pose = V.object_pose(pts) if len(pts) >= 3 else None
    box = None if pose is None else {"R": pose[0].tolist(), "t": (pose[1] + args.bias).tolist(), "rms": pose[2], "n": pose[3]}
    wrist = [list(b) for b in V.blobs(latest["wrist_right"][1])]
    sock.sendto(json.dumps({"stamp": stamp, "box": box, "wrist": wrist}).encode(), ("127.0.0.1", 5102))


node.create_timer(0.1, tick)
print("perception running, sending to udp/5102", flush=True)
rclpy.spin(node)
