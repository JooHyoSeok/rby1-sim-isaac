"""Save one synchronized snapshot (4 RGB images + joint state + sim timestamps) on every "SAVE <i>" UDP message.

Runs in the ROS 2 container, with the simulator started via docker/run_sdk.sh ... --vision:
  export FASTDDS_BUILTIN_TRANSPORTS=UDPv4   # the sim container's DDS shared memory does not reach this container
  python3 record_snapshots.py --out /tmp/calib_data
Replies "OK <i> <red pixel counts>" so the mover knows the file is written.
"""
import argparse
import os
import socket
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

CAMS = ("stereo_left", "stereo_right", "wrist_left", "wrist_right")

p = argparse.ArgumentParser()
p.add_argument("--out", default="/tmp/calib_data")
p.add_argument("--port", type=int, default=5101)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

rclpy.init()
node = Node("record_snapshots")
latest = {}  # name -> (stamp [s], data)


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def on_image(name):
    def cb(m):
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
        latest[name] = (stamp(m), img.copy())
    return cb


for c in CAMS:
    node.create_subscription(Image, f"/rby1/{c}/image_raw", on_image(c), 10)
node.create_subscription(JointState, "/rby1/joint_states",
                         lambda m: latest.__setitem__("joints", (stamp(m), m)), 10)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", args.port))
sock.setblocking(False)
print(f"waiting for SAVE on udp/{args.port}, writing to {args.out}", flush=True)


def red_pixels(img):
    r, g, b = (img[:, :, k].astype(int) for k in range(3))
    return int(((r > 150) & (g < 60) & (b < 60)).sum())


while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.05)
    try:
        msg, addr = sock.recvfrom(256)
    except BlockingIOError:
        continue
    idx = int(msg.decode().split()[1])
    t0 = time.time()
    while time.time() - t0 < 0.4:  # drain messages queued while the robot was still moving
        rclpy.spin_once(node, timeout_sec=0.05)
    latest.clear()  # the robot is settled; wait for a fresh complete set
    t0 = time.time()
    while len(latest) < len(CAMS) + 1 and time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if len(latest) < len(CAMS) + 1:
        sock.sendto(f"FAIL {idx} missing {sorted(set(CAMS + ('joints',)) - set(latest))}".encode(), addr)
        continue
    js = latest["joints"][1]
    stamps = {k: v[0] for k, v in latest.items()}
    np.savez_compressed(f"{args.out}/snap_{idx:03d}.npz", **{c: latest[c][1] for c in CAMS},
                        joint_names=np.array(js.name), joint_position=np.array(js.position),
                        joint_velocity=np.array(js.velocity), stamps=np.array(list(stamps.values())),
                        stamp_keys=np.array(list(stamps)))
    skew = max(stamps.values()) - min(stamps.values())
    sock.sendto(f"OK {idx} red={[red_pixels(latest[c][1]) for c in CAMS]} skew={skew:.3f}s".encode(), addr)
