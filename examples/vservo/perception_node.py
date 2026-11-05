"""Perception for the hybrid grasp, in the ROS 2 container: stereo pose of every box + wrist-camera blobs, sent to the executor.

  docker cp examples/vservo/{vision.py,perception_node.py} intelligent-robotic-manipulation:/tmp/
  docker cp ~/docker/isaac-sim/data/rby1_gt.json intelligent-robotic-manipulation:/tmp/calib.json   # stereo extrinsics
  docker exec intelligent-robotic-manipulation bash -c 'source /opt/ros/jazzy/setup.bash;
      export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; cd /tmp; python3 perception_node.py [--bias DX DY DZ]'

Also publishes debug images for RViz and servo_viewer.py (the executor, hybrid_grasp.py, sends its servo state to udp/5103):
  /rby1/servo_debug/wrist_right   blobs (cyan), predicted landmarks (yellow x), matched pairs (green), the measured box centre
                                  (magenta) and where it must be, on the tool line (red cross), with the error in mm
  /rby1/servo_debug/stereo_right  blobs and the box poses fitted from the stereo pair, landmarks reprojected
Sends JSON datagrams to 127.0.0.1:5102 at ~10 Hz:
  {"stamp": sim time [s], "boxes": {"blue": {"R": 3x3, "t": [x, y, z], "n": landmarks used, "rms": m}, ...},   boxes seen with >= 3 landmarks
   "wrist": {"blue": [[u, v, area], ...], ...}}   blobs of each box colour in the right hand's camera
The stereo extrinsics come from rby1_gt.json here (the simulator's ground truth); the real pipeline puts the result of
the eye-to-hand calibration (HW7) there. The servo stage does not depend on them being exact.
"""
import argparse
import json
import socket

import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import vision as V

parser = argparse.ArgumentParser()
parser.add_argument("--calib", default="/tmp/calib.json")
parser.add_argument("--record", default="", metavar="FILE.mp4",
                    help="write the two debug images side by side to this video until the executor reports 'done'")
parser.add_argument("--bias", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"),
                    help="add this to the stereo box positions [m]: emulates a calibration error")
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
                             lambda m, cam=cam: latest.__setitem__(cam, (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, to_array(m), m.header.stamp)), 2)


# servo state from the executor (udp/5103)
dbg = {"t": 0.0, "d": {}}
dbg_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dbg_rx.bind(("127.0.0.1", 5103))


def listen_debug():
    while True:
        dbg["d"], dbg["t"] = json.loads(dbg_rx.recv(65536)), time.time()


threading.Thread(target=listen_debug, daemon=True).start()
writer = None
pub = {n: node.create_publisher(Image, f"/rby1/servo_debug/{n}", 2) for n in ("wrist_right", "stereo_right")}


def text(img, lines, y0=22):
    for i, line in enumerate(lines):  # white on a black outline: readable on any background
        cv2.putText(img, line, (8, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def cross(img, p, size, color, thick=2):
    x, y = int(p[0]), int(p[1])
    cv2.line(img, (x - size, y), (x + size, y), color, thick)
    cv2.line(img, (x, y - size), (x, y + size), color, thick)


def to_msg(img, stamp_src):
    m = Image()
    m.header.stamp, m.header.frame_id = stamp_src, "debug"
    m.height, m.width, m.encoding, m.step = img.shape[0], img.shape[1], "rgb8", img.shape[1] * 3
    m.data = np.ascontiguousarray(img).tobytes()
    return m


def draw_wrist(img, blobs):
    out = img.copy()  # RGB
    d = dbg["d"] if time.time() - dbg["t"] < 4.0 else {}
    for color_blobs in blobs.values():
        for u, v, area in color_blobs:
            cv2.circle(out, (int(u), int(v)), 9, (0, 255, 255), 1)
    for k, (u, v) in enumerate(d.get("pred", [])):
        cross(out, (u, v), 7, (255, 255, 0), 1)
        cv2.putText(out, f"Q{k + 1}", (int(u) + 8, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    for k, u, v in d.get("matched", []):
        pu, pv = d["pred"][int(k)]
        cv2.line(out, (int(pu), int(pv)), (int(u), int(v)), (0, 255, 0), 1)
        cv2.circle(out, (int(u), int(v)), 4, (0, 255, 0), -1)
    if "centre_px" in d:
        cv2.arrowedLine(out, tuple(int(x) for x in d["centre_px"]), tuple(int(x) for x in d["goal_px"]), (255, 0, 255), 2, tipLength=0.15)
        cv2.circle(out, tuple(int(x) for x in d["centre_px"]), 10, (255, 0, 255), 2)
        cross(out, d["goal_px"], 14, (255, 0, 0))
    lines = list(d["text"]) if "text" in d else [d.get("stage", "waiting for the executor")]
    if "centre_px" in d:
        lines.append(f"camera depth {d['depth_cm']:.1f} cm   magenta: measured box centre   red +: where it must be (tool line)")
    text(out, lines)
    return out


def draw_stereo(img, poses):
    out = img.copy()
    for color in ("red", *V.BOX_COLORS):
        for u, v, area in V.blobs(img, color=color, max_area=V.LANDMARK_MAX_AREA):
            cv2.circle(out, (int(u), int(v)), 9, (0, 255, 255), 1)
    lines = ["no box pose"] if not poses else []
    for color, (R, t, rms, n) in poses.items():
        px, _ = V.project(T_R, t + V.OBJECT_MODEL @ R.T)
        for u, v in px:
            cross(out, (u, v), 6, (255, 255, 0), 1)
        c = V.project(T_R, t[None])[0][0]
        cv2.circle(out, (int(c[0]), int(c[1])), 8, (255, 0, 255), 2)
        cv2.putText(out, color, (int(c[0]) + 10, int(c[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        lines.append(f"{color}: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m, {n} landmarks, fit {rms * 1000:.1f} mm")
    text(out, lines)
    return out


def record(wrist_img, stereo_img):
    """Video of the debug images while the executor works: starts with its first message, ends with 'done'."""
    global writer
    if not args.record or "stage" not in dbg["d"] or time.time() - dbg["t"] > 4.0:
        return
    if writer is None and dbg["d"]["stage"] != "done":
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 480))
    if writer is not None:
        writer.write(cv2.cvtColor(np.hstack([wrist_img, stereo_img]), cv2.COLOR_RGB2BGR))
        if dbg["d"]["stage"] == "done":
            writer.release()
            writer = None
            print(f"video written: {args.record}", flush=True)


def tick():
    if not all(k in latest for k in ("stereo_left", "stereo_right", "wrist_right")):
        return
    stamp = latest["stereo_left"][0]
    poses = V.box_poses(latest["stereo_left"][1], latest["stereo_right"][1], T_L, T_R)
    boxes = {c: {"R": R.tolist(), "t": (t + args.bias).tolist(), "rms": rms, "n": n} for c, (R, t, rms, n) in poses.items()}
    wrist_blobs = {c: V.blobs(latest["wrist_right"][1], color=c) for c in V.BOX_COLORS}   # close up the spheres are big: no area limit
    wrist = {c: [list(b) for b in bl] for c, bl in wrist_blobs.items()}
    stamp_msg = latest["stereo_right"][2]
    wrist_img, stereo_img = draw_wrist(latest["wrist_right"][1], wrist_blobs), draw_stereo(latest["stereo_right"][1], poses)
    pub["wrist_right"].publish(to_msg(wrist_img, stamp_msg))
    pub["stereo_right"].publish(to_msg(stereo_img, stamp_msg))
    record(wrist_img, stereo_img)
    sock.sendto(json.dumps({"stamp": stamp, "boxes": boxes, "wrist": wrist}).encode(), ("127.0.0.1", 5102))


node.create_timer(0.1, tick)
print("perception running, sending to udp/5102", flush=True)
rclpy.spin(node)
