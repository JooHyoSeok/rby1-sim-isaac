"""Forward kinematics of the RB-Y1 from rby1_kinematics.json (numpy only; see make_kinematics.py). Frame: the SDK's "base" (the robot
stands at the origin of the world, x forward, y left, z up). Joint values in rad; joints not given count as 0."""
import json
import os

import numpy as np

TREE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rby1_kinematics.json")))
JOINTS = {j["name"]: j for j in TREE["joints"]}
_PARENT_JOINT = {j["child"]: j for j in TREE["joints"]}
TORSO = [f"torso_{i}" for i in range(6)]
RIGHT_ARM = [f"right_arm_{i}" for i in range(7)]
LEFT_ARM = [f"left_arm_{i}" for i in range(7)]
HEAD = ["head_0", "head_1"]
# points on an arm that matter for keeping clear of things: (link, offset in the link frame). The hand: the wrist link origin, the ee frame
# lies 0.1261 below it and the fingertips 0.12 further along -z (examples/vservo/servo.py LINK_TO_EE, grasp_core TOOL). The left arm mirrors the
# right one (same link layout; the sideways and roll joints 1, 2, 4, 6 have the opposite sign, e.g. positive left_arm_1 lifts it sideways).
POINTS_BY_ARM = {side: {"elbow": (f"link_{side}_arm_3", (0, 0, 0)), "forearm": (f"link_{side}_arm_4", (0, 0, 0)), "wrist": (f"link_{side}_arm_6", (0, 0, 0)),
                        "hand_mid": (f"link_{side}_arm_6", (0, 0, -0.12)), "hand_tip": (f"link_{side}_arm_6", (0, 0, -0.2461))} for side in ("right", "left")}
POINTS = POINTS_BY_ARM["right"]


def _rot(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _rpy(rpy):
    r, p, y = rpy   # URDF: R = Rz(yaw) Ry(pitch) Rx(roll)
    return _rot((0, 0, 1), y) @ _rot((0, 1, 0), p) @ _rot((1, 0, 0), r)


def chain(link):
    """Joints from the root to `link`, in order."""
    out = []
    while link in _PARENT_JOINT:
        out.append(_PARENT_JOINT[link])
        link = _PARENT_JOINT[link]["parent"]
    return out[::-1]


def link_pose(q, link):
    """4x4 pose of `link` in the base frame for joint values q = {name: rad}."""
    T = np.eye(4)
    for j in chain(link):
        step = np.eye(4)
        step[:3, :3] = _rpy(j["rpy"])
        step[:3, 3] = j["xyz"]
        T = T @ step
        if j["type"] in ("revolute", "continuous"):
            m = np.eye(4)
            m[:3, :3] = _rot(j["axis"], q.get(j["name"], 0.0))
            T = T @ m
    return T


def points(q, arm="right"):
    """{name: xyz} of the key points of one arm in the base frame."""
    return {name: (link_pose(q, link) @ np.r_[off, 1.0])[:3] for name, (link, off) in POINTS_BY_ARM[arm].items()}


def all_points(q):
    """{"right_hand_tip": xyz, ..., "left_elbow": xyz, ...} for both arms."""
    return {f"{side}_{name}": p for side in ("right", "left") for name, p in points(q, side).items()}


def tip_direction(q, arm="right"):
    """Unit vector the fingertips point along (the tool axis -z of the wrist link)."""
    return -link_pose(q, f"link_{arm}_arm_6")[:3, 2]


def head_forward(q):
    """Unit vector the head looks along (x of the last head link)."""
    return link_pose(q, "link_head_2")[:3, 0]


def limits(name):
    j = JOINTS[name]
    return (j["lower"], j["upper"])
