"""Inverse kinematics of one RB-Y1 arm by damped least squares (numpy, the FK of kinematics.py next to this file, torso fixed in the squat of
examples/gui/grasp_core.py). Used offline by examples/assembly/meet_study.py and by the executor to plan joint-space moves of both hands."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics as K  # noqa: E402

TORSO = K.TORSO
SEED_TORSO = dict(zip(TORSO, [0, 1, -2, 1, 0, 0.0]))
READY = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 0.3, -1.57])
GRASP_EE = np.array([0.0, 0.0, -0.115])                    # grasp point in the ee frame: the fingertips (0.12 from the ee origin) 5 mm in (TIP_BEYOND_CENTRE)
LINK_TO_EE = np.array([0.0, 0.0, -0.1261])                 # the ee origin in the wrist link frame
MARGIN = 0.1                                                 # rad kept inside every joint limit
POS_TOL, ROT_TOL = 0.001, math.radians(1.0)
_CHAIN = {}


def _fk(q7, side):
    """4x4 pose of the wrist link of `side` for the 7 arm joint values (torso in the squat)."""
    link = f"link_{side}_arm_6"
    if link not in _CHAIN:
        _CHAIN[link] = K.chain(link)
    q = dict(SEED_TORSO) | dict(zip(K.RIGHT_ARM if side == "right" else K.LEFT_ARM, q7))
    T = np.eye(4)
    for j in _CHAIN[link]:
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = K._rpy(j["rpy"]), j["xyz"]
        T = T @ step
        if j["type"] in ("revolute", "continuous"):
            m = np.eye(4)
            m[:3, :3] = K._rot(j["axis"], q.get(j["name"], 0.0))
            T = T @ m
    return T


def fk_ee(q7, side):
    """4x4 pose of the ee frame of `side` (the SDK's "ee_right" / "ee_left") from the URDF chain of kinematics.py."""
    T = _fk(q7, side)
    T[:3, 3] = T[:3, :3] @ LINK_TO_EE + T[:3, 3]
    return T


def _rotvec(R):
    c = (np.trace(R) - 1) / 2
    a = math.acos(max(-1.0, min(1.0, c)))
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * math.sin(a)) * a if a < math.pi - 1e-3 else v / max(np.linalg.norm(v), 1e-9) * a


def hand_pose(tool, closing):
    """Wrist link orientation with the fingertips along `tool` and the closing axis along `closing` (perpendicular unit vectors, base frame)."""
    z = -np.asarray(tool, float)
    x = np.asarray(closing, float)
    return np.column_stack([x, np.cross(z, x), z])


def solve(side, p, R, seeds=6, rng=None, q_start=None, fk=None):
    """Joint values (7) that put the grasp point at p with the hand orientation R, or None. `fk(q7, side)` gives the ee pose (default: fk_ee of the URDF chain;
    the executor passes the SDK's own forward kinematics). Damped least squares from `q_start` (default READY), then random seeds;
    starting from the joints the arm has now keeps the solution close to them, so a straight joint-space move there stays smooth."""
    rng = rng or np.random.default_rng(0)
    fk = fk or fk_ee
    names = K.RIGHT_ARM if side == "right" else K.LEFT_ARM
    lo = np.array([K.limits(n)[0] for n in names]) + MARGIN
    hi = np.array([K.limits(n)[1] for n in names]) - MARGIN
    for s in range(seeds):
        q = np.clip((READY if q_start is None else np.asarray(q_start, float)) if s == 0 else rng.uniform(lo, hi), lo, hi)
        for _ in range(120):
            T = fk(q, side)
            e = np.r_[p - (T[:3, :3] @ GRASP_EE + T[:3, 3]), _rotvec(R @ T[:3, :3].T)]
            if np.linalg.norm(e[:3]) < POS_TOL and np.linalg.norm(e[3:]) < ROT_TOL:
                return q
            J = np.zeros((6, 7))
            for i in range(7):
                dq = q.copy()
                dq[i] += 1e-5
                Td = fk(dq, side)
                J[:3, i] = ((Td[:3, :3] @ GRASP_EE + Td[:3, 3]) - (T[:3, :3] @ GRASP_EE + T[:3, 3])) / 1e-5
                J[3:, i] = _rotvec(Td[:3, :3] @ T[:3, :3].T) / 1e-5
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), lo, hi)
    return None


def margin(q, side):
    names = K.RIGHT_ARM if side == "right" else K.LEFT_ARM
    return min(min(v - K.limits(n)[0], K.limits(n)[1] - v) for n, v in zip(names, q))


def arm_gap(qr, ql):
    """Smallest distance between the elbows / forearms of the two arms [m]. The wrists and hands are left out: two hands holding parts to be mated are close by design."""
    q = dict(SEED_TORSO) | dict(zip(K.RIGHT_ARM, qr)) | dict(zip(K.LEFT_ARM, ql))
    P = K.all_points(q)
    names = ("elbow", "forearm")
    return min(np.linalg.norm(P[f"right_{a}"] - P[f"left_{b}"]) for a in names for b in names)
