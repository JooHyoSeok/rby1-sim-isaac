"""Can both hands hold a part each at the mating plane in front of the chest? Offline inverse kinematics of the two arms (numpy, the FK of
examples/agent/kinematics.py, torso fixed in the squat of examples/gui/grasp_core.py), no simulator.

The insertion axis is the base y axis, the mating plane is x = x_m, z = z_m, y = 0. The female part (left hand) extends to +y, the male part
(right hand) to -y; each hand holds its part GRASP_BACK behind the mating face, so with the parts `gap` apart the two grasp points are 2 * GRASP_BACK + gap
apart. The fingers close across the part, so the closing axis (x of the wrist link, see R_FRONT in grasp_core.py) is vertical, perpendicular to the
part axis; the tool axis points forward, or turned 30 degrees inwards or down. Everything except the hand pose (torso, other joints, the physical
space the arms and fingers take) is checked only by joint limits and the gap between the elbows and forearms.
  /usr/bin/python3 meet_study.py            table of what is reachable
"""
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
import kinematics as K  # noqa: E402

TORSO = K.TORSO
SEED_TORSO = dict(zip(TORSO, [0, 1, -2, 1, 0, 0.0]))
READY = np.array([0.0, 0.0, 0.0, -1.8, 0.0, 0.3, -1.57])
GRASP = np.array([0.0, 0.0, -(0.1261 + 0.12 - 0.005)])      # grasp point in the wrist link frame: the fingertips 5 mm in (TIP_BEYOND_CENTRE)
GRASP_BACK = 0.03                                            # [m] from the mating face to the middle of the fingers
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


def solve(side, p, R, seeds=6, rng=None):
    """Joint values (7) that put the grasp point at p with the wrist link orientation R, or None. Damped least squares from READY, then random seeds."""
    rng = rng or np.random.default_rng(0)
    names = K.RIGHT_ARM if side == "right" else K.LEFT_ARM
    lo = np.array([K.limits(n)[0] for n in names]) + MARGIN
    hi = np.array([K.limits(n)[1] for n in names]) - MARGIN
    for s in range(seeds):
        q = np.clip(READY if s == 0 else rng.uniform(lo, hi), lo, hi)
        for _ in range(120):
            T = _fk(q, side)
            e = np.r_[p - (T[:3, :3] @ GRASP + T[:3, 3]), _rotvec(R @ T[:3, :3].T)]
            if np.linalg.norm(e[:3]) < POS_TOL and np.linalg.norm(e[3:]) < ROT_TOL:
                return q
            J = np.zeros((6, 7))
            for i in range(7):
                dq = q.copy()
                dq[i] += 1e-5
                Td = _fk(dq, side)
                J[:3, i] = ((Td[:3, :3] @ GRASP + Td[:3, 3]) - (T[:3, :3] @ GRASP + T[:3, 3])) / 1e-5
                J[3:, i] = _rotvec(Td[:3, :3] @ T[:3, :3].T) / 1e-5
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), lo, hi)
    return None


def margin(q, side):
    names = K.RIGHT_ARM if side == "right" else K.LEFT_ARM
    return min(min(v - K.limits(n)[0], K.limits(n)[1] - v) for n, v in zip(names, q))


def arm_gap(qr, ql):
    """Smallest distance between the elbows / forearms of the two arms [m]. The wrists and hands are left out: they are 2 * GRASP_BACK + gap apart by design."""
    q = dict(SEED_TORSO) | dict(zip(K.RIGHT_ARM, qr)) | dict(zip(K.LEFT_ARM, ql))
    P = K.all_points(q)
    names = ("elbow", "forearm")
    return min(np.linalg.norm(P[f"right_{a}"] - P[f"left_{b}"]) for a in names for b in names)


VARIANTS = {   # name: (tool axis for the right hand, closing axis); the left hand mirrors y
    "forward, close vertical": ((1, 0, 0), (0, 0, 1)),
    "forward, close vertical (flipped)": ((1, 0, 0), (0, 0, -1)),
    "inward 30 deg, close vertical": ((math.cos(math.radians(30)), math.sin(math.radians(30)), 0), (0, 0, 1)),
    "down 30 deg, close across": ((math.cos(math.radians(30)), 0, -math.sin(math.radians(30))), (math.sin(math.radians(30)), 0, math.cos(math.radians(30)))),
}


def study(xs=(0.30, 0.36, 0.42, 0.48), zs=(0.90, 1.00, 1.10), gaps=(0.03, 0.0)):
    """{variant: {(x, z): (reachable at every gap, smallest elbow/forearm/wrist gap between the arms [m])}}"""
    out = {}
    for name, (tool, closing) in VARIANTS.items():
        grid = {}
        for x, z in itertools.product(xs, zs):
            gaps_seen, ok = [], True
            for gap in gaps:
                sols = {}
                for side, sign in (("right", 1), ("left", -1)):
                    t = np.array([tool[0], sign * tool[1], tool[2]])
                    c = np.array([closing[0], sign * closing[1], closing[2]])
                    # right hand: male part, grasp point at y = -(gap + back); left hand: female, y = +back
                    y = -(gap + GRASP_BACK) if side == "right" else GRASP_BACK
                    sols[side] = solve(side, np.array([x, y, z]), hand_pose(t, c))
                if any(v is None for v in sols.values()):
                    ok = False
                    break
                gaps_seen.append(arm_gap(sols["right"], sols["left"]))
            grid[(x, z)] = (ok, min(gaps_seen) if ok else None)
        out[name] = grid
    return out


if __name__ == "__main__":
    xs, zs = (0.30, 0.36, 0.42, 0.48), (1.10, 1.00, 0.90)
    print(f"both hands hold a part each (axis along base y), grasp point {GRASP_BACK * 100:.0f} cm behind the mating face, parts 3 cm apart and seated;")
    print(f"torso squat fixed, every joint {MARGIN} rad inside its limit. Cell: arm gap in cm (elbows/forearms, smallest of the two gaps), '--' not reachable")
    print("rows z = 1.10 / 1.00 / 0.90 m, columns x = " + " / ".join(f"{x:.2f}" for x in xs))
    for name, grid in study(xs, tuple(sorted(zs))).items():
        print(f"\n{name}")
        for z in zs:
            print("   " + "  ".join("--" if not grid[(x, z)][0] else f"{grid[(x, z)][1] * 100:2.0f}" for x in xs))
