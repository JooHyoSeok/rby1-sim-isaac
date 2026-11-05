"""Safety check for joint trajectories that a language model designs. Runs on the host before anything is sent to the robot (numpy only).

A trajectory is {"unit": "deg" | "rad", "waypoints": [{"t": seconds from the previous waypoint, "<joint>": value, ...}, ...]}. A joint that a
waypoint does not mention keeps its value from the one before (the first one: the robot's current value). `check` returns every violation it
finds, each with the numbers, so the model can fix its plan; the robot moves only when there is none.

What is checked: only the joints of both arms, the head and the torso yaw may move; each stays inside its limit (URDF) minus a margin
(and the smaller ranges below); joint speed and the length of a segment and of the whole motion; and along the motion (sampled every
few degrees) the elbow, forearm, wrist and hand stay above the table and the floor, off the body's middle, and clear of the boxes. There is no
collision model of the hand and the gripper in the simulator's USD, so the clearance rules are deliberately generous. These rules are
code, not prompt: the model cannot talk its way past them.
"""
import numpy as np

import kinematics as K

ALLOWED = set(K.RIGHT_ARM) | set(K.LEFT_ARM) | set(K.HEAD) | {"torso_5"}
RANGE = {"torso_5": (-0.6, 0.6), "head_0": (-1.2, 1.2), "head_1": (-0.6, 0.6)}   # tighter than the URDF
MARGIN = 0.10                                    # rad kept inside the URDF limit
V_MAX = {**{n: 0.8 for n in K.RIGHT_ARM + K.LEFT_ARM}, "head_0": 1.0, "head_1": 1.0, "torso_5": 0.3}   # rad/s
ARM_GAP = 0.15                                   # the two arms (elbow, forearm, wrist, hand) stay this far apart [m]
MIN_SEGMENT, MAX_TOTAL, MAX_WAYPOINTS = 0.3, 30.0, 16
TABLE_TOP, TABLE_CLEAR = 0.72, 0.10              # points over the table stay this far above it ...
TABLE_X, TABLE_Y = 0.55, 0.50                    # ... where x > TABLE_X and |y| < TABLE_Y (the default table's front edge is at 0.58; the READY pose hovers just in front of it)
FLOOR_CLEAR = 0.20
CHEST_X, CHEST_Z, CHEST_Y = 0.25, 0.5, 0.12      # near the body (x < CHEST_X, z > CHEST_Z) a hand stays on its own side: right hand y <= -CHEST_Y, left hand y >= CHEST_Y
BOX_SIZE, BOX_CLEAR = (0.05, 0.05, 0.08), 0.04   # boxes are obstacles with this margin all around
STEP = np.radians(3.0)                           # joint-space sampling step along a segment


def _unit(traj):
    unit = traj.get("unit", "deg")
    if unit not in ("deg", "rad"):
        raise ValueError(f"unit must be 'deg' or 'rad', not {unit!r}")
    return np.pi / 180.0 if unit == "deg" else 1.0


def expand(current, traj):
    """[(t, {joint: rad}, not-allowed joint names, joints set here)] with every allowed joint filled in (carried forward), t = seconds from the
    previous waypoint. Raises ValueError for a malformed trajectory."""
    k = _unit(traj)
    wps, state, out = traj.get("waypoints"), {n: float(current.get(n, 0.0)) for n in ALLOWED}, []
    if not isinstance(wps, list) or not wps:
        raise ValueError('"waypoints" must be a non-empty list')
    for i, w in enumerate(wps, 1):
        if not isinstance(w, dict) or not isinstance(w.get("t"), (int, float)):
            raise ValueError(f'waypoint {i}: needs "t" (seconds) and joint values')
        for name, v in w.items():
            if name == "t":
                continue
            if name not in K.JOINTS:
                raise ValueError(f"waypoint {i}: unknown joint {name!r}")
            if not isinstance(v, (int, float)):
                raise ValueError(f"waypoint {i}: {name} must be a number")
            if name in ALLOWED:
                state[name] = float(v) * k
        out.append((float(w["t"]), dict(state), [n for n in w if n != "t" and n not in ALLOWED], {n for n in w if n in ALLOWED}))
    return out


def _limit(name):
    lo, hi = K.limits(name)
    lo, hi = lo + MARGIN, hi - MARGIN
    if name in RANGE:
        lo, hi = max(lo, RANGE[name][0]), min(hi, RANGE[name][1])
    return lo, hi


def check(current, traj, boxes=None, base=None):
    """(violations, summary). `current`: {joint: rad} now; `boxes`: {name: (x, y, z)} centres in the base frame or None; `base`: (x, y, yaw [rad]) of the
    base in the world when it is not at its start pose (the table is where it is in the world, everything else here is in the base frame). The summary lists where the hand tip, the
    wrist and the elbow are at each waypoint (m, base frame) and the lowest points, so a valid plan can still be improved."""
    try:
        plan = expand(current, traj)
    except ValueError as exc:
        return [str(exc)], {}
    v, deg = [], 180.0 / np.pi
    if len(plan) > MAX_WAYPOINTS:
        v.append(f"at most {MAX_WAYPOINTS} waypoints per trajectory (got {len(plan)})")
    if sum(p[0] for p in plan) > MAX_TOTAL:
        v.append(f"the motion lasts {sum(p[0] for p in plan):.1f} s, at most {MAX_TOTAL:.0f} s")
    for i, (t, q, not_allowed, given) in enumerate(plan, 1):
        for name in not_allowed:
            v.append(f"waypoint {i}: {name} may not be moved (allowed: right_arm_0..6, left_arm_0..6, head_0, head_1, torso_5)")
        if t < MIN_SEGMENT:
            v.append(f"waypoint {i}: t = {t:.2f} s is too short (at least {MIN_SEGMENT} s)")
        for name in given:           # a value taken over from the robot's current pose is not the plan's to answer for (the arm hangs at its limits)
            val = q[name]
            lo, hi = _limit(name)
            if not lo - 1e-9 <= val <= hi + 1e-9:
                v.append(f"waypoint {i}: {name} = {val * deg:.1f} deg is outside [{lo * deg:.1f}, {hi * deg:.1f}] deg")
    prev = {n: float(current.get(n, 0.0)) for n in ALLOWED}
    for i, (t, q, _, _) in enumerate(plan, 1):
        for name in ALLOWED:
            speed = abs(q[name] - prev[name]) / max(t, 1e-6)
            if speed > V_MAX[name] + 1e-9:
                need = abs(q[name] - prev[name]) / V_MAX[name]
                v.append(f"waypoint {i}: {name} would turn at {speed * deg:.0f} deg/s, at most {V_MAX[name] * deg:.0f}: use t >= {need:.1f} s")
        prev = q
    summary = {"duration_s": round(sum(p[0] for p in plan), 2), "waypoints": []}
    lowest, clear = {}, {}
    full = dict(current)
    prev = {n: float(current.get(n, 0.0)) for n in ALLOWED}
    for i, (t, q, _, _) in enumerate(plan, 1):
        n_steps = max(1, int(np.ceil(max(abs(q[n] - prev[n]) for n in ALLOWED) / STEP)))
        for s in range(1, n_steps + 1):
            a = s / n_steps
            full.update({n: prev[n] + a * (q[n] - prev[n]) for n in ALLOWED})
            pts = K.all_points(full)
            for name, p in pts.items():
                lowest[name] = min(lowest.get(name, 9.0), float(p[2]))
                for rule, m in _clearance(name, p, boxes, base):
                    clear.setdefault((i, name, rule), f"waypoint {i} (at {a * 100:.0f}% of the way there): {m}")   # the first sample that breaks it
            gap = min((np.linalg.norm(pts[f"right_{r}"] - pts[f"left_{l}"]), r, l) for r in ARM_POINTS for l in ARM_POINTS)
            if gap[0] < ARM_GAP:
                clear.setdefault((i, "arms", "gap"), f"waypoint {i} (at {a * 100:.0f}% of the way there): the arms would touch: right {gap[1]} and left {gap[2]} "
                                                    f"only {gap[0] * 100:.0f} cm apart (keep {ARM_GAP * 100:.0f} cm)")
            if s == n_steps:
                summary["waypoints"].append({"index": i, "t_total": round(sum(p[0] for p in plan[:i]), 2),
                                             **{n: [round(float(c), 3) for c in pts[f"right_{n}"]] for n in ("hand_tip", "wrist", "elbow")},
                                             **{f"left_{n}": [round(float(c), 3) for c in pts[f"left_{n}"]] for n in ("hand_tip", "wrist", "elbow")}})
        prev = q
    summary["lowest_z"] = {n: round(z, 3) for n, z in lowest.items()}
    return (v + list(clear.values()))[:12], summary


ARM_POINTS = ("elbow", "forearm", "wrist", "hand_mid", "hand_tip")


def _clearance(name, p, boxes, base=None):
    """[(rule, message)] for one point of the arm: floor, table, chest, boxes."""
    x, y, z = (float(c) for c in p)
    wx, wy = (x, y) if base is None else (base[0] + np.cos(base[2]) * x - np.sin(base[2]) * y, base[1] + np.sin(base[2]) * x + np.cos(base[2]) * y)
    out = []
    if z < FLOOR_CLEAR:
        out.append(("floor", f"{name} at z = {z:.2f} m is below the floor clearance ({FLOOR_CLEAR} m)"))
    if wx > TABLE_X and abs(wy) < TABLE_Y and z < TABLE_TOP + TABLE_CLEAR:
        out.append(("table", f"{name} at ({x:.2f}, {y:.2f}, {z:.2f}) is within {TABLE_CLEAR * 100:.0f} cm of the table (top {TABLE_TOP} m): keep z >= {TABLE_TOP + TABLE_CLEAR:.2f}"))
    side = name.split("_")[0]
    wrong_side = y > -CHEST_Y if side == "right" else y < CHEST_Y
    if name.split("_", 1)[1] in ("wrist", "hand_mid", "hand_tip") and x < CHEST_X and z > CHEST_Z and wrong_side:
        out.append(("chest", f"{name} at ({x:.2f}, {y:.2f}, {z:.2f}) is too close to the body (head and chest are not modelled): keep y {'<= -' if side == 'right' else '>= '}{CHEST_Y} "
                             f"({side} of the body) or x >= {CHEST_X}"))
    for b, (bx, by, bz) in (boxes or {}).items():
        if abs(x - bx) < BOX_SIZE[0] / 2 + BOX_CLEAR and abs(y - by) < BOX_SIZE[1] / 2 + BOX_CLEAR and abs(z - bz) < BOX_SIZE[2] / 2 + BOX_CLEAR:
            out.append((f"box {b}", f"{name} at ({x:.2f}, {y:.2f}, {z:.2f}) would touch the {b} box"))
    return out
