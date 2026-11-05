"""Shared by the control centre (host) and the executor service (simulator container): ports, scene defaults, reachability.

Plain python (no numpy, no Qt) so both sides can import it. The scene defaults repeat src/rby1_vision.py.
"""
import math

# UDP ports, all on 127.0.0.1 (the containers share the host network)
PORT_SCENE = 5110        # control centre -> simulator: move the box / table (src/rby1_vision.py poll_commands)
PORT_EXEC_CMD = 5120     # control centre -> executor service: pick, place, ...
PORT_EXEC_STATUS = 5121  # executor service -> control centre: status and log lines

# Scene defaults (src/rby1_vision.py)
TABLE_CENTER = (0.88, 0.0)
TABLE_SIZE = (0.6, 0.8)   # x, y extent [m]
TABLE_TOP = 0.72
BOX_SIZE = (0.05, 0.05, 0.08)
BOX_NAMES = ("blue", "green", "yellow")
BOX_DEFAULTS = {"blue": (0.62, -0.34), "green": (0.62, -0.22), "yellow": (0.62, -0.10)}   # a row, 12 cm apart
BOX_DEFAULT = BOX_DEFAULTS["blue"]

# Reachability. The grasp posture (torso squat) puts the right shoulder at SHOULDER; the tool points forward, so the wrist
# link sits TIP_TO_WRIST behind the fingertips. Measured: the box at (0.64, -0.20) on the 0.72 m table (wrist 0.494 m from
# the shoulder) is grasped and lifted; a straight arm is a Cartesian singularity, so the limit is a little above that.
# MIN_REACH (elbow folded against the torso) is a guess; nothing closer has been tried.
SHOULDER = (0.0, -0.22, 1.05)
TIP_TO_WRIST = 0.12 + 0.1261
MAX_REACH = 0.50
MIN_REACH = 0.33
TIP_BEYOND_CENTRE = 0.005  # the fingertips end this far past the box centre


def wrist_distance(tip):
    wx, wy, wz = tip[0] - TIP_TO_WRIST, tip[1], tip[2]
    return math.dist((wx, wy, wz), SHOULDER)


def tip_reachable(tip) -> bool:
    return MIN_REACH <= wrist_distance(tip) <= MAX_REACH


def box_tip(x, y, top, lift=0.0):
    """Fingertip position that holds a box standing at (x, y) on a table of height `top`."""
    return (x + TIP_BEYOND_CENTRE, y, top + BOX_SIZE[2] / 2 + lift)


def box_reachable(x, y, top) -> bool:
    return tip_reachable(box_tip(x, y, top))


def on_table(x, y, table_center=TABLE_CENTER, table_size=TABLE_SIZE, margin=0.04) -> bool:
    return (abs(x - table_center[0]) <= table_size[0] / 2 - margin) and (abs(y - table_center[1]) <= table_size[1] / 2 - margin)


def check_pose(x, y, top, table_center=TABLE_CENTER, table_size=TABLE_SIZE):
    """None when a box can stand at (x, y) and be reached there, else the reason."""
    if not on_table(x, y, table_center, table_size):
        return "not on the table"
    d = wrist_distance(box_tip(x, y, top))
    if d > MAX_REACH:
        return f"out of reach ({d:.2f} m from the shoulder, limit {MAX_REACH:.2f})"
    if d < MIN_REACH:
        return f"too close to the body ({d:.2f} m, limit {MIN_REACH:.2f})"
    return None

# The seed motion (torso squat, then the arm swings up from the hanging pose) passes through x = 0.5..0.53 below the table top:
# it only clears the table slab while the slab's front edge is at x >= SEED_EDGE_X (measured: an edge at 0.50 blocked the elbow).
# Once the arm is above the table (seeded), the table may move closer.
SEED_EDGE_X = 0.57


def table_front_edge(table_center=TABLE_CENTER, table_size=TABLE_SIZE):
    return table_center[0] - table_size[0] / 2


def table_ok_before_seed(table_center=TABLE_CENTER, table_size=TABLE_SIZE) -> bool:
    return table_front_edge(table_center, table_size) >= SEED_EDGE_X


# ---- several boxes: clearance, stacks, free spots ------------------------------------------------------------------------------
# The fingers open sideways (y) to +-0.063 m around the box centre: a box at the same height closer than this in y (or in x, the
# fingers are 3 cm wide in x ... and the hand comes from the front) would be hit when the hand goes down around the box.
FINGER_CLEARANCE = 0.11   # [m] minimum centre distance to the next box at the same level
STACK_XY = 0.035          # [m] a box whose centre is this close (in x and y) to another one, one box height above it, sits on it
SPOT_MARGIN = 0.01        # [m] free spots stay this far inside the reach limit
HOVER_STACK = 0.06        # [m] same as grasp_core.HOVER: the held box's bottom this far above the surface it is carried over


def stacked_on(boxes):
    """{name: name of the box it rests on, or None} for {name: (x, y, z)} box centres."""
    out = {}
    for n, (x, y, z) in boxes.items():
        out[n] = None
        for m, (bx, by, bz) in boxes.items():
            if m != n and abs(x - bx) < STACK_XY and abs(y - by) < STACK_XY and abs((z - bz) - BOX_SIZE[2]) < 0.025:
                out[n] = m
    return out


def pick_problem(name, boxes):
    """None when box `name` can be picked where it is ({name: (x, y, z)}), else the reason: out of reach, or a neighbour at the
    same level that the open fingers would hit."""
    x, y, z = boxes[name]
    tip = (x + TIP_BEYOND_CENTRE, y, z)
    d = wrist_distance(tip)
    if d > MAX_REACH:
        return f"the {name} box at ({x:.2f}, {y:.2f}) is out of reach ({d:.2f} m from the shoulder, limit {MAX_REACH:.2f})"
    if d < MIN_REACH:
        return f"the {name} box at ({x:.2f}, {y:.2f}) is too close to the body ({d:.2f} m, limit {MIN_REACH:.2f})"
    for m, (bx, by, bz) in boxes.items():
        if m != name and abs(bz - z) < 0.5 * BOX_SIZE[2] and math.hypot(bx - x, by - y) < FINGER_CLEARANCE:
            return (f"the {name} box is too close to the {m} box ({math.hypot(bx - x, by - y) * 100:.0f} cm apart, the open fingers need "
                    f"{FINGER_CLEARANCE * 100:.0f} cm): move one of them first")
    return None


def stack_problems(order, boxes, top=TABLE_TOP):
    """Reasons why `order` (box names, bottom first; the first one stays where it is) cannot be stacked from the current
    scene {name: (x, y, z)}. Empty list: go."""
    missing = [n for n in order if n not in boxes]
    if missing:
        return [f"cannot see the {', '.join(missing)} box" + ("es" if len(missing) > 1 else "")]
    if len(set(order)) != len(order) or len(order) < 2:
        return ["a stack needs two or more different boxes"]
    bx, by, _ = boxes[order[0]]
    problems = []
    for k, n in enumerate(order[1:], 1):
        p = pick_problem(n, {m: boxes[m] for m in boxes if m not in order[:k]})   # boxes already on the stack are out of the way
        if p:
            problems.append(p)
        hover = (bx + TIP_BEYOND_CENTRE, by, top + k * BOX_SIZE[2] + BOX_SIZE[2] / 2 + HOVER_STACK)
        if not tip_reachable(hover):
            problems.append(f"level {k + 1} of the stack at ({bx:.2f}, {by:.2f}) is out of reach "
                            f"({wrist_distance(hover):.2f} m from the shoulder)")
    return problems


def direction_spot(direction, boxes, exclude=None, top=TABLE_TOP, table_center=TABLE_CENTER, table_size=TABLE_SIZE):
    """Free table spot that is furthest 'right' (-y, the right arm's side), 'left' (+y), 'near' (-x) or 'far' (+x), or the one
    closest to the shoulder line ('center'): reachable with a margin, on the table, FINGER_CLEARANCE from every box but `exclude`."""
    others = [(x, y) for n, (x, y, _) in boxes.items() if n != exclude]
    best = None
    for i in range(int(table_size[0] / 0.01) + 1):
        for j in range(int(table_size[1] / 0.01) + 1):
            x, y = table_center[0] - table_size[0] / 2 + i * 0.01, table_center[1] - table_size[1] / 2 + j * 0.01
            if check_pose(x, y, top, table_center, table_size) is not None or wrist_distance(box_tip(x, y, top)) > MAX_REACH - SPOT_MARGIN:
                continue
            if any(math.hypot(x - ox, y - oy) < FINGER_CLEARANCE for ox, oy in others):
                continue
            key = {"right": y, "left": -y, "near": x, "far": -x, "center": abs(y - SHOULDER[1]) + 0.2 * abs(x - 0.60)}[direction]
            if best is None or key < best[0]:
                best = (key, (round(x, 3), round(y, 3)))
    return None if best is None else best[1]
