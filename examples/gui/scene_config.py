"""Shared by the control centre (host) and the executor service (simulator container): ports, scene defaults, reachability.

Plain python (no numpy, no Qt) so both sides can import it. The scene defaults repeat src/rby1_vision.py.
"""
import math

# UDP ports, all on 127.0.0.1 (the containers share the host network)
PORT_SCENE = 5110        # control centre -> simulator: move the box / table (src/rby1_vision.py poll_commands)
PORT_EXEC_CMD = 5120     # control centre -> executor service: pick, place, ...
PORT_EXEC_STATUS = 5121  # executor service -> control centre: status and log lines
PORT_BASE_POSE = 5130    # simulator -> executor service: the base's true pose {"base": [x, y, yaw], "t": ..} at 20 Hz (the wheels slip when it turns)

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
# The left arm mirrors it: its shoulder is at y = +0.22 (verified: forward kinematics of the URDF, FK of the left wrist matches the simulator).
SHOULDERS = {"right": (0.0, -0.22, 1.05), "left": (0.0, 0.22, 1.05)}
ARMS = ("right", "left")
SHOULDER = SHOULDERS["right"]
TIP_TO_WRIST = 0.12 + 0.1261
MAX_REACH = 0.50
MIN_REACH = 0.33
TIP_BEYOND_CENTRE = 0.005  # the fingertips end this far past the box centre


def _arms(arm):
    """The arms to consider: ("right",), ("left",), or both for arm=None (reachable by either)."""
    return ARMS if arm is None else (arm,)


def wrist_distance(tip, arm="right"):
    wx, wy, wz = tip[0] - TIP_TO_WRIST, tip[1], tip[2]
    return math.dist((wx, wy, wz), SHOULDERS[arm])


def tip_reachable(tip, arm="right") -> bool:
    """Can `arm` hold its fingertips at `tip` (base frame)? arm=None: can either arm?"""
    return any(MIN_REACH <= wrist_distance(tip, a) <= MAX_REACH for a in _arms(arm))


def comfort(tip, arm="right"):
    """How comfortably `arm` holds its fingertips at `tip`: the distance of the wrist from the shoulder to the nearer limit of the reach (m). Negative =
    out of reach; near zero = arm almost straight (singular, the whole body leans and stretches) or folded against the chest; largest in the middle."""
    d = wrist_distance(tip, arm)
    return min(MAX_REACH - d, d - MIN_REACH)


def box_tip(x, y, top, lift=0.0):
    """Fingertip position that holds a box standing at (x, y) on a table of height `top`."""
    return (x + TIP_BEYOND_CENTRE, y, top + BOX_SIZE[2] / 2 + lift)


def box_reachable(x, y, top, arm="right") -> bool:
    return tip_reachable(box_tip(x, y, top), arm)


def base_to_world(base, x, y):
    """World position of the point (x, y) given in the base frame; `base` = (x, y, yaw [rad]) of the base in the world (the simulator's true pose)."""
    c, s = math.cos(base[2]), math.sin(base[2])
    return base[0] + c * x - s * y, base[1] + s * x + c * y


def world_to_base(base, x, y):
    c, s = math.cos(base[2]), math.sin(base[2])
    dx, dy = x - base[0], y - base[1]
    return c * dx + s * dy, -s * dx + c * dy


def on_table(x, y, table_center=TABLE_CENTER, table_size=TABLE_SIZE, margin=0.04, base=None) -> bool:
    """Is the point (x, y) on the table? In the world, or in the base frame when `base` (the base's world pose) is given: the table stays where it is."""
    if base is not None:
        x, y = base_to_world(base, x, y)
    return (abs(x - table_center[0]) <= table_size[0] / 2 - margin) and (abs(y - table_center[1]) <= table_size[1] / 2 - margin)


def check_pose(x, y, top, table_center=TABLE_CENTER, table_size=TABLE_SIZE, base=None, arm="right"):
    """None when a box can stand at (x, y) and be reached there by `arm` (None: by either arm), else the reason. With `base` (the base's world pose)
    the point is in the base frame."""
    if not on_table(x, y, table_center, table_size, base=base):
        return "not on the table"
    ds = {a: wrist_distance(box_tip(x, y, top), a) for a in _arms(arm)}
    if any(MIN_REACH <= d <= MAX_REACH for d in ds.values()):
        return None
    a, d = min(ds.items(), key=lambda kv: min(abs(kv[1] - MAX_REACH), abs(kv[1] - MIN_REACH)))
    who = "" if arm is not None and arm == "right" else f"{a} arm: "
    if d > MAX_REACH:
        return f"{who}out of reach ({d:.2f} m from the shoulder, limit {MAX_REACH:.2f})"
    return f"{who}too close to the body ({d:.2f} m, limit {MIN_REACH:.2f})"

# The seed motion (torso squat, then the arm swings up from the hanging pose) passes through x = 0.5..0.53 below the table top:
# it only clears the table slab while the slab's front edge is at x >= SEED_EDGE_X (measured: an edge at 0.50 blocked the elbow).
# Once the arm is above the table (seeded), the table may move closer.
SEED_EDGE_X = 0.57


def table_front_edge(table_center=TABLE_CENTER, table_size=TABLE_SIZE):
    return table_center[0] - table_size[0] / 2


def table_front_distance(base, table_center=TABLE_CENTER, table_size=TABLE_SIZE, y_range=(-0.45, 0.0)):
    """How far ahead of the base (along its x axis) the nearest table point is among those with base-frame y in `y_range` (the right arm's side, where
    the seed motion swings the arm and the elbow); infinity when none is. The seed motion needs SEED_EDGE_X of free room."""
    best = math.inf
    nx, ny = int(table_size[0] / 0.02) + 1, int(table_size[1] / 0.02) + 1
    for i in range(nx):
        for j in range(ny):
            wx, wy = table_center[0] - table_size[0] / 2 + i * 0.02, table_center[1] - table_size[1] / 2 + j * 0.02
            bx, by = world_to_base(base, wx, wy)
            if bx > 0 and y_range[0] <= by <= y_range[1]:
                best = min(best, bx)
    return best


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


def pick_problem(name, boxes, arm="right"):
    """None when box `name` can be picked where it is ({name: (x, y, z)}) with `arm`, else the reason: out of reach, or a neighbour at the
    same level that the open fingers would hit."""
    x, y, z = boxes[name]
    tip = (x + TIP_BEYOND_CENTRE, y, z)
    d = wrist_distance(tip, arm)
    who = "" if arm == "right" else f" for the {arm} arm"
    if d > MAX_REACH:
        return f"the {name} box at ({x:.2f}, {y:.2f}) is out of reach{who} ({d:.2f} m from the shoulder, limit {MAX_REACH:.2f})"
    if d < MIN_REACH:
        return f"the {name} box at ({x:.2f}, {y:.2f}) is too close to the body{who} ({d:.2f} m, limit {MIN_REACH:.2f})"
    for m, (bx, by, bz) in boxes.items():
        if m != name and abs(bz - z) < 0.5 * BOX_SIZE[2] and math.hypot(bx - x, by - y) < FINGER_CLEARANCE:
            return (f"the {name} box is too close to the {m} box ({math.hypot(bx - x, by - y) * 100:.0f} cm apart, the open fingers need "
                    f"{FINGER_CLEARANCE * 100:.0f} cm): move one of them first")
    return None


def best_arm(name, boxes, spot=None, top=TABLE_TOP, prefer=None, slack=0.03):
    """Which arm should pick box `name` (and put it at `spot` = (x, y) on a surface of height `top`, if given): the one that can do it and holds
    its hand most comfortably (see `comfort`); `prefer` (the arm that is already in position) keeps the job unless the other arm is more
    comfortable by more than `slack` m. None when neither arm can."""
    x, y, z = boxes[name]
    options = {}
    for arm in ARMS:
        if pick_problem(name, boxes, arm) is not None:
            continue
        c = comfort((x + TIP_BEYOND_CENTRE, y, z), arm)
        if spot is not None:
            tip = box_tip(spot[0], spot[1], top)
            if not tip_reachable(tip, arm):
                continue
            c = min(c, comfort(tip, arm))
        options[arm] = c
    if not options:
        return None
    best = max(options, key=options.get)
    if prefer in options and options[best] - options[prefer] <= slack:
        best = prefer
    return best


def arm_options(name, boxes):
    """{arm: {"problem": reason or None, "comfort": m}} for picking box `name` where it is: what the scene reply shows the planner."""
    x, y, z = boxes[name]
    return {arm: {"problem": pick_problem(name, boxes, arm), "comfort": round(comfort((x + TIP_BEYOND_CENTRE, y, z), arm), 3)} for arm in ARMS}


def stack_problems(order, boxes, top=TABLE_TOP):
    """Reasons why `order` (box names, bottom first; the first one stays where it is) cannot be stacked from the current
    scene {name: (x, y, z)}: every other box is picked by whichever arm can do its pick and its level of the stack. Empty list: go."""
    missing = [n for n in order if n not in boxes]
    if missing:
        return [f"cannot see the {', '.join(missing)} box" + ("es" if len(missing) > 1 else "")]
    if len(set(order)) != len(order) or len(order) < 2:
        return ["a stack needs two or more different boxes"]
    bx, by, _ = boxes[order[0]]
    problems = []
    for k, n in enumerate(order[1:], 1):
        others = {m: boxes[m] for m in boxes if m not in order[:k]}   # boxes already on the stack are out of the way
        hover = (bx + TIP_BEYOND_CENTRE, by, top + k * BOX_SIZE[2] + BOX_SIZE[2] / 2 + HOVER_STACK)
        level = (bx, by)
        arm = best_arm(n, others, spot=level, top=top + k * BOX_SIZE[2])
        if arm is not None and tip_reachable(hover, arm):
            continue
        can_pick = [a for a in ARMS if pick_problem(n, others, a) is None]
        if can_pick:      # some arm can pick it, but none of those reaches this level of the stack
            problems.append(f"level {k + 1} of the stack at ({bx:.2f}, {by:.2f}) is out of reach "
                            f"({min(wrist_distance(hover, a) for a in can_pick):.2f} m from the shoulder)")
        else:             # no arm can pick it: the reasons
            problems.append(next(p for a in ARMS if (p := pick_problem(n, others, a))))
    return problems


def direction_spot(direction, boxes, exclude=None, top=TABLE_TOP, table_center=TABLE_CENTER, table_size=TABLE_SIZE, base=None, arm="right"):
    """Free table spot that is furthest 'right' (-y, the right arm's side), 'left' (+y), 'near' (-x) or 'far' (+x), or the one
    closest to the shoulder line ('center'): reachable with a margin, on the table, FINGER_CLEARANCE from every box but `exclude`."""
    others = [(x, y) for n, (x, y, _) in boxes.items() if n != exclude]
    best = None
    # the grid: the table's extent when the base stands at the world origin, else the base-frame region the arm can reach (the table is rotated in it)
    x0, y0, nx, ny = (table_center[0] - table_size[0] / 2, table_center[1] - table_size[1] / 2, int(table_size[0] / 0.01) + 1, int(table_size[1] / 0.01) + 1) \
        if base is None else (0.15, -0.60, 71, 121)
    for i in range(nx):
        for j in range(ny):
            x, y = x0 + i * 0.01, y0 + j * 0.01
            if check_pose(x, y, top, table_center, table_size, base, arm) is not None or \
                    min(wrist_distance(box_tip(x, y, top), a) for a in _arms(arm)) > MAX_REACH - SPOT_MARGIN:
                continue
            if any(math.hypot(x - ox, y - oy) < FINGER_CLEARANCE for ox, oy in others):
                continue
            centre_y = SHOULDERS[arm][1] if arm is not None else 0.0
            key = {"right": y, "left": -y, "near": x, "far": -x, "center": abs(y - centre_y) + 0.2 * abs(x - 0.60)}[direction]
            if best is None or key < best[0]:
                best = (key, (round(x, 3), round(y, 3)))
    return None if best is None else best[1]


def unstack_order(boxes):
    """Boxes that rest on another one, highest first: the order in which a stack is taken apart ({name: (x, y, z)})."""
    on = stacked_on(boxes)
    return sorted((n for n in boxes if on[n]), key=lambda n: -boxes[n][2])


def move_spot(direction, name, boxes, top=TABLE_TOP, table_center=TABLE_CENTER, table_size=TABLE_SIZE, base=None, arm=None):
    """The free spot furthest in `direction` where box `name` can be put down by an arm that can also pick it where it stands (`arm`: only that
    one). Boxes a hand can pick but not carry across the middle of the far table stay on their side. None when there is none."""
    arms = [arm] if arm else [a for a in ARMS if pick_problem(name, boxes, a) is None]
    best = None
    for a in arms:
        spot = direction_spot(direction, boxes, exclude=name, top=top, table_center=table_center, table_size=table_size, base=base, arm=a)
        if spot is None:
            continue
        key = {"right": spot[1], "left": -spot[1], "near": spot[0], "far": -spot[0], "center": 0.0}[direction]
        if best is None or key < best[0]:
            best = (key, spot)
    return None if best is None else best[1]


def table_spot(name, boxes, top=TABLE_TOP, table_center=TABLE_CENTER, table_size=TABLE_SIZE, base=None, arm="right"):
    """A free table spot for the `name` box: its own start spot if nothing is closer than FINGER_CLEARANCE, else another start spot (boxes were
    picked there, so the arm reaches them), else the best free spot the reach model allows; None when there is none."""
    others = [(x, y) for n, (x, y, _) in boxes.items() if n != name]
    starts = [BOX_DEFAULTS[name]] + [xy for n, xy in BOX_DEFAULTS.items() if n != name]     # world coordinates ...
    if base is not None:
        starts = [world_to_base(base, *xy) for xy in starts]                                  # ... as the boxes and the answer are in the base frame
    for x, y in starts:
        if check_pose(x, y, top, table_center, table_size, base, arm) is None and all(math.hypot(x - ox, y - oy) >= FINGER_CLEARANCE for ox, oy in others):
            return (x, y)
    for direction in ("center", "right", "left", "near", "far"):
        spot = direction_spot(direction, boxes, exclude=name, top=top, table_center=table_center, table_size=table_size, base=base, arm=arm)
        if spot is not None:
            return spot
    return None


# ---- two hands at the mating plane (examples/assembly/meet_study.py: reachable at x 0.42..0.48, z 0.9..1.1 with the torso squatted) --------------------
# The insertion axis is the base y axis, the mating plane is y = 0. The left hand holds the female part (it extends to +y), the right hand the male part
# (to -y); each grasp point is MEET_GRASP_BACK behind its part's mating face, so the grasp points are 2 * MEET_GRASP_BACK + gap apart. The fingers close
# across the part (vertical closing axis), the tool axis points forward ("forward") or 30 degrees inward ("inward": the elbows spread, 33 cm apart
# instead of 9 cm in the study). The wrist link frame: x = the closing axis, z = against the tool axis; the ee frame has the same orientation, its origin
# EE_BEHIND_GRASP behind the grasp point along the tool axis (fingertips 5 mm past the grasp point, ee 0.12 behind the fingertips).
MEET_GRASP_BACK = 0.03
EE_BEHIND_GRASP = 0.115
MEET_VARIANTS = {"forward": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
                 "inward": ((math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0), (0.0, 0.0, 1.0))}    # tool axis and closing axis of the RIGHT hand


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def meet_targets(x, z, gap=MEET_GRASP_BACK, variant="forward"):
    """{"right"/"left": {"R": 3x3 rows (the ee orientation), "ee": [x, y, z] (the ee origin), "grasp": [x, y, z]}} for the two hands holding their parts at
    (x, 0, z) in the base frame, the parts `gap` apart (0 = seated)."""
    tool, closing = MEET_VARIANTS[variant]
    out = {}
    for side, sign in (("right", 1.0), ("left", -1.0)):
        t, c = (tool[0], sign * tool[1], tool[2]), (closing[0], sign * closing[1], closing[2])
        zc = (-t[0], -t[1], -t[2])
        yc = _cross(zc, c)
        R = [[c[i], yc[i], zc[i]] for i in range(3)]             # columns x, y, z
        y = -(gap + MEET_GRASP_BACK) if side == "right" else MEET_GRASP_BACK
        g = (x, y, z)
        out[side] = {"R": R, "grasp": list(g), "ee": [g[i] + EE_BEHIND_GRASP * zc[i] for i in range(3)]}
    return out
