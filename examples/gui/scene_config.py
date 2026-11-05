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
BOX_DEFAULT = (0.64, -0.20)

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
