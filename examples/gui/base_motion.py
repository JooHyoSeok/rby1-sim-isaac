"""Driving the RB-Y1's base (two wheels, differential drive): odometry from the wheel joints, the speed profile of a closed-loop move, and the
check that a move does not run the base or the arm into the table. Pure python + numpy, no SDK: the executor (grasp_core.py) does the commanding.

Frames: the base at its start pose is the world origin, x forward, y left, yaw counter-clockwise (a right turn is negative). Everything the
rest of the code does (perception, servo, reach model, safety.py) assumes the base is still there: grasp_core refuses manipulation while it is not.
"""
import math

import numpy as np

WHEEL_R, TRACK = 0.1, 0.53        # [m] wheel radius (URDF collision cylinder), distance between the wheels (joints at y = +-0.265)
FORWARD_SIGN = -1.0               # the wheel joints turn about -y: a positive joint angle drives backward. To be confirmed in the simulator.
V_MAX, A_LIN, V_MIN = 0.15, 0.15, 0.04      # [m/s], [m/s^2]: the speed profile of a straight move
W_MAX, A_ANG, W_MIN = 0.40, 0.50, 0.12      # [rad/s], [rad/s^2]: of a turn
TOL_LIN, TOL_ANG = 0.005, math.radians(0.6)  # a move is done within this of its target
MAX_FORWARD, MAX_TURN = 0.30, math.radians(90.0)   # one drive command
MAX_RANGE, MAX_YAW = 0.45, math.radians(180.0)     # how far from the start pose the base may get in total
HOME_TOL_LIN, HOME_TOL_ANG = 0.03, math.radians(3.0)   # "at the start pose": manipulation is allowed
# Where things are, in the world, for the collision check. The table is assumed at its default place (front edge 0.58) with a 5 cm margin: if the
# control centre pulled it closer the executor is not told and this check does not know. The footprint is the base and wheels (from the
# simulator's ground truth file when it has it, see src/rby1_vision.py write_gt), otherwise this guess. [xmin, ymin, xmax, ymax]
TABLE_RECT = (0.53, -0.45, 1.23, 0.45)
TABLE_TOP, ARM_CLEAR = 0.72, 0.10
FOOTPRINT = (-0.30, -0.35, 0.35, 0.35)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Odometry:
    """Pose (x, y, yaw) of the base from the wheel joint angles [rad]; call update() often (the arc is approximated by its mean heading)."""

    def __init__(self, q_right, q_left, pose=(0.0, 0.0, 0.0)):
        self.q = (float(q_right), float(q_left))
        self.pose = np.array(pose, float)

    def update(self, q_right, q_left):
        dr, dl = (q_right - self.q[0]) * WHEEL_R * FORWARD_SIGN, (q_left - self.q[1]) * WHEEL_R * FORWARD_SIGN   # distance each wheel rolled forward
        self.q = (float(q_right), float(q_left))
        d, dyaw = (dr + dl) / 2.0, (dr - dl) / TRACK
        x, y, yaw = self.pose
        self.pose = np.array([x + d * math.cos(yaw + dyaw / 2), y + d * math.sin(yaw + dyaw / 2), wrap(yaw + dyaw)])
        return self.pose

    def at_home(self):
        x, y, yaw = self.pose
        return math.hypot(x, y) <= HOME_TOL_LIN and abs(wrap(yaw)) <= HOME_TOL_ANG


def speed(remaining, v_max, accel, v_min):
    """Speed for the distance (or angle) still to go: as fast as `v_max`, slowing so it can stop with `accel`, never below `v_min` (friction)."""
    return float(np.clip(math.sqrt(2.0 * accel * max(abs(remaining), 0.0)), v_min, v_max))


def _world(points, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * px - s * py, y + s * px + c * py, pz) for px, py, pz in points]


def _footprint_points(rect):
    x0, y0, x1, y1 = rect
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    return [(x, y, 0.0) for x, y in ((x0, y0), (x0, ym), (x0, y1), (xm, y1), (x1, y1), (x1, ym), (x1, y0), (xm, y0))]


def _in_rect(x, y, rect):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def check_move(pose, forward, turn, arm_points=(), footprint=FOOTPRINT, table=TABLE_RECT):
    """Reasons why a drive command (first turn by `turn` rad, then `forward` m along the new heading) from `pose` is refused ([] = fine): size of the
    command, distance from the start pose, and along the way (sampled every 1 degree / 2 cm) the base outline or any `arm_points` (xyz in the
    base frame, from forward kinematics) inside the table or below the arm clearance over it."""
    out = []
    if abs(forward) > MAX_FORWARD + 1e-9:
        out.append(f"at most {MAX_FORWARD * 100:.0f} cm per drive command (asked {forward * 100:.0f} cm)")
    if abs(turn) > MAX_TURN + 1e-9:
        out.append(f"at most {math.degrees(MAX_TURN):.0f} degrees per drive command (asked {math.degrees(turn):.0f})")
    x, y, yaw = (float(v) for v in pose)
    yaw1 = wrap(yaw + turn)
    x1, y1 = x + forward * math.cos(yaw1), y + forward * math.sin(yaw1)
    if math.hypot(x1, y1) > MAX_RANGE + 1e-9:
        out.append(f"that would take the base {math.hypot(x1, y1) * 100:.0f} cm from its start pose (at most {MAX_RANGE * 100:.0f} cm)")
    if abs(wrap(yaw1)) > MAX_YAW + 1e-9:
        out.append("that would turn the base more than 180 degrees from its start heading")
    if out:
        return out
    n_turn = max(1, int(abs(turn) / math.radians(1.0))) if turn else 0
    n_fwd = max(1, int(abs(forward) / 0.02)) if forward else 0
    samples = [(x, y, wrap(yaw + turn * i / n_turn), f"{math.degrees(turn * i / n_turn):.0f} degrees into the turn") for i in range(1, n_turn + 1)]
    samples += [(x + forward * i / n_fwd * math.cos(yaw1), y + forward * i / n_fwd * math.sin(yaw1), yaw1,
                 f"{abs(forward) * i / n_fwd * 100:.0f} cm into the move") for i in range(1, n_fwd + 1)]
    for sx, sy, syaw, where in samples:
        for wx, wy, _ in _world(_footprint_points(footprint), (sx, sy, syaw)):
            if _in_rect(wx, wy, table):
                return [f"the base would run into the table ({where}; the table is assumed to start at x = {table[0]:.2f} m)"]
        for wx, wy, wz in _world(list(arm_points), (sx, sy, syaw)):
            if _in_rect(wx, wy, table) and wz < TABLE_TOP + ARM_CLEAR:
                return [f"the arm would hit the table ({where}): the hand or elbow would be at ({wx:.2f}, {wy:.2f}, {wz:.2f}); move the arm up and back first"]
    return []


def way_home(pose):
    """(turn rad, forward m, final turn rad) that bring the base from `pose` back to the start pose: face the origin, drive, face forward."""
    x, y, yaw = (float(v) for v in pose)
    dist = math.hypot(x, y)
    if dist <= HOME_TOL_LIN / 2:
        return wrap(-yaw), 0.0, 0.0
    heading = math.atan2(-y, -x)
    return wrap(heading - yaw), dist, wrap(-heading)
