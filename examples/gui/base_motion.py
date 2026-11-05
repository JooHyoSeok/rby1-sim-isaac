"""Driving the RB-Y1's base (two wheels, differential drive): odometry from the wheel joints, the speed profile of a closed-loop move, and the
check that a move does not run the base or the arm into the table. Pure python + numpy, no SDK: the executor (grasp_core.py) does the commanding.

Frames: the base at its start pose is the world origin, x forward, y left, yaw counter-clockwise (a right turn is negative). The rest of the code
(servo, reach model, safety.py) works in the base frame: grasp_core converts what the stereo rig sees (world) with the base's true pose, so
manipulation works wherever the base stands. The simulated base turns poorly (see PIVOTS): precise driving home is best effort only.
"""
import math

import numpy as np

WHEEL_R, TRACK = 0.1, 0.53        # [m] wheel radius (URDF collision cylinder), distance between the wheels (joints at y = +-0.265)
FORWARD_SIGN = -1.0               # the wheel joints turn about -y: a positive joint angle drives backward. To be confirmed in the simulator.
V_MAX, A_LIN, V_MIN = 0.15, 0.15, 0.05      # [m/s], [m/s^2]: the speed profile of a straight move
W_MAX, A_ANG, W_MIN = 0.40, 0.50, 0.20      # [rad/s], [rad/s^2]: of a turn (below ~0.15 rad/s the simulated base stalls)
STALL_TIME, STALL_BOOST = 1.0, 1.5          # no progress for this long: command this much faster (up to 3x the profile)
# Turning in place does not pivot about the middle of the wheels in the simulator: the base origin moved 0.2 m in a 30 degree turn and 0.58 m
# in an 81 degree one (the base swings about a point near one wheel). The turn sweeps are therefore checked about all of these pivots, given in the
# base frame: the base origin, the middle of the wheel axle, and the two points the simulator was seen to pivot about (a 30 degree right turn: (0.05, 0.18)).
PIVOTS = ((0.0, 0.0), (0.228, 0.0), (0.05, 0.18), (0.05, -0.18))
TOL_LIN, TOL_ANG = 0.005, math.radians(0.6)  # a move is done within this of its target
MAX_FORWARD, MAX_TURN = 0.30, math.radians(90.0)   # one drive command
MAX_RANGE, MAX_YAW = 0.45, math.radians(180.0)     # how far from the start pose the base may get in total
HOME_TOL_LIN, HOME_TOL_ANG = 0.03, math.radians(3.0)   # "at the start pose": manipulation is allowed
# Where things are, in the world, for the collision check. The table is assumed at its default place (front edge 0.58) with a 5 cm margin: if the
# control centre pulled it closer the executor is not told and this check does not know. The footprint is the base and wheels (from the
# simulator's ground truth file when it has it, see src/rby1_vision.py write_gt), otherwise the outline measured there. [xmin, ymin, xmax, ymax]
TABLE_RECT = (0.53, -0.45, 1.23, 0.45)
TABLE_TOP, ARM_CLEAR = 0.72, 0.10
FOOTPRINT = (-0.335, -0.29, 0.328, 0.29)    # as measured in the simulator (src/rby1_vision.py write_gt: '[Vision] base outline')


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


def turn_about(pose, delta, pivot):
    """Pose after turning the base by `delta` rad about `pivot` (given in the base frame at `pose`)."""
    x, y, yaw = (float(v) for v in pose)
    c, s = math.cos(yaw), math.sin(yaw)
    px, py = x + c * pivot[0] - s * pivot[1], y + s * pivot[0] + c * pivot[1]       # the pivot in the world
    cd, sd = math.cos(delta), math.sin(delta)
    return (px + cd * (x - px) - sd * (y - py), py + sd * (x - px) + cd * (y - py), wrap(yaw + delta))


def check_move(pose, forward, turn, arm_points=(), footprint=FOOTPRINT, table=TABLE_RECT, check_range=True):
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
    if check_range and math.hypot(x1, y1) > MAX_RANGE + 1e-9:
        out.append(f"that would take the base {math.hypot(x1, y1) * 100:.0f} cm from its start pose (at most {MAX_RANGE * 100:.0f} cm)")
    if check_range and abs(wrap(yaw1)) > MAX_YAW + 1e-9:
        out.append("that would turn the base more than 180 degrees from its start heading")
    if out:
        return out
    n_turn = max(1, int(abs(turn) / math.radians(1.0))) if turn else 0
    n_fwd = max(1, int(abs(forward) / 0.02)) if forward else 0
    samples = [(*turn_about(pose, turn * i / n_turn, pv), f"{math.degrees(turn * i / n_turn):.0f} degrees into the turn") for i in range(1, n_turn + 1) for pv in PIVOTS]
    end = turn_about(pose, turn, PIVOTS[1]) if turn else (x, y, yaw)      # after the turn: where the base stands (the move starts there)
    samples += [(end[0] + forward * i / n_fwd * math.cos(end[2]), end[1] + forward * i / n_fwd * math.sin(end[2]), end[2],
                 f"{abs(forward) * i / n_fwd * 100:.0f} cm into the move") for i in range(1, n_fwd + 1)]
    start = penetration(pose, arm_points, footprint, table)      # already too close to the table (the margin is generous): moving AWAY is fine, deeper is not
    for sx, sy, syaw, where in samples:
        if penetration((sx, sy, syaw), arm_points, footprint, table) > start + 1e-6:
            arm_in = any(_in_rect(wx, wy, table) and wz < TABLE_TOP + ARM_CLEAR for wx, wy, wz in _world(list(arm_points), (sx, sy, syaw)))
            base_in = any(_in_rect(wx, wy, table) for wx, wy, _ in _world(_footprint_points(footprint), (sx, sy, syaw)))
            if base_in:
                return [f"the base would run into the table ({where}; the table is assumed to start at x = {table[0]:.2f} m)" + ("; it is already close: only moves away from it are allowed" if start else "")]
            if arm_in:
                return [f"the arm would hit the table ({where}); move the arm up and back first"]
    return []


def penetration(pose, arm_points=(), footprint=FOOTPRINT, table=TABLE_RECT):
    """How far (m, along x) the base outline or an arm point below the arm clearance reaches into the table at `pose`; 0 when clear."""
    depth = 0.0
    for wx, wy, _ in _world(_footprint_points(footprint), pose):
        if _in_rect(wx, wy, table):
            depth = max(depth, wx - table[0] + 1e-9)
    for wx, wy, wz in _world(list(arm_points), pose):
        if _in_rect(wx, wy, table) and wz < TABLE_TOP + ARM_CLEAR:
            depth = max(depth, wx - table[0] + 1e-9)
    return depth


def hits_table(pose, arm_points=(), footprint=FOOTPRINT, table=TABLE_RECT):
    """True when at `pose` the base outline is inside the table or an arm point is inside it below the arm clearance."""
    for wx, wy, _ in _world(_footprint_points(footprint), pose):
        if _in_rect(wx, wy, table):
            return True
    return any(_in_rect(wx, wy, table) and wz < TABLE_TOP + ARM_CLEAR for wx, wy, wz in _world(list(arm_points), pose))


def way_home(pose):
    """(turn rad, forward m, final turn rad) that bring the base from `pose` back to the start pose: face the origin, drive, face forward. A turn
    moves the origin (see PIVOTS): the caller does the first step that is needed, then plans again from where the base really is."""
    x, y, yaw = (float(v) for v in pose)
    dist = math.hypot(x, y)
    if dist <= HOME_TOL_LIN / 2:
        return wrap(-yaw), 0.0, 0.0
    heading = math.atan2(-y, -x)
    return wrap(heading - yaw), dist, wrap(-heading)
