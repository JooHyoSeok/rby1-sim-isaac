"""Mixed-part assembly (term project, week 9 slides): the five interface families, the relative pose of two held parts, and what it takes to
bring the male part to the mating pose of the female one. numpy only, no simulator.

Frames. Every part has its own frame {P}: the origin at the centre of its mating face, +z pointing out of that face towards the partner (the peg tip
direction for a male part, out of the opening for a female one), +x towards the key / flat face / small pin (the roll reference). A male part {M}
is seated in a female part {F} when their origins coincide and +z_M = -z_F, +x_M = +x_F: F_T_M = MATED. Approaching, F_T_M has t = (0, 0, gap).
Poses are 4x4 homogeneous matrices; F_T_M = inv(C_T_F) @ C_T_M for the two parts seen by one camera (the camera-to-base transform cancels,
which is why an unknown camera or joint-zero error does not matter for this measurement: slide 10).
  /usr/bin/python3 assembly.py     prints the odds of a peg fitting for a few clearances and alignment accuracies
"""
import itertools
import math
from dataclasses import dataclass

import numpy as np

# family: (name, roll symmetry n: the part looks the same after a roll of 2 pi / n about the insertion axis, 0 = any roll)
FAMILIES = {"A": ("circular", 0), "B": ("square", 4), "C": ("D-shaped", 1), "D": ("keyed", 1), "E": ("dual-peg", 1)}
MATED = np.diag([1.0, -1.0, -1.0])      # R of F_T_M when seated: z flipped, x shared


def wrap(a, period=2 * math.pi):
    return (a + period / 2) % period - period / 2


def relative_pose(T_cf, T_cm):
    """F_T_M from the two part poses in one camera frame."""
    return np.linalg.inv(T_cf) @ T_cm


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _quat(R):
    """(w, x, y, z) of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[1 + i] = s / 4
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _rotvec(R):
    q = _quat(R)
    if q[0] < 0:
        q = -q
    n = np.linalg.norm(q[1:])
    return np.zeros(3) if n < 1e-12 else q[1:] / n * 2 * math.atan2(n, q[0])


def rodrigues(r):
    a = np.linalg.norm(r)
    if a < 1e-12:
        return np.eye(3)
    k = r / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def swing_twist(R):
    """R = swing @ rot_z(twist), the swing about an axis in the xy plane: (swing angle, signed twist angle) [rad]."""
    q = _quat(R)
    return math.acos(float(np.clip(R[2, 2], -1.0, 1.0))), wrap(2 * math.atan2(q[3], q[0]))


@dataclass
class Mating:
    """How far M is from the seated pose (F frame): lateral [m], axial [m] (along +z_F, 0 at the gap), tilt of the insertion axes [rad], roll about
    the axis [rad] (to the nearest equivalent roll of the family), and the move that removes it: translation `dt` and rotation vector `dr` [rad]
    in the F frame, the rotation about the origin of M."""
    lateral: float
    axial: float
    tilt: float
    roll: float
    dt: np.ndarray
    dr: np.ndarray


def mating_error(T_fm, family, gap=0.0):
    """Error of the male part's pose in the female frame against the target `gap` in front of the seated pose (gap = 0: seated)."""
    n = FAMILIES[family][1]
    R, t = T_fm[:3, :3], T_fm[:3, 3]
    best = None
    for k in range(max(n, 1)):
        Rt = MATED @ rot_z(2 * math.pi * k / n) if n else MATED       # equivalent seated orientations (rolls of M about its own axis)
        Re = R @ Rt.T                                                   # rotation of M away from that target, in the F frame
        tilt, roll = swing_twist(Re)
        if n == 0:                                                      # any roll is fine: remove only the swing (Re = swing @ rot_z(roll))
            dr, roll = -_rotvec(Re @ rot_z(-roll)), 0.0
        else:
            dr = -_rotvec(Re)
        if best is None or tilt ** 2 + roll ** 2 < best[0] ** 2 + best[1] ** 2:
            best = (tilt, roll, dr)
    tilt, roll, dr = best
    target = np.array([0.0, 0.0, gap])
    return Mating(float(np.hypot(t[0], t[1])), float(t[2] - gap), tilt, roll, target - t, dr)


def apply(T_fm, dt, dr):
    """T_fm after moving M by `dt` and rotating it by `dr` (rotation vector) about its own origin, both in the F frame."""
    T = T_fm.copy()
    T[:3, :3] = rodrigues(np.asarray(dr, float)) @ T_fm[:3, :3]
    T[:3, 3] = T_fm[:3, 3] + dt
    return T


def to_base(T_base_f, dt, dr):
    """The same correction in the base frame (the frame the arm controllers use): rotate the F-frame vectors by the pose of F."""
    R = T_base_f[:3, :3]
    return R @ np.asarray(dt, float), R @ np.asarray(dr, float)


@dataclass
class Tolerance:
    lateral: float      # [m]
    tilt: float         # [rad]
    roll: float         # [rad]


def tolerance(clearance, engage, half_width, chamfer=0.0):
    """First-order limits for a straight insertion to start without jamming. `clearance` is the radial gap peg to hole [m], `engage` the length that
    must be in the hole before the peg is guided [m], `half_width` the peg's radius or half side [m], `chamfer` the lead-in at the hole mouth [m].
    A peg tilted by a needs a hole wider by about engage * a, so the tilt is at most 2 * clearance / engage; a roll b moves the peg's corners by
    half_width * b, which must stay inside the clearance. Each limit holds ALONE: lateral error and tilt add up (see fits(), which checks them jointly)."""
    return Tolerance(clearance + chamfer, 2 * clearance / engage, clearance / half_width)


def axis_offsets(T_fm, depth):
    """Where the male part's axis crosses the mouth plane of the hole (z_F = 0) and the plane `depth` inside it (z_F = -depth): lateral offsets (x, y) in the F
    frame [m]. The peg tip direction is z_M, which is -z_F when seated."""
    R, t = T_fm[:3, :3], T_fm[:3, 3]
    axis = R[:, 2]
    return [t[:2] + (zp - t[2]) / axis[2] * axis[:2] for zp in (0.0, -depth)]


def fits(T_fm, clearance, depth, chamfer=0.0):
    """Can a round peg with radial `clearance` slide into the hole to `depth` from this pose without touching the wall? Its axis must stay within the
    clearance circle at the mouth (plus the `chamfer` lead-in) and at depth; in between it is a straight line, so that is enough. Lateral error and tilt
    are NOT independent: a tilt moves the axis by depth * tilt at depth, on top of the lateral error (tolerance() gives each limit alone)."""
    mouth, deep = axis_offsets(T_fm, depth)
    return bool(np.linalg.norm(mouth) <= clearance + chamfer and np.linalg.norm(deep) <= clearance)


def within(m, tol):
    return m.lateral <= tol.lateral and abs(m.tilt) <= tol.tilt and abs(m.roll) <= tol.roll


# Landmark spheres (radius LANDMARK_R) of a part in its own frame: an ASSUMED layout for the interim cylinder-and-hole parts (parts.py: body radius 22.5 mm,
# 70 mm long behind the mating face). The parts stand upright on the table like the boxes and are picked from the side. Spheres are visual only (no collider,
# as on the boxes), so what limits them is visibility: three sit on a ring of radius 32 mm around the lower rear of the body (the lowest clears the base, the
# fingers hold the body above them) and one floats 40 mm out beside the mating face. Found by a random search maximising how badly any WRONG assignment of the
# landmarks fits: >= 6.1 mm rms for every assignment of 3 or 4 of them (noise is ~0.5 mm). A flat ring alone was ambiguous (a 180 degree turn about an in-plane
# axis fit it to ~1 mm); all distances >= 36 mm. At the 0.3 m the head rig sees the parts from, 12 mm spheres (35 px across) merged into one blob wherever two landmarks lined up along the
# view, so the spheres are 6 mm in radius (18 px across). Replace it when the final CAD is known.
LANDMARK_R = 0.006
HELD_MODEL = np.array([(0.0401, 0.0005, -0.0021), (0.0306, 0.0094, -0.0507), (0.0001, 0.0320, -0.0573), (-0.0300, -0.0112, -0.0509)])


# ---- pose of a held part from triangulated landmarks ---------------------------------------------------------------------------------------
def part_pose(points, model, max_rms=0.004, ambiguity=1.5):
    """Pose (R, t, rms) of a part from 3D points of its landmark colour, or None. `model` (n, 3) is the landmark layout in the part frame. The
    landmark ids are unknown: every assignment of the model landmarks to the points is fitted rigidly and the best kept; it is refused when the
    second best fit (a different assignment) is within `ambiguity` times the best rms, i.e. when the layout does not tell the assignments apart.
    No upright prior: held parts can have any orientation (examples/vservo/vision.py object_pose has one for boxes on the table)."""
    points, model = np.asarray(points, float), np.asarray(model, float)
    k = len(points)
    if k < 3 or k > len(model):
        return None
    fits = []
    for perm in itertools.permutations(range(len(model)), k):
        cm, co = model[list(perm)].mean(0), points.mean(0)
        U, _, Vt = np.linalg.svd((model[list(perm)] - cm).T @ (points - co))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t = co - R @ cm
        fits.append((float(np.sqrt(np.mean(np.sum((model[list(perm)] @ R.T + t - points) ** 2, axis=1)))), R, t))
    fits.sort(key=lambda f: f[0])
    best = fits[0]
    if best[0] > max_rms or (len(fits) > 1 and fits[1][0] < ambiguity * max(best[0], 1e-4)):
        return None
    return best[1], best[2], best[0]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    engage, half = 0.020, 0.0225
    print("insertion odds with 1 mm lateral / 1 deg tilt / 1 deg roll (1 sigma, per axis) alignment error, 20 mm of the 30 mm peg to go in, 1 mm chamfer")
    print("radial clearance | peg alone (circular) | with the roll limit (square, ...) | tilt limit 2c/L alone")
    for c in (0.0005, 0.001, 0.002, 0.003):
        n, ok_round, ok_square = 4000, 0, 0
        for _ in range(n):
            T = np.eye(4)
            T[:3, :3] = rodrigues(np.r_[rng.normal(0, math.radians(1), 2), 0.0]) @ MATED
            T[:3, 3] = np.r_[rng.normal(0, 0.001, 2), 0.030 - engage]
            f = fits(T, c, engage, chamfer=0.001)
            ok_round += f
            ok_square += f and abs(rng.normal(0, math.radians(1))) <= tolerance(c, engage, half).roll
        tol = tolerance(c, engage, half, chamfer=0.001)
        print(f"{c * 1000:8.1f} mm      | {ok_round / n:6.1%}               | {ok_square / n:6.1%}                            | {math.degrees(tol.tilt):.1f} deg")
    print("\nodds vs alignment accuracy (lateral mm, tilt deg: both 1 sigma per axis, tilt = lateral * 1 deg/mm), circular peg, 1 mm chamfer")
    print("radial clearance |  0.25 mm |  0.5 mm |  1 mm  |  2 mm")
    for c in (0.001, 0.002, 0.003):
        row = []
        for sl in (0.25, 0.5, 1.0, 2.0):
            n, ok = 3000, 0
            for _ in range(n):
                T = np.eye(4)
                T[:3, :3] = rodrigues(np.r_[rng.normal(0, math.radians(sl), 2), 0.0]) @ MATED
                T[:3, 3] = np.r_[rng.normal(0, sl / 1000, 2), 0.030 - engage]
                ok += fits(T, c, engage, chamfer=0.001)
            row.append(f"{ok / n:6.1%}")
        print(f"{c * 1000:8.1f} mm      | " + " | ".join(row))
