"""Mixed-part assembly (term project, week 9 slides): the five interface families, the relative pose of two held parts, and what it takes to
bring the male part to the mating pose of the female one. numpy only, no simulator.

Frames. Every part has its own frame {P}: the origin at the centre of its mating face, +z pointing out of that face towards the partner (the peg tip
direction for a male part, out of the opening for a female one), +x towards the key / flat face / small pin (the roll reference). A male part {M}
is seated in a female part {F} when their origins coincide and +z_M = -z_F, +x_M = +x_F: F_T_M = MATED. Approaching, F_T_M has t = (0, 0, gap).
Poses are 4x4 homogeneous matrices; F_T_M = inv(C_T_F) @ C_T_M for the two parts seen by one camera (the camera-to-base transform cancels,
which is why an unknown camera or joint-zero error does not matter for this measurement: slide 10).
  /usr/bin/python3 assembly.py     prints the tolerance and the odds of fitting for a few clearances
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
    half_width * b, which must stay inside the clearance."""
    return Tolerance(clearance + chamfer, 2 * clearance / engage, clearance / half_width)


def within(m, tol):
    return m.lateral <= tol.lateral and abs(m.tilt) <= tol.tilt and abs(m.roll) <= tol.roll


# Landmark spheres (radius 12 mm) of a held part in its own frame: an ASSUMED layout for the interim cylinder-and-hole parts (body radius 22.5 mm, 70 mm
# long behind the mating face, held by the fingers near the face), on the rear half so the fingers do not hide them. Found by a random search
# for a layout whose every triangle of landmarks is scalene by at least 4.9 mm (two equal sides would let two assignments of the landmarks fit
# equally well, and with a landmark hidden the pose would be ambiguous) with all distances >= 35 mm (the spheres, 24 mm across, stay separate blobs).
# Replace it when the final CAD is known.
HELD_MODEL = np.array([(0.0, 0.0, -0.083), (0.009, -0.0271, -0.0473), (-0.0018, 0.0284, -0.0614), (-0.0236, -0.0159, -0.054)])


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
    print("clearance  engage  lateral  tilt   roll   | P(fits) with 1 mm / 1 deg / 1 deg (1 sigma) alignment error")
    for c in (0.0005, 0.001, 0.002, 0.003):
        tol = tolerance(c, 0.020, 0.0225, chamfer=0.001)
        n, ok = 4000, 0
        for _ in range(n):
            lat = np.hypot(*rng.normal(0, 0.001, 2))
            tilt = np.hypot(*rng.normal(0, math.radians(1), 2))
            roll = rng.normal(0, math.radians(1))
            ok += lat <= tol.lateral and tilt <= tol.tilt and abs(roll) <= tol.roll
        print(f"{c * 1000:6.1f} mm {0.020 * 1000:4.0f} mm {tol.lateral * 1000:5.1f} mm {math.degrees(tol.tilt):4.1f} deg {math.degrees(tol.roll):4.1f} deg | {ok / n:5.1%}")
