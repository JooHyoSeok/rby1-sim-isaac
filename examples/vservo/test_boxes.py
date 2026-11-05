"""Offline check of the multi-box perception (no simulator): synthetic stereo images of coloured landmark spheres, with look-alike
distractors (pale box bodies, the blue floor, the white table, a red target sphere, shaded olive). Every box must come out with its own
colour and a pose within a few mm, also in a stack where the landmark on top of each lower box is hidden by the box above it.
  /usr/bin/python3 test_boxes.py
"""
import numpy as np

import vision as V

rng = np.random.default_rng(1)
RGB = {"blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0), "red": (255, 0, 0)}
EYE, TARGET, BASELINE = np.array([1.4, 0.0, 1.2]), np.array([0.5, 0.0, 0.85]), 0.12   # src/rby1_vision.py stereo rig
f = (TARGET - EYE) / np.linalg.norm(TARGET - EYE)
right = np.cross(f, [0, 0, 1.0])
right /= np.linalg.norm(right)
T_L = V.look_at_optical(EYE - right * BASELINE / 2, TARGET - right * BASELINE / 2)
T_R = V.look_at_optical(EYE + right * BASELINE / 2, TARGET + right * BASELINE / 2)


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def render(T, spheres, patches=()):
    """spheres: [(world point, colour name)], patches: [(u0, v0, u1, v1, rgb)] drawn first. Disc radius from the 12 mm sphere."""
    img = np.zeros((V.H, V.W, 3), np.uint8)
    for u0, v0, u1, v1, rgb in patches:
        img[v0:v1, u0:u1] = rgb
    vv, uu = np.mgrid[0:V.H, 0:V.W]
    for p, color in spheres:
        (u, v), z = (a[0] for a in V.project(T, np.array([p])))
        r = V.FX * 0.012 / z
        img[np.hypot(uu - u - rng.normal(0, 0.15), vv - v - rng.normal(0, 0.15)) < r] = RGB[color]
    return img


def landmarks(color, centre, yaw=0.0, hidden=()):
    R = rotz(yaw)
    return [(np.asarray(centre) + R @ m, color) for k, m in enumerate(V.OBJECT_MODEL) if k not in hidden], R


DISTRACTORS = [(100, 380, 540, 470, (20, 70, 130)),     # blue floor, g/b ~ 0.5
               (60, 300, 120, 340, (90, 128, 255)),     # pale blue body
               (200, 300, 260, 340, (115, 255, 115)),   # pale green body
               (300, 300, 360, 340, (255, 242, 140)),   # pale yellow body
               (400, 300, 460, 340, (127, 121, 64)),    # a yellow body in shade
               (500, 300, 560, 340, (230, 225, 215)),   # the white table
               (20, 20, 220, 120, (0, 0, 200))]         # a large flat blue area (not a sphere): must be rejected by size


def check(scene, label, tol=0.004):
    """scene: {colour: (centre, yaw, hidden landmark indices)} -> every box found with its own colour, within tol [m] and ~ yaw."""
    sph, truth = [(np.array([1.08, 0.24, 0.732]), "blue"), (np.array([1.02, 0.16, 0.732]), "yellow")], {}   # the calibration spheres (far end of the table)
    for color, (c, yaw, hidden) in scene.items():
        pts, R = landmarks(color, c, yaw, hidden)
        sph += pts
        truth[color] = (np.asarray(c), R)
    L, Rr = render(T_L, sph, DISTRACTORS), render(T_R, sph, DISTRACTORS)
    poses = V.box_poses(L, Rr, T_L, T_R)
    assert set(poses) == set(truth), f"{label}: found {sorted(poses)}, expected {sorted(truth)}"
    worst = 0.0
    for color, (c, R) in truth.items():
        err = np.linalg.norm(poses[color][1] - c)
        assert err < tol, f"{label}: {color} box off by {err * 1000:.1f} mm"
        assert np.allclose(poses[color][0], R, atol=0.15), f"{label}: {color} box orientation wrong"
        worst = max(worst, err)
    print(f"ok  {label}: {len(truth)} boxes, worst error {worst * 1000:.1f} mm")


for color in ("blue", "green", "yellow"):   # each mask finds its sphere and only that
    img = render(T_L, [(np.array([0.62, -0.2, 0.8]), color)], DISTRACTORS)
    found = {c: len(V.blobs(img, color=c, max_area=V.LANDMARK_MAX_AREA)) for c in V.BOX_COLORS + ("red",)}
    assert found[color] == 1 and sum(found.values()) == 1, (color, found)
print("ok  colour masks: one sphere per colour, none of the look-alikes")

row = {"blue": ((0.62, -0.34, 0.76), 0, ()), "green": ((0.62, -0.22, 0.76), 0, ()), "yellow": ((0.62, -0.10, 0.76), 0, ())}
check(row, "three boxes in a row")
check({**row, "blue": ((0.62, -0.34, 0.76), 25, ())}, "one box turned 25 degrees")
stack = {"green": ((0.62, -0.22, 0.76), 0, (0,)),            # the landmark on top of a lower box sits inside the box above it
         "blue": ((0.621, -0.221, 0.84), 0, (0,)),
         "yellow": ((0.620, -0.219, 0.92), 0, ())}
check(stack, "stack of three (top landmarks of the lower two hidden)")
check({"green": row["green"], "blue": ((0.62, -0.28, 0.76), 0, ())}, "two boxes 6 cm apart")   # closer than the finger clearance: still told apart
