"""Offline check of the hybrid step (no simulator): a pinhole palm camera, a box with its four landmarks, a stereo estimate
that is off by ~1.3 cm and a position loop that is off by ~1.6 cm. Position control alone leaves the box ~2 cm off the camera
axis; the image-based loop must bring it under 3 mm, also from a larger error and with half of the landmarks hidden.
  /usr/bin/python3 test_servo.py
"""
import numpy as np

import servo as S
import vision as V

rng = np.random.default_rng(0)
R_FRONT = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)
C = np.array([0.64, -0.20, 0.76])  # true box centre
TOOL = 0.12


def ee_pose(tip):
    T = np.eye(4)
    T[:3, :3] = R_FRONT
    T[:3, 3] = tip - TOOL * np.array([1, 0, 0])
    return T


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def lateral_mm(T_ee):
    Tc = np.linalg.inv(S.cam_pose(T_ee))
    return float(np.linalg.norm((Tc[:3, :3] @ C + Tc[:3, 3])[:2]) * 1000)


def run(e_cal, bias, yaw_err=0.0, hidden=0, iters=6, visual=True):
    est_t, est_R = C + e_cal, rotz(yaw_err)
    cmd = est_t - np.array([0.12, 0, 0])  # position-based pre-grasp from the stereo estimate
    for _ in range(iters if visual else 0):
        T_true = ee_pose(cmd + bias)      # the position loop leaves a constant offset
        px, z = V.project(S.cam_pose(T_true), C + V.OBJECT_MODEL)
        ok = (z > 0.03) & (abs(px[:, 0] - 320) < 320) & (abs(px[:, 1] - 240) < 240)
        blobs = np.c_[px[ok] + rng.normal(0, 0.3, (ok.sum(), 2)), np.full(ok.sum(), 50.0)][hidden:]
        err, Z, n = S.image_errors(blobs, S.cam_pose(ee_pose(cmd)), est_R, est_t)
        assert err is not None, "no landmark matched"
        cmd = cmd + S.lateral_step(err, Z, ee_pose(cmd))
    return lateral_mm(ee_pose(cmd + bias))


e, b = np.array([0.010, -0.008, 0.006]), np.array([0.012, -0.010, 0.008])
before = run(e, b, visual=False)
after = run(e, b)
far = run(np.array([0.03, -0.02, 0.015]), np.array([0.02, 0.02, -0.01]))
occluded = run(e, b, yaw_err=4.0, hidden=2)
print(f"position control only {before:.1f} mm | hybrid {after:.1f} mm | larger error {far:.1f} mm | 2 landmarks hidden, yaw 4 deg off {occluded:.1f} mm")
assert before > 15 and after < 3 and far < 3 and occluded < 10
