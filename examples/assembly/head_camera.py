"""A stereo rig on the robot's head, for the assembly: where its two cameras are for given joint values (forward kinematics of the torso and head
chain), and a synthetic observation of 3D points through it (pixel noise, optionally a wrong head joint reading), with no simulator.
The mount is an ASSUMPTION (eyes in the frame of link_head_2, which looks along its +x with +y to the left and +z up); change MOUNT when the
camera is built in src/rby1_vision.py. Camera poses are 4x4 T_base_camera in the ROS optical convention (x right, y down, z forward), like
examples/vservo/vision.py expects.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("agent", "vservo"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
import kinematics as K  # noqa: E402
import vision as V  # noqa: E402

MOUNT = np.array([0.06, 0.0, 0.03])      # middle between the eyes in the frame of link_head_2 [m]
BASELINE = 0.12                          # distance between the eyes [m] (the external rig has 0.12 too)
SEED_TORSO = dict(zip(K.TORSO, [0, 1, -2, 1, 0, 0.0]))     # the squat of examples/gui/grasp_core.py


def stereo_poses(q):
    """(T_base_left, T_base_right) of the head rig for joint values q = {name: rad} (torso and head joints used)."""
    T = K.link_pose(q, "link_head_2")
    R = np.column_stack([-T[:3, 1], -T[:3, 2], T[:3, 0]])      # x right = -y_head, y down = -z_head, z forward = x_head
    out = []
    for side in (+1, -1):                                       # the left eye is on the +y side
        Tc = np.eye(4)
        Tc[:3, :3] = R
        Tc[:3, 3] = (T @ np.r_[MOUNT + np.array([0.0, side * BASELINE / 2, 0.0]), 1.0])[:3]
        out.append(Tc)
    return out[0], out[1]


def observe(points, q_true, q_believed=None, noise_px=0.2, rng=None):
    """3D points (n, 3) in the base frame -> pixels in both eyes (with noise) -> triangulated with the camera poses computed from `q_believed`
    (default: the true joints). Returns (n, 3). A wrong q_believed (joint zero offsets) moves the whole cloud, the same way for every point."""
    rng = rng or np.random.default_rng(0)
    TL, TR = stereo_poses(q_true)
    BL, BR = stereo_poses(q_believed if q_believed is not None else q_true)
    out = []
    for p in np.asarray(points, float):
        uv = [V.project(T, p[None])[0][0] + rng.normal(0, noise_px, 2) for T in (TL, TR)]
        out.append(V.triangulate(V.projection(BL), V.projection(BR), uv[0], uv[1]))
    return np.array(out)


def in_view(points, q, margin=40):
    """True when every point is in front of both eyes and inside the image (with a margin in px)."""
    for T in stereo_poses(q):
        uv, z = V.project(T, np.asarray(points, float))
        if np.any(z <= 0.1) or np.any(uv < margin) or np.any(uv[:, 0] > V.W - margin) or np.any(uv[:, 1] > V.H - margin):
            return False
    return True
