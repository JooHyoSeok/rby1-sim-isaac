"""Hybrid visual servoing for the grasp: global position-based step from the stereo rig, local image-based step from the
palm camera, orientation held by the position loop. numpy only.

Why hybrid: the Cartesian position loop and the stereo calibration leave a steady-state error of a centimetre or so.
Position-based servoing cannot see it (it trusts its own pose estimate); the error between where the box *appears* in the
palm camera and where it should appear is measured in the image, independent of calibration and of the position loop.

Palm camera (src/rby1_vision.py WRIST_CAM): at the finger plane, looking along the tool axis (-z of the wrist link, which
is -z of the ee frame). Optical axes in the ee frame: x_opt = x_ee, y_opt = -y_ee, z_opt = -z_ee.
"""
import itertools

import numpy as np

from vision import FX, FY, K, OBJECT_MODEL, W, H

CAM_IN_EE = np.array([[1, 0, 0, 0.0], [0, -1, 0, 0.0], [0, 0, -1, -0.0789], [0, 0, 0, 1.0]])  # ee frame is 0.1261 below the wrist link origin, the camera at link z=-0.205
SEARCH_PX = 260    # blobs farther than this from the predicted box are other red things (table spheres, the other hand)
MAX_RESIDUAL_PX = 14  # blobs that differ from the predicted layout by more than a common shift are not the landmarks
MIN_DEPTH = 0.03   # [m] the lens must stay this far from the box


def cam_pose(T_world_ee):
    return T_world_ee @ CAM_IN_EE


def predicted_landmarks(T_world_cam, box_R, box_t):
    """Pixels and depths the box landmarks should have in the palm camera, given the box pose estimate."""
    world = box_t + OBJECT_MODEL @ box_R.T
    pc = (np.linalg.inv(T_world_cam) @ np.c_[world, np.ones(len(world))].T)[:3]
    return (K @ pc)[:2].T / pc[2][:, None], pc[2]


def match_landmarks(blob_px, pred_px):
    """Pair blobs with landmarks. The stereo estimate is off by a centimetre or so, which at 10-15 cm range shifts every
    landmark by tens of pixels but keeps their layout, so the blobs are assigned by how well they fit the predicted layout
    up to one common shift (all assignments of up to 4 blobs: at most a few hundred). More landmarks beat fewer."""
    if not len(blob_px):
        return []
    near = blob_px[np.hypot(*(blob_px[:, :2] - pred_px.mean(0)).T) < SEARCH_PX][:, :2]
    best = None
    for n in (4, 3, 2):
        for lms in itertools.combinations(range(len(pred_px)), n):
            for bls in itertools.permutations(range(len(near)), n):
                d = near[list(bls)] - pred_px[list(lms)]
                res = float(np.sqrt(np.mean(np.sum((d - d.mean(0)) ** 2, axis=1))))
                if res < MAX_RESIDUAL_PX and (best is None or res < best[0]):
                    best = (res, lms, bls)
        if best is not None:
            return [(k, near[b]) for k, b in zip(best[1], best[2])]
    j = int(np.argmin(np.hypot(*(near - pred_px.mean(0)).T))) if len(near) else None  # one blob: the nearest
    return [] if j is None else [(int(np.argmin(np.hypot(*(pred_px - near[j]).T))), near[j])]


def image_errors(blob_px, T_world_cam, box_R, box_t):
    """Pixel error of each visible landmark against where it appears when the box centre is on the optical axis.

    Landmark k sits at offset o_k = R q_k from the box centre. With the centre on the axis at depth Z it appears at
    (cx, cy) + f * (o_k expressed in the camera)_xy / Z, so the desired pixel depends only on the box orientation and on Z.
    Z is taken from the image itself (the spread of the matched landmarks against their known spread), not from the pose
    estimate: a wrong Z would shift the desired pixels by f * o / dZ and leave a steady-state error again.
    Returns (mean error [px], depth Z [m], number of matched landmarks), error None when nothing matched."""
    Tcw = np.linalg.inv(T_world_cam)
    Rcw = Tcw[:3, :3]
    px, _ = predicted_landmarks(T_world_cam, box_R, box_t)
    matched = match_landmarks(blob_px, px)
    if not matched:
        return None, float((Rcw @ box_t + Tcw[:3, 3])[2]), 0
    o = np.array([(Rcw @ (box_R @ OBJECT_MODEL[k]))[:2] for k, _ in matched])   # offsets in the camera, x/y [m]
    obs = np.array([p for _, p in matched])
    Z = max(float((Rcw @ box_t + Tcw[:3, 3])[2]), MIN_DEPTH)                     # fallback: the pose estimate
    if len(matched) >= 2:
        do, dp = o - o.mean(0), obs - obs.mean(0)
        Z = max(FX * float(np.sum(do * do) / np.sum(do * dp)), MIN_DEPTH) if np.sum(do * dp) > 0 else Z
    want = np.array([W / 2, H / 2]) + np.array([FX, FY]) * o / Z
    return np.mean(obs - want, axis=0), Z, len(matched)


def lateral_step(err_px, Z, T_world_ee, gain=0.7, max_step=0.03):
    """World-frame translation that moves the palm camera so the box centre comes to the optical axis."""
    d_cam = np.array([err_px[0], err_px[1], 0.0]) * Z / FX        # box right/below the axis -> move right/down
    d_ee = np.diag([1.0, -1.0, -1.0]) @ d_cam                      # optical -> ee axes
    step = gain * (T_world_ee[:3, :3] @ d_ee)
    n = np.linalg.norm(step)
    return step if n <= max_step else step * max_step / n
