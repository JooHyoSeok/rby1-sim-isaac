"""Hybrid visual servoing for the grasp: global position-based step from the stereo rig, local image-based step from the
wrist camera on the back of the hand, orientation held by the position loop. numpy only.

Why hybrid: the Cartesian position loop and the stereo calibration leave a steady-state error of a centimetre or so.
Position-based servoing cannot see it (it trusts its own pose estimate); the box position measured in the *hand's* camera
is independent of calibration and of the position loop. Here it comes from the box landmarks in the image, the landmark
spread (depth) and the known box orientation, and the correction drives the box centre onto the tool line.

The wrist camera (src/rby1_vision.py WRIST_CAM): on the back of the hand, looking forward and down. Because its optical
axis is not the tool axis, the servo goal is "box centre on the tool line" (x = y = 0 in the ee frame), not "box centre
on the optical axis".
"""
import itertools

import numpy as np

from vision import FX, FY, K, OBJECT_MODEL, W, H

# Wrist camera pose in the wrist link frame (repeats src/rby1_vision.py): eye, aim point, up. Link z is 0.1261 above the ee frame.
WRIST_EYE = np.array([0.0, 0.08, -0.12])
WRIST_AIM = np.array([0.0, 0.0, -0.40])
WRIST_UP = np.array([0.0, 1.0, 0.0])
LINK_TO_EE = np.array([0.0, 0.0, 0.1261])  # ee origin = link origin + (0, 0, -0.1261): p_ee = p_link + (0, 0, 0.1261)


def _cam_in_ee():
    f = WRIST_AIM - WRIST_EYE
    f /= np.linalg.norm(f)
    r = np.cross(f, WRIST_UP)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([r, -u, f])  # optical frame: x right, y down, z forward (the ee frame has the link's axes)
    T[:3, 3] = WRIST_EYE + LINK_TO_EE
    return T


CAM_IN_EE = _cam_in_ee()
SEARCH_PX = 260       # blobs farther than this from the predicted box are other red things (table spheres, the other hand)
MAX_RESIDUAL_PX = 14  # blobs that differ from the predicted layout by more than a common shift are not the landmarks
MIN_DEPTH = 0.03      # [m]


def cam_pose(T_world_ee):
    return T_world_ee @ CAM_IN_EE


def predicted_landmarks(T_world_cam, box_R, box_t):
    """Pixels and depths the box landmarks should have in the wrist camera, given the box pose estimate."""
    world = box_t + OBJECT_MODEL @ box_R.T
    pc = (np.linalg.inv(T_world_cam) @ np.c_[world, np.ones(len(world))].T)[:3]
    return (K @ pc)[:2].T / pc[2][:, None], pc[2]


def match_landmarks(blob_px, pred_px):
    """Pair blobs with landmarks. The stereo estimate is off by a centimetre or so, which at 10-30 cm range shifts every
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


def box_in_ee(blob_px, T_world_cam, box_R, box_t):
    """Box centre in the ee frame, measured from the wrist camera image: (centre [m] or None, depth Z [m], landmarks used).

    Each matched landmark gives a ray; its depth is the depth Z of the box centre plus the landmark's known offset along the
    optical axis (box orientation R from the stereo estimate). Z comes from the image itself: the depth at which the centres
    implied by the individual landmarks agree best (two landmarks are enough). The pose estimate only provides the landmark
    association and R, so its position error (calibration) does not enter the result."""
    Tcw = np.linalg.inv(T_world_cam)
    Rcw = Tcw[:3, :3]
    px, _ = predicted_landmarks(T_world_cam, box_R, box_t)
    matched = match_landmarks(blob_px, px)
    Z_est = max(float((Rcw @ box_t + Tcw[:3, 3])[2]), MIN_DEPTH)
    if not matched:
        return None, Z_est, 0
    o = np.array([Rcw @ (box_R @ OBJECT_MODEL[k]) for k, _ in matched])   # landmark offsets from the centre, camera frame
    obs = np.array([p for _, p in matched])
    ray = np.c_[(obs[:, 0] - W / 2) / FX, (obs[:, 1] - H / 2) / FY]

    def centre_at(Z):  # every landmark k gives centre_k = ray_k * (Z + o_z,k) - o_k; they agree only at the right Z
        ck = np.c_[ray * (Z + o[:, 2:3]), Z + o[:, 2]] - o
        return ck.mean(0), float(np.sum((ck[:, :2] - ck[:, :2].mean(0)) ** 2))

    Z = Z_est
    if len(matched) >= 2:
        grid = np.linspace(MIN_DEPTH, 1.2, 600)
        Z = float(grid[np.argmin([centre_at(z)[1] for z in grid])])
        fine = np.linspace(Z - 0.004, Z + 0.004, 81)
        Z = float(fine[np.argmin([centre_at(z)[1] for z in fine])])
    c_cam = centre_at(Z)[0]
    return CAM_IN_EE[:3, :3] @ c_cam + CAM_IN_EE[:3, 3], Z, len(matched)


def lateral_step(center_ee, T_world_ee, gain=0.7, max_step=0.03):
    """World-frame translation that brings the box centre onto the tool line (x = y = 0 in the ee frame)."""
    step = gain * (T_world_ee[:3, :3] @ np.array([center_ee[0], center_ee[1], 0.0]))
    n = np.linalg.norm(step)
    return step if n <= max_step else step * max_step / n
