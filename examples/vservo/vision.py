"""Perception for the hybrid visual servoing: coloured-sphere blobs, stereo triangulation, box poses. numpy only.

Camera model = the simulator's (src/rby1_vision.py): 640x480, focal 15 mm on a 20.955 mm wide aperture, square pixels.
Poses are 4x4 T_world_camera with the ROS optical convention (x right, y down, z forward), as in rby1_gt.json.
"""
import itertools

import numpy as np

W, H = 640, 480
FX = FY = W * 15.0 / 20.955
K = np.array([[FX, 0, W / 2], [0, FY, H / 2], [0, 0, 1.0]])

# Landmarks of a box in the box frame (src/rby1_vision.py OBJECT_LANDMARKS); the box centre is the frame origin. Every box has
# the same layout in its own colour (src/rby1_vision.py BOXES); red spheres are the hand and calibration target landmarks.
OBJECT_MODEL = np.array([(0.0, 0.0, 0.052), (0.037, 0.0, 0.0), (-0.02, 0.037, 0.0), (-0.02, -0.037, 0.0)])
BOX_COLORS = ("blue", "green", "yellow")
LANDMARK_MAX_AREA = 900  # [px] a sphere seen by the stereo rig or the wrist camera is smaller than this: bigger blobs are scenery


def look_at_optical(eye, target, up=(0.0, 0.0, 1.0)):
    """T_world_camera of a camera at `eye` looking at `target` (the same construction as the simulator's USD cameras)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([r, -u, f])
    T[:3, 3] = eye
    return T


def color_mask(img, color="red"):
    """Pixels of a pure sphere colour. The tests on the other channels are ratios, so they hold in shade too, and they reject the
    pale tint of the box bodies, the white table and the blue floor (which has g/b ~ 0.4)."""
    r, g, b = (img[:, :, i].astype(int) for i in range(3))
    if color == "red":
        return (r > 120) & (g < 70) & (b < 70) & (r - g > 80)
    if color == "blue":
        return (b > 120) & (r < 0.2 * b) & (g < 0.2 * b)
    if color == "green":
        return (g > 120) & (r < 0.2 * g) & (b < 0.2 * g)
    if color == "yellow":
        return (r > 120) & (g > 120) & (b < 0.3 * np.minimum(r, g)) & (np.abs(r - g) < 0.25 * np.maximum(r, g))
    raise ValueError(f"unknown colour {color!r}")


def red_mask(img):
    return color_mask(img, "red")


def blobs(img, min_area=12, color="red", max_area=None):
    """Connected regions of one colour as (u, v, area), centroids in pixels (4-connectivity, breadth-first)."""
    mask = color_mask(img, color)
    ys, xs = np.nonzero(mask)
    todo = set(zip(xs.tolist(), ys.tolist()))
    out = []
    while todo:
        stack = [todo.pop()]
        pix = []
        while stack:
            x, y = stack.pop()
            pix.append((x, y))
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n in todo:
                    todo.remove(n)
                    stack.append(n)
        if len(pix) >= min_area and (max_area is None or len(pix) <= max_area):
            p = np.array(pix, float)
            out.append((p[:, 0].mean(), p[:, 1].mean(), len(pix)))
    return out


def projection(T_world_cam):
    Tc = np.linalg.inv(T_world_cam)
    return K @ Tc[:3]


def project(T_world_cam, pts):
    """Pixel coordinates (n, 2) and depths (n,) of world points."""
    pc = (np.linalg.inv(T_world_cam) @ np.c_[pts, np.ones(len(pts))].T)[:3]
    return (K @ pc)[:2].T / pc[2][:, None], pc[2]


def triangulate(P1, P2, uv1, uv2):
    A = np.array([uv1[0] * P1[2] - P1[0], uv1[1] * P1[2] - P1[1], uv2[0] * P2[2] - P2[0], uv2[1] * P2[2] - P2[1]])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def stereo_points(img_l, img_r, T_l, T_r, max_reproj=3.0, color="red", max_area=None, all_pairs=False):
    """3D positions of the spheres of one colour seen in both images: every blob pair is triangulated and kept when the point
    reprojects close to both blobs (the rig is nearly parallel, so this is an epipolar test); greedy by error.
    Two spheres in the same image row (the box landmarks Q3 and Q4) can be paired either way: both pairings pass the epipolar test and
    only the sub-pixel errors tell them apart. With `all_pairs` every passing pair is returned (the wrong ones are ghost points at
    other depths) and the caller picks the set that fits its model (box_poses: the rigid landmark layout)."""
    P1, P2 = projection(T_l), projection(T_r)
    bl, br = blobs(img_l, color=color, max_area=max_area), blobs(img_r, color=color, max_area=max_area)
    pairs = []
    for i, a in enumerate(bl):
        for j, b in enumerate(br):
            X = triangulate(P1, P2, a[:2], b[:2])
            (u1, v1), (u2, v2) = project(T_l, X[None])[0][0], project(T_r, X[None])[0][0]
            err = max(np.hypot(u1 - a[0], v1 - a[1]), np.hypot(u2 - b[0], v2 - b[1]))
            if err < max_reproj and project(T_l, X[None])[1][0] > 0:
                pairs.append((err, i, j, X))
    pairs.sort(key=lambda p: p[0])
    if all_pairs:
        return np.array([X for _, _, _, X in pairs])
    used_l, used_r, pts = set(), set(), []
    for err, i, j, X in pairs:
        if i not in used_l and j not in used_r:
            used_l.add(i), used_r.add(j), pts.append(X)
    return np.array(pts)


def fit_rigid(model, obs):
    """Arun/Kabsch: R, t with obs ~ R @ model + t, and the RMS residual."""
    cm, co = model.mean(0), obs.mean(0)
    U, _, Vt = np.linalg.svd((model - cm).T @ (obs - co))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = co - R @ cm
    return R, t, float(np.sqrt(np.mean(np.sum((model @ R.T + t - obs) ** 2, axis=1))))


def object_pose(points, model=OBJECT_MODEL, radius=0.12, max_rms=0.006, max_tilt_deg=8.0):
    """Pose of the box from the triangulated red points. The landmark ids are unknown: take the cluster of >=3 points
    within `radius` of each other and try every assignment of them to the model landmarks. The layout is close to mirror
    symmetric, so with three visible landmarks two assignments fit equally well (1.4 mm rms, 3 cm apart); the box stands
    upright on the table, which rejects the tilted one: fits whose z axis is more than `max_tilt_deg` off vertical are skipped."""
    best = None
    for k in (4, 3):
        for idx in itertools.combinations(range(len(points)), k):
            sub = points[list(idx)]
            if np.max(np.linalg.norm(sub - sub.mean(0), axis=1)) > radius:
                continue
            for perm in itertools.permutations(range(len(model)), k):
                R, t, rms = fit_rigid(model[list(perm)], sub)
                if R[2, 2] < np.cos(np.radians(max_tilt_deg)):
                    continue
                if rms < max_rms and (best is None or (k, -rms) > (best[0], -best[3])):
                    best = (k, R, t, rms, idx)
        if best is not None:
            break
    return None if best is None else (best[1], best[2], best[3], best[0])


def box_poses(img_l, img_r, T_l, T_r, colors=BOX_COLORS):
    """{colour: (R, t, rms, landmarks used)} of every box whose landmarks are seen by the stereo rig. A box needs three of its
    four landmarks: the one on top is hidden under a box stacked on it."""
    out = {}
    for c in colors:
        pts = stereo_points(img_l, img_r, T_l, T_r, color=c, max_area=LANDMARK_MAX_AREA, all_pairs=True)
        pose = object_pose(pts) if len(pts) >= 3 else None
        if pose is not None:
            out[c] = pose
    return out
