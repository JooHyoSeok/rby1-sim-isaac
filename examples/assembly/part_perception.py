"""Poses of the coloured parts from a stereo pair (the head rig or any other): the landmark spheres of one colour are triangulated (every epipolar-consistent
pair is kept, the wrong ones are ghosts at other depths) and the subset of 4 (else 3) that fits the part's landmark model rigidly, with no ambiguity, wins.
numpy only; the colour masks and the triangulation are examples/vservo/vision.py's.
"""
import itertools

import numpy as np

import assembly as A
import head_camera as H   # puts examples/vservo on the path
import vision as V

PART_COLORS = ("blue", "yellow")      # female / male
MAX_AREA = 600                        # [px] a 6 mm sphere seen by the head rig is up to ~300 px (0.3 m away); bigger blobs are scenery or merged spheres


def part_poses(img_l, img_r, T_l, T_r, colors=PART_COLORS, model=A.HELD_MODEL, radius=0.15):
    """{colour: (R, t, rms, landmarks used)} in the frame of T_l / T_r, for every part whose landmarks (at least 3) are seen by both cameras."""
    out = {}
    for c in colors:
        pts = V.stereo_points(img_l, img_r, T_l, T_r, color=c, max_area=MAX_AREA, all_pairs=True)
        best = None
        for k in (4, 3):
            for idx in itertools.combinations(range(len(pts)), k):
                sub = pts[list(idx)]
                if np.max(np.linalg.norm(sub - sub.mean(0), axis=1)) > radius:
                    continue
                got = A.part_pose(sub, model)
                if got is not None and (best is None or got[2] < best[2]):
                    best = (*got, k)
            if best is not None:
                break
        if best is not None:
            out[c] = best
    return out
