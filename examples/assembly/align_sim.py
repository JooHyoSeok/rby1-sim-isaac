"""Closed-loop alignment and insertion of two held parts, simulated without the simulator: every step the head stereo rig sees both parts (pixel noise,
landmark triangulation, pose fit), the relative pose gives the correction (assembly.mating_error), and the arm carries it out with a scale error and
tracking noise. Tells how much the observe-move-reobserve loop (slide 7) can achieve and which clearance the parts need.
  /usr/bin/python3 align_sim.py     success rate for a few clearances and measurement noise levels
"""
import math

import numpy as np

import assembly as A
import head_camera as H
import parts as P

Q = dict(H.SEED_TORSO) | {"head_1": 0.5}
CENTRE = (0.45, 0.0, 1.05)                        # the mating plane in front of the chest (meet_study.py: reachable)
START_GAP = P.PEG_L + 0.02                        # tip 20 mm in front of the mouth
GAPS = (0.035, 0.020, 0.010, 0.0)                 # then down in steps, looking again at each


def pose(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def meeting(gap, centre=CENTRE):
    """T_base_F (socket, left hand) and T_base_M (peg, right hand) with the insertion axis along base y, F's opening towards -y, key up."""
    RF = np.column_stack([(0, 0, 1), (-1, 0, 0), (0, -1, 0)]).astype(float)
    RM = np.column_stack([(0, 0, 1), (1, 0, 0), (0, 1, 0)]).astype(float)
    c = np.asarray(centre, float)
    return pose(RF, c), pose(RM, c + RF[:, 2] * gap)


def landmarks_base(T):
    return (T[:3, :3] @ A.HELD_MODEL.T).T + T[:3, 3]


def measure(T_f, T_m, rng, noise_px):
    """F_T_M as the head rig sees it (None when a part cannot be fitted)."""
    est = []
    for T in (T_f, T_m):
        got = A.part_pose(H.observe(landmarks_base(T), Q, noise_px=noise_px, rng=rng), A.HELD_MODEL)
        if got is None:
            return None
        est.append(pose(got[0], got[1]))
    return A.relative_pose(*est)


def trial(rng, clearance=P.CLEARANCE, noise_px=0.2, scale_err=0.10, track=(0.0003, math.radians(0.3)), tol=(0.0005, math.radians(0.5)), gain=0.7, max_iter=12, frames=5,
          start=(0.008, math.radians(3.0))):
    """One run from a start pose off by `start` (lateral m, tilt rad). Returns (success, iterations of the alignment, why not)."""
    T_f, T_m = meeting(START_GAP)
    T_fm = A.relative_pose(T_f, T_m)
    T_fm = A.apply(T_fm, np.r_[rng.normal(0, start[0] / 1.4, 2), 0.0], np.r_[rng.normal(0, start[1] / 1.4, 2), 0.0])
    scale = 1 + rng.uniform(-scale_err, scale_err, 6)               # the arm moves this much more or less than commanded, per axis, fixed in a run

    def move(dt, dr):
        nonlocal T_fm
        T_fm = A.apply(T_fm, dt * scale[:3] + rng.normal(0, track[0], 3), dr * scale[3:] + rng.normal(0, track[1], 3))

    def look(gap):
        """(dt, dr) of the correction, averaged over `frames` camera frames (the rig runs at 30 Hz; the arm is still while it looks)."""
        ms = [measure(T_f, T_f @ T_fm, rng, noise_px) for _ in range(frames)]
        if any(rel is None for rel in ms):
            return None
        ms = [A.mating_error(rel, "A", gap=gap) for rel in ms]
        return np.mean([m.dt for m in ms], axis=0), np.mean([m.dr for m in ms], axis=0)

    for it in range(1, max_iter + 1):                               # align in front of the hole
        m = look(START_GAP)
        if m is None:
            return False, it, "no pose"
        dt, dr = m
        if math.hypot(dt[0], dt[1]) <= tol[0] and math.hypot(dr[0], dr[1]) <= tol[1]:
            break
        move(gain * dt * np.array([1, 1, 0]), gain * dr)            # lateral and tilt only: the depth is the insertion's job
    # not converged within max_iter (the tolerance is below what the noise lets it see): insert anyway, the physical fit decides
    for g in GAPS:                                                  # insert, looking again at every step
        m = look(g)
        if m is None:
            return False, it, "no pose"
        move(*m)                                                    # full correction of lateral, tilt and depth for this step
        depth = P.PEG_L - g
        if depth > 0 and not A.fits(T_fm, clearance, depth, chamfer=0.001):
            return False, it, f"jammed at gap {g * 1000:.0f} mm"
    return A.fits(T_fm, clearance, P.PEG_L, chamfer=0.001) and abs(T_fm[2, 3]) < 0.002, it, "seated"


def rate(n=60, seed=0, **kw):
    rng = np.random.default_rng(seed)
    res = [trial(rng, **kw) for _ in range(n)]
    ok = [r for r in res if r[0]]
    return len(ok) / n, (np.mean([r[1] for r in ok]) if ok else float("nan")), res


if __name__ == "__main__":
    print("insertion of a 20 mm-deep peg after observe-move-reobserve (60 runs each; start 8 mm / 3 deg off; arm scale error 10 %, tracking noise 0.3 mm / 0.3 deg)")
    for frames in (5, 1):
        print(f"\naveraging {frames} camera frame(s) per look. radial clearance | pixel noise 0.2 | 0.5 | 1.0   (success rate, mean alignment iterations)")
        for c in (0.001, 0.002, 0.003):
            cells = []
            for npx in (0.2, 0.5, 1.0):
                r, it, _ = rate(clearance=c, noise_px=npx, frames=frames)
                cells.append(f"{r:5.0%} {it:4.1f} it")
            print(f"{c * 1000:8.1f} mm      | " + " | ".join(cells))
