"""Offline checks of the assembly maths (no simulator): the mating error and its correction for every interface family, the tolerance numbers, the
pose of a held part from its landmarks, and what a head stereo rig measures of two held parts, also with wrong head joint readings.
  /usr/bin/python3 test_assembly.py
"""
import math

import numpy as np

import align_sim as S
import assembly as A
import head_camera as H
import parts as P
import part_perception as PP
import vision as V


def pose(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rot(axis, deg):
    return A.rodrigues(np.asarray(axis, float) * math.radians(deg))


SEATED = pose(A.MATED)


def test_mating():
    for fam in A.FAMILIES:
        m = A.mating_error(SEATED, fam)
        assert m.lateral < 1e-9 and abs(m.axial) < 1e-9 and m.tilt < 1e-6 and abs(m.roll) < 1e-6, fam
    # position: 3 mm and 4 mm sideways, 50 mm in front, asked for a 50 mm gap
    T = pose(A.MATED, (0.003, 0.004, 0.05))
    m = A.mating_error(T, "A", gap=0.05)
    assert abs(m.lateral - 0.005) < 1e-9 and abs(m.axial) < 1e-9
    assert abs(A.mating_error(T, "A").axial - 0.05) < 1e-9
    # tilt: M turned 2 degrees about the x axis of F
    m = A.mating_error(pose(rot((1, 0, 0), 2) @ A.MATED), "B")
    assert abs(math.degrees(m.tilt) - 2) < 1e-6 and abs(m.roll) < 1e-6
    # roll about the insertion axis, by family: circular any, square every 90, D / keyed / dual-peg only 360
    rolled = lambda deg: pose(A.MATED @ A.rot_z(math.radians(deg)))
    assert abs(A.mating_error(rolled(37), "A").roll) < 1e-9
    assert abs(math.degrees(A.mating_error(rolled(100), "B").roll)) - 10 < 1e-6
    assert abs(math.degrees(abs(A.mating_error(rolled(45), "B").roll)) - 45) < 1e-6
    for fam in "CDE":
        assert abs(math.degrees(abs(A.mating_error(rolled(90), fam).roll)) - 90) < 1e-6
        assert abs(math.degrees(abs(A.mating_error(rolled(180), fam).roll)) - 180) < 1e-6
    # the correction brings any pose to the target, whatever the family
    rng = np.random.default_rng(3)
    for fam in A.FAMILIES:
        for _ in range(20):
            R = A.rodrigues(rng.normal(0, 0.15, 3)) @ A.MATED @ A.rot_z(rng.uniform(-math.pi, math.pi))
            T = pose(R, rng.normal(0, 0.02, 3))
            before = A.mating_error(T, fam, gap=0.03)
            after = A.mating_error(A.apply(T, before.dt, before.dr), fam, gap=0.03)
            assert after.lateral < 1e-9 and abs(after.axial) < 1e-9 and after.tilt < 1e-6 and abs(after.roll) < 1e-6 or fam == "A" and after.tilt < 1e-6, (fam, after)
    # the same correction in the base frame: F standing rotated 90 degrees about z
    Rb = A.rot_z(math.pi / 2)
    dt, dr = A.to_base(pose(Rb), [0.01, 0, 0], [0, 0, 0.1])
    assert np.allclose(dt, [0, 0.01, 0]) and np.allclose(dr, [0, 0, 0.1])
    print("ok  mating error and correction, five families")


def test_tolerance():
    tol = A.tolerance(0.001, 0.020, 0.0225)
    assert abs(tol.lateral - 0.001) < 1e-12 and abs(math.degrees(tol.tilt) - math.degrees(0.1)) < 1e-9 and abs(math.degrees(tol.roll) - math.degrees(0.001 / 0.0225)) < 1e-9
    assert A.tolerance(0.001, 0.020, 0.0225, chamfer=0.002).lateral == 0.003
    assert A.within(A.mating_error(pose(A.MATED, (0.0005, 0, 0)), "A"), tol)
    assert not A.within(A.mating_error(pose(A.MATED, (0.002, 0, 0)), "A"), tol)
    print("ok  tolerance of a straight insertion")


def test_parts():
    walls = P.socket_walls()
    seated = pose(A.MATED, (0, 0, P.PEG_L - 0.020))                     # peg 20 mm in
    assert A.fits(seated, P.CLEARANCE, 0.020) and not P.collides(seated, walls)
    # sideways by 0.9 clearance fits, by clearance + 0.5 mm does not (towards a wall face, where the polygon hole is tightest)
    for k in range(P.N_WALLS):
        a = 2 * math.pi * k / P.N_WALLS
        d = lambda r: pose(A.MATED, (r * math.cos(a), r * math.sin(a), P.PEG_L - 0.020))
        assert not P.collides(d(0.9 * P.CLEARANCE), walls) and A.fits(d(0.9 * P.CLEARANCE), P.CLEARANCE, 0.020)
        assert P.collides(d(P.CLEARANCE + 0.0005), walls) and not A.fits(d(P.CLEARANCE + 0.0005), P.CLEARANCE, 0.020)
    # the analytic condition and the ring of boxes agree on random poses (outside a 0.3 mm band around the boundary: the polygon hole is up to 0.19 mm wider)
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(600):
        T = pose(A.rodrigues(np.r_[rng.normal(0, math.radians(1.5), 2), 0.0]) @ A.MATED, np.r_[rng.normal(0, 0.0015, 2), P.PEG_L - 0.020])
        mouth, deep = A.axis_offsets(T, 0.020)
        if abs(P.CLEARANCE - max(np.linalg.norm(mouth), np.linalg.norm(deep))) < 0.0003:
            continue
        assert A.fits(T, P.CLEARANCE, 0.020) == (not P.collides(T, walls))
        checked += 1
    assert checked > 300
    # tilt alone: the peg turned about the mouth by more than clearance / depth does not go in, a little less does
    tilt = lambda deg: pose(rot((1, 0, 0), deg) @ A.MATED, (0, 0, 0.0))
    limit = math.degrees(P.CLEARANCE / 0.020)
    assert A.fits(tilt(0.8 * limit), P.CLEARANCE, 0.020) and not A.fits(tilt(1.3 * limit), P.CLEARANCE, 0.020)
    print(f"ok  parts: hole as {P.N_WALLS} boxes agrees with the fit condition on {checked} random poses")


def test_part_pose():
    rng = np.random.default_rng(5)
    et, er = [], []
    for _ in range(60):
        R, t = A.rodrigues(rng.normal(0, 1.0, 3)), rng.normal(0, 0.2, 3)
        pts = A.HELD_MODEL @ R.T + t + rng.normal(0, 0.0005, (4, 3))      # 0.5 mm per coordinate; the landmarks span only ~5 cm, so the angle error is ~ noise / span
        Rf, tf, rms = A.part_pose(pts[rng.permutation(4)], A.HELD_MODEL)
        et.append(np.linalg.norm(tf - t))
        er.append(np.degrees(np.arccos(np.clip((np.trace(Rf.T @ R) - 1) / 2, -1, 1))))
    assert np.mean(et) < 0.002 and np.mean(er) < 2.0, f"pose of a part with 4 landmarks, 0.5 mm noise: {np.mean(et) * 1000:.1f} mm, {np.mean(er):.1f} deg"
    # three of the four: still unique, as the layout is scalene
    R, t = A.rodrigues(np.array([0.3, -0.5, 0.2])), np.array([0.4, 0.0, 1.0])
    for drop in range(4):
        pts = np.delete(A.HELD_MODEL, drop, axis=0) @ R.T + t
        got = A.part_pose(pts, A.HELD_MODEL)
        assert got is not None and np.linalg.norm(got[1] - t) < 1e-6, f"landmark {drop} hidden"
    assert A.part_pose(A.HELD_MODEL[:2] @ R.T + t, A.HELD_MODEL) is None                  # two points do not fix a pose
    iso = np.array([(0.0, 0.0, 0.0), (0.03, 0.0, 0.0), (-0.03, 0.0, 0.0), (0.0, 0.04, 0.0)])   # mirror symmetric: a 180 degree turn about y maps it to itself
    assert A.part_pose(iso[:3] @ R.T + t, iso) is None, "ambiguous layout must be refused"
    assert A.part_pose(A.HELD_MODEL @ R.T + t + np.array([0.02, 0, 0]) * np.array([[1], [0], [0], [0]]), A.HELD_MODEL) is None   # one landmark 2 cm off: not rigid
    print("ok  pose of a held part from landmarks (any orientation, 3 or 4 landmarks, ambiguity refused)")


def test_head_stereo():
    TF, TM = S.meeting(0.02)
    assert np.allclose(A.relative_pose(TF, TM), pose(A.MATED, (0, 0, 0.02)), atol=1e-12)
    lm = {name: (T[:3, :3] @ A.HELD_MODEL.T).T + T[:3, 3] for name, T in (("F", TF), ("M", TM))}
    q = dict(H.SEED_TORSO) | {"head_1": 0.5}
    assert H.in_view(np.vstack(list(lm.values())), q), "both parts must be in both images with the head pitched 0.5 rad"
    rng = np.random.default_rng(7)

    def measure(q_believed, runs=40):
        """(absolute error of F [m], error of F_T_M: lateral+axial [m], tilt+roll [deg]) over several noisy observations."""
        absolute, rel_t, rel_r = [], [], []
        for _ in range(runs):
            poses = {}
            for name in lm:
                pts = H.observe(lm[name], q, q_believed, noise_px=0.2, rng=rng)
                got = A.part_pose(pts, A.HELD_MODEL)
                assert got is not None, name
                poses[name] = pose(got[0], got[1])
            absolute.append(np.linalg.norm(poses["F"][:3, 3] - TF[:3, 3]))
            m = A.mating_error(A.relative_pose(poses["F"], poses["M"]), "C", gap=0.02)
            rel_t.append(math.hypot(m.lateral, m.axial))
            rel_r.append(math.degrees(math.hypot(m.tilt, m.roll)))
        return np.mean(absolute), np.mean(rel_t), np.mean(rel_r)

    a0, t0, r0 = measure(None)
    assert t0 < 0.004 and r0 < 2.5, f"relative pose of two held parts with exact joints: {t0 * 1000:.1f} mm, {r0:.1f} deg"
    bias = {k: math.radians(1.0) for k in ("head_0", "head_1", "torso_1", "torso_2", "torso_3")}      # every joint of the chain 1 degree off
    qb = {k: q.get(k, 0.0) + bias.get(k, 0.0) for k in set(q) | set(bias)}
    a1, t1, r1 = measure(qb)
    assert a1 > 3 * a0 and a1 > 0.008, f"1 degree of joint zero error should move the absolute pose by a centimetre or so, got {a1 * 1000:.1f} mm"
    assert t1 < 0.006 and r1 < 3.0, "the relative pose must stay as good (slide 10: camera-to-base cancels)"
    print(f"ok  head stereo, two held parts 0.4 m ahead: exact joints: absolute {a0 * 1000:.1f} mm, relative {t0 * 1000:.1f} mm / {r0:.1f} deg; "
          f"joints 1 deg off: absolute {a1 * 1000:.1f} mm, relative {t1 * 1000:.1f} mm / {r1:.1f} deg")


RGB = {"blue": (0, 0, 255), "yellow": (255, 255, 0), "red": (255, 0, 0)}


def render(T, spheres, rng, hidden=()):
    """Image of coloured spheres (radius A.LANDMARK_R) seen from T_base_camera: [(point, colour)]; `hidden` indices are left out."""
    img = np.zeros((V.H, V.W, 3), np.uint8)
    vv, uu = np.mgrid[0:V.H, 0:V.W]
    for i, (p, color) in enumerate(spheres):
        if i in hidden:
            continue
        (u, v), z = (a[0] for a in V.project(T, np.array([p])))
        img[np.hypot(uu - u - rng.normal(0, 0.15), vv - v - rng.normal(0, 0.15)) < V.FX * A.LANDMARK_R / z] = RGB[color]
    return img


def test_part_perception():
    rng = np.random.default_rng(2)
    TF, TM = S.meeting(0.02)
    q = dict(H.SEED_TORSO) | {"head_1": 0.5}
    TL, TR = H.stereo_poses(q)
    spheres = [(p, "blue") for p in S.landmarks_base(TF)] + [(p, "yellow") for p in S.landmarks_base(TM)] + [(np.array([0.30, -0.05, 0.95]), "red")]
    for label, hidden_l, hidden_r in (("all landmarks seen", (), ()), ("one landmark of the female hidden in the left image", (1,), ()),
                                      ("one of each part hidden in different images", (2,), (7,))):
        got = PP.part_poses(render(TL, spheres, rng, hidden_l), render(TR, spheres, rng, hidden_r), TL, TR)
        assert set(got) == {"blue", "yellow"}, f"{label}: found {sorted(got)}"
        for color, T in (("blue", TF), ("yellow", TM)):
            R, t, rms, k = got[color]
            assert np.linalg.norm(t - T[:3, 3]) < 0.003 and np.degrees(np.arccos(np.clip((np.trace(R.T @ T[:3, :3]) - 1) / 2, -1, 1))) < 3.0, f"{label}: {color}"
        rel = A.mating_error(A.relative_pose(*(np.block([[got[c][0], got[c][1][:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) for c in ("blue", "yellow"))), "A", gap=0.02)
        print(f"ok  part perception from images, {label}: relative pose {math.hypot(rel.lateral, rel.axial) * 1000:.1f} mm, {math.degrees(rel.tilt):.1f} deg off the truth")


def test_align_loop():
    easy, _, why = S.rate(n=20, seed=1, clearance=0.003, noise_px=0.2, frames=5)
    hard, _, _ = S.rate(n=20, seed=1, clearance=0.0003, noise_px=0.2, frames=5)
    assert easy >= 0.9, f"observe-move-reobserve with 3 mm clearance: {easy:.0%} seated"
    assert hard <= 0.5, f"a 0.3 mm clearance must fail often with this alignment accuracy, got {hard:.0%}"
    print(f"ok  alignment loop: {easy:.0%} seated with 3 mm clearance, {hard:.0%} with 0.3 mm (head stereo noise 0.2 px, 5 frames per look)")


if __name__ == "__main__":
    test_mating()
    test_tolerance()
    test_parts()
    test_part_pose()
    test_head_stereo()
    test_part_perception()
    test_align_loop()
    print("7 assembly checks passed")
