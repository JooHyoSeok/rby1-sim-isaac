"""Offline checks of the agent's tools (no simulator, no model): the FK against positions the simulator reported, the safety checker on a wave and on
every rule it enforces, and the tool CLI's file and argument handling.
  /usr/bin/python3 test_agent.py
"""
import os
import sys

import numpy as np

import kinematics as K
import safety as S

HANG = {}   # arm hanging, torso straight: all joints 0


def wave(osc=(-65, -115, -65, -115, -90)):
    wps = [{"t": 2.0, "right_arm_1": -90}, {"t": 2.0, "right_arm_2": -90, "right_arm_3": -90}]
    wps += [{"t": 1.2, "right_arm_2": a} for a in osc]
    wps += [{"t": 2.0, "right_arm_1": -10, "right_arm_2": 0, "right_arm_3": -10}]      # back down (not onto the limit: the margin keeps 5.7 deg)
    return {"unit": "deg", "waypoints": wps}


def test_kinematics():
    p = K.points(HANG)
    assert np.allclose(p["hand_tip"], [0.0006, -0.2198, 0.5917], atol=0.002)       # what the simulator reported for the hanging pose
    seed = dict(zip(K.TORSO, np.array([0, 1, -2, 1, 0, 0.0]))) | dict(zip(K.RIGHT_ARM, [0, 0, 0, -1.8, 0, 0.3, -1.57]))
    assert np.allclose(K.link_pose(seed, "link_right_arm_0")[:3, 3], [0, -0.22, 1.05], atol=0.005)   # the shoulder of the squat pose (scene_config.SHOULDER)
    assert K.tip_direction(seed)[0] > 0.95                                          # in the READY pose the fingertips point forward
    up = K.points({"right_arm_1": -np.pi / 2, "right_arm_2": -np.pi / 2, "right_arm_3": -np.pi / 2})
    assert up["wrist"][2] > up["elbow"][2] + 0.2 and up["hand_tip"][2] > 1.8        # forearm straight up at the side


def test_expand():
    plan = S.expand({"right_arm_3": -0.5}, wave())
    assert len(plan) == 8 and abs(plan[0][1]["right_arm_3"] - (-0.5)) < 1e-9 and abs(plan[1][1]["right_arm_3"] + np.pi / 2) < 1e-9
    assert abs(plan[0][1]["right_arm_1"] + np.pi / 2) < 1e-9 and abs(plan[0][1]["right_arm_2"]) < 1e-9     # unmentioned joints carry forward
    assert S.expand({}, {"unit": "rad", "waypoints": [{"t": 1, "head_0": 0.5}]})[0][1]["head_0"] == 0.5
    for bad in ({"waypoints": []}, {"waypoints": [{"head_0": 1}]}, {"waypoints": [{"t": 1, "bogus": 1}]}, {"unit": "grad", "waypoints": [{"t": 1}]}):
        try:
            S.expand({}, bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_wave_is_fine():
    v, summary = S.check(HANG, wave())
    assert v == [], v
    assert summary["duration_s"] == 2 + 2 + 5 * 1.2 + 2 and len(summary["waypoints"]) == 8
    assert summary["waypoints"][1]["hand_tip"][2] > 1.8 and summary["lowest_z"]["hand_tip"] > 0.5


def only(traj, **kw):
    v, _ = S.check(HANG, traj, **kw)
    return v


def test_rules():
    wp = lambda **j: {"waypoints": [{"t": 3.0, **j}]}   # noqa: E731
    # joints that may not move, unknown joints
    assert any("left_arm_0 may not be moved" in m for m in only(wp(left_arm_0=30)))
    assert any("right_wheel may not be moved" in m for m in only(wp(right_wheel=30)))
    assert any("torso_1 may not be moved" in m for m in only(wp(torso_1=30)))
    # limits: the URDF range minus a margin, and the tighter head / torso ranges
    assert any("right_arm_3" in m and "outside" in m for m in only(wp(right_arm_3=20)))             # the elbow only bends one way (-150..0 deg)
    assert any("right_arm_1" in m and "outside" in m for m in only(wp(right_arm_1=10)))
    assert any("head_0" in m and "outside" in m for m in only(wp(head_0=80)))
    assert any("torso_5" in m and "outside" in m for m in only(wp(torso_5=50)))
    assert only(wp(head_0=40, head_1=-20, torso_5=20)) == []
    # speed and timing
    assert any("deg/s" in m and "use t >=" in m for m in only({"waypoints": [{"t": 0.5, "right_arm_1": -90}]}))
    assert any("too short" in m for m in only({"waypoints": [{"t": 0.1, "head_0": 1}]}))
    assert any("at most 16 waypoints" in m for m in only({"waypoints": [{"t": 0.5, "head_0": i % 5} for i in range(17)]}))
    assert any("lasts" in m for m in only({"waypoints": [{"t": 3.0, "head_0": (i % 2) * 10} for i in range(11)]}))
    # clearance rules, point by point: the table (only over it, and only below the clearance), the floor, boxes
    rules = lambda name, p, boxes=None: {r for r, _ in S._clearance(name, p, boxes)}   # noqa: E731
    assert rules("hand_tip", (0.70, 0.0, 0.78)) == {"table"} and rules("hand_tip", (0.70, 0.0, 0.85)) == set()
    assert rules("hand_tip", (0.45, 0.0, 0.78)) == set()                        # in front of the table edge (0.58) it is free
    assert rules("elbow", (0.30, -0.5, 0.15)) == {"floor"}
    assert rules("hand_tip", (0.62, -0.22, 0.80), {"green": (0.62, -0.22, 0.76)}) == {"table", "box green"}
    assert rules("hand_tip", (0.62, -0.22, 1.00), {"green": (0.62, -0.22, 0.76)}) == set()
    seeded = dict(zip(K.TORSO, [0, 1, -2, 1, 0, 0.0])) | dict(zip(K.RIGHT_ARM, [0, 0, 0, -1.8, 0, 0.3, -1.57]))
    across = {"right_arm_0": 124, "right_arm_1": -29, "right_arm_2": 131, "right_arm_3": -79, "right_arm_4": -79, "right_arm_5": -83, "right_arm_6": 44}
    v, _ = S.check(HANG, {"waypoints": [{"t": 8.0, **across}]})
    assert any("too close to the body" in m for m in v), v
    seeded_ok, _ = S.check(seeded, {"unit": "rad", "waypoints": [{"t": 2.0, "right_arm_3": -1.8}]})   # holding the READY pose (hand just in front of the table) is fine
    assert seeded_ok == []
    box_in_the_way = {"blue": (0.0, -0.53, 1.9)}
    assert any("would touch the blue box" in m for m in only(wave(), boxes=box_in_the_way))
    # the same violation is reported once per waypoint and point, not once per sample
    v = only(wp(right_arm_3=20))
    assert len(v) == len(set(v))


def test_tools():
    """The tool CLI, offline: what it prints for the model, exit codes, and save / play / skills with a temporary skills directory."""
    import contextlib
    import io
    import json
    import tempfile

    import robot_tools as R
    R.SKILLS, R.LAST = tempfile.mkdtemp(), os.path.join(tempfile.mkdtemp(), "last.json")

    def run(*argv, stdin=None):
        out, old = io.StringIO(), sys.stdin
        sys.stdin = io.StringIO(stdin or "")
        try:
            with contextlib.redirect_stdout(out):
                rc = R.main(["--offline", *argv])
        except SystemExit as exc:
            rc, out = exc.code, io.StringIO(str(exc.code))
        finally:
            sys.stdin = old
        text = out.getvalue()
        return rc, (json.loads(text) if text.lstrip().startswith("{") else text)
    rc, st = run("state")
    assert rc == 0 and st["joints_deg"]["right_arm_3"] == 0.0 and st["holding_box"] is None and "head_looks_along" in st
    rc, lim = run("limits")
    assert lim["joints"]["right_arm_1"]["max"] < 0 and lim["joints"]["head_0"]["max"] == round(S._limit("head_0")[1] * 180 / np.pi, 1) and "left_arm_0" not in lim["joints"]
    rc, fk = run("fk", "right_arm_1=-90", "right_arm_2=-90", "right_arm_3=-90")
    assert fk["hand_tip"][2] > 1.8 and fk["with"]["right_arm_3"] == -90.0
    rc, fk = run("fk", "head_0=40")
    assert fk["head_looks_along"][1] > 0.5                                                  # positive head_0 looks left
    assert run("fk", "bogus=1")[0] != 0 and run("fk", "right_arm_1=abc")[0] != 0
    # check / move: a good trajectory passes, a bad one comes back with the reasons (exit code 1), garbage is refused
    good = json.dumps(wave())
    rc, res = run("check", "--json", good)
    assert rc == 0 and res["ok"] and res["duration_s"] == 12.0
    rc, res = run("check", "--json", json.dumps({"waypoints": [{"t": 0.5, "right_arm_1": -90}]}))
    assert rc == 1 and not res["ok"] and "deg/s" in res["violations"][0]
    rc, res = run("move", stdin=good)                                                       # standard input works too
    assert rc == 0 and res["ok"] and res["sent"] is False
    assert run("move", "--json", "{nope")[0] != 0
    # keep what ran, play it again, list it
    assert run("skills")[1] == {"saved_motions": {}}
    assert run("save", "wave")[0] == 0 and run("skills")[1] == {"saved_motions": {"wave": 12.0}}
    rc, res = run("play", "wave")
    assert rc == 0 and res["ok"] and run("play", "nothing")[0] != 0


if __name__ == "__main__":
    tests = [test_kinematics, test_expand, test_wave_is_fine, test_rules, test_tools]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} agent checks passed")
