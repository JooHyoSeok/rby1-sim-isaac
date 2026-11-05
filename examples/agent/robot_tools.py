#!/usr/bin/env python3
"""Tools a language model uses to look at the RB-Y1, try joint values with forward kinematics, check a motion and run it.

  robot_tools.py state                          joint angles [deg], where the hand is [m], what it holds, the boxes
  robot_tools.py limits                         which joints may move, their ranges and speed limits, what each one does
  robot_tools.py fk right_arm_1=-90 head_0=30   where the arm / hand / head would be with these joint values (changes from the current pose)
  robot_tools.py check --json '{"unit": "deg", "waypoints": [{"t": 2, "right_arm_1": -90}, ...]}'     safety check only, nothing moves
  robot_tools.py move  --json '{...}'           the same check, and if it passes the robot plays the trajectory
  robot_tools.py save NAME | play NAME | skills   keep the last trajectory that ran under a name, play it again, list them

A trajectory: "t" is seconds from the previous waypoint, a joint a waypoint does not mention keeps its value. Frame: the robot's base, x forward,
y left, z up, metres. `--offline` pretends the robot hangs at all-zero joints and sends nothing (to try plans without the simulator).
Plain python + numpy. The executor (examples/gui/grasp_core.py move_joints) checks again before it moves: this check is a preview of that one.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "gui"))
import kinematics as K  # noqa: E402
import safety as S  # noqa: E402

SKILLS = os.path.join(HERE, "skills")
LAST = os.path.join(tempfile.gettempdir(), "rby1_last_trajectory.json")
D = 180.0 / np.pi
WHAT = {   # derived from the URDF axes and probed with forward kinematics; `fk` shows the real effect of any value
    "right_arm_0": "shoulder pitch: negative swings the arm forward and up, positive backward",
    "right_arm_1": "shoulder abduction: 0 = arm hangs at the side, negative lifts it sideways (-90 = horizontal)",
    "right_arm_2": "upper-arm roll: turns the forearm about the upper arm (nothing visible while the arm hangs straight)",
    "right_arm_3": "elbow: 0 = straight, negative bends it",
    "right_arm_4": "forearm roll",
    "right_arm_5": "wrist flex",
    "right_arm_6": "hand roll about the finger axis",
    "head_0": "head yaw: positive looks left, negative looks right",
    "head_1": "head pitch: positive looks down",
    "torso_5": "torso yaw: positive turns the upper body left (the arm moves with it)",
}


def offline_state():
    return {"joints": {}, "tip": K.points({})["hand_tip"].round(4).tolist(), "robot": {"holding": False, "held": None, "seeded": False}}


def get_state(offline):
    if offline:
        return offline_state()
    import rbyctl
    r = rbyctl.ask({"cmd": "state"}, 6.0, quiet=True)
    if r is None or r.get("type") != "state":
        sys.exit("FAILED: the executor does not answer" if r is None else f"FAILED: {r.get('text')}")
    return r


def get_boxes(offline):
    if offline:
        return {}
    import rbyctl
    r = rbyctl.ask({"cmd": "scene"}, 6.0, quiet=True)
    return {c: tuple(b["pos"]) for c, b in r["boxes"].items()} if r and r.get("type") == "scene" else {}


def hand_report(q):
    p = K.points(q)
    return {"hand_tip": p["hand_tip"].round(3).tolist(), "wrist": p["wrist"].round(3).tolist(), "elbow": p["elbow"].round(3).tolist(),
            "fingers_point_along": K.tip_direction(q).round(2).tolist(), "head_looks_along": K.head_forward(q).round(2).tolist()}


def load_traj(args):
    text = args.json if args.json else (open(args.file).read() if args.file else sys.stdin.read())
    try:
        return json.loads(text)
    except ValueError as exc:
        sys.exit(f"FAILED: the trajectory is not valid JSON ({exc})")


def out(obj):
    print(json.dumps(obj, indent=None, separators=(", ", ": ")))


def cmd_state(args):
    st = get_state(args.offline)
    q = st["joints"]
    out({"joints_deg": {n: round(q.get(n, 0.0) * D, 1) for n in sorted(S.ALLOWED)}, "other_joints": "torso_0..4, left arm, wheels: fixed, not yours to move",
         **hand_report(q), "holding_box": st["robot"].get("held"), "arm_in_ready_pose": bool(st["robot"].get("seeded")),
         "boxes_m": {c: [round(v, 3) for v in p] for c, p in get_boxes(args.offline).items()}})


def cmd_limits(args):
    out({"unit": "deg", "frame": "base: x forward, y left, z up [m]", "margin_kept_from_hard_limits_deg": round(S.MARGIN * D, 1),
         "joints": {n: {"min": round(S._limit(n)[0] * D, 1), "max": round(S._limit(n)[1] * D, 1), "max_speed_deg_s": round(S.V_MAX[n] * D),
                        "what": WHAT[n]} for n in sorted(S.ALLOWED)},
         "trajectory": {"max_waypoints": S.MAX_WAYPOINTS, "min_segment_s": S.MIN_SEGMENT, "max_total_s": S.MAX_TOTAL},
         "keep_clear": f"table top z={S.TABLE_TOP} (stay {S.TABLE_CLEAR} m above it where x > {S.TABLE_X}), the floor, the body's middle, the boxes"})


def cmd_fk(args):
    st = get_state(args.offline)
    q, given = dict(st["joints"]), {}
    k = np.pi / 180.0 if args.unit == "deg" else 1.0
    for item in args.assign:
        name, _, val = item.partition("=")
        if name not in K.JOINTS:
            sys.exit(f"FAILED: unknown joint {name!r}")
        try:
            q[name], given[name] = float(val) * k, float(val)
        except ValueError:
            sys.exit(f"FAILED: {item!r} is not joint=value")
    out({"with": given, "unit": args.unit, **hand_report(q)})


def cmd_check(args):
    traj = load_traj(args)
    q = get_state(args.offline)["joints"]
    violations, summary = S.check(q, traj, get_boxes(args.offline))
    out({"ok": not violations, "violations": violations, **summary})
    return 0 if not violations else 1


def run_trajectory(traj, offline):
    q = get_state(offline)["joints"]
    violations, summary = S.check(q, traj, get_boxes(offline))
    if violations:
        out({"ok": False, "violations": violations})
        return 1
    if offline:
        out({"ok": True, "sent": False, "note": "offline: checked, not sent", **summary})
        json.dump(traj, open(LAST, "w"))
        return 0
    import rbyctl
    r = rbyctl.ask({"cmd": "joints", "trajectory": traj}, 120.0, log=lambda text: print(text, file=sys.stderr, flush=True))
    ok = r is not None and r.get("type") == "result" and r.get("ok")
    out({"ok": bool(ok), "result": "the executor does not answer" if r is None else r.get("text"), **({"summary": summary} if ok else {})})
    if ok:
        json.dump(traj, open(LAST, "w"))
    return 0 if ok else 1


def cmd_move(args):
    return run_trajectory(load_traj(args), args.offline)


def cmd_save(args):
    if not os.path.exists(LAST):
        sys.exit("FAILED: no trajectory has run yet")
    os.makedirs(SKILLS, exist_ok=True)
    shutil.copy(LAST, os.path.join(SKILLS, f"{args.name}.json"))
    out({"saved": args.name})
    return 0


def cmd_play(args):
    path = os.path.join(SKILLS, f"{args.name}.json")
    if not os.path.exists(path):
        sys.exit(f"FAILED: no saved motion {args.name!r} (see `skills`)")
    return run_trajectory(json.load(open(path)), args.offline)


def cmd_skills(args):
    names = sorted(f[:-5] for f in os.listdir(SKILLS) if f.endswith(".json")) if os.path.isdir(SKILLS) else []
    out({"saved_motions": {n: round(sum(w["t"] for w in json.load(open(os.path.join(SKILLS, f"{n}.json")))["waypoints"]), 1) for n in names}})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", default=bool(os.environ.get("ROBOT_TOOLS_OFFLINE")),
                    help="pretend the robot hangs at zero joints; send nothing (also: environment ROBOT_TOOLS_OFFLINE=1)")
    sub = ap.add_subparsers(dest="verb", required=True)
    sub.add_parser("state").set_defaults(fn=cmd_state)
    sub.add_parser("limits").set_defaults(fn=cmd_limits)
    p = sub.add_parser("fk")
    p.add_argument("assign", nargs="*", metavar="joint=value")
    p.add_argument("--unit", choices=("deg", "rad"), default="deg")
    p.set_defaults(fn=cmd_fk)
    for verb, fn in (("check", cmd_check), ("move", cmd_move)):
        p = sub.add_parser(verb)
        g = p.add_mutually_exclusive_group()
        g.add_argument("--json", help="the trajectory as a JSON string")
        g.add_argument("--file", help="the trajectory in a file (default: standard input)")
        p.set_defaults(fn=fn)
    for verb, fn in (("save", cmd_save), ("play", cmd_play)):
        p = sub.add_parser(verb)
        p.add_argument("name")
        p.set_defaults(fn=fn)
    sub.add_parser("skills").set_defaults(fn=cmd_skills)
    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
