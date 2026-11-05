#!/usr/bin/env python3
"""Command-line skills for the RB-Y1 executor (servo_server.py): what a script or a language model calls to drive the robot.

  rbyctl.py scene                          the boxes as the stereo rig sees them (centre, what each rests on), the robot's state, free spots
  rbyctl.py pick green                     visual-servo to the green box, grasp, lift
  rbyctl.py place 0.62 -0.30               put the held box on the table at (x, y)
  rbyctl.py place-on green                 put the held box on top of the green box (aligned by the stereo rig)
  rbyctl.py move blue right                pick the blue box and put it on the free spot furthest right (left | near | far | center), or at x y
  rbyctl.py stack green blue yellow        bottom first; the first box stays where it is
  rbyctl.py unstack                        take a stack apart: top box first onto free table spots (its own start spot when free)
  rbyctl.py drive [--forward M] [--turn DEG] [--home]   roll the base (turn first, positive = left; at most 30 cm / 90 deg per command), or back to the start
  rbyctl.py home | seed | abort | gripper 0.0..1.0
  rbyctl.py set-box blue 0.62 -0.30 [--z Z] [--yaw DEG]   simulator only: teleport a box (a box dropped on another one: give --z)
  rbyctl.py reset                          simulator only: table and boxes back to their start positions

Plain python, standard library only. Prints the executor's log while a task runs, then OK or FAILED: <reason>. Exit code 0 when the task
succeeded, 1 when it failed or was refused (busy, out of reach, box not seen), 2 when the executor does not answer. Ctrl-C aborts the task.
"right" is the robot's right (-y, the right arm's side); the table is assumed to be at its default place.
"""
import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C  # noqa: E402

HOST = "127.0.0.1"
DIRECTIONS = ("right", "left", "near", "far", "center")


def send(msg, port):
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(json.dumps(msg).encode(), (HOST, port))
    tx.close()


def ask(msg, timeout, quiet=False, log=None):
    """Send a command to the executor with a reply port and return its final message ({"type": "result"|"scene", ...}) or None. The log
    lines of the job are printed on the way (stderr when quiet), or passed to `log` when given."""
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind((HOST, 0))
    rx.settimeout(0.5)
    send({**msg, "reply": rx.getsockname()[1]}, C.PORT_EXEC_CMD)
    t0, heard = time.time(), False
    try:
        while time.time() - t0 < timeout:
            try:
                m = json.loads(rx.recv(65536))
            except socket.timeout:
                if not heard and time.time() - t0 > 5.0:   # a live executor answers at once (its first log line, a result or the scene)
                    return None
                continue
            heard = True
            if m["type"] == "log":
                if log is not None:
                    log(m["text"])
                else:
                    print(m["text"], file=sys.stderr if quiet else sys.stdout, flush=True)
            else:
                return m
    except KeyboardInterrupt:
        send({"cmd": "abort"}, C.PORT_EXEC_CMD)
        print("interrupted: abort sent", file=sys.stderr)
        return {"type": "result", "ok": False, "text": "interrupted"}
    finally:
        rx.close()
    return None


def run_task(msg, timeout=600.0):
    r = ask(msg, timeout)
    if r is None:
        print("FAILED: the executor does not answer (is run_gui.sh / servo_server.py running?)")
        return 2
    if r["type"] == "result" and r["ok"]:
        print("OK")
        return 0
    print(f"FAILED: {r.get('text', r)}")
    return 1


def get_scene():
    r = ask({"cmd": "scene"}, 6.0, quiet=True)
    if r is None or r.get("type") != "scene":
        print("FAILED: the executor does not answer (is run_gui.sh / servo_server.py running?)" if r is None else f"FAILED: {r.get('text')}")
        sys.exit(2)
    return r


def positions(scene):
    return {c: tuple(b["pos"]) for c, b in scene["boxes"].items()}


def cmd_scene(args):
    s = get_scene()
    pos = positions(s)
    s["free_spots"] = {d: C.direction_spot(d, pos) for d in DIRECTIONS}
    s["unseen"] = [c for c in C.BOX_NAMES if c not in pos]
    print(json.dumps(s, indent=1))
    return 0


def cmd_move(args):
    if args.where[0] in DIRECTIONS:
        s = get_scene()
        pos = positions(s)
        if args.color not in pos:
            print(f"FAILED: cannot see the {args.color} box")
            return 1
        spot = C.direction_spot(args.where[0], pos, exclude=args.color)
        if spot is None:
            print(f"FAILED: no free reachable spot to the {args.where[0]} (the arm reaches a narrow band of the table: "
                  f"see `scene` free_spots)")
            return 1
        print(f"target spot {spot}", flush=True)
        x, y = spot
    else:
        x, y = float(args.where[0]), float(args.where[1])
    return run_task({"cmd": "pick_place", "color": args.color, "x": x, "y": y, "top": C.TABLE_TOP})


def cmd_set_box(args):
    msg = {"cmd": "set_box", "name": args.color, "x": args.x, "y": args.y, "yaw_deg": args.yaw}
    if args.z is not None:
        msg["z"] = args.z
    send(msg, C.PORT_SCENE)
    print("sent")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    color = lambda p: p.add_argument("color", choices=C.BOX_NAMES)  # noqa: E731
    sub.add_parser("scene").set_defaults(fn=cmd_scene)
    p = sub.add_parser("pick")
    color(p)
    p.set_defaults(fn=lambda a: run_task({"cmd": "pick", "color": a.color}))
    p = sub.add_parser("place")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.set_defaults(fn=lambda a: run_task({"cmd": "place", "x": a.x, "y": a.y, "top": C.TABLE_TOP}))
    p = sub.add_parser("place-on")
    p.add_argument("base", choices=C.BOX_NAMES)
    p.set_defaults(fn=lambda a: run_task({"cmd": "place_on", "base": a.base}))
    p = sub.add_parser("move")
    color(p)
    p.add_argument("where", nargs="+", help="right | left | near | far | center, or x y")
    p.set_defaults(fn=cmd_move)
    p = sub.add_parser("stack")
    p.add_argument("order", nargs="+", choices=C.BOX_NAMES, help="bottom first")
    p.set_defaults(fn=lambda a: run_task({"cmd": "stack", "order": a.order}))
    for verb in ("home", "seed", "unstack"):
        sub.add_parser(verb).set_defaults(fn=lambda a, v=verb: run_task({"cmd": v}))
    p = sub.add_parser("drive")
    p.add_argument("--forward", type=float, default=0.0, help="metres, negative = back")
    p.add_argument("--turn", type=float, default=0.0, help="degrees, positive = left")
    p.add_argument("--home", action="store_true", help="back to the start pose")
    p.set_defaults(fn=lambda a: run_task({"cmd": "drive", "forward": a.forward, "turn_deg": a.turn, "home": a.home}))
    sub.add_parser("abort").set_defaults(fn=lambda a: run_task({"cmd": "abort"}, 5.0))
    p = sub.add_parser("gripper")
    p.add_argument("closeness", type=float)
    p.set_defaults(fn=lambda a: run_task({"cmd": "gripper", "closeness": a.closeness}))
    p = sub.add_parser("set-box")
    color(p)
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("--z", type=float)
    p.add_argument("--yaw", type=float, default=0.0)
    p.set_defaults(fn=cmd_set_box)
    sub.add_parser("reset").set_defaults(fn=lambda a: (send({"cmd": "reset"}, C.PORT_SCENE), print("sent"), 0)[-1])
    args = ap.parse_args(argv)
    if args.verb == "move" and (args.where[0] not in DIRECTIONS) and len(args.where) != 2:
        ap.error("move: give one of right/left/near/far/center, or x y")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
