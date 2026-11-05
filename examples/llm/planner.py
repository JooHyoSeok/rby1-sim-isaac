#!/usr/bin/env python3
"""Natural language -> robot skills. A language model plans, the visual-servo executor does the work.

  planner.py "파란 걸 오른쪽에 놔"
  planner.py "stack all three boxes"            --dry-run     (print the plan, move nothing)
  planner.py "노란 상자를 맨 아래로 해서 쌓아"  --scene default --dry-run   (no simulator: the start row of the boxes)

Requests no skill covers (waving, nodding, looking left or right) go to the `improvise` skill: a second model with joint-level tools
(examples/agent/improvise.py). The model gets the scene as text (boxes seen by the stereo rig, what rests on what, what the hand holds, where a box could go) and
answers with a short JSON plan over a handful of skills (stack, move, pick, place, place_on, home). It is called once per command, not
in any control loop (the servo loop runs at 20 Hz in the executor and uses no model), and no image is sent. The plan is checked
against the same reach and clearance rules the executor uses before anything moves; an invalid plan goes back to the model once with
the reasons. After a failed step the scene is read again and the model plans once more from there (--replan).

The model is called with `claude -p` (no tools, no settings, no session): it uses your Claude Code login, no API key. Default model
claude-haiku-5-5, escalating to claude-sonnet-5-5 when the first plan is invalid. Plain python, standard library only.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gui"))
import scene_config as C  # noqa: E402

DIRECTIONS = ("right", "left", "near", "far", "center")
SKILLS = {"stack": {"order"}, "unstack": set(), "move": {"color", "to"}, "pick": {"color"}, "place": {"x", "y"}, "place_on": {"base"}, "home": set(),
          "improvise": {"request"}, "drive": set()}
OPTIONAL = {"drive": {"forward_m", "turn_deg", "home"}}
MODEL, ESCALATE = "claude-haiku-5-5", "claude-sonnet-5-5"

SYSTEM = """You plan tasks for a humanoid robot (RB-Y1) that moves three 5 cm boxes on a table with its RIGHT arm. You do not control the motors: you
choose skills; a visual-servo executor carries them out and checks reach and clearance.

Boxes: "blue", "green", "yellow". Frame: x forward from the robot, y to the robot's left, so the robot's RIGHT is -y. Words like
right / left / near (towards the robot) / far (away from it) / center refer to that frame.

Skills (JSON objects in "steps", run in order). "Put the boxes back / take the stack down / unstack" means the unstack skill:
  {"skill": "stack", "order": ["green", "blue", "yellow"]}   stack boxes, bottom first. The first box stays where it is; every other one is
                                                             picked up and put on top of the one before it.
  {"skill": "unstack"}                                       take a stack apart again: the boxes that rest on others are put back on the
                                                             table, top first, each on a free spot (its own start spot if free). Needs a stack.
  {"skill": "move", "color": "blue", "to": "right"}          pick the box and put it on the free table spot furthest right | left | near |
                                                             far, or the best central spot ("center"), or at {"to": [x, y]}.
  {"skill": "pick", "color": "blue"}                         pick the box up (the hand must be empty)
  {"skill": "place", "x": 0.62, "y": -0.3}                   put the held box on the table at (x, y)
  {"skill": "place_on", "base": "green"}                     put the held box on top of the green box
  {"skill": "home"}                                          open the hand and return to the ready pose (hand must be empty)
  {"skill": "drive", "forward_m": -0.2, "turn_deg": 30}      roll the base on its wheels: first turn_deg degrees (positive = left, negative = right, at
                                                             most 90), then forward_m metres (negative = back, at most 0.3); or {"skill": "drive", "home": true}
                                                             to roll back where it started. Refused when the arm or base would hit the table. Until the base is
                                                             back, pick / stack / home do not work.
  {"skill": "improvise", "request": "손 흔들어"}               a body motion none of the skills above covers (waving, nodding, looking left or right,
                                                             raising the arm): a second model designs it with joint-level tools. NOT for moving boxes;
                                                             the hand must be empty. Pass the user's wish as "request", in the user's words.

You are given the scene as JSON: "boxes" (position in metres, "on": the box it rests on), "unseen" (boxes the cameras cannot see), "holding"
(the colour in the hand or null), "possible_moves" (for each box and direction the spot it would be put on, or null when there is no free
reachable spot or it is already there), and "suggested_stack_order" (the order that is easiest to reach: use it unless the user names an order).

Rules: use only these skills and only boxes that are seen. Do not invent positions: prefer directions over coordinates. If the request is
impossible, unclear, or already satisfied, return no steps and say why. If the hand holds a box, the plan must start by putting it down
(place or place_on). Answer with ONE JSON object and nothing else:
  {"say": "<one short sentence to the user, in the user's language>", "steps": [ ... ]}"""


class PlanError(Exception):
    pass


def resolve_spot(color, to, boxes):
    """Where the `color` box would be put for direction `to`, or None: no free reachable spot that way, or it already stands there."""
    spot = C.direction_spot(to, boxes, exclude=color)
    if spot is None or ((spot[0] - boxes[color][0]) ** 2 + (spot[1] - boxes[color][1]) ** 2) ** 0.5 <= 0.03:
        return None
    return list(spot)


# ---- context for the model (pure) -----------------------------------------------------------------------------------------------
def build_context(scene):
    boxes = {c: tuple(b["pos"]) for c, b in scene["boxes"].items()}
    ctx = {"boxes": {c: {"pos": [round(v, 2) for v in b["pos"]], "on": b.get("on")} for c, b in scene["boxes"].items()},
           "unseen": [c for c in C.BOX_NAMES if c not in boxes], "holding": scene.get("robot", {}).get("held"),
           "possible_moves": {}}
    for c in boxes:
        moves = {}
        for d in DIRECTIONS:
            moves[d] = resolve_spot(c, d, boxes)
        ctx["possible_moves"][c] = moves
    ctx["suggested_stack_order"] = sorted(boxes, key=lambda c: (abs(boxes[c][1] - C.SHOULDER[1]), c))
    return ctx


# ---- the model's answer (pure) ----------------------------------------------------------------------------------------------------
def parse_plan(text):
    """The first JSON object in the answer (the model may wrap it in a code fence or a sentence)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise PlanError("the answer contains no JSON object")
    try:
        plan = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise PlanError(f"the answer is not valid JSON ({exc})") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("steps", []), list):
        raise PlanError('the answer must be {"say": ..., "steps": [...]}')
    plan.setdefault("steps", [])
    plan.setdefault("say", "")
    return plan


def validate(plan, scene):
    """Reasons why `plan` cannot be run on `scene` (empty list: fine). The same rules the executor applies, checked up front."""
    errors, boxes = [], {c: tuple(b["pos"]) for c, b in scene["boxes"].items()}
    holding = scene.get("robot", {}).get("held")
    for i, step in enumerate(plan["steps"], 1):
        where = f"step {i}"
        skill = step.get("skill") if isinstance(step, dict) else None
        if skill not in SKILLS:
            errors.append(f"{where}: unknown skill {skill!r}")
            continue
        extra = set(step) - SKILLS[skill] - OPTIONAL.get(skill, set()) - {"skill"}
        missing = SKILLS[skill] - set(step)
        if extra or missing:
            errors.append(f"{where}: {skill} takes {sorted(SKILLS[skill])}" + (f", got extra {sorted(extra)}" if extra else "")
                          + (f", missing {sorted(missing)}" if missing else ""))
            continue
        needs_empty = skill in ("stack", "unstack", "move", "pick", "home", "improvise")
        if i == 1 and needs_empty and holding:
            errors.append(f"{where}: the hand holds the {holding} box: put it down first (place or place_on)")
        if i == 1 and skill in ("place", "place_on") and not holding:
            errors.append(f"{where}: {skill} needs a box in the hand and the hand is empty")
        if skill == "drive":
            f, t, home = step.get("forward_m", 0.0), step.get("turn_deg", 0.0), step.get("home", False)
            if not (isinstance(f, (int, float)) and isinstance(t, (int, float)) and isinstance(home, bool)):
                errors.append(f"{where}: forward_m and turn_deg are numbers, home is true or false")
            elif not home and not f and not t:
                errors.append(f"{where}: drive needs forward_m, turn_deg or home")
            elif abs(f) > 0.3 or abs(t) > 90:
                errors.append(f"{where}: at most 0.3 m and 90 degrees per drive step (split it into several steps)")
        elif skill == "improvise":
            if not isinstance(step["request"], str) or not step["request"].strip():
                errors.append(f"{where}: request must be the user's wish as text")
        elif skill == "unstack":
            if not C.unstack_order(boxes):
                errors.append(f"{where}: nothing is stacked")
        elif skill == "stack":
            errors += [f"{where}: {p}" for p in C.stack_problems(step["order"], {c: p for c, p in boxes.items()})] \
                if isinstance(step["order"], list) else [f"{where}: order must be a list of colours"]
        elif skill in ("move", "pick"):
            c = step["color"]
            if c not in boxes:
                errors.append(f"{where}: the {c} box is not visible")
            elif skill == "pick" and (p := C.pick_problem(c, boxes)):
                errors.append(f"{where}: {p}")
            elif skill == "move":
                to = step["to"]
                if isinstance(to, str) and to in DIRECTIONS:
                    if resolve_spot(c, to, boxes) is None:
                        errors.append(f"{where}: there is no free reachable spot to the {to} for the {c} box (or it is already there)")
                elif isinstance(to, list) and len(to) == 2 and all(isinstance(v, (int, float)) for v in to):
                    if (p := C.check_pose(to[0], to[1], C.TABLE_TOP)):
                        errors.append(f"{where}: ({to[0]}, {to[1]}) is not possible: {p}")
                else:
                    errors.append(f"{where}: 'to' must be one of {list(DIRECTIONS)} or [x, y]")
        elif skill == "place":
            if not all(isinstance(step[k], (int, float)) for k in ("x", "y")):
                errors.append(f"{where}: x and y must be numbers")
            elif (p := C.check_pose(step["x"], step["y"], C.TABLE_TOP)):
                errors.append(f"{where}: ({step['x']}, {step['y']}) is not possible: {p}")
        elif skill == "place_on" and step["base"] not in boxes:
            errors.append(f"{where}: the {step['base']} box is not visible")
    return errors


# ---- the model call ---------------------------------------------------------------------------------------------------------------
def ask_claude(system, user, model, timeout=120):
    """(answer text, cost in USD at list price, usage dict) from `claude -p`: no tools, no settings (hooks, plugins, memory), no session."""
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--system-prompt", system, "--no-session-persistence",
           "--disable-slash-commands", "--setting-sources", "", "--strict-mcp-config", "--output-format", "json"]
    try:   # run from /tmp: a CLAUDE.md in the project tree would be added to the prompt
        done = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=timeout, cwd=tempfile.gettempdir())
    except FileNotFoundError as exc:
        raise PlanError("the `claude` command is not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise PlanError(f"claude -p did not answer within {timeout} s") from exc
    if done.returncode != 0:
        raise PlanError(f"claude -p failed ({done.returncode}): {done.stderr.strip()[:300] or done.stdout.strip()[:300]}")
    out = json.loads(done.stdout)
    if out.get("is_error"):
        raise PlanError(f"claude -p reported an error: {out.get('result')}")
    return out["result"], out.get("total_cost_usd", 0.0), out.get("usage", {})


def make_plan(command, scene, llm, models, say=print, verbose=False):
    """Ask the model, validate; on an invalid plan ask again (next model of `models`) with the reasons. Returns (plan, calls)."""
    user = f"Scene:\n{json.dumps(build_context(scene), ensure_ascii=False)}\n\nUser: {command}"
    calls, last_errors = [], []
    for model in models:
        prompt = user if not last_errors else user + "\n\nYour previous plan was rejected:\n- " + "\n- ".join(last_errors) + "\nPlan again."
        try:
            text, cost, usage = llm(SYSTEM, prompt, model)
            calls.append({"model": model, "cost": cost, "usage": usage})
            if verbose:
                say(f"[{model}] {text}")
            plan = parse_plan(text)
            last_errors = validate(plan, scene)
        except PlanError as exc:
            last_errors = [str(exc)]
            plan = None
        if not last_errors:
            return plan, calls
        say(f"plan rejected ({model}): " + "; ".join(last_errors))
    raise PlanError("no valid plan: " + "; ".join(last_errors))


# ---- running the plan -------------------------------------------------------------------------------------------------------------
def to_command(step, scene):
    """Executor command for one step; `move` is resolved on the scene as it is when the step starts."""
    skill = step["skill"]
    if skill == "stack":
        return {"cmd": "stack", "order": step["order"]}
    if skill == "improvise":
        return {"cmd": "improvise", "request": step["request"]}
    if skill == "drive":
        return {"cmd": "drive", "forward": step.get("forward_m", 0.0), "turn_deg": step.get("turn_deg", 0.0), "home": step.get("home", False)}
    if skill == "unstack":
        return {"cmd": "unstack"}
    if skill == "pick":
        return {"cmd": "pick", "color": step["color"]}
    if skill == "place":
        return {"cmd": "place", "x": step["x"], "y": step["y"], "top": C.TABLE_TOP}
    if skill == "place_on":
        return {"cmd": "place_on", "base": step["base"]}
    if skill == "home":
        return {"cmd": "home"}
    boxes = {c: tuple(b["pos"]) for c, b in scene["boxes"].items()}
    to = step["to"]
    spot = resolve_spot(step["color"], to, boxes) if isinstance(to, str) else tuple(to)
    if spot is None:
        raise PlanError(f"no free reachable spot to the {to} for the {step['color']} box (or it is already there)")
    return {"cmd": "pick_place", "color": step["color"], "x": spot[0], "y": spot[1], "top": C.TABLE_TOP}


def execute(plan, run_task, get_scene, say=print):
    """Run the steps in order. `run_task(cmd)` returns (0 | 1 | 2, text): done, failed, no executor. Returns (ok, why not): stops at the first
    failing step."""
    for i, step in enumerate(plan["steps"], 1):
        say(f"== step {i}/{len(plan['steps'])}: {json.dumps(step, ensure_ascii=False)}")
        try:
            cmd = to_command(step, get_scene())
        except PlanError as exc:
            return False, str(exc)
        rc, text = run_task(cmd)       # "improvise" is not an executor command: the run_task of main() hands it to examples/agent/improvise.py
        if rc != 0:
            return False, f"step {i} ({step['skill']}) failed: {text}"
    return True, ""


def run(command, scene_source, llm=ask_claude, models=(MODEL, ESCALATE), dry_run=False, replan=1, run_task=None, say=print, verbose=False):
    """Plan and (unless dry_run) execute. Returns an exit code: 0 done or nothing to do, 1 failed."""
    total = 0.0
    for attempt in range(replan + 1):
        scene = scene_source()
        try:
            plan, calls = make_plan(command, scene, llm, models, say, verbose)
        except PlanError as exc:
            say(f"FAILED: {exc}")
            return 1
        total += sum(c["cost"] for c in calls)
        say(f"robot: {plan['say']}" if plan["say"] else "robot: (no comment)")
        say(f"plan ({calls[-1]['model']}, ${sum(c['cost'] for c in calls):.4f} list price): {json.dumps(plan['steps'], ensure_ascii=False)}")
        if not plan["steps"] or dry_run:
            return 0
        ok, why = execute(plan, run_task, scene_source, say)
        if ok:
            say(f"OK (language model cost ${total:.4f} at list price)")
            return 0
        say(f"FAILED: {why}")
        if attempt == replan or any(w in why for w in ("aborted", "interrupted", "does not answer")):
            return 1                      # no endless retries on hardware; an abort or a dead executor is not retried at all
        say("planning again from the current scene ...")
        command = f"{command}\n(The previous attempt failed: {why}. The scene above is the current one: plan what is still to do.)"
    return 1


def default_scene():
    return {"boxes": {n: {"pos": [*xy, 0.76], "on": None} for n, xy in C.BOX_DEFAULTS.items()}, "robot": {"held": None, "holding": False}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", help="what the robot should do, in any language")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--escalate", default=ESCALATE, help="model for the second try when the first plan is invalid ('' = none)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, move nothing")
    ap.add_argument("--scene", default="", metavar="FILE|default", help="plan on a scene from a file (rbyctl.py scene > f.json) or the start row, not the live one")
    ap.add_argument("--replan", type=int, default=1, help="how often to plan again after a failed step (default 1)")
    ap.add_argument("--no-exec-log", action="store_true", help="do not print the executor's log lines (the control centre shows them itself)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the model's raw answers")
    args = ap.parse_args(argv)
    import rbyctl
    if args.scene == "default":
        scene_source = default_scene
    elif args.scene:
        scene_source = lambda: json.load(open(args.scene))  # noqa: E731
    else:
        scene_source = rbyctl.get_scene
    models = (args.model, args.escalate) if args.escalate else (args.model,)
    if not args.dry_run and args.scene:
        ap.error("--scene is for --dry-run only: a recorded scene cannot be acted on")
    def run_task(cmd):
        if cmd["cmd"] == "improvise":
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
            import improvise
            return improvise.run(cmd["request"], improvise.MODEL, offline=False, verbose=False), "improvise failed"
        r = rbyctl.ask(cmd, 600.0, log=(lambda text: None) if args.no_exec_log else None)   # prints the executor's log lines while it works
        if r is None:
            return 2, "the executor does not answer (is run_gui.sh running?)"
        return (0 if r.get("type") == "result" and r.get("ok") else 1), r.get("text", "")
    return run(args.command, scene_source, models=models, dry_run=args.dry_run, replan=args.replan, run_task=run_task,
               verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
