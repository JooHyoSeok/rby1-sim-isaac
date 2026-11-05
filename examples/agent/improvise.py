#!/usr/bin/env python3
"""A language model that designs a robot motion with tools, and runs it: for requests no fixed skill covers ("손 흔들어", "고개를 오른쪽으로 돌려").

  improvise.py "손 흔들어"                    design and play it on the robot (examples/gui/run_gui.sh must be running)
  improvise.py "wave your hand" --offline    the same without a robot: the tools pretend the arm hangs at zero joints and send nothing
  improvise.py "..." --model claude-haiku-5-5 -v

The model (default claude-sonnet-5-5) gets one tool: shell commands of robot_tools.py (state, limits, fk, check, move, save, play, skills), through
`claude -p --tools Bash --allowedTools "Bash(<robot_tools.py>:*)"`: nothing else on the machine. It looks at the robot, uses forward kinematics to learn
what joints do, designs waypoints, checks them against the safety rules (code, see safety.py), plays them, and may save a motion that worked.
The executor checks again before anything moves. Each tool call is a model turn: expect 5 to 15 of them, tens of seconds, and many times the tokens
of examples/llm/planner.py. Uses your Claude Code login (subscription usage, no API key).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "robot_tools.py")
PYTHON = "/usr/bin/python3"
MODEL = "claude-sonnet-5-5"

SYSTEM = f"""You move a humanoid robot (RB-Y1) by designing joint trajectories for its RIGHT ARM, HEAD and TORSO YAW. You can run shell commands, only these:
  {PYTHON} {TOOLS} state                      joint angles [deg], hand position [m], what the hand holds, the boxes
  {PYTHON} {TOOLS} limits                     the joints you may move, their ranges, speed limits and what each does
  {PYTHON} {TOOLS} fk joint=deg joint=deg ...  where the arm, hand and head would be with these joint values (changes from the current pose)
  {PYTHON} {TOOLS} check --json '<trajectory>'  safety check, nothing moves
  {PYTHON} {TOOLS} move --json '<trajectory>'   check, then the robot plays it
  {PYTHON} {TOOLS} skills | play NAME | save NAME   motions saved earlier / save the last one that ran
Trajectory: {{"unit": "deg", "waypoints": [{{"t": 2.0, "right_arm_1": -90}}, {{"t": 1.2, "right_arm_2": -60, "head_0": 20}}, ...]}}. "t" = seconds from the previous
waypoint (the robot stops briefly at each waypoint), a joint a waypoint does not mention keeps its value. Frame: base, x forward, y left, z up, metres;
the robot's RIGHT is -y.

Work like this: (1) `skills`: if a saved motion fits the request, `play` it. (2) `state` and `limits`. (3) Do not guess what a joint does: use `fk` to
try values and read where the hand, wrist, elbow and head end up; probe, then decide the poses. (4) Write the trajectory and `check` it; if there are
violations, fix exactly those (they contain numbers). (5) `move` it. (6) Answer in ONE short sentence in the user's language saying what you did. If the
motion is a reusable gesture and it ran, `save` it first with a short English name (wave, nod, ...).
Rules: only these commands; small simple motions; stay inside the limits `limits` reports; never try to get around a violation, change the plan.
If the request needs something you cannot do (walking, turning the whole body, picking things up, the left arm), say so in one sentence and do not move.
Do not narrate between commands."""


def run(request, model, offline, verbose, timeout=600):
    cmd = ["claude", "-p", "--model", model, "--tools", "Bash", "--allowedTools", f"Bash({PYTHON} {TOOLS}:*)", "--system-prompt", SYSTEM,
           "--no-session-persistence", "--disable-slash-commands", "--setting-sources", "", "--strict-mcp-config", "--output-format", "stream-json",
           "--verbose"]   # --strict-mcp-config: without it the MCP servers of the user's plugins added ~270k tokens to every turn after the first tool call
    env = {**os.environ, **({"ROBOT_TOOLS_OFFLINE": "1"} if offline else {})}
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tempfile.gettempdir(), env=env)
    proc.stdin.write(request)
    proc.stdin.close()
    result, calls = None, 0
    try:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "assistant":
                for block in msg["message"]["content"]:
                    if block.get("type") == "tool_use":
                        calls += 1
                        text = block["input"].get("command", "").replace(f"{PYTHON} {TOOLS}", "tools").replace("\n", " ")
                        print(f"tool {calls}: {text[:600]}", flush=True)
                    elif block.get("type") == "text" and verbose and block["text"].strip():
                        print(f"model: {block['text'].strip()}", flush=True)
            elif kind == "user" and verbose:
                for block in msg["message"]["content"] if isinstance(msg["message"]["content"], list) else []:
                    if block.get("type") == "tool_result":
                        content = block.get("content")
                        text = content if isinstance(content, str) else " ".join(c.get("text", "") for c in content or [])
                        print(f"  -> {text.strip()[:400]}", flush=True)
            elif kind == "result":
                result = msg
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("FAILED: the model did not finish in time")
        return 1
    if result is None:
        print(f"FAILED: no answer from claude -p ({proc.stderr.read().strip()[:300]})")
        return 1
    print(f"robot: {result.get('result', '').strip()}")
    usage = result.get("usage", {})
    print(f"({model}, {calls} tool calls, {result.get('num_turns')} turns, {result.get('duration_ms', 0) / 1000:.0f} s, list price ${result.get('total_cost_usd', 0):.3f}; "
          f"tokens in {usage.get('input_tokens', 0)} + cache write {usage.get('cache_creation_input_tokens', 0)} + cache read {usage.get('cache_read_input_tokens', 0)}, "
          f"out {usage.get('output_tokens', 0)})")
    return 1 if result.get("is_error") else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("request")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--offline", action="store_true", help="no robot: check and print, send nothing")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the model's text and the tools' answers")
    args = ap.parse_args(argv)
    return run(args.request, args.model, args.offline, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
