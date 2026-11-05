"""Offline checks of the language planner (no model, no simulator): the context the model sees, parsing and validating its answers, the retry
with the reasons, the translation into executor commands, and the run loop with a fake model and a fake robot.
  /usr/bin/python3 test_planner.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planner as P  # noqa: E402

C = P.C
ROW = P.default_scene()


def scene(boxes=None, held=None):
    s = json.loads(json.dumps(ROW))
    s["boxes"].update(boxes or {})
    s["robot"] = {"held": held, "holding": bool(held)}
    return s


def test_context():
    ctx = P.build_context(ROW)
    assert ctx["suggested_stack_order"][0] == "green" and ctx["unseen"] == [] and ctx["holding"] is None   # the middle box is the easiest base
    assert all(v is None for m in ctx["possible_moves"].values() for v in m.values())                      # the default table has no free spot
    pulled = P.build_context(ROW)  # a box taken away leaves its own spot: moving another box there is possible
    s = scene()
    del s["boxes"]["yellow"]
    ctx = P.build_context(s)
    assert ctx["unseen"] == ["yellow"] and "yellow" not in ctx["possible_moves"]
    assert len(json.dumps(P.build_context(ROW))) < 1200               # small: it is paid for with every call
    del pulled


def test_parse():
    assert P.parse_plan('{"say": "ok", "steps": []}') == {"say": "ok", "steps": []}
    plan = P.parse_plan('Sure!\n```json\n{"steps": [{"skill": "home"}]}\n```')   # a fence and a sentence around it
    assert plan["steps"] == [{"skill": "home"}] and plan["say"] == ""
    for bad in ("no json here", '{"steps": 3}', '{"steps": [}'):
        try:
            P.parse_plan(bad)
            raise AssertionError(bad)
        except P.PlanError:
            pass


def test_validate():
    ok = {"steps": [{"skill": "stack", "order": ["green", "blue", "yellow"]}]}
    assert P.validate(ok, ROW) == []
    stacked = scene({"green": {"pos": [0.62, -0.34, 0.84], "on": "blue"}, "yellow": {"pos": [0.62, -0.34, 0.92], "on": "green"}})
    stacked["boxes"]["blue"]["pos"] = [0.62, -0.34, 0.76]
    assert P.validate({"steps": [{"skill": "unstack"}]}, stacked) == []
    assert any("nothing is stacked" in e for e in P.validate({"steps": [{"skill": "unstack"}]}, ROW))
    assert any("holds the" in e for e in P.validate({"steps": [{"skill": "unstack"}]}, {**stacked, "robot": {"held": "blue", "holding": True}}))
    assert P.to_command({"skill": "unstack"}, stacked) == {"cmd": "unstack"}
    assert P.validate({"steps": [{"skill": "improvise", "request": "손 흔들어"}]}, ROW) == []
    assert any("request must be" in e for e in P.validate({"steps": [{"skill": "improvise", "request": " "}]}, ROW))
    assert any("holds the" in e for e in P.validate({"steps": [{"skill": "improvise", "request": "wave"}]}, scene(held="blue")))
    assert P.to_command({"skill": "improvise", "request": "wave"}, ROW) == {"cmd": "improvise", "request": "wave"}
    # drive: numbers in range, at least one of forward / turn / home, nothing else
    assert P.validate({"steps": [{"skill": "drive", "forward_m": -0.2, "turn_deg": 30}, {"skill": "drive", "home": True}]}, ROW) == []
    assert any("needs forward_m" in e for e in P.validate({"steps": [{"skill": "drive"}]}, ROW))
    assert any("at most 0.3 m" in e for e in P.validate({"steps": [{"skill": "drive", "forward_m": 0.5}]}, ROW))
    assert any("at most 0.3 m" in e for e in P.validate({"steps": [{"skill": "drive", "turn_deg": 120}]}, ROW))
    assert any("takes" in e or "extra" in e for e in P.validate({"steps": [{"skill": "drive", "sideways_m": 0.1}]}, ROW))
    assert P.to_command({"skill": "drive", "turn_deg": -30}, ROW) == {"cmd": "drive", "forward": 0.0, "turn_deg": -30, "home": False}
    bad = {"steps": [{"skill": "fly"}, {"skill": "pick"}, {"skill": "pick", "color": "red"}, {"skill": "stack", "order": ["green"]},
                     {"skill": "move", "color": "blue", "to": "up"}, {"skill": "place", "x": 0.3, "y": 0.0}, {"skill": "home", "x": 1}]}
    errors = P.validate(bad, ROW)
    assert len(errors) == 7 and "unknown skill 'fly'" in errors[0] and "missing ['color']" in errors[1] and "red box is not visible" in errors[2], errors
    # the hand and the first step
    assert any("holds the blue box" in e for e in P.validate({"steps": [{"skill": "pick", "color": "green"}]}, scene(held="blue")))
    assert P.validate({"steps": [{"skill": "place_on", "base": "green"}]}, scene(held="blue")) == []
    assert any("hand is empty" in e for e in P.validate({"steps": [{"skill": "place", "x": 0.62, "y": -0.3}]}, ROW))
    # a move to the right on the default table: no free spot, or already there
    assert any("no free reachable spot" in e for e in P.validate({"steps": [{"skill": "move", "color": "blue", "to": "right"}]}, ROW))
    s = scene()
    del s["boxes"]["yellow"]
    real = P.resolve_spot
    P.resolve_spot = lambda color, to, boxes, base=None: [0.55, -0.30]                  # a table pulled closer would have room: pretend
    try:
        assert P.validate({"steps": [{"skill": "move", "color": "blue", "to": "left"}]}, s) == []
    finally:
        P.resolve_spot = real
    assert any("too close" in e for e in P.validate({"steps": [{"skill": "pick", "color": "blue"}]}, scene({"yellow": {"pos": [0.62, -0.30, 0.76], "on": None}})))


def test_commands():
    s = scene()
    del s["boxes"]["yellow"]
    assert P.to_command({"skill": "stack", "order": ["a", "b"]}, s) == {"cmd": "stack", "order": ["a", "b"]}
    real = P.resolve_spot
    P.resolve_spot = lambda color, to, boxes, base=None: [0.55, -0.30]
    try:
        cmd = P.to_command({"skill": "move", "color": "blue", "to": "left"}, s)
    finally:
        P.resolve_spot = real
    assert cmd == {"cmd": "pick_place", "color": "blue", "x": 0.55, "y": -0.30, "top": C.TABLE_TOP}
    assert P.to_command({"skill": "move", "color": "blue", "to": [0.62, -0.3]}, s)["x"] == 0.62
    try:
        P.to_command({"skill": "move", "color": "blue", "to": "right"}, ROW)
        raise AssertionError("expected a PlanError")
    except P.PlanError:
        pass


class FakeModel:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, system, user, model):
        self.calls.append((model, user))
        return self.answers.pop(0), 0.001, {}


def test_make_plan():
    said = []
    good = '{"say": "쌓을게", "steps": [{"skill": "stack", "order": ["green", "blue", "yellow"]}]}'
    llm = FakeModel(good)
    plan, calls = P.make_plan("상자 3개 쌓아", ROW, llm, ("haiku", "sonnet"), said.append)
    assert plan["steps"][0]["skill"] == "stack" and len(calls) == 1 and "상자 3개 쌓아" in llm.calls[0][1] and said == []
    # the first answer is invalid: it goes back with the reasons, to the next model
    llm = FakeModel('{"steps": [{"skill": "pick", "color": "red"}]}', good)
    plan, calls = P.make_plan("stack", ROW, llm, ("haiku", "sonnet"), said.append)
    assert [m for m, _ in llm.calls] == ["haiku", "sonnet"] and "red box is not visible" in llm.calls[1][1] and "rejected" in llm.calls[1][1]
    # garbage twice: no plan, with the reasons
    llm = FakeModel("nope", "still nope")
    try:
        P.make_plan("stack", ROW, llm, ("haiku", "sonnet"), said.append)
        raise AssertionError("expected a PlanError")
    except P.PlanError as exc:
        assert "no valid plan" in str(exc)


def test_run():
    out, ran = [], []
    good = '{"say": "쌓을게", "steps": [{"skill": "stack", "order": ["green", "blue", "yellow"]}]}'

    def robot(cmd):
        ran.append(cmd)
        return 0, "done"
    rc = P.run("쌓아", lambda: ROW, llm=FakeModel(good), models=("m",), run_task=robot, say=out.append)
    assert rc == 0 and ran == [{"cmd": "stack", "order": ["green", "blue", "yellow"]}] and any(l.startswith("robot: 쌓을게") for l in out)
    assert any("OK" in l for l in out)
    # dry run: planned, not run
    ran.clear()
    assert P.run("쌓아", lambda: ROW, llm=FakeModel(good), models=("m",), dry_run=True, run_task=robot, say=out.append) == 0 and ran == []
    # nothing to do / impossible: the model says why, no steps, exit 0
    ran.clear()
    refuse = '{"say": "파란 상자는 이미 맨 오른쪽이야", "steps": []}'
    assert P.run("파란 걸 오른쪽에", lambda: ROW, llm=FakeModel(refuse), models=("m",), run_task=robot, say=out.append) == 0 and ran == []
    # a failing step: one new plan from the fresh scene, and the failure goes into the request
    llm = FakeModel(good, good)
    results = iter([(1, "grasp missed"), (0, "done")])
    out.clear()
    rc = P.run("쌓아", lambda: ROW, llm=llm, models=("m",), run_task=lambda cmd: next(results), say=out.append)
    assert rc == 0 and len(llm.calls) == 2 and "previous attempt failed" in llm.calls[1][1] and "grasp missed" in llm.calls[1][1] and any("planning again" in l for l in out)
    # an abort is not retried either
    llm = FakeModel(good)
    assert P.run("쌓아", lambda: ROW, llm=llm, models=("m",), run_task=lambda cmd: (1, "aborted: stack"), say=out.append) == 1 and len(llm.calls) == 1
    # twice failing: gives up (no endless retries on hardware); an abort / dead executor is not retried at all
    results = iter([(1, "grasp missed"), (1, "grasp missed")])
    assert P.run("쌓아", lambda: ROW, llm=FakeModel(good, good), models=("m",), run_task=lambda cmd: next(results), say=out.append) == 1
    llm = FakeModel(good)
    assert P.run("쌓아", lambda: ROW, llm=llm, models=("m",), run_task=lambda cmd: (2, "the executor does not answer"), say=out.append) == 1 and len(llm.calls) == 1


if __name__ == "__main__":
    tests = [test_context, test_parse, test_validate, test_commands, test_make_plan, test_run]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} planner checks passed")
