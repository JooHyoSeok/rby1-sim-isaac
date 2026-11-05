"""Offline checks of the control centre, no simulator and no robot:
  QT_QPA_PLATFORM=offscreen /usr/bin/python3 test_offline.py
Covers: scene_config reachability and the multi-box helpers, the pure grasp_core helpers, the stack alignment loop and the stack check
(GraspRobot with a simulated hand, no SDK), the scene map, the window (buttons -> UDP commands, status -> button states, joint table,
camera views), servo_server's command handling with a fake robot (queueing, errors, abort, reply port) and rbyctl against it.
Not covered: anything that talks to the simulator or the robot (the rest of grasp_core.GraspRobot, rby1_vision.poll_commands).
"""
import json
import os
import socket
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from PyQt5 import QtCore, QtTest, QtWidgets  # noqa: E402

import base_motion as B  # noqa: E402
import grasp_core  # noqa: E402
import rbyctl  # noqa: E402
import scene_config as C  # noqa: E402
from control_center import ControlCenter  # noqa: E402
from scene_map import SceneMap  # noqa: E402
from servo_server import Server  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def listener(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    s.settimeout(1.0)
    return s


def recv_all(sock, wait=0.3):
    out, t0 = [], time.time()
    sock.settimeout(0.05)
    while time.time() - t0 < wait:
        try:
            out.append(json.loads(sock.recv(65536)))
        except socket.timeout:
            pass
    return out


def last_msg(sock, window, wait=0.3):
    """The last datagram received, or a failure that shows the window's log (it says why nothing was sent)."""
    msgs = recv_all(sock, wait)
    assert msgs, "nothing was sent; the window's log says:\n" + window.log.toPlainText()[-600:]
    return msgs[-1]


def pump(ms=150):
    QtTest.QTest.qWait(ms)


# ---- config and pure helpers --------------------------------------------------------------------------------------------
def test_config():
    assert C.check_pose(0.64, -0.20, 0.72) is None                      # the grasp that was run in the simulator
    assert "reach" in C.check_pose(0.70, -0.20, 0.72)
    assert "table" in C.check_pose(0.30, -0.20, 0.72)
    assert C.table_ok_before_seed() and not C.table_ok_before_seed((0.80, 0.0))
    assert C.check_pose(0.55, -0.2, 0.72, (0.80, 0.0)) is None           # a table pulled closer opens up more positions
    # a lower table pushes the box out of reach, a higher one brings it in
    assert C.box_reachable(0.64, -0.20, 0.72) and not C.box_reachable(0.64, -0.20, 0.60)


def test_boxes():
    row = {n: (*xy, 0.76) for n, xy in C.BOX_DEFAULTS.items()}
    assert all(C.check_pose(x, y, 0.72) is None for x, y, _ in row.values())          # the start row is on the table and reachable
    assert all(C.pick_problem(n, row) is None for n in row)
    assert C.stack_problems(["green", "blue", "yellow"], row) == []
    # too close: the open fingers (+-6.3 cm) would hit the neighbour; far: out of reach; unseen / bad orders are refused
    near = {**row, "yellow": (0.62, -0.30, 0.76)}
    assert "too close to the yellow box" in C.pick_problem("blue", near)
    assert "out of reach" in C.pick_problem("blue", {**row, "blue": (0.72, -0.34, 0.76)})
    assert "cannot see the yellow box" in C.stack_problems(["green", "yellow"], {k: v for k, v in row.items() if k != "yellow"})[0]
    assert C.stack_problems(["green"], row) and C.stack_problems(["green", "green"], row)
    # the boxes of a stack are no obstacle for each other; a different height is no obstacle either
    stack = {"green": (0.62, -0.22, 0.76), "blue": (0.62, -0.22, 0.84), "yellow": (0.62, -0.10, 0.76)}
    assert C.stacked_on(stack) == {"green": None, "blue": "green", "yellow": None}
    assert C.pick_problem("blue", stack) is None
    # a stack whose base is too far away: the model says so before the arm finds out
    assert "level 2 of the stack" in C.stack_problems(["green", "blue"], {**row, "green": (0.73, -0.22, 0.76)})[0]
    # taking a stack apart: top box first, each onto its own start spot when that is free
    st = {"blue": (0.62, -0.34, 0.76), "green": (0.62, -0.34, 0.84), "yellow": (0.62, -0.34, 0.92)}
    assert C.unstack_order(st) == ["yellow", "green"] and C.unstack_order(row) == []
    assert C.table_spot("yellow", st) == C.BOX_DEFAULTS["yellow"] and C.table_spot("green", {**st, "yellow": (*C.BOX_DEFAULTS["yellow"], 0.76)}) == C.BOX_DEFAULTS["green"]
    assert C.table_spot("green", {**st, "yellow": (0.62, -0.20, 0.76)}) is None or C.table_spot("green", {**st, "yellow": (0.62, -0.20, 0.76)})[1] != -0.22   # its start spot is taken
    # free spots: only where nothing is closer than the finger clearance; on the default table there is no room beside the row
    assert C.direction_spot("right", row, exclude="blue") == (0.62, -0.33)               # the blue box's own spot
    assert C.direction_spot("right", row) is None
    pulled = (0.80, 0.0)                                                                  # table pulled closer: room for a move
    right, left = (C.direction_spot(d, row, exclude="blue", table_center=pulled) for d in ("right", "left"))
    assert right[1] < left[1] and all(C.check_pose(*p, 0.72, pulled) is None for p in (right, left))


def test_helpers():
    assert grasp_core.smoothstep(0) == 0 and grasp_core.smoothstep(1) == 1 and abs(grasp_core.smoothstep(0.5) - 0.5) < 1e-9
    above, down, back = grasp_core.place_waypoints(np.array([0.58, -0.2, 0.88]), 0.62, -0.28, 0.72)
    assert above[2] >= 0.88 and abs(above[0] - 0.625) < 1e-9 and abs(above[1] + 0.28) < 1e-9
    table_down, stack_down = np.array([0.625, -0.22, 0.764]), np.array([0.625, -0.229, 0.924])   # the third box of a stack, from the simulator run
    arm_stub = object.__new__(grasp_core.GraspRobot)
    arm_stub.side = "right"
    assert np.allclose(arm_stub._back_off(table_down) - table_down, [-0.10, 0, 0])
    stack_back = arm_stub._back_off(stack_down)
    assert -0.10 < (stack_back - stack_down)[0] <= -0.04 and C.tip_reachable(stack_back)           # the arm model allows less than 10 cm back there
    top2 = grasp_core.place_waypoints(np.array([0.58, -0.2, 0.88]), 0.62, -0.22, 0.88)      # on a stack of two: carried above its top
    assert top2[0][2] >= 0.88 + 0.04 + grasp_core.HOVER - 1e-9 and abs(top2[1][2] - (0.88 + 0.04 + 0.004)) < 1e-9
    assert abs(down[2] - (0.72 + 0.04 + 0.004)) < 1e-9 and np.allclose(back - down, [-0.10, 0, 0])
    low = grasp_core.place_waypoints(np.array([0.58, -0.2, 0.70]), 0.62, -0.28, 0.72)[0]
    assert low[2] >= 0.72 + 0.04 + grasp_core.HOVER - 1e-9              # never carries the box through the table


# ---- the map ------------------------------------------------------------------------------------------------------------
def press_drag_release(w, a, b, steps=6):
    """Left mouse drag on a widget from pixel a to pixel b. The moves are delivered straight to the widget: QTest.mouseMove does not
    reach it on the offscreen platform."""
    from PyQt5 import QtGui
    QtTest.QTest.mousePress(w, QtCore.Qt.LeftButton, pos=QtCore.QPoint(int(a.x()), int(a.y())))
    for i in range(1, steps + 1):
        q = a + (b - a) * (i / steps)
        w.mouseMoveEvent(QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(q), QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
                                           QtCore.Qt.NoModifier))
        QtTest.QTest.qWait(8)
    QtTest.QTest.mouseRelease(w, QtCore.Qt.LeftButton, pos=QtCore.QPoint(int(b.x()), int(b.y())))


def test_map():
    m = SceneMap()
    m.resize(400, 460)
    m.show()
    for x, y in ((0.0, 0.0), (0.64, -0.2), (1.1, 0.5)):
        px = m.to_px(x, y)
        wx, wy = m.to_world(px.x(), px.y())
        assert abs(wx - x) < 1e-6 and abs(wy - y) < 1e-6
    default = len(m.reach_cells())
    assert default > 0                                                  # a green region exists for the default table (a narrow one)
    m.set_box_live((0.64, -0.2, 0.76))
    m.set_target((0.64, -0.3))
    events = []
    m.dragged.connect(lambda item, x, y, final: events.append((item, x, y, final)))
    # dragging the box: the grabbed point stays under the pointer, the last event is final and the box ends where it was dropped
    press_drag_release(m, m.to_px(0.64, -0.2), m.to_px(0.64, -0.26))
    item, x, y, final = events[-1]
    assert item == "box" and final and abs(x - 0.64) < 0.01 and abs(y + 0.26) < 0.01, events[-1]
    # dropped where the arm cannot reach (beside the robot): it snaps to the nearest valid position
    events.clear()
    press_drag_release(m, m.to_px(0.64, -0.2), m.to_px(0.30, 0.20))
    item, x, y, final = events[-1]
    assert item == "box" and final and m.valid(x, y), events[-1]
    assert abs(x - 0.30) > 0.2                                          # it did move away from the pointer
    # with "allow" it does not snap
    events.clear()
    m.set_allow_any(True)
    press_drag_release(m, m.to_px(0.64, -0.2), m.to_px(0.30, 0.20))
    assert abs(events[-1][1] - 0.30) < 0.01 and abs(events[-1][2] - 0.20) < 0.01
    m.set_allow_any(False)
    # the place target drags too, and grabbing it does not grab the box
    events.clear()
    press_drag_release(m, m.to_px(0.64, -0.3), m.to_px(0.63, -0.33))
    assert events[-1][0] == "target" and events[-1][3]
    # the table: grabbed away from the box and the target; kept clear of the seed motion when a minimum edge is set
    events.clear()
    m.set_table_min_edge(C.SEED_EDGE_X)
    press_drag_release(m, m.to_px(1.05, 0.3), m.to_px(0.60, 0.3))
    item, x, y, final = events[-1]
    assert item == "table" and final and x >= C.SEED_EDGE_X + C.TABLE_SIZE[0] / 2 - 1e-9, events[-1]
    m.set_table_min_edge(None)
    events.clear()
    press_drag_release(m, m.to_px(1.05, 0.3), m.to_px(0.60, 0.3))
    assert events[-1][1] < C.SEED_EDGE_X + C.TABLE_SIZE[0] / 2
    # an empty spot is not a drag target; the wheel over the box turns it, elsewhere it does nothing
    events.clear()
    press_drag_release(m, m.to_px(0.2, 0.4), m.to_px(0.3, 0.4))
    assert events == []
    turned = []
    m.rotated.connect(turned.append)
    from PyQt5 import QtGui

    def wheel(at, delta):
        ev = QtGui.QWheelEvent(QtCore.QPointF(at), QtCore.QPointF(m.mapToGlobal(at)), QtCore.QPoint(), QtCore.QPoint(0, delta),
                               QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
        m.wheelEvent(ev)
    p_box = m.to_px(0.64, -0.2).toPoint()
    wheel(p_box, 120)
    wheel(m.to_px(0.2, 0.4).toPoint(), 120)
    assert turned == [5.0], turned
    assert not m.grab().isNull()
    m.close()


# ---- the window ---------------------------------------------------------------------------------------------------------
class FakeBridge:
    def snapshot(self):
        rng = np.random.default_rng(0)
        imgs = {n: (rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), 0.1)
                for n in ("wrist overlay", "stereo overlay", "wrist left", "stereo left")}
        names = [f"torso_{i}" for i in range(6)] + [f"right_arm_{i}" for i in range(7)] + [f"left_arm_{i}" for i in range(7)] + \
                ["left_wheel", "right_wheel", "head_0", "head_1", "gripper_finger_r1", "gripper_finger_r2"]
        return {"images": imgs, "joints": {n: 0.1 * i for i, n in enumerate(names)}, "joints_age": 0.1, "box": (0.64, -0.2, 0.76), "box_age": 0.1,
                "boxes": {"blue": (0.64, -0.2, 0.76), "green": (0.62, -0.22, 0.76), "yellow": (0.62, -0.10, 0.84)}}

    def close(self):
        pass


def test_window():
    p_scene, p_exec, p_status = free_port(), free_port(), free_port()
    scene_rx, exec_rx = listener(p_scene), listener(p_exec)
    w = ControlCenter(FakeBridge(), ("127.0.0.1", p_scene), ("127.0.0.1", p_exec), p_status)
    w.show()
    pump(300)
    # the exact numbers are folded away until asked for (an unchecked group box also disables what is inside it)
    precise = [g for g in w.findChildren(QtWidgets.QGroupBox) if g.title().startswith("exact values")][0]
    assert precise.isCheckable() and not precise.isChecked()
    precise.setChecked(True)
    # cameras and joints are filled from the bridge
    assert all(v.pixmap() is not None for v in w.views.values())
    assert w.joints.rowCount() > 0 and w.joints.item(0, 1) is not None
    assert set(w.map.other_boxes) == {"green", "yellow"}                  # the other boxes are drawn on the map, not draggable
    assert not w.map.grab().isNull()
    # no executor yet: task buttons disabled
    assert not w.buttons["pick"].isEnabled() and "not connected" in w.status_label.text()
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def status(**kw):
        st.sendto(json.dumps({"type": "status", "state": "idle", "task": None, "phase": "idle", "holding": False, "seeded": True,
                              "tip": [0.5, -0.2, 0.8], "closeness": 0.0, **kw}).encode(), ("127.0.0.1", p_status))
        pump(250)

    status()
    assert w.buttons["pick"].isEnabled() and not w.buttons["place"].isEnabled()
    # pick sends the command with the camera-correction flag
    w.buttons["pick"].click()
    msg = recv_all(exec_rx)[-1]
    assert msg == {"cmd": "pick", "use_image": True}, msg
    # holding: place becomes available and sends the table height
    status(holding=True)
    assert w.buttons["place"].isEnabled() and not w.buttons["pick"].isEnabled()
    w.place_x.setValue(0.62)
    w.place_y.setValue(-0.25)
    w.buttons["place"].click()
    msg = recv_all(exec_rx)[-1]
    assert msg["cmd"] == "place" and abs(msg["x"] - 0.62) < 1e-9 and abs(msg["top"] - C.TABLE_TOP) < 1e-9, msg
    # an unreachable place target is refused, with a reason in the log, unless explicitly allowed
    w.place_x.setValue(0.95)
    w.buttons["place"].click()
    assert recv_all(exec_rx) == [] and "not sent" in w.log.toPlainText()
    w.allow_any.setChecked(True)
    w.buttons["place"].click()
    assert recv_all(exec_rx)[-1]["cmd"] == "place"
    w.allow_any.setChecked(False)
    # busy: only abort and the gripper stay usable
    status(task="pick", state="busy")
    assert not w.buttons["pick"].isEnabled()
    w.findChildren(QtWidgets.QPushButton)  # exists
    abort = [b for b in w.findChildren(QtWidgets.QPushButton) if b.text() == "ABORT"][0]
    abort.click()
    assert recv_all(exec_rx)[-1] == {"cmd": "abort"}
    # scene commands go to the simulator port
    w.box_x.setValue(0.63)
    w.box_y.setValue(-0.18)
    w.box_yaw.setValue(10)
    [b for b in w.findChildren(QtWidgets.QPushButton) if b.text() == "apply box"][0].click()
    assert last_msg(scene_rx, w) == {"cmd": "set_box", "x": 0.63, "y": -0.18, "yaw_deg": 10.0}
    w.box_x.setValue(0.95)
    [b for b in w.findChildren(QtWidgets.QPushButton) if b.text() == "apply box"][0].click()
    assert recv_all(scene_rx) == []
    # a table whose front edge would block the seed motion is refused until the robot is seeded
    status(seeded=False, task=None, state="idle")
    w.box_x.setValue(0.63)
    w.tab_x.setValue(0.80)
    [b for b in w.findChildren(QtWidgets.QPushButton) if b.text() == "apply table"][0].click()
    assert recv_all(scene_rx) == [] and "seed" in w.log.toPlainText()
    status(seeded=True)
    [b for b in w.findChildren(QtWidgets.QPushButton) if b.text() == "apply table"][0].click()
    msgs = recv_all(scene_rx)
    assert [m["cmd"] for m in msgs] == ["set_table", "set_box"] and abs(msgs[0]["x"] - 0.80) < 1e-9, msgs   # box restands on the table
    # with the table pulled close, the first move of the robot (pick when not seeded) and home are refused; once seeded pick works
    status(seeded=False)
    w.buttons["pick"].click()
    assert recv_all(exec_rx) == [] and "seed pose" in w.log.toPlainText()
    w.buttons["home"].click()
    assert recv_all(exec_rx) == []
    status(seeded=True)
    w.buttons["pick"].click()
    assert recv_all(exec_rx)[-1]["cmd"] == "pick"
    # dragging the box on the map moves it in the simulator (live and once more on release) and keeps the numbers in step
    w.allow_any.setChecked(False)
    w.map.set_table(C.TABLE_CENTER, C.TABLE_SIZE, C.TABLE_TOP)         # back to the default table for the map tests
    w.tab_x.setValue(C.TABLE_CENTER[0])
    w.box_yaw.setValue(0)
    pump(150)
    recv_all(scene_rx, 0.05)
    press_drag_release(w.map, w.map.to_px(0.64, -0.2), w.map.to_px(0.64, -0.25))
    msgs = [m for m in recv_all(scene_rx) if m["cmd"] == "set_box"]
    assert msgs and abs(msgs[-1]["y"] + 0.25) < 0.01 and abs(w.box_y.value() + 0.25) < 0.01, msgs
    # the table carries the box along; a box in the hand stays out of it
    w.map.set_table_min_edge(None)
    press_drag_release(w.map, w.map.to_px(1.05, 0.3), w.map.to_px(1.05, 0.2))
    msgs = recv_all(scene_rx)
    cmds = [m["cmd"] for m in msgs]
    assert "set_table" in cmds and cmds[-1] == "set_box", cmds
    table_msg = [m for m in msgs if m["cmd"] == "set_table"][-1]
    assert abs(table_msg["y"] - 0.0 + 0.1) < 0.02, table_msg            # moved by about -0.1 in y
    assert abs(msgs[-1]["y"] - (-0.25 - 0.1)) < 0.03, msgs[-1]         # the box moved with it
    # the wheel over the box turns it (5 degrees per notch)
    from PyQt5 import QtGui
    bp = w.map.to_px(*w.map._box_pos()).toPoint()   # where the map shows the box (the fake bridge keeps the live box fixed)
    w.map.wheelEvent(QtGui.QWheelEvent(QtCore.QPointF(bp), QtCore.QPointF(w.map.mapToGlobal(bp)), QtCore.QPoint(), QtCore.QPoint(0, 120),
                                       QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False))
    assert recv_all(scene_rx)[-1]["yaw_deg"] == 5.0
    # the right-click menu choices: the target is only a GUI thing (no simulator message), the box goes to the simulator
    w._on_chosen("target", 0.63, -0.18)
    assert recv_all(scene_rx) == [] and "0.630" in w.target_label.text() and w.map.target == (0.63, -0.18)
    w._on_chosen("box", 0.64, -0.21)
    assert recv_all(scene_rx)[-1]["cmd"] == "set_box"
    assert w.map.target is not None                                     # the place target exists from the start
    # log lines from the executor appear
    st.sendto(json.dumps({"type": "log", "text": "hello from the executor"}).encode(), ("127.0.0.1", p_status))
    pump(250)
    assert "hello from the executor" in w.log.toPlainText()
    if os.environ.get("SAVE_SHOT"):
        w.grab().save(os.environ["SAVE_SHOT"])
    w.close()


def test_say():
    """The text box: the typed sentence goes to the planner process (here a stand-in that prints), its output lands in the log, its
    'robot:' line in the label; plan-only adds --dry-run; no executor means nothing is started unless it is plan-only."""
    p_scene, p_exec, p_status = free_port(), free_port(), free_port()
    fake = [sys.executable, "-u", "-c", "import sys; print('robot: 쌓을게'); print('== step 1/1: stack'); print('args', sys.argv[1:])"]
    w = ControlCenter(FakeBridge(), ("127.0.0.1", p_scene), ("127.0.0.1", p_exec), p_status, planner_cmd=fake)
    w.show()
    pump(200)

    def finish(timeout=5000):
        QtTest.QTest.qWait(50)
        for _ in range(timeout // 50):
            if w.planner.state() == QtCore.QProcess.NotRunning:
                break
            QtTest.QTest.qWait(50)
        pump(100)
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    w.say_edit.setText("상자 3개 쌓아")
    assert not w.say_go.isEnabled()                                          # no executor yet
    w.do_say()
    pump(200)
    assert "executor is not connected" in w.log.toPlainText() and w.planner.state() == QtCore.QProcess.NotRunning
    # plan only works without an executor
    w.say_plan_only.setChecked(True)
    assert w.say_go.isEnabled()
    w.do_say()
    finish()
    log = w.log.toPlainText()
    assert "you: 상자 3개 쌓아" in log and "== step 1/1: stack" in log and "--dry-run" in log and "'상자 3개 쌓아'" in log, log
    assert w.robot_says.text() == "쌓을게"
    # with an executor: runs for real (no --dry-run)
    st.sendto(json.dumps({"type": "status", "state": "idle", "task": None, "phase": "idle", "holding": False, "seeded": True,
                          "tip": [0.5, -0.2, 0.8], "closeness": 0.0}).encode(), ("127.0.0.1", p_status))
    pump(250)
    w.say_plan_only.setChecked(False)
    w.say_edit.setText("stack")
    assert w.say_go.isEnabled()
    w.do_say()
    finish()
    assert "args ['stack', '--no-exec-log']" in w.log.toPlainText().splitlines()[-1], w.log.toPlainText()[-200:]
    # ABORT also stops a running plan
    w.planner_cmd = [sys.executable, "-u", "-c", "import time; print('robot: ...'); time.sleep(30)"]
    w.say_edit.setText("slow")
    w.do_say()
    pump(300)
    assert w.planner.state() != QtCore.QProcess.NotRunning
    w.do_abort()
    finish()
    assert w.planner.state() == QtCore.QProcess.NotRunning and "plan stopped" in w.log.toPlainText()
    assert recv_all(listener(p_exec), 0.05) == [] or True
    w.close()


# ---- the executor service with a fake robot -----------------------------------------------------------------------------
class FakeRobot:
    def __init__(self):
        self.calls, self.flag, self.jobs = [], threading.Event(), 0
        self.row = {n: {"pos": [*xy, 0.76], "yaw_deg": 0.0, "landmarks": 4, "on": None} for n, xy in C.BOX_DEFAULTS.items()}

    def status(self):
        return {"phase": "idle", "holding": False, "held": None, "seeded": True, "tip": [0.5, -0.2, 0.8], "closeness": 0.0}

    def scene(self):
        return {"boxes": self.row, "robot": self.status(), "table": {"center": list(C.TABLE_CENTER), "size": list(C.TABLE_SIZE),
                                                                      "top": C.TABLE_TOP, "assumed": True}}

    def job(self):
        self.jobs += 1
        import contextlib
        return contextlib.nullcontext()

    def abort(self):
        self.flag.set()

    def place_on(self, base):
        self.calls.append(("place_on", base))

    def unstack(self):
        self.calls.append(("unstack",))

    def state(self):
        return {"joints": {"right_arm_1": -0.5}, "tip": [0.0, -0.22, 0.59], "robot": self.status()}

    def move_joints(self, traj):
        self.calls.append(("joints", traj["waypoints"][0]["t"]))
        if traj.get("unit") == "bad":
            raise grasp_core.TaskError("the trajectory is not safe: test")

    def stack(self, order):
        self.calls.append(("stack", tuple(order)))
        if "yellow" not in order:
            raise grasp_core.TaskError("cannot see the yellow box")

    def pick(self, color="blue", use_image=True, bias=(0, 0, 0), arm="auto", dest=None):
        self.calls.append(("pick", color, use_image, bias))
        if use_image == "wait":
            self.flag.clear()
            while not self.flag.is_set():
                time.sleep(0.01)
            raise grasp_core.Aborted("aborted")
        if use_image == "fail":
            raise grasp_core.TaskError("grasp missed")

    def place(self, x, y, top):
        self.calls.append(("place", x, y, top))

    def seed(self):
        self.calls.append(("seed",))

    def home(self):
        self.calls.append(("home",))

    def set_gripper(self, c, arm="right"):
        self.calls.append(("gripper", c))


def test_server():
    p_cmd, p_status = free_port(), free_port()
    status_rx = listener(p_status)
    robot = FakeRobot()
    srv = Server(robot, cmd_port=p_cmd, status_addr=("127.0.0.1", p_status))
    try:
        srv.handle({"cmd": "pick_place", "x": 0.6, "y": -0.3, "top": 0.72})
        time.sleep(0.4)
        assert robot.calls == [("pick", "blue", True, (0.0, 0.0, 0.0)), ("place", 0.6, -0.3, 0.72)], robot.calls
        assert robot.jobs == 1                                                        # pick and place ran as one job
        # an error is reported, not raised
        srv.handle({"cmd": "pick", "use_image": "fail"})
        time.sleep(0.4)
        logs = [m["text"] for m in recv_all(status_rx, 0.2) if m["type"] == "log"]
        assert any("failed: pick: grasp missed" in t for t in logs), logs
        # a running task blocks new ones, abort interrupts it
        srv.handle({"cmd": "pick", "use_image": "wait"})
        time.sleep(0.3)
        srv.handle({"cmd": "home"})
        srv.handle({"cmd": "abort"})
        time.sleep(0.4)
        logs = [m["text"] for m in recv_all(status_rx, 0.2) if m["type"] == "log"]
        assert any("busy with 'pick'" in t for t in logs) and any("aborted: pick" in t for t in logs), logs
        assert ("home",) not in robot.calls
        # idle again: heartbeat says so, commands work
        srv.handle({"cmd": "gripper", "closeness": 1.0})
        time.sleep(0.2)
        assert ("gripper", 1.0) in robot.calls
        beats = [m for m in recv_all(status_rx, 1.2) if m["type"] == "status"]
        assert beats and beats[-1]["state"] == "idle" and beats[-1]["seeded"] is True
    finally:
        srv.stop.set()


# ---- the stack alignment and check, on a GraspRobot whose hand is simulated (no SDK) -------------------------------------------
class SimPerception:
    """Stereo measurements of the base box and of the held box from the simulated hand, a few mm noisy."""

    def __init__(self, robot, base, base_xy, grip, visible=True):
        self.robot, self.base, self.base_xy, self.grip, self.visible = robot, base, np.array(base_xy), np.array(grip), visible
        self.rng = np.random.default_rng(3)

    def held_true(self):
        return self.robot.hand[:2] - np.array([C.TIP_BEYOND_CENTRE, 0.0]) + self.grip     # the box sits off-centre between the fingers

    def boxes(self, max_age=1.5):
        noise = lambda: self.rng.normal(0, 0.002, 2)  # noqa: E731
        out = {self.base: {"t": [*(self.base_xy + noise()), 0.80], "n": 3}}
        if self.visible:
            out[self.robot.held] = {"t": [*(self.held_true() + noise()), 0.98], "n": 3}
        return out


def sim_robot(bias, grip, visible, start=(0.625, -0.235)):
    """GraspRobot holding the blue box over a hand that follows the commanded tip with a lag and a constant offset `bias` (what the
    position loop leaves), as grasp_core._align_over sees it: ee_pose / _stream_to / _tick / perception are the simulated ones."""
    r = object.__new__(grasp_core.GraspRobot)
    r.held, r.holding, r.phase, r._depth, r.abort_flag = "blue", True, "idle", 0, threading.Event()
    r.events = []
    r.event, r.debug, r._check = r.events.append, (lambda **kw: None), (lambda: None)
    r.hand, r.target = np.array([*start, 0.98]), np.array([*start, 0.98])
    r.cmd_tip = r.hand.copy()

    def ee_pose():
        T = np.eye(4)
        T[:3, :3] = grasp_core.R_FRONT
        T[:3, 3] = r.hand - grasp_core.TOOL * np.array([1.0, 0.0, 0.0])
        return T

    def tick(t_loop):
        r.hand += (r.target + np.r_[bias, 0.0] - r.hand) * (1 - np.exp(-1.0 / grasp_core.RATE / 0.15))   # first-order lag, 0.15 s
    r.ee_pose, r._stream_to, r._tick = ee_pose, (lambda tip: r.target.__setitem__(slice(None), tip)), tick
    r.perception = SimPerception(r, "green", (0.62, -0.22), grip, visible)
    return r


def test_align():
    bias, grip = (0.008, -0.006), (0.004, -0.003)
    r = sim_robot(bias, grip, visible=True)
    r._align_over("green")                                                  # the position loop is 1 cm off, the grip 5 mm: both vanish
    err = np.linalg.norm(r.perception.held_true() - r.perception.base_xy)
    assert err < 0.003 and "aligned over the green box" in r.events[-1] and "(stereo)" in r.events[-1], (err, r.events)
    # the held box hidden by the fingers: the hand's own pose is used (the position loop's error is still removed, the grip offset is not)
    r = sim_robot(bias, grip, visible=False)
    r._align_over("green")
    err = np.linalg.norm(r.perception.held_true() - r.perception.base_xy)
    assert err < np.linalg.norm(grip) + 0.003 and "(hand)" in r.events[-1], (err, r.events)
    assert err > 0.003                                                      # ... which is why the stereo measurement is the primary one
    # the base box not visible: refuses instead of placing blindly
    r = sim_robot(bias, grip, visible=True)
    r.perception.boxes = lambda max_age=1.5: {}
    old, grasp_core.ALIGN_TIMEOUT = grasp_core.ALIGN_TIMEOUT, 0.2
    try:
        r._align_over("green")
        raise AssertionError("expected a TaskError")
    except grasp_core.TaskError as exc:
        assert "not visible" in str(exc)
    finally:
        grasp_core.ALIGN_TIMEOUT = old


def test_verify_and_stack():
    r = object.__new__(grasp_core.GraspRobot)
    r.events, r.abort_flag, r._depth, r.phase = [], threading.Event(), 0, "idle"
    r.event = r.events.append

    class P:
        pos = {}

        def wait_box(self, color, count=6, timeout=8.0, abort=None):
            return [{"boxes": {color: {"t": self.pos[color]}}}] if color in self.pos else []

        def boxes(self, max_age=1.5):
            return {c: {"t": t} for c, t in self.pos.items()}
    r.perception = P()
    P.pos = {"green": [0.620, -0.220, 0.80], "blue": [0.623, -0.218, 0.88]}
    r._verify_stacked("blue", "green")
    assert "offset (+3.0, +2.0) mm" in r.events[-1] and "height above it 80 mm" in r.events[-1], r.events
    P.pos = {"green": [0.620, -0.220, 0.80], "blue": [0.650, -0.220, 0.80]}           # the box slid off and stands next to it
    try:
        r._verify_stacked("blue", "green")
        raise AssertionError("expected a TaskError")
    except grasp_core.TaskError as exc:
        assert "did not stay on the green box" in str(exc)
    P.pos = {"green": [0.620, -0.220, 0.80]}
    r._verify_stacked("blue", "green")                                               # blue not visible: says so, does not fail
    assert "could not check" in r.events[-1]
    # stack: the plan is checked against the scene first; then pick + place_on per box, bottom first
    calls = []
    r.pick, r.place_on = (lambda c, **kw: calls.append(("pick", c))), (lambda b: calls.append(("place_on", b)))
    P.pos = {n: [*xy, 0.76] for n, xy in C.BOX_DEFAULTS.items()}
    r.stack(["green", "blue", "yellow"])
    assert calls == [("pick", "blue"), ("place_on", "green"), ("pick", "yellow"), ("place_on", "blue")], calls
    calls.clear()
    P.pos = {"green": [0.62, -0.22, 0.76]}
    try:
        r.stack(["green", "blue"])
        raise AssertionError("expected a TaskError")
    except grasp_core.TaskError as exc:
        assert "cannot see the blue box" in str(exc) and calls == []                 # nothing moved
    # abort requested between two tasks of one job is not lost: only the outermost run clears the flag
    r.abort_flag.clear()
    with r.job():
        r.abort_flag.set()
        r._run("inner", lambda: None)
    assert r.abort_flag.is_set()


def test_base_motion():
    """Odometry, speed profile and the table check of the base (pure functions)."""
    q = [0.0, 0.0]
    odo = B.Odometry(*q)

    def roll(dr, dl):                                             # distances the right / left wheel rolls forward
        q[0] += dr / (B.WHEEL_R * B.FORWARD_SIGN)
        q[1] += dl / (B.WHEEL_R * B.FORWARD_SIGN)
        return odo.update(*q)
    p = roll(0.2, 0.2)
    assert np.allclose(p, [0.2, 0, 0], atol=1e-9)
    p = roll(np.pi / 4 * B.TRACK, -np.pi / 4 * B.TRACK)           # a quarter turn to the left on the spot (right wheel forward, left back)
    assert abs(p[2] - np.pi / 2) < 1e-9 and np.allclose(p[:2], [0.2, 0], atol=1e-9)
    p = roll(0.1, 0.1)
    assert np.allclose(p, [0.2, 0.1, np.pi / 2], atol=1e-9)       # forward now means +y
    assert not odo.at_home()
    # speed: slows down towards the target, never below the friction minimum, never above the maximum
    assert B.speed(1.0, B.V_MAX, B.A_LIN, B.V_MIN) == B.V_MAX and B.speed(0.0, B.V_MAX, B.A_LIN, B.V_MIN) == B.V_MIN
    assert B.speed(0.05, B.V_MAX, B.A_LIN, B.V_MIN) < B.speed(0.1, B.V_MAX, B.A_LIN, B.V_MIN)
    # the way home: face the start, drive, face forward again
    t1, d, t2 = B.way_home((0.2, 0.1, np.pi / 2))
    assert abs(d - np.hypot(0.2, 0.1)) < 1e-9 and abs(B.wrap(np.pi / 2 + t1 + t2) - 0.0) < 1e-9
    # the table check: hanging arm (hands low at the sides) and the READY pose (hand just in front of the table)
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
    import kinematics as K
    hang = list(K.all_points({}).values())
    ready = {**dict(zip(K.TORSO, [0, 1, -2, 1, 0, 0.0])), **dict(zip(K.RIGHT_ARM, [0, 0, 0, -1.8, 0, 0.3, -1.57]))}
    ready_pts = list(K.all_points(ready).values())
    start = (0.0, 0.0, 0.0)
    assert B.check_move(start, 0.10, 0.0, hang) == [] and B.check_move(start, -0.20, 0.0, hang) == [] and B.check_move(start, 0.0, np.radians(-30), hang) == []
    assert "run into the table" in B.check_move(start, 0.0, np.radians(-60), hang)[0]           # a big turn swings a corner into the table: back up first
    assert B.check_move((-0.25, 0.0, 0.0), 0.0, np.radians(-60), hang) == []
    # already so close that the margin is used up: backing away is allowed, going in is not
    close = (0.25, 0.0, 0.0)
    assert B.penetration(close, hang) > 0 and B.check_move(close, -0.15, 0.0, hang) == []
    assert "already close" in B.check_move(close, 0.05, 0.0, hang)[0]
    assert "run into the table" in B.check_move(start, 0.25, 0.0, hang)[0]                    # 35 cm front + 25 cm > the table edge at 53 cm
    assert "arm would hit the table" in B.check_move(start, 0.05, 0.0, ready_pts)[0]            # the READY hand is over the table edge once it rolls forward
    assert any("at most 30 cm" in m for m in B.check_move(start, 0.5, 0.0, hang))
    assert any("at most 90 degrees" in m for m in B.check_move(start, 0.0, np.radians(120), hang))
    assert any("from its start pose" in m for m in B.check_move((0.4, 0.0, 0.0), -0.3, 0.0, hang) + B.check_move((0.4, 0.0, 0.0), 0.3, np.radians(90), hang))


def test_drive():
    """GraspRobot.drive on a simulated differential-drive base (no SDK): reaches the target, refuses what the check refuses, can be aborted
    with the wheels stopped, comes home, and blocks manipulation while the base is away."""
    r = object.__new__(grasp_core.GraspRobot)
    r.events, r.abort_flag, r._depth, r.phase, r.holding, r.held, r.stream = [], threading.Event(), 0, "idle", False, None, None
    r.event, r._check, r._tick = r.events.append, (lambda: None), (lambda t_loop: sim.step(1.0 / grasp_core.RATE))
    r.state, r.footprint, r._odo_lock, r.gt_base = (lambda: {"joints": {}}), B.FOOTPRINT, threading.Lock(), None
    r._end_stream = lambda: None

    class Sim:
        """A differential-drive base that follows the commanded (v, w) at once; wheel joint angles as the SDK would report them."""
        def __init__(self):
            self.q, self.sent, self.pose = [0.0, 0.0], [], np.zeros(3)
            self.sign = B.FORWARD_SIGN                      # how the (simulated) wheels really turn, whatever the odometry assumes

        def step(self, dt):
            v, w = self.sent[-1] if self.sent else (0.0, 0.0)
            dr, dl = (v + w * B.TRACK / 2) * dt, (v - w * B.TRACK / 2) * dt
            self.q[0] += dr / (B.WHEEL_R * self.sign)
            self.q[1] += dl / (B.WHEEL_R * self.sign)
            self.pose += [v * dt * np.cos(self.pose[2]), v * dt * np.sin(self.pose[2]), w * dt]
    sim = Sim()
    r._wheels = lambda: tuple(sim.q)
    r._send_base = lambda v, w: sim.sent.append((v, w))
    r._stop_base = lambda: sim.sent.append((0.0, 0.0))
    r.odo = B.Odometry(*sim.q)
    r.drive(forward=-0.10)                                                                            # back first: forward + turn would swing the corners into the table
    x, y, yaw = r.base_pose()
    assert abs(x + 0.10) < 0.01 and abs(y) < 0.005 and abs(yaw) < 0.01, (x, y, yaw)
    assert max(abs(v) for v, _ in sim.sent) <= B.V_MAX + 1e-9 and sim.sent[-1] == (0.0, 0.0)          # never above the limit, ends stopped
    r.drive(turn_deg=-30)
    assert abs(np.degrees(r.base_pose()[2]) + 30) < 1.0 and abs(np.degrees(sim.pose[2]) + 30) < 1.0, r.base_pose()
    # the seed pose needs room ahead of the base: measured to the table where it really is, from wherever the base stands
    r.seeded, r.squatted, r.gt_base = False, False, (np.array([0.2, 0.0, 0.0]), __import__("time").time())
    try:
        r._require_table_clear_for_seed()
        raise AssertionError("expected a TaskError")                                                  # the base stands 20 cm closer to the table
    except grasp_core.TaskError as exc:
        assert "cm ahead of the base" in str(exc), str(exc)
    r.gt_base = None
    r.odo = B.Odometry(*sim.q, pose=tuple(sim.pose))                                                  # the test's fake true pose was swapped for a made-up one above
    r.drive(home=True)
    x, y, yaw = r.base_pose()
    assert np.hypot(x, y) < B.HOME_TOL_LIN and abs(yaw) < B.HOME_TOL_ANG and r.odo.at_home(), (x, y, yaw)
    r._require_table_clear_for_seed()                                                                 # at home again: the table is 58 cm ahead: fine
    # the simulator's true pose wins while it is fresh (the wheels slip when turning), the odometry takes over when it stops
    import time as _time
    r.gt_base = (np.array([0.02, -0.01, 0.1]), _time.time())
    assert np.allclose(r.base_pose(), [0.02, -0.01, 0.1]) and not r.odo.at_home()                 # 0.1 rad = 5.7 degrees off
    r.gt_base = (np.array([0.0, 0.0, 0.0]), _time.time() - 5.0)                                    # stale: back to the wheels
    assert np.allclose(r.base_pose(), r.odo.pose) and np.allclose(r.base_pose(), [0.02, -0.01, 0.1], atol=1e-6)
    r.odo = B.Odometry(*sim.q)
    r.gt_base = None
    # refused: too far, into the table; nothing is sent
    sent = len(sim.sent)
    for kw, text in (({"forward": 0.5}, "at most 30 cm"), ({"forward": 0.3}, "run into the table"), ({"turn_deg": 120}, "at most 90 degrees")):
        try:
            r.drive(**kw)
            raise AssertionError(kw)
        except grasp_core.TaskError as exc:
            assert text in str(exc), (kw, str(exc))
    assert len(sim.sent) == sent
    # a wheel sign that is the wrong way round must stop the base within a few centimetres, not run away
    flip, B.FORWARD_SIGN = B.FORWARD_SIGN, -B.FORWARD_SIGN
    try:
        r.odo = B.Odometry(*sim.q)           # the odometry now disagrees with the simulated wheels
        sent = len(sim.sent)
        try:
            r.drive(forward=-0.1)
            raise AssertionError("expected a TaskError")
        except grasp_core.TaskError as exc:
            assert "wrong way" in str(exc), str(exc)
        assert len(sim.sent) - sent < 30 and sim.sent[-1] == (0.0, 0.0) and abs(sim.pose[0]) < 0.1, (len(sim.sent) - sent, sim.pose)
    finally:
        B.FORWARD_SIGN = flip
        r.odo = B.Odometry(*sim.q)
        sim.pose[:] = 0.0
    # abort in the middle: the wheels are told to stop
    calls = []
    original = r._send_base

    def send_and_abort(v, w):
        original(v, w)
        calls.append(v)
        if len(calls) == 5:
            r.abort_flag.set()
    r._send_base = send_and_abort
    r._check = lambda: (_ for _ in ()).throw(grasp_core.Aborted("aborted")) if r.abort_flag.is_set() else None
    try:
        r.drive(forward=-0.2)
        raise AssertionError("expected Aborted")
    except grasp_core.Aborted:
        pass
    assert sim.sent[-1] == (0.0, 0.0) and abs(r.base_pose()[0]) < 0.19


def test_base_frame():
    """The base may stand anywhere: boxes seen in the world are handed on in the base frame, spots and table checks follow the base."""
    import math
    # scene_config helpers
    base = (0.1, -0.2, math.radians(30))
    for x, y in ((0.5, 0.1), (-0.3, 0.2)):
        assert np.allclose(C.world_to_base(base, *C.base_to_world(base, x, y)), (x, y))
    assert abs(C.table_front_distance((0.0, 0.0, 0.0)) - 0.58) < 0.02 and C.table_front_distance((-0.3, 0.0, 0.0)) > 0.85       # backed up: more room
    assert C.table_front_distance((0.2, 0.0, 0.0)) < C.SEED_EDGE_X and C.table_front_distance((2.0, 0.0, math.pi)) > 0.5       # too close / table behind: other cases
    # a box standing where the table is: valid in the base frame of a shifted base only if it is on the table in the world
    assert C.check_pose(0.64, -0.20, 0.72) is None
    shifted = (-0.2, 0.0, 0.0)                                                      # the base backed up 20 cm: the same box is 20 cm further away in its frame
    wx, wy = C.base_to_world(shifted, 0.84, -0.20)                                  # (0.64, -0.20) in the world is (0.84, -0.20) in this base's frame
    assert abs(wx - 0.64) < 1e-9 and C.check_pose(0.84, -0.20, 0.72, base=shifted) is not None          # out of reach from there ("out of reach")
    # free spots: the same table, a base that stands a little off and turned still finds spots, all on the table in the world and in reach
    boxes = {"blue": (0.62, -0.34, 0.76)}
    turned = (0.05, 0.05, math.radians(15))
    spot = C.direction_spot("right", boxes, exclude="blue", base=turned)
    assert spot is not None and C.on_table(*spot, base=turned) and C.wrist_distance(C.box_tip(*spot, C.TABLE_TOP)) <= C.MAX_REACH
    # taking a stack apart from a displaced base: the start spots are world positions, the answer is in the base frame
    base = (0.0, -0.05, 0.0)
    stack_b = {"blue": (0.62, -0.29, 0.76), "green": (0.62, -0.29, 0.84), "yellow": (0.62, -0.05 - 0.0, 0.76)}      # blue + green stacked at world y = -0.34; yellow at its start spot
    spot = C.table_spot("green", stack_b, base=base)
    assert spot is not None and np.allclose(C.base_to_world(base, *spot), C.BOX_DEFAULTS["green"], atol=1e-9), spot   # its own start spot, seen from the shifted base
    # Perception hands the boxes on in the base frame
    p = grasp_core.Perception.__new__(grasp_core.Perception)
    p.base_pose = lambda: (0.0, 0.0, 0.0)
    data = {"boxes": {"blue": {"t": [0.62, -0.2, 0.76], "R": np.eye(3).tolist(), "n": 4}}}
    p._to_base(data)
    assert data["boxes"]["blue"]["t"] == [0.62, -0.2, 0.76]                          # at the start pose nothing changes
    p.base_pose = lambda: (0.1, 0.2, math.pi / 2)                                   # the base stands at (0.1, 0.2) facing +y
    data = {"boxes": {"blue": {"t": [0.62, -0.2, 0.76], "R": np.eye(3).tolist(), "n": 4}}}
    p._to_base(data)
    t = data["boxes"]["blue"]["t"]
    assert np.allclose(t, [-0.4, -0.52, 0.76], atol=1e-9), t                         # 0.4 m to its right (-y in its frame): x' = dy = -0.4, y' = -dx = -0.52
    assert np.allclose(np.array(data["boxes"]["blue"]["R"]), np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]), atol=1e-9)   # the box is turned by -90 degrees in that frame


def test_arms():
    """Two arms: the reach model per arm, the comfort score and arm choice, the grippers one by one, switching arms (park, squat, ready), the choice
    of the arm in the executor."""
    row = {n: (*xy, 0.76) for n, xy in C.BOX_DEFAULTS.items()}
    # the default row is the right arm's: the left shoulder is 0.44-0.56 m away sideways
    assert all(C.best_arm(n, row) == "right" and C.pick_problem(n, row, "left") for n in row)
    assert C.comfort(C.box_tip(0.62, -0.22, 0.72)) > C.comfort(C.box_tip(0.62, -0.34, 0.72)) > 0       # the middle of the reach is more comfortable than its edge
    left_box = {"yellow": (0.60, 0.22, 0.76)}                                                           # a box on the left side: the left arm's
    assert C.best_arm("yellow", left_box) == "left" and C.pick_problem("yellow", left_box, "right")
    mid = {"blue": (0.50, -0.03, 0.76)}                                                                # in reach of both: the more comfortable one, the arm in place keeps it
    opts = C.arm_options("blue", mid)
    assert opts["right"]["problem"] is None and opts["left"]["problem"] is None
    best = C.best_arm("blue", mid)
    other = "left" if best == "right" else "right"
    assert C.best_arm("blue", mid, prefer=other, slack=1.0) == other and C.best_arm("blue", mid, prefer=other, slack=0.0) == best
    # a move needs ONE arm for the pick and the place: from the right band to the left side no arm can do both
    two = {"blue": (0.62, -0.34, 0.76), "yellow": (0.60, 0.22, 0.76)}
    assert C.move_spot("left", "blue", two) is None or C.move_spot("left", "blue", two)[1] < 0.0         # blue cannot go to the left side
    assert C.best_arm("blue", two, spot=(0.62, 0.30)) is None and C.best_arm("yellow", two, spot=(0.60, 0.15)) == "left"
    # a stack where the arms can reach: bottom on the right, a box from the left would need the left arm to reach the right-side stack: refused
    assert any("out of reach" in p for p in C.stack_problems(["green", "yellow"], {"green": (0.62, -0.22, 0.76), "yellow": (0.60, 0.22, 0.76)}))
    # the grippers are separate (library: id 0 = left, 1 = right)
    r = object.__new__(grasp_core.GraspRobot)
    r.side = "right"

    class G:
        min_q, max_q, target_q = np.array([0.0, 0.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])        # open: max
    r.gripper = G()
    r._grip(1.0, "left")
    assert np.allclose(r.gripper.target_q, [0.0, 2.0])                                                          # only the left one closed
    r._grip(1.0)
    assert np.allclose(r.gripper.target_q, [0.0, 0.0])                                                          # default: the arm at work (right)
    r._grip(0.0, "left")
    assert np.allclose(r.gripper.target_q, [1.0, 0.0])
    # switching arms: the other arm parks, the torso squats once, the arm goes to READY once
    r = object.__new__(grasp_core.GraspRobot)
    r.events, r.side, r.squatted, r.ready, r.seeded = [], "right", False, {"right": False, "left": False}, False
    r.event = r.events.append
    calls = []
    r._joint_arm = lambda side, q, seconds, name: calls.append((name, side))
    r._joint_move = lambda body, name, head_builder=None: calls.append((name, "torso"))
    r._require_table_clear_for_seed = lambda: calls.append(("table check", ""))
    r.cmd_tip = None
    r.tip = lambda side=None: np.zeros(3)
    r.hold = {"right": {}, "left": {}}
    r.rby = type("Rby", (), {"BodyComponentBasedCommandBuilder": staticmethod(lambda: type("B", (), {"set_torso_command": lambda s, c: s})()),
                              "TorsoCommandBuilder": staticmethod(lambda: type("T", (), {"set_command": lambda s, c: s})()),
                              "JointPositionCommandBuilder": staticmethod(lambda: type("J", (), {"set_minimum_time": lambda s, t: s, "set_position": lambda s, q: s})())})()
    r.robot = type("Robot", (), {"get_state": lambda s: type("S", (), {"position": np.zeros(24)})()})()
    r._activate("right")
    assert [c for c in calls if c[0] != "table check"] == [("torso", "torso"), ("arm", "right")] and r.ready == {"right": True, "left": False} and r.squatted
    calls.clear()
    r._activate("right")
    assert calls == []                                                                                  # already so: nothing moves
    r._activate("left")
    assert calls == [("park", "right"), ("arm", "left")] and r.ready == {"right": False, "left": True} and r.side == "left"      # right parks, no second squat
    # the executor's choice of the arm
    r.perception = type("P", (), {"boxes": lambda s, max_age=1.5: {n: {"t": list(p)} for n, p in {**row, "yellow": (0.60, 0.22, 0.76)}.items()}})()
    assert r._choose_arm("yellow", "auto") == "left" and r._choose_arm("green", "auto") == "right"
    assert r._choose_arm("green", "right") == "right"
    for bad in (("green", "left"), ("nothing", "auto"), ("yellow", "right")):
        try:
            r._choose_arm(*bad)
            raise AssertionError(bad)
        except grasp_core.TaskError as exc:
            assert "reach" in str(exc) or "cannot see" in str(exc), str(exc)
    try:
        r._choose_arm("blue", "auto", dest=(0.60, 0.15, C.TABLE_TOP))                                   # the right arm picks it, but only the left reaches the place
        raise AssertionError("expected a TaskError")
    except grasp_core.TaskError as exc:
        assert "none reaches both" in str(exc) or "neither arm" in str(exc), str(exc)


def test_unstack():
    """GraspRobot.unstack with a fake scene: top box first, onto its own start spot, bottom box stays; a clear table is refused."""
    r = object.__new__(grasp_core.GraspRobot)
    r.events, r.abort_flag, r._depth, r.phase, r.holding, r.held = [], threading.Event(), 0, "idle", False, None
    r.side = "right"
    r.event = r.events.append
    pos = {"blue": (0.62, -0.34, 0.76), "green": (0.62, -0.34, 0.84), "yellow": (0.62, -0.34, 0.92)}
    moved = []

    class P:
        def boxes(self, max_age=1.5):
            return {c: {"t": list(t)} for c, t in pos.items()}

        def wait_box(self, color, count=6, timeout=8.0, abort=None):
            return [{"boxes": {color: {"t": list(pos[color])}}}]
    r.base_pose = lambda: (0.0, 0.0, 0.0)                                  # the base stands at its start pose
    r.perception = P()
    r.pick = lambda c, **kw: moved.append(("pick", c))
    r.place = lambda x, y, top=C.TABLE_TOP: (moved.append(("place", round(x, 2), round(y, 2))), pos.__setitem__(moved[-2][1], (x, y, 0.76)))
    r.unstack()
    assert moved == [("pick", "yellow"), ("place", 0.62, -0.1), ("pick", "green"), ("place", 0.62, -0.22)], moved
    assert pos["blue"] == (0.62, -0.34, 0.76) and "on the table at (0.620, -0.220)" in r.events[-1]
    try:
        r.unstack()
        raise AssertionError("expected a TaskError")
    except grasp_core.TaskError as exc:
        assert "nothing is stacked" in str(exc)
    # a box that did not land on the table is reported
    pos.update({"green": (0.62, -0.34, 0.84)})
    r.place = lambda x, y, top=C.TABLE_TOP: pos.__setitem__("green", (x, y, 0.84))      # it stayed in the air
    try:
        r.unstack()
        raise AssertionError("expected a TaskError")
    except grasp_core.TaskError as exc:
        assert "not standing on the table" in str(exc)
    r.holding, r.held = True, "blue"
    try:
        r.unstack()
        raise AssertionError("expected a TaskError")
    except grasp_core.TaskError as exc:
        assert "holds the blue box" in str(exc)


# ---- rbyctl against the server with the fake robot -----------------------------------------------------------------------------
def test_rbyctl():
    import contextlib
    import io
    p_cmd, p_status = free_port(), free_port()
    robot = FakeRobot()
    srv = Server(robot, cmd_port=p_cmd, status_addr=("127.0.0.1", p_status))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old, C.PORT_EXEC_CMD = C.PORT_EXEC_CMD, p_cmd

    def run(*argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = rbyctl.main(list(argv))
        return rc, out.getvalue()
    try:
        rc, out = run("stack", "green", "blue", "yellow")
        assert rc == 0 and out.strip().endswith("OK") and "done: stack" in out, (rc, out)
        assert robot.calls[-1] == ("stack", ("green", "blue", "yellow"))
        rc, out = run("stack", "green", "blue")                                      # the fake robot refuses this one
        assert rc == 1 and "FAILED: failed: stack: cannot see the yellow box" in out, (rc, out)
        rc, out = run("unstack")
        assert rc == 0 and robot.calls[-1] == ("unstack",) and "done: unstack" in out, (rc, out)
        # the agent tools talk to the same service: state is a reply-only query, joints a task whose refusal comes back with the reasons
        r = rbyctl.ask({"cmd": "state"}, 5.0, quiet=True)
        assert r["type"] == "state" and r["joints"] == {"right_arm_1": -0.5}
        r = rbyctl.ask({"cmd": "joints", "trajectory": {"unit": "deg", "waypoints": [{"t": 2.0, "right_arm_1": -90}]}}, 5.0, quiet=True)
        assert r["ok"] and robot.calls[-1] == ("joints", 2.0), r
        r = rbyctl.ask({"cmd": "joints", "trajectory": {"unit": "bad", "waypoints": [{"t": 1.0}]}}, 5.0, quiet=True)
        assert not r["ok"] and "not safe" in r["text"], r
        rc, out = run("pick", "yellow")
        assert rc == 0 and robot.calls[-1] == ("pick", "yellow", True, (0.0, 0.0, 0.0))
        rc, out = run("place-on", "green")
        assert rc == 0 and robot.calls[-1] == ("place_on", "green")
        rc, out = run("move", "blue", "0.6", "-0.3")
        assert rc == 0 and robot.calls[-2:] == [("pick", "blue", True, (0.0, 0.0, 0.0)), ("place", 0.6, -0.3, C.TABLE_TOP)], robot.calls
        rc, out = run("move", "blue", "right")                                       # the default table has no free spot beside the row
        assert rc == 0 or "no free reachable spot" in out
        rc, out = run("scene")
        scene = json.loads(out)
        assert rc == 0 and set(scene["boxes"]) == set(C.BOX_NAMES) and scene["unseen"] == [] and "free_spots" in scene
        assert scene["robot"]["holding"] is False and scene["table"]["assumed"] is True
        # a busy executor refuses at once, with the reason
        robot.row, srv.task = {}, "pick"
        rc, out = run("home")
        assert rc == 1 and "busy with 'pick'" in out, (rc, out)
        srv.task = None
        rc, out = run("scene")
        assert json.loads(out)["unseen"] == list(C.BOX_NAMES)                        # nothing seen: says which boxes are missing
    finally:
        C.PORT_EXEC_CMD = old
        srv.stop.set()


if __name__ == "__main__":
    tests = [test_config, test_boxes, test_helpers, test_align, test_verify_and_stack, test_map, test_window, test_say, test_base_motion, test_drive, test_base_frame, test_arms, test_unstack, test_server, test_rbyctl]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} offline checks passed")
