"""Offline checks of the control centre, no simulator and no robot:
  QT_QPA_PLATFORM=offscreen /usr/bin/python3 test_offline.py
Covers: scene_config reachability, the pure grasp_core helpers, the scene map, the window (buttons -> UDP commands, status -> button
states, joint table, camera views), and servo_server's command handling with a fake robot (queueing, errors, abort).
Not covered: anything that talks to the simulator or the robot (grasp_core.GraspRobot, rby1_vision.poll_commands).
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

import grasp_core  # noqa: E402
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


def test_helpers():
    assert grasp_core.smoothstep(0) == 0 and grasp_core.smoothstep(1) == 1 and abs(grasp_core.smoothstep(0.5) - 0.5) < 1e-9
    above, down, back = grasp_core.place_waypoints(np.array([0.58, -0.2, 0.88]), 0.62, -0.28, 0.72)
    assert above[2] >= 0.88 and abs(above[0] - 0.625) < 1e-9 and abs(above[1] + 0.28) < 1e-9
    assert abs(down[2] - (0.72 + 0.04 + 0.004)) < 1e-9 and np.allclose(back - down, [-0.10, 0, 0])
    low = grasp_core.place_waypoints(np.array([0.58, -0.2, 0.70]), 0.62, -0.28, 0.72)[0]
    assert low[2] >= 0.72 + 0.04 + grasp_core.HOVER - 1e-9              # never carries the box through the table


# ---- the map ------------------------------------------------------------------------------------------------------------
def test_map():
    m = SceneMap()
    m.resize(400, 460)
    m.show()
    for x, y in ((0.0, 0.0), (0.64, -0.2), (1.1, 0.5)):
        px = m.to_px(x, y)
        wx, wy = m.to_world(px.x(), px.y())
        assert abs(wx - x) < 1e-6 and abs(wy - y) < 1e-6
    got = []
    m.placed.connect(lambda mode, x, y: got.append((mode, x, y)))
    target = m.to_px(0.63, -0.18)
    QtTest.QTest.mousePress(m, QtCore.Qt.LeftButton, pos=QtCore.QPoint(int(target.x()), int(target.y())))
    QtTest.QTest.mouseRelease(m, QtCore.Qt.LeftButton, pos=QtCore.QPoint(int(target.x()), int(target.y())))
    assert len(got) == 1 and got[0][0] == "box" and abs(got[0][1] - 0.63) < 0.01 and abs(got[0][2] + 0.18) < 0.01
    m.set_mode("table")
    QtTest.QTest.mousePress(m, QtCore.Qt.LeftButton, pos=QtCore.QPoint(100, 100))
    QtTest.QTest.mouseMove(m, QtCore.QPoint(120, 140))
    QtTest.QTest.mouseRelease(m, QtCore.Qt.LeftButton, pos=QtCore.QPoint(120, 140))
    assert got[-1][0] == "table"
    default = len(m.reach_cells())
    assert default > 0                                                  # a green region exists for the default table (a narrow one)
    m.set_table((0.80, 0.0), C.TABLE_SIZE, C.TABLE_TOP)
    assert len(m.reach_cells()) > 2 * default                           # and a table pulled closer opens up many more positions
    img = m.grab()
    assert not img.isNull()
    m.close()


# ---- the window ---------------------------------------------------------------------------------------------------------
class FakeBridge:
    def snapshot(self):
        rng = np.random.default_rng(0)
        imgs = {n: (rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), 0.1)
                for n in ("wrist overlay", "stereo overlay", "wrist left", "stereo left")}
        names = [f"torso_{i}" for i in range(6)] + [f"right_arm_{i}" for i in range(7)] + [f"left_arm_{i}" for i in range(7)] + \
                ["left_wheel", "right_wheel", "head_0", "head_1", "gripper_finger_r1", "gripper_finger_r2"]
        return {"images": imgs, "joints": {n: 0.1 * i for i, n in enumerate(names)}, "joints_age": 0.1, "box": (0.64, -0.2, 0.76), "box_age": 0.1}

    def close(self):
        pass


def test_window():
    p_scene, p_exec, p_status = free_port(), free_port(), free_port()
    scene_rx, exec_rx = listener(p_scene), listener(p_exec)
    w = ControlCenter(FakeBridge(), ("127.0.0.1", p_scene), ("127.0.0.1", p_exec), p_status)
    w.show()
    pump(300)
    # cameras and joints are filled from the bridge
    assert all(v.pixmap() is not None for v in w.views.values())
    assert w.joints.rowCount() > 0 and w.joints.item(0, 1) is not None
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
    assert recv_all(scene_rx)[-1] == {"cmd": "set_box", "x": 0.63, "y": -0.18, "yaw_deg": 10.0}
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
    # clicks on the map are applied
    px = w.map.to_px(0.60, -0.15)
    QtTest.QTest.mousePress(w.map, QtCore.Qt.LeftButton, pos=QtCore.QPoint(int(px.x()), int(px.y())))
    QtTest.QTest.mouseRelease(w.map, QtCore.Qt.LeftButton, pos=QtCore.QPoint(int(px.x()), int(px.y())))
    assert recv_all(scene_rx)[-1]["cmd"] == "set_box"
    # log lines from the executor appear
    st.sendto(json.dumps({"type": "log", "text": "hello from the executor"}).encode(), ("127.0.0.1", p_status))
    pump(250)
    assert "hello from the executor" in w.log.toPlainText()
    if os.environ.get("SAVE_SHOT"):
        w.grab().save(os.environ["SAVE_SHOT"])
    w.close()


# ---- the executor service with a fake robot -----------------------------------------------------------------------------
class FakeRobot:
    def __init__(self):
        self.calls, self.flag = [], threading.Event()

    def status(self):
        return {"phase": "idle", "holding": False, "seeded": True, "tip": [0.5, -0.2, 0.8], "closeness": 0.0}

    def abort(self):
        self.flag.set()

    def pick(self, use_image=True, bias=(0, 0, 0)):
        self.calls.append(("pick", use_image, bias))
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

    def set_gripper(self, c):
        self.calls.append(("gripper", c))


def test_server():
    p_cmd, p_status = free_port(), free_port()
    status_rx = listener(p_status)
    robot = FakeRobot()
    srv = Server(robot, cmd_port=p_cmd, status_addr=("127.0.0.1", p_status))
    try:
        srv.handle({"cmd": "pick_place", "x": 0.6, "y": -0.3, "top": 0.72})
        time.sleep(0.4)
        assert robot.calls == [("pick", True, (0.0, 0.0, 0.0)), ("place", 0.6, -0.3, 0.72)], robot.calls
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


if __name__ == "__main__":
    tests = [test_config, test_helpers, test_map, test_window, test_server]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} offline checks passed")
