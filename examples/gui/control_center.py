"""RBY1 control centre: cameras, joint states, scene editing and the pick / place tasks in one window (PyQt5).

  source /opt/ros/jazzy/setup.bash; export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
  DISPLAY=:1 /usr/bin/python3 control_center.py          (examples/gui/run_gui.sh starts everything it talks to)

  cameras     the wrist and stereo views with the servo overlay, and the left-side raw views (ros_bridge.py)
  scene map   click or drag to put the box, the place target or the table where you want them (scene_map.py)
  scene       box / table numbers, sent to the simulator over UDP (src/rby1_vision.py poll_commands)
  tasks       pick (visual servoing), place, pick & place, home, gripper, ABORT: sent to the executor service (servo_server.py)
  robot       all joint positions and the executor's status

NOT RUN YET against a simulator (see grasp_core.py); the window, the map and the commands are covered by test_offline.py.
"""
import json
import socket
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import scene_config as C
from scene_map import SceneMap

CAMERAS = [("wrist overlay", "wrist camera (back of the hand) + servo overlay"), ("stereo overlay", "stereo + box pose"),
           ("wrist left", "left wrist camera (raw)"), ("stereo left", "left stereo camera (raw)")]


class ImageView(QtWidgets.QLabel):
    def __init__(self, caption):
        super().__init__()
        self.caption = caption
        self.setMinimumSize(240, 180)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background: #1b1c1f; color: #9aa0a6;")
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setText(f"{caption}\n(no image yet)")
        self._img = None

    def show_array(self, arr, age):
        h, w, _ = arr.shape
        self._img = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._age = age
        self._rescale()

    def _rescale(self):
        if self._img is not None:
            self.setPixmap(QtGui.QPixmap.fromImage(self._img).scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()


def spin(value, lo, hi, step=0.01, decimals=3, suffix=" m"):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    s.setValue(value)
    return s


class ControlCenter(QtWidgets.QMainWindow):
    def __init__(self, bridge, scene_addr=("127.0.0.1", C.PORT_SCENE), exec_addr=("127.0.0.1", C.PORT_EXEC_CMD),
                 status_port=C.PORT_EXEC_STATUS):
        super().__init__()
        self.bridge, self.scene_addr, self.exec_addr = bridge, scene_addr, exec_addr
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", status_port))
        self.rx.setblocking(False)
        self.exec_status, self.exec_status_t = {}, 0.0
        self.table_top = C.TABLE_TOP
        self.setWindowTitle("RBY1 control center")
        self.resize(1500, 980)
        self._joint_names, self._tick = [], 0
        self._build()
        self._set_target(0.64, -0.30, final=False)  # visible and draggable from the start
        self.fast = QtCore.QTimer(self, interval=66, timeout=self._refresh)
        self.fast.start()
        self.slow = QtCore.QTimer(self, interval=100, timeout=self._poll_status)
        self.slow.start()

    # ---- layout ------------------------------------------------------------------------------------------------------
    def _build(self):
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)
        # left: cameras and log
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        grid = QtWidgets.QGridLayout()
        self.views = {}
        for i, (name, caption) in enumerate(CAMERAS):
            box = QtWidgets.QGroupBox(caption)
            bl = QtWidgets.QVBoxLayout(box)
            bl.setContentsMargins(2, 2, 2, 2)
            self.views[name] = ImageView(name)
            bl.addWidget(self.views[name])
            grid.addWidget(box, i // 2, i % 2)
        lv.addLayout(grid, 5)
        self.log = QtWidgets.QPlainTextEdit(readOnly=True, maximumBlockCount=500)
        self.log.setMaximumHeight(170)
        lv.addWidget(self.log, 1)
        split.addWidget(left)
        # right: map, scene, tasks, robot
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        self.map = SceneMap()
        self.map.dragged.connect(self._on_dragged)
        self.map.chosen.connect(self._on_chosen)
        self.map.rotated.connect(self._on_rotated)
        rv.addWidget(self.map, 8)

        scene = QtWidgets.QGroupBox("scene (simulator)")
        sv = QtWidgets.QVBoxLayout(scene)
        row = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("reset scene")
        reset.clicked.connect(self.reset_scene)
        row.addWidget(reset)
        self.allow_any = QtWidgets.QCheckBox("allow positions the arm cannot reach")
        self.allow_any.toggled.connect(self.map.set_allow_any)
        row.addWidget(self.allow_any, 1)
        sv.addLayout(row)
        self.box_x, self.box_y = spin(C.BOX_DEFAULT[0], 0.0, 1.3), spin(C.BOX_DEFAULT[1], -0.8, 0.8)
        self.box_yaw = spin(0.0, -30, 30, 5, 0, " deg")  # the gripper approach is fixed: a box turned further than this is not graspable
        self.tab_x, self.tab_y = spin(C.TABLE_CENTER[0], 0.3, 1.4), spin(C.TABLE_CENTER[1], -0.6, 0.6)
        self.tab_top = spin(C.TABLE_TOP, 0.5, 1.0)
        self.place_x, self.place_y = spin(0.60, 0.0, 1.3), spin(-0.30, -0.8, 0.8)
        precise = QtWidgets.QGroupBox("exact values (everything above can be dragged on the map)")
        precise.setCheckable(True)
        precise.setChecked(False)
        inner = QtWidgets.QWidget()
        sg = QtWidgets.QGridLayout(inner)
        sg.setContentsMargins(0, 0, 0, 0)
        for r, (label, widgets) in enumerate((("box x / y / yaw", (self.box_x, self.box_y, self.box_yaw)),
                                              ("table x / y / top", (self.tab_x, self.tab_y, self.tab_top)),
                                              ("place target x / y", (self.place_x, self.place_y)))):
            sg.addWidget(QtWidgets.QLabel(label), r, 0)
            for c, w in enumerate(widgets):
                sg.addWidget(w, r, c + 1)
        for r, (text, slot) in enumerate((("apply box", self.apply_box), ("apply table", self.apply_table),
                                          ("set target", self.apply_target))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            sg.addWidget(b, r, 4)
        pl = QtWidgets.QVBoxLayout(precise)
        pl.addWidget(inner)
        inner.setVisible(False)
        precise.toggled.connect(inner.setVisible)
        sv.addWidget(precise)
        rv.addWidget(scene)

        tasks = QtWidgets.QGroupBox("tasks (executor)")
        tg = QtWidgets.QGridLayout(tasks)
        self.target_label = QtWidgets.QLabel("place target: drag the blue square on the map")
        tg.addWidget(self.target_label, 0, 0, 1, 3)
        self.use_image = QtWidgets.QCheckBox("camera correction (hybrid servo)")
        self.use_image.setChecked(True)
        tg.addWidget(self.use_image, 0, 3, 1, 2)
        self.buttons = {}
        for c, (key, text, slot) in enumerate((("pick", "pick", self.do_pick), ("place", "place", self.do_place),
                                               ("pick_place", "pick and place", self.do_pick_place), ("home", "home / ready", self.do_home),
                                               ("open", "open gripper", lambda: self.send_exec({"cmd": "gripper", "closeness": 0.0})),
                                               ("close", "close gripper", lambda: self.send_exec({"cmd": "gripper", "closeness": 1.0})))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            tg.addWidget(b, 1 + c // 3, c % 3)
            self.buttons[key] = b
        stop = QtWidgets.QPushButton("ABORT")
        stop.setStyleSheet("background: #c0392b; color: white; font-weight: bold;")
        stop.clicked.connect(lambda: self.send_exec({"cmd": "abort"}))
        tg.addWidget(stop, 1, 3, 2, 2)
        rv.addWidget(tasks)

        robot = QtWidgets.QGroupBox("robot")
        rl = QtWidgets.QVBoxLayout(robot)
        self.status_label = QtWidgets.QLabel("executor: not connected")
        self.status_label.setWordWrap(True)
        rl.addWidget(self.status_label)
        self.joints = QtWidgets.QTableWidget(0, 6)
        self.joints.horizontalHeader().setVisible(False)
        self.joints.verticalHeader().setVisible(False)
        self.joints.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.joints.setMaximumHeight(120)
        self.joints.setStyleSheet("font-size: 10px;")
        rl.addWidget(self.joints)
        rv.addWidget(robot, 1)
        split.addWidget(right)
        split.setSizes([760, 740])
        self._enable_buttons()

    # ---- messages ----------------------------------------------------------------------------------------------------
    def say(self, text):
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    def send_scene(self, msg):
        self.tx.sendto(json.dumps(msg).encode(), self.scene_addr)

    def send_exec(self, msg):
        self.tx.sendto(json.dumps(msg).encode(), self.exec_addr)
        self.say(f"-> {msg}")

    def _poll_status(self):
        while True:
            try:
                msg = json.loads(self.rx.recv(65536))
            except BlockingIOError:
                break
            except ValueError:
                continue
            if msg.get("type") == "status":
                self.exec_status, self.exec_status_t = msg, time.time()
            elif msg.get("type") == "log":
                self.say(msg["text"])
        self._update_status()

    def _connected(self):
        return time.time() - self.exec_status_t < 2.0

    def _update_status(self):
        s = self.exec_status
        if not self._connected():
            self.status_label.setText("executor: not connected (is servo_server.py running?)")
        else:
            tip = ", ".join(f"{v:.3f}" for v in s.get("tip", []))
            self.status_label.setText(
                f"executor: {s.get('state')}  task: {s.get('task') or '-'}  phase: {s.get('phase')}   "
                f"holding box: {'yes' if s.get('holding') else 'no'}   seeded: {'yes' if s.get('seeded') else 'no'}   "
                f"gripper closeness {s.get('closeness', 0):.2f}   fingertips ({tip}) m")
        self.map.set_table_min_edge(None if s.get("seeded") else C.SEED_EDGE_X)
        self._enable_buttons()

    def _enable_buttons(self):
        s, ok = self.exec_status, self._connected()
        busy, holding = bool(s.get("task")), bool(s.get("holding"))
        self.buttons["pick"].setEnabled(ok and not busy and not holding)
        self.buttons["place"].setEnabled(ok and not busy and holding)
        self.buttons["pick_place"].setEnabled(ok and not busy and not holding)
        self.buttons["home"].setEnabled(ok and not busy and not holding)
        self.buttons["open"].setEnabled(ok and not busy)
        self.buttons["close"].setEnabled(ok and not busy)

    # ---- scene -------------------------------------------------------------------------------------------------------
    def _table(self):
        return (self.tab_x.value(), self.tab_y.value()), C.TABLE_SIZE, self.tab_top.value()

    def _check_box(self, x, y):
        (c, size, top) = self._table()
        reason = C.check_pose(x, y, top, c, size)
        if reason and not self.allow_any.isChecked():
            self.say(f"not sent: ({x:.3f}, {y:.3f}) is {reason} (tick 'allow positions the arm cannot reach' to place it anyway)")
            return False
        if reason:
            self.say(f"warning: ({x:.3f}, {y:.3f}) is {reason}")
        return True

    def apply_box(self):
        x, y = self.box_x.value(), self.box_y.value()
        if not self._check_box(x, y):
            return
        self.send_scene({"cmd": "set_box", "x": x, "y": y, "yaw_deg": self.box_yaw.value()})
        self.map.set_box((x, y), self.box_yaw.value())
        self.say(f"box -> ({x:.3f}, {y:.3f})")

    def apply_table(self):
        c, size, top = self._table()
        if not C.table_ok_before_seed(c, size) and not self.exec_status.get("seeded") and not self.allow_any.isChecked():
            self.say(f"not sent: the table's front edge ({C.table_front_edge(c, size):.2f} m) would block the arm's seed motion "
                     f"(needs >= {C.SEED_EDGE_X} m); press 'home / ready' first, or tick 'allow ...'")
            return
        self.send_scene({"cmd": "set_table", "x": c[0], "y": c[1], "top": top})
        self.table_top = top
        self.map.set_table(c, size, top)
        self.say(f"table -> centre ({c[0]:.3f}, {c[1]:.3f}) top {top:.3f}")
        self.apply_box()  # the box stays where it was: stand it on the new table

    def reset_scene(self):
        self.send_scene({"cmd": "reset"})
        for w, v in ((self.box_x, C.BOX_DEFAULT[0]), (self.box_y, C.BOX_DEFAULT[1]), (self.box_yaw, 0.0), (self.tab_x, C.TABLE_CENTER[0]),
                     (self.tab_y, C.TABLE_CENTER[1]), (self.tab_top, C.TABLE_TOP)):
            w.setValue(v)
        self.table_top = C.TABLE_TOP
        self.map.set_table(C.TABLE_CENTER, C.TABLE_SIZE, C.TABLE_TOP)
        self.map.set_box(None, 0.0)
        self.map.set_target(None)
        self.target_label.setText("place target: drag the blue square on the map")
        self.say("scene reset")

    def _set_box(self, x, y, final=True):
        self.box_x.setValue(x)
        self.box_y.setValue(y)
        self.send_scene({"cmd": "set_box", "x": x, "y": y, "yaw_deg": self.box_yaw.value()})
        self.map.set_box((x, y), self.box_yaw.value())
        if final:
            self.say(f"box -> ({x:.3f}, {y:.3f})")

    def _set_table(self, x, y, final=True):
        """Moves the table; a box standing on it comes along (a box in the robot's hand does not)."""
        dx, dy = x - self.tab_x.value(), y - self.tab_y.value()
        self.tab_x.setValue(x)
        self.tab_y.setValue(y)
        self.table_top = self.tab_top.value()
        self.send_scene({"cmd": "set_table", "x": x, "y": y, "top": self.table_top})
        self.map.set_table((x, y), C.TABLE_SIZE, self.table_top)
        if not self.exec_status.get("holding"):
            self._set_box(self.box_x.value() + dx, self.box_y.value() + dy, final=False)
        if final:
            self.say(f"table -> centre ({x:.3f}, {y:.3f}) top {self.table_top:.3f}")

    def _set_target(self, x, y, final=True):
        self.place_x.setValue(x)
        self.place_y.setValue(y)
        self.map.set_target((x, y))
        self.target_label.setText(f"place target: ({x:.3f}, {y:.3f}) m")
        if final:
            self.say(f"place target -> ({x:.3f}, {y:.3f})")

    def _on_dragged(self, item, x, y, final):
        {"box": self._set_box, "target": self._set_target, "table": self._set_table}[item](x, y, final)

    def _on_chosen(self, item, x, y):  # right-click menu
        {"box": self._set_box, "target": self._set_target, "table": self._set_table}[item](x, y, True)

    def _on_rotated(self, degrees):    # mouse wheel over the box
        yaw = min(30.0, max(-30.0, self.box_yaw.value() + degrees))
        self.box_yaw.setValue(yaw)
        self._set_box(self.box_x.value(), self.box_y.value(), final=False)
        self.say(f"box turned to {yaw:.0f} deg")

    def apply_target(self):
        self._set_target(self.place_x.value(), self.place_y.value())

    # ---- tasks -------------------------------------------------------------------------------------------------------
    def _target_ok(self):
        x, y = self.place_x.value(), self.place_y.value()
        self.map.set_target((x, y))
        return self._check_box(x, y)

    def _seed_ok(self):
        """The first move of the arm (seed pose) needs the table's front edge out of the way; later moves do not."""
        c, size, _ = self._table()
        if self.exec_status.get("seeded") or C.table_ok_before_seed(c, size) or self.allow_any.isChecked():
            return True
        self.say(f"not sent: the robot is not in its seed pose yet and the table's front edge ({C.table_front_edge(c, size):.2f} m) "
                 f"is in the way of that move (needs >= {C.SEED_EDGE_X} m): move the table back first, or tick 'allow ...'")
        return False

    def do_pick(self):
        if self._seed_ok():
            self.send_exec({"cmd": "pick", "use_image": self.use_image.isChecked()})

    def do_place(self):
        if self._target_ok():
            self.send_exec({"cmd": "place", "x": self.place_x.value(), "y": self.place_y.value(), "top": self.tab_top.value()})

    def do_pick_place(self):
        if self._seed_ok() and self._target_ok():
            self.send_exec({"cmd": "pick_place", "x": self.place_x.value(), "y": self.place_y.value(), "top": self.tab_top.value(),
                            "use_image": self.use_image.isChecked()})

    def do_home(self):
        c, size, _ = self._table()
        if C.table_ok_before_seed(c, size) or self.allow_any.isChecked():  # home goes through the seed motion again
            self.send_exec({"cmd": "home"})
        else:
            self.say(f"not sent: home repeats the seed motion, which the table's front edge ({C.table_front_edge(c, size):.2f} m) "
                     f"would block (needs >= {C.SEED_EDGE_X} m): move the table back first")

    # ---- periodic refresh --------------------------------------------------------------------------------------------
    def _refresh(self):
        snap = self.bridge.snapshot()
        for name, (arr, age) in snap["images"].items():
            if name in self.views and age < 5.0:
                self.views[name].show_array(arr, age)
        self._tick += 1
        if snap["box"] is not None and (snap["box_age"] or 0) < 3.0:
            self.map.set_box_live(snap["box"])
        self.map.set_other_boxes({c: p for c, p in snap.get("boxes", {}).items() if c != "blue"})
        if self._tick % 4 == 0 and snap["joints"]:
            self._fill_joints(snap["joints"])

    def _fill_joints(self, joints):
        names = list(joints)
        per_col = (len(names) + 2) // 3
        if names != self._joint_names:
            self._joint_names = names
            self.joints.setRowCount(per_col)
            for i, n in enumerate(names):
                col = (i // per_col) * 2
                item = QtWidgets.QTableWidgetItem(n)
                item.setForeground(QtGui.QBrush(QtGui.QColor("#7f8c8d")))
                self.joints.setItem(i % per_col, col, item)
        first = names != getattr(self, "_filled", None)
        for i, n in enumerate(names):
            col = (i // per_col) * 2 + 1
            unit_deg = "finger" not in n and "wheel" not in n
            text = f"{np.degrees(joints[n]):7.1f}°" if unit_deg else f"{joints[n] * 1000:7.1f} mm" if "finger" in n else f"{joints[n]:7.2f} rad"
            self.joints.setItem(i % per_col, col, QtWidgets.QTableWidgetItem(text))
        if first:
            self._filled = names
            self.joints.resizeColumnsToContents()   # once the values are in, so the columns fit them

    def closeEvent(self, ev):
        self.bridge.close()
        super().closeEvent(ev)


def main():
    from ros_bridge import RosBridge

    app = QtWidgets.QApplication(sys.argv)
    win = ControlCenter(RosBridge())
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
