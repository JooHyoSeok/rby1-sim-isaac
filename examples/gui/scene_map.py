"""Top-down map of the scene for the control centre: robot, table, box, place target and the region the arm can reach.

World frame: x forward (up on the screen), y to the left. Direct manipulation, no modes:
  drag the orange box / the blue place target / the table      (the cursor turns into a hand over things that can be dragged)
  mouse wheel over the box                                      turns it (5 degrees per notch)
  right click anywhere                                          menu: put the box / the place target / the table here
A dragged box or place target snaps to the nearest position where a box can stand and be reached (unless `allow_any`), so a drag
never ends somewhere the robot would fail; a dragged table is kept clear of the arm's seed motion when `table_min_edge` is set.
The widget only reports: `dragged(item, x, y, final)` (live while dragging, throttled, and once more on release), `chosen(item, x, y)`
(context menu) and `rotated(degrees)`.
"""
import math

from PyQt5 import QtCore, QtGui, QtWidgets

import scene_config as C

X_RANGE = (-0.35, 1.25)   # m, shown bottom to top
Y_RANGE = (-0.65, 0.65)   # m, shown right to left
GRID = 0.1
CELL = 0.025
HIT_MARGIN_PX = 10
LIVE_INTERVAL_MS = 50


class SceneMap(QtWidgets.QWidget):
    dragged = QtCore.pyqtSignal(str, float, float, bool)   # item ("box" | "target" | "table"), x, y, final
    chosen = QtCore.pyqtSignal(str, float, float)
    rotated = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(330, 300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.table_center, self.table_size, self.table_top = C.TABLE_CENTER, C.TABLE_SIZE, C.TABLE_TOP
        self.table_min_edge = None  # keep the table's front edge at x >= this (before the robot has made its seed move)
        self.allow_any = False
        self.box_live = None        # ground truth from the simulator (x, y, z)
        self.box_xy = None          # where the box was last asked to go (x, y)
        self.box_yaw = 0.0
        self.target = None          # place target (x, y)
        self.hover = None
        self.drag = None            # {"item", "offset", "pos", "raw"} while dragging
        self._last_emit = QtCore.QElapsedTimer()
        self._last_emit.start()
        self._reach_key, self._reach_cells = None, []

    # ---- state -------------------------------------------------------------------------------------------------------
    def set_table(self, center, size, top):
        self.table_center, self.table_size, self.table_top = tuple(center), tuple(size), float(top)
        self.update()

    def set_box_live(self, xyz):
        self.box_live = xyz
        self.update()

    def set_box(self, xy, yaw=None):
        self.box_xy = xy
        if yaw is not None:
            self.box_yaw = yaw
        self.update()

    def set_target(self, xy):
        self.target = xy
        self.update()

    def set_allow_any(self, on):
        self.allow_any = bool(on)

    def set_table_min_edge(self, x):
        self.table_min_edge = x

    # ---- geometry ----------------------------------------------------------------------------------------------------
    def _scale(self):
        return min(self.width() / (Y_RANGE[1] - Y_RANGE[0]), self.height() / (X_RANGE[1] - X_RANGE[0]))

    def _origin(self):
        s = self._scale()
        return (self.width() - s * (Y_RANGE[1] - Y_RANGE[0])) / 2, (self.height() - s * (X_RANGE[1] - X_RANGE[0])) / 2

    def to_px(self, x, y):
        s, (ox, oy) = self._scale(), self._origin()
        return QtCore.QPointF(ox + s * (Y_RANGE[1] - y), oy + s * (X_RANGE[1] - x))

    def to_world(self, px, py):
        s, (ox, oy) = self._scale(), self._origin()
        return X_RANGE[1] - (py - oy) / s, Y_RANGE[1] - (px - ox) / s

    def _rect(self, cx, cy, sx, sy):
        return QtCore.QRectF(self.to_px(cx + sx / 2, cy + sy / 2), self.to_px(cx - sx / 2, cy - sy / 2)).normalized()

    def reach_cells(self):
        key = (self.table_center, self.table_size, round(self.table_top, 3))
        if key != self._reach_key:
            self._reach_cells = [(x, y) for i in range(int((X_RANGE[1] - X_RANGE[0]) / CELL)) for j in range(int((Y_RANGE[1] - Y_RANGE[0]) / CELL))
                                 for x, y in [(X_RANGE[0] + (i + 0.5) * CELL, Y_RANGE[0] + (j + 0.5) * CELL)]   # cell centres
                                 if C.check_pose(x, y, self.table_top, self.table_center, self.table_size) is None]
            self._reach_key = key
        return self._reach_cells

    def valid(self, x, y):
        return C.check_pose(x, y, self.table_top, self.table_center, self.table_size) is None

    def snap(self, x, y):
        """The closest position where a box can stand and be reached (the point itself when that is already so)."""
        if self.allow_any or self.valid(x, y):
            return x, y
        cells = self.reach_cells()
        if not cells:
            return x, y
        return min(cells, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2)

    def constrain(self, item, x, y):
        if item in ("box", "target"):
            return self.snap(x, y)
        if self.table_min_edge is not None and not self.allow_any:     # table: keep the front edge clear of the seed motion
            x = max(x, self.table_min_edge + self.table_size[0] / 2)
        return min(max(x, X_RANGE[0]), X_RANGE[1]), min(max(y, Y_RANGE[0]), Y_RANGE[1])

    # ---- what is under the pointer -----------------------------------------------------------------------------------
    def _box_pos(self):
        if self.box_live is not None:
            return self.box_live[0], self.box_live[1]
        return self.box_xy

    def _item_rect(self, item):
        if item == "box" and self._box_pos() is not None:
            return self._rect(*self._box_pos(), *C.BOX_SIZE[:2])
        if item == "target" and self.target is not None:
            return self._rect(*self.target, *C.BOX_SIZE[:2])
        if item == "table":
            return self._rect(*self.table_center, *self.table_size)
        return None

    def hit(self, px, py):
        for item in ("box", "target", "table"):      # small things first: they sit on top of the table
            r = self._item_rect(item)
            if r is not None and r.adjusted(-HIT_MARGIN_PX, -HIT_MARGIN_PX, HIT_MARGIN_PX, HIT_MARGIN_PX).contains(px, py):
                return item
        return None

    def _item_centre(self, item):
        return {"box": self._box_pos(), "target": self.target, "table": self.table_center}[item]

    # ---- drawing -----------------------------------------------------------------------------------------------------
    def _draw_box(self, p, cx, cy, yaw, pen, brush):
        p.save()
        c = self.to_px(cx, cy)
        p.translate(c)
        p.rotate(-yaw)  # x up / y left on the screen: a positive yaw turns counterclockwise
        s = self._scale()
        p.setPen(pen)
        p.setBrush(brush)
        p.drawRect(QtCore.QRectF(-C.BOX_SIZE[1] * s / 2, -C.BOX_SIZE[0] * s / 2, C.BOX_SIZE[1] * s, C.BOX_SIZE[0] * s))
        p.restore()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(38, 40, 44))
        p.setPen(QtGui.QPen(QtGui.QColor(70, 73, 80), 1))
        x = X_RANGE[0] - X_RANGE[0] % GRID + GRID
        while x < X_RANGE[1]:
            p.drawLine(self.to_px(x, Y_RANGE[0]), self.to_px(x, Y_RANGE[1]))
            x += GRID
        y = Y_RANGE[0] - Y_RANGE[0] % GRID + GRID
        while y < Y_RANGE[1]:
            p.drawLine(self.to_px(X_RANGE[0], y), self.to_px(X_RANGE[1], y))
            y += GRID
        # the table (draggable) and the region where a box can stand and be reached
        dragging_table = self.drag is not None and self.drag["item"] == "table"
        tc = self.drag["pos"] if dragging_table else self.table_center
        p.setPen(QtGui.QPen(QtGui.QColor(190, 150, 105) if dragging_table else QtGui.QColor(150, 120, 90), 2))
        p.setBrush(QtGui.QColor(120, 95, 70, 120))
        p.drawRect(self._rect(*tc, *self.table_size))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(70, 190, 110, 90))
        cell = self._scale() * CELL
        for cx, cy in self.reach_cells():
            q = self.to_px(cx, cy)
            p.drawRect(QtCore.QRectF(q.x() - cell / 2, q.y() - cell / 2, cell + 0.6, cell + 0.6))
        # the robot
        p.setPen(QtGui.QPen(QtGui.QColor(160, 165, 175), 2))
        p.setBrush(QtGui.QColor(90, 95, 105))
        p.drawRect(self._rect(0.0, 0.0, 0.58, 0.50))
        p.setBrush(QtGui.QColor(230, 230, 235))
        p.drawEllipse(self.to_px(C.SHOULDER[0], C.SHOULDER[1]), 5, 5)
        p.setPen(QtGui.QColor(200, 205, 215))
        p.drawText(self.to_px(0.05, 0.22), "robot")
        # the place target
        if self.target is not None:
            t = self.drag["pos"] if self.drag is not None and self.drag["item"] == "target" else self.target
            p.setPen(QtGui.QPen(QtGui.QColor(90, 170, 255), 2, QtCore.Qt.DashLine))
            p.setBrush(QtGui.QColor(90, 170, 255, 40))
            r = self._rect(*t, *C.BOX_SIZE[:2])
            p.drawRect(r)
            p.drawLine(r.center() - QtCore.QPointF(9, 0), r.center() + QtCore.QPointF(9, 0))
            p.drawLine(r.center() - QtCore.QPointF(0, 9), r.center() + QtCore.QPointF(0, 9))
            p.setPen(QtGui.QColor(140, 195, 255))
            p.drawText(r.bottomRight() + QtCore.QPointF(4, 12), "place here")
        # the box: the simulator's actual position, and the dragged one while a drag is going on
        bp = self._box_pos()
        if bp is not None:
            self._draw_box(p, bp[0], bp[1], self.box_yaw, QtGui.QPen(QtGui.QColor(255, 160, 40), 2), QtGui.QColor(230, 130, 20))
            if self.box_live is not None:
                p.setPen(QtGui.QColor(255, 200, 120))
                p.drawText(self.to_px(bp[0] + 0.06, bp[1] - 0.045), f"z {self.box_live[2]:.2f}")
        if self.drag is not None and self.drag["item"] == "box":
            gx, gy = self.drag["pos"]
            self._draw_box(p, gx, gy, self.box_yaw, QtGui.QPen(QtGui.QColor(255, 225, 90), 2), QtGui.QColor(255, 225, 90, 90))
            rx, ry = self.drag["raw"]
            if math.hypot(rx - gx, ry - gy) > 0.01:   # the pointer is somewhere the box cannot go: show where it snapped from
                p.setPen(QtGui.QPen(QtGui.QColor(235, 90, 90), 1, QtCore.Qt.DotLine))
                p.drawLine(self.to_px(rx, ry), self.to_px(gx, gy))
        # hints
        p.setPen(QtGui.QColor(200, 205, 215))
        p.drawText(8, 16, "drag the box / blue target / table   wheel on the box = turn   right click = menu")
        p.drawText(8, 32, "green: a box can stand and be reached here")
        if self.hover is not None:
            p.drawText(8, self.height() - 8, f"x {self.hover[0]:.3f}  y {self.hover[1]:.3f} m")

    # ---- mouse -------------------------------------------------------------------------------------------------------
    def mousePressEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        item = self.hit(ev.x(), ev.y())
        if item is None:
            return
        cx, cy = self._item_centre(item)
        wx, wy = self.to_world(ev.x(), ev.y())
        self.drag = {"item": item, "offset": (cx - wx, cy - wy), "pos": (cx, cy), "raw": (cx, cy)}
        self.setCursor(QtCore.Qt.ClosedHandCursor)
        self.update()

    def mouseMoveEvent(self, ev):
        self.hover = self.to_world(ev.x(), ev.y())
        if self.drag is None:
            self.setCursor(QtCore.Qt.OpenHandCursor if self.hit(ev.x(), ev.y()) else QtCore.Qt.ArrowCursor)
        else:
            d = self.drag
            raw = (self.hover[0] + d["offset"][0], self.hover[1] + d["offset"][1])
            d["raw"], d["pos"] = raw, self.constrain(d["item"], *raw)
            if self._last_emit.elapsed() >= LIVE_INTERVAL_MS:
                self._last_emit.restart()
                self.dragged.emit(d["item"], d["pos"][0], d["pos"][1], False)
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self.drag is not None:
            d, self.drag = self.drag, None
            self.dragged.emit(d["item"], d["pos"][0], d["pos"][1], True)
            self.setCursor(QtCore.Qt.OpenHandCursor if self.hit(ev.x(), ev.y()) else QtCore.Qt.ArrowCursor)
            self.update()

    def wheelEvent(self, ev):
        if self.hit(ev.pos().x(), ev.pos().y()) == "box":
            self.rotated.emit(5.0 * ev.angleDelta().y() / 120.0)
            ev.accept()
        else:
            ev.ignore()

    def contextMenuEvent(self, ev):
        x, y = self.to_world(ev.pos().x(), ev.pos().y())
        menu = QtWidgets.QMenu(self)
        for item, text in (("box", "put the box here"), ("target", "put the place target here"), ("table", "move the table here")):
            menu.addAction(text).setData(item)
        picked = menu.exec_(ev.globalPos())
        if picked is not None:
            item = picked.data()
            x, y = self.constrain(item, x, y)
            self.chosen.emit(item, x, y)
