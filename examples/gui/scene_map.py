"""Top-down map of the scene for the control centre: robot, table, box, place target and the region the arm can reach.

World frame: x forward (up on the screen), y to the left. Click or drag to place things; the mode decides what moves:
  "box"    where the box should stand        "target"  where a held box should be put down        "table"  the table centre
Nothing is sent from here: the widget emits `placed(mode, x, y)` when the mouse is released.
"""
from PyQt5 import QtCore, QtGui, QtWidgets

import scene_config as C

X_RANGE = (-0.35, 1.25)   # m, shown bottom to top
Y_RANGE = (-0.65, 0.65)   # m, shown right to left
GRID = 0.1


class SceneMap(QtWidgets.QWidget):
    placed = QtCore.pyqtSignal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(330, 300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.mode = "box"
        self.table_center, self.table_size, self.table_top = C.TABLE_CENTER, C.TABLE_SIZE, C.TABLE_TOP
        self.box_live = None       # ground truth from the simulator (x, y, z)
        self.box_request = None    # where the box was asked to go (x, y)
        self.target = None         # place target (x, y)
        self.ghost = None          # (mode, x, y) while dragging
        self.hover = None
        self._reach_key, self._reach_cells = None, []

    # ---- state -------------------------------------------------------------------------------------------------------
    def set_mode(self, mode):
        self.mode = mode
        self.ghost = None
        self.update()

    def set_table(self, center, size, top):
        self.table_center, self.table_size, self.table_top = tuple(center), tuple(size), float(top)
        self.update()

    def set_box_live(self, xyz):
        self.box_live = xyz
        self.update()

    def set_box_request(self, xy):
        self.box_request = xy
        self.update()

    def set_target(self, xy):
        self.target = xy
        self.update()

    # ---- geometry ----------------------------------------------------------------------------------------------------
    def _scale(self):
        return min(self.width() / (Y_RANGE[1] - Y_RANGE[0]), self.height() / (X_RANGE[1] - X_RANGE[0]))

    def to_px(self, x, y):
        s = self._scale()
        ox = (self.width() - s * (Y_RANGE[1] - Y_RANGE[0])) / 2
        oy = (self.height() - s * (X_RANGE[1] - X_RANGE[0])) / 2
        return QtCore.QPointF(ox + s * (Y_RANGE[1] - y), oy + s * (X_RANGE[1] - x))

    def to_world(self, px, py):
        s = self._scale()
        ox = (self.width() - s * (Y_RANGE[1] - Y_RANGE[0])) / 2
        oy = (self.height() - s * (X_RANGE[1] - X_RANGE[0])) / 2
        return X_RANGE[1] - (py - oy) / s, Y_RANGE[1] - (px - ox) / s

    def _rect(self, cx, cy, sx, sy):
        a, b = self.to_px(cx + sx / 2, cy + sy / 2), self.to_px(cx - sx / 2, cy - sy / 2)
        return QtCore.QRectF(a, b).normalized()

    def reach_cells(self):
        key = (self.table_center, self.table_size, round(self.table_top, 3))
        if key != self._reach_key:
            step = 0.025
            self._reach_cells = [(x, y) for i in range(int((X_RANGE[1] - X_RANGE[0]) / step)) for j in range(int((Y_RANGE[1] - Y_RANGE[0]) / step))
                                 for x, y in [(X_RANGE[0] + i * step, Y_RANGE[0] + j * step)]
                                 if C.check_pose(x, y, self.table_top, self.table_center, self.table_size) is None]
            self._reach_key = key
        return self._reach_cells

    # ---- drawing -----------------------------------------------------------------------------------------------------
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(38, 40, 44))
        grid = QtGui.QPen(QtGui.QColor(70, 73, 80), 1)
        p.setPen(grid)
        x = X_RANGE[0] - X_RANGE[0] % GRID + GRID
        while x < X_RANGE[1]:
            a, b = self.to_px(x, Y_RANGE[0]), self.to_px(x, Y_RANGE[1])
            p.drawLine(a, b)
            x += GRID
        y = Y_RANGE[0] - Y_RANGE[0] % GRID + GRID
        while y < Y_RANGE[1]:
            p.drawLine(self.to_px(X_RANGE[0], y), self.to_px(X_RANGE[1], y))
            y += GRID
        # table, with the region where a box can stand and be reached
        p.setPen(QtGui.QPen(QtGui.QColor(150, 120, 90), 2))
        p.setBrush(QtGui.QColor(120, 95, 70, 120))
        p.drawRect(self._rect(*self.table_center, *self.table_size))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(70, 190, 110, 90))
        cell = self._scale() * 0.025
        for cx, cy in self.reach_cells():
            q = self.to_px(cx + 0.0125, cy + 0.0125)
            p.drawRect(QtCore.QRectF(q.x() - cell / 2, q.y() - cell / 2, cell + 0.6, cell + 0.6))
        # robot: base footprint, shoulder
        p.setPen(QtGui.QPen(QtGui.QColor(160, 165, 175), 2))
        p.setBrush(QtGui.QColor(90, 95, 105))
        p.drawRect(self._rect(0.0, 0.0, 0.58, 0.50))
        sh = self.to_px(C.SHOULDER[0], C.SHOULDER[1])
        p.setBrush(QtGui.QColor(230, 230, 235))
        p.drawEllipse(sh, 5, 5)
        p.setPen(QtGui.QColor(200, 205, 215))
        p.drawText(self.to_px(0.05, 0.22), "robot")
        # box (live ground truth), requested box, place target
        if self.box_live is not None:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 160, 40), 2))
            p.setBrush(QtGui.QColor(230, 130, 20))
            p.drawRect(self._rect(self.box_live[0], self.box_live[1], C.BOX_SIZE[0], C.BOX_SIZE[1]))
            p.setPen(QtGui.QColor(255, 200, 120))
            p.drawText(self.to_px(self.box_live[0] + 0.05, self.box_live[1] - 0.04), f"z {self.box_live[2]:.2f}")
        dashed = QtGui.QPen(QtGui.QColor(255, 160, 40), 2, QtCore.Qt.DashLine)
        if self.box_request is not None:
            p.setPen(dashed)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(self._rect(*self.box_request, C.BOX_SIZE[0], C.BOX_SIZE[1]))
        if self.target is not None:
            p.setPen(QtGui.QPen(QtGui.QColor(90, 170, 255), 2, QtCore.Qt.DashLine))
            p.setBrush(QtCore.Qt.NoBrush)
            r = self._rect(*self.target, C.BOX_SIZE[0], C.BOX_SIZE[1])
            p.drawRect(r)
            p.drawLine(r.center() - QtCore.QPointF(8, 0), r.center() + QtCore.QPointF(8, 0))
            p.drawLine(r.center() - QtCore.QPointF(0, 8), r.center() + QtCore.QPointF(0, 8))
        if self.ghost is not None:
            mode, gx, gy = self.ghost
            bad = mode != "table" and C.check_pose(gx, gy, self.table_top, self.table_center, self.table_size) is not None
            col = QtGui.QColor(235, 70, 70) if bad else (QtGui.QColor(90, 170, 255) if mode == "target" else QtGui.QColor(255, 220, 80))
            p.setPen(QtGui.QPen(col, 2))
            p.setBrush(QtGui.QColor(col.red(), col.green(), col.blue(), 70))
            p.drawRect(self._rect(gx, gy, *(self.table_size if mode == "table" else C.BOX_SIZE[:2])))
        # legend and the pointer position
        p.setPen(QtGui.QColor(200, 205, 215))
        p.drawText(8, 16, "green: a box can stand and be reached here")
        if self.hover is not None:
            p.drawText(8, self.height() - 8, f"x {self.hover[0]:.3f}  y {self.hover[1]:.3f} m   [{self.mode}]")

    # ---- mouse -------------------------------------------------------------------------------------------------------
    def _world(self, ev):
        return self.to_world(ev.x(), ev.y())

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            x, y = self._world(ev)
            self.ghost = (self.mode, x, y)
            self.update()
        elif ev.button() == QtCore.Qt.RightButton:
            self.ghost = None
            self.update()

    def mouseMoveEvent(self, ev):
        self.hover = self._world(ev)
        if self.ghost is not None:
            self.ghost = (self.mode, *self.hover)
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self.ghost is not None:
            mode, x, y = self.ghost
            self.ghost = None
            self.placed.emit(mode, x, y)
            self.update()
