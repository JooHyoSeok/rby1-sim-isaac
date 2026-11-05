"""Executor service for the control centre: a persistent process in the simulator container that takes commands over UDP and runs
them one at a time on the robot (grasp_core.py).

  docker cp examples/gui/{grasp_core,servo_server,scene_config}.py examples/vservo/{vision,servo}.py examples/gripper_example.py rby1-sim-isaac:/tmp/
  docker exec rby1-sim-isaac /isaac-sim/python.sh /tmp/servo_server.py        (examples/gui/run_gui.sh does all of it)

Commands (JSON datagrams to 127.0.0.1:5120); boxes are "blue", "green" or "yellow" ("blue" when no colour is given):
  {"cmd": "seed"}                                  torso squat + bent arm (done automatically before the first pick)
  {"cmd": "pick", "color": "green", "use_image": true, "bias": [0,0,0]}
  {"cmd": "place", "x": .., "y": .., "top": ..}    put the held box down at (x, y) on a table of height `top`
  {"cmd": "place_on", "base": "green"}             put the held box on top of the green box
  {"cmd": "pick_place", "color": .., "x": .., "y": .., "top": ..}
  {"cmd": "stack", "order": ["green", "blue", "yellow"]}   bottom first; the first box stays where it is
  {"cmd": "unstack"}                               take a stack apart: top box first onto free table spots (its start spot if free)
  {"cmd": "home"}                                  open the gripper and go back to the seed pose
  {"cmd": "gripper", "closeness": 0..1}
  {"cmd": "scene"}                                 reply only (see below): the boxes as the robot sees them, and its state
  {"cmd": "state"}                                 reply only: joint positions by name [rad], the right fingertips, status
  {"cmd": "joints", "trajectory": {"unit": "deg", "waypoints": [{"t": 2, "right_arm_1": -90}, ...]}}   a joint trajectory (examples/agent)
  {"cmd": "abort"}                                 stops the running task at once (also while it is moving)
Status goes to 127.0.0.1:5121: {"type": "status", "state": "idle"|"busy", "task": .., "phase": .., "holding": .., "held": .., "seeded": ..,
"tip": [x, y, z], "closeness": ..} twice a second and {"type": "log", "text": ..} for events and errors.
Any command may carry "reply": PORT (examples/gui/rbyctl.py does): the log lines of that job, then {"type": "result", "ok": bool, "text": ..}
go to 127.0.0.1:PORT as well, and "scene" is answered there as {"type": "scene", "boxes": .., "robot": .., "table": ..}.

pick/place/pick_place/home/stack/scene were run in the simulator; unstack is NOT RUN YET (see grasp_core.py): the
dispatch logic is covered by test_offline.py with a fake robot.
"""
import contextlib
import json
import queue
import socket
import threading
import time
import traceback

import scene_config as C
from grasp_core import Aborted, TaskError


class Server:
    def __init__(self, robot, cmd_port=C.PORT_EXEC_CMD, status_addr=("127.0.0.1", C.PORT_EXEC_STATUS)):
        self.robot = robot
        self.status_addr = status_addr
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", cmd_port))
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.jobs = queue.Queue()
        self.task = None
        self.reply_to = None      # (host, port) of the client whose job is running
        self.stop = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    # ---- output ------------------------------------------------------------------------------------------------------
    def send(self, msg):
        self.tx.sendto(json.dumps(msg).encode(), self.status_addr)

    def log(self, text):
        print(text, flush=True)
        self.send({"type": "log", "text": text})
        self.reply({"type": "log", "text": text})

    def reply(self, msg, to=None):
        to = to or self.reply_to
        if to:
            self.tx.sendto(json.dumps(msg).encode(), to)

    def _heartbeat(self):
        while not self.stop.is_set():
            try:
                st = self.robot.status()
            except Exception as exc:  # noqa: BLE001 - the heartbeat must survive a failing status read
                st = {"error": str(exc)}
            self.send({"type": "status", "state": "busy" if self.task else "idle", "task": self.task, **st})
            time.sleep(0.5)

    # ---- input -------------------------------------------------------------------------------------------------------
    def handle(self, msg):
        """Called for every datagram; abort acts immediately, everything else is queued (one task at a time)."""
        cmd, to = msg.get("cmd"), ("127.0.0.1", msg["reply"]) if "reply" in msg else None
        if cmd == "abort":
            self.robot.abort()
            self.log("abort requested")
            self.reply({"type": "result", "ok": True, "text": "abort requested"}, to)
        elif cmd in ("scene", "state"):
            try:
                self.reply({"type": cmd, **getattr(self.robot, cmd)()}, to)
            except Exception as exc:  # noqa: BLE001 - a failing query must not take the service down
                self.reply({"type": "result", "ok": False, "text": f"scene: {exc}"}, to)
        elif cmd in ("seed", "pick", "place", "place_on", "pick_place", "stack", "unstack", "joints", "home", "gripper"):
            if self.task and cmd != "gripper":
                self.log(f"busy with '{self.task}': {cmd} ignored (abort first)")
                self.reply({"type": "result", "ok": False, "text": f"busy with '{self.task}'"}, to)
            else:
                self.jobs.put(msg)
        else:
            self.log(f"unknown command: {msg}")

    def serve_forever(self):
        self.log("executor ready")
        while not self.stop.is_set():
            try:
                self.handle(json.loads(self.rx.recv(4096)))
            except ValueError:
                self.log("ignored a malformed command")

    # ---- worker ------------------------------------------------------------------------------------------------------
    def _worker(self):
        while not self.stop.is_set():
            try:
                msg = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            self.task = msg["cmd"]
            self.reply_to = ("127.0.0.1", msg["reply"]) if "reply" in msg else None
            self.log(f"start: {msg}")
            ok, text = False, ""
            try:
                with getattr(self.robot, "job", contextlib.nullcontext)():   # one job: an abort between its tasks is kept
                    self._dispatch(msg)
                ok, text = True, f"done: {msg['cmd']}"
            except Aborted:
                text = f"aborted: {msg['cmd']}"
            except TaskError as exc:
                text = f"failed: {msg['cmd']}: {exc}"
            except Exception:  # noqa: BLE001 - report anything unexpected to the user instead of dying
                text = f"error in {msg['cmd']}:\n{traceback.format_exc()}"
            finally:
                self.log(text)
                self.reply({"type": "result", "ok": ok, "text": text})
                self.reply_to = None
                self.task = None

    def _dispatch(self, msg):
        r, cmd = self.robot, msg["cmd"]
        if cmd == "seed":
            r.seed()
        elif cmd == "pick":
            r.pick(msg.get("color", "blue"), use_image=msg.get("use_image", True), bias=tuple(msg.get("bias", (0.0, 0.0, 0.0))))
        elif cmd == "place":
            r.place(msg["x"], msg["y"], msg.get("top", C.TABLE_TOP))
        elif cmd == "place_on":
            r.place_on(msg["base"])
        elif cmd == "pick_place":
            r.pick(msg.get("color", "blue"), use_image=msg.get("use_image", True), bias=tuple(msg.get("bias", (0.0, 0.0, 0.0))))
            r.place(msg["x"], msg["y"], msg.get("top", C.TABLE_TOP))
        elif cmd == "stack":
            r.stack(list(msg["order"]))
        elif cmd == "unstack":
            r.unstack()
        elif cmd == "joints":
            r.move_joints(msg["trajectory"])
        elif cmd == "home":
            r.home()
        elif cmd == "gripper":
            r.set_gripper(msg["closeness"])


if __name__ == "__main__":
    from grasp_core import GraspRobot, Perception

    perception = Perception()
    holder = {}
    robot = GraspRobot(perception, event=lambda text: holder["server"].log(text) if "server" in holder else print(text, flush=True))
    holder["server"] = server = Server(robot)
    server.serve_forever()
