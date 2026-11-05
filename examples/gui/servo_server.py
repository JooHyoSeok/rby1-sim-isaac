"""Executor service for the control centre: a persistent process in the simulator container that takes commands over UDP and runs
them one at a time on the robot (grasp_core.py).

  docker cp examples/gui/{grasp_core,servo_server,scene_config}.py examples/vservo/{vision,servo}.py examples/gripper_example.py rby1-sim-isaac:/tmp/
  docker exec rby1-sim-isaac /isaac-sim/python.sh /tmp/servo_server.py        (examples/gui/run_gui.sh does all of it)

Commands (JSON datagrams to 127.0.0.1:5120):
  {"cmd": "seed"}                                  torso squat + bent arm (done automatically before the first pick)
  {"cmd": "pick", "use_image": true, "bias": [0,0,0]}
  {"cmd": "place", "x": .., "y": .., "top": ..}    put the held box down at (x, y) on a table of height `top`
  {"cmd": "pick_place", "x": .., "y": .., "top": ..}
  {"cmd": "home"}                                  open the gripper and go back to the seed pose
  {"cmd": "gripper", "closeness": 0..1}
  {"cmd": "abort"}                                 stops the running task at once (also while it is moving)
Status goes to 127.0.0.1:5121: {"type": "status", "state": "idle"|"busy", "task": .., "phase": .., "holding": .., "seeded": ..,
"tip": [x, y, z], "closeness": ..} twice a second and {"type": "log", "text": ..} for events and errors.

NOT RUN YET (see grasp_core.py): the dispatch logic is covered by test_offline.py with a fake robot.
"""
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
        self.stop = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    # ---- output ------------------------------------------------------------------------------------------------------
    def send(self, msg):
        self.tx.sendto(json.dumps(msg).encode(), self.status_addr)

    def log(self, text):
        print(text, flush=True)
        self.send({"type": "log", "text": text})

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
        cmd = msg.get("cmd")
        if cmd == "abort":
            self.robot.abort()
            self.log("abort requested")
        elif cmd in ("seed", "pick", "place", "pick_place", "home", "gripper"):
            if self.task and cmd != "gripper":
                self.log(f"busy with '{self.task}': {cmd} ignored (abort first)")
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
            self.log(f"start: {msg}")
            try:
                self._dispatch(msg)
                self.log(f"done: {msg['cmd']}")
            except Aborted:
                self.log(f"aborted: {msg['cmd']}")
            except TaskError as exc:
                self.log(f"failed: {msg['cmd']}: {exc}")
            except Exception:  # noqa: BLE001 - report anything unexpected to the user instead of dying
                self.log(f"error in {msg['cmd']}:\n{traceback.format_exc()}")
            finally:
                self.task = None

    def _dispatch(self, msg):
        r, cmd = self.robot, msg["cmd"]
        if cmd == "seed":
            r.seed()
        elif cmd == "pick":
            r.pick(use_image=msg.get("use_image", True), bias=tuple(msg.get("bias", (0.0, 0.0, 0.0))))
        elif cmd == "place":
            r.place(msg["x"], msg["y"], msg.get("top", C.TABLE_TOP))
        elif cmd == "pick_place":
            r.pick(use_image=msg.get("use_image", True), bias=tuple(msg.get("bias", (0.0, 0.0, 0.0))))
            r.place(msg["x"], msg["y"], msg.get("top", C.TABLE_TOP))
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
