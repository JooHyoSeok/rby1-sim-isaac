"""ROS 2 side of the control centre (host, system python with rclpy): the latest images, joint states and ground-truth poses.

A background thread spins the node; the GUI reads `snapshot()` from its timer, so no Qt object is touched from the ROS thread.
Needs `source /opt/ros/jazzy/setup.bash` and FASTDDS_BUILTIN_TRANSPORTS=UDPv4 (the sim container's DDS shared memory does not
reach the host).
"""
import threading
import time

import numpy as np

IMAGE_TOPICS = {   # name -> topic
    "wrist overlay": "/rby1/servo_debug/wrist_right",
    "stereo overlay": "/rby1/servo_debug/stereo_right",
    "wrist left": "/rby1/wrist_left/image_raw",
    "stereo left": "/rby1/stereo_left/image_raw",
}


class RosBridge:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, JointState
        from tf2_msgs.msg import TFMessage

        self._rclpy = rclpy
        rclpy.init()
        self.node = Node("control_center")
        self._lock = threading.Lock()
        self._images, self._stamps = {}, {}
        self._joints, self._joints_t = {}, 0.0
        self._box, self._box_t = None, 0.0
        for name, topic in IMAGE_TOPICS.items():
            self.node.create_subscription(Image, topic, lambda m, name=name: self._on_image(name, m), 2)
        self.node.create_subscription(JointState, "/rby1/joint_states", self._on_joints, 5)
        self.node.create_subscription(TFMessage, "/rby1/gt/tf", self._on_gt, 5)
        self._stop = threading.Event()
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        while not self._stop.is_set() and self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.05)

    def _on_image(self, name, m):
        if m.encoding not in ("rgb8", "bgr8"):
            return
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == "bgr8":
            img = img[:, :, ::-1]
        with self._lock:
            self._images[name] = img.copy()
            self._stamps[name] = time.time()

    def _on_joints(self, m):
        with self._lock:
            self._joints, self._joints_t = dict(zip(m.name, m.position)), time.time()

    def _on_gt(self, m):
        for t in m.transforms:
            if t.child_frame_id == "Object":
                with self._lock:
                    self._box = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
                    self._box_t = time.time()

    def snapshot(self):
        """{'images': {name: (rgb array, age s)}, 'joints': {name: rad} or {}, 'joints_age', 'box': (x, y, z) or None, 'box_age'}"""
        now = time.time()
        with self._lock:
            return {"images": {n: (img, now - self._stamps[n]) for n, img in self._images.items()},
                    "joints": dict(self._joints), "joints_age": now - self._joints_t if self._joints_t else None,
                    "box": self._box, "box_age": now - self._box_t if self._box_t else None}

    def close(self):
        self._stop.set()
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
