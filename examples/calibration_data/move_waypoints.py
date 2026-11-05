"""Move both arms through random waypoints and ask record_snapshots.py to save a snapshot at each one.

Runs inside the simulator container (rby1_sdk is installed only there):
  docker cp move_waypoints.py rby1-sim-isaac:/tmp/ && docker exec rby1-sim-isaac /isaac-sim/python.sh /tmp/move_waypoints.py -n 30
Needs the sim started with --vision via docker/run_sdk.sh and record_snapshots.py already running.
"""
import argparse
import socket
import time

import numpy as np
import rby1_sdk as rby

p = argparse.ArgumentParser()
p.add_argument("-n", type=int, default=30, help="number of waypoints")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--move-time", type=float, default=2.5)
p.add_argument("--settle", type=float, default=0.7, help="wait after the arms stop [s]")
p.add_argument("--recorder", default="127.0.0.1:5101")
args = p.parse_args()

robot = rby.create_robot_a("127.0.0.1:50051")
assert robot.connect(), "cannot reach app_main (is run_sdk.sh running?)"
robot.power_on(".*")
robot.servo_on(".*")
robot.reset_fault_control_manager()
assert robot.enable_control_manager()

RIGHT, LEFT = slice(8, 15), slice(15, 22)  # arm joints in the SDK joint order
q0 = np.array(robot.get_state().position)
rng = np.random.default_rng(args.seed)


def waypoint(sign):
    """Arm joint offsets; sign mirrors the shoulder so both arms reach toward the table."""
    d = np.array([rng.uniform(-0.6, 0.6), sign * rng.uniform(0.1, 0.6), rng.uniform(-0.4, 0.4),
                  -rng.uniform(0.5, 1.6), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.8, 0.8)])
    return d


def move(right_q, left_q):
    body = rby.BodyComponentBasedCommandBuilder()
    for setter, q in ((body.set_right_arm_command, right_q), (body.set_left_arm_command, left_q)):
        setter(rby.ArmCommandBuilder().set_command(
            rby.JointPositionCommandBuilder().set_minimum_time(args.move_time).set_position(q)))
    rc = rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(body))
    return robot.send_command(rc, 10).get().finish_code


host, port = args.recorder.split(":")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(8.0)
for i in range(args.n):
    tr, tl = q0[RIGHT] + waypoint(-1), q0[LEFT] + waypoint(+1)
    code = move(tr, tl)
    time.sleep(args.settle)
    q = np.array(robot.get_state().position)
    err = max(np.abs(q[RIGHT] - tr).max(), np.abs(q[LEFT] - tl).max())
    sock.sendto(f"SAVE {i}".encode(), (host, int(port)))
    try:
        reply = sock.recvfrom(256)[0].decode()
    except socket.timeout:
        reply = "NO REPLY"
    print(f"[{i}] finish={code} max_joint_err={err:.4f} rad  recorder: {reply}")
move(q0[RIGHT], q0[LEFT])
