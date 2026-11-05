#!/usr/bin/env bash
# One command: simulator + perception node + executor service + the control centre window. Closing the window stops everything.
#
#   examples/gui/run_gui.sh [options]
#     --rviz            also open RViz (robot model, camera frames, table and box)
#     --keep            leave the simulator running after the window is closed
#     --no-window       no control centre: start the rest and wait (Ctrl-C stops everything); drive it with examples/gui/rbyctl.py
#     --cpuset CPUS     pin the simulator to these cores (default: the last 6; "" = no pinning)
#     --image TAG       simulator image tag (default 0.10.7-a_v1.2-isaac6.1.0-r1)
#
# Needs the ROS 2 container (intelligent-robotic-manipulation) running and the sim image present. The simulator takes ~2 minutes to
# come up. In the window: set the box / table, press "pick" (the robot first goes to its seed pose), then "place" or "pick and place".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROS=intelligent-robotic-manipulation
SIM=rby1-sim-isaac
IMAGE=0.10.7-a_v1.2-isaac6.1.0-r1
CORES="$(nproc)"
CPUSET="$((CORES - 6))-$((CORES - 1))"
RVIZ=0; KEEP=0; WINDOW=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rviz) RVIZ=1; shift ;;
        --keep) KEEP=1; shift ;;
        --no-window) WINDOW=0; shift ;;
        --cpuset) CPUSET="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

WORK="$(mktemp -d /tmp/rby1_gui.XXXXXX)"
say() { printf '\n== %s\n' "$*"; }
SIM_PID=""
VIEWER_GROUPS=()   # process groups of the viewers this script started (killed by group, never by name pattern)

cleanup() {
    set +e
    for g in "${VIEWER_GROUPS[@]:-}"; do [[ -n "${g}" ]] && kill -- "-${g}" 2>/dev/null; done
    if [[ "${KEEP}" == "1" ]]; then
        say "left running (--keep). Stop with:  docker stop ${SIM}; docker exec ${ROS} pkill -f perception_node.py"
        return
    fi
    say "cleaning up"
    docker exec "${ROS}" pkill -f perception_node.py 2>/dev/null
    docker stop "${SIM}" >/dev/null 2>&1
    [[ -n "${SIM_PID}" ]] && kill "${SIM_PID}" 2>/dev/null
    sleep 1
    for g in "${VIEWER_GROUPS[@]:-}"; do [[ -n "${g}" ]] && kill -9 -- "-${g}" 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT

docker image inspect "rainbowroboticsofficial/rby1-sim-isaac:${IMAGE}" >/dev/null 2>&1 || { echo "image ${IMAGE} not found" >&2; exit 1; }
[[ "$(docker inspect -f '{{.State.Running}}' "${ROS}" 2>/dev/null)" == "true" ]] || { echo "container ${ROS} is not running" >&2; exit 1; }
for m in PyQt5 numpy rclpy; do
    ( set +u; source /opt/ros/jazzy/setup.bash; /usr/bin/python3 -c "import ${m}" ) 2>/dev/null || { echo "/usr/bin/python3 cannot import ${m}" >&2; exit 1; }
done

say "1/4 starting the simulator (headless, cores ${CPUSET:-all}); log: ${WORK}/sim.log"
docker rm -f "${SIM}" >/dev/null 2>&1 || true
( cd "${REPO}" && HEADLESS=1 CPUSET="${CPUSET}" ./docker/run_sdk.sh --image "${IMAGE}" --gripper --gripper-name rb_gripper --vision ) \
    > "${WORK}/sim.log" 2>&1 &
SIM_PID=$!
for ((i = 0; i < 300; i++)); do
    grep -q "wall_dt=" "${WORK}/sim.log" 2>/dev/null && break   # the simulation loop is running
    kill -0 "${SIM_PID}" 2>/dev/null || { echo "simulator exited:" >&2; tail -15 "${WORK}/sim.log" >&2; exit 1; }
    sleep 1
done
grep -q "wall_dt=" "${WORK}/sim.log" || { echo "simulator not ready after 300 s" >&2; exit 1; }
sleep 3
echo "simulator ready"

say "2/4 perception node (ROS 2 container)"
for f in vision.py perception_node.py; do docker cp "${HERE}/../vservo/${f}" "${ROS}:/tmp/${f}"; done
docker cp "${HOME}/docker/isaac-sim/data/rby1_gt.json" "${ROS}:/tmp/calib.json"   # stereo extrinsics, rewritten at every simulator start
docker exec -d "${ROS}" bash -c "source /opt/ros/jazzy/setup.bash; export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; cd /tmp; python3 perception_node.py > /tmp/perception.log 2>&1"
for ((i = 0; i < 30; i++)); do
    docker exec "${ROS}" grep -q "perception running" /tmp/perception.log 2>/dev/null && break
    sleep 1
done
docker exec "${ROS}" grep -q "perception running" /tmp/perception.log || { echo "perception node failed:" >&2; docker exec "${ROS}" tail -8 /tmp/perception.log >&2; exit 1; }
echo "perception running"

say "3/4 executor service (simulator container)"
for f in vision.py servo.py; do docker cp "${HERE}/../vservo/${f}" "${SIM}:/tmp/${f}"; done
for f in scene_config.py base_motion.py grasp_core.py servo_server.py; do docker cp "${HERE}/${f}" "${SIM}:/tmp/${f}"; done
for f in kinematics.py safety.py rby1_kinematics.json; do docker cp "${HERE}/../agent/${f}" "${SIM}:/tmp/${f}"; done   # the joint-trajectory check
docker cp "${REPO}/examples/gripper_example.py" "${SIM}:/tmp/gripper_example.py"
docker exec -d "${SIM}" bash -c "/isaac-sim/python.sh /tmp/servo_server.py > /tmp/server.log 2>&1"
for ((i = 0; i < 60; i++)); do
    docker exec "${SIM}" grep -q "executor ready" /tmp/server.log 2>/dev/null && break
    sleep 1
done
docker exec "${SIM}" grep -q "executor ready" /tmp/server.log || { echo "executor failed:" >&2; docker exec "${SIM}" tail -15 /tmp/server.log >&2; exit 1; }
echo "executor ready"

if [[ "${RVIZ}" == "1" ]]; then
    export DISPLAY="${DISPLAY:-:1}"
    if [[ -f "${HOME}/docker/isaac-sim/data/rby1_urdf/rsp_params.yaml" ]]; then
        setsid bash -c "set +u; source /opt/ros/jazzy/setup.bash; export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; \
            exec /usr/bin/python3 '${HERE}/../rviz/object_marker.py'" > "${WORK}/marker.log" 2>&1 &
        VIEWER_GROUPS+=($!)
        setsid "${HERE}/../rviz/rby1_model.sh" view > "${WORK}/rviz.log" 2>&1 &
        VIEWER_GROUPS+=($!)
    else
        echo "no URDF yet (run examples/rviz/rby1_model.sh export once): skipping RViz"
    fi
fi

if [[ "${WINDOW}" == "0" ]]; then
    say "4/4 no window: ready. Drive it with examples/gui/rbyctl.py; Ctrl-C (or killing this script) stops everything"
    trap 'exit 0' INT TERM      # the EXIT trap (cleanup) runs; `wait` lets the signal through at once
    while true; do sleep 3600 & wait $!; done
fi

say "4/4 control centre (close the window to stop everything)"
export DISPLAY="${DISPLAY:-:1}"
set +e
( set +u; source /opt/ros/jazzy/setup.bash; set -u; export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; cd "${HERE}"; exec /usr/bin/python3 control_center.py )
GUI_RC=$?
set -e
exit "${GUI_RC}"
