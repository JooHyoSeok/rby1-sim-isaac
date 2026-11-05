#!/usr/bin/env bash
# One command: start the simulator, wait until it is ready, start the perception node, run the hybrid visual-servoing grasp
# (stereo + wrist camera -> box -> continuous servo -> grasp -> lift), report the result and clean everything up.
#
#   examples/vservo/run_demo.sh [options]
#     --view            also open RViz (robot model, camera frames, table and box) and the two servo overlay windows
#     --record          record the servo overlay video (path printed at the end)
#     --bias DX DY DZ   add this to the stereo box position [m] (emulates a calibration error the image servo must remove)
#     --no-servo        stereo measurement only, no image correction (for comparison)
#     --keep            leave the simulator and the viewers running afterwards
#     --cpuset CPUS     pin the simulator to these cores (default: the last 6; "" = no pinning)
#     --image TAG       simulator image tag (default 0.10.7-a_v1.2-isaac6.1.0-r1)
#
# Needs the ROS 2 container (intelligent-robotic-manipulation) running and the sim image present. Takes ~3 minutes.
# Each run starts a fresh simulator, so the box is back on the table every time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ROS=intelligent-robotic-manipulation
SIM=rby1-sim-isaac
IMAGE=0.10.7-a_v1.2-isaac6.1.0-r1
CORES="$(nproc)"
CPUSET="$((CORES - 6))-$((CORES - 1))"
VIEW=0; KEEP=0; RECORD=0; NOSERVO=0; BIAS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --view) VIEW=1; shift ;;
        --keep) KEEP=1; shift ;;
        --record) RECORD=1; shift ;;
        --no-servo) NOSERVO=1; shift ;;
        --bias) BIAS="$2 $3 $4"; shift 4 ;;
        --cpuset) CPUSET="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

WORK="$(mktemp -d /tmp/rby1_demo.XXXXXX)"
say() { printf '\n== %s\n' "$*"; }
SIM_PID=""
VIEWER_GROUPS=()   # process groups of the viewers this script started (killed by group, never by name pattern)

cleanup() {
    set +e
    if [[ "${KEEP}" == "1" ]]; then
        say "left running (--keep). Stop with:  docker stop ${SIM}; docker exec ${ROS} pkill -f perception_node.py"
        return
    fi
    say "cleaning up"
    docker exec "${ROS}" pkill -f perception_node.py 2>/dev/null
    for g in "${VIEWER_GROUPS[@]:-}"; do [[ -n "${g}" ]] && kill -- "-${g}" 2>/dev/null; done
    sleep 2
    for g in "${VIEWER_GROUPS[@]:-}"; do [[ -n "${g}" ]] && kill -9 -- "-${g}" 2>/dev/null; done   # rqt_image_view ignores SIGTERM
    docker stop "${SIM}" >/dev/null 2>&1
    [[ -n "${SIM_PID}" ]] && kill "${SIM_PID}" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

docker image inspect "rainbowroboticsofficial/rby1-sim-isaac:${IMAGE}" >/dev/null 2>&1 || { echo "image ${IMAGE} not found" >&2; exit 1; }
[[ "$(docker inspect -f '{{.State.Running}}' "${ROS}" 2>/dev/null)" == "true" ]] || { echo "container ${ROS} is not running" >&2; exit 1; }

say "1/5 starting the simulator (headless, cores ${CPUSET:-all}); log: ${WORK}/sim.log"
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

say "2/5 perception node (ROS 2 container)"
for f in vision.py perception_node.py check_result.py; do docker cp "${HERE}/${f}" "${ROS}:/tmp/${f}"; done
docker cp "${HOME}/docker/isaac-sim/data/rby1_gt.json" "${ROS}:/tmp/calib.json"   # stereo extrinsics, rewritten at every simulator start
PERC_ARGS=""
[[ -n "${BIAS}" ]] && PERC_ARGS+=" --bias ${BIAS}"
[[ "${RECORD}" == "1" ]] && PERC_ARGS+=" --record /tmp/servo_demo.mp4"
docker exec -d "${ROS}" bash -c "source /opt/ros/jazzy/setup.bash; export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; cd /tmp; rm -f servo_demo.mp4; python3 perception_node.py${PERC_ARGS} > /tmp/perception.log 2>&1"
for ((i = 0; i < 30; i++)); do
    docker exec "${ROS}" grep -q "perception running" /tmp/perception.log 2>/dev/null && break
    sleep 1
done
docker exec "${ROS}" grep -q "perception running" /tmp/perception.log || { echo "perception node failed:" >&2; docker exec "${ROS}" tail -8 /tmp/perception.log >&2; exit 1; }
echo "perception running${BIAS:+ (stereo bias ${BIAS} m)}"

if [[ "${VIEW}" == "1" ]]; then
    say "3/5 viewers (RViz + servo overlay windows on ${DISPLAY:-:1})"
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
    setsid "${HERE}/servo_view.sh" > "${WORK}/servo_view.log" 2>&1 &
    VIEWER_GROUPS+=($!)
    sleep 15
else
    say "3/5 viewers: skipped (use --view)"
fi

say "4/5 visual servoing grasp"
docker cp "${HERE}/vision.py" "${SIM}:/tmp/vision.py"; docker cp "${HERE}/servo.py" "${SIM}:/tmp/servo.py"
docker cp "${HERE}/hybrid_grasp.py" "${SIM}:/tmp/hybrid_grasp.py"; docker cp "${REPO}/examples/gripper_example.py" "${SIM}:/tmp/gripper_example.py"
EXEC_ARGS=""
[[ "${NOSERVO}" == "1" ]] && EXEC_ARGS+=" --no-servo"
set +e
docker exec "${SIM}" /isaac-sim/python.sh /tmp/hybrid_grasp.py${EXEC_ARGS} 2>&1 | grep -vE "^\[|^20[0-9]{2}-|Hub|PoseTree|^INFO|^$" | tee "${WORK}/grasp.log"
RUN_RC=${PIPESTATUS[0]}
set -e

say "5/5 result (simulator ground truth)"
set +e
docker exec "${ROS}" bash -c "source /opt/ros/jazzy/setup.bash; export FASTDDS_BUILTIN_TRANSPORTS=UDPv4; python3 /tmp/check_result.py"
RESULT_RC=$?
set -e
if [[ "${RECORD}" == "1" ]] && docker cp "${ROS}:/tmp/servo_demo.mp4" "${WORK}/servo_demo.mp4" 2>/dev/null; then
    echo "video: ${WORK}/servo_demo.mp4"
fi
echo "logs: ${WORK}"
[[ "${RUN_RC}" == "0" ]] || { echo "the grasp script failed (exit ${RUN_RC})" >&2; exit "${RUN_RC}"; }
exit "${RESULT_RC}"
