#!/usr/bin/env bash
# RBY1 model in RViz, driven by the simulator's /rby1/joint_states (start the sim with docker/run_sdk.sh ... --vision).
#   ./rby1_model.sh export   one-off: USD -> URDF + meshes (headless Isaac Sim in the sim image, ~1 min)
#   ./rby1_model.sh view     robot_state_publisher + RViz (host ROS 2 Jazzy, display $DISPLAY)
# Colors come from the exported mtl files. TF goes to /tf_urdf because the sim's own /tf uses the same
# frame names with a different tree.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
IMAGE="${IMAGE:-rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2-isaac6.1.0-r1}"
H="${ISAAC_HOST_DIR:-${HOME}/docker/isaac-sim}"
OUT="${H}/data/rby1_urdf"

case "${1:-}" in
export)
    docker run --rm --gpus all -u 1234:1234 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y --entrypoint bash \
        -v "${REPO}/assets:/a:ro" -v "${H}/cache/main:/isaac-sim/.cache:rw" -v "${H}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
        -v "${H}/cache/kit:/isaac-sim/kit/cache:rw" -v "${H}/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
        -v "${H}/config:/isaac-sim/.nvidia-omniverse/config:rw" -v "${H}/data:/isaac-sim/.local/share/ov/data:rw" "${IMAGE}" \
        -c '/isaac-sim/python.sh /isaac-sim/standalone_examples/api/isaacsim.asset.exporter.urdf/urdf_export.py --usd-path /a/model_v_1_2_a.usd --output-dir /isaac-sim/.local/share/ov/data/rby1_urdf'
    # the mtl files export Ke=1 (emissive white), which RViz renders as flat white links
    sed -i 's/^Ke .*/Ke 0.000000 0.000000 0.000000/' "${OUT}"/meshes/*.mtl
    # floating base -> fixed, absolute mesh paths, and the velocity/effort every <limit> needs
    python3 - "${OUT}" <<'PY'
import re, sys
d = sys.argv[1]
s = open(f"{d}/model_v_1_2_a.urdf").read()
s = s.replace('type="floating"', 'type="fixed"').replace('filename="./meshes/', f'filename="file://{d}/meshes/')
def fix(m):
    t = m.group(0)
    if "velocity=" not in t: t = t.replace("<limit", '<limit velocity="10"', 1)
    if "effort=" not in t: t = t.replace("<limit", '<limit effort="100"', 1)
    return t
s = re.sub(r"<limit[^>]*>", fix, s)
open(f"{d}/rby1.urdf", "w").write(s)
with open(f"{d}/rsp_params.yaml", "w") as f:
    f.write("robot_state_publisher:\n  ros__parameters:\n    robot_description: |\n")
    f.write("".join("      " + l + "\n" for l in s.splitlines()))
PY
    ;;
view)
    set +u; source /opt/ros/jazzy/setup.bash; set -u
    export FASTDDS_BUILTIN_TRANSPORTS=UDPv4   # the sim container's DDS shared memory does not reach the host
    R=(-r /tf:=/tf_urdf -r /tf_static:=/tf_static_urdf -p use_sim_time:=true)
    ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "${OUT}/rsp_params.yaml" \
        -r joint_states:=/rby1/joint_states "${R[@]}" &
    trap 'kill $! 2>/dev/null' EXIT
    rviz2 -d "${HERE}/rby1_model.rviz" --ros-args "${R[@]}"
    ;;
*) echo "usage: $0 export|view" >&2; exit 1 ;;
esac
