#!/usr/bin/env bash
# RBY1 model in RViz, driven by the simulator's /rby1/joint_states (start the sim with docker/run_sdk.sh ... --vision).
#   ./rby1_model.sh export   one-off: USD -> URDF + meshes, robot and grippers (headless Isaac Sim in the sim image, ~3 min)
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
    EXP=/isaac-sim/standalone_examples/api/isaacsim.asset.exporter.urdf/urdf_export.py
    OUTC=/isaac-sim/.local/share/ov/data/rby1_urdf
    # The gripper USDs have no articulation root, which the exporter requires: the copy of the example adds one.
    docker run --rm --gpus all -u 1234:1234 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y --entrypoint bash \
        -v "${REPO}/assets:/a:ro" -v "${H}/cache/main:/isaac-sim/.cache:rw" -v "${H}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
        -v "${H}/cache/kit:/isaac-sim/kit/cache:rw" -v "${H}/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
        -v "${H}/config:/isaac-sim/.nvidia-omniverse/config:rw" -v "${H}/data:/isaac-sim/.local/share/ov/data:rw" "${IMAGE}" -c "
        /isaac-sim/python.sh ${EXP} --usd-path /a/model_v_1_2_a.usd --output-dir ${OUTC}
        cp ${EXP} /tmp/exp.py
        sed -i 's/^        stage = omni.usd.get_context().get_stage()/&\n        from pxr import UsdPhysics; UsdPhysics.ArticulationRootAPI.Apply(stage.GetDefaultPrim())/' /tmp/exp.py
        for s in left right; do /isaac-sim/python.sh /tmp/exp.py --usd-path /a/gripper/rb_gripper/rb_gripper_\$s.usd --output-dir ${OUTC}/gripper_\$s; done"
    python3 "${HERE}/merge_urdf.py" "${OUT}"
    ;;
view)
    set +u; source /opt/ros/jazzy/setup.bash; set -u
    export FASTDDS_BUILTIN_TRANSPORTS=UDPv4   # the sim container's DDS shared memory does not reach the host
    R=(-r /tf:=/tf_urdf -r /tf_static:=/tf_static_urdf -p use_sim_time:=true)
    ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "${OUT}/rsp_params.yaml" \
        -r joint_states:=/rby1/joint_states "${R[@]}" &
    PIDS=($!)
    # Camera frames: fixed to the wrist links, so static TFs from the simulator's export of the mount (rby1_gt.json). The
    # simulator's own TF node cannot publish camera prims.
    if [[ -f "${H}/data/rby1_gt.json" ]]; then
        while read -r args; do
            ros2 run tf2_ros static_transform_publisher ${args} --ros-args -r /tf_static:=/tf_static_urdf &
            PIDS+=($!)
        done < <(/usr/bin/python3 "${HERE}/camera_tf_args.py" "${H}/data/rby1_gt.json")
    fi
    trap 'kill "${PIDS[@]}" 2>/dev/null' EXIT
    # RViz rewrites its config file on exit: open a temporary copy so the one in the repo stays as written
    CFG="$(mktemp --suffix=.rviz)"
    cp "${HERE}/rby1_model.rviz" "${CFG}"
    trap 'kill "${PIDS[@]}" 2>/dev/null; rm -f "${CFG}"' EXIT
    rviz2 -d "${CFG}" --ros-args "${R[@]}"
    ;;
*) echo "usage: $0 export|view" >&2; exit 1 ;;
esac
