#!/usr/bin/env bash
# Two windows with the servo debug images (wrist camera overlay, stereo box pose), on the host display.
# The images are drawn by perception_node.py (ROS 2 container) and also show in RViz (examples/rviz/rby1_model.sh view).
# (The host's OpenCV has no GUI backend, hence rqt_image_view instead of cv2.imshow; /usr/bin/python3 because `ros2` needs
# the system python, not a virtualenv's, and the script's `env python3` shebang would pick the virtualenv.)
set +u; source /opt/ros/jazzy/setup.bash; set -u
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4   # the sim container's DDS shared memory does not reach the host
/usr/bin/python3 /opt/ros/jazzy/lib/rqt_image_view/rqt_image_view /rby1/servo_debug/wrist_right &
/usr/bin/python3 /opt/ros/jazzy/lib/rqt_image_view/rqt_image_view /rby1/servo_debug/stereo_right &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
wait
