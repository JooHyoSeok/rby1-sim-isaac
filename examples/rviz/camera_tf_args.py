"""Arguments for tf2_ros static_transform_publisher, one line per wrist camera, from rby1_gt.json (T_link_camera)."""
import json
import math
import sys

gt = json.load(open(sys.argv[1]))
for side, w in gt["wrist"].items():
    T = w["T_link_camera"]
    R = [row[:3] for row in T[:3]]
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        q = ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, s / 4)
    else:  # largest diagonal element
        i = max(range(3), key=lambda k: R[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(R[i][i] - R[j][j] - R[k][k] + 1) * 2
        v = [0.0] * 3
        v[i], v[j], v[k] = s / 4, (R[j][i] + R[i][j]) / s, (R[k][i] + R[i][k]) / s
        q = (v[0], v[1], v[2], (R[k][j] - R[j][k]) / s) if i == 0 else \
            (v[0], v[1], v[2], (R[0][2] - R[2][0]) / s) if i == 1 else (v[0], v[1], v[2], (R[1][0] - R[0][1]) / s)
    print(f"--x {T[0][3]} --y {T[1][3]} --z {T[2][3]} --qx {q[0]} --qy {q[1]} --qz {q[2]} --qw {q[3]} "
          f"--frame-id {w['link']} --child-frame-id wrist_{side}_cam")
