"""Can both hands hold a part each at the mating plane in front of the chest? Offline inverse kinematics of the two arms (numpy, the FK of
examples/agent/kinematics.py, torso fixed in the squat of examples/gui/grasp_core.py), no simulator.

The insertion axis is the base y axis, the mating plane is x = x_m, z = z_m, y = 0. The female part (left hand) extends to +y, the male part
(right hand) to -y; each hand holds its part GRASP_BACK behind the mating face, so with the parts `gap` apart the two grasp points are 2 * GRASP_BACK + gap
apart. The fingers close across the part, so the closing axis (x of the wrist link, see R_FRONT in grasp_core.py) is vertical, perpendicular to the
part axis; the tool axis points forward, or turned 30 degrees inwards or down. Everything except the hand pose (torso, other joints, the physical
space the arms and fingers take) is checked only by joint limits and the gap between the elbows and forearms.
  /usr/bin/python3 meet_study.py            table of what is reachable
"""
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
import kinematics as K  # noqa: E402
from arm_ik import SEED_TORSO, READY, MARGIN, solve, margin, arm_gap, hand_pose  # noqa: E402,F401

GRASP_BACK = 0.03                                            # [m] from the mating face to the middle of the fingers


VARIANTS = {   # name: (tool axis for the right hand, closing axis); the left hand mirrors y
    "forward, close vertical": ((1, 0, 0), (0, 0, 1)),
    "forward, close vertical (flipped)": ((1, 0, 0), (0, 0, -1)),
    "inward 30 deg, close vertical": ((math.cos(math.radians(30)), math.sin(math.radians(30)), 0), (0, 0, 1)),
    "down 30 deg, close across": ((math.cos(math.radians(30)), 0, -math.sin(math.radians(30))), (math.sin(math.radians(30)), 0, math.cos(math.radians(30)))),
}


def study(xs=(0.30, 0.36, 0.42, 0.48), zs=(0.90, 1.00, 1.10), gaps=(0.03, 0.0)):
    """{variant: {(x, z): (reachable at every gap, smallest elbow/forearm/wrist gap between the arms [m])}}"""
    out = {}
    for name, (tool, closing) in VARIANTS.items():
        grid = {}
        for x, z in itertools.product(xs, zs):
            gaps_seen, ok = [], True
            for gap in gaps:
                sols = {}
                for side, sign in (("right", 1), ("left", -1)):
                    t = np.array([tool[0], sign * tool[1], tool[2]])
                    c = np.array([closing[0], sign * closing[1], closing[2]])
                    # right hand: male part, grasp point at y = -(gap + back); left hand: female, y = +back
                    y = -(gap + GRASP_BACK) if side == "right" else GRASP_BACK
                    sols[side] = solve(side, np.array([x, y, z]), hand_pose(t, c))
                if any(v is None for v in sols.values()):
                    ok = False
                    break
                gaps_seen.append(arm_gap(sols["right"], sols["left"]))
            grid[(x, z)] = (ok, min(gaps_seen) if ok else None)
        out[name] = grid
    return out


if __name__ == "__main__":
    xs, zs = (0.30, 0.36, 0.42, 0.48), (1.10, 1.00, 0.90)
    print(f"both hands hold a part each (axis along base y), grasp point {GRASP_BACK * 100:.0f} cm behind the mating face, parts 3 cm apart and seated;")
    print(f"torso squat fixed, every joint {MARGIN} rad inside its limit. Cell: arm gap in cm (elbows/forearms, smallest of the two gaps), '--' not reachable")
    print("rows z = 1.10 / 1.00 / 0.90 m, columns x = " + " / ".join(f"{x:.2f}" for x in xs))
    for name, grid in study(xs, tuple(sorted(zs))).items():
        print(f"\n{name}")
        for z in zs:
            print("   " + "  ".join("--" if not grid[(x, z)][0] else f"{grid[(x, z)][1] * 100:2.0f}" for x in xs))
