#!/usr/bin/env python3
"""Extract the kinematic tree of the RB-Y1 (joint origins, axes, limits) from its URDF into rby1_kinematics.json (no meshes, ~10 kB), so
the agent tools run without the 30 MB URDF export. The URDF is the one the SDK ships (model_v_1_2_a.urdf, also in the sim image).
  make_kinematics.py ~/docker/isaac-sim/data/rby1_urdf/model_v_1_2_a.urdf
"""
import json
import os
import sys
import xml.etree.ElementTree as ET


def floats(text, default):
    return [float(v) for v in text.split()] if text else list(default)


def main(urdf, out=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rby1_kinematics.json")):
    joints = []
    for j in ET.parse(urdf).getroot().findall("joint"):
        if j.get("type") == "floating":
            continue
        o, ax, lim = j.find("origin"), j.find("axis"), j.find("limit")
        joints.append({"name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                       "xyz": floats(o.get("xyz") if o is not None else None, (0, 0, 0)), "rpy": floats(o.get("rpy") if o is not None else None, (0, 0, 0)),
                       "axis": floats(ax.get("xyz") if ax is not None else None, (1, 0, 0)),
                       "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                       "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None})
    json.dump({"source": os.path.basename(urdf), "root": "base", "joints": joints}, open(out, "w"), indent=1)
    print(f"{len(joints)} joints -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
