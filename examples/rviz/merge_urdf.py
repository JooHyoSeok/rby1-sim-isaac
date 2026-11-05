"""Turn the exported URDFs (robot + grippers) into one URDF that robot_state_publisher and RViz accept.

usage: merge_urdf.py <dir with model_v_1_2_a.urdf, gripper_left/, gripper_right/>   (see rby1_model.sh export)
Writes <dir>/rby1.urdf and <dir>/rsp_params.yaml. Fixes up the exporter's output: floating base -> fixed, absolute mesh
paths, the velocity/effort every <limit> needs, emissive white materials (Ke=1) and the grippers, which are exported
as separate URDFs (no articulation root in their USD) and attached to the wrist links like src/rby1_task.py does.
"""
import glob
import re
import sys
import xml.etree.ElementTree as ET

d = sys.argv[1]
tree = ET.parse(f"{d}/model_v_1_2_a.urdf")
robot = tree.getroot()
for j in robot.iter("joint"):
    if j.get("type") == "floating":
        j.set("type", "fixed")


def absolute_meshes(root, folder):
    for m in root.iter("mesh"):
        m.set("filename", m.get("filename").replace("./meshes/", f"file://{folder}/meshes/"))
    for mtl in glob.glob(f"{folder}/meshes/*.mtl"):  # Ke 1 renders every link as flat white in RViz
        text = open(mtl).read()
        open(mtl, "w").write(re.sub(r"(?m)^Ke .*", "Ke 0.000000 0.000000 0.000000", text))


absolute_meshes(robot, d)
for side in ("left", "right"):
    g = ET.parse(glob.glob(f"{d}/gripper_{side}/*.urdf")[0]).getroot()
    absolute_meshes(g, f"{d}/gripper_{side}")
    for link in g.findall("link"):
        if link.get("name") != "world":
            robot.append(link)
    for joint in g.findall("joint"):
        if joint.get("type") != "floating":  # the exporter's world -> root joint
            robot.append(joint)
    tool = ET.SubElement(robot, "joint", name=f"tool_{side}", type="fixed")  # assets/gripper/rb_gripper/gripper.json
    ET.SubElement(tool, "origin", xyz="0 0 -0.1261", rpy="0 0 0")
    ET.SubElement(tool, "parent", link=f"link_{side}_arm_6")
    ET.SubElement(tool, "child", link=f"ee_{side}")
for lim in robot.iter("limit"):
    lim.attrib.setdefault("velocity", "10")
    lim.attrib.setdefault("effort", "100")
ET.indent(tree)
tree.write(f"{d}/rby1.urdf")
text = open(f"{d}/rby1.urdf").read()
with open(f"{d}/rsp_params.yaml", "w") as f:
    f.write("robot_state_publisher:\n  ros__parameters:\n    robot_description: |\n")
    f.write("".join("      " + line + "\n" for line in text.splitlines()))
print(f"wrote {d}/rby1.urdf ({len(list(robot.iter('link')))} links, {len(list(robot.iter('joint')))} joints)")
