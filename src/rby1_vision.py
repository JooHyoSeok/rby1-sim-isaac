# SPDX-License-Identifier: Apache-2.0
"""Course scene add-on: stereo rig, wrist cameras, coloured landmarks, three stackable boxes and ROS 2 publishing.

Enabled with ``--vision``. Everything here is visual or kinematic (no new rigid bodies except a static table), so
the robot dynamics and the rby1-sdk UDP path are untouched. All poses below are first guesses in metres: check them
in a rendered frame and tune (the wrist camera and EE landmark poses depend on the gripper geometry).

ROS 2 topics (sim-time stamped, published at the render rate, so pass ``--render-hz`` when headless):
  /clock, /rby1/joint_states
  /rby1/{stereo_left,stereo_right,wrist_left,wrist_right}/image_raw and /camera_info
  /rby1/gt/tf   ground-truth wrist link and object poses w.r.t. the robot root (evaluation only)
  rby1_gt.json  ground-truth fixed poses: stereo cameras, wrist camera and landmark offsets, object, targets
"""
from __future__ import annotations

import json
import os
import socket
import time

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROBOT_ROOT = "/World/RBY1"
WRIST_LINKS = {"left": "link_left_arm_6", "right": "link_right_arm_6"}

IMAGE_W, IMAGE_H = (int(v) for v in os.environ.get("RBY1_CAM_RES", "640x480").split("x"))
FOCAL_MM, H_APERTURE_MM = 15.0, 20.955  # fx = IMAGE_W * FOCAL_MM / H_APERTURE_MM

# External stereo rig: eye centre, look-at point, baseline. Override with RBY1_STEREO_EYE="x,y,z".
STEREO_EYE = np.array([float(v) for v in os.environ.get("RBY1_STEREO_EYE", "1.4,0.0,1.2").split(",")])
STEREO_TARGET = np.array([0.5, 0.0, 0.85])
STEREO_BASELINE = 0.12

# Wrist camera pose in the wrist link frame: (translation, look direction, up). The tool points along -z of the wrist link;
# the gripper spans x +-0.063, y +-0.032, z -0.246..-0.11 in that frame (printed by setup()), the palm ends at about z=-0.2
# where the fingers attach. The camera sits on the back of the hand (+y face: the top when the fingers close sideways, as in
# examples/grasp), 8 cm above the hand axis, looking forward and 15 degrees down: the tops of the fingertips and everything
# from 6 cm past the tips on stay visible (projected gripper geometry, see the commit message).
# A camera on the thin side or under the palm was tried first; see git history. Image right = x of the wrist link.
# examples/vservo/servo.py repeats this pose (WRIST_EYE / WRIST_AIM): keep them equal.
WRIST_CAM = {side: (np.array([0.0, 0.08, -0.12]), np.array([0.0, -0.08, -0.28]), np.array([0.0, 1.0, 0.0]))
             for side in ("left", "right")}

# Stereo rig on the head (opt-in, RBY1_HEAD_CAM=1: two more cameras to render; verified in the simulator 2026-11-12: FK-predicted landmark pixels within 1 px): the eyes in the frame of link_head_2, which looks along +x with y to the
# left and z up. examples/assembly/head_camera.py repeats this mount: keep them equal.

HEAD_CAM = os.environ.get("RBY1_HEAD_CAM", "0") == "1"
HEAD_MOUNT, HEAD_BASELINE = np.array([0.06, 0.0, 0.03]), 0.12

FINGER_MASS = 0.02  # kg, for finger bodies whose USD sets no mass
SPHERE_R = 0.012
# Asymmetric red landmarks P1..P4 in the wrist link frame (P1 is the single sphere for eye-to-hand calibration).
EE_LANDMARKS = [(0.04, 0.0, 0.05), (-0.04, 0.03, 0.06), (0.0, -0.05, 0.08), (0.03, 0.05, 0.10)]
# Table, object, object landmarks Q1..Q4 (object frame) and the R/B/Y calibration target (table frame, on top).
# Table height for the lowered-torso grasp posture (grasp_object.py TORSO: shoulders at z=1.05).
TABLE_TOP = 0.72
# A thin top without legs: a solid box down to the floor gets hit by the hand when it swings up from the hanging pose.
TABLE_CENTER, TABLE_SIZE = (0.88, 0.0, TABLE_TOP - 0.02), (0.6, 0.8, 0.04)  # front edge x=0.58
OBJECT_SIZE = (0.05, 0.05, 0.08)  # tall enough to be grasped above the table surface
OBJECT_MASS, OBJECT_FRICTION = 0.1, 1.5  # a dynamic rigid body the gripper can lift
# The side landmarks sit 2.5 cm off the faces (1.3 cm clear of them): at 3.7 cm the stereo camera on the far side saw them half hidden
# behind the box edge, which biased their centroids (found in a stack of three).
OBJECT_LANDMARKS = [(0.0, 0.0, 0.052), (0.037, 0.0, 0.0), (-0.02, 0.05, 0.0), (-0.02, -0.05, 0.0)]
# Three boxes that can be stacked. Each has its own landmark colour (pure sphere colour), so the perception tells the boxes apart
# and keeps the landmarks of a stack apart (examples/vservo/vision.py BOX_COLORS repeats the names); the body is a paler tint that
# the colour masks reject. Start positions: in a row 12 cm apart at x=0.62, the band of the default table the arm reaches
# (examples/gui/scene_config.py BOX_DEFAULTS repeats them).
BOXES = {"blue": {"rgb": (0.0, 0.0, 1.0), "body": (0.35, 0.5, 1.0), "xy": (0.62, -0.34)},
         "green": {"rgb": (0.0, 1.0, 0.0), "body": (0.45, 1.0, 0.45), "xy": (0.62, -0.22)},
         "yellow": {"rgb": (1.0, 1.0, 0.0), "body": (1.0, 0.95, 0.55), "xy": (0.62, -0.10)}}
# The R/B/Y calibration target (HW7), off by default: its blue and yellow spheres have the colours of the blue and yellow boxes' landmarks, and the
# wrist camera matched a sphere as a landmark of the box next to it (and later, from afar): the servo measured the box 4 cm off and missed the grasp.
# Set RBY1_CALIB_TARGET=1 in the simulator's environment to build it for the calibration data collection (examples/calibration_data).
CALIBRATION_TARGET = os.environ.get("RBY1_CALIB_TARGET", "0") == "1"
TARGET_SPHERES = {"R": ((1.00, 0.30, TABLE_TOP + SPHERE_R), (1, 0, 0)),
                  "B": ((1.08, 0.24, TABLE_TOP + SPHERE_R), (0, 0, 1)),
                  "Y": ((1.02, 0.16, TABLE_TOP + SPHERE_R), (1, 1, 0)),
                  "R2": ((1.10, 0.14, TABLE_TOP + SPHERE_R + 0.03), (1, 0, 0))}  # 4th, off-plane: removes P3P ambiguity


def _material(stage, path: str, rgb) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb) * 0.5)  # stable under lighting
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _sphere(stage, path: str, pos, rgb, mat) -> None:
    s = UsdGeom.Sphere.Define(stage, path)
    s.GetRadiusAttr().Set(SPHERE_R)
    UsdGeom.XformCommonAPI(s).SetTranslate(Gf.Vec3d(*pos))
    UsdShade.MaterialBindingAPI(s.GetPrim()).Bind(mat)


def _cuboid(stage, path: str, center, size, rgb, static_collider: bool) -> None:
    c = UsdGeom.Cube.Define(stage, path)
    c.GetSizeAttr().Set(1.0)
    api = UsdGeom.XformCommonAPI(c)
    api.SetTranslate(Gf.Vec3d(*center))
    api.SetScale(Gf.Vec3f(*size))
    c.GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
    if static_collider:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())


def _look_at_matrix(eye, forward, up) -> Gf.Matrix4d:
    """Camera-to-parent transform for a USD camera (looks along -Z, +Y up)."""
    eye, forward, up = (Gf.Vec3d(*map(float, v)) for v in (eye, forward, up))
    return Gf.Matrix4d().SetLookAt(eye, eye + forward, up).GetInverse()


def _camera(stage, path: str, matrix: Gf.Matrix4d, near: float = 0.05) -> None:
    cam = UsdGeom.Camera.Define(stage, path)
    cam.GetFocalLengthAttr().Set(FOCAL_MM)
    cam.GetHorizontalApertureAttr().Set(H_APERTURE_MM)
    cam.GetVerticalApertureAttr().Set(H_APERTURE_MM * IMAGE_H / IMAGE_W)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(near, 10.0))
    cam.AddTransformOp().Set(matrix)


def _friction_material(stage) -> UsdShade.Material:
    """High-friction physics material shared by the object and the gripper fingers (PhysX averages the two)."""
    mat = UsdShade.Material.Define(stage, "/World/Looks/object_physics")
    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr(OBJECT_FRICTION)
    phys.CreateDynamicFrictionAttr(OBJECT_FRICTION)
    phys.CreateRestitutionAttr(0.0)
    return mat


def add_gripper_colliders(stage) -> None:
    """The robot USDs ship without collision geometry on the wrist link and the whole gripper, so the hand passes
    through the table and the object. Give the finger meshes convex-hull colliders and the high-friction material
    (the palm and wrist hulls jam against the forearm when the elbow folds, so they stay without).
    Called from RBY1Task.set_up_scene, i.e. before physics starts (shapes added later are not covered by the
    articulation's disabled self-collisions and jam the arm against itself)."""
    friction = _friction_material(stage)
    roots = [f"{ROBOT_ROOT}/{n}" for n in (*WRIST_LINKS.values(), "left_gripper", "right_gripper")]
    for _ in range(4):  # meshes inside instanceable prims cannot be edited: un-instance (nested instances need passes)
        for root in roots:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if prim.IsInstance():
                    prim.SetInstanceable(False)
    count = 0
    finger_bodies = set()
    for root in roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.IsA(UsdGeom.Mesh) and "finger" in str(prim.GetPath()):
                body = prim
                while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = body.GetParent()
                if body:
                    finger_bodies.add(body)
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(friction, UsdShade.Tokens.weakerThanDescendants, "physics")
                count += 1
    # The articulation's disabled self-collisions do not cover these shapes: the fingers jam against the forearm when
    # the elbow folds. Filter every finger body against every other robot body; the object and table still collide.
    robot_bodies = [p.GetPath() for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_ROOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for body in finger_bodies:
        rel = UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel()
        for other in robot_bodies:
            if other != body.GetPath():
                rel.AddTarget(other)
    # A collider without an authored mass adds volume*density to its body: heavy convex hulls would overload the elbow.
    for body in finger_bodies:
        mass = UsdPhysics.MassAPI.Apply(body).GetMassAttr()
        print(f"[Vision] {body.GetPath()} mass authored={mass.HasAuthoredValue()} value={mass.Get()}")
        if not mass.HasAuthoredValue() or not mass.Get():
            mass.Set(FINGER_MASS)
    print(f"[Vision] {count} finger meshes got convex-hull colliders, {len(finger_bodies)} finger bodies filtered "
          f"against {len(robot_bodies)} robot bodies")


def build_scene(stage) -> dict[str, str]:
    """Create cameras, landmarks, table and the boxes. Returns {topic name: camera prim path}."""
    red = _material(stage, "/World/Looks/red", (1, 0, 0))
    mats = {(1, 0, 0): red, (0, 0, 1): _material(stage, "/World/Looks/blue", (0, 0, 1)),
            (1, 1, 0): _material(stage, "/World/Looks/yellow", (1, 1, 0)), (0, 1, 0): _material(stage, "/World/Looks/green", (0, 1, 0))}

    forward = STEREO_TARGET - STEREO_EYE
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    cams = {}
    UsdGeom.Xform.Define(stage, "/World/StereoRig")
    for name, sign in (("stereo_left", -1), ("stereo_right", 1)):
        path = f"/World/StereoRig/{name}"
        _camera(stage, path, _look_at_matrix(STEREO_EYE + sign * right * STEREO_BASELINE / 2, forward, [0, 0, 1]))
        cams[name] = path
    for side, link in WRIST_LINKS.items():
        pos, look, up = WRIST_CAM[side]
        path = f"{ROBOT_ROOT}/{link}/wrist_cam"
        _camera(stage, path, _look_at_matrix(pos, look, up), near=0.01)  # the object gets within 3 cm of the lens
        cams[f"wrist_{side}"] = path
        for i, p in enumerate(EE_LANDMARKS, 1):
            _sphere(stage, f"{ROBOT_ROOT}/{link}/P{i}", p, (1, 0, 0), red)

    if HEAD_CAM:
        for name, sign in (("head_left", 1), ("head_right", -1)):
            path = f"{ROBOT_ROOT}/link_head_2/{name}"
            _camera(stage, path, _look_at_matrix(HEAD_MOUNT + np.array([0.0, sign * HEAD_BASELINE / 2, 0.0]), [1, 0, 0], [0, 0, 1]))
            cams[name] = path

    _cuboid(stage, "/World/Table", TABLE_CENTER, TABLE_SIZE, (0.6, 0.5, 0.4), static_collider=True)
    friction = _friction_material(stage)
    for name, box in BOXES.items():
        path = f"/World/Box_{name}"
        obj = UsdGeom.Xform.Define(stage, path)  # unscaled parent so the landmark spheres stay round
        UsdGeom.XformCommonAPI(obj).SetTranslate(Gf.Vec3d(*box["xy"], TABLE_TOP + OBJECT_SIZE[2] / 2))
        _cuboid(stage, f"{path}/body", (0, 0, 0), OBJECT_SIZE, box["body"], static_collider=True)
        UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
        UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(OBJECT_MASS)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{path}/body")).Bind(
            friction, UsdShade.Tokens.weakerThanDescendants, "physics")
        for i, p in enumerate(OBJECT_LANDMARKS, 1):
            _sphere(stage, f"{path}/Q{i}", p, box["rgb"], mats[box["rgb"]])
    for name, (p, rgb) in (TARGET_SPHERES.items() if CALIBRATION_TARGET else ()):
        _sphere(stage, f"/World/Target/{name}", p, rgb, mats[rgb])
    return cams


def robot_links(stage) -> list[str]:
    """Prim paths of every body linked by a physics joint (arm, torso, head, wheels, gripper), for RViz."""
    bodies = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_ROOT)):
        if prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
                bodies.update(str(t) for t in rel.GetTargets())
    return sorted(bodies)


def build_ros_graph(cams: dict[str, str], stage) -> None:
    """One execution graph: clock, joint states, GT transforms, and image + camera_info per camera."""
    import omni.graph.core as og
    import usdrt.Sdf

    keys = og.Controller.Keys
    nodes = [
        ("Tick", "omni.graph.action.OnPlaybackTick"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("ReadJoints", "isaacsim.sensors.physics.IsaacReadJointState"),
        ("PubJoints", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ("GtTf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
    ]
    conn = [
        ("Tick.outputs:tick", "Clock.inputs:execIn"), ("Context.outputs:context", "Clock.inputs:context"),
        ("SimTime.outputs:simulationTime", "Clock.inputs:timeStamp"),
        ("Tick.outputs:tick", "ReadJoints.inputs:execIn"), ("ReadJoints.outputs:execOut", "PubJoints.inputs:execIn"),
        ("Context.outputs:context", "PubJoints.inputs:context"),
        ("Tick.outputs:tick", "GtTf.inputs:execIn"), ("Context.outputs:context", "GtTf.inputs:context"),
        ("SimTime.outputs:simulationTime", "GtTf.inputs:timeStamp"),
    ]
    conn += [(f"ReadJoints.outputs:{a}", f"PubJoints.inputs:{a}") for a in (
        "jointNames", "jointPositions", "jointVelocities", "jointEfforts", "jointDofTypes", "stageMetersPerUnit",
        "sensorTime")]
    gt_targets = [f"{ROBOT_ROOT}/{l}" for l in WRIST_LINKS.values()] + [f"/World/Box_{n}" for n in BOXES] + [f"{ROBOT_ROOT}/base"]  # rigid bodies only; base = where it drove
    values = []
    # /tf: link poses for RViz. Isaac publishes the flat RBY1 -> link frames AND the articulation chain (rooted at
    # link_torso_5); examples/rviz/tf_chain_relay.py keeps only the chain, which is a proper tree.
    links = robot_links(stage)
    nodes.append(("Tf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"))
    conn += [("Tick.outputs:tick", "Tf.inputs:execIn"), ("Context.outputs:context", "Tf.inputs:context"),
             ("SimTime.outputs:simulationTime", "Tf.inputs:timeStamp")]
    values += [("Tf.inputs:parentPrim", [usdrt.Sdf.Path(ROBOT_ROOT)]),
               ("Tf.inputs:targetPrims", [usdrt.Sdf.Path(p) for p in links])]
    values += [
        ("ReadJoints.inputs:prim", [usdrt.Sdf.Path(ROBOT_ROOT)]),
        ("PubJoints.inputs:topicName", "rby1/joint_states"),
        ("GtTf.inputs:parentPrim", [usdrt.Sdf.Path(ROBOT_ROOT)]),
        ("GtTf.inputs:targetPrims", [usdrt.Sdf.Path(p) for p in gt_targets]),
        ("GtTf.inputs:topicName", "rby1/gt/tf"),
    ]
    for name, path in cams.items():
        nodes += [(f"RP_{name}", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                  (f"Img_{name}", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                  (f"Info_{name}", "isaacsim.ros2.bridge.ROS2CameraInfoHelper")]
        conn += [("Tick.outputs:tick", f"RP_{name}.inputs:execIn"),
                 (f"RP_{name}.outputs:execOut", f"Img_{name}.inputs:execIn"),
                 (f"RP_{name}.outputs:execOut", f"Info_{name}.inputs:execIn"),
                 ("Context.outputs:context", f"Img_{name}.inputs:context"),
                 ("Context.outputs:context", f"Info_{name}.inputs:context"),
                 (f"RP_{name}.outputs:renderProductPath", f"Img_{name}.inputs:renderProductPath"),
                 (f"RP_{name}.outputs:renderProductPath", f"Info_{name}.inputs:renderProductPath")]
        values += [(f"RP_{name}.inputs:cameraPrim", [usdrt.Sdf.Path(path)]),
                   (f"RP_{name}.inputs:width", IMAGE_W), (f"RP_{name}.inputs:height", IMAGE_H),
                   (f"Img_{name}.inputs:topicName", f"rby1/{name}/image_raw"), (f"Img_{name}.inputs:type", "rgb"),
                   (f"Img_{name}.inputs:frameId", name),
                   (f"Info_{name}.inputs:topicName", f"rby1/{name}/camera_info"), (f"Info_{name}.inputs:frameId", name)]
    og.Controller.edit({"graph_path": "/ROS_RBY1", "evaluator_name": "execution"},
                       {keys.CREATE_NODES: nodes, keys.CONNECT: conn, keys.SET_VALUES: values})


def _mat(m: Gf.Matrix4d) -> list:
    """Row-vector USD matrix -> column-vector 4x4 (nested lists)."""
    return np.array(m).T.tolist()


def write_gt(stage, path: str) -> None:
    """Ground truth that does not move with the robot (evaluation only), plus the gripper bounding box per wrist link."""
    import json

    flip = Gf.Matrix4d(1.0).SetDiagonal(Gf.Vec4d(1, -1, -1, 1))  # USD camera (-z forward, +y up) -> ROS optical (+z, +y down)
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    xf = lambda p: UsdGeom.Xformable(stage.GetPrimAtPath(p))
    gt = {"units": "m", "note": "T_a_b maps b-frame points into a; camera frames are ROS optical (+z forward, +y down)",
          "stereo": {n: _mat(flip * xf(f"/World/StereoRig/{n}").ComputeLocalToWorldTransform(0)) for n in ("stereo_left", "stereo_right")},
          "wrist": {}, "ee_landmarks_in_wrist_link": {f"P{i}": p for i, p in enumerate(EE_LANDMARKS, 1)},
          "boxes_world": {n: _mat(xf(f"/World/Box_{n}").ComputeLocalToWorldTransform(0)) for n in BOXES},
          "box_landmarks_in_box": {f"Q{i}": p for i, p in enumerate(OBJECT_LANDMARKS, 1)},
          "target_spheres_world": {n: p for n, (p, _) in TARGET_SPHERES.items()} if CALIBRATION_TARGET else {}, "sphere_radius": SPHERE_R}
    for side, link in WRIST_LINKS.items():
        link_path = f"{ROBOT_ROOT}/{link}"
        link_to_world = xf(link_path).ComputeLocalToWorldTransform(0)
        cam_local = xf(f"{link_path}/wrist_cam").GetLocalTransformation()
        gt["wrist"][side] = {"link": link, "T_link_camera": _mat(flip * cam_local)}
        gripper = stage.GetPrimAtPath(f"{ROBOT_ROOT}/{side}_gripper")
        if not gripper.IsValid():  # no --gripper
            continue
        box = cache.ComputeWorldBound(gripper).ComputeAlignedBox()
        lo, hi = (link_to_world.GetInverse().Transform(v) for v in (box.GetMin(), box.GetMax()))
        print(f"[Vision] {side} gripper extent in {link} frame: {[round(min(a, b), 3) for a, b in zip(lo, hi)]} .. "
              f"{[round(max(a, b), 3) for a, b in zip(lo, hi)]}")
    try:   # outline of the base and the wheels in the base frame (the base stands at the origin now): the executor's drive check uses it
        lo, hi = np.full(2, 9.0), np.full(2, -9.0)
        for prim in stage.GetPrimAtPath(ROBOT_ROOT).GetChildren():
            if prim.GetName() == "base" or "wheel" in prim.GetName():
                box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                lo, hi = np.minimum(lo, [box.GetMin()[0], box.GetMin()[1]]), np.maximum(hi, [box.GetMax()[0], box.GetMax()[1]])
        if lo[0] < hi[0]:
            gt["base_footprint"] = [round(float(v), 3) for v in (*lo, *hi)]
            print(f"[Vision] base outline (x0, y0, x1, y1): {gt['base_footprint']}")
    except Exception as exc:  # noqa: BLE001 - only the drive check needs it; the rest of the scene must come up anyway
        print(f"[Vision] no base outline: {exc}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt, f, indent=1)
    print(f"[Vision] ground truth written to {path}")


# ---------------------------------------------------------------------------------------------------------------------
# Runtime scene commands (UDP, JSON, port COMMAND_PORT): move the box and the table while the simulation runs.
#   {"cmd": "set_box", "name": "blue", "x": .., "y": .., "yaw_deg": 0}   one box ("blue" if no name) on the table, upright (z follows the
#                                                                          table top; a box put on another one is dropped on it)
#   {"cmd": "set_box", "name": "green", "x": .., "y": .., "z": ..}        same, at a given centre height
#   {"cmd": "set_table", "x": .., "y": .., "top": ..}         table centre and top height [m]
#   {"cmd": "reset"}                                           defaults (table and all boxes)
# Sent by examples/gui (the control centre); applied on the simulation thread, between physics steps.
# ---------------------------------------------------------------------------------------------------------------------
COMMAND_PORT = 5110
_cmd_sock = None
_table_top = TABLE_TOP


def _start_command_server() -> None:
    global _cmd_sock
    _cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _cmd_sock.bind(("127.0.0.1", COMMAND_PORT))
    _cmd_sock.setblocking(False)


def _place_box(stage, name: str, x: float, y: float, yaw_deg: float = 0.0, z: float | None = None) -> None:
    obj = stage.GetPrimAtPath(f"/World/Box_{name}")
    api = UsdGeom.XformCommonAPI(obj)
    api.SetTranslate(Gf.Vec3d(x, y, (_table_top + OBJECT_SIZE[2] / 2 if z is None else z) + 0.002))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    body = UsdPhysics.RigidBodyAPI(obj)
    body.CreateVelocityAttr().Set(Gf.Vec3f(0.0))
    body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0))


def _place_table(stage, x: float, y: float, top: float) -> None:
    global _table_top
    _table_top = top
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/Table")).SetTranslate(Gf.Vec3d(x, y, top - TABLE_SIZE[2] / 2))


BASE_POSE_PORT = 5130
_base_tx, _base_last = None, 0.0


def publish_base_pose(robot) -> None:
    """The base's true pose to the executor (examples/gui/grasp_core.py base_pose), at 20 Hz: the wheels slip when the base turns, so the
    executor's odometry drifts (measured: 30 degrees commanded, 15 turned). In a real robot this is the job of SLAM / localization."""
    global _base_tx, _base_last
    now = time.time()
    if now - _base_last < 0.05:
        return
    _base_last = now
    try:
        if _base_tx is None:
            _base_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # The articulation's own root pose (the physics view): the pose of robot.base, a rigid-prim wrapper, jumped by 10-20 cm during arm motions
        # while the transform tree (and the rendered scene) showed the base standing still, which made the executor convert every box wrongly.
        pos, quat = robot.get_world_pose()
        w, x, y, z = (float(v) for v in quat)
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        _base_tx.sendto(json.dumps({"base": [float(pos[0]), float(pos[1]), yaw], "t": now}).encode(), ("127.0.0.1", BASE_POSE_PORT))
    except Exception as exc:  # noqa: BLE001 - a missing pose must never stop the simulation
        if _base_last > 0:
            print(f"[Vision] base pose not published: {exc}", flush=True)
        _base_last = float("inf")


def poll_commands(stage) -> None:
    """Apply the pending scene commands (call once per simulation step)."""
    if _cmd_sock is None:
        return
    while True:
        try:
            msg = json.loads(_cmd_sock.recv(4096))
        except BlockingIOError:
            return
        except ValueError:
            continue
        cmd = msg.get("cmd")
        if cmd == "set_table":
            _place_table(stage, msg["x"], msg["y"], msg["top"])
        elif cmd == "set_box":
            name = msg.get("name", "blue")
            if name in BOXES:
                _place_box(stage, name, msg["x"], msg["y"], msg.get("yaw_deg", 0.0), msg.get("z"))
        elif cmd == "reset":
            _place_table(stage, TABLE_CENTER[0], TABLE_CENTER[1], TABLE_TOP)
            for name, box in BOXES.items():
                _place_box(stage, name, *box["xy"])
        print(f"[Vision] scene command: {msg}", flush=True)


def setup(stage_getter) -> None:
    """Enable the ROS 2 bridge, build the scene and the publishing graph (call after the robot is in the stage)."""
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.ros2.bridge")
    stage = stage_getter()
    cams = build_scene(stage)
    build_ros_graph(cams, stage)
    write_gt(stage, os.environ.get("RBY1_GT_PATH", "/isaac-sim/.local/share/ov/data/rby1_gt.json"))
    _start_command_server()
    print(f"[Vision] cameras {list(cams)} at {IMAGE_W}x{IMAGE_H}, ROS 2 topics under /rby1")
