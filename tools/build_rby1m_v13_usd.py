"""Build rby1-sim-isaac/assets/model_v_1_3_m.usd from model_v_1_2_m.usd.

RB-Y1 Model M v1.3 differs from v1.2 only in the wrist modules (link_*_arm_4/5/6:
WY / WP / WR) and the end effector.  The base, mecanum wheels, torso, head and
upper arms are byte-identical in the two URDFs, and the end effector is not part
of the robot USD (the modular gripper is mounted at run time), so the v1.3 asset
is the hand-tuned v1.2 asset with the six wrist links swapped:

  * visual meshes      -> rby1-sdk models/rby1m/urdf/meshes/LINK_{11,12,13,18,19,20}_W?.dae
  * mass / CoM / inertia -> model_v1.3.urdf <inertial>
  * *_arm_4 joint origin  z: -0.256 -> -0.3095
  * *_arm_5 joint limits  [-90, 110] deg -> [-50, 50] deg
  * *_arm_6 joint axis    Z -> X, limits +-155 deg -> +-90 deg

Everything else (drive torque limits, max joint velocities, colliders,
articulation settings, wheel models) is kept from v1.2.

Run with the Isaac Sim python from a directory that does not contain an
``isaacsim`` folder (it shadows the pip package):

    cd /tmp && OMNI_KIT_ACCEPT_EULA=YES \
      ~/miniforge3/envs/env_isaaclab/bin/python \
      ~/Downloads/isaac_rby1/tools/build_rby1m_v13_usd.py
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import asyncio  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import omni.kit.asset_converter as asset_converter  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ASSETS = os.path.expanduser("~/rby1-sim-isaac/assets")
SRC = os.path.join(ASSETS, "model_v_1_2_m.usd")
DST = os.path.join(ASSETS, "model_v_1_3_m.usd")
MESH_DIR = os.path.expanduser("~/rby1/rby1-sdk/models/rby1m/urdf/meshes")
OLD_ROOT = "/RBY1_M_v1_2"
NEW_ROOT = "/RBY1_M_v1_3"

# URDF "black" material; the robot's other black parts use this value.
BLACK = Gf.Vec3f(0.039, 0.039, 0.039)

# model_v1.3.urdf <inertial> for the wrist links: mass, CoM, (ixx, ixy, ixz, iyy, iyz, izz)
WRIST_INERTIAL = {
    "link_right_arm_4": (1.49373286, (-0.00006159, 0.00402933, 0.09539632),
                         (0.00327354, -0.00000405, -0.00000027, 0.00252231, -0.00001749, 0.00186828)),
    "link_right_arm_5": (0.87163780, (0.00178191, 0.00092083, 0.00006831),
                         (0.00047456, 0.00002110, 0.00000222, 0.00056446, -0.00000011, 0.00067619)),
    "link_right_arm_6": (0.74448182, (0.00706872, -0.00102308, -0.08649522),
                         (0.00187469, -0.00003209, -0.00035074, 0.00160335, 0.00000005, 0.00115053)),
    "link_left_arm_4": (1.49373286, (0.00006159, -0.00402933, 0.09539632),
                        (0.00327354, -0.00000405, 0.00000027, 0.00252231, 0.00001749, 0.00186828)),
    "link_left_arm_5": (0.87163780, (0.00178191, 0.00092083, 0.00006831),
                        (0.00047456, -0.00002110, -0.00000222, 0.00056446, -0.00000011, 0.00067619)),
    "link_left_arm_6": (0.74448182, (0.00706872, -0.00102308, -0.08649522),
                        (0.00187469, -0.00003209, -0.00035074, 0.00160335, 0.00000005, 0.00115053)),
}

# model_v1.2.urdf values, used only to check that the principal-axes math
# reproduces what the v1.2 asset already contains.
WRIST_INERTIAL_V12 = {
    "link_right_arm_4": (0.42508710, (-0.00000011, -0.02696036, 0.06182110),
                         (0.00082466, 0.0, 0.0, 0.00075398, -0.00025338, 0.00039794)),
    "link_right_arm_5": (1.60506693, (0.00592642, -0.00537804, -0.02584713),
                         (0.00247557, -0.00004984, -0.00009861, 0.00268241, 0.00021452, 0.00166724)),
    "link_right_arm_6": (0.42751673, (0.00024951, -0.00019562, -0.10711314),
                         (0.00029613, -0.00000186, -0.00000157, 0.00030651, 0.00000221, 0.00048946)),
}

WRIST_MESHES = {
    "link_right_arm_4": "LINK_11_WY",
    "link_right_arm_5": "LINK_12_WP",
    "link_right_arm_6": "LINK_13_WR",
    "link_left_arm_4": "LINK_18_WY",
    "link_left_arm_5": "LINK_19_WP",
    "link_left_arm_6": "LINK_20_WR",
}

# joint name -> (localPos0 z, axis, lower deg, upper deg); None keeps the v1.2 value
WRIST_JOINTS = {
    "right_arm_4": (-0.3095, None, None, None),
    "left_arm_4": (-0.3095, None, None, None),
    "right_arm_5": (None, None, -50.0, 50.0),
    "left_arm_5": (None, None, -50.0, 50.0),
    "right_arm_6": (None, "X", -90.0, 90.0),
    "left_arm_6": (None, "X", -90.0, 90.0),
}


def principal_axes(ixx, ixy, ixz, iyy, iyz, izz):
    """URDF inertia tensor -> (diagonal inertia, principal axes quaternion wxyz).

    Matches the Isaac Sim URDF importer: eigenvectors are assigned to the body
    axis they are most aligned with and signed so the rotation stays close to
    identity (not sorted by eigenvalue).
    """
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    values, vectors = np.linalg.eigh(tensor)
    order = [-1, -1, -1]
    taken = set()
    for axis in np.argsort(-np.abs(vectors).max(axis=0)):  # most decisive eigenvector first
        candidates = np.argsort(-np.abs(vectors[:, axis]))
        target = next(c for c in candidates if c not in taken)
        order[target] = axis
        taken.add(target)
    rot = vectors[:, order].copy()
    diag = values[order]
    for i in range(3):
        if rot[i, i] < 0:
            rot[:, i] *= -1
    if np.linalg.det(rot) < 0:  # flip the column with the weakest alignment
        rot[:, int(np.argmin(np.diag(rot)))] *= -1
    q = Rotation.from_matrix(rot).as_quat()  # x, y, z, w
    return Gf.Vec3f(*map(float, diag)), Gf.Quatf(float(q[3]), Gf.Vec3f(float(q[0]), float(q[1]), float(q[2])))


def quat_close(a, b, tol=1e-4):
    va = np.array([a.GetReal(), *a.GetImaginary()])
    vb = np.array([b.GetReal(), *b.GetImaginary()])
    return min(np.abs(va - vb).max(), np.abs(va + vb).max()) < tol


def check_principal_axes_against_v12(stage):
    for link, (mass, com, tensor) in WRIST_INERTIAL_V12.items():
        prim = stage.GetPrimAtPath(f"{OLD_ROOT}/{link}")
        mass_api = UsdPhysics.MassAPI(prim)
        diag, quat = principal_axes(*tensor)
        have_diag = mass_api.GetDiagonalInertiaAttr().Get()
        have_quat = mass_api.GetPrincipalAxesAttr().Get()
        ok = np.abs(np.array(diag) - np.array(have_diag)).max() < 1e-6 and quat_close(quat, have_quat)
        print(f"[check] {link}: diag {tuple(diag)} vs {tuple(have_diag)} | axes {quat} vs {have_quat} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("principal-axes computation does not reproduce the v1.2 asset")


def set_inertial(stage, link, mass, com, tensor):
    prim = stage.GetPrimAtPath(f"{OLD_ROOT}/{link}")
    mass_api = UsdPhysics.MassAPI(prim)
    diag, quat = principal_axes(*tensor)
    mass_api.GetMassAttr().Set(float(mass))
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com))
    mass_api.GetDiagonalInertiaAttr().Set(diag)
    mass_api.GetPrincipalAxesAttr().Set(quat)
    print(f"[inertial] {link}: mass={mass} com={com} diag={tuple(diag)} axes={quat}")


def set_joint(stage, name, pos0_z, axis, lower, upper):
    prim = stage.GetPrimAtPath(f"{OLD_ROOT}/joints/{name}")
    joint = UsdPhysics.RevoluteJoint(prim)
    if pos0_z is not None:
        pos0 = joint.GetLocalPos0Attr().Get()
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(pos0[0], pos0[1], pos0_z))
    if axis is not None:
        joint.GetAxisAttr().Set(axis)
    if lower is not None:
        joint.GetLowerLimitAttr().Set(float(lower))
        joint.GetUpperLimitAttr().Set(float(upper))
    print(f"[joint] {name}: pos0={joint.GetLocalPos0Attr().Get()} axis={joint.GetAxisAttr().Get()} "
          f"limits=({joint.GetLowerLimitAttr().Get()}, {joint.GetUpperLimitAttr().Get()}) "
          f"maxForce={prim.GetAttribute('drive:angular:physics:maxForce').Get()} "
          f"maxVel={prim.GetAttribute('physxJoint:maxJointVelocity').Get()}")


def shift_wrist_rest_poses(stage):
    """Keep the authored (zero-configuration) link poses consistent with the joints.

    The importer authored each link's xformOp:translate as its world pose at
    q = 0.  The *_arm_4 joint origin moved 53.5 mm down, so the wrist links it
    carries (arm_4/5/6, both sides) move down by the same amount; otherwise the
    bodies start 53.5 mm away from where their joints put them.
    """
    dz = WRIST_JOINTS["right_arm_4"][0] - (-0.256)
    for link in WRIST_MESHES:
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(f"{OLD_ROOT}/{link}"))
        op = next(o for o in xform.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        old = op.Get()
        assert abs(old[2] - 0.854) < 1e-6, f"{link}: unexpected rest pose {old}"
        op.Set(Gf.Vec3d(old[0], old[1], old[2] + dz))
        print(f"[rest pose] {link}: {tuple(old)} -> {tuple(op.Get())}")


def convert_dae(name, out_dir):
    src = os.path.join(MESH_DIR, f"{name}.dae")
    dst = os.path.join(out_dir, f"{name}.usd")
    ctx = asset_converter.AssetConverterContext()
    ctx.ignore_materials = False
    ctx.ignore_animations = True
    ctx.ignore_camera = True
    ctx.ignore_light = True
    ctx.single_mesh = True
    ctx.merge_all_meshes = True
    ctx.use_meter_as_world_unit = True
    ctx.create_world_as_default_root_prim = True
    ctx.embed_textures = True
    ctx.convert_fbx_to_y_up = False
    ctx.convert_fbx_to_z_up = False
    task = asset_converter.get_instance().create_converter_task(src, dst, None, ctx)
    ok = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
    if not ok:
        raise SystemExit(f"asset converter failed on {src}: {task.get_status()} {task.get_error_message()}")
    return dst


def mesh_bbox(stage, root_path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(root_path)).ComputeAlignedRange()
    lo, hi = box.GetMin(), box.GetMax()
    return tuple(round(v, 4) for v in lo), tuple(round(v, 4) for v in hi)


def replace_visual(stage, link, mesh_name, tmp_dir):
    """Replace the instanced v1.2 visual of ``link`` with the converted v1.3 mesh."""
    vis = stage.GetPrimAtPath(f"{OLD_ROOT}/{link}/visuals")
    old_refs = [str(r.primPath) for r in vis.GetMetadata("references").GetAddedOrExplicitItems()] if vis.GetMetadata("references") else []
    vis.SetInstanceable(False)
    vis.GetReferences().ClearReferences()

    holder = UsdGeom.Xform.Define(stage, vis.GetPath().AppendChild(mesh_name))
    holder.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    holder.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, Gf.Vec3d(0, 0, 0)))
    holder.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))

    converted = convert_dae(mesh_name, tmp_dir)
    conv_layer = Sdf.Layer.FindOrOpen(converted)
    scene_path = holder.GetPath().AppendChild("Scene")
    if not Sdf.CopySpec(conv_layer, Sdf.Path("/World"), stage.GetRootLayer(), scene_path):
        raise SystemExit(f"CopySpec failed for {mesh_name}")

    # The DAE carries no material; apply the URDF "black".
    looks = stage.GetPrimAtPath(scene_path.AppendChild("Looks"))
    for material in looks.GetChildren():
        for shader in material.GetChildren():
            attr = shader.GetAttribute("inputs:diffuse_color_constant")
            if attr:
                attr.Set(BLACK)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(scene_path.AppendChild("mesh")))
    bound = UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()[0]
    npts = len(mesh.GetPointsAttr().Get())
    print(f"[visual] {link}: {old_refs} -> {mesh_name} ({npts} points, material {bound.GetPath() if bound else None}) "
          f"bbox={mesh_bbox(stage, str(vis.GetPath()))}")
    return old_refs


GRIPPER_DIR = os.path.join(ASSETS, "gripper", "rb_gripper")
EE_MOUNT_Z = -0.125  # URDF END_right / END_left
FINGER_TRAVEL = 0.05
# model_v1.3.urdf ee_right / ee_left <inertial> (identical on both sides)
EE_INERTIAL = (0.47470074, (0.00041345, 0.00160451, -0.02702162),
               (0.00053136, 0.00001907, 0.00000050, 0.00031059, 0.00000880, 0.00060261))
FINGER_MASS = 0.03326999  # unchanged from v1.2; checked against the copied bodies

ROT_Z_POS90 = Gf.Quatf(0.70710678, Gf.Vec3f(0, 0, 0.70710678))
ROT_Z_180 = Gf.Quatf(0, Gf.Vec3f(0, 0, 1))
ROT_Z_NEG90 = Gf.Quatf(0.70710678, Gf.Vec3f(0, 0, -0.70710678))


def set_rest_pose(stage, link, translate, orient=None):
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(f"{OLD_ROOT}/{link}"))
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*translate))
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient is not None:
            op.Set(Gf.Quatd(orient.GetReal(), Gf.Vec3d(*orient.GetImaginary())))


def add_integrated_gripper(stage, side, tmp_dir):
    """Build the v1.3 end effector into the robot from the rb_gripper asset.

    v1.3 has no detachable gripper: ee_<side> (EE_GR_TF body) hangs from
    link_<side>_arm_6 through the fixed joint END_<side>, and the two finger
    bodies ride on prismatic joints whose frames, axes and limits follow
    model_v1.3.urdf.  Bodies, finger meshes, drives and the finger-2 mimic
    setup are copied from assets/gripper/rb_gripper/rb_gripper_<side>.usd.
    """
    s = side[0]
    grip_layer = Sdf.Layer.FindOrOpen(os.path.join(GRIPPER_DIR, f"rb_gripper_{side}.usd"))
    grip_root = f"/Gripper_{side.capitalize()}"
    layer = stage.GetRootLayer()
    ee, f1, f2 = f"ee_{side}", f"ee_finger_{s}1", f"ee_finger_{s}2"
    j1, j2, j_end = f"gripper_finger_{s}1", f"gripper_finger_{s}2", f"END_{side}"

    copies = {f"{grip_root}/{n}": f"{OLD_ROOT}/{n}" for n in (ee, f1, f2)}
    copies.update({f"{grip_root}/joints/{n}": f"{OLD_ROOT}/joints/{n}" for n in (j1, j2)})
    for src, dst in copies.items():
        if not grip_layer.GetPrimAtPath(src):
            raise SystemExit(f"{src} missing in {grip_layer.identifier}")
        if layer.GetPrimAtPath(dst):
            raise SystemExit(f"{dst} already exists")
        if not Sdf.CopySpec(grip_layer, Sdf.Path(src), layer, Sdf.Path(dst)):
            raise SystemExit(f"CopySpec failed for {src}")
    # relationships (joint bodies, mimic reference joint) still point into the gripper file
    for dst in copies.values():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(dst)):
            for rel in prim.GetRelationships():
                targets = rel.GetTargets()
                new = [Sdf.Path(str(t).replace(grip_root, OLD_ROOT, 1)) for t in targets]
                if new != list(targets):
                    rel.SetTargets(new)

    # finger visuals in rb_gripper_right.usd are internal references to prototypes of that file
    for dst in copies.values():
        for prim in Usd.PrimRange(stage.GetPrimAtPath(dst)):
            refs = prim.GetMetadata("references")
            if not refs:
                continue
            items, changed = [], False
            for ref in refs.GetAddedOrExplicitItems():
                if not ref.assetPath and not layer.GetPrimAtPath(ref.primPath):
                    new_path = Sdf.Path(f"/Flattened_Prototype_{side}_{ref.primPath.name.rsplit('_', 1)[-1]}")
                    if not layer.GetPrimAtPath(new_path):
                        if not Sdf.CopySpec(grip_layer, ref.primPath, layer, new_path):
                            raise SystemExit(f"CopySpec failed for prototype {ref.primPath}")
                    print(f"[gripper] {prim.GetPath()}: prototype {ref.primPath} -> {new_path}")
                    ref, changed = Sdf.Reference(primPath=new_path), True
                items.append(ref)
            if changed:
                prim.GetReferences().SetReferences(items)

    arm6 = UsdGeom.Xformable(stage.GetPrimAtPath(f"{OLD_ROOT}/link_{side}_arm_6"))
    arm6_pos = next(o.Get() for o in arm6.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    ee_pos = (arm6_pos[0], arm6_pos[1], arm6_pos[2] + EE_MOUNT_Z)

    # --- end-effector body: EE_GR_TF visual (URDF visual origin rpy = (pi, 0, 0)), v1.3 inertial
    ee_prim = stage.GetPrimAtPath(f"{OLD_ROOT}/{ee}")
    for child in list(ee_prim.GetChildren()):
        if child.GetName() in ("EE_BODY", "visuals"):
            stage.RemovePrim(child.GetPath())
    vis = UsdGeom.Xform.Define(stage, ee_prim.GetPath().AppendChild("visuals"))
    holder = UsdGeom.Xform.Define(stage, vis.GetPath().AppendChild("EE_GR_TF"))
    holder.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 0))
    holder.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0, Gf.Vec3d(1, 0, 0)))  # 180 deg about X
    holder.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 1, 1))
    conv_layer = Sdf.Layer.FindOrOpen(convert_dae("EE_GR_TF", tmp_dir))
    scene_path = holder.GetPath().AppendChild("Scene")
    if not Sdf.CopySpec(conv_layer, Sdf.Path("/World"), layer, scene_path):
        raise SystemExit("CopySpec failed for EE_GR_TF")
    for material in stage.GetPrimAtPath(scene_path.AppendChild("Looks")).GetChildren():
        for shader in material.GetChildren():
            attr = shader.GetAttribute("inputs:diffuse_color_constant")
            if attr:
                attr.Set(BLACK)
    set_inertial(stage, ee, *EE_INERTIAL)
    set_rest_pose(stage, ee, ee_pos, Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))

    # --- fingers: same mesh and inertial as v1.2, new rest frames (joint origin rpy = (0, 0, +-pi/2))
    for finger, orient in ((f1, ROT_Z_POS90), (f2, ROT_Z_NEG90)):
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{OLD_ROOT}/{finger}")).GetMassAttr().Get()
        if abs(mass - FINGER_MASS) > 1e-6:
            raise SystemExit(f"{finger}: unexpected mass {mass}")
        set_rest_pose(stage, finger, ee_pos, orient)
        # rb_gripper_left.usd binds LINK_9 / LINK_3 (arm and torso meshes) to its finger
        # visuals; every finger uses the EE_FINGER mesh, so point all of them at the
        # right-side prototype (copied above, since the right side is built first).
        vis = stage.GetPrimAtPath(f"{OLD_ROOT}/{finger}/visuals")
        proto = Sdf.Path("/Flattened_Prototype_right_258")
        if not layer.GetPrimAtPath(proto):
            raise SystemExit("EE_FINGER prototype missing; build the right side first")
        for child in list(vis.GetChildren()):
            if not child.IsInstanceProxy():
                stage.RemovePrim(child.GetPath())
        vis.GetReferences().SetReferences([Sdf.Reference(primPath=proto)])
        vis.SetInstanceable(True)
        print(f"[gripper] {finger}: visuals -> {proto} bbox={mesh_bbox(stage, str(vis.GetPath()))}")

    # --- prismatic finger joints.  URDF: *1 frame rotZ(+90), axis +X, [0, 0.05];
    #     *2 frame rotZ(-90), axis -X, [-0.05, 0].  As the importer does for a
    #     negative axis, *2 gets both joint frames turned 180 deg about Z so the
    #     USD axis is +X: localRot0 = rotZ(-90)*rotZ(180) = rotZ(+90), localRot1 = rotZ(180).
    for name, rot0, rot1, lower, upper in (
        (j1, ROT_Z_POS90, Gf.Quatf(1, Gf.Vec3f(0, 0, 0)), 0.0, FINGER_TRAVEL),
        (j2, ROT_Z_POS90, ROT_Z_180, -FINGER_TRAVEL, 0.0),
    ):
        prim = stage.GetPrimAtPath(f"{OLD_ROOT}/joints/{name}")
        joint = UsdPhysics.PrismaticJoint(prim)
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.GetLocalRot0Attr().Set(rot0)
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.GetLocalRot1Attr().Set(rot1)
        joint.GetAxisAttr().Set("X")
        joint.GetLowerLimitAttr().Set(lower)
        joint.GetUpperLimitAttr().Set(upper)
        prim.GetAttribute("drive:linear:physics:targetPosition").Set(0.0)
        props = {a.GetName(): a.Get() for a in prim.GetAttributes()
                 if a.IsAuthored() and a.GetName().startswith(("drive:", "physxJoint", "physxMimic"))}
        rels = {r.GetName(): [str(t) for t in r.GetTargets()] for r in prim.GetRelationships()}
        print(f"[gripper] {name}: b0/b1={rels.get('physics:body0')}/{rels.get('physics:body1')} "
              f"limits=({lower}, {upper}) rot0={rot0} rot1={rot1} {props} "
              f"mimic={ {k: v for k, v in rels.items() if 'imic' in k} }")

    # --- END_<side>: fixed joint arm_6 -> ee, like the v1.2 head_base joint
    end = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{OLD_ROOT}/joints/{j_end}"))
    end.GetBody0Rel().SetTargets([Sdf.Path(f"{OLD_ROOT}/link_{side}_arm_6")])
    end.GetBody1Rel().SetTargets([Sdf.Path(f"{OLD_ROOT}/{ee}")])
    end.GetLocalPos0Attr().Set(Gf.Vec3f(0, 0, EE_MOUNT_Z))
    end.GetLocalRot0Attr().Set(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    end.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    end.GetLocalRot1Attr().Set(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    end.CreateCollisionEnabledAttr(False)
    end.CreateExcludeFromArticulationAttr(False)

    root = stage.GetPrimAtPath(OLD_ROOT)
    for rel_name, names in (("isaac:physics:robotLinks", (ee, f1, f2)),
                            ("isaac:physics:robotJoints", (j_end, j1, j2))):
        rel = root.GetRelationship(rel_name)
        if rel:
            prefix = f"{OLD_ROOT}/joints/" if "Joints" in rel_name else f"{OLD_ROOT}/"
            targets = list(rel.GetTargets())  # v1.2 still lists the old ee prims; keep each path once
            rel.SetTargets(targets + [Sdf.Path(prefix + n) for n in names if Sdf.Path(prefix + n) not in targets])
    print(f"[gripper] {side}: ee rest pose {ee_pos}, EE_GR_TF bbox={mesh_bbox(stage, str(vis.GetPath()))}")


def remove_unreferenced_prototypes(stage, expected):
    """Delete the internal Flattened_Prototype_* roots nothing references.

    The v1.2 asset already carries ~210 leftover prototypes that nothing uses
    (the same residue commit af05f78 removed from the gripper files); the six
    replaced wrist visuals (``expected``) join them.  If any of those six is
    still reported as referenced the scan is wrong, so stop before saving.
    """
    layer = stage.GetRootLayer()
    referenced = set()

    def visit(spec):
        for ref in spec.referenceList.GetAddedOrExplicitItems():
            if not ref.assetPath:
                referenced.add(str(ref.primPath))
        for child in spec.nameChildren:
            visit(child)

    for root in layer.rootPrims:
        visit(root)
    unreferenced = sorted(
        "/" + root.name for root in layer.rootPrims
        if root.name.startswith("Flattened_Prototype_") and "/" + root.name not in referenced
    )
    print(f"[cleanup] {len(referenced)} prototypes referenced, {len(unreferenced)} unreferenced "
          f"(replaced wrist visuals: {sorted(expected)})")
    missing = set(expected) - set(unreferenced)
    if missing:
        raise SystemExit(f"replaced visual prototypes still referenced: {sorted(missing)}")
    for path in unreferenced:
        stage.RemovePrim(path)


def dangling_targets(stage):
    bad = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        for rel in prim.GetRelationships():
            for target in rel.GetTargets():
                if not stage.GetPrimAtPath(target.GetPrimPath()):
                    bad.append((str(rel.GetPath()), str(target)))
        refs = prim.GetMetadata("references")
        if refs:
            for ref in refs.GetAddedOrExplicitItems():
                if not ref.assetPath and not stage.GetPrimAtPath(ref.primPath):
                    bad.append((str(prim.GetPath()), f"ref {ref.primPath}"))
    return bad


def build():
    shutil.copyfile(SRC, DST)
    usd_context = omni.usd.get_context()
    if not usd_context.open_stage(DST):
        raise SystemExit(f"could not open {DST}")
    stage = usd_context.get_stage()

    # v1.2 wrist bboxes, for comparison with the v1.3 meshes
    for link in WRIST_MESHES:
        print(f"[v1.2 visual] {link}: bbox={mesh_bbox(stage, f'{OLD_ROOT}/{link}/visuals')}")

    check_principal_axes_against_v12(stage)

    tmp_dir = tempfile.mkdtemp(prefix="rby1m_v13_meshes_")
    old_visual_prototypes = []
    for link, mesh_name in WRIST_MESHES.items():
        old_visual_prototypes += replace_visual(stage, link, mesh_name, tmp_dir)
    for link, (mass, com, tensor) in WRIST_INERTIAL.items():
        set_inertial(stage, link, mass, com, tensor)
    for name, args in WRIST_JOINTS.items():
        set_joint(stage, name, *args)
    shift_wrist_rest_poses(stage)
    for side in ("right", "left"):
        add_integrated_gripper(stage, side, tmp_dir)

    remove_unreferenced_prototypes(stage, old_visual_prototypes)

    omni.kit.commands.execute("MovePrim", path_from=OLD_ROOT, path_to=NEW_ROOT, keep_world_transform=False)
    stage.SetDefaultPrim(stage.GetPrimAtPath(NEW_ROOT))

    # The v1.2 asset's IsaacRobotAPI link/joint lists still name the end-effector
    # prims that were removed when the modular gripper was introduced.
    for rel_name in ("isaac:physics:robotLinks", "isaac:physics:robotJoints"):
        rel = stage.GetPrimAtPath(NEW_ROOT).GetRelationship(rel_name)
        if rel:
            targets = rel.GetTargets()
            live = [t for t in targets if stage.GetPrimAtPath(t)]
            rel.SetTargets(live)
            print(f"[cleanup] {rel_name}: dropped {len(targets) - len(live)} stale targets "
                  f"{[t.name for t in targets if t not in live]}, {len(live)} kept")

    bad = dangling_targets(stage)
    print(f"[check] dangling relationship/reference targets: {bad if bad else 'none'}")
    if bad:
        raise SystemExit("dangling targets after rename")

    layer = stage.GetRootLayer()
    layer.documentation = (
        "RB-Y1 Model M v1.3 (RBY1_M_v1_3). Built from model_v_1_2_m.usd by "
        "tools/build_rby1m_v13_usd.py: wrist links link_*_arm_4/5/6 (WY/WP/WR) take their visual "
        "meshes, inertial properties, joint origin, axes and limits from rby1-sdk "
        "models/rby1m/urdf/model_v1.3.urdf; everything else is the v1.2 asset.\n\n" + layer.documentation
    )
    layer.Save()
    usd_context.close_stage()
    print(f"[build] saved {DST} ({os.path.getsize(DST) / 1e6:.1f} MB)")


def verify():
    stage = Usd.Stage.Open(DST)
    print(f"[verify] defaultPrim={stage.GetDefaultPrim().GetPath()} roots={[p.GetName() for p in stage.GetPseudoRoot().GetChildren()]}")
    assert not stage.GetPrimAtPath(OLD_ROOT)
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    print(f"[verify] joints={len(joints)} rigid bodies={len(bodies)} "
          f"articulation roots={[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]}")
    for link in WRIST_MESHES:
        print(f"[verify] {link}: bbox={mesh_bbox(stage, f'{NEW_ROOT}/{link}/visuals')}")
    print(f"[verify] dangling: {dangling_targets(stage) or 'none'}")

    # Load it the way rby1-sim-isaac does and step the articulation.
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=DST, prim_path="/World/rby1")
    robot = Articulation("/World/rby1")
    world.reset()
    print(f"[sim] dof_names ({robot.num_dof}): {robot.dof_names}")
    print(f"[sim] body_names ({robot.num_bodies}): {robot.body_names}")
    limits = robot.get_dof_limits()[0]
    for i, name in enumerate(robot.dof_names):
        if name.endswith(("arm_4", "arm_5", "arm_6")):
            print(f"[sim] {name}: limits=({math.degrees(limits[i][0]):.1f}, {math.degrees(limits[i][1]):.1f}) deg "
                  f"max_effort={robot.get_max_efforts()[0][i]:.1f}")
    for _ in range(120):
        world.step(render=False)
    q = robot.get_joint_positions()[0]
    pos, _ = robot.get_world_poses()
    print(f"[sim] after 120 steps: base pos={pos[0]} joint pos finite={bool(np.isfinite(q).all())} "
          f"|q|max={float(np.abs(q).max()):.3f}")

    # Open both grippers through the finger-1 drives; finger 2 should follow through its mimic.
    fingers = {n: robot.dof_names.index(n) for n in ("gripper_finger_r1", "gripper_finger_r2", "gripper_finger_l1", "gripper_finger_l2")}
    idx = [fingers["gripper_finger_r1"], fingers["gripper_finger_l1"]]
    robot.set_joint_position_targets(np.array([[FINGER_TRAVEL, FINGER_TRAVEL]]), joint_indices=np.array(idx))
    for _ in range(240):
        world.step(render=False)
    q = robot.get_joint_positions()[0]
    print("[sim] finger positions after opening: " + ", ".join(f"{n}={q[i]:+.4f}" for n, i in fingers.items()))
    ee_idx = [robot.body_names.index(n) for n in ("ee_right", "ee_finger_r1", "ee_finger_r2", "link_right_arm_6")]
    body_pos = robot.get_link_world_poses()[0] if hasattr(robot, "get_link_world_poses") else None
    if body_pos is not None:
        print("[sim] ee_right / fingers / arm_6 world positions: " + "; ".join(str(np.round(body_pos[0][i], 4)) for i in ee_idx))
    world.clear()


if __name__ == "__main__":
    import traceback
    try:
        build()
        verify()
    except BaseException:
        traceback.print_exc()  # app.close() exits the process before Python would print it
    finally:
        app.close()
