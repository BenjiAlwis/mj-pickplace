"""Programmatic scene construction with ``mujoco.MjSpec``.

The scene is a UR5e with a Robotiq 2F-85 attached at the tool flange, a table,
a divider wall separating a *pick zone* from a *place zone* (so free-space
motions genuinely require a planner), a 4 cm graspable box and a visual-only
target pad.

Building the scene in code (rather than a hand-edited XML) keeps every
geometric constant in one place, which the planner, the grasp generator and
the evaluator all import.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MENAGERIE = Path(os.environ.get("MENAGERIE_PATH", REPO_ROOT / "third_party" / "mujoco_menagerie"))

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
GRIPPER_PREFIX = "gripper/"
TCP_SITE = GRIPPER_PREFIX + "pinch"
GRIPPER_ACTUATOR = GRIPPER_PREFIX + "fingers_actuator"
OBJECT_BODY = "box"
OBJECT_JOINT = "box_free"

# Planning joint limits (rad). The UR5e allows +-2*pi on most joints; restricting
# to +-pi removes the duplicate configurations that make joint-space planning
# wander, and is what most UR cells use in practice.
JOINT_LIMITS = np.array([[-np.pi, np.pi]] * 6)
# Conservative velocity / acceleration limits (rad/s, rad/s^2), well inside the
# UR5e datasheet (180 deg/s) so the position servos track tightly.
VEL_LIMITS = np.full(6, 1.2)
ACC_LIMITS = np.full(6, 2.5)

HOME_Q = np.array([0.0, -1.9, 1.9, -1.571, -1.571, 0.0])


@dataclass(frozen=True)
class SceneGeometry:
    """Geometric constants of the workcell (metres)."""

    table_center: tuple[float, float] = (0.55, 0.0)
    table_half: tuple[float, float] = (0.35, 0.55)
    table_top_z: float = 0.0
    divider_center: tuple[float, float] = (0.55, 0.0)
    divider_half: tuple[float, float, float] = (0.22, 0.01, 0.10)  # 20 cm tall wall
    box_half: tuple[float, float, float] = (0.02, 0.02, 0.025)
    box_mass: float = 0.1
    # Sampling regions for object / target (x range, y range).
    pick_x: tuple[float, float] = (0.40, 0.65)
    pick_y: tuple[float, float] = (-0.38, -0.14)
    place_x: tuple[float, float] = (0.40, 0.65)
    place_y: tuple[float, float] = (0.14, 0.38)


GEOM = SceneGeometry()


CAM_POS = np.array([1.45, 0.8, 0.95])
CAM_TARGET = np.array([0.35, 0.0, 0.2])


def _lookat_xyaxes(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo camera ``xyaxes`` for a camera at ``pos`` looking at ``target`` (z up)."""
    f = (target - pos) / np.linalg.norm(target - pos)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return np.concatenate([x, y])


def build_spec(geom: SceneGeometry = GEOM) -> mujoco.MjSpec:
    """Assemble the workcell ``MjSpec``."""
    arm = mujoco.MjSpec.from_file(str(MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"))
    grip = mujoco.MjSpec.from_file(str(MENAGERIE / "robotiq_2f85" / "2f85.xml"))

    # The stock keyframe has the wrong qpos size once the gripper/box are added.
    arm.delete(arm.key("home"))

    # Menagerie mounts the base rotated 180 deg about z; undo it so that
    # shoulder_pan = 0 faces the table (+x) and the workspace sits mid-range.
    arm.body("base").quat = [1.0, 0.0, 0.0, 0.0]

    # Planning limits are also the model's limits, so "limits respected" can be
    # checked against the simulator itself.
    for name, (lo, hi) in zip(ARM_JOINTS, JOINT_LIMITS):
        j = arm.joint(name)
        j.range = [lo, hi]

    arm.option.timestep = 0.002
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10
    arm.option.noslip_iterations = 3  # reduces grasp slip under fast transport

    arm.attach(grip, site=arm.site("attachment_site"), prefix=GRIPPER_PREFIX)

    # Gravity compensation for every robot body so the stock position servos
    # (kp=2000) track trajectories without steady-state sag.
    for b in arm.bodies:
        if b.name and b.name != "world":
            b.gravcomp = 1.0

    wb = arm.worldbody
    tex = arm.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                          rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], width=300, height=300)
    arm.add_material(name="grid", textures=["", "grid"], texrepeat=[5, 5], reflectance=0.1)
    _ = tex
    arm.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                    rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=512)
    arm.visual.headlight.ambient = [0.4, 0.4, 0.4]
    arm.visual.global_.offwidth = 1280
    arm.visual.global_.offheight = 960

    wb.add_light(pos=[0.5, 0, 2.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05],
                pos=[0, 0, -0.75], material="grid")

    # Robot pedestal (visual + collision) so the base is not floating.
    wb.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.08, 0.375, 0],
                pos=[0, 0, -0.375], rgba=[0.3, 0.3, 0.3, 1])

    tx, ty = geom.table_center
    hx, hy = geom.table_half
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.02],
                pos=[tx, ty, geom.table_top_z - 0.02], rgba=[0.55, 0.42, 0.3, 1], friction=[1.0, 0.005, 0.0001])
    dx, dy = geom.divider_center
    wb.add_geom(name="divider", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(geom.divider_half),
                pos=[dx, dy, geom.table_top_z + geom.divider_half[2]], rgba=[0.7, 0.7, 0.75, 1])

    box = wb.add_body(name=OBJECT_BODY, pos=[0.5, -0.25, geom.box_half[2]])
    box.add_freejoint(name=OBJECT_JOINT)
    box.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(geom.box_half),
                 mass=geom.box_mass, rgba=[0.85, 0.2, 0.15, 1], friction=[1.2, 0.01, 0.0001],
                 condim=4)

    # Target pad: mocap body so its pose can be set per episode; visual only.
    tgt = wb.add_body(name="target", mocap=True, pos=[0.5, 0.25, 0.0005])
    tgt.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[geom.box_half[0] + 0.006, geom.box_half[1] + 0.006, 0.0005],
                 rgba=[0.1, 0.8, 0.2, 0.6], contype=0, conaffinity=0)

    wb.add_camera(name="overview", pos=list(CAM_POS), xyaxes=list(_lookat_xyaxes(CAM_POS, CAM_TARGET)))

    return arm


def build_model(geom: SceneGeometry = GEOM) -> mujoco.MjModel:
    """Compile the workcell model."""
    return build_spec(geom).compile()


@dataclass
class Indices:
    """Cached address lookups into ``qpos``/``qvel``/``ctrl``."""

    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    arm_ctrl: np.ndarray
    gripper_ctrl: int
    tcp_site: int
    obj_body: int
    obj_qpos: int  # start of the 7-vector free joint
    obj_dof: int
    target_mocap: int
    robot_bodies: frozenset[int]
    gripper_bodies: frozenset[int]
    pad_geoms: frozenset[int]
    obj_geom: int


def indices(model: mujoco.MjModel) -> Indices:
    """Resolve names to indices once."""
    jid = [model.joint(n).id for n in ARM_JOINTS]
    arm_qpos = np.array([model.jnt_qposadr[j] for j in jid])
    arm_dof = np.array([model.jnt_dofadr[j] for j in jid])
    arm_ctrl = np.array([model.actuator(n).id for n in ARM_ACTUATORS])
    oj = model.joint(OBJECT_JOINT).id
    base = model.body("base").id
    robot = set()
    for b in range(model.nbody):
        p = b
        while p != 0:
            if p == base:
                robot.add(b)
                break
            p = model.body_parentid[p]
    gripper = {b for b in range(model.nbody) if model.body(b).name.startswith(GRIPPER_PREFIX)}
    pads = {model.geom(GRIPPER_PREFIX + n).id for n in ("left_pad1", "left_pad2", "right_pad1", "right_pad2")}
    return Indices(
        arm_qpos=arm_qpos,
        arm_dof=arm_dof,
        arm_ctrl=arm_ctrl,
        gripper_ctrl=model.actuator(GRIPPER_ACTUATOR).id,
        tcp_site=model.site(TCP_SITE).id,
        obj_body=model.body(OBJECT_BODY).id,
        obj_qpos=int(model.jnt_qposadr[oj]),
        obj_dof=int(model.jnt_dofadr[oj]),
        target_mocap=int(model.body_mocapid[model.body("target").id]),
        robot_bodies=frozenset(robot),
        gripper_bodies=frozenset(gripper),
        pad_geoms=frozenset(pads),
        obj_geom=model.geom("box_geom").id,
    )
