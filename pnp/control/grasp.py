"""Grasp synthesis and grasp-state sensing for a parallel-jaw gripper.

Candidate generation exploits the object's symmetry: a box with a square
cross-section can be pinched across either pair of side faces, and the 2F-85
is itself symmetric under a half turn, so the top-down candidates are the TCP
yaws ``psi_obj + k * pi/2``, k = 0..3. Each candidate gets an IK solution
(collision-validated, at the *pre-grasp* and *grasp* poses), and feasible
candidates are ranked by weighted joint distance from the current
configuration -- a proxy for motion time that, in particular, avoids needless
half-turns of wrist 3.

Closing direction: the fingers close along the TCP y-axis (measured from the
model: open pads sit at TCP y = +-46.6 mm). With ``R_tcp = Rz(yaw) diag(1,-1,-1)``
the tool z-axis points down, i.e. a top-down approach.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from pnp.ik.solver import DLSSolver
from pnp.planning.rrt_connect import DEFAULT_WEIGHTS
from pnp.se3 import Array, make_T, rot_z
from pnp.sim.collision import CollisionChecker
from pnp.sim.scene import GEOM, Indices

TOP_DOWN = np.diag([1.0, -1.0, -1.0])
GRASP_DEPTH = 0.022  # TCP sits this far below the object's top face (m)
PREGRASP_HEIGHT = 0.12  # approach offset above the grasp pose (m)


def top_down_pose(xyz: Array, yaw: float) -> Array:
    return make_T(rot_z(yaw) @ TOP_DOWN, np.asarray(xyz, dtype=float))


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class GraspCandidate:
    yaw: float
    T_grasp: Array
    T_pre: Array
    q_pre: Array
    q_grasp: Array
    cost: float


def object_pose(data: mujoco.MjData, ix: Indices) -> tuple[Array, float]:
    """Object position and yaw (rotation about world z) from the sim state."""
    p = data.xpos[ix.obj_body].copy()
    R = data.xmat[ix.obj_body].reshape(3, 3)
    return p, float(np.arctan2(R[1, 0], R[0, 0]))


def grasp_candidates(
    ik: DLSSolver,
    checker: CollisionChecker,
    obj_pos: Array,
    obj_yaw: float,
    q_now: Array,
    rng: np.random.Generator,
) -> tuple[list[GraspCandidate], int]:
    """Return feasible candidates sorted by cost, and the number of IK failures."""
    top_z = obj_pos[2] + GEOM.box_half[2]
    grasp_xyz = np.array([obj_pos[0], obj_pos[1], top_z - GRASP_DEPTH])
    cands, ik_fail = [], 0
    for k in range(4):
        yaw = wrap_pi(obj_yaw + k * np.pi / 2)
        T_g = top_down_pose(grasp_xyz, yaw)
        T_p = top_down_pose(grasp_xyz + [0, 0, PREGRASP_HEIGHT], yaw)
        r_pre = ik.solve(T_p, q_now, rng=rng, validator=lambda q: not checker.in_collision(q))
        if not r_pre.success:
            ik_fail += 1
            continue
        # Grasp pose: seeded from pre-grasp, single descent -> same IK branch.
        r_g = ik.solve(T_g, r_pre.q, rng=rng, restarts=0)
        if not r_g.success:
            ik_fail += 1
            continue
        cost = float(np.linalg.norm((r_pre.q - q_now) * DEFAULT_WEIGHTS))
        cands.append(GraspCandidate(yaw, T_g, T_p, r_pre.q, r_g.q, cost))
    cands.sort(key=lambda c: c.cost)
    return cands, ik_fail


def pad_contacts(model: mujoco.MjModel, data: mujoco.MjData, ix: Indices) -> tuple[bool, bool]:
    """Whether the left / right finger pads currently touch the object."""
    left = right = False
    for c in data.contact[: data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if ix.obj_geom not in (g1, g2):
            continue
        other = g2 if g1 == ix.obj_geom else g1
        if other in ix.pad_geoms:
            name = model.geom(other).name
            left |= "left" in name
            right |= "right" in name
    return left, right


def gripper_touching_object(model: mujoco.MjModel, data: mujoco.MjData, ix: Indices) -> bool:
    for c in data.contact[: data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if ix.obj_geom in (g1, g2):
            other = g2 if g1 == ix.obj_geom else g1
            if model.geom_bodyid[other] in ix.gripper_bodies:
                return True
    return False
