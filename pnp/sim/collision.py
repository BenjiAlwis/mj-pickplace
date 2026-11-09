"""Configuration-space collision checking backed by MuJoCo's narrow phase.

A :class:`CollisionChecker` owns a private model copy and ``MjData``, so
planning never perturbs the live simulation. For each query it writes the arm
configuration (and, when an object is attached, the object pose implied by the
grasp transform), runs ``mj_kinematics`` + ``mj_collision`` and classifies
contacts:

* robot vs. environment (table, divider, pedestal, free object) -> collision
* robot vs. robot (gripper-internal linkage pairs ignored)      -> collision
* gripper vs. object while *attached*                          -> allowed (the grasp)
* attached object vs. environment                              -> collision
* free object vs. table                                        -> ignored (resting)

**Clearance margin.** ``geom_margin`` is inflated on a private model (set in the
spec *before* compiling -- see ``__init__``) so MuJoCo reports near-contacts,
and any contact with ``dist < margin`` is a collision.
(MuJoCo *adds* the two geoms' margins, so filtering on ``dist`` explicitly is
what makes the clearance exactly ``margin`` -- verified empirically.)

**Edge checking with a guarantee.** Rather than a fixed joint-space step, edges
are discretised with the swept-displacement bound of Schwarzer et al. (2004):
if joint *i* moves by ``|dq_i|``, no robot point moves more than
``sum_i r_i |dq_i|``, where ``r_i`` bounds the distance from joint *i* to any
collision geometry distal to it (computed from link offsets and geom bounding
radii; extended by the held object when one is attached). Sampling so that
this bound is ``<= margin`` per step means every intermediate configuration is
within ``margin/2`` of a checked one for static obstacles and within
``margin`` for self-collision pairs, so a free edge at the samples is free in
continuum with clearance ``>= 0``.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import mujoco
import numpy as np

from pnp.se3 import Array, mat_to_quat
from pnp.sim.scene import ARM_JOINTS, Indices, build_spec, indices

ALLOWED_BODY_GEOM: tuple[tuple[str, str], ...] = (("shoulder_link", "pedestal"),)


@dataclass
class CollisionStats:
    checks: int = 0


class CollisionChecker:
    def __init__(self, model: mujoco.MjModel, margin: float = 0.01) -> None:
        # Private model compiled with the inflated margin. Setting
        # ``geom_margin`` on an already-compiled model is NOT enough: MuJoCo's
        # mid-phase bounding-volume hierarchy is built at compile time, and the
        # test ``test_margin_is_exact`` showed runtime margin edits silently
        # miss near-contacts (the finger pads were never reported).
        spec = build_spec()
        for g in spec.geoms:
            g.margin = margin
        self.model = spec.compile()
        if self.model.nq != model.nq or self.model.ngeom != model.ngeom:
            raise ValueError("CollisionChecker expects the workcell model from pnp.sim.scene")
        self.margin = margin
        self.data = mujoco.MjData(self.model)
        self.ix: Indices = indices(self.model)
        m = self.model
        self.geom_body = m.geom_bodyid.copy()
        self.is_robot_geom = np.array([b in self.ix.robot_bodies for b in self.geom_body])
        self.is_gripper_geom = np.array([b in self.ix.gripper_bodies for b in self.geom_body])
        self.table_geom = m.geom("table").id
        # Allowed-collision entries (as in a MoveIt SRDF): the shoulder link
        # rotates 3 mm above its mounting pedestal, so that pair is always
        # inside the clearance margin and must be whitelisted explicitly.
        self.allowed = {(m.body(b).id, m.geom(g).id) for b, g in ALLOWED_BODY_GEOM}  # (body, geom)
        self.attached_T: Array | None = None  # TCP -> object transform
        # Contact relaxations for the near-contact Cartesian phases (descending
        # around the object, lifting it off / setting it down on the table).
        self.allow_gripper_object = False
        self.allow_object_table = False
        self.stats = CollisionStats()
        self._base_lever = self._lever_arms()
        self.lever = self._base_lever.copy()

    # --------------------------------------------------------------- geometry
    def _lever_arms(self, extra_tool_radius: float = 0.0) -> Array:
        """Upper bounds r_i on the distance from joint i to any distal collision point."""
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[self.ix.obj_qpos + 3] = 1.0
        mujoco.mj_kinematics(m, d)
        jid = [m.joint(n).id for n in ARM_JOINTS]
        anchors = d.xanchor[jid].copy()
        jbody = [m.jnt_bodyid[j] for j in jid]
        rho = np.zeros(len(jid))  # reach of geometry owned by each joint's link group
        for g in range(m.ngeom):
            if not self.is_robot_geom[g] or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            b, owner = int(m.geom_bodyid[g]), None
            while b != 0 and owner is None:
                if b in jbody:
                    owner = jbody.index(b)
                b = int(m.body_parentid[b])
            if owner is not None:
                r = np.linalg.norm(d.geom_xpos[g] - anchors[owner]) + m.geom_rbound[g]
                rho[owner] = max(rho[owner], r)
        rho[-1] += extra_tool_radius
        seg = np.linalg.norm(np.diff(anchors, axis=0), axis=1)  # config-invariant
        n = len(jid)
        r = np.zeros(n)
        for i in range(n):
            r[i] = max(sum(seg[i:j]) + rho[j] for j in range(i, n))
        return r

    # ------------------------------------------------------------------ state
    def sync(self, live: mujoco.MjData) -> None:
        """Copy the full state (gripper finger joints, object pose) from the live sim."""
        self.data.qpos[:] = live.qpos
        self.data.mocap_pos[:] = live.mocap_pos
        self.data.mocap_quat[:] = live.mocap_quat

    def attach(self, T_tcp_obj: Array | None) -> None:
        """Treat the object as rigidly attached to the TCP (``None`` detaches)."""
        if T_tcp_obj is None:
            self.attached_T = None
            self.lever = self._base_lever.copy()
            return
        self.attached_T = T_tcp_obj.copy()
        # Object extends the tool: TCP offset from the wrist-3 anchor is already in
        # rho_6, so add the object's offset from the TCP plus its bounding radius.
        obj_r = float(np.linalg.norm(T_tcp_obj[:3, 3]) + self.model.geom_rbound[self.ix.obj_geom])
        self.lever = self._base_lever + obj_r

    @contextmanager
    def relaxed(self, gripper_object: bool = False, object_table: bool = False) -> Iterator[None]:
        """Temporarily allow the named contact classes."""
        old = (self.allow_gripper_object, self.allow_object_table)
        self.allow_gripper_object, self.allow_object_table = gripper_object, object_table
        try:
            yield
        finally:
            self.allow_gripper_object, self.allow_object_table = old

    # ---------------------------------------------------------------- queries
    def _set(self, q: Array) -> None:
        d, m, ix = self.data, self.model, self.ix
        d.qpos[ix.arm_qpos] = q
        if self.attached_T is not None:
            mujoco.mj_kinematics(m, d)
            T = np.eye(4)
            T[:3, :3] = d.site_xmat[ix.tcp_site].reshape(3, 3)
            T[:3, 3] = d.site_xpos[ix.tcp_site]
            To = T @ self.attached_T
            d.qpos[ix.obj_qpos: ix.obj_qpos + 3] = To[:3, 3]
            d.qpos[ix.obj_qpos + 3: ix.obj_qpos + 7] = mat_to_quat(To[:3, :3])
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)

    def _offending(self, g1: int, g2: int) -> bool:
        if self.allowed and ((int(self.geom_body[g1]), g2) in self.allowed
                             or (int(self.geom_body[g2]), g1) in self.allowed):
            return False
        r1, r2 = self.is_robot_geom[g1], self.is_robot_geom[g2]
        og = self.ix.obj_geom
        if r1 and r2:
            return not (self.is_gripper_geom[g1] and self.is_gripper_geom[g2])
        if og in (g1, g2):
            other = g2 if g1 == og else g1
            if self.is_gripper_geom[other] and (self.attached_T is not None or self.allow_gripper_object):
                return False
            if other == self.table_geom:
                return self.attached_T is not None and not self.allow_object_table
            if self.attached_T is not None:
                return True  # held object vs environment / arm
            return bool(self.is_robot_geom[other])
        return bool(r1 or r2)

    def contacts(self, q: Array) -> list[tuple[str, str, float]]:
        """Offending contacts ``(geom1, geom2, dist)`` at ``q`` (debugging aid)."""
        self._set(q)
        out = []
        for c in self.data.contact[: self.data.ncon]:
            if c.dist < self.margin and self._offending(int(c.geom1), int(c.geom2)):
                out.append((self.model.geom(c.geom1).name, self.model.geom(c.geom2).name, float(c.dist)))
        return out

    def min_distance(self, q: Array) -> float:
        """Smallest signed distance over offending pairs (``inf`` if none within 2*margin)."""
        self._set(q)
        best = np.inf
        for c in self.data.contact[: self.data.ncon]:
            if self._offending(int(c.geom1), int(c.geom2)):
                best = min(best, float(c.dist))
        return best

    def in_collision(self, q: Array) -> bool:
        self.stats.checks += 1
        self._set(q)
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < self.margin and self._offending(int(c.geom1), int(c.geom2)):
                return True
        return False

    def edge_steps(self, qa: Array, qb: Array) -> int:
        """Number of samples so that no robot point moves more than ``margin`` per step."""
        sweep = float(self.lever @ np.abs(qb - qa))
        return max(1, int(np.ceil(sweep / self.margin)))

    def segment_free(self, qa: Array, qb: Array, resolution: float | None = None) -> bool:
        """Check the straight joint-space segment ``qa -> qb`` (``qa`` assumed free).

        ``resolution`` (rad, max-norm) optionally forces a finer discretisation;
        by default the swept-displacement bound sets the step count. Samples are
        visited coarse-to-fine so mid-edge collisions are found early.
        """
        n = self.edge_steps(qa, qb)
        if resolution is not None:
            n = max(n, int(np.ceil(np.max(np.abs(qb - qa)) / resolution)))
        for k in _bisection_order(n):
            if self.in_collision(qa + (qb - qa) * (k / n)):
                return False
        return True


def _bisection_order(n: int) -> list[int]:
    """Indices 1..n in coarse-to-fine order (n first, then halves, quarters, ...)."""
    out, seen = [], set()
    step = n
    while step >= 1:
        for k in range(step, n + 1, step):
            if k not in seen:
                seen.add(k)
                out.append(k)
        step //= 2
    return out
