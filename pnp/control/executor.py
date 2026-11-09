"""Trajectory execution in the live MuJoCo simulation.

The UR5e actuators in Menagerie are joint position servos whose bias term is
``-kp q - kd qdot`` (kp=2000/kd=400 on the big joints, 500/100 on the wrist),
and the scene enables gravity compensation. Streaming ``ctrl = q_des(t)`` alone
therefore has a steady-state lag of ``kd * qd / kp`` -- 0.24 rad at 1.2 rad/s,
measured at 0.22 rad in the first version. That is far larger than the 1 cm
planning clearance, i.e. the executed path would *not* be the checked path.

Fix: shift the servo set-point,

    ctrl = q_des + (kd * qd_des + M(q) qdd_des) / kp

so the actuator force becomes ``kp (q_des - q) + kd (qd_des - qd) + M qdd_des``,
a proper PD-plus-feedforward tracking law. A first attempt injected the same
feedforward through ``qfrc_applied``; it went unstable, because the actuator's
own damping term saturates at its +-150 Nm force range (kd*qd = 480 Nm at
1.2 rad/s) while the external torque does not, so the two stop cancelling.
Routing feedforward through ``ctrl`` keeps every torque inside the actuator
and its force limits. Tracking error is logged so
the evaluator can separate *planning* failures from *control* failures.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from pnp.se3 import Array
from pnp.sim.scene import Indices

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0


class Traj(Protocol):
    @property
    def duration(self) -> float: ...

    def sample(self, t: float) -> tuple[Array, Array, Array]: ...


@dataclass
class ExecLog:
    sim_time: float = 0.0
    max_track_err: float = 0.0  # rad, max-norm, during arm motions
    unplanned_contact_steps: int = 0  # robot/held-object vs environment contacts observed in physics
    unplanned_pairs: dict = field(default_factory=dict)  # "geomA|geomB" -> monitored steps
    max_penetration: float = 0.0  # m, deepest unplanned contact
    frames: list[np.ndarray] = field(default_factory=list)


class Executor:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ix: Indices,
                 frame_cb: Callable[[mujoco.MjData], None] | None = None, frame_every: float = 1 / 30) -> None:
        self.m, self.d, self.ix = model, data, ix
        self.log = ExecLog()
        self.frame_cb = frame_cb
        self._frame_every = frame_every
        self._next_frame = 0.0
        self._kd = -model.actuator_biasprm[ix.arm_ctrl, 2].copy()
        self._kp = model.actuator_gainprm[ix.arm_ctrl, 0].copy()
        self._vec = np.zeros(model.nv)
        self._res = np.zeros(model.nv)
        self.feedforward = True
        robot = np.array([b in ix.robot_bodies for b in model.geom_bodyid])
        self._robot_geom = robot
        env = ~robot
        env[ix.obj_geom] = False
        env &= (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self._env_geom = env
        self._table = model.geom("table").id
        self._monitor_every = 5
        self._k = 0

    @property
    def q(self) -> Array:
        return self.d.qpos[self.ix.arm_qpos].copy()

    def _unplanned_contact(self) -> bool:
        """Robot touching the environment, or the object touching non-table environment."""
        hit = False
        for c in self.d.contact[: self.d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if c.dist > 0:
                continue
            for a, b in ((g1, g2), (g2, g1)):
                if (self._robot_geom[a] and self._env_geom[b]) or (
                        a == self.ix.obj_geom and self._env_geom[b] and b != self._table):
                    key = f"{self.m.geom(a).name}|{self.m.geom(b).name}"
                    self.log.unplanned_pairs[key] = self.log.unplanned_pairs.get(key, 0) + 1
                    self.log.max_penetration = max(self.log.max_penetration, -float(c.dist))
                    hit = True
                    break
        return hit

    def _step(self) -> None:
        mujoco.mj_step(self.m, self.d)
        self.log.sim_time += self.m.opt.timestep
        self._k += 1
        if self._k % self._monitor_every == 0 and self._unplanned_contact():
            self.log.unplanned_contact_steps += 1
        if self.frame_cb is not None and self.log.sim_time >= self._next_frame:
            self.frame_cb(self.d)
            self._next_frame += self._frame_every

    def follow(self, traj: Traj, settle: float = 0.25) -> None:
        """Track ``traj`` then hold its final configuration for ``settle`` seconds."""
        dt = self.m.opt.timestep
        n = int(np.ceil(traj.duration / dt))
        for k in range(n + 1):
            q_des, qd_des, qdd_des = traj.sample(k * dt)
            u = q_des
            if self.feedforward:
                self._vec[:] = 0.0
                self._vec[self.ix.arm_dof] = qdd_des
                mujoco.mj_mulM(self.m, self.d, self._res, self._vec)
                u = q_des + (self._kd * qd_des + self._res[self.ix.arm_dof]) / self._kp
            self.d.ctrl[self.ix.arm_ctrl] = u
            self._step()
            err = float(np.max(np.abs(self.d.qpos[self.ix.arm_qpos] - q_des)))
            self.log.max_track_err = max(self.log.max_track_err, err)
        self.d.ctrl[self.ix.arm_ctrl] = traj.sample(traj.duration)[0]
        self.hold(settle)

    def hold(self, seconds: float) -> None:
        for _ in range(int(seconds / self.m.opt.timestep)):
            self._step()

    def gripper(self, value: float, seconds: float = 0.6) -> None:
        """Command the gripper and wait until the fingers stop (or ``seconds``)."""
        self.d.ctrl[self.ix.gripper_ctrl] = value
        dt = self.m.opt.timestep
        drv = self.m.joint("gripper/right_driver_joint")
        adr = self.m.jnt_dofadr[drv.id]
        t = 0.0
        while t < seconds:
            self._step()
            t += dt
            if t > 0.15 and abs(self.d.qvel[adr]) < 1e-3:
                break
        self.hold(0.1)
