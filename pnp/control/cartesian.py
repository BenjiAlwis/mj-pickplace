"""Straight-line Cartesian moves for the near-contact phases.

Sampling planners give no guarantee about the *shape* of the TCP path, which is
exactly what matters when descending around an object or lifting it clear. So
pre-grasp -> grasp, lift, place-descent and retreat are Cartesian lines:

1. a trapezoidal profile on the scalar arc parameter ``s`` with Cartesian speed
   and acceleration limits (position and orientation, the slower one wins),
2. pose interpolation ``p(s) = p0 + s (p1 - p0)``, ``R(s) = R0 exp(s log(R0^T R1))``,
3. IK at every control sample, seeded from the previous solution with *no*
   restarts, so the arm stays on one IK branch; a joint jump larger than
   ``max_joint_step`` is treated as a branch flip / singularity and fails the move,
4. every sample is collision checked.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from pnp.ik.solver import DLSSolver
from pnp.se3 import Array, so3_exp, so3_log
from pnp.sim.collision import CollisionChecker


@dataclass
class SampledTrajectory:
    """Joint trajectory stored at a fixed sample period, linearly interpolated."""

    q: Array  # (N, 6)
    dt: float

    @property
    def duration(self) -> float:
        return (len(self.q) - 1) * self.dt

    def sample(self, t: float) -> tuple[Array, Array, Array]:
        x = min(max(t / self.dt, 0.0), len(self.q) - 1.0)
        i = min(int(x), len(self.q) - 2) if len(self.q) > 1 else 0
        if len(self.q) == 1:
            z = np.zeros_like(self.q[0])
            return self.q[0].copy(), z, z
        a = x - i
        qd = (self.q[i + 1] - self.q[i]) / self.dt
        return self.q[i] + a * (self.q[i + 1] - self.q[i]), qd, np.zeros_like(qd)


@dataclass
class CartesianResult:
    success: bool
    traj: SampledTrajectory | None
    reason: str
    time_s: float
    ik_failures: int
    tcp_length: float


def _profile(L: float, v: float, a: float, dt: float) -> Array:
    """Samples of s in [0, 1] for a trapezoid covering normalised distance 1."""
    if L < 1e-9:
        return np.array([0.0, 1.0])
    vs, as_ = v / L, a / L
    if vs * vs / as_ >= 1.0:
        ta = np.sqrt(1.0 / as_)
        T, vp = 2 * ta, as_ * ta
    else:
        ta, vp = vs / as_, vs
        T = 2 * ta + (1.0 - vs * ta) / vs
    t = np.append(np.arange(0.0, T, dt), T)
    s = np.where(t < ta, 0.5 * as_ * t**2,
                 np.where(t <= T - ta, 0.5 * as_ * ta**2 + vp * (t - ta), 1.0 - 0.5 * as_ * (T - t) ** 2))
    return np.clip(s, 0.0, 1.0)


def cartesian_move(
    ik: DLSSolver,
    checker: CollisionChecker,
    q0: Array,
    T_goal: Array,
    v_lin: float = 0.15,
    a_lin: float = 0.5,
    v_ang: float = 0.8,
    a_ang: float = 2.0,
    dt: float = 0.01,
    max_joint_step: float = 0.05,
) -> CartesianResult:
    """Plan a straight-line TCP move from ``fk(q0)`` to ``T_goal``."""
    t0 = time.perf_counter()
    T0 = ik.chain.fk(q0)
    dp = T_goal[:3, 3] - T0[:3, 3]
    dw = so3_log(T0[:3, :3].T @ T_goal[:3, :3])
    L_lin, L_ang = float(np.linalg.norm(dp)), float(np.linalg.norm(dw))
    s_lin, s_ang = _profile(L_lin, v_lin, a_lin, dt), _profile(L_ang, v_ang, a_ang, dt)
    s_all = s_lin if len(s_lin) >= len(s_ang) else s_ang  # slower constraint governs
    qs, q, fails = [q0.copy()], q0.copy(), 0
    for s in s_all[1:]:
        T = np.eye(4)
        T[:3, :3] = T0[:3, :3] @ so3_exp(s * dw)
        T[:3, 3] = T0[:3, 3] + s * dp
        r = ik.solve(T, q, restarts=0)
        if not r.success:
            fails += 1
            return CartesianResult(False, None, "ik", time.perf_counter() - t0, fails, L_lin)
        if np.max(np.abs(r.q - q)) > max_joint_step:
            return CartesianResult(False, None, "branch_jump", time.perf_counter() - t0, fails, L_lin)
        if checker.in_collision(r.q):
            return CartesianResult(False, None, "collision", time.perf_counter() - t0, fails, L_lin)
        q = r.q
        qs.append(q.copy())
    return CartesianResult(True, SampledTrajectory(np.array(qs), dt), "ok", time.perf_counter() - t0, fails, L_lin)
