"""Damped-least-squares inverse kinematics on the geometric Jacobian.

Update rule (one iteration)::

    e   = [p_d - p ; log(R_d R^T)]            # 6-vector, world frame
    e   = clamp(e)                             # bounded task-space step
    dq  = J^T (J J^T + lambda^2 I)^-1 e        # DLS / Levenberg-Marquardt
    q  <- clip(q + alpha * dq, limits)

Design choices
--------------
* **Orientation error via the SO(3) log map**, not Euler-angle differences, so
  the error is a true geodesic and has no gimbal singularities.
* **Two-term adaptive damping**: ``lambda^2 = lambda_s^2 + beta |e|^2``.
  ``lambda_s`` ramps up smoothly only when the smallest singular value of J
  drops below a threshold (Maciejewski & Klein 1988), bounding joint steps near
  wrist/elbow/shoulder singularities. The ``beta |e|^2`` term (Sugihara 2011)
  damps large far-from-solution steps but vanishes as the error does, so the
  final iterations are undamped Gauss-Newton with quadratic convergence. A
  first version used manipulability ``sqrt(det(J J^T))`` as the trigger; it
  was active in >50% of random configurations (its units mix m and rad), which
  made convergence linear and slow -- the sigma_min trigger fixed that.
* **Task-space step clamping** keeps the linearisation valid for far targets.
* **Joint limits by projection** plus **random restarts**: for a 6-DoF arm
  there is no null space to push away from limits. Joints whose range covers a
  full turn are *wrapped* (exact, because revolute FK is 2*pi periodic);
  narrower ranges are clipped. Clipping all joints made 75% of single descents
  stall against a limit -- wrapping removed that failure mode. Stalled descents
  are abandoned early and restarted. Restarts handle the remaining local
  minima and select among the (up to 8) UR inverse-kinematic branches.
* An optional ``validator`` (e.g. a collision check) is applied to each
  converged solution, so "IK succeeded" means "IK found a *usable* solution".
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pnp.ik.kinematics import SerialChain
from pnp.se3 import Array, so3_log


@dataclass
class IKConfig:
    pos_tol: float = 1e-4  # m   (0.1 mm)
    rot_tol: float = 1e-3  # rad (0.057 deg)
    max_iters: int = 150
    lambda_max: float = 0.05  # maximum singularity damping
    sigma_threshold: float = 0.03  # min singular value below which it kicks in
    err_damping: float = 0.01  # Sugihara-style damping ~ |e|^2, vanishes at the solution
    max_pos_step: float = 0.08  # m per iteration
    max_rot_step: float = 0.4  # rad per iteration
    restarts: int = 20
    wrap_full_turn: bool = True  # False = clip every joint (ablation)


@dataclass
class IKResult:
    q: Array
    success: bool
    pos_err: float
    rot_err: float
    iters: int  # total iterations across all attempts
    attempts: int
    time_s: float
    rejected_by_validator: int = 0
    extra: dict = field(default_factory=dict)


def pose_error(T: Array, T_d: Array) -> Array:
    """6-vector ``[p_d - p; log(R_d R^T)]`` in the world frame."""
    e = np.empty(6)
    e[:3] = T_d[:3, 3] - T[:3, 3]
    e[3:] = so3_log(T_d[:3, :3] @ T[:3, :3].T)
    return e


class DLSSolver:
    """Damped-least-squares IK for a :class:`SerialChain`."""

    def __init__(self, chain: SerialChain, config: IKConfig | None = None) -> None:
        self.chain = chain
        self.cfg = config or IKConfig()

    def _damping_sq(self, J: Array, err_sq: float) -> float:
        """Squared damping: singularity term + error-dependent (Sugihara) term."""
        c = self.cfg
        s_min = np.linalg.svd(J, compute_uv=False)[-1]
        sing = 0.0
        if s_min < c.sigma_threshold:
            sing = c.lambda_max**2 * (1.0 - (s_min / c.sigma_threshold) ** 2)
        return float(sing + c.err_damping * err_sq + 1e-9)

    def _descend(self, T_d: Array, q0: Array) -> tuple[Array, float, float, int]:
        c = self.cfg
        q = self._project(q0.copy())
        last_err = np.inf
        it = 0
        for it in range(1, c.max_iters + 1):
            T, J = self.chain.fk_and_jacobian(q)
            e = pose_error(T, T_d)
            pe, re = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if pe < c.pos_tol and re < c.rot_tol:
                return q, pe, re, it
            if pe > c.max_pos_step:
                e[:3] *= c.max_pos_step / pe
            if re > c.max_rot_step:
                e[3:] *= c.max_rot_step / re
            lam_sq = self._damping_sq(J, float(e @ e))
            dq = J.T @ np.linalg.solve(J @ J.T + lam_sq * np.eye(6), e)
            q = self._project(q + dq)
            # Stall detection: abandon a descent that has stopped improving
            # (typically stuck against a limit) so a restart can be tried.
            err = pe + re
            if it % 10 == 0:
                if err > 0.99 * last_err:
                    break
                last_err = err
        T = self.chain.fk(q)
        e = pose_error(T, T_d)
        return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])), it

    def _project(self, q: Array) -> Array:
        """Project onto the joint limits.

        Revolute FK is 2*pi-periodic, so for joints whose allowed range spans a
        full turn, wrapping is an *exact* projection (same pose, no artificial
        boundary). Joints with a narrower range are clipped.
        """
        lo, hi = self.chain.limits[:, 0], self.chain.limits[:, 1]
        full = ((hi - lo) >= 2 * np.pi - 1e-9) & self.cfg.wrap_full_turn
        q = q.copy()
        q[full] = lo[full] + np.mod(q[full] - lo[full], 2 * np.pi)
        return np.clip(q, lo, hi)

    def solve(
        self,
        T_d: Array,
        q_seed: Array,
        rng: np.random.Generator | None = None,
        validator: Callable[[Array], bool] | None = None,
        restarts: int | None = None,
    ) -> IKResult:
        """Solve for ``T_d`` starting from ``q_seed``, then from random seeds.

        The seeded attempt runs first so that, when it succeeds, the solution
        lies on the same IK branch as the seed (important for continuous
        Cartesian paths). Returns the first converged, validator-approved
        solution.
        """
        rng = rng if rng is not None else np.random.default_rng()
        n_restarts = self.cfg.restarts if restarts is None else restarts
        t0 = time.perf_counter()
        total_iters, rejected = 0, 0
        best = (q_seed, np.inf, np.inf)
        for attempt in range(n_restarts + 1):
            seed = q_seed if attempt == 0 else self.chain.random_q(rng)
            q, pe, re, it = self._descend(T_d, seed)
            total_iters += it
            converged = pe < self.cfg.pos_tol and re < self.cfg.rot_tol
            if converged and validator is not None and not validator(q):
                rejected += 1
                converged = False
            if converged:
                return IKResult(q, True, pe, re, total_iters, attempt + 1, time.perf_counter() - t0, rejected)
            if pe + re < best[1] + best[2]:
                best = (q, pe, re)
        return IKResult(best[0], False, best[1], best[2], total_iters, n_restarts + 1,
                        time.perf_counter() - t0, rejected)
