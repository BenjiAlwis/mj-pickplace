"""Forward kinematics and geometric Jacobian via the product of exponentials.

The kinematic description is *extracted* from the compiled MuJoCo model at the
zero configuration (each joint's world axis ``w_i`` and anchor ``p_i`` give the
space-frame screw ``S_i = [-w_i x p_i, w_i]``; the TCP site pose gives the
home transform ``M``). After that, FK and Jacobians are pure numpy and never
touch ``MjData``. MuJoCo is used only as ground truth in the tests, so the two
implementations are genuinely independent.

    T(q) = exp([S_1] q_1) ... exp([S_n] q_n) M
    J_s(q)_i = Ad_{exp([S_1]q_1)...exp([S_{i-1}]q_{i-1})} S_i
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from pnp.se3 import Array, adjoint, se3_exp, skew
from pnp.sim.scene import ARM_JOINTS, JOINT_LIMITS, TCP_SITE


@dataclass(frozen=True)
class SerialChain:
    """A revolute serial chain in product-of-exponentials form."""

    screws: Array  # (n, 6) space-frame screws [v, w]
    M: Array  # (4, 4) TCP pose at q = 0
    limits: Array  # (n, 2)

    @property
    def n(self) -> int:
        return self.screws.shape[0]

    @classmethod
    def from_mujoco(cls, model: mujoco.MjModel, joints: tuple[str, ...] = ARM_JOINTS,
                    tcp_site: str = TCP_SITE, limits: Array = JOINT_LIMITS) -> SerialChain:
        """Extract screws and home pose from ``model`` at the zero configuration."""
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        # Keep any free joints at a valid unit quaternion.
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                adr = model.jnt_qposadr[j]
                data.qpos[adr + 3] = 1.0
        mujoco.mj_kinematics(model, data)
        screws = []
        for name in joints:
            jid = model.joint(name).id
            w = data.xaxis[jid].copy()
            p = data.xanchor[jid].copy()
            screws.append(np.concatenate([-np.cross(w, p), w]))
        sid = model.site(tcp_site).id
        M = np.eye(4)
        M[:3, :3] = data.site_xmat[sid].reshape(3, 3)
        M[:3, 3] = data.site_xpos[sid]
        return cls(screws=np.array(screws), M=M, limits=np.asarray(limits, dtype=float))

    def fk(self, q: Array) -> Array:
        """TCP pose (4x4) in the world frame."""
        T = np.eye(4)
        for S, qi in zip(self.screws, q):
            T = T @ se3_exp(S, qi)
        return T @ self.M

    def fk_and_jacobian(self, q: Array) -> tuple[Array, Array]:
        """Return TCP pose and the 6xn *geometric* Jacobian at the TCP point.

        Rows are ``[linear velocity of the TCP point; angular velocity]``, both
        expressed in the world frame -- the same quantity as MuJoCo's
        ``mj_jacSite`` (``jacp`` stacked on ``jacr``).
        """
        T = np.eye(4)
        Js = np.zeros((6, self.n))
        for i, (S, qi) in enumerate(zip(self.screws, q)):
            Js[:, i] = adjoint(T) @ S
            T = T @ se3_exp(S, qi)
        T_tcp = T @ self.M
        p = T_tcp[:3, 3]
        # Space twist (v_s, w) -> velocity of the point p: v_p = v_s + w x p.
        J = np.empty_like(Js)
        J[3:] = Js[3:]
        J[:3] = Js[:3] - skew(p) @ Js[3:]
        return T_tcp, J

    def within_limits(self, q: Array, tol: float = 1e-9) -> bool:
        return bool(np.all(q >= self.limits[:, 0] - tol) and np.all(q <= self.limits[:, 1] + tol))

    def random_q(self, rng: np.random.Generator) -> Array:
        return rng.uniform(self.limits[:, 0], self.limits[:, 1])
