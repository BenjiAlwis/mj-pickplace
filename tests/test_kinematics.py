"""Our product-of-exponentials FK/Jacobian vs MuJoCo as ground truth."""
import mujoco
import numpy as np

from pnp.se3 import rotation_angle, so3_log


def test_fk_matches_mujoco(model, ix, chain, rng):
    d = mujoco.MjData(model)
    for _ in range(200):
        q = chain.random_q(rng)
        d.qpos[ix.arm_qpos] = q
        mujoco.mj_kinematics(model, d)
        T = chain.fk(q)
        assert np.linalg.norm(T[:3, 3] - d.site_xpos[ix.tcp_site]) < 1e-10
        assert rotation_angle(T[:3, :3], d.site_xmat[ix.tcp_site].reshape(3, 3)) < 1e-10


def test_jacobian_matches_mj_jacsite(model, ix, chain, rng):
    d = mujoco.MjData(model)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(100):
        q = chain.random_q(rng)
        d.qpos[ix.arm_qpos] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        mujoco.mj_jacSite(model, d, jp, jr, ix.tcp_site)
        _, J = chain.fk_and_jacobian(q)
        assert np.allclose(np.vstack([jp, jr])[:, ix.arm_dof], J, atol=1e-10)


def test_jacobian_matches_finite_differences(chain, rng):
    q = chain.random_q(rng)
    T, J = chain.fk_and_jacobian(q)
    h = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = h
        T2 = chain.fk(q + dq)
        assert np.allclose((T2[:3, 3] - T[:3, 3]) / h, J[:3, i], atol=1e-5)
        assert np.allclose(so3_log(T2[:3, :3] @ T[:3, :3].T) / h, J[3:, i], atol=1e-5)


def test_fk_is_2pi_periodic(chain, rng):
    q = chain.random_q(rng)
    for i in range(6):
        q2 = q.copy()
        q2[i] += 2 * np.pi
        assert np.allclose(chain.fk(q), chain.fk(q2), atol=1e-10)
