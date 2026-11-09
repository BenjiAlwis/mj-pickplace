"""IK round trips: pose = FK(q_true) -> IK from a random seed -> FK(q_sol) == pose."""
import numpy as np

from pnp.ik.solver import IKConfig, pose_error


def test_ik_roundtrip_random_poses(chain, ik, rng):
    n, ok = 150, 0
    for _ in range(n):
        T = chain.fk(chain.random_q(rng))
        r = ik.solve(T, chain.random_q(rng), rng=rng)
        if r.success:
            ok += 1
            e = pose_error(chain.fk(r.q), T)
            assert np.linalg.norm(e[:3]) <= ik.cfg.pos_tol
            assert np.linalg.norm(e[3:]) <= ik.cfg.rot_tol
            assert chain.within_limits(r.q)
    assert ok / n >= 0.97, f"IK success rate {ok / n:.3f}"


def test_seeded_ik_stays_on_branch(chain, ik, rng):
    """Small pose perturbations solved from the nearby seed give nearby joints."""
    for _ in range(50):
        q = chain.random_q(rng) * 0.7
        T = chain.fk(q + rng.normal(0, 0.01, 6))
        r = ik.solve(T, q, rng=rng, restarts=0)
        if r.success:
            assert np.max(np.abs(r.q - q)) < 0.2


def test_ik_at_wrist_singularity(chain, ik, rng):
    """wrist_2 = 0 aligns wrist_1 and wrist_3 axes (UR wrist singularity)."""
    q = np.array([0.3, -1.2, 1.4, -1.0, 0.0, 0.5])
    _, J = chain.fk_and_jacobian(q)
    assert np.linalg.svd(J, compute_uv=False)[-1] < 1e-9  # genuinely singular
    r = ik.solve(chain.fk(q), q + 0.2, rng=rng)
    assert r.success
    assert np.all(np.isfinite(r.q))


def test_validator_is_respected(chain, ik, rng):
    T = chain.fk(chain.random_q(rng))
    r = ik.solve(T, chain.random_q(rng), rng=rng, validator=lambda q: q[0] > 0, restarts=40)
    if r.success:
        assert r.q[0] > 0


def test_unreachable_pose_fails_cleanly(ik, rng):
    T = np.eye(4)
    T[:3, 3] = [3.0, 0.0, 0.0]  # far outside the ~0.85 m reach
    r = ik.solve(T, np.zeros(6), rng=rng, restarts=3)
    assert not r.success
    assert r.pos_err > 1.0


def test_damping_is_finite_near_singularity(chain, ik):
    q = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    _, J = chain.fk_and_jacobian(q)
    assert ik._damping_sq(J, 0.0) > IKConfig().lambda_max ** 2 * 0.5
