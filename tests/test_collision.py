import numpy as np

from pnp.control.grasp import top_down_pose
from pnp.sim.collision import CollisionChecker
from pnp.sim.scene import HOME_Q


def test_home_is_free(checker):
    assert not checker.in_collision(HOME_Q)


def test_arm_through_table_is_detected(checker, ik, rng):
    T = top_down_pose([0.5, 0.25, -0.05], 0.0)  # TCP 5 cm *inside* the table
    r = ik.solve(T, HOME_Q, rng=rng)
    assert r.success
    assert checker.in_collision(r.q)


def test_margin_does_not_leak_into_live_model(model):
    CollisionChecker(model, margin=0.05)
    assert np.all(model.geom_margin == 0.0)


def test_margin_is_exact(model, data, ix):
    """Lowering the TCP over open table: collision iff gripper-table distance < margin."""
    c0 = CollisionChecker(model, margin=0.0)
    c1 = CollisionChecker(model, margin=0.01)
    c0.sync(data)
    c1.sync(data)
    from pnp.ik.kinematics import SerialChain
    from pnp.ik.solver import DLSSolver
    ik = DLSSolver(SerialChain.from_mujoco(model))
    q = HOME_Q
    first0 = first1 = None
    for z in np.arange(0.10, -0.02, -0.001):
        q = ik.solve(top_down_pose([0.5, 0.25, z], 0.0), q, restarts=0).q
        if first1 is None and c1.in_collision(q):
            first1 = z
        if first0 is None and c0.in_collision(q):
            first0 = z
    assert first0 is not None and first1 is not None
    assert abs((first1 - first0) - 0.01) < 0.0025


def test_attached_object_is_checked(model, data, ix, checker):
    """Holding the box 10 cm below the TCP pushes it into the table at a pose that is free when empty."""
    from pnp.ik.kinematics import SerialChain
    from pnp.ik.solver import DLSSolver
    ik = DLSSolver(SerialChain.from_mujoco(model))
    q = ik.solve(top_down_pose([0.5, 0.25, 0.08], 0.0), HOME_Q).q
    assert not checker.in_collision(q)
    T = np.eye(4)
    T[2, 3] = 0.10  # along TCP z (pointing down)
    checker.attach(T)
    assert checker.in_collision(q)
    checker.attach(None)
    assert not checker.in_collision(q)


def test_lever_bound_dominates_tcp_displacement(chain, checker, rng):
    """Swept-volume bound: |dp_tcp| <= sum_i r_i |dq_i| for random joint motions."""
    for _ in range(500):
        qa = chain.random_q(rng)
        qb = qa + rng.normal(0, 0.3, 6)
        dp = np.linalg.norm(chain.fk(qb)[:3, 3] - chain.fk(qa)[:3, 3])
        assert dp <= checker.lever @ np.abs(qb - qa) + 1e-12


def test_checker_does_not_touch_live_data(model, data, ix, checker, rng):
    before = data.qpos.copy()
    for _ in range(20):
        checker.in_collision(rng.uniform(-np.pi, np.pi, 6))
    assert np.array_equal(before, data.qpos)
