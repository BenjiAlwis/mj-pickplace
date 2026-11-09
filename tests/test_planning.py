import numpy as np
import pytest

from pnp.control.grasp import top_down_pose
from pnp.planning.planner import MotionPlanner, validate_trajectory
from pnp.planning.postprocess import path_length
from pnp.planning.timing import time_parameterise
from pnp.sim.collision import CollisionChecker
from pnp.sim.scene import ACC_LIMITS, GEOM, HOME_Q, VEL_LIMITS


@pytest.fixture()
def query(ik, checker, rng):
    """Pre-grasp over the pick zone -> pre-place over the place zone (across the divider)."""
    qs = []
    for xy in ([0.5, -0.25], [0.55, 0.28]):
        r = ik.solve(top_down_pose([*xy, 0.06], 0.3), HOME_Q, rng=rng,
                     validator=lambda q: not checker.in_collision(q))
        assert r.success
        qs.append(r.q)
    assert not checker.segment_free(qs[0], qs[1]), "query should require a detour"
    return qs


def test_rrt_connect_finds_collision_free_path(model, data, checker, query, rng):
    plan = MotionPlanner(checker, rng=rng).plan(*query)
    assert plan.success
    assert np.allclose(plan.path[0], query[0]) and np.allclose(plan.path[-1], query[1])
    # Independent re-check: zero margin, fine time sampling of the timed trajectory.
    c0 = CollisionChecker(model, margin=0.0)
    c0.sync(data)
    v = validate_trajectory(plan.trajectory, c0, dt=0.005)
    assert v["collisions"] == 0
    assert v["within_joint_limits"]
    assert v["vel_ratio"] <= 1.0 + 1e-9
    assert v["acc_ratio"] <= 1.0 + 1e-9


def test_shortcutting_shortens(checker, query, rng):
    plan = MotionPlanner(checker, rng=rng).plan(*query)
    assert plan.length <= plan.raw_length + 1e-9
    assert len(plan.path) <= len(plan.raw_path)


def test_infeasible_start_fails(checker, ik, rng, query):
    bad = ik.solve(top_down_pose([0.5, 0.25, -0.05], 0.0), HOME_Q, rng=rng).q
    assert not MotionPlanner(checker, rng=rng).plan(bad, query[1]).success


@pytest.mark.parametrize("profile", ["trapezoid", "quintic"])
def test_time_parameterisation_respects_limits(profile, rng):
    path = [rng.uniform(-2, 2, 6) for _ in range(5)]
    tr = time_parameterise(path, VEL_LIMITS, ACC_LIMITS, profile)
    q0, qd0, _ = tr.sample(0.0)
    q1, qd1, _ = tr.sample(tr.duration)
    assert np.allclose(q0, path[0]) and np.allclose(q1, path[-1])
    assert np.allclose(qd0, 0) and np.allclose(qd1, 0)
    ts = np.linspace(0, tr.duration, 4000)
    qd = np.array([tr.sample(t)[1] for t in ts])
    qdd = np.array([tr.sample(t)[2] for t in ts])
    assert np.all(np.abs(qd) <= VEL_LIMITS + 1e-9)
    assert np.all(np.abs(qdd) <= ACC_LIMITS + 1e-9)
    # Velocity is the derivative of position (numerical check).
    q = np.array([tr.sample(t)[0] for t in ts])
    assert np.allclose(np.gradient(q, ts, axis=0)[5:-5], qd[5:-5], atol=5e-3)
    # Path-preserving: each segment's samples are q_a + u * dq with u in [0, 1].
    for seg in tr.segments:
        for s in np.linspace(0, seg.T, 7):
            qt = seg.qa + seg.s(s)[0] * seg.dq
            u = seg.s(s)[0]
            assert -1e-12 <= u <= 1 + 1e-12 and np.allclose(qt, seg.qa + u * seg.dq)


def test_trapezoid_matches_closed_form():
    """Single joint, 2 rad, v=1, a=2 -> t_acc=0.5, cruise 1.5 s, total 2.5 s."""
    v, a = np.array([1.0] * 6), np.array([2.0] * 6)
    q0, q1 = np.zeros(6), np.r_[2.0, np.zeros(5)]
    tr = time_parameterise([q0, q1], v, a)
    assert np.isclose(tr.duration, 2.5)
    assert np.isclose(tr.sample(1.25)[0][0], 1.0)  # symmetric midpoint
    assert np.isclose(tr.sample(1.0)[1][0], 1.0)  # cruising


def test_path_length():
    p = [np.zeros(6), np.r_[3.0, 4.0, 0, 0, 0, 0], np.r_[3.0, 4.0, 1.0, 0, 0, 0]]
    assert np.isclose(path_length(p), 6.0)


def test_scene_zones_are_on_the_table():
    tx, ty = GEOM.table_center
    hx, hy = GEOM.table_half
    for (x0, x1), (y0, y1) in ((GEOM.pick_x, GEOM.pick_y), (GEOM.place_x, GEOM.place_y)):
        assert tx - hx < x0 < x1 < tx + hx and ty - hy < y0 < y1 < ty + hy
