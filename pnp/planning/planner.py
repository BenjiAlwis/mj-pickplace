"""High-level motion planner: RRT-Connect -> shortcut -> time parameterisation."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from pnp.planning.postprocess import path_length, shortcut
from pnp.planning.rrt_connect import PlannerConfig, RRTConnect
from pnp.planning.timing import Profile, Trajectory, time_parameterise
from pnp.se3 import Array
from pnp.sim.collision import CollisionChecker
from pnp.sim.scene import ACC_LIMITS, JOINT_LIMITS, VEL_LIMITS


@dataclass
class MotionPlan:
    success: bool
    trajectory: Trajectory | None
    raw_path: list[Array] | None
    path: list[Array] | None  # after shortcutting
    plan_time_s: float  # RRT + shortcut + timing, wall clock
    rrt_time_s: float
    raw_length: float
    length: float  # joint-space length after shortcutting (rad, unweighted L2)
    nodes: int
    collision_checks: int


class MotionPlanner:
    def __init__(self, checker: CollisionChecker, config: PlannerConfig | None = None,
                 rng: np.random.Generator | None = None, shortcut_iters: int = 150,
                 profile: Profile = "trapezoid") -> None:
        self.checker = checker
        self.cfg = config or PlannerConfig()
        self.rng = rng or np.random.default_rng()
        self.shortcut_iters = shortcut_iters
        self.profile = profile

    def _seg(self, a: Array, b: Array, res: float | None = None) -> bool:
        return self.checker.segment_free(a, b, res or self.cfg.edge_resolution)

    def plan(self, q_start: Array, q_goal: Array) -> MotionPlan:
        t0 = time.perf_counter()
        c0 = self.checker.stats.checks
        rrt = RRTConnect(
            is_free=lambda q: not self.checker.in_collision(q),
            segment_free=self._seg,
            limits=JOINT_LIMITS,
            config=self.cfg,
            rng=self.rng,
        )
        res = rrt.plan(q_start, q_goal)
        if not res.success or res.path is None:
            return MotionPlan(False, None, None, None, time.perf_counter() - t0, res.time_s, 0.0, 0.0,
                              res.nodes, self.checker.stats.checks - c0)
        path = shortcut(res.path, self._seg, iters=self.shortcut_iters, rng=self.rng)
        traj = time_parameterise(path, VEL_LIMITS, ACC_LIMITS, self.profile)
        return MotionPlan(True, traj, res.path, path, time.perf_counter() - t0, res.time_s,
                          path_length(res.path), path_length(path), res.nodes,
                          self.checker.stats.checks - c0)


def validate_trajectory(traj: Trajectory, checker: CollisionChecker, dt: float = 0.01,
                        vmax: Array = VEL_LIMITS, amax: Array = ACC_LIMITS) -> dict:
    """Densely re-check a trajectory: collisions, joint/velocity/acceleration limits."""
    n_coll, max_v, max_a, lim_ok = 0, np.zeros(6), np.zeros(6), True
    for t in np.arange(0.0, traj.duration + dt, dt):
        q, qd, qdd = traj.sample(t)
        n_coll += int(checker.in_collision(q))
        lim_ok &= bool(np.all(q >= JOINT_LIMITS[:, 0] - 1e-9) and np.all(q <= JOINT_LIMITS[:, 1] + 1e-9))
        max_v = np.maximum(max_v, np.abs(qd))
        max_a = np.maximum(max_a, np.abs(qdd))
    return {
        "collisions": n_coll,
        "within_joint_limits": lim_ok,
        "vel_ratio": float(np.max(max_v / vmax)),
        "acc_ratio": float(np.max(max_a / amax)),
    }
