"""RRT-Connect (Kuffner & LaValle, 2000) in joint space, written from scratch.

Two trees grow from start and goal. Each iteration samples a configuration,
*extends* one tree a single step toward it, then greedily *connects* the other
tree toward the new node (repeated steps until blocked or joined). Trees swap
roles every iteration. The greedy connect is what makes RRT-Connect dramatically
faster than single-tree RRT in the relatively open C-spaces of pick cells.

Nearest-neighbour search is brute force over a preallocated numpy array. For
the tree sizes seen here (10^2-10^3 nodes) a vectorised linear scan beats a
KD-tree, which would need rebuilding as the tree grows.

Distances use a weighted Euclidean metric: proximal joints move more of the
arm, so they get larger weights. That makes "nearest" mean "least swept volume"
rather than "smallest raw angle change".
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pnp.se3 import Array

DEFAULT_WEIGHTS = np.array([1.5, 1.5, 1.2, 0.8, 0.6, 0.4])


@dataclass
class PlannerConfig:
    step: float = 0.25  # max extension per step in the weighted metric (rad)
    edge_resolution: float | None = None  # None: use the swept-displacement bound
    max_iters: int = 6000
    timeout_s: float = 5.0
    weights: Array = field(default_factory=DEFAULT_WEIGHTS.copy)


@dataclass
class PlanResult:
    path: list[Array] | None
    success: bool
    iters: int
    nodes: int
    time_s: float
    collision_checks: int


class _Tree:
    def __init__(self, root: Array, cap: int) -> None:
        self.q = np.empty((cap, root.size))
        self.parent = np.empty(cap, dtype=np.int64)
        self.q[0], self.parent[0], self.n = root, -1, 1

    def add(self, q: Array, parent: int) -> int:
        if self.n == len(self.q):
            self.q = np.vstack([self.q, np.empty_like(self.q)])
            self.parent = np.concatenate([self.parent, np.empty_like(self.parent)])
        self.q[self.n], self.parent[self.n] = q, parent
        self.n += 1
        return self.n - 1

    def nearest(self, q: Array, w: Array) -> int:
        d = (((self.q[: self.n] - q) * w) ** 2).sum(axis=1)
        return int(np.argmin(d))

    def branch(self, i: int) -> list[Array]:
        out = []
        while i != -1:
            out.append(self.q[i].copy())
            i = int(self.parent[i])
        return out  # node -> root


class RRTConnect:
    """Bidirectional RRT with greedy connect."""

    def __init__(
        self,
        is_free: Callable[[Array], bool],
        segment_free: Callable[[Array, Array, float], bool],
        limits: Array,
        config: PlannerConfig | None = None,
        rng: np.random.Generator | None = None,
        count_checks: Callable[[], int] | None = None,
    ) -> None:
        self.is_free = is_free
        self.segment_free = segment_free
        self.limits = np.asarray(limits)
        self.cfg = config or PlannerConfig()
        self.rng = rng or np.random.default_rng()
        self._count = count_checks or (lambda: 0)

    def _steer(self, qa: Array, qb: Array) -> Array:
        d = qb - qa
        dist = float(np.linalg.norm(d * self.cfg.weights))
        if dist <= self.cfg.step:
            return qb.copy()
        return qa + d * (self.cfg.step / dist)

    def _extend(self, tree: _Tree, q_target: Array) -> tuple[str, int]:
        i = tree.nearest(q_target, self.cfg.weights)
        q_new = self._steer(tree.q[i], q_target)
        if not self.segment_free(tree.q[i], q_new, self.cfg.edge_resolution):
            return "trapped", -1
        j = tree.add(q_new, i)
        return ("reached" if np.allclose(q_new, q_target) else "advanced"), j

    def _connect(self, tree: _Tree, q_target: Array) -> tuple[str, int]:
        status, j = "advanced", -1
        while status == "advanced":
            status, k = self._extend(tree, q_target)
            if k >= 0:
                j = k
        return status, j

    def plan(self, q_start: Array, q_goal: Array) -> PlanResult:
        t0, c0 = time.perf_counter(), self._count()
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        if not (self.is_free(q_start) and self.is_free(q_goal)):
            return PlanResult(None, False, 0, 0, time.perf_counter() - t0, self._count() - c0)
        # Trivial case: the straight segment is free.
        if self.segment_free(q_start, q_goal, self.cfg.edge_resolution):
            return PlanResult([q_start.copy(), q_goal.copy()], True, 0, 2,
                              time.perf_counter() - t0, self._count() - c0)
        ta, tb = _Tree(q_start, 1024), _Tree(q_goal, 1024)
        a_is_start = True
        for it in range(1, self.cfg.max_iters + 1):
            if time.perf_counter() - t0 > self.cfg.timeout_s:
                break
            q_rand = self.rng.uniform(lo, hi)
            status, j = self._extend(ta, q_rand)
            if status != "trapped":
                status_b, k = self._connect(tb, ta.q[j])
                if status_b == "reached":
                    pa, pb = ta.branch(j), tb.branch(k)
                    path = pa[::-1] + pb[1:]  # root_a ... joint ... root_b
                    if not a_is_start:
                        path = path[::-1]
                    return PlanResult(path, True, it, ta.n + tb.n, time.perf_counter() - t0,
                                      self._count() - c0)
            ta, tb = tb, ta
            a_is_start = not a_is_start
        return PlanResult(None, False, it, ta.n + tb.n, time.perf_counter() - t0, self._count() - c0)
