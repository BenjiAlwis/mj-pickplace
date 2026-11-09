"""Path post-processing: randomised shortcutting and waypoint pruning.

RRT paths are feasible but jagged. Randomised shortcutting (Geraerts &
Overmars 2007) repeatedly picks two points *anywhere on the path* (not just at
vertices), and if the straight joint-space segment between them is
collision-free, replaces the section in between. Sampling on continuous
arc-length rather than vertex indices lets shortcuts cut corners mid-edge,
which converges to much shorter paths for the same number of attempts.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pnp.se3 import Array


def path_length(path: list[Array], weights: Array | None = None) -> float:
    """Sum of (optionally weighted) Euclidean joint-space segment lengths (rad)."""
    w = np.ones_like(path[0]) if weights is None else weights
    return float(sum(np.linalg.norm((b - a) * w) for a, b in zip(path[:-1], path[1:])))


def _point_at(path: list[Array], cum: Array, s: float) -> tuple[int, Array]:
    """Return (segment index, configuration) at arc length ``s``."""
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(path) - 2))
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg < 1e-12 else (s - cum[i]) / seg
    return i, path[i] + t * (path[i + 1] - path[i])


def shortcut(
    path: list[Array],
    segment_free: Callable[[Array, Array], bool],
    iters: int = 150,
    rng: np.random.Generator | None = None,
) -> list[Array]:
    """Randomised continuous shortcutting. Endpoints are preserved exactly."""
    rng = rng or np.random.default_rng()
    path = [p.copy() for p in path]
    for _ in range(iters):
        if len(path) < 3:
            break
        seglen = np.array([np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])])
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        s1, s2 = np.sort(rng.uniform(0.0, cum[-1], 2))
        i, qa = _point_at(path, cum, s1)
        j, qb = _point_at(path, cum, s2)
        if j <= i:  # same segment: already straight
            continue
        if segment_free(qa, qb):
            path = path[: i + 1] + [qa, qb] + path[j + 1:]
    return prune(path, segment_free)


def prune(path: list[Array], segment_free: Callable[[Array, Array], bool]) -> list[Array]:
    """Greedy vertex pruning: from each vertex jump to the farthest visible one."""
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    # Drop near-duplicate consecutive vertices.
    clean = [out[0]]
    for q in out[1:]:
        if np.linalg.norm(q - clean[-1]) > 1e-6:
            clean.append(q)
    if len(clean) == 1:
        clean.append(out[-1])
    return clean
