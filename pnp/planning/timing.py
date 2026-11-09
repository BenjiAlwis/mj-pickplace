"""Time parameterisation of piecewise-linear joint-space paths.

Design choice: **path-preserving, synchronised scaling**. Every segment
``q(s) = q_a + s (q_b - q_a)`` is timed with a scalar profile ``s(t)`` shared by
all joints, so the executed trajectory traces *exactly* the straight segments
that were collision-checked -- time parameterisation can never introduce a
collision. The price is a stop at each interior vertex; after shortcutting
there are typically only 1-3 of them.

Per segment, joint i needs ``|dq_i| s' <= v_i`` and ``|dq_i| s'' <= a_i``, so
the scalar limits are ``v_s = min_i v_i/|dq_i|`` and ``a_s = min_i a_i/|dq_i|``.

* ``trapezoid``: time-optimal bang-coast-bang profile for those scalar limits
  (degenerates to a triangle for short segments). C1 position, bounded jerk-free
  piecewise-constant acceleration.
* ``quintic``: ``s = 10 tau^3 - 15 tau^4 + 6 tau^5`` (zero vel *and* accel at both
  ends, C2). Peak ``s' = 15/8 / T``, peak ``s'' = 10/sqrt(3) / T^2``, so the
  minimum feasible duration is ``T = max(15/8 / v_s, sqrt(10/sqrt(3) / a_s))``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from pnp.se3 import Array

Profile = Literal["trapezoid", "quintic"]

_QUINTIC_VPEAK = 15.0 / 8.0
_QUINTIC_APEAK = 10.0 / np.sqrt(3.0)


@dataclass
class _Segment:
    qa: Array
    dq: Array
    T: float
    vs: float
    as_: float
    profile: Profile
    t_acc: float = 0.0  # trapezoid only
    v_peak: float = 0.0

    def s(self, t: float) -> tuple[float, float, float]:
        """Scalar path parameter and its first two time derivatives."""
        t = min(max(t, 0.0), self.T)
        if self.T <= 0.0:
            return 1.0, 0.0, 0.0
        if self.profile == "quintic":
            tau = t / self.T
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.T
            sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / self.T**2
            return s, sd, sdd
        ta, vp, a = self.t_acc, self.v_peak, self.as_
        if t < ta:
            return 0.5 * a * t * t, a * t, a
        if t <= self.T - ta:
            return 0.5 * a * ta * ta + vp * (t - ta), vp, 0.0
        r = self.T - t
        return 1.0 - 0.5 * a * r * r, a * r, -a


@dataclass
class Trajectory:
    """Piecewise time-parameterised joint trajectory."""

    segments: list[_Segment] = field(default_factory=list)
    starts: Array = field(default_factory=lambda: np.zeros(1))

    @property
    def duration(self) -> float:
        return float(self.starts[-1])

    def sample(self, t: float) -> tuple[Array, Array, Array]:
        """Return ``(q, qd, qdd)`` at time ``t`` (clamped to the end)."""
        if not self.segments:
            raise ValueError("empty trajectory")
        t = min(max(t, 0.0), self.duration)
        k = int(np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, len(self.segments) - 1))
        seg = self.segments[k]
        s, sd, sdd = seg.s(t - self.starts[k])
        return seg.qa + s * seg.dq, sd * seg.dq, sdd * seg.dq

    def waypoints(self) -> list[Array]:
        return [self.segments[0].qa] + [sg.qa + sg.dq for sg in self.segments]


def time_parameterise(path: list[Array], vmax: Array, amax: Array, profile: Profile = "trapezoid") -> Trajectory:
    """Time a piecewise-linear path with synchronised per-segment profiles."""
    segs: list[_Segment] = []
    for qa, qb in zip(path[:-1], path[1:]):
        dq = qb - qa
        mag = np.abs(dq)
        if mag.max() < 1e-9:
            continue
        nz = mag > 1e-12
        vs = float(np.min(vmax[nz] / mag[nz]))
        as_ = float(np.min(amax[nz] / mag[nz]))
        if profile == "quintic":
            T = max(_QUINTIC_VPEAK / vs, np.sqrt(_QUINTIC_APEAK / as_))
            segs.append(_Segment(qa.copy(), dq, T, vs, as_, profile))
            continue
        # Trapezoid on s in [0, 1]: triangle if the cruise velocity is never reached.
        if vs * vs / as_ >= 1.0:
            t_acc = np.sqrt(1.0 / as_)
            v_peak = as_ * t_acc
            T = 2.0 * t_acc
        else:
            t_acc = vs / as_
            v_peak = vs
            T = 2.0 * t_acc + (1.0 - vs * t_acc) / vs
        segs.append(_Segment(qa.copy(), dq, float(T), vs, as_, profile, float(t_acc), float(v_peak)))
    if not segs:  # zero-length path: hold position for one tick
        q = path[0].copy()
        segs.append(_Segment(q, np.zeros_like(q), 0.0, 1.0, 1.0, profile))
    starts = np.concatenate([[0.0], np.cumsum([s.T for s in segs])])
    return Trajectory(segs, starts)
