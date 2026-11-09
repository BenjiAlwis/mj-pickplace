"""Planner benchmark: pick-zone pre-grasp -> place-zone pre-place queries across the divider.

Every timed trajectory is re-validated against a *zero-margin* checker at 5 ms
resolution, plus velocity/acceleration/joint limits.

    python scripts/bench_planner.py --n 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from pnp.planning.planner import MotionPlanner, validate_trajectory
from pnp.sim.collision import CollisionChecker
from pnp.sim.scene import build_model
from scripts.compare_ompl import make_queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    m = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cc, cc0 = CollisionChecker(m, 0.01), CollisionChecker(m, 0.0)
    cc.sync(d)
    cc0.sync(d)
    out = {}
    for profile in ("trapezoid", "quintic"):
        planner = MotionPlanner(cc, rng=np.random.default_rng(a.seed), profile=profile)
        rows = []
        for qa, qb in make_queries(a.n, a.seed):
            p = planner.plan(qa, qb)
            v = validate_trajectory(p.trajectory, cc0, dt=0.005) if p.success else None
            rows.append((p, v))
        ok = [(p, v) for p, v in rows if p.success]
        f = lambda xs, q: float(np.percentile(xs, q))  # noqa: E731
        out[profile] = {
            "n": a.n, "solved": len(ok),
            "plan_time_ms_median": 1e3 * f([p.plan_time_s for p, _ in ok], 50),
            "plan_time_ms_p95": 1e3 * f([p.plan_time_s for p, _ in ok], 95),
            "rrt_time_ms_median": 1e3 * f([p.rrt_time_s for p, _ in ok], 50),
            "raw_length_rad_median": f([p.raw_length for p, _ in ok], 50),
            "length_rad_median": f([p.length for p, _ in ok], 50),
            "length_reduction_median": f([1 - p.length / p.raw_length for p, _ in ok], 50),
            "vertices_median": f([len(p.path) for p, _ in ok], 50),
            "duration_s_median": f([p.trajectory.duration for p, _ in ok], 50),
            "trajectories_with_collision": sum(v["collisions"] > 0 for _, v in ok),
            "all_within_joint_limits": all(v["within_joint_limits"] for _, v in ok),
            "max_vel_ratio": max(v["vel_ratio"] for _, v in ok),
            "max_acc_ratio": max(v["acc_ratio"] for _, v in ok),
        }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "planner_benchmark.json").write_text(json.dumps(out, indent=1))
    lines = ["| Profile | Solved | Plan time med / p95 (ms) | Path length raw -> shortcut (rad) | Vertices | "
             "Duration (s) | Collisions (0-margin re-check) | Max v/v_lim, a/a_lim |",
             "|---|---|---|---|---|---|---|---|"]
    for k, s in out.items():
        lines.append(f"| {k} | {s['solved']}/{s['n']} | "
                     f"{s['plan_time_ms_median']:.0f} / {s['plan_time_ms_p95']:.0f} | "
                     f"{s['raw_length_rad_median']:.2f} -> {s['length_rad_median']:.2f} | {s['vertices_median']:.0f} | "
                     f"{s['duration_s_median']:.2f} | {s['trajectories_with_collision']} | "
                     f"{s['max_vel_ratio']:.3f}, {s['max_acc_ratio']:.3f} |")
    md = "\n".join(lines)
    (a.out / "planner_benchmark.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
