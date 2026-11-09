"""Randomised pick-and-place benchmark.

Usage::

    python -m pnp.eval.run_eval --n 200 --configs nominal noisy --out results

Each episode samples an object pose in the pick zone and a target pose in the
place zone (uniform x/y, uniform yaw in [-pi, pi)), runs the state machine, lets
the scene settle and scores it with :mod:`pnp.eval.metrics`. Episodes are
seeded (episode *i* uses seed ``base_seed + i``) so every configuration sees the
same object/target poses: comparisons between configurations are paired.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from pnp.control.pipeline import PickAndPlace, reset_episode
from pnp.eval.metrics import EpisodeMetrics, judge
from pnp.sim.scene import GEOM, build_model, indices

CONFIGS: dict[str, dict] = {
    # Perfect state estimate, all components enabled.
    "nominal": {},
    # Observed object pose corrupted by sigma = 5 mm / 5 deg Gaussian noise.
    "noisy": {"obs_noise_xy": 0.005, "obs_noise_yaw": np.deg2rad(5.0)},
    # Stress test: sigma = 12 mm / 10 deg -- comparable to the 22 mm per-side jaw clearance.
    "noisy_large": {"obs_noise_xy": 0.012, "obs_noise_yaw": np.deg2rad(10.0)},
    # Moderate noise, but no in-hand re-measurement before placing (ablation).
    "noisy_no_inhand": {"obs_noise_xy": 0.005, "obs_noise_yaw": np.deg2rad(5.0), "inhand_compensation": False},
    # Servo set-point only, no velocity/inertia feedforward (ablation).
    "no_feedforward": {"feedforward": False},
}


def sample_task(rng: np.random.Generator) -> tuple[np.ndarray, float, np.ndarray, float]:
    oxy = np.array([rng.uniform(*GEOM.pick_x), rng.uniform(*GEOM.pick_y)])
    txy = np.array([rng.uniform(*GEOM.place_x), rng.uniform(*GEOM.place_y)])
    return oxy, float(rng.uniform(-np.pi, np.pi)), txy, float(rng.uniform(-np.pi, np.pi))


def run_episode(seed: int, config: str = "nominal", settle: float = 0.5) -> dict:
    model = build_model()
    data = mujoco.MjData(model)
    ix = indices(model)
    rng = np.random.default_rng(seed)
    oxy, oyaw, txy, tyaw = sample_task(rng)
    reset_episode(model, data, ix, oxy, oyaw, txy, tyaw)
    opts = dict(CONFIGS[config])
    ff = opts.pop("feedforward", True)
    # Planner randomness uses a separate stream so the task is identical across configs.
    pp = PickAndPlace(model, data, ix, np.random.default_rng(10_000 + seed), **opts)
    pp.ex.feedforward = ff
    t0 = time.perf_counter()
    completed = pp.run(txy, tyaw)
    pp.ex.hold(settle)
    ok, failure, (e_xy, e_yaw, tilt) = judge(model, data, ix, txy, tyaw, completed, pp.log.failure)
    m = EpisodeMetrics(
        seed=seed,
        success=ok,
        pipeline_completed=completed,
        grasp_success=pp.log.lifted,
        placement_error_xy_mm=e_xy * 1e3,
        yaw_error_deg=float(np.rad2deg(e_yaw)),
        tilt_deg=float(np.rad2deg(tilt)),
        plan_time_s=pp.log.plan_time_s,
        rrt_queries=pp.log.rrt_queries,
        joint_path_length_rad=pp.log.joint_path_length,
        tcp_path_length_m=pp.log.tcp_path_length,
        ik_failures=pp.log.ik_failures,
        cycle_time_s=pp.ex.log.sim_time - settle,
        max_tracking_err_rad=pp.ex.log.max_track_err,
        in_hand_slip_mm=pp.log.in_hand_slip_mm,
        failure=failure,
    )
    d = m.to_dict()
    d["config"] = config
    d["unplanned_contact_steps"] = pp.ex.log.unplanned_contact_steps
    d["unplanned_pairs"] = pp.ex.log.unplanned_pairs
    d["max_penetration_mm"] = pp.ex.log.max_penetration * 1e3
    d["wall_s"] = time.perf_counter() - t0
    return d


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    done = [r for r in rows if r["pipeline_completed"]]

    def pct(key: str, sub: list[dict], q: float) -> float:
        v = [r[key] for r in sub if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.percentile(v, q)) if v else float("nan")

    k = len(succ)
    lo, hi = wilson(k, n)
    return {
        "n": n,
        "success_rate": k / n,
        "success_ci95": [lo, hi],
        "grasp_success_rate": sum(r["grasp_success"] for r in rows) / n,
        "pipeline_completion_rate": len(done) / n,
        "placement_error_mm_median": pct("placement_error_xy_mm", done, 50),
        "placement_error_mm_p95": pct("placement_error_xy_mm", done, 95),
        "placement_error_mm_max": pct("placement_error_xy_mm", done, 100),
        "yaw_error_deg_median": pct("yaw_error_deg", done, 50),
        "yaw_error_deg_p95": pct("yaw_error_deg", done, 95),
        "plan_time_s_median": pct("plan_time_s", rows, 50),
        "plan_time_s_p95": pct("plan_time_s", rows, 95),
        "joint_path_rad_median": pct("joint_path_length_rad", done, 50),
        "tcp_path_m_median": pct("tcp_path_length_m", done, 50),
        "cycle_time_s_median": pct("cycle_time_s", done, 50),
        "ik_failures_total": int(sum(r["ik_failures"] for r in rows)),
        "episodes_with_ik_failure": int(sum(r["ik_failures"] > 0 for r in rows)),
        "max_tracking_err_rad_median": pct("max_tracking_err_rad", rows, 50),
        "episodes_with_unplanned_contact": int(sum(r["unplanned_contact_steps"] > 0 for r in rows)),
        "failures": dict(Counter(r["failure"] for r in rows if r["failure"])),
    }


def to_markdown(summaries: dict[str, dict]) -> str:
    cols = [
        ("Success [95% CI]", lambda s: f"{100*s['success_rate']:.1f}% "
                                       f"[{100*s['success_ci95'][0]:.1f}, {100*s['success_ci95'][1]:.1f}]"),
        ("Grasp", lambda s: f"{100*s['grasp_success_rate']:.1f}%"),
        ("Place err med / p95 (mm)",
         lambda s: f"{s['placement_error_mm_median']:.1f} / {s['placement_error_mm_p95']:.1f}"),
        ("Yaw err med / p95 (deg)", lambda s: f"{s['yaw_error_deg_median']:.1f} / {s['yaw_error_deg_p95']:.1f}"),
        ("Plan time med / p95 (s)", lambda s: f"{s['plan_time_s_median']:.2f} / {s['plan_time_s_p95']:.2f}"),
        ("Joint path (rad)", lambda s: f"{s['joint_path_rad_median']:.2f}"),
        ("TCP path (m)", lambda s: f"{s['tcp_path_m_median']:.2f}"),
        ("Cycle (s)", lambda s: f"{s['cycle_time_s_median']:.1f}"),
        ("IK fails (eps)", lambda s: f"{s['ik_failures_total']} ({s['episodes_with_ik_failure']})"),
        ("Track err (rad)", lambda s: f"{s['max_tracking_err_rad_median']:.3f}"),
        ("Unplanned contact (eps)", lambda s: f"{s['episodes_with_unplanned_contact']}"),
    ]
    lines = ["| Config | N | " + " | ".join(c for c, _ in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['n']} | " + " | ".join(f(s) for _, f in cols) + " |")
    lines.append("")
    lines.append("Failure breakdown:")
    for name, s in summaries.items():
        lines.append(f"- **{name}**: " + (", ".join(f"{k} x{v}" for k, v in sorted(s["failures"].items()))
                                          if s["failures"] else "none"))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", nargs="+", default=["nominal"], choices=list(CONFIGS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for cfg in a.configs:
        t0 = time.perf_counter()
        seeds = list(range(a.seed, a.seed + a.n))
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(run_episode, seeds, [cfg] * a.n, chunksize=4))
        (a.out / f"episodes_{cfg}.json").write_text(json.dumps(rows, indent=1))
        summaries[cfg] = summarise(rows)
        print(f"[{cfg}] {a.n} episodes in {time.perf_counter() - t0:.0f}s: "
              f"success {summaries[cfg]['success_rate']:.3f}", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summaries, indent=1))
    md = to_markdown(summaries)
    (a.out / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
