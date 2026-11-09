"""Head-to-head: this repo's RRT-Connect + shortcutting vs OMPL's RRTConnect + simplifier.

Both planners use the *same* state validity checker (MuJoCo, 1 cm clearance)
and the *same* edge validator (swept-displacement-bounded discretisation), so
the comparison isolates the planning algorithms and post-processing. Every
final path is re-validated with a zero-margin checker at fine resolution.

Queries: pre-grasp pose over the pick zone -> pre-place pose over the place
zone (across the divider), random yaws, TCP 6 cm above the table.

    python scripts/compare_ompl.py --n 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from pnp.control.grasp import top_down_pose
from pnp.ik.kinematics import SerialChain
from pnp.ik.solver import DLSSolver
from pnp.planning.postprocess import path_length, shortcut
from pnp.planning.rrt_connect import PlannerConfig, RRTConnect
from pnp.sim.collision import CollisionChecker
from pnp.sim.scene import GEOM, HOME_Q, JOINT_LIMITS, build_model


def make_queries(n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    m = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cc = CollisionChecker(m)
    cc.sync(d)
    ik = DLSSolver(SerialChain.from_mujoco(m))
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        qs = []
        for xr, yr in ((GEOM.pick_x, GEOM.pick_y), (GEOM.place_x, GEOM.place_y)):
            T = top_down_pose([rng.uniform(*xr), rng.uniform(*yr), 0.06], rng.uniform(-np.pi / 2, np.pi / 2))
            r = ik.solve(T, HOME_Q, rng=rng, validator=lambda q: not cc.in_collision(q))
            qs.append(r.q if r.success else None)
        if all(q is not None for q in qs):
            out.append((qs[0], qs[1]))
    return out


def dense_valid(path: list[np.ndarray], cc0: CollisionChecker) -> bool:
    for a, b in zip(path[:-1], path[1:]):
        n = max(2, int(np.ceil(np.max(np.abs(b - a)) / 0.002)))
        for k in range(n + 1):
            if cc0.in_collision(a + (b - a) * k / n):
                return False
    return True


def run_ours(queries, cc, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for qa, qb in queries:
        c0 = cc.stats.checks
        rrt = RRTConnect(lambda q: not cc.in_collision(q), cc.segment_free, JOINT_LIMITS,
                         PlannerConfig(timeout_s=10.0), rng)
        res = rrt.plan(qa, qb)
        t0 = time.perf_counter()
        path = shortcut(res.path, cc.segment_free, iters=150, rng=rng) if res.success else None
        t_sc = time.perf_counter() - t0
        rows.append(dict(ok=res.success, t_plan=res.time_s, t_post=t_sc,
                         raw_len=path_length(res.path) if res.success else np.nan,
                         len=path_length(path) if path else np.nan, verts=len(path) if path else 0,
                         checks=cc.stats.checks - c0, path=path))
    return rows


def run_ompl(queries, cc):
    from ompl import base as ob
    from ompl import geometric as og

    sp = ob.RealVectorStateSpace(6)
    b = ob.RealVectorBounds(6)
    for i in range(6):
        b.setLow(i, float(JOINT_LIMITS[i, 0]))
        b.setHigh(i, float(JOINT_LIMITS[i, 1]))
    sp.setBounds(b)
    ss = og.SimpleSetup(sp)
    si = ss.getSpaceInformation()

    def to_np(s) -> np.ndarray:
        return np.array([s[i] for i in range(6)])

    ss.setStateValidityChecker(lambda s: not cc.in_collision(to_np(s)))

    class SweptMotionValidator(ob.MotionValidator):
        def __init__(self, si_):
            super().__init__(si_)

        def checkMotion(self, s1, s2, *args):  # noqa: N802 (OMPL API)
            return cc.segment_free(to_np(s1), to_np(s2))

    mv = SweptMotionValidator(si)
    si.setMotionValidator(mv)
    planner = og.RRTConnect(si)
    planner.setRange(0.25 * 1.2)  # comparable step to ours (weighted metric ~1.2x L2)
    ss.setPlanner(planner)
    rows = []
    for qa, qb in queries:
        ss.clear()
        st, gl = sp.allocState(), sp.allocState()
        for i in range(6):
            st[i], gl[i] = float(qa[i]), float(qb[i])
        ss.setStartAndGoalStates(st, gl)
        c0 = cc.stats.checks
        t0 = time.perf_counter()
        solved = ss.solve(10.0)
        t_plan = time.perf_counter() - t0
        ok = bool(solved) and str(solved) == "Exact solution"
        raw = path = None
        t_post = 0.0
        if ok:
            p = ss.getSolutionPath()
            raw = [to_np(p.getState(i)) for i in range(p.getStateCount())]
            t1 = time.perf_counter()
            ss.simplifySolution()
            t_post = time.perf_counter() - t1
            p = ss.getSolutionPath()
            path = [to_np(p.getState(i)) for i in range(p.getStateCount())]
        rows.append(dict(ok=ok, t_plan=t_plan, t_post=t_post, raw_len=path_length(raw) if raw else np.nan,
                         len=path_length(path) if path else np.nan, verts=len(path) if path else 0,
                         checks=cc.stats.checks - c0, path=path))
    return rows


def summary(rows, cc0) -> dict:
    ok = [r for r in rows if r["ok"]]
    valid = sum(dense_valid(r["path"], cc0) for r in ok)
    f = lambda k, q: float(np.percentile([r[k] for r in ok], q))  # noqa: E731
    return {
        "success": f"{len(ok)}/{len(rows)}",
        "plan_ms_median": 1e3 * f("t_plan", 50), "plan_ms_p95": 1e3 * f("t_plan", 95),
        "post_ms_median": 1e3 * f("t_post", 50),
        "raw_len_median": f("raw_len", 50), "len_median": f("len", 50),
        "verts_median": f("verts", 50), "checks_median": f("checks", 50),
        "dense_valid": f"{valid}/{len(ok)}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    m = build_model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cc, cc0 = CollisionChecker(m, 0.01), CollisionChecker(m, 0.0)
    cc.sync(d)
    cc0.sync(d)
    q = make_queries(a.n, a.seed)
    res = {"ours (RRT-Connect + shortcut)": summary(run_ours(q, cc, a.seed), cc0)}
    try:
        res["OMPL 2.x (RRTConnect + simplifySolution)"] = summary(run_ompl(q, cc), cc0)
    except ImportError:
        print("OMPL not installed; skipping")
    a.out.mkdir(exist_ok=True, parents=True)
    (a.out / "ompl_comparison.json").write_text(json.dumps(res, indent=1))
    hdr = ("| Planner | Solved | RRT time med / p95 (ms) | Post-proc med (ms) | Raw length (rad) | "
           "Final length (rad) | Vertices | Collision checks | Dense re-check clean |")
    lines = [hdr, "|---|---|---|---|---|---|---|---|---|"]
    for k, s in res.items():
        lines.append(f"| {k} | {s['success']} | {s['plan_ms_median']:.0f} / {s['plan_ms_p95']:.0f} | "
                     f"{s['post_ms_median']:.0f} | {s['raw_len_median']:.2f} | {s['len_median']:.2f} | "
                     f"{s['verts_median']:.0f} | {s['checks_median']:.0f} | {s['dense_valid']} |")
    md = "\n".join(lines)
    (a.out / "ompl_comparison.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
