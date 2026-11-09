"""Render one pick-and-place episode to GIF/MP4 (headless; uses MUJOCO_GL=egl if unset).

    python scripts/record.py --seed 3 --out media/pick_place.gif
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pnp.control.executor import Executor  # noqa: E402
from pnp.control.pipeline import PickAndPlace, reset_episode  # noqa: E402
from pnp.eval.metrics import judge  # noqa: E402
from pnp.eval.run_eval import sample_task  # noqa: E402
from pnp.sim.scene import build_model, indices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("media/pick_place.gif"))
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--speedup", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--still", action="store_true", help="only save the first frame as PNG")
    a = ap.parse_args()

    m = build_model()
    d = mujoco.MjData(m)
    ix = indices(m)
    rng = np.random.default_rng(a.seed)
    oxy, oyaw, txy, tyaw = sample_task(rng)
    reset_episode(m, d, ix, oxy, oyaw, txy, tyaw)

    renderer = mujoco.Renderer(m, a.height, a.width)
    opt = mujoco.MjvOption()
    frames: list[np.ndarray] = []

    def grab(data: mujoco.MjData) -> None:
        renderer.update_scene(data, camera="overview", scene_option=opt)
        frames.append(renderer.render().copy())

    if a.still:
        grab(d)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(a.out.with_suffix(".png"), frames[0])
        return

    ex = Executor(m, d, ix, frame_cb=grab, frame_every=a.speedup / a.fps)
    pp = PickAndPlace(m, d, ix, np.random.default_rng(10_000 + a.seed), executor=ex)
    done = pp.run(txy, tyaw)
    ex.hold(0.5)
    ok, fail, (e_xy, e_yaw, _) = judge(m, d, ix, txy, tyaw, done, pp.log.failure)
    print(f"success={ok} failure={fail} placement error={e_xy*1e3:.1f} mm, yaw error={np.rad2deg(e_yaw):.1f} deg, "
          f"{len(frames)} frames")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.suffix == ".gif":
        iio.imwrite(a.out, frames, duration=1000 / a.fps, loop=0)
    else:
        iio.imwrite(a.out, np.stack(frames), fps=a.fps, codec="libx264")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
