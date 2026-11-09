"""End-to-end: a full physical pick-and-place episode, scored from the sim state."""
import numpy as np
import pytest

from pnp.eval.run_eval import run_episode, wilson


@pytest.mark.parametrize("seed", [0, 1])
def test_episode_succeeds(seed):
    r = run_episode(seed, "nominal")
    assert r["success"], r["failure"]
    assert r["placement_error_xy_mm"] < 5.0
    assert r["grasp_success"]
    assert r["unplanned_contact_steps"] == 0
    assert r["max_tracking_err_rad"] < 0.02


def test_episode_is_deterministic():
    a, b = run_episode(3, "nominal"), run_episode(3, "nominal")
    assert a["placement_error_xy_mm"] == pytest.approx(b["placement_error_xy_mm"], abs=1e-9)


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and np.isclose(lo + hi, 1.0)
    assert wilson(100, 100)[1] == pytest.approx(1.0)
