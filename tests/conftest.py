"""Shared fixtures: the compiled workcell is built once per test session."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from pnp.ik.kinematics import SerialChain
from pnp.ik.solver import DLSSolver
from pnp.sim.collision import CollisionChecker
from pnp.sim.scene import HOME_Q, build_model, indices


@pytest.fixture(scope="session")
def model() -> mujoco.MjModel:
    return build_model()


@pytest.fixture()
def data(model):
    d = mujoco.MjData(model)
    ix = indices(model)
    d.qpos[ix.arm_qpos] = HOME_Q
    mujoco.mj_forward(model, d)
    return d


@pytest.fixture(scope="session")
def ix(model):
    return indices(model)


@pytest.fixture(scope="session")
def chain(model) -> SerialChain:
    return SerialChain.from_mujoco(model)


@pytest.fixture(scope="session")
def ik(chain) -> DLSSolver:
    return DLSSolver(chain)


@pytest.fixture()
def checker(model, data) -> CollisionChecker:
    c = CollisionChecker(model, margin=0.01)
    c.sync(data)
    return c


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
