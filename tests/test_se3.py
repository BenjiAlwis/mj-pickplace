import numpy as np
import pytest

from pnp.se3 import mat_to_quat, quat_to_mat, rotation_angle, so3_exp, so3_log


def _random_rotvec(rng, max_angle=np.pi):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return axis * rng.uniform(0, max_angle)


def test_exp_is_rotation(rng):
    for _ in range(100):
        R = so3_exp(_random_rotvec(rng))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


@pytest.mark.parametrize("max_angle", [1e-8, 1e-3, 1.0, np.pi - 1e-6])
def test_log_exp_roundtrip(rng, max_angle):
    for _ in range(100):
        w = _random_rotvec(rng, max_angle)
        R = so3_exp(w)
        assert rotation_angle(so3_exp(so3_log(R)), R) < 1e-7


def test_log_at_exactly_pi():
    for axis in np.eye(3):
        R = so3_exp(np.pi * axis)
        w = so3_log(R)
        assert np.isclose(np.linalg.norm(w), np.pi)
        assert np.allclose(so3_exp(w), R, atol=1e-9)


def test_quaternion_roundtrip(rng):
    for _ in range(100):
        R = so3_exp(_random_rotvec(rng))
        assert np.allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-12)
