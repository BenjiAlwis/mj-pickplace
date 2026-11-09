"""Minimal SO(3)/SE(3) Lie-group utilities written from scratch.

Conventions
-----------
* Rotations are 3x3 numpy arrays, homogeneous transforms 4x4.
* Twists are ordered ``[v, w]`` (linear first, angular second), matching the
  ``[jacp; jacr]`` stacking used by MuJoCo and by :mod:`pnp.ik`.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

_EPS = 1e-12


def skew(w: Array) -> Array:
    """Return the 3x3 skew-symmetric matrix ``[w]x`` such that ``[w]x @ v = w x v``."""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(w: Array) -> Array:
    """Rodrigues' formula: exponential of the rotation vector ``w`` (axis * angle)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-9:
        return np.eye(3) + skew(w)
    k = skew(w / theta)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def so3_log(R: Array) -> Array:
    """Logarithm of a rotation matrix as a rotation vector, robust near 0 and pi.

    Near ``theta = pi`` the antisymmetric part vanishes, so the axis is recovered
    from the symmetric part ``(R + I) / 2 = k k^T`` instead.
    """
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    if theta < 1e-6:
        # First-order: R ~ I + [w]x
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-4:
        B = 0.5 * (R + np.eye(3))
        i = int(np.argmax(np.diag(B)))
        k = B[:, i] / np.sqrt(max(B[i, i], _EPS))
        # Fix the sign using the (tiny) antisymmetric part when available.
        anti = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if anti @ k < 0:
            k = -k
        return theta * k / np.linalg.norm(k)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    return theta * np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])


def se3_exp(S: Array, q: float) -> Array:
    """Exponential of a unit screw ``S = [v, w]`` scaled by ``q`` (Lynch & Park eq. 3.88).

    For a revolute joint ``|w| = 1`` and ``q`` is the joint angle.
    """
    v, w = S[:3], S[3:]
    T = np.eye(4)
    if np.linalg.norm(w) < 1e-12:  # prismatic
        T[:3, 3] = v * q
        return T
    W = skew(w)
    R = so3_exp(w * q)
    G = np.eye(3) * q + (1.0 - np.cos(q)) * W + (q - np.sin(q)) * (W @ W)
    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


def adjoint(T: Array) -> Array:
    """6x6 adjoint of ``T`` acting on twists ordered ``[v, w]``."""
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[:3, 3:] = skew(p) @ R
    return Ad


def make_T(R: Array, p: Array) -> Array:
    """Assemble a homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rot_z(yaw: float) -> Array:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_mat(q: Array) -> Array:
    """MuJoCo-convention quaternion ``[w, x, y, z]`` to rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R: Array) -> Array:
    """Rotation matrix to ``[w, x, y, z]`` quaternion (Shepperd's method), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def rotation_angle(R_a: Array, R_b: Array) -> float:
    """Geodesic distance (rad) between two rotations."""
    return float(np.linalg.norm(so3_log(R_a.T @ R_b)))
