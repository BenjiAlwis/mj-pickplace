"""Pick-and-place quality metrics -- explicit definitions.

All quantities are measured from the **physical simulation state after the
gripper has opened, retreated and the object has been allowed to settle**,
never from the plan. A plan that "should" have worked but dropped the box is a
failure.

Per-episode metrics
-------------------
success
    All of: (a) the pipeline completed; (b) planar placement error
    ``e_xy <= TOL_XY``; (c) yaw error ``e_yaw <= TOL_YAW``; (d) tilt
    ``<= TOL_TILT``; (e) the gripper is not touching the object; (f) the object
    is at rest (``|v| < 1 cm/s``).
placement_error_xy
    ``|| p_obj,xy - p_target,xy ||`` in mm.
yaw_error
    Smallest rotation about z between object and target, modulo the box's
    90-degree symmetry, in degrees (range [0, 45]).
tilt
    Angle between the object's z-axis and world z, in degrees.
grasp_success
    The object was lifted >= LIFT_HEIGHT - 3 cm with both pads in contact.
plan_time_s
    Wall-clock time spent in IK, grasp synthesis, RRT-Connect, shortcutting and
    Cartesian planning for the whole episode (single-threaded Python).
joint_path_length
    Executed joint-space path length, sum of L2 norms of joint increments (rad).
tcp_path_length
    Executed TCP path length (m), from the planned samples.
ik_failures
    Count of IK queries that failed to return a valid (converged and
    collision-free) solution during the episode (including rejected grasp
    candidates), i.e. pressure on the IK layer, not necessarily an episode failure.
cycle_time_s
    Simulated time from episode start to the end of the retreat.
failure
    ``FAIL_<STATE>_<reason>`` for pipeline failures, ``PLACE_TOL_<which>`` for
    physical tolerance violations, ``None`` on success.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from pnp.control.grasp import gripper_touching_object
from pnp.sim.scene import Indices

TOL_XY = 0.015  # m
TOL_YAW = np.deg2rad(10.0)
TOL_TILT = np.deg2rad(5.0)
TOL_REST = 0.01  # m/s


@dataclass
class EpisodeMetrics:
    seed: int
    success: bool
    pipeline_completed: bool
    grasp_success: bool
    placement_error_xy_mm: float
    yaw_error_deg: float
    tilt_deg: float
    plan_time_s: float
    rrt_queries: int
    joint_path_length_rad: float
    tcp_path_length_m: float
    ik_failures: int
    cycle_time_s: float
    max_tracking_err_rad: float
    in_hand_slip_mm: float
    failure: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def placement_errors(model: mujoco.MjModel, data: mujoco.MjData, ix: Indices,
                     target_xy: np.ndarray, target_yaw: float) -> tuple[float, float, float, float]:
    """Return ``(e_xy [m], e_yaw [rad], tilt [rad], speed [m/s])`` of the object."""
    p = data.xpos[ix.obj_body]
    R = data.xmat[ix.obj_body].reshape(3, 3)
    e_xy = float(np.linalg.norm(p[:2] - target_xy))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    d = (yaw - target_yaw) % (np.pi / 2)
    e_yaw = float(min(d, np.pi / 2 - d))
    tilt = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
    speed = float(np.linalg.norm(data.qvel[ix.obj_dof: ix.obj_dof + 3]))
    return e_xy, e_yaw, tilt, speed


def judge(model: mujoco.MjModel, data: mujoco.MjData, ix: Indices, target_xy: np.ndarray,
          target_yaw: float, completed: bool, failure: str | None) -> tuple[bool, str | None, tuple]:
    e_xy, e_yaw, tilt, speed = placement_errors(model, data, ix, target_xy, target_yaw)
    if not completed:
        return False, failure, (e_xy, e_yaw, tilt)
    reasons = []
    if e_xy > TOL_XY:
        reasons.append("xy")
    if e_yaw > TOL_YAW:
        reasons.append("yaw")
    if tilt > TOL_TILT:
        reasons.append("tilt")
    if gripper_touching_object(model, data, ix):
        reasons.append("still_in_gripper")
    if speed > TOL_REST:
        reasons.append("moving")
    if reasons:
        return False, "PLACE_TOL_" + "+".join(reasons), (e_xy, e_yaw, tilt)
    return True, None, (e_xy, e_yaw, tilt)
