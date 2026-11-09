"""Pick-and-place as an explicit finite-state machine.

    APPROACH -> PRE_GRASP -> GRASP -> LIFT -> TRANSPORT -> PLACE -> RETREAT -> DONE
       (RRT)    (Cartesian)  (close) (Cart.)   (RRT,held)  (Cart.+open) (Cart.+RRT)

Every state either advances or terminates the episode with a named failure
(``FAIL_<STATE>_<reason>``), so evaluation can attribute each failure to a
pipeline stage instead of reporting a single opaque success rate.

Perception is assumed perfect (object pose read from the simulator), but the
*in-hand* object pose is re-measured after the lift, and the place pose is
computed from it: ``T_tcp_place = T_obj_target * inv(T_tcp_obj_measured)``.
That compensates for any slip or rotation during grasp closure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import mujoco
import numpy as np

from pnp.control.cartesian import cartesian_move
from pnp.control.executor import GRIPPER_CLOSED, GRIPPER_OPEN, Executor
from pnp.control.grasp import PREGRASP_HEIGHT, grasp_candidates, object_pose, pad_contacts, wrap_pi
from pnp.ik.kinematics import SerialChain
from pnp.ik.solver import DLSSolver
from pnp.planning.planner import MotionPlanner
from pnp.planning.postprocess import path_length
from pnp.se3 import Array, make_T, rot_z
from pnp.sim.collision import CollisionChecker
from pnp.sim.scene import GEOM, HOME_Q, Indices

LIFT_HEIGHT = 0.15
PLACE_CLEARANCE = 0.003  # release the object this far above the table (m)
RETREAT_HEIGHT = 0.10


class State(Enum):
    APPROACH = "approach"
    PRE_GRASP = "pre_grasp"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    PLACE = "place"
    RETREAT = "retreat"
    DONE = "done"


@dataclass
class EpisodeLog:
    state_trace: list[str] = field(default_factory=list)
    failure: str | None = None
    plan_time_s: float = 0.0  # all planning (IK, RRT, shortcut, Cartesian) wall clock
    rrt_queries: int = 0
    joint_path_length: float = 0.0  # rad, executed joint-space path (all phases)
    tcp_path_length: float = 0.0  # m, executed TCP path (all phases, from planned samples)
    ik_failures: int = 0
    grasp_candidates_feasible: int = 0
    grasp_yaw: float = float("nan")
    lifted: bool = False
    in_hand_slip_mm: float = float("nan")


class PickAndPlace:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ix: Indices,
                 rng: np.random.Generator, executor: Executor | None = None,
                 obs_noise_xy: float = 0.0, obs_noise_yaw: float = 0.0,
                 inhand_compensation: bool = True) -> None:
        self.m, self.d, self.ix = model, data, ix
        self.rng = rng
        self.chain = SerialChain.from_mujoco(model)
        self.ik = DLSSolver(self.chain)
        self.checker = CollisionChecker(model)
        self.planner = MotionPlanner(self.checker, rng=rng)
        self.ex = executor or Executor(model, data, ix)
        self.log = EpisodeLog()
        # Perception model: Gaussian noise on the observed object pose (m, rad).
        self.obs_noise_xy = obs_noise_xy
        self.obs_noise_yaw = obs_noise_yaw
        # If False, the place pose assumes the object sits exactly where the
        # grasp was planned (no re-measurement after lift) -- an ablation.
        self.inhand_compensation = inhand_compensation

    # --------------------------------------------------------------- helpers
    def _tcp_T(self) -> Array:
        return make_T(self.d.site_xmat[self.ix.tcp_site].reshape(3, 3).copy(),
                      self.d.site_xpos[self.ix.tcp_site].copy())

    def _obj_T(self) -> Array:
        return make_T(self.d.xmat[self.ix.obj_body].reshape(3, 3).copy(), self.d.xpos[self.ix.obj_body].copy())

    def _account(self, q_samples: list[Array] | Array) -> None:
        qs = list(q_samples)
        self.log.joint_path_length += path_length(qs) if len(qs) > 1 else 0.0
        P = [self.chain.fk(q)[:3, 3] for q in qs]
        self.log.tcp_path_length += float(sum(np.linalg.norm(b - a) for a, b in zip(P[:-1], P[1:])))

    def _free_move(self, q_goal: Array) -> bool:
        self.checker.sync(self.d)
        t0 = time.perf_counter()
        plan = self.planner.plan(self.ex.q, q_goal)
        self.log.plan_time_s += time.perf_counter() - t0
        self.log.rrt_queries += 1
        if not plan.success:
            return False
        dense = [plan.trajectory.sample(t)[0] for t in np.linspace(0, plan.trajectory.duration, 200)]
        self._account(dense)
        self.ex.follow(plan.trajectory)
        return True

    def _cart_move(self, T_goal: Array, **relax: bool) -> str:
        self.checker.sync(self.d)
        with self.checker.relaxed(**relax):
            res = cartesian_move(self.ik, self.checker, self.ex.q, T_goal)
        self.log.plan_time_s += res.time_s
        self.log.ik_failures += res.ik_failures
        if not res.success:
            return res.reason
        self._account(res.traj.q)
        self.ex.follow(res.traj)
        return "ok"

    def _fail(self, state: State, reason: str) -> bool:
        self.log.failure = f"FAIL_{state.name}_{reason}"
        return False

    # ------------------------------------------------------------ the machine
    def run(self, target_xy: Array, target_yaw: float) -> bool:
        """Execute one pick-and-place. Returns True if the pipeline completed.

        Completion is *not* success: success is judged afterwards from the
        physical object pose (see :mod:`pnp.eval.metrics`).
        """
        state = State.APPROACH
        cand = None
        while state != State.DONE:
            self.log.state_trace.append(state.value)

            if state == State.APPROACH:
                obj_p, obj_yaw = object_pose(self.d, self.ix)
                obj_p = obj_p + np.r_[self.rng.normal(0.0, self.obs_noise_xy, 2), 0.0]
                obj_yaw = obj_yaw + float(self.rng.normal(0.0, self.obs_noise_yaw))
                T_obj_belief = make_T(rot_z(obj_yaw), obj_p)
                self.checker.sync(self.d)
                self.checker.attach(None)
                t0 = time.perf_counter()
                cands, fails = grasp_candidates(self.ik, self.checker, obj_p, obj_yaw, self.ex.q, self.rng)
                self.log.plan_time_s += time.perf_counter() - t0
                self.log.ik_failures += fails
                self.log.grasp_candidates_feasible = len(cands)
                moved = False
                for cand in cands:  # best first; fall back if a plan fails
                    if self._free_move(cand.q_pre):
                        moved = True
                        break
                if not cands:
                    return self._fail(state, "no_ik_grasp")
                if not moved:
                    return self._fail(state, "plan")
                self.log.grasp_yaw = cand.yaw
                self._T_tcp_obj_nominal = np.linalg.inv(cand.T_grasp) @ T_obj_belief
                state = State.PRE_GRASP

            elif state == State.PRE_GRASP:
                self.ex.gripper(GRIPPER_OPEN)
                r = self._cart_move(cand.T_grasp, gripper_object=True)
                if r != "ok":
                    return self._fail(state, r)
                state = State.GRASP

            elif state == State.GRASP:
                self.ex.gripper(GRIPPER_CLOSED, seconds=0.8)
                left, right = pad_contacts(self.m, self.d, self.ix)
                if not (left and right):
                    return self._fail(state, "no_pad_contact")
                state = State.LIFT

            elif state == State.LIFT:
                T_grip_obj = np.linalg.inv(self._tcp_T()) @ self._obj_T()
                self.checker.attach(T_grip_obj)
                T_up = self._tcp_T()
                T_up[2, 3] += LIFT_HEIGHT
                r = self._cart_move(T_up, object_table=True)
                if r != "ok":
                    return self._fail(state, r)
                lifted_z = self.d.xpos[self.ix.obj_body][2] - GEOM.box_half[2] - GEOM.table_top_z
                left, right = pad_contacts(self.m, self.d, self.ix)
                self.log.lifted = bool(lifted_z > LIFT_HEIGHT - 0.03 and left and right)
                if not self.log.lifted:
                    return self._fail(state, "dropped")
                # Re-measure the in-hand pose after the lift (slip compensation).
                T_meas = np.linalg.inv(self._tcp_T()) @ self._obj_T()
                self.log.in_hand_slip_mm = float(np.linalg.norm(T_meas[:3, 3] - T_grip_obj[:3, 3]) * 1e3)
                self.checker.attach(T_meas if self.inhand_compensation else self._T_tcp_obj_nominal)
                state = State.TRANSPORT

            elif state == State.TRANSPORT:
                T_tcp_obj = self.checker.attached_T
                z_obj = GEOM.table_top_z + GEOM.box_half[2] + PLACE_CLEARANCE
                best = None
                self.checker.sync(self.d)
                t0 = time.perf_counter()
                for k in range(4):  # square cross-section: 4 equivalent object yaws
                    yaw = wrap_pi(target_yaw + k * np.pi / 2)
                    T_obj = make_T(rot_z(yaw), np.array([target_xy[0], target_xy[1], z_obj]))
                    T_place = T_obj @ np.linalg.inv(T_tcp_obj)
                    T_pre = T_place.copy()
                    T_pre[2, 3] += PREGRASP_HEIGHT
                    r = self.ik.solve(T_pre, self.ex.q, rng=self.rng,
                                      validator=lambda q: not self.checker.in_collision(q))
                    if not r.success:
                        self.log.ik_failures += 1
                        continue
                    cost = float(np.linalg.norm(r.q - self.ex.q))
                    if best is None or cost < best[0]:
                        best = (cost, r.q, T_place)
                self.log.plan_time_s += time.perf_counter() - t0
                if best is None:
                    return self._fail(state, "no_ik_place")
                self._place_T = best[2]
                if not self._free_move(best[1]):
                    return self._fail(state, "plan")
                state = State.PLACE

            elif state == State.PLACE:
                r = self._cart_move(self._place_T, object_table=True)
                if r != "ok":
                    return self._fail(state, r)
                self.ex.gripper(GRIPPER_OPEN, seconds=0.6)
                self.checker.attach(None)
                state = State.RETREAT

            elif state == State.RETREAT:
                T_up = self._tcp_T()
                T_up[2, 3] += RETREAT_HEIGHT
                r = self._cart_move(T_up, gripper_object=True)
                if r != "ok":
                    return self._fail(state, r)
                if not self._free_move(HOME_Q):
                    return self._fail(state, "plan")
                state = State.DONE

        self.log.state_trace.append(State.DONE.value)
        return True


def reset_episode(model: mujoco.MjModel, data: mujoco.MjData, ix: Indices,
                  obj_xy: Array, obj_yaw: float, target_xy: Array, target_yaw: float) -> None:
    """Put the robot at home, the object on the table and the target pad in place."""
    mujoco.mj_resetData(model, data)
    data.qpos[ix.arm_qpos] = HOME_Q
    data.ctrl[ix.arm_ctrl] = HOME_Q
    data.ctrl[ix.gripper_ctrl] = GRIPPER_OPEN
    a = ix.obj_qpos
    data.qpos[a: a + 3] = [obj_xy[0], obj_xy[1], GEOM.table_top_z + GEOM.box_half[2]]
    data.qpos[a + 3: a + 7] = [np.cos(obj_yaw / 2), 0, 0, np.sin(obj_yaw / 2)]
    data.mocap_pos[ix.target_mocap] = [target_xy[0], target_xy[1], GEOM.table_top_z + 0.0005]
    data.mocap_quat[ix.target_mocap] = [np.cos(target_yaw / 2), 0, 0, np.sin(target_yaw / 2)]
    mujoco.mj_forward(model, data)
    # Let contacts settle before the episode starts.
    for _ in range(100):
        mujoco.mj_step(model, data)
