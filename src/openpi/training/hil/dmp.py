"""Dynamical movement primitive (DMP) base policy for residual RL.

This is a small NumPy port of the parts of ``movement_primitives`` that the
reference implementation uses (``DMP`` with ``imitate`` / ``configure`` /
``open_loop``):

    https://github.com/ColeDewis/wam-haptic-dmps
    https://github.com/dfki-ric/movement_primitives

Keeping the implementation local avoids adding a Cython/optional-dependency
chain to the LIBERO training environment.  The equations and the public method
names follow ``movement_primitives`` closely so behaviour matches the reference.

``DmpBasePolicy`` trains one DMP on a demonstrated end-effector pose trajectory
and exposes a *normalized delta-pose* action for the current step.  The RL agent
learns a residual on top of that action:

    final_delta = dmp_delta + residual_delta

The gripper is deliberately not part of the DMP; the gripper DQN keeps learning
it independently.

The DMP implementation is adapted from ``movement_primitives``, which is
distributed under the BSD-3-Clause licence:

    Copyright Alexander Fabisch (DFKI GmbH, Robotics Innovation Center),
    Jannik Klemm (University of Bremen, Robotics Research Group),
    and movement_primitives contributors.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the conditions of the
    BSD-3-Clause licence are met.  See the upstream repository for the full
    licence text.
"""

# ruff: noqa: N803, N806, FBT001, FBT002

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Small helpers shared by the DMP implementation
# ---------------------------------------------------------------------------


def _ensure_1d_array(value, n_dims: int, name: str) -> np.ndarray:
    value = np.atleast_1d(value).astype(np.float64)
    if value.ndim == 1 and value.shape[0] == 1:
        value = np.repeat(value, n_dims)
    if value.ndim > 1 or value.shape[0] != n_dims:
        raise ValueError(f"{name} has incorrect shape, expected ({n_dims},), got {value.shape}")
    return value


def _canonical_system_alpha(goal_z: float, goal_t: float, start_t: float) -> float:
    if goal_z <= 0.0:
        raise ValueError("final phase must be > 0")
    if start_t >= goal_t:
        raise ValueError("goal must be chronologically after start")
    return float(-np.log(goal_z))


def _phase(t, alpha: float, goal_t: float, start_t: float):
    """Map time to the DMP phase variable ``z``."""
    execution_time = goal_t - start_t
    return np.exp(-alpha * (np.asarray(t) - start_t) / execution_time)


class ForcingTerm:
    """Radial-basis forcing term that shapes a DMP trajectory."""

    def __init__(
        self,
        n_dims: int,
        n_weights_per_dim: int,
        goal_t: float,
        start_t: float,
        overlap: float = 0.8,
        alpha_z: float | None = None,
    ):
        if n_weights_per_dim <= 1:
            raise ValueError("n_weights_per_dim must be > 1")
        if start_t >= goal_t:
            raise ValueError("goal must be chronologically after start")
        self.n_weights_per_dim = int(n_weights_per_dim)
        self.goal_t = float(goal_t)
        self.start_t = float(start_t)
        self.overlap = float(overlap)
        self.alpha_z = _canonical_system_alpha(0.01, self.goal_t, self.start_t) if alpha_z is None else float(alpha_z)

        self.log_overlap = float(-math.log(self.overlap))
        self.execution_time = self.goal_t - self.start_t
        self.weights_ = np.zeros((int(n_dims), self.n_weights_per_dim))
        self.centers = np.empty(self.n_weights_per_dim)
        self.widths = np.empty(self.n_weights_per_dim)

        # ``n_weights_per_dim - 1`` intervals between start and goal.
        step = self.execution_time / (self.n_weights_per_dim - 1)
        self.centers[0] = _phase(self.start_t, self.alpha_z, self.goal_t, self.start_t)
        for i in range(1, self.n_weights_per_dim):
            t = self.start_t + i * step
            self.centers[i] = _phase(t, self.alpha_z, self.goal_t, self.start_t)
            diff = self.centers[i] - self.centers[i - 1]
            self.widths[i - 1] = self.log_overlap / diff**2
        self.widths[-1] = self.widths[-2]

    def _activations(self, z):
        z = np.atleast_2d(z)  # (1, n_steps)
        squared_dist = (z - self.centers[:, np.newaxis]) ** 2
        activations = np.exp(-self.widths[:, np.newaxis] * squared_dist)
        return activations / activations.sum(axis=0)

    def design_matrix(self, T: np.ndarray) -> np.ndarray:
        """Return the (n_weights, n_steps) regression design matrix."""
        Z = _phase(T, self.alpha_z, T[-1], T[0])
        return Z[np.newaxis, :] * self._activations(Z)

    def phase(self, t):
        return _phase(t, self.alpha_z, self.goal_t, self.start_t)

    def forcing_term(self, z) -> np.ndarray:
        z = np.atleast_1d(z)
        return z[np.newaxis, :] * self.weights_.dot(self._activations(z))

    def __call__(self, t):
        return self.forcing_term(self.phase(t))

    @property
    def shape(self) -> tuple[int, int]:
        return self.weights_.shape


# ---------------------------------------------------------------------------
# DMP integration
# ---------------------------------------------------------------------------


def _dmp_acc(
    Y,
    V,
    cdd,
    alpha_y,
    beta_y,
    goal_y,
    goal_yd,
    goal_ydd,
    start_y,
    z,
    execution_time,
    f,
    coupling_term,
    tdd,
    smooth_scaling,
):
    if coupling_term is not None:
        _, cdd = coupling_term.coupling(Y, V)
    smoothing = beta_y * (goal_y - start_y) * z if smooth_scaling else 0.0
    return (alpha_y * (beta_y * (goal_y - Y) - execution_time * V - smoothing) + f + cdd + tdd) / execution_time**2


def _dmp_step_rk4(
    last_t,
    t,
    current_y,
    current_yd,
    goal_y,
    goal_yd,
    goal_ydd,
    start_y,
    start_yd,
    start_ydd,
    goal_t,
    start_t,
    alpha_y,
    beta_y,
    forcing_term,
    coupling_term=None,
    coupling_term_precomputed=None,
    int_dt=0.001,
    p_gain=0.0,
    tracking_error=0.0,
    smooth_scaling=False,
):
    if coupling_term is None:
        cd, cdd = np.zeros_like(current_y), np.zeros_like(current_y)
        if coupling_term_precomputed is not None:
            cd += coupling_term_precomputed[0]
            cdd += coupling_term_precomputed[1]
    else:
        cd, cdd = None, None

    execution_time = goal_t - start_t
    dt = t - last_t
    dt_2 = 0.5 * dt
    T = np.array([t, t + dt_2, t + dt])
    Z = forcing_term.phase(T)
    F = forcing_term.forcing_term(Z)
    tdd = p_gain * tracking_error / dt

    C0 = current_yd
    K0 = _dmp_acc(
        current_y,
        C0,
        cdd,
        alpha_y,
        beta_y,
        goal_y,
        goal_yd,
        goal_ydd,
        start_y,
        Z[0],
        execution_time,
        F[:, 0],
        coupling_term,
        tdd,
        smooth_scaling,
    )
    C1 = current_yd + dt_2 * K0
    K1 = _dmp_acc(
        current_y + dt_2 * C0,
        C1,
        cdd,
        alpha_y,
        beta_y,
        goal_y,
        goal_yd,
        goal_ydd,
        start_y,
        Z[1],
        execution_time,
        F[:, 1],
        coupling_term,
        tdd,
        smooth_scaling,
    )
    C2 = current_yd + dt_2 * K1
    K2 = _dmp_acc(
        current_y + dt_2 * C1,
        C2,
        cdd,
        alpha_y,
        beta_y,
        goal_y,
        goal_yd,
        goal_ydd,
        start_y,
        Z[1],
        execution_time,
        F[:, 1],
        coupling_term,
        tdd,
        smooth_scaling,
    )
    C3 = current_yd + dt * K2
    K3 = _dmp_acc(
        current_y + dt * C2,
        C3,
        cdd,
        alpha_y,
        beta_y,
        goal_y,
        goal_yd,
        goal_ydd,
        start_y,
        Z[2],
        execution_time,
        F[:, 2],
        coupling_term,
        tdd,
        smooth_scaling,
    )

    current_y += dt * (current_yd + dt / 6.0 * (K0 + K1 + K2))
    current_yd += dt / 6.0 * (K0 + 2 * K1 + 2 * K2 + K3)

    if coupling_term is not None:
        cd, _ = coupling_term.coupling(current_y, current_yd)
        current_yd += cd / execution_time


def _determine_forces(T, Y, alpha_y, beta_y, alpha_z, allow_final_velocity, smooth_scaling=False):
    n_dims = Y.shape[1]
    DT = np.gradient(T)
    Yd = np.empty_like(Y)
    for d in range(n_dims):
        Yd[:, d] = np.gradient(Y[:, d]) / DT
    if not allow_final_velocity:
        Yd[-1, :] = 0.0
    Ydd = np.empty_like(Y)
    for d in range(n_dims):
        Ydd[:, d] = np.gradient(Yd[:, d]) / DT
    Ydd[-1, :] = 0.0

    execution_time = T[-1] - T[0]
    goal_y = Y[-1]
    start_y = Y[0]
    Z = _phase(T, alpha_z, T[-1], T[0])
    F = np.empty((len(T), n_dims))
    for t in range(len(T)):
        smoothing = beta_y * (goal_y - start_y) * Z[t] if smooth_scaling else 0.0
        F[t, :] = execution_time**2 * Ydd[t] - alpha_y * (beta_y * (goal_y - Y[t]) - Yd[t] * execution_time - smoothing)
    return F, Y[0], Yd[0], Ydd[0], Y[-1], Yd[-1], Ydd[-1]


def _ridge_regression(X, Y, regularization_coefficient):
    return np.linalg.pinv(X.dot(X.T) + regularization_coefficient * np.eye(X.shape[0])).dot(X).dot(Y).T


def _dmp_imitate(
    T,
    Y,
    n_weights_per_dim,
    regularization_coefficient,
    alpha_y,
    beta_y,
    overlap,
    alpha_z,
    allow_final_velocity,
    smooth_scaling=False,
):
    if regularization_coefficient < 0.0:
        raise ValueError("regularization coefficient must be >= 0")
    forcing_term = ForcingTerm(Y.shape[1], n_weights_per_dim, T[-1], T[0], overlap, alpha_z)
    F, start_y, start_yd, start_ydd, goal_y, goal_yd, goal_ydd = _determine_forces(
        T, Y, alpha_y, beta_y, alpha_z, allow_final_velocity, smooth_scaling
    )
    X = forcing_term.design_matrix(T)
    weights = _ridge_regression(X, F, regularization_coefficient)
    return weights, start_y, start_yd, start_ydd, goal_y, goal_yd, goal_ydd


def _dmp_open_loop(
    goal_t,
    start_t,
    dt,
    start_y,
    goal_y,
    alpha_y,
    beta_y,
    forcing_term,
    run_t=None,
    int_dt=0.001,
    smooth_scaling=False,
    start_yd=None,
    start_ydd=None,
    goal_yd=None,
    goal_ydd=None,
):
    goal_yd = np.zeros_like(goal_y) if goal_yd is None else goal_yd
    goal_ydd = np.zeros_like(goal_y) if goal_ydd is None else goal_ydd
    start_yd = np.zeros_like(start_y) if start_yd is None else start_yd
    start_ydd = np.zeros_like(start_y) if start_ydd is None else start_ydd
    run_t = goal_t if run_t is None else run_t

    current_y = np.copy(start_y)
    current_yd = np.copy(start_yd)
    T = np.arange(start_t, run_t + dt, dt)
    Y = np.empty((len(T), len(current_y)))
    Y[0] = current_y
    for i in range(1, len(T)):
        _dmp_step_rk4(
            T[i - 1],
            T[i],
            current_y,
            current_yd,
            goal_y=goal_y,
            goal_yd=goal_yd,
            goal_ydd=goal_ydd,
            start_y=start_y,
            start_yd=start_yd,
            start_ydd=start_ydd,
            goal_t=goal_t,
            start_t=start_t,
            alpha_y=alpha_y,
            beta_y=beta_y,
            forcing_term=forcing_term,
            int_dt=int_dt,
            smooth_scaling=smooth_scaling,
        )
        Y[i] = current_y
    return T, Y


class DMP:
    """Discrete dynamical movement primitive (NumPy port)."""

    def __init__(
        self,
        n_dims: int,
        execution_time: float = 1.0,
        dt: float = 0.01,
        n_weights_per_dim: int = 10,
        int_dt: float = 0.001,
        smooth_scaling: bool = False,
        alpha_y=25.0,
        beta_y=6.25,
    ):
        self.n_dims = int(n_dims)
        self._execution_time = float(execution_time)
        self.dt_ = float(dt)
        self.n_weights_per_dim = int(n_weights_per_dim)
        self.int_dt = float(int_dt)
        self.smooth_scaling = bool(smooth_scaling)
        self.alpha_y = _ensure_1d_array(alpha_y, self.n_dims, "alpha_y")
        self.beta_y = _ensure_1d_array(beta_y, self.n_dims, "beta_y")

        self._init_forcing_term()

        self.t = 0.0
        self.last_t = None
        self.start_y = np.zeros(self.n_dims)
        self.start_yd = np.zeros(self.n_dims)
        self.start_ydd = np.zeros(self.n_dims)
        self.goal_y = np.zeros(self.n_dims)
        self.goal_yd = np.zeros(self.n_dims)
        self.goal_ydd = np.zeros(self.n_dims)
        self.current_y = np.zeros(self.n_dims)
        self.current_yd = np.zeros(self.n_dims)
        self.initialized = False

    def _init_forcing_term(self) -> None:
        alpha_z = _canonical_system_alpha(0.01, self.execution_time_, 0.0)
        self.forcing_term = ForcingTerm(self.n_dims, self.n_weights_per_dim, self.execution_time_, 0.0, 0.8, alpha_z)

    def get_execution_time_(self) -> float:
        return self._execution_time

    def set_execution_time_(self, execution_time: float) -> None:
        weights = self.forcing_term.weights_.copy()
        self._execution_time = float(execution_time)
        self._init_forcing_term()
        self.forcing_term.weights_ = weights

    execution_time_ = property(get_execution_time_, set_execution_time_)

    def configure(
        self,
        t=None,
        start_y=None,
        start_yd=None,
        start_ydd=None,
        goal_y=None,
        goal_yd=None,
        goal_ydd=None,
    ) -> None:
        if t is not None:
            self.t = float(t)
        for value, name in (
            (start_y, "start_y"),
            (start_yd, "start_yd"),
            (start_ydd, "start_ydd"),
            (goal_y, "goal_y"),
            (goal_yd, "goal_yd"),
            (goal_ydd, "goal_ydd"),
        ):
            if value is not None:
                setattr(self, name, _ensure_1d_array(value, self.n_dims, name))

    def reset(self) -> None:
        self.t = 0.0
        self.last_t = None
        self.current_y = np.copy(self.start_y)
        self.current_yd = np.copy(self.start_yd)
        self.initialized = False

    def step(self, last_y, last_yd):
        self.last_t = self.t
        self.t += self.dt_
        if not self.initialized:
            self.current_y = np.copy(self.start_y)
            self.current_yd = np.copy(self.start_yd)
            self.initialized = True

        tracking_error = self.current_y - np.asarray(last_y, dtype=np.float64)
        _dmp_step_rk4(
            self.last_t,
            self.t,
            self.current_y,
            self.current_yd,
            self.goal_y,
            self.goal_yd,
            self.goal_ydd,
            self.start_y,
            self.start_yd,
            self.start_ydd,
            self.execution_time_,
            0.0,
            self.alpha_y,
            self.beta_y,
            self.forcing_term,
            int_dt=self.int_dt,
            tracking_error=tracking_error,
            smooth_scaling=self.smooth_scaling,
        )
        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t: float | None = None):
        return _dmp_open_loop(
            self.execution_time_,
            0.0,
            self.dt_,
            self.start_y,
            self.goal_y,
            self.alpha_y,
            self.beta_y,
            self.forcing_term,
            run_t=run_t,
            int_dt=self.int_dt,
            smooth_scaling=self.smooth_scaling,
        )

    def imitate(self, T, Y, regularization_coefficient: float = 0.0) -> None:
        T = np.asarray(T, dtype=np.float64).reshape(-1)
        Y = np.asarray(Y, dtype=np.float64)
        if T.shape[0] != Y.shape[0]:
            raise ValueError(f"T and Y must have the same length, got {T.shape[0]} and {Y.shape[0]}")
        if T.shape[0] < 3:
            raise ValueError("a DMP demonstration needs at least 3 samples")
        if Y.ndim != 2 or Y.shape[1] != self.n_dims:
            raise ValueError(f"Y must have shape (n_steps, {self.n_dims}), got {Y.shape}")
        if not np.all(np.isfinite(T)) or not np.all(np.isfinite(Y)):
            raise ValueError("DMP demonstration contains non-finite values")

        weights, start_y, _, _, goal_y, _, _ = _dmp_imitate(
            T,
            Y,
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y,
            beta_y=self.beta_y,
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=False,
            smooth_scaling=self.smooth_scaling,
        )
        self.forcing_term.weights_[:, :] = weights
        self.configure(start_y=start_y, goal_y=goal_y)


class DmpBasePolicy:
    """DMP base policy that emits normalized delta-pose actions.

    Parameters
    ----------
    trajectory:
        Demonstrated pose trajectory, shape ``(n_steps, 3)`` for position-only
        runs, ``(n_steps, 6)`` for position + axis-angle rotation runs, or one
        extra column when ``gripper=True``.
    dt:
        Control period (seconds) used for imitation and rollout.
    rotation:
        Whether the DMP models the 3 rotation dimensions.
    gripper:
        Whether the DMP models the gripper command as an extra trajectory
        dimension.  The gripper entry of the emitted action is then the DMP's
        absolute normalized gripper target (not a delta), because LIBERO's
        gripper action is absolute.  When false, the gripper entry is zero and
        the RL residual supplies it.
    action_dim:
        Width of the RL action vector (4 = xyz + gripper, 7 = xyz + rot +
        gripper).
    max_step_m / max_step_rad:
        Normalization scales used by the LIBERO actor.  A DMP delta in metres
        is divided by ``max_step_m``; a delta in radians by ``max_step_rad``.
    """

    def __init__(
        self,
        trajectory: np.ndarray,
        *,
        dt: float,
        rotation: bool = False,
        gripper: bool = False,
        action_dim: int = 7,
        max_step_m: float = 0.02,
        max_step_rad: float = 0.05,
        n_weights_per_dim: int = 20,
        regularization_coefficient: float = 0.0,
    ):
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim != 2:
            raise ValueError(f"trajectory must be 2-D, got shape {trajectory.shape}")
        if len(trajectory) < 3:
            raise ValueError("a DMP demonstration needs at least 3 samples")
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("DMP demonstration contains non-finite values")

        self.rotation = bool(rotation)
        self.gripper = bool(gripper)
        self.pose_dim = 3 + (3 if self.rotation else 0) + (1 if self.gripper else 0)
        if trajectory.shape[1] != self.pose_dim:
            raise ValueError(
                f"trajectory must have {self.pose_dim} columns for rotation={self.rotation}, "
                f"gripper={self.gripper}, got {trajectory.shape[1]}"
            )
        if action_dim not in (4, 7):
            raise ValueError(f"action_dim must be 4 or 7, got {action_dim}")
        expected_action_dim = self.pose_dim if self.gripper else self.pose_dim + 1
        if action_dim != expected_action_dim:
            raise ValueError(
                f"action_dim={action_dim} does not match pose_dim={self.pose_dim} for "
                f"rotation={self.rotation}, gripper={self.gripper}"
            )
        if not (max_step_m > 0.0 and max_step_rad > 0.0):
            raise ValueError("max_step_m and max_step_rad must be positive")

        self.action_dim = int(action_dim)
        self.max_step_m = float(max_step_m)
        self.max_step_rad = float(max_step_rad)
        self.dt = float(dt)
        self.n_steps = len(trajectory)
        self.execution_time = (self.n_steps - 1) * self.dt
        self.goal = np.array(trajectory[-1], copy=True)

        self.dmp = DMP(
            n_dims=self.pose_dim,
            execution_time=self.execution_time,
            dt=self.dt,
            n_weights_per_dim=n_weights_per_dim,
        )
        T = np.linspace(0.0, self.execution_time, self.n_steps)
        self.dmp.imitate(T, trajectory, regularization_coefficient=regularization_coefficient)
        self.dmp.set_execution_time_(self.execution_time)

        self._trajectory: np.ndarray | None = None
        self._index = 0

    def reset(self, start_pose: np.ndarray) -> None:
        """Re-anchor the DMP at the current pose and generate a fresh rollout."""
        start_pose = np.asarray(start_pose, dtype=np.float64).reshape(-1)
        if start_pose.shape[0] != self.pose_dim:
            raise ValueError(f"start_pose must have {self.pose_dim} entries, got {start_pose.shape[0]}")
        zeros = np.zeros(self.pose_dim)
        self.dmp.configure(
            start_y=start_pose,
            start_yd=zeros,
            start_ydd=zeros,
            goal_y=self.goal,
            goal_yd=zeros,
            goal_ydd=zeros,
        )
        _, self._trajectory = self.dmp.open_loop()
        # Drop the first point: it is the current pose, so it would produce a
        # zero base action for the first control step.
        self._index = 1

    def action(self, current_pose: np.ndarray) -> np.ndarray:
        """Return the normalized base action for the current step.

        Position/rotation entries are ``next_dmp_target - current_pose`` divided
        by the actor's per-dimension normalization scale.  When ``gripper`` is
        enabled, the last entry is the DMP's absolute normalized gripper target
        (LIBERO gripper actions are absolute, not deltas).
        """
        if self._trajectory is None:
            raise RuntimeError("DmpBasePolicy.reset() must be called before action()")
        current_pose = np.asarray(current_pose, dtype=np.float64).reshape(-1)
        if current_pose.shape[0] != self.pose_dim:
            raise ValueError(f"current_pose must have {self.pose_dim} entries, got {current_pose.shape[0]}")

        if self._index < len(self._trajectory):
            target = self._trajectory[self._index]
            self._index += 1
        else:
            target = self.goal

        delta = target - current_pose
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[:3] = delta[:3] / self.max_step_m
        if self.rotation:
            action[3:6] = delta[3:6] / self.max_step_rad
        if self.gripper:
            action[-1] = target[-1]
        return np.clip(action, -1.0, 1.0)

    @property
    def rollout(self) -> np.ndarray | None:
        """The absolute DMP pose rollout generated by the last ``reset``."""
        return self._trajectory


def base_weight_from_residual(
    residual: np.ndarray,
    *,
    residual_scale: float = 1.0,
    min_base_weight: float = 0.0,
) -> np.ndarray:
    """Per-dimension weight on the base policy, driven by residual magnitude.

    ``residual_scale`` is in normalized action units: at ``|residual| ==
    residual_scale`` the base policy is fully suppressed, and at ``residual ==
    0`` it keeps full weight.  This is the simulation stand-in for the real
    robot's per-joint ``policy_torque_scale``; pass an externally supplied
    array as ``base_weight`` to ``combine_base_residual`` when integrating it.
    """
    residual = np.asarray(residual, dtype=np.float32)
    if residual_scale <= 0.0:
        raise ValueError("residual_scale must be positive")
    weight = 1.0 - np.abs(residual) / float(residual_scale)
    return np.clip(weight, float(min_base_weight), 1.0).astype(np.float32)


def combine_base_residual(
    base: np.ndarray,
    residual: np.ndarray,
    *,
    base_weight: np.ndarray | None = None,
    residual_scale: float = 1.0,
    min_base_weight: float = 0.0,
) -> np.ndarray:
    """Compose a base action and a residual action with a residual-driven weight.

    ``final = base_weight * base + residual`` (clipped to [-1, 1]).  The base
    term shrinks as the residual grows, so a small residual leaves the base
    policy in charge while a large residual can override it.  When
    ``base_weight`` is given (e.g. the real robot's ``policy_torque_scale``),
    it is used directly instead of being derived from the residual.
    """
    base = np.asarray(base, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    if base.shape != residual.shape:
        raise ValueError(f"base and residual shapes differ: {base.shape} vs {residual.shape}")
    if base_weight is None:
        base_weight = base_weight_from_residual(
            residual,
            residual_scale=residual_scale,
            min_base_weight=min_base_weight,
        )
    else:
        base_weight = np.asarray(base_weight, dtype=np.float32)
        if base_weight.shape != base.shape:
            raise ValueError(f"base_weight must have shape {base.shape}, got {base_weight.shape}")
        base_weight = np.clip(base_weight, 0.0, 1.0)
    return np.clip(base_weight * base + residual, -1.0, 1.0).astype(np.float32)


def residual_from_final(
    final: np.ndarray,
    base: np.ndarray,
    *,
    residual_scale: float = 1.0,
    min_base_weight: float = 0.0,
) -> np.ndarray:
    """Invert ``combine_base_residual`` for demonstration residual targets.

    Given the demonstrated final action and the base action, solve for the
    residual that would have produced it under the same magnitude-dependent
    base weight.  Done per dimension by bisection because the weight itself
    depends on the residual.  With ``residual_scale=1`` the composition is
    monotone on ``[-1, 1]``, so the bracket always contains the solution.
    """
    final = np.asarray(final, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    if final.shape != base.shape:
        raise ValueError(f"final and base shapes differ: {final.shape} vs {base.shape}")
    if residual_scale <= 0.0:
        raise ValueError("residual_scale must be positive")

    def _final_for_residual(b: float, r: float) -> float:
        weight = float(np.clip(1.0 - abs(r) / float(residual_scale), float(min_base_weight), 1.0))
        return float(np.clip(weight * b + r, -1.0, 1.0))

    out = np.empty_like(final, dtype=np.float32)
    for idx in np.ndindex(final.shape):
        target = float(final[idx])
        b = float(base[idx])
        lo, hi = -1.0, 1.0
        if _final_for_residual(b, lo) > target or _final_for_residual(b, hi) < target:
            # Fallback for a non-monotone scale (<1).  The linear difference is
            # still a useful residual target, and the actor can correct it.
            out[idx] = target - b
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _final_for_residual(b, mid) < target:
                lo = mid
            else:
                hi = mid
        out[idx] = 0.5 * (lo + hi)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
