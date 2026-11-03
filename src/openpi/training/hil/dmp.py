"""DMP base policy for residual RL.

The dynamics come from ``movement_primitives``.  This module only adapts the
library's absolute-pose rollout to LIBERO's normalized delta-pose action space
and provides the residual composition used by the actor.
"""

from __future__ import annotations

from movement_primitives.dmp import DMP
import numpy as np


class DmpPolicy:
    """Fit a DMP to an end-effector trajectory and emit normalized deltas."""

    def __init__(
        self,
        trajectory: np.ndarray,
        *,
        dt: float,
        rotation: bool = False,
        gripper: bool = False,
        action_dim: int = 4,
        max_step_m: float = 0.02,
        max_step_rad: float = 0.05,
        n_weights_per_dim: int = 20,
        regularization_coefficient: float = 0.0,
    ):
        trajectory = np.asarray(trajectory, np.float64)
        self.rotation = bool(rotation)
        self.gripper = bool(gripper)
        self.action_dim = int(action_dim)
        self.max_step_m = float(max_step_m)
        self.max_step_rad = float(max_step_rad)
        self.goal = trajectory[-1].copy()

        execution_time = (len(trajectory) - 1) * float(dt)
        self.dmp = DMP(
            n_dims=trajectory.shape[1],
            execution_time=execution_time,
            dt=float(dt),
            n_weights_per_dim=int(n_weights_per_dim),
        )
        time = np.linspace(0.0, execution_time, len(trajectory))
        self.dmp.imitate(time, trajectory, regularization_coefficient=float(regularization_coefficient))

        self._rollout: np.ndarray | None = None
        self._index = 0

    def reset(self, start_pose: np.ndarray) -> None:
        """Re-anchor the DMP at the current pose and generate its rollout."""
        start_pose = np.asarray(start_pose, np.float64).reshape(-1)
        self.dmp.reset()
        self.dmp.configure(start_y=start_pose, goal_y=self.goal)
        _, self._rollout = self.dmp.open_loop()
        self._index = 1  # rollout[0] is the current pose

    def action(self, current_pose: np.ndarray) -> np.ndarray:
        """Return the normalized delta-pose action for the current step."""
        if self._rollout is None:
            raise RuntimeError("DmpPolicy.reset() must be called before action()")

        current_pose = np.asarray(current_pose, np.float64).reshape(-1)
        target = self._rollout[self._index] if self._index < len(self._rollout) else self.goal
        self._index += 1

        delta = target - current_pose
        action = np.zeros(self.action_dim, np.float32)
        action[:3] = delta[:3] / self.max_step_m
        if self.rotation:
            action[3:6] = delta[3:6] / self.max_step_rad
        if self.gripper:
            action[-1] = target[-1]
        return np.clip(action, -1.0, 1.0)

    @property
    def rollout(self) -> np.ndarray | None:
        return self._rollout


def residual_target(final: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Residual that would move ``base`` toward the demonstrated action."""
    return np.clip(np.asarray(final, np.float32) - np.asarray(base, np.float32), -1.0, 1.0)


def combine_base_residual(
    base: np.ndarray,
    residual: np.ndarray,
    *,
    max_residual: float = 1.0,
) -> np.ndarray:
    """Blend base and residual by how close the residual is to its maximum.

    ``w = sigmoid(10 * (|residual| / max_residual - 0.5))`` is applied per
    dimension, then ``final = (1 - w) * base + w * residual``.
    """
    base = np.asarray(base, np.float32)
    residual = np.asarray(residual, np.float32)
    closeness = np.clip(np.abs(residual) / float(max_residual), 0.0, 1.0)
    weight = 1.0 / (1.0 + np.exp(-10.0 * (closeness - 0.5)))
    return np.clip((1.0 - weight) * base + weight * residual, -1.0, 1.0)
