"""Map the agent's normalized action to the environment's action space.

The SAC/BC/HG-Dagger agents operate on ``Box(-1, 1, (action_dim,))``; LIBERO
expects a 7-dim ``[dx, dy, dz, dax, day, daz, gripper]`` command. What those
numbers *mean* is robot-specific:

    delta_pose   LIBERO / VLA convention: [dx, dy, dz, dax, day, daz, gripper].
                 LIBERO's OSC controller already applies its own metric scale,
                 so the adapter is the identity here.
    wrench       WAM convention: [fx, fy, fz, tx, ty, tz, gripper] in N / N*m.
                 The adapter multiplies by the force/torque limits.
    joint_torque reserved for the next expansion: same normalized->metric
                 scaling, with per-joint torque limits.

Keeping this in one small dataclass means the simulator, the real WAM actor,
and any future torque controller share the same transition storage: transitions
always store the normalized action, and only the boundary adapter knows the
robot units.
"""

from __future__ import annotations

import dataclasses

import numpy as np

ACTION_KINDS = ("delta_pose", "wrench", "joint_torque")


@dataclasses.dataclass(frozen=True)
class ActionAdapter:
    """Convert between normalized agent actions and environment actions."""

    kind: str = "delta_pose"
    action_dim: int = 7
    # Per-dimension metric scale. delta_pose -> 1s (the env scales internally);
    # wrench -> (5, 5, 5, 5, 5, 5, 1); joint_torque -> the joint torque limits.
    scales: tuple[float, ...] = (1.0,) * 7
    clip: float = 1.0

    def __post_init__(self):
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"action kind must be one of {ACTION_KINDS}, got {self.kind!r}")
        if len(self.scales) != self.action_dim:
            raise ValueError(f"expected {self.action_dim} scales, got {len(self.scales)}")

    @property
    def scale_vector(self) -> np.ndarray:
        return np.asarray(self.scales, np.float32)

    def to_env(self, action_norm: np.ndarray) -> np.ndarray:
        """Normalized [-1, 1] action -> the action the environment expects."""
        action = np.asarray(action_norm, np.float32).reshape(self.action_dim)
        action = np.clip(action, -self.clip, self.clip)
        # All current kinds are a per-dimension scale. A future controller with
        # a non-diagonal mapping (e.g. a Jacobian) belongs here.
        return (action * self.scale_vector).astype(np.float32)

    def from_env(self, action_env: np.ndarray) -> np.ndarray:
        """Environment action -> the normalized action stored in the buffer."""
        action = np.asarray(action_env, np.float32).reshape(self.action_dim)
        return np.clip(action / (self.scale_vector + 1e-8), -self.clip, self.clip).astype(np.float32)
