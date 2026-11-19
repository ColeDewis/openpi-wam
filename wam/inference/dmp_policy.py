"""DMP base policy for the real WAM robot.

The trajectory is loaded from an HDF5 episode written by ``HDF5Recorder``.
The DMP emits absolute joint positions; the residual policy and controller
handle the correction on top of that target.
"""

from __future__ import annotations

import h5py
import numpy as np
from movement_primitives.dmp import DMP

from wam.inference.policy import PolicyTemplate


class DMPPolicy(PolicyTemplate):
    """Fit a DMP to a recorded trajectory and stream its open-loop rollout."""

    def __init__(
        self,
        trajectory_path: str,
        *,
        dt: float,
        dof: int = 7,
        use_gripper: bool = False,
        horizon: int = 8,
        n_weights_per_dim: int = 20,
        inf_buffer: float = 0.1,
    ):
        self.dt = dt
        trajectory = self._load_trajectory(trajectory_path, dof, use_gripper, self.dt)


        execution_time = (len(trajectory) - 1) * self.dt
        self.dmp = DMP(
            n_dims=trajectory.shape[1],
            dt=self.dt,
            n_weights_per_dim=n_weights_per_dim,
        )
        time = np.linspace(0.0, execution_time, len(trajectory))
        self.dmp.imitate(time, trajectory)
        self.dmp.set_execution_time_(execution_time)
        self.goal = trajectory[-1].copy()

        self.dof = dof
        self.use_gripper = use_gripper
        self.horizon = horizon
        self.action_size = dof + int(use_gripper)
        self.needs_images = False
        self.image_size = None
        self.inf_buffer_ns = int(inf_buffer * 1e9)

        self._rollout: np.ndarray | None = None
        self._index = 0
        self._sent = False

    @staticmethod
    def _load_trajectory(path: str, dof: int, use_gripper: bool, dt: float) -> np.ndarray:
        with h5py.File(path, "r") as f:
            joint_positions = np.asarray(f["low_dim/leader_jp"], np.float64)
            if joint_positions.shape[1] != dof:
                raise ValueError(
                    f"trajectory has {joint_positions.shape[1]} joints, expected {dof}"
                )
            if use_gripper:
                gripper = np.asarray(f["low_dim/gripper_pos"], np.float64).reshape(-1, 1)
                trajectory = np.concatenate([joint_positions, gripper], axis=1)
            else:
                trajectory = joint_positions

        # Downsample from 500 Hz to the DMP execution frequency.
        stride = int(round(500 * dt))
        idx = np.arange(0, len(trajectory), stride)
        trajectory = trajectory[idx]

        return trajectory

    def start(self) -> None:
        self._rollout = None
        self._index = 0
        self._sent = False

    def pre_infer(self, obs: dict) -> None:
        if self._rollout is not None:
            return

        low_dim = obs["low_dim"]
        start_y = list(np.asarray(low_dim["leader_jp"], np.float64))
        start_yd = list(np.asarray(low_dim.get("leader_jv", np.zeros(self.dof)), np.float64))
        if self.use_gripper:
            start_y.append(float(low_dim.get("gripper_pos", 0.0)))
            start_yd.append(float(low_dim.get("gripper_vel", 0.0)))

        self.dmp.reset()
        self.dmp.configure(
            start_y=np.asarray(start_y),
            start_yd=np.asarray(start_yd),
            goal_y=self.goal,
        )
        _, self._rollout = self.dmp.open_loop()
        self._index = 0

    def infer(self, obs: dict) -> np.ndarray | None:
        if self._rollout is None or self._index >= len(self._rollout):
            return None

        time_to_chunk_end_ns = int(
            obs["low_dim"].get("time_to_chunk_end_ns", 0)
        )
        if time_to_chunk_end_ns > self.inf_buffer_ns:
            return None

        end = min(self._index + self.horizon, len(self._rollout))
        chunk = self._rollout[self._index:end].copy()
        if len(chunk) < self.horizon:
            pad = np.tile(chunk[-1], (self.horizon - len(chunk), 1))
            chunk = np.vstack([chunk, pad])

        self._sent = True
        return chunk.astype(np.float32)

    def post_infer(self, obs: dict) -> None:
        if self._sent:
            self._index += self.horizon
            self._sent = False
