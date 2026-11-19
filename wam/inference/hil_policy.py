"""Trained RLPD / HG-Dagger residual policy for the real WAM robot."""

from __future__ import annotations

import json
from pathlib import Path
import time

from flax.training import checkpoints
import jax
import numpy as np
from termcolor import cprint

from openpi.training.hil.encoders import EncoderConfig
from openpi.training.hil.encoders import make_encoder_agent
from openpi.training.hil.transitions import TransitionRecorder
from wam.inference.hil_encoder import discrete_gripper, make_encoder
from wam.inference.policy import PolicyTemplate

# RL actions are 7 delta joint positions + a discrete gripper. The agents work
# in [-1, 1], so each channel is scaled before being stored and after sampling.
ACTION_JP_SCALE = 0.05
ACTION_GRIPPER_SCALE = 1.0

# The controller reports a per-joint policy torque scale; a human takeover is
# any scale above this threshold.
INTERVENTION_POLICY_TORQUE_SCALE_THRESHOLD = 0.5

HIL_POLICY_RELOAD_INTERVAL_S = 30.0


class HilPolicy(PolicyTemplate):
    """Residual policy that records transitions and detects interventions."""

    def __init__(
        self,
        checkpoint_dir: str,
        data_dir: str,
        *,
        dof: int = 7,
        image_size: tuple[int, int] | None = None,
    ):
        self.checkpoint_dir = str(checkpoint_dir)
        config_path = Path(checkpoint_dir) / "config.json"
        config = json.loads(config_path.read_text())

        self.algorithm = config["algorithm"]
        self.action_dim = int(config["action_dim"])
        self.use_gripper = bool(config.get("use_gripper", True))
        self.agent_kind = config.get("agent")
        if self.agent_kind is None:
            self.agent_kind = (
                "bc"
                if self.algorithm == "hgdagger"
                else ("sac_hybrid_single" if self.use_gripper else "sac")
            )
        default_cost = (
            self.agent_kind == "sac_hybrid_single"
            and "target_intervention_rate" in config
        )
        self.use_intervention_cost = bool(
            config.get("use_intervention_cost", default_cost)
        )

        encoder_config = dict(config["encoder"])
        encoder_config["image_keys"] = tuple(
            encoder_config.get("image_keys", ("front_image", "wrist_image"))
        )
        self.encoder_config = EncoderConfig(**encoder_config)

        self.agent = make_encoder_agent(
            self.agent_kind,
            self.encoder_config,
            seed=0,
            action_dim=self.action_dim,
            use_gripper=self.use_gripper,
            use_intervention_cost=self.use_intervention_cost,
        )
        self.agent = self.reload(self.agent)

        self.image_size = tuple(
            self.encoder_config.image_size if image_size is None else image_size
        )
        self.encoder = make_encoder(
            self.encoder_config,
            self.image_size,
            use_gripper=self.use_gripper,
        )

        joint_dim = self.action_dim - int(self.use_gripper)
        self.action_scale = np.array(
            [ACTION_JP_SCALE] * joint_dim
            + [ACTION_GRIPPER_SCALE] * int(self.use_gripper),
            np.float32,
        )
        self.action_size = self.action_dim
        self.needs_images = True

        self.transition_recorder = TransitionRecorder(data_dir, self.algorithm)
        self.dof = dof
        self._argmax = self.agent_kind == "bc"
        self._rng = jax.random.PRNGKey(0)
        self._intervened = False
        self._last_obs_enc: dict | None = None
        self._last_action_norm: np.ndarray | None = None
        self._last_reload = time.time()

    def reload(self, agent=None):
        agent = self.agent if agent is None else agent
        latest = checkpoints.latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(f"no flax checkpoint under {self.checkpoint_dir}")
        state = checkpoints.restore_checkpoint(latest, target=agent.state)
        return agent.replace(state=state)

    def start(self) -> None:
        self.transition_recorder.reset()
        self._intervened = False
        self._last_obs_enc = None
        self._last_action_norm = None

    def pre_infer(self, obs: dict) -> None:
        self._maybe_reload()
        self._intervened = self._detect_intervention(obs["low_dim"])

    def infer(self, obs: dict) -> np.ndarray:
        observation = self.encoder.encode(obs)
        self._rng, key = jax.random.split(self._rng)
        action = np.asarray(
            self.agent.sample_actions(
                observation, seed=key, argmax=self._argmax
            ),
            np.float32,
        ).clip(-1.0, 1.0)

        self._last_obs_enc = observation
        self._last_action_norm = action
        return (action * self.action_scale)[None, :]

    def post_infer(self, obs: dict) -> None:
        if self._intervened:
            action_norm = self._intervention_action_norm(obs["low_dim"])
        else:
            action_norm = self._last_action_norm

        self.transition_recorder.add_step(
            self._last_obs_enc,
            action_norm,
            intervened=self._intervened,
        )

    def finish(self, success: bool) -> None:
        intervened_steps, total_steps = self.transition_recorder.stats()
        self.transition_recorder.flush(success=success)
        rate = intervened_steps / max(total_steps, 1)
        cprint(
            f"[RECORDER] intervention rate: {rate:.1%} "
            f"({intervened_steps}/{total_steps} steps)",
            "cyan",
        )

    def _detect_intervention(self, state_dict: dict) -> bool:
        """Whether the policy torque scale indicates a human takeover."""
        scale = np.asarray(
            state_dict.get("policy_torque_scale", np.zeros(self.dof)),
            np.float32,
        )
        return bool(
            np.max(np.abs(scale)) > INTERVENTION_POLICY_TORQUE_SCALE_THRESHOLD
        )

    def _intervention_action_norm(self, state_dict: dict) -> np.ndarray:
        """Residual the human is applying: leader joint positions minus the base."""
        action = np.asarray(state_dict["leader_jp"], np.float32) - np.asarray(
            state_dict["policy_jp"], np.float32
        )
        if self.use_gripper:
            gripper = discrete_gripper(float(state_dict.get("gripper_pos", 0.0)))
            action = np.concatenate([action, [gripper]]).astype(np.float32)
        return np.clip(action / self.action_scale, -1.0, 1.0)

    def _maybe_reload(self) -> None:
        """Pick up a newer learner checkpoint without restarting the actor."""
        if time.time() - self._last_reload < HIL_POLICY_RELOAD_INTERVAL_S:
            return
        self._last_reload = time.time()
        try:
            self.agent = self.reload()
            cprint("[HIL] policy checkpoint reloaded", "green")
        except Exception as e:
            cprint(f"[HIL] policy reload skipped: {e}", "yellow")
