"""Load a trained RLPD / HG-Dagger agent and expose the same ``infer`` surface
as ``OpenPIPolicy`` so ``wam/inference/main.py`` can drive the arm with either.

The checkpoint is a standard flax ``save_checkpoint`` directory plus a
``config.json`` written by the learner (algorithm, action dim, encoder choice).
The same ``EncoderConfig`` rebuilds whichever frozen feature extractor the run
used, so deployment cannot silently pair a policy with the wrong observations.
"""

from __future__ import annotations

import json
from pathlib import Path

from flax.training import checkpoints
import jax
import numpy as np

from openpi.training.hil.encoders import EncoderConfig
from openpi.training.hil.encoders import make_encoder_agent
from wam.inference.hil_encoder import make_encoder


class HilPolicy:
    def __init__(
        self,
        checkpoint_dir: str,
        action_scale: np.ndarray,
        image_size: tuple[int, int] = (224, 224),
    ):
        self.checkpoint_dir = str(checkpoint_dir)
        config_path = Path(checkpoint_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} not found — {checkpoint_dir} is not an RLPD/HG-Dagger run dir")
        config = json.loads(config_path.read_text())
        self.algorithm = config["algorithm"]  # "rlpd" | "hgdagger"
        if self.algorithm == "rlpd" and config.get("target_intervention_rate") is None:
            raise ValueError(
                f"{config_path} predates the intervention-cost networks; "
                "retrain with a fresh checkpoint directory"
            )
        self.action_dim = int(config["action_dim"])
        self.base_policy = config.get("base_policy", "none")
        self.base_action_dim = int(config.get("base_action_dim", 0))
        self.action_scale = np.asarray(action_scale, np.float32)
        encoder_config = dict(config["encoder"])
        encoder_config["image_keys"] = tuple(encoder_config.get("image_keys", ("front_image", "wrist_image")))
        self.encoder_config = EncoderConfig(**encoder_config)

        self.agent = make_encoder_agent(
            "bc" if self.algorithm == "hgdagger" else "sac",
            self.encoder_config,
            seed=0,
            action_dim=self.action_dim,
        )
        self.agent = self.reload(self.agent)
        self.encoder = make_encoder(self.encoder_config, image_size)
        # BC is deterministic at deployment (mode); SAC samples, matching the
        # upstream actor loop (``argmax=False``).
        self._argmax = self.algorithm == "hgdagger"
        self._rng = jax.random.PRNGKey(0)

    def reload(self, agent=None):
        agent = self.agent if agent is None else agent
        latest = checkpoints.latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(f"no flax checkpoint under {self.checkpoint_dir}")
        state = checkpoints.restore_checkpoint(latest, target=agent.state)
        return agent.replace(state=state)

    def encode(self, raw_obs: dict) -> dict:
        return self.encoder.encode(raw_obs)

    def act(self, raw_obs: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        """Returns ``(wrench_action, normalized_action, encoded_observation)``."""
        observation = self.encode(raw_obs)
        self._rng, key = jax.random.split(self._rng)
        action = np.asarray(
            self.agent.sample_actions(observation, seed=key, argmax=self._argmax),
            np.float32,
        ).clip(-1.0, 1.0)
        return action * self.action_scale, action, observation

    def infer(self, raw_obs: dict) -> np.ndarray:
        """OpenPIPolicy-compatible entry point: the 7-dim wrench chunk step."""
        return self.act(raw_obs)[0]
