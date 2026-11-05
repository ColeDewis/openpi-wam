"""Encoder selection for RLPD / HG-Dagger on the WAM robot.

Three observation encoders are selectable through one config field:

    hilserl  -- the original hil-serl path: raw front/wrist frames plus proprio
                go through the agent's own ResNet-10 (``serl_launcher``).
    rltoken  -- the frozen VLA's RL token (``openpi.models.rl_token``) plus
                proprio, produced outside the agent (``WamRltokenEncoder``).
    timm     -- a frozen torch/timm vision encoder plus proprio
                (``WamTimmEncoder``).

For ``rltoken`` and ``timm`` the feature extractor runs outside the JAX agent,
so the agent sees a plain vector under the ``state`` key. For ``hilserl`` the
agent owns the encoder and sees the raw dict, exactly as upstream.
"""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import numpy as np

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.data.data_store import ReplayBufferDataStore
from serl_launcher.utils.launcher import make_agent

ENCODER_KINDS = ("hilserl", "rltoken", "timm")

# WAM proprio fed to every encoder/agent:
#   follower_jp(7) + discrete gripper position(1)
# This mirrors the simulator's compact state and the 8-dim WAM joint state.
WAM_PROPRIO_DIM = 8


@dataclasses.dataclass
class EncoderConfig:
    """Everything needed to rebuild one encoder choice on either side."""

    kind: str = "hilserl"

    # Camera keys, shared by all encoders (the images feed the VLA for rltoken).
    image_keys: tuple[str, str] = ("front_image", "wrist_image")

    # hilserl
    hilserl_encoder_type: str = "resnet-pretrained"

    # Observation shape. The WAM defaults match the tested simulator:
    # 8-dim proprio (joint positions + discrete gripper) and 128x128 frames.
    #
    # ``proprio_dim`` is the TOTAL state width. When a base policy is used,
    # ``base_action_dim`` values are appended to the raw WAM proprio (or are
    # already included by the LIBERO adapter). For WAM:
    #   proprio_dim == WAM_PROPRIO_DIM + base_action_dim
    proprio_dim: int = WAM_PROPRIO_DIM
    base_action_dim: int = 0
    image_size: tuple[int, int] = (128, 128)

    # rltoken
    rlt_config_name: str = "rlt_wam_pi05"
    rlt_checkpoint_dir: str = ""
    rlt_prompt: str = "reach for the green toy plushy"
    rlt_width: int = 0  # 0 = use the config's value

    # timm
    timm_model_name: str = "resnet18.a1_in1k"
    timm_pretrained: bool = True
    timm_feature_dim: int = 0  # 0 = query the model architecture

    def __post_init__(self):
        if self.kind not in ENCODER_KINDS:
            raise ValueError(f"encoder kind must be one of {ENCODER_KINDS}, got {self.kind!r}")
        # JSON round-trips tuples as lists; normalize so a checkpoint's
        # config.json rebuilds the same observation space.
        self.image_keys = tuple(self.image_keys)
        self.image_size = tuple(self.image_size)
        if self.proprio_dim <= 0:
            raise ValueError(f"proprio_dim must be positive, got {self.proprio_dim}")
        if self.base_action_dim < 0:
            raise ValueError(f"base_action_dim must be non-negative, got {self.base_action_dim}")
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must be (height, width), got {self.image_size}")


def rltoken_dim(cfg: EncoderConfig) -> int:
    import openpi.training.config as _config

    return int(_config.get_config(cfg.rlt_config_name).model.rlt_zrl_dim)


def timm_num_features(cfg: EncoderConfig) -> int:
    if cfg.timm_feature_dim:
        return int(cfg.timm_feature_dim)
    import timm

    # Architecture only; no weight download just to learn the output width.
    model = timm.create_model(cfg.timm_model_name, pretrained=False, num_classes=0)
    return int(model.num_features)


def observation_dim(cfg: EncoderConfig) -> int:
    """Width of the ``state`` vector the agent consumes.

    Only meaningful for ``rltoken`` / ``timm``; ``hilserl`` consumes the raw
    image dict and its agent encoder handles the rest.
    """
    if cfg.kind == "rltoken":
        return rltoken_dim(cfg) + cfg.proprio_dim
    if cfg.kind == "timm":
        return len(cfg.image_keys) * timm_num_features(cfg) + cfg.proprio_dim
    return cfg.proprio_dim


def observation_space(cfg: EncoderConfig, image_size: tuple[int, int] | None = None) -> gym.Space:
    image_size = cfg.image_size if image_size is None else tuple(image_size)
    if cfg.kind == "hilserl":
        # One stacked frame per camera (T=1), exactly the layout the upstream
        # MemoryEfficientReplayBuffer expects.
        return gym.spaces.Dict(
            {key: gym.spaces.Box(0, 255, (1, *image_size, 3), np.uint8) for key in cfg.image_keys}
            | {"state": gym.spaces.Box(-np.inf, np.inf, (1, cfg.proprio_dim), np.float32)}
        )
    return gym.spaces.Dict({"state": gym.spaces.Box(-np.inf, np.inf, (observation_dim(cfg),), np.float32)})


def sample_obs(cfg: EncoderConfig, image_size: tuple[int, int] | None = None) -> dict:
    image_size = cfg.image_size if image_size is None else tuple(image_size)
    if cfg.kind == "hilserl":
        obs = {key: np.zeros((1, *image_size, 3), np.uint8) for key in cfg.image_keys}
        obs["state"] = np.zeros((1, cfg.proprio_dim), np.float32)
        return obs
    return {"state": np.zeros((observation_dim(cfg),), np.float32)}


def action_space(action_dim: int = 8) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32)


def make_encoder_agent(
    algorithm: str,
    cfg: EncoderConfig,
    seed: int,
    action_dim: int = 8,
    *,
    reward_bias: float = 0.0,
    target_entropy: float | None = None,
    discount: float = 0.985,
    soft_target_update_rate: float = 0.005,
    target_intervention_rate: float = 0.05,
):
    """RLPD (``sac``) or HG-Dagger (``bc``) agent for the configured encoder.

    RLPD is hil-serl's hybrid agent: continuous SAC over the first ``action_dim - 1``
    dims plus a DQN "grasp critic" over the discrete gripper {-1 open, 0 hold, +1 close}.
    Every caller that asks for ``"sac"`` (learner, sim, ``HilPolicy``) therefore gets the
    same architecture, so checkpoints restore on both sides.
    """
    if algorithm == "sac":
        algorithm = "sac_hybrid"
    return make_agent(
        algorithm,
        seed,
        sample_obs(cfg),
        np.zeros((action_dim,), np.float32),
        encoder_kind=cfg.kind,
        image_keys=cfg.image_keys,
        encoder_type=cfg.hilserl_encoder_type,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        discount=discount,
        soft_target_update_rate=soft_target_update_rate,
        target_intervention_rate=target_intervention_rate,
    )


def make_encoder_buffer(cfg: EncoderConfig, capacity: int, action_dim: int = 8):
    space = observation_space(cfg)
    act_space = action_space(action_dim)
    if cfg.kind == "hilserl":
        return MemoryEfficientReplayBufferDataStore(
            space,
            act_space,
            capacity,
            image_keys=cfg.image_keys,
            include_grasp_penalty=True,
        )
    return ReplayBufferDataStore(
        space, act_space, capacity, include_grasp_penalty=True
    )
