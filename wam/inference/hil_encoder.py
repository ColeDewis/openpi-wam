"""WAM observation encoders for the three selectable feature spaces.

One object per encoder kind, all exposing ``encode(raw_obs) -> observation``
where ``observation`` is exactly what the SAC/BC agent expects:

    WamHilserlEncoder  -> {"front_image": (1,H,W,C), "wrist_image": (...),
                           "state": (1, PROPRIO)}
    WamRltokenEncoder  -> {"state": (z_rl + PROPRIO,)}
    WamTimmEncoder     -> {"state": (timm_feat + FORCE + PROPRIO,)}

The first one owns no network: its images go straight into the agent's
ResNet-10, exactly as in hil-serl. The other two are frozen feature extractors
that run outside the JAX agent (the design decision that keeps a single JAX
training loop for all three options).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from openpi.training.hil.encoders import WAM_FORCE_DIM
from openpi.training.hil.encoders import WAM_PROPRIO_DIM
from openpi.training.hil.encoders import EncoderConfig
from wam.inference.openpi_policy import WAM_TORQUE_LIMIT
from wam.inference.openpi_policy import quat_to_rotvec

logger = logging.getLogger(__name__)

# Keep in sync with the pi05 data pipeline's 8-dim state (see OpenPIPolicy).
VLA_GRIPPER_STATE = np.array([0.015, -0.015], np.float32)


def vla_state(low_dim: dict) -> np.ndarray:
    """8-dim pi0 state: cart_pos + rotvec + the two fixed gripper channels."""
    return np.concatenate(
        [
            np.asarray(low_dim["follower_cart_pos"], np.float32),
            quat_to_rotvec(np.asarray(low_dim["follower_cart_rot"], np.float32)),
            VLA_GRIPPER_STATE,
        ]
    ).astype(np.float32)


def wam_proprio(low_dim: dict, base_action: np.ndarray | None = None) -> np.ndarray:
    """16-dim RL proprio shared by all three WAM encoders.

    When a base policy is active, its next action (``policy_jp``) is appended,
    making the state ``WAM_PROPRIO_DIM + base_action_dim`` wide.
    """
    gripper = float(low_dim.get("gripper_pos", 0.0))
    jv = np.asarray(low_dim.get("follower_jv", np.zeros(7)), np.float32)
    gv = np.asarray([low_dim.get("gripper_vel", 0.0)], np.float32)
    parts = [
        np.asarray(low_dim["follower_cart_pos"], np.float32),
        quat_to_rotvec(np.asarray(low_dim["follower_cart_rot"], np.float32)),
        np.asarray([gripper, -gripper], np.float32),
        jv,
        gv,
    ]
    if base_action is not None:
        parts.append(np.asarray(base_action, np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def _base_action_from_low_dim(cfg: EncoderConfig, low_dim: dict) -> np.ndarray | None:
    """Read the base policy's delta joint action from the extended UDP packet."""
    if cfg.base_action_dim <= 0:
        return None
    if "policy_jp" not in low_dim:
        raise ValueError(
            "encoder.base_action_dim > 0 requires a 'policy_jp' field; "
            "use the extended recorder packet, not the legacy one."
        )
    action = np.asarray(low_dim["policy_jp"], np.float32).reshape(-1)
    if action.shape[0] != cfg.base_action_dim:
        raise ValueError(
            f"policy_jp has {action.shape[0]} dims, expected encoder.base_action_dim={cfg.base_action_dim}"
        )
    return action


def wam_force(low_dim: dict) -> np.ndarray:
    """7-dim measured external torque, used only by the timm encoder."""
    force = np.asarray(low_dim.get("follower_ext_torque", np.zeros(WAM_FORCE_DIM)), np.float32)
    return force / float(WAM_TORQUE_LIMIT)


def _require_wam_proprio(cfg: EncoderConfig) -> None:
    """The WAM encoders emit ``WAM_PROPRIO_DIM + base_action_dim`` features.

    The LIBERO simulator has a different proprio (8/17 dims) and its own
    adapter (``openpi.training.hil.libero``); catching the mismatch here keeps
    a sim checkpoint from being deployed against WAM packets by accident.
    """
    expected = WAM_PROPRIO_DIM + cfg.base_action_dim
    if cfg.proprio_dim != expected:
        raise ValueError(
            f"WAM encoders produce {expected}-dim proprio "
            f"({WAM_PROPRIO_DIM} raw + {cfg.base_action_dim} base action), "
            f"got encoder.proprio_dim={cfg.proprio_dim}. "
            "Use the LIBERO adapter for simulator checkpoints."
        )


class WamHilserlEncoder:
    """Raw images + proprio for the agent's own ResNet-10."""

    def __init__(self, cfg: EncoderConfig, image_size: tuple[int, int] = (224, 224)):
        _require_wam_proprio(cfg)
        self.cfg = cfg
        self.image_size = image_size

    def encode(self, raw_obs: dict) -> dict:
        obs = {
            key: np.asarray(raw_obs[key], np.uint8)[None]  # (T=1, H, W, C)
            for key in self.cfg.image_keys
        }
        base_action = _base_action_from_low_dim(self.cfg, raw_obs["low_dim"])
        obs["state"] = wam_proprio(raw_obs["low_dim"], base_action)[None]  # (T=1, D)
        return obs


class WamRltokenEncoder:
    """Frozen VLA prefix -> RL token, concatenated with proprio.

    This is the same computation ``scripts/train_rlt.py`` uses to build the
    stage-2 state (``rlt/vla.py``'s ``_prefix_cache`` + ``RLTokenizer.encode``);
    the action denoising loop is skipped because only ``z_rl`` is needed here.
    """

    def __init__(self, cfg: EncoderConfig, image_size: tuple[int, int] = (224, 224)):
        _require_wam_proprio(cfg)
        if not cfg.rlt_checkpoint_dir:
            raise ValueError("encoder.kind='rltoken' needs --encoder.rlt_checkpoint_dir")

        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        import openpi.models.model as _model
        from openpi.models.tokenizer import PaligemmaTokenizer
        import openpi.shared.nnx_utils as nnx_utils
        from openpi.training.rlt.envs import load_pi0_model
        from openpi.training.rlt.vla import _prefix_cache

        self._jnp = jnp
        self._model = _model
        self.cfg = cfg

        graphdef, state, model_config, train_cfg = load_pi0_model(
            cfg.rlt_checkpoint_dir,
            cfg.rlt_config_name,
            rlt_width=(cfg.rlt_width or None),
        )
        if not getattr(model_config, "rlt_enabled", False):
            raise ValueError(f"config {cfg.rlt_config_name} has no RLT module")
        # The tokenizer was trained in f32; a bf16 restore rounds it.
        rlt_state, rest_state = state.split(nnx_utils.PathRegex(".*rlt.*"), ...)
        rlt_state = jax.tree.map(lambda x: x.astype(jnp.float32), rlt_state)
        state = nnx.State.merge(rlt_state, rest_state)

        self._state = state
        self._num_image_tokens = model_config.rlt_num_input_cameras * 256
        self._tok = PaligemmaTokenizer(max_len=model_config.max_token_len)
        self._prompt, self._prompt_mask = self._tok.tokenize(cfg.rlt_prompt)
        self._state_norm = _load_state_norm(train_cfg, model_config, cfg.rlt_checkpoint_dir)

        @jax.jit
        def _encode(state, observation):
            m = nnx.merge(graphdef, state)
            observation = _model.preprocess_observation(None, observation, train=False)
            prefix_out, _, _ = _prefix_cache(m, observation)
            return m.rlt.encode(prefix_out[:, : self._num_image_tokens].astype(jnp.float32))

        self._encode_impl = _encode
        logger.info(
            "RL-token encoder ready: config=%s checkpoint=%s zrl_dim=%d",
            cfg.rlt_config_name,
            cfg.rlt_checkpoint_dir,
            model_config.rlt_zrl_dim,
        )

    def encode(self, raw_obs: dict) -> dict:
        jnp = self._jnp
        front = np.asarray(raw_obs[self.cfg.image_keys[0]], np.float32) / 255.0 * 2.0 - 1.0
        wrist = np.asarray(raw_obs[self.cfg.image_keys[1]], np.float32) / 255.0 * 2.0 - 1.0
        batch = 1
        images = {
            "base_0_rgb": jnp.asarray(front)[None],
            "left_wrist_0_rgb": jnp.asarray(wrist)[None],
            "right_wrist_0_rgb": jnp.zeros_like(jnp.asarray(front))[None],
        }
        masks = {
            "base_0_rgb": jnp.ones(batch, bool),
            "left_wrist_0_rgb": jnp.ones(batch, bool),
            "right_wrist_0_rgb": jnp.zeros(batch, bool),
        }
        state = self._normalized_vla_state(raw_obs["low_dim"])
        observation = self._model.Observation(
            images=images,
            image_masks=masks,
            state=jnp.asarray(state, jnp.float32)[None],
            tokenized_prompt=jnp.asarray(self._prompt, jnp.int32)[None],
            tokenized_prompt_mask=jnp.asarray(self._prompt_mask)[None],
        )
        z_rl = np.asarray(self._encode_impl(self._state, observation))[0]
        base_action = _base_action_from_low_dim(self.cfg, raw_obs["low_dim"])
        return {"state": np.concatenate([z_rl, wam_proprio(raw_obs["low_dim"], base_action)]).astype(np.float32)}

    def _normalized_vla_state(self, low_dim: dict) -> np.ndarray:
        state = vla_state(low_dim)
        if self._state_norm is not None:
            kind, p1, p2 = self._state_norm
            n = len(p1)
            if kind == "quantile":
                state[:n] = (state[:n] - p1) / (p2 - p1 + 1e-6) * 2.0 - 1.0
            else:
                state[:n] = (state[:n] - p1) / (p2 + 1e-6)
        return state


class WamTimmEncoder:
    """Frozen torch/timm vision encoder + force/proprio.

    Close to cr-dagger's ``TimmObsEncoderWithForce``: one timm model per RGB
    key, the measured force vector, and proprio fused by concatenation. It is
    deliberately frozen (no torch backward path), so the JAX SAC/BC agents see
    a fixed feature vector. ``fuse_mode='concat'`` is the cr-dagger default.
    """

    def __init__(self, cfg: EncoderConfig, image_size: tuple[int, int] = (224, 224)):
        _require_wam_proprio(cfg)
        import timm
        import torch

        self.cfg = cfg
        self.image_size = image_size
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._model = timm.create_model(cfg.timm_model_name, pretrained=cfg.timm_pretrained, num_classes=0)
        except Exception as e:  # offline box / weight download failure
            logger.warning("could not load pretrained timm weights (%s); falling back to random init", e)
            self._model = timm.create_model(cfg.timm_model_name, pretrained=False, num_classes=0)
        self._model.eval().to(self._device)
        for param in self._model.parameters():
            param.requires_grad = False
        self.feature_dim = int(self._model.num_features)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self._device).view(1, 3, 1, 1)
        logger.info("timm encoder ready: %s (%d features)", cfg.timm_model_name, self.feature_dim)

    def _features(self, image: np.ndarray):
        torch = self._torch
        x = torch.as_tensor(np.ascontiguousarray(image), dtype=torch.float32, device=self._device)
        x = x.permute(2, 0, 1)[None] / 255.0
        x = (x - self._mean) / self._std
        with torch.no_grad():
            return self._model(x)[0].cpu().numpy().astype(np.float32)

    def encode(self, raw_obs: dict) -> dict:
        parts = [self._features(np.asarray(raw_obs[key])) for key in self.cfg.image_keys]
        parts.append(wam_force(raw_obs["low_dim"]))
        base_action = _base_action_from_low_dim(self.cfg, raw_obs["low_dim"])
        parts.append(wam_proprio(raw_obs["low_dim"], base_action))
        return {"state": np.concatenate(parts).astype(np.float32)}


def make_encoder(cfg: EncoderConfig, image_size: tuple[int, int] = (224, 224)):
    if cfg.kind == "hilserl":
        return WamHilserlEncoder(cfg, image_size)
    if cfg.kind == "rltoken":
        return WamRltokenEncoder(cfg, image_size)
    if cfg.kind == "timm":
        return WamTimmEncoder(cfg, image_size)
    raise ValueError(f"unknown encoder kind {cfg.kind!r}")


# ── norm stats for the frozen VLA ────────────────────────────────────────────
# Mirrors scripts/train_rlt.py's resolution order so the RL-token encoder sees
# the same state normalization the stage-2 policy does.


def _build_norm(norm_stats, *, use_quantile: bool, key: str):
    if norm_stats is None or key not in norm_stats:
        logger.warning("no %s norm stats found; RL-token state is unnormalized", key)
        return None
    s = norm_stats[key]
    if use_quantile:
        return ("quantile", np.array(s.q01, np.float32), np.array(s.q99, np.float32))
    return ("zscore", np.array(s.mean, np.float32), np.array(s.std, np.float32))


def _load_state_norm(train_cfg, model_config, checkpoint_dir: str):
    import openpi.training.checkpoints as _checkpoints

    data_config = train_cfg.data.create(train_cfg.assets_dirs, model_config)
    if data_config.asset_id is None:
        return None
    ckpt = Path(checkpoint_dir)
    candidates = [ckpt / "assets", ckpt.parent / "assets"]
    if ckpt.name.isdigit() or ckpt.name.startswith("step_"):
        candidates.append(ckpt.parent / "assets")
    elif ckpt.is_dir():
        steps = sorted(int(p.name) for p in ckpt.iterdir() if p.name.isdigit())
        if steps:
            candidates.append(ckpt / str(steps[-1]) / "assets")
    for assets_dir in candidates:
        try:
            if (assets_dir / data_config.asset_id).exists():
                stats = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)
                if stats is not None:
                    return _build_norm(stats, use_quantile=data_config.use_quantile_norm, key="state")
        except Exception as e:
            logger.warning("norm-stats load from %s failed: %s", assets_dir, e)
    if data_config.norm_stats is not None:
        return _build_norm(data_config.norm_stats, use_quantile=data_config.use_quantile_norm, key="state")
    return None
