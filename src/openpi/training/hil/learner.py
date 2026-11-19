"""Shared learner loop for RLPD (SAC) and HG-Dagger (BC).

The actor writes transition pickles under ``data_dir`` (``online/`` and
``demo/``); this loop tails them, trains, and checkpoints.

The agent is selected explicitly with ``agent``:

    "bc"               -- HG-Dagger behaviour cloning
    "sac"              -- continuous SAC (with or without a gripper dim)
    "sac_hybrid_single"-- SAC plus a DQN grasp critic for a discrete gripper
"""

from __future__ import annotations

import collections
import dataclasses
import json
from pathlib import Path
import threading
import time

from flax.training import checkpoints
import numpy as np
import tqdm

from openpi.training.hil import encoders as _encoders
from openpi.training.hil import transitions as _transitions
from serl_launcher.utils.train_utils import concat_batches

AGENT_KINDS = ("bc", "sac", "sac_hybrid_single")


def update_sets(agent: str, use_intervention_cost: bool) -> tuple[frozenset[str], frozenset[str]]:
    """Networks updated on critic-only steps and on full CTA steps."""
    critic_only = {"critic"}
    if agent == "sac_hybrid_single":
        critic_only.add("grasp_critic")
    if use_intervention_cost:
        critic_only.add("cost_critic")

    full_update = critic_only | {"actor", "temperature"}
    if use_intervention_cost:
        full_update.add("intervention")
    return frozenset(critic_only), frozenset(full_update)


@dataclasses.dataclass
class LearnerConfig:
    data_dir: str = ""
    """Directory written by ``wam/inference/main.py`` (contains online/ and demo/)."""

    checkpoint_dir: str = ""
    """Where agents are saved. Also read back by ``wam/inference``."""

    encoder: _encoders.EncoderConfig = dataclasses.field(default_factory=_encoders.EncoderConfig)

    agent: str = "sac"
    seed: int = 42

    use_gripper: bool = False
    use_intervention_cost: bool = False

    # Base policy metadata. ``encoder.proprio_dim`` includes these values.
    base_policy: str = "none"
    base_action_dim: int = 0

    # Defaults mirror the tested LIBERO trainer.
    batch_size: int = 256
    cta_ratio: int = 5
    discount: float = 0.985
    soft_target_update_rate: float = 0.005
    reward_bias: float = 0.0
    target_entropy: float | None = None
    target_intervention_rate: float = 0.15
    intervention_window: int = 20

    training_starts: int = 20
    max_steps: int = 100_000
    replay_buffer_capacity: int = 100_000
    pretrain_steps: int = 2_000
    poll_interval: float = 5.0

    log_interval: int = 100
    checkpoint_period: int = 1_000
    keep: int = 20

    @property
    def action_dim(self) -> int:
        return 7 + int(self.use_gripper)

    @property
    def proprio_dim(self) -> int:
        return self.action_dim + self.base_action_dim

    @property
    def algorithm(self) -> str:
        return "hgdagger" if self.agent == "bc" else "rlpd"

    @property
    def is_bc(self) -> bool:
        return self.agent == "bc"

    @property
    def uses_intervention_cost(self) -> bool:
        return self.agent == "sac_hybrid_single" and self.use_intervention_cost

    @property
    def critic_only(self) -> frozenset[str]:
        return update_sets(self.agent, self.uses_intervention_cost)[0]

    @property
    def full_update(self) -> frozenset[str]:
        return update_sets(self.agent, self.uses_intervention_cost)[1]

    def __post_init__(self):
        if not self.data_dir:
            raise ValueError("--data-dir is required (it is written by wam/inference/main.py)")
        if not self.checkpoint_dir:
            raise ValueError("--checkpoint-dir is required")
        if self.agent not in AGENT_KINDS:
            raise ValueError(f"agent must be one of {AGENT_KINDS}, got {self.agent!r}")
        if self.agent == "sac_hybrid_single" and not self.use_gripper:
            raise ValueError("agent='sac_hybrid_single' requires use_gripper=True")
        if self.base_action_dim < 0:
            raise ValueError(f"base_action_dim must be non-negative, got {self.base_action_dim}")

        # One source of truth: widths come from the agent/gripper/base layout.
        self.encoder.base_action_dim = self.base_action_dim
        self.encoder.proprio_dim = self.proprio_dim

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def _agent_from_config(cfg: LearnerConfig):
    return _encoders.make_encoder_agent(
        cfg.agent,
        cfg.encoder,
        cfg.seed,
        cfg.action_dim,
        use_gripper=cfg.use_gripper,
        use_intervention_cost=cfg.uses_intervention_cost,
        reward_bias=cfg.reward_bias,
        target_entropy=cfg.target_entropy,
        discount=cfg.discount,
        soft_target_update_rate=cfg.soft_target_update_rate,
        target_intervention_rate=cfg.target_intervention_rate,
    )


def save_run_config(cfg: LearnerConfig) -> None:
    payload = {
        "agent": cfg.agent,
        "algorithm": cfg.algorithm,
        "action_dim": cfg.action_dim,
        "use_gripper": bool(cfg.use_gripper),
        "use_intervention_cost": bool(cfg.uses_intervention_cost),
        "encoder": dataclasses.asdict(cfg.encoder),
        "base_policy": cfg.base_policy,
        "base_action_dim": cfg.base_action_dim,
        "bottleneck_label": bool(getattr(cfg, "bottleneck_label", False)),
        "dmp_gripper": bool(getattr(cfg, "dmp_gripper", False)),
        "residual_max": float(getattr(cfg, "residual_max", 1.0)),
        "target_intervention_rate": float(cfg.target_intervention_rate),
        "intervention_window": int(cfg.intervention_window),
    }
    (Path(cfg.checkpoint_dir) / "config.json").write_text(json.dumps(payload, indent=2))


def save_agent_checkpoint(cfg: LearnerConfig, agent, step: int) -> None:
    # The main loop can land on the same step as the final save.
    if (Path(cfg.checkpoint_dir) / f"checkpoint_{step}").exists():
        return
    checkpoints.save_checkpoint(cfg.checkpoint_dir, agent.state, step=step, keep=cfg.keep)


def sample_buffer(buffer, batch_size: int):
    # The pixel path stores ``num_stack + 1`` frames per sample; low-dim
    # buffers have no pixel keys and therefore no packing option.
    if hasattr(buffer, "pixel_keys"):
        return buffer.sample(batch_size, pack_obs_and_next_obs=True)
    return buffer.sample(batch_size)


def flatten_info(info, prefix: str = "") -> dict[str, float]:
    """SAC returns ``{"critic": {...}, "actor": {...}, ...}``; flatten for logs."""
    flat: dict[str, float] = {}
    for key, value in info.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_info(value, name))
        else:
            flat[name] = value
    return flat


class InterventionRateTracker:
    """Pooled human-takeover rate over the last ``window`` episodes."""

    def __init__(self, window: int = 20):
        self.window = window
        self._episodes: collections.deque[tuple[int, int]] = collections.deque(
            maxlen=window
        )
        self._lock = threading.Lock()

    def add_episode(self, intervened_steps: int, total_steps: int) -> None:
        if total_steps <= 0:
            return
        with self._lock:
            self._episodes.append((intervened_steps, total_steps))

    def add_episodes(self, episodes: list[tuple[int, int]]) -> None:
        for intervened_steps, total_steps in episodes:
            self.add_episode(intervened_steps, total_steps)

    @property
    def rate(self) -> float:
        with self._lock:
            episodes = list(self._episodes)
        total_steps = sum(total for _, total in episodes)
        intervened_steps = sum(intervened for intervened, _ in episodes)
        return intervened_steps / max(total_steps, 1)

    @property
    def num_episodes(self) -> int:
        with self._lock:
            return len(self._episodes)


def _sync(cfg: LearnerConfig, seen: set[str], online, demo) -> tuple[int, list[tuple[int, int]]]:
    """Insert new pickles and return online episode stats for the rate window."""
    added = 0
    episodes: list[tuple[int, int]] = []
    if online is not None:
        n, episodes = _transitions.sync_transitions(
            Path(cfg.data_dir) / "online", seen, online
        )
        added += n
    n, _ = _transitions.sync_transitions(
        Path(cfg.data_dir) / "demo",
        seen,
        demo,
        force_episode_start=True,
    )
    added += n
    return added, episodes


def _add_online_mask(batch, value: float):
    mask = np.full(batch["rewards"].shape[0], value, dtype=np.float32)
    return batch.copy(add_or_replace={"online_mask": mask})


def mixed_batch(online, demo, batch_size: int):
    """RLPD's 50/50 online/demo batch (splitting online when no demos exist).

    The ``online_mask`` lets the cost critic ignore demo transitions, which are
    human corrections rather than policy behavior.
    """
    if len(demo) == 0:
        return _add_online_mask(sample_buffer(online, batch_size), 1.0)
    if len(online) == 0:
        return _add_online_mask(sample_buffer(demo, batch_size), 0.0)
    half = batch_size // 2
    online_batch = _add_online_mask(sample_buffer(online, batch_size - half), 1.0)
    demo_batch = _add_online_mask(sample_buffer(demo, half), 0.0)
    return concat_batches(online_batch, demo_batch, axis=0)


def run(cfg: LearnerConfig) -> None:
    """Train one agent from the pickles under ``cfg.data_dir``."""
    save_run_config(cfg)
    print(
        f"[learner] agent={cfg.agent} use_gripper={cfg.use_gripper} "
        f"use_intervention_cost={cfg.uses_intervention_cost}"
    )
    print(f"[learner] CRITIC_ONLY={sorted(cfg.critic_only)}")
    print(f"[learner] FULL_UPDATE={sorted(cfg.full_update)}")

    agent = _agent_from_config(cfg)

    # Always resume from the latest checkpoint, if there is one.
    latest = checkpoints.latest_checkpoint(cfg.checkpoint_dir)
    if latest is not None:
        state = checkpoints.restore_checkpoint(latest, target=agent.state)
        agent = agent.replace(state=state)
        step = int(Path(latest).name.rsplit("_", 1)[-1])
        print(f"restored agent from {latest} (step {step})")
    else:
        step = 0
        # Publish the initial network so the actor can start immediately.
        save_agent_checkpoint(cfg, agent, 0)

    online = (
        None
        if cfg.is_bc
        else _encoders.make_encoder_buffer(
            cfg.encoder, cfg.replay_buffer_capacity, cfg.action_dim
        )
    )
    demo = _encoders.make_encoder_buffer(cfg.encoder, cfg.replay_buffer_capacity, cfg.action_dim)
    tracker = InterventionRateTracker(cfg.intervention_window)
    seen: set[str] = set()
    _, episodes = _sync(cfg, seen, online, demo)
    tracker.add_episodes(episodes)
    print(
        f"loaded transitions: online={len(online) if online is not None else 0} demo={len(demo)} "
        f"intervention_rate={tracker.rate:.3f} ({tracker.num_episodes} episodes)"
    )

    # HG-Dagger: BC pretraining on the initial intervention set.
    if cfg.is_bc and cfg.pretrain_steps > 0 and step == 0:
        while len(demo) == 0:
            time.sleep(cfg.poll_interval)
            _, episodes = _sync(cfg, seen, online, demo)
            tracker.add_episodes(episodes)
        pbar = tqdm.tqdm(range(cfg.pretrain_steps), desc="bc_pretraining", dynamic_ncols=True)
        for _ in pbar:
            agent, info = agent.update(sample_buffer(demo, cfg.batch_size))
            step += 1
            if cfg.log_interval > 0 and step % cfg.log_interval == 0:
                pbar.set_postfix({k: f"{float(v):.4f}" for k, v in flatten_info(info).items()})
            if cfg.checkpoint_period > 0 and step % cfg.checkpoint_period == 0:
                save_agent_checkpoint(cfg, agent, step)
        save_agent_checkpoint(cfg, agent, step)
        print(f"BC pretraining done ({step} steps)")

    pbar = tqdm.tqdm(total=cfg.max_steps, initial=step, desc="learner", dynamic_ncols=True)
    last_sync = 0.0
    while step < cfg.max_steps:
        now = time.monotonic()
        if now - last_sync >= cfg.poll_interval:
            added, episodes = _sync(cfg, seen, online, demo)
            tracker.add_episodes(episodes)
            if added:
                pbar.write(
                    f"[step {step}] +{added} transitions "
                    f"(online={len(online) if online is not None else 0} demo={len(demo)} "
                    f"intervention_rate={tracker.rate:.3f} episodes={tracker.num_episodes})"
                )
            last_sync = now

        if not cfg.is_bc:
            if len(online) < cfg.training_starts:
                time.sleep(cfg.poll_interval)
                last_sync = 0.0
                continue
            # hil-serl's CTA ratio: critic-only steps, then a full update.
            n_this_iter = max(cfg.cta_ratio - 1, 0) + 1
            for _ in range(max(cfg.cta_ratio - 1, 0)):
                agent, _ = agent.update(
                    mixed_batch(online, demo, cfg.batch_size),
                    networks_to_update=cfg.critic_only,
                )
                step += 1

            update_kwargs = {"networks_to_update": cfg.full_update}
            if cfg.uses_intervention_cost:
                # Before the first episode completes, pass the target so the
                # multiplier does not decay for no reason.
                update_kwargs["intervention_rate"] = (
                    tracker.rate
                    if tracker.num_episodes > 0
                    else cfg.target_intervention_rate
                )
            agent, info = agent.update(
                mixed_batch(online, demo, cfg.batch_size),
                **update_kwargs,
            )
        else:
            if len(demo) == 0:
                time.sleep(cfg.poll_interval)
                last_sync = 0.0
                continue
            n_this_iter = 1
            agent, info = agent.update(sample_buffer(demo, cfg.batch_size))

        step += 1
        pbar.update(n_this_iter)
        flat = flatten_info(info)
        flat["intervention_episodes"] = tracker.num_episodes
        pbar.set_postfix({k: f"{float(v):.3f}" for k, v in flat.items()})
        if cfg.log_interval > 0 and step % cfg.log_interval == 0:
            pbar.write(f"[step {step}] " + " ".join(f"{k}={float(v):.4f}" for k, v in sorted(flat.items())))
        if cfg.checkpoint_period > 0 and step % cfg.checkpoint_period == 0:
            save_agent_checkpoint(cfg, agent, step)

    save_agent_checkpoint(cfg, agent, step)
    print("Done.")
