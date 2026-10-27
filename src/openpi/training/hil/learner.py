"""Shared learner loop for RLPD (SAC) and HG-Dagger (BC).

The two upstream scripts differ only in which buffers they sample and whether
they run a BC pretraining phase; everything else (agent construction, the
file-based actor handoff, checkpointing) is shared here. Keeping the algorithm
structure in one place makes the two entry points thin and readable:

    scripts/train_rlpd.py      -> run(..., algorithm="rlpd")
    scripts/train_hgdagger.py  -> run(..., algorithm="hgdagger")
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import time

from flax.training import checkpoints
import tqdm

from openpi.training.hil import encoders as _encoders
from openpi.training.hil import transitions as _transitions
from serl_launcher.utils.train_utils import concat_batches

# Networks updated per step. RLPD follows hil-serl's CTA schedule: ``cta_ratio-1``
# critic-only updates, then one update that also moves the actor and temperature.
# The DQN gripper ("grasp_critic") trains on every step, like hil-serl's hybrid example.
CRITIC_ONLY = frozenset({"critic", "grasp_critic"})
FULL_UPDATE = frozenset({"critic", "grasp_critic", "actor", "temperature"})


@dataclasses.dataclass
class LearnerConfig:
    data_dir: str = ""
    """Directory written by ``wam/inference/main.py`` (contains online/ and demo/)."""

    checkpoint_dir: str = ""
    """Where agents are saved. Also read back by ``wam/inference``."""

    encoder: _encoders.EncoderConfig = dataclasses.field(default_factory=_encoders.EncoderConfig)

    algorithm: str = "rlpd"
    seed: int = 42
    action_dim: int = 7

    # Data / optimization. Defaults mirror hil-serl's examples.
    batch_size: int = 256
    cta_ratio: int = 5
    discount: float = 0.97
    soft_target_update_rate: float = 0.005
    reward_bias: float = 0.0
    target_entropy: float | None = None

    training_starts: int = 1_000
    max_steps: int = 100_000
    replay_buffer_capacity: int = 200_000
    pretrain_steps: int = 20_000
    poll_interval: float = 5.0

    log_interval: int = 100
    checkpoint_period: int = 5_000
    keep: int = 20
    resume: bool = False

    def __post_init__(self):
        if not self.data_dir:
            raise ValueError("--data_dir is required (it is written by wam/inference/main.py)")
        if not self.checkpoint_dir:
            raise ValueError("--checkpoint_dir is required")
        if self.algorithm not in ("rlpd", "hgdagger"):
            raise ValueError(f"algorithm must be 'rlpd' or 'hgdagger', got {self.algorithm!r}")
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def _agent_from_config(cfg: LearnerConfig):
    return _encoders.make_encoder_agent(
        "bc" if cfg.algorithm == "hgdagger" else "sac",
        cfg.encoder,
        cfg.seed,
        cfg.action_dim,
        reward_bias=cfg.reward_bias,
        target_entropy=cfg.target_entropy,
        discount=cfg.discount,
        soft_target_update_rate=cfg.soft_target_update_rate,
    )


def save_run_config(cfg: LearnerConfig) -> None:
    payload = {
        "algorithm": cfg.algorithm,
        "action_dim": cfg.action_dim,
        "encoder": dataclasses.asdict(cfg.encoder),
    }
    (Path(cfg.checkpoint_dir) / "config.json").write_text(json.dumps(payload, indent=2))


def _restore_if_requested(cfg: LearnerConfig, agent):
    if not cfg.resume:
        return agent, 0
    latest = checkpoints.latest_checkpoint(cfg.checkpoint_dir)
    if latest is None:
        return agent, 0
    state = checkpoints.restore_checkpoint(latest, target=agent.state)
    step = int(Path(latest).name.rsplit("_", 1)[-1])
    print(f"restored agent from {latest} (step {step})")
    return agent.replace(state=state), step


def save_agent_checkpoint(cfg: LearnerConfig, agent, step: int) -> None:
    # The main loop can land on the same step as the final save (and a resume
    # re-enters at an existing step); saving twice would raise.
    if (Path(cfg.checkpoint_dir) / f"checkpoint_{step}").exists():
        return
    checkpoints.save_checkpoint(cfg.checkpoint_dir, agent.state, step=step, keep=cfg.keep)


def sample_buffer(buffer, batch_size: int):
    # The pixel path stores ``num_stack + 1`` frames per sample; the low-dim
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


def _sync(cfg: LearnerConfig, seen: set[str], online, demo) -> int:
    added = 0
    if online is not None:
        added += _transitions.sync_transitions(Path(cfg.data_dir) / "online", seen, online)
    added += _transitions.sync_transitions(Path(cfg.data_dir) / "demo", seen, demo)
    return added


def mixed_batch(online, demo, batch_size: int):
    """RLPD's 50/50 online/demo batch (splitting online when no demos exist)."""
    if len(demo) == 0:
        return sample_buffer(online, batch_size)
    if len(online) == 0:
        return sample_buffer(demo, batch_size)
    half = batch_size // 2
    return concat_batches(sample_buffer(online, batch_size - half), sample_buffer(demo, half), axis=0)


def run(cfg: LearnerConfig) -> None:
    """Train one agent from the pickles under ``cfg.data_dir``."""
    save_run_config(cfg)
    agent = _agent_from_config(cfg)
    agent, start_step = _restore_if_requested(cfg, agent)
    if checkpoints.latest_checkpoint(cfg.checkpoint_dir) is None:
        # Publish the initial (randomly initialized) network so the actor can
        # start its first episode immediately, matching hil-serl's
        # `server.publish_network(agent.state.params)` before the buffer fills.
        save_agent_checkpoint(cfg, agent, 0)

    online = (
        _encoders.make_encoder_buffer(cfg.encoder, cfg.replay_buffer_capacity, cfg.action_dim)
        if cfg.algorithm == "rlpd"
        else None
    )
    demo = _encoders.make_encoder_buffer(cfg.encoder, cfg.replay_buffer_capacity, cfg.action_dim)
    seen: set[str] = set()
    _sync(cfg, seen, online, demo)
    print(f"loaded transitions: online={len(online) if online is not None else 0} demo={len(demo)}")

    # ── HG-Dagger: BC pretraining on the initial intervention set ────────────
    step = start_step
    if cfg.algorithm == "hgdagger" and cfg.pretrain_steps > 0 and step == 0:
        pbar = tqdm.tqdm(range(cfg.pretrain_steps), desc="bc_pretraining", dynamic_ncols=True)
        for _ in pbar:
            if len(demo) == 0:
                # No interventions yet: wait for the operator instead of
                # spinning on an empty buffer.
                time.sleep(cfg.poll_interval)
                _sync(cfg, seen, online, demo)
                continue
            agent, info = agent.update(sample_buffer(demo, cfg.batch_size))
            step += 1
            if step % cfg.log_interval == 0:
                pbar.set_postfix({k: f"{float(v):.4f}" for k, v in flatten_info(info).items()})
            if step % cfg.checkpoint_period == 0:
                save_agent_checkpoint(cfg, agent, step)
        save_agent_checkpoint(cfg, agent, step)
        print(f"BC pretraining done ({step} steps)")

    # ── main loop ────────────────────────────────────────────────────────────
    pbar = tqdm.tqdm(total=cfg.max_steps, initial=step, desc="learner", dynamic_ncols=True)
    last_sync = 0.0
    while step < cfg.max_steps:
        now = time.monotonic()
        if now - last_sync >= cfg.poll_interval:
            added = _sync(cfg, seen, online, demo)
            if added:
                pbar.write(
                    f"[step {step}] +{added} transitions "
                    f"(online={len(online) if online is not None else 0} demo={len(demo)})"
                )
            last_sync = now

        if cfg.algorithm == "rlpd":
            if len(online) < cfg.training_starts:
                time.sleep(cfg.poll_interval)
                last_sync = 0.0
                continue
            # hil-serl's CTA ratio: critic-only steps, then a full update.
            n_this_iter = max(cfg.cta_ratio - 1, 0) + 1
            for _ in range(max(cfg.cta_ratio - 1, 0)):
                agent, _ = agent.update(
                    mixed_batch(online, demo, cfg.batch_size),
                    networks_to_update=CRITIC_ONLY,
                )
                step += 1
            batch = mixed_batch(online, demo, cfg.batch_size)
            agent, info = agent.update(batch, networks_to_update=FULL_UPDATE)
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
        pbar.set_postfix({k: f"{float(v):.3f}" for k, v in flat.items()})
        if step % cfg.log_interval == 0:
            pbar.write(f"[step {step}] " + " ".join(f"{k}={float(v):.4f}" for k, v in sorted(flat.items())))
        if step % cfg.checkpoint_period == 0:
            save_agent_checkpoint(cfg, agent, step)

    save_agent_checkpoint(cfg, agent, step)
    print("Done.")
