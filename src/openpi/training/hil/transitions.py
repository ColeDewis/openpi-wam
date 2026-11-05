"""On-disk transition handoff between the actor and the RLPD / HG-Dagger learners.

A transition is a plain dict:

    observations      dict, encoder-convention observation at step t
    actions           [action_dim] float32, normalized to [-1, 1]
    next_observations dict, observation at step t+1
    rewards           float32, terminal success = 1.0 (sparse)
    masks             float32, 1 - dones
    dones             bool
    intervened        bool, True when a human takeover produced the action
    active            bool, True while the human owns the robot (cost label)
    grasp_penalty     float32, <= 0 reward shaping for the gripper DQN only

Directories:
    <root>/online   every transition (RLPD's online buffer)
    <root>/demo     intervened transitions, or all bottleneck-section
                    transitions when bottleneck labels are used
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import pickle as pkl
import time

import numpy as np


@dataclasses.dataclass
class Demo:
    transitions: list[dict]
    bottlenecks: list[dict]
    metadata: dict


def make_transition(
    observations,
    actions,
    next_observations,
    reward,
    done,
    intervened,
    active=None,
    grasp_penalty=0.0,
):
    """Build one transition dict.

    ``active`` is the intervention-cost label: True while the human owns the
    robot. It defaults to ``intervened`` for the real-robot recorder, where a
    detected takeover is the same thing.
    """
    return {
        "observations": observations,
        "actions": np.asarray(actions, np.float32),
        "next_observations": next_observations,
        "rewards": np.float32(reward),
        "masks": np.float32(1.0 - float(done)),
        "dones": bool(done),
        "intervened": bool(intervened),
        "active": bool(intervened if active is None else active),
        "grasp_penalty": np.float32(grasp_penalty),
    }


def save_transitions(directory: Path | str, transitions: list[dict]) -> Path | None:
    if not transitions:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"transitions_{time.time_ns()}_{os.getpid()}.pkl"
    with path.open("wb") as f:
        pkl.dump(transitions, f)
    return path


def save_demo(
    directory: Path | str,
    transitions: list[dict],
    *,
    bottlenecks: list[dict] | None = None,
    metadata: dict | None = None,
) -> Path | None:
    """Save a teleop demo with its bottleneck-section labels.

    Old demo pickles are plain lists of transitions; the bottleneck-aware
    format wraps them in a dict so the section boundaries and the full MuJoCo
    states needed to restore each section start travel with the transitions.
    """
    if not transitions:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"demo_{time.time_ns()}_{os.getpid()}.pkl"
    payload = {
        "format": "bottleneck_demo_v1",
        "transitions": transitions,
        "bottlenecks": list(bottlenecks or []),
    }
    if metadata:
        payload.update(metadata)
    with path.open("wb") as f:
        pkl.dump(payload, f)
    return path


def load_demo(path: Path | str) -> Demo:
    """Load either an old list-format demo or the bottleneck-aware dict format."""
    with Path(path).open("rb") as f:
        data = pkl.load(f)
    if isinstance(data, dict) and "transitions" in data:
        return Demo(list(data["transitions"]), list(data.get("bottlenecks", ())), data)
    return Demo(list(data), [], {})


def episode_stats(transitions: list[dict]) -> tuple[int, int]:
    """Return ``(active_steps, total_steps)`` for one episode."""
    active_steps = sum(
        bool(t.get("active", t.get("intervened", False))) for t in transitions
    )
    return active_steps, len(transitions)


def sync_transitions(
    directory: Path | str, seen: set[str], data_store=None
) -> tuple[int, list[tuple[int, int]]]:
    """Insert pickles not seen before into ``data_store`` (or just count them).

    Returns the number of transitions added and one ``(active_steps,
    total_steps)`` pair per pickle. The learner uses the pairs to maintain a
    moving intervention-rate window.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0, []
    added = 0
    episodes = []
    for path in sorted(directory.glob("*.pkl")):
        if str(path) in seen:
            continue
        seen.add(str(path))
        transitions = load_demo(path).transitions
        for transition in transitions:
            # Old pickles predate the ``active`` field.
            transition.setdefault("active", transition.get("intervened", False))
        episodes.append(episode_stats(transitions))
        if data_store is not None:
            for transition in transitions:
                data_store.insert(transition)
        added += len(transitions)
    return added, episodes


class TransitionRecorder:
    """Accumulates one episode and flushes it as hil-serl transition pickles.

    The actor calls ``add_step`` once per control step with the encoder-convention
    observation and the normalized action that was executed. ``flush`` closes the
    episode: the final transition carries the sparse success reward and
    ``done=True``.
    """

    def __init__(self, root: Path | str, algorithm: str):
        self.root = Path(root)
        self.algorithm = algorithm
        self.reset()

    def reset(self) -> None:
        self._observations: list[dict] = []
        self._actions: list[np.ndarray] = []
        self._intervened: list[bool] = []

    def add_step(self, observation: dict, action: np.ndarray, *, intervened: bool = False) -> None:
        self._observations.append(observation)
        self._actions.append(np.asarray(action, np.float32))
        self._intervened.append(bool(intervened))

    def stats(self) -> tuple[int, int]:
        """Return ``(intervened_steps, total_steps)`` for the pending episode."""
        n = len(self._observations) - 1
        if n <= 0:
            return 0, 0
        return sum(self._intervened[:n]), n

    def _transitions(self, *, success: bool) -> list[dict]:
        n = len(self._observations) - 1
        if n <= 0:
            return []
        out = []
        for i in range(n):
            terminal = i == n - 1
            out.append(
                make_transition(
                    self._observations[i],
                    self._actions[i],
                    self._observations[i + 1],
                    reward=1.0 if (success and terminal) else 0.0,
                    done=terminal,
                    intervened=self._intervened[i],
                )
            )
        return out

    def flush(self, *, success: bool) -> list[Path]:
        """Write the episode. Online gets everything (RLPD); demo gets the
        intervened subset (both algorithms)."""
        transitions = self._transitions(success=success)
        self.reset()
        if not transitions:
            return []
        written = []
        if self.algorithm == "rlpd":
            path = save_transitions(self.root / "online", transitions)
            if path is not None:
                written.append(path)
        demo = [t for t in transitions if t["intervened"]]
        if demo:
            path = save_transitions(self.root / "demo", demo)
            if path is not None:
                written.append(path)
        return written
