"""On-disk transition handoff between the actor (``wam/inference/main.py``) and
the RLPD / HG-Dagger learners.

hil-serl writes transition pickles and lets the learner read them; this module
keeps that format and adds the small amount of bookkeeping needed to tail a
directory. A transition is a plain dict:

    observations      dict, encoder-convention observation at step t
    actions           [action_dim] float32, normalized to [-1, 1]
    next_observations dict, observation at step t+1
    rewards           float32, terminal success = 1.0 (sparse)
    masks             float32, 1 - dones
    dones             bool
    intervened        bool, True when a human takeover produced the action
    grasp_penalty     float32, <= 0 reward shaping for the gripper DQN only (a toggle of the
                      commanded gripper state); 0.0 when the actor does not compute it

Directories:
    <root>/online   every transition (RLPD's online buffer)
    <root>/demo     intervened transitions only (RLPD's demo buffer and
                    HG-Dagger's entire dataset)
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle as pkl
import time

import numpy as np

TRANSITION_FIELDS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
    "intervened",
    "grasp_penalty",
)


def make_transition(observations, actions, next_observations, reward, done, intervened, grasp_penalty=0.0):
    return {
        "observations": observations,
        "actions": np.asarray(actions, np.float32),
        "next_observations": next_observations,
        "rewards": np.float32(reward),
        "masks": np.float32(1.0 - float(done)),
        "dones": bool(done),
        "intervened": bool(intervened),
        "grasp_penalty": np.float32(grasp_penalty),
    }


def _new_pickle_path(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # Unique even when two flushes land in the same clock tick.
    name = f"transitions_{time.time_ns()}_{os.getpid()}.pkl"
    return directory / name


def save_transitions(directory: Path | str, transitions: list[dict]) -> Path | None:
    if not transitions:
        return None
    path = _new_pickle_path(Path(directory))
    with path.open("wb") as f:
        pkl.dump(transitions, f)
    return path


def load_transitions(directory: Path | str) -> list[dict]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(directory.glob("*.pkl")):
        with path.open("rb") as f:
            out.extend(pkl.load(f))
    return out


def sync_transitions(directory: Path | str, seen: set[str], data_store=None) -> int:
    """Insert pickles not seen before into ``data_store`` (or just count them).

    Returns the number of transitions added. ``seen`` is updated in place so a
    learner can call this every iteration without re-reading old files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    added = 0
    for path in sorted(directory.glob("*.pkl")):
        if str(path) in seen:
            continue
        seen.add(str(path))
        with path.open("rb") as f:
            transitions = pkl.load(f)
        if data_store is not None:
            for transition in transitions:
                data_store.insert(transition)
        added += len(transitions)
    return added


class TransitionRecorder:
    """Accumulates one episode and flushes it as hil-serl transition pickles.

    The actor loop calls ``add_step`` once per control step with the *encoder
    convention* observation (the policy's ``encode`` output), the normalized
    action that was actually executed, and whether a human intervened. The
    next observation is filled in on the following step. ``flush`` closes the
    episode: the final transition carries the sparse success reward and
    ``done=True``.
    """

    def __init__(
        self,
        root: Path | str,
        algorithm: str,
        *,
        only_interventions: bool | None = None,
    ):
        self.root = Path(root)
        self.algorithm = algorithm
        if only_interventions is None:
            only_interventions = algorithm == "hgdagger"
        self.only_interventions = only_interventions
        self.reset()

    def reset(self) -> None:
        self._observations: list[dict] = []
        self._actions: list[np.ndarray] = []
        self._intervened: list[bool] = []

    def add_step(self, observation: dict, action: np.ndarray, *, intervened: bool = False) -> None:
        self._observations.append(observation)
        self._actions.append(np.asarray(action, np.float32))
        self._intervened.append(bool(intervened))

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
        if self.algorithm == "rlpd" and not self.only_interventions:
            path = save_transitions(self.root / "online", transitions)
            if path is not None:
                written.append(path)
        demo = [t for t in transitions if t["intervened"]]
        if demo:
            path = save_transitions(self.root / "demo", demo)
            if path is not None:
                written.append(path)
        return written
