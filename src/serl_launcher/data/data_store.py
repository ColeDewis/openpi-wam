from threading import Lock
from typing import Iterable

import gymnasium as gym
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)
from serl_launcher.data.replay_buffer import ReplayBuffer


class ReplayBufferDataStore(ReplayBuffer):
    """Thread-safe ``ReplayBuffer``.

    Upstream this also inherits agentlace's ``DataStoreBase`` for the actor /
    learner network transport. This repo uses a file-based handoff instead, so
    only the locking behaviour (which is what the buffer itself needs) is kept.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        include_grasp_penalty: bool = False,
    ):
        ReplayBuffer.__init__(
            self,
            observation_space,
            action_space,
            capacity,
            include_grasp_penalty=include_grasp_penalty,
        )
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    def latest_data_id(self):
        return self._insert_index

    def last_inserted_index(self):
        with self._lock:
            return super().last_inserted_index()

    def update_transition(self, index, **fields):
        with self._lock:
            return super().update_transition(index, **fields)

    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer):
    """Thread-safe ``MemoryEfficientReplayBuffer`` (see the class above)."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity, pixel_keys=image_keys, **kwargs
        )
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(*args, **kwargs)

    def latest_data_id(self):
        return self._insert_index

    def last_inserted_index(self):
        with self._lock:
            return super().last_inserted_index()

    def update_transition(self, index, **fields):
        with self._lock:
            return super().update_transition(index, **fields)

    def force_episode_start(self):
        with self._lock:
            return super().force_episode_start()

    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


def populate_data_store(data_store, demos_path: Iterable[str]):
    """Load a list of hil-serl transition pickles into ``data_store``."""
    import pickle as pkl

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                transition.pop("active", None)
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
