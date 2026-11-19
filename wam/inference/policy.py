"""Common lifecycle for policies driven by ``wam/inference/main.py``."""

from __future__ import annotations

import numpy as np


class PolicyTemplate:
    """Base class for a policy that emits an action chunk each control step.

    ``main.py`` calls ``pre_infer``, ``infer``, and ``post_infer`` in that order.
    ``infer`` returns an ``(horizon, action_size)`` chunk, or ``None`` when the
    policy does not want to send anything this step.
    """

    horizon: int = 1
    action_size: int = 7
    needs_images: bool = True
    image_size: tuple[int, int] | None = None

    def start(self) -> None:
        """Reset per-episode state."""

    def pre_infer(self, obs: dict) -> None:
        """Prepare the policy before inference."""

    def infer(self, obs: dict) -> np.ndarray | None:
        """Return an action chunk to send."""
        raise NotImplementedError

    def post_infer(self, obs: dict) -> None:
        """Bookkeeping after inference, before the chunk is sent."""

    def finish(self, success: bool) -> None:
        """Called when the episode ends."""

    def close(self) -> None:
        """Release resources."""
