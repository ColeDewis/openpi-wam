#!/usr/bin/env python3
"""RLPD on the WAM robot.

Faithful port of hil-serl's ``examples/train_rlpd.py``: SAC with 50/50 online /
demo replay sampling and a ``cta_ratio`` critic-to-actor update schedule. The
actor side lives in ``wam/inference/main.py`` (it writes transition pickles);
this is the learner.

The observation encoder is selected with ``--encoder.kind``:

    hilserl  -- the original hil-serl ResNet-10 path
    rltoken  -- frozen VLAs RL token + proprio
    timm     -- frozen torch/timm encoder + force/proprio

Example:
    uv run scripts/train_rlpd.py \
        --data-dir ./interventions/book \
        --checkpoint-dir ./checkpoints/hil/book_rlpd \
        --encoder.kind hilserl
"""

from __future__ import annotations

import tyro

from openpi.training.hil import learner as _learner


def main(cfg: _learner.LearnerConfig) -> None:
    cfg.algorithm = "rlpd"
    _learner.run(cfg)


if __name__ == "__main__":
    main(tyro.cli(_learner.LearnerConfig))
