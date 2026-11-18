#!/usr/bin/env python3
"""RLPD on the WAM robot.

SAC with 50/50 online/demo replay sampling and hil-serl's CTA update schedule.
The actor side lives in ``wam/inference/main.py`` and writes transition pickles;
this is the learner. The latest checkpoint in ``--checkpoint-dir`` is restored
automatically.

Example:
uv run scripts/train_rlpd.py \
        --data-dir /home/serg/projects/openpi-wam/interventions/test \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/test_rlpd
"""

from __future__ import annotations
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

import tyro

from openpi.training.hil import learner as _learner


def main(cfg: _learner.LearnerConfig) -> None:
    cfg.algorithm = "rlpd"
    _learner.run(cfg)


if __name__ == "__main__":
    main(tyro.cli(_learner.LearnerConfig))
