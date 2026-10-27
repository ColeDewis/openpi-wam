#!/usr/bin/env python3
"""HG-Dagger on the WAM robot.

Faithful port of hil-serl's ``examples/train_hgdagger.py``: BC on the human
intervention dataset, with a BC pretraining phase followed by continued
training as new interventions arrive. The actor side lives in
``wam/inference/main.py`` (it writes the demo pickles); this is the learner.

The observation encoder is selected with ``--encoder.kind``:

    hilserl  -- the original hil-serl ResNet-10 path
    rltoken  -- frozen VLAs RL token + proprio
    timm     -- frozen torch/timm encoder + force/proprio

Example:
    uv run scripts/train_hgdagger.py \
        --data-dir ./interventions/book \
        --checkpoint-dir ./checkpoints/hil/book_hgdagger \
        --encoder.kind hilserl
"""

from __future__ import annotations

import tyro

from openpi.training.hil import learner as _learner


def main(cfg: _learner.LearnerConfig) -> None:
    cfg.algorithm = "hgdagger"
    _learner.run(cfg)


if __name__ == "__main__":
    main(tyro.cli(_learner.LearnerConfig))
