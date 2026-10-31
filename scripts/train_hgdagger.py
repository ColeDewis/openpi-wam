#!/usr/bin/env python3
"""HG-Dagger on the WAM robot.

BC on the human intervention dataset, with a BC pretraining phase followed by
continued training as new interventions arrive. The actor side lives in
``wam/inference/main.py`` and writes the demo pickles; this is the learner. The
latest checkpoint in ``--checkpoint-dir`` is restored automatically.

Example:
    uv run scripts/train_hgdagger.py \
        --data-dir ./interventions/book \
        --checkpoint-dir ./checkpoints/hil/book_hgdagger
"""

from __future__ import annotations

import tyro

from openpi.training.hil import learner as _learner


def main(cfg: _learner.LearnerConfig) -> None:
    cfg.algorithm = "hgdagger"
    _learner.run(cfg)


if __name__ == "__main__":
    main(tyro.cli(_learner.LearnerConfig))
