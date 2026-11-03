"""Train RLPD or HG-Dagger in LIBERO with browser-gamepad interventions.

Record:
    uv run --extra libero scripts/train_hil_libero.py --record-demos 20 \
        --bottleneck-label --demo-path demos/goal_t1 --suite libero_goal --task-id 1

Train:
    uv run --extra libero scripts/train_hil_libero.py --algorithm rlpd \
        --base-policy dmp --demo-path demos/goal_t1 --suite libero_goal --task-id 1

Controls:
    OPTIONS/SHARE: success/failure
    D-pad up/down: select bottleneck
    D-pad left/right: mark sections while recording, otherwise skip eval
    L1/R1: skip eval
"""

import os

# Leave GPU headroom for the LIBERO EGL contexts (one per env subprocess).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import tyro

from openpi.training.hil import libero as _libero


def main(cfg: _libero.SimConfig) -> None:
    _libero.HilLiberoTrainer(cfg).run()


if __name__ == "__main__":
    main(tyro.cli(_libero.SimConfig))
