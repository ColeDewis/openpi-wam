"""Train RLPD or HG-Dagger in LIBERO with browser-gamepad interventions.

Record:
    cd /home/serg/projects/openpi-wam
    uv run --extra libero /home/serg/projects/openpi-wam/scripts/train_hil_libero.py \
        --record-demos 20 --bottleneck-label --suite libero_goal --task-id 1 \
        --demo-path /home/serg/projects/openpi-wam/demos/dmp_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_demos \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/goal_t1_demos

Train:
    cd /home/serg/projects/openpi-wam
    uv run --extra libero /home/serg/projects/openpi-wam/scripts/train_hil_libero.py \
        --algorithm rlpd --suite libero_goal --task-id 1 --base-policy dmp \
        --demo-path /home/serg/projects/openpi-wam/demos/dmp_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_rlpd_dmp_bottleneck_lagrange \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/rlpd_goal_t1_dmp_bottleneck_lagrange

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
