"""Train RLPD or HG-Dagger in LIBERO, with keyboard interventions.

The GPU box runs the simulator, agent, and learner. `--display` starts a
headless-safe MJPEG stream plus a browser keyboard endpoint:

    http://<gb10-ip>:8080/

Open that URL from the laptop, or tunnel it:
    ssh -L 8080:localhost:8080 serg@promaxgb10-7c7f

Examples:

    # RLPD: policy rollouts fill `online`; arrow takeovers fill `demo`.
    uv run scripts/train_hil_libero.py --algorithm rlpd \
        --suite libero_goal --task-id 1 --display

    # HG-Dagger: only arrow takeovers fill the BC buffer.
    uv run scripts/train_hil_libero.py --algorithm hgdagger \
        --suite libero_goal --task-id 1 --display --pretrain-steps 2000

Browser keys (and local pynput, if a display exists): R start/pause,
S save(success), D discard(failure), arrows = Cartesian takeover
(up/down=x, left/right=y), q/e=z, g=gripper.

Headless smoke test (no live view, auto-start):
    MUJOCO_GL=egl uv run scripts/train_hil_libero.py --algorithm rlpd \
        --suite libero_goal --task-id 1 --keyboard none --no-display \
        --max-episodes 1 --max-env-steps 40 --training-starts 20 \
        --batch-size 8 --checkpoint-period 100
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
