"""Train RLPD or HG-Dagger in LIBERO with browser-gamepad interventions.

Record demos first (pure teleop, one pickle per successful episode):

    uv run --extra libero scripts/train_hil_libero.py --record-demos 20 \
        --demo-path /home/serg/projects/openpi-wam/demos/goal_t1 --suite libero_goal --task-id 1

Then train. The latest checkpoint in ``--checkpoint-dir`` is restored
automatically:

    uv run --extra libero scripts/train_hil_libero.py \
        --algorithm rlpd --suite libero_goal --task-id 1 \
        --demo-path /home/serg/projects/openpi-wam/demos/goal_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_rlpd \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/rlpd_goal_t1

To learn a residual on top of a DMP base policy, add ``--base-policy dmp``.
The DMP is fit to the first trajectory under ``--demo-path``:

    uv run --extra libero scripts/train_hil_libero.py --algorithm rlpd \
        --suite libero_goal --task-id 1 --base-policy dmp \
        --demo-path /home/serg/projects/openpi-wam/demos/goal_t1 \
        --checkpoint-dir /home/serg/projects/openpi-wam/checkpoints/hil/libero_goal_t1_rlpd_dmp \
        --video-dir /home/serg/projects/openpi-wam/videos/hil_libero/rlpd_goal_t1_dmp

Open the printed URL in Chrome (tunnel it if the box is remote) and use the
controller: OPTIONS starts an episode / marks success, SHARE marks failure,
left stick moves x/y, triangle/cross moves z, L2/R2 opens/closes the gripper,
and PS hands control back to the policy. D-pad left/right (or L1/R1) skips the
current eval episode.

Headless smoke test:

    MUJOCO_GL=egl uv run --extra libero scripts/train_hil_libero.py \
        --suite libero_goal --task-id 1 --no-display \
        --max-episodes 1 --max-env-steps 40
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
