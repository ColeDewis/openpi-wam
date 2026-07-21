import dataclasses
from pathlib import Path

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib.pyplot as plt

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training.data_loader import TransformedDataset, create_torch_dataset

if __name__ == "__main__":

    config = _config.get_config("haptic_wam_pi05")
    config = dataclasses.replace(config, batch_size=10)
    checkpoint_dir = Path(
        "/mnt/10tb/dyanmiller/openpi-wam/checkpoints/haptic_wam_pi05/haptic_wam_pi05/29999"
    )

    # Create a trained policy.
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    dataset = lerobot_dataset.LeRobotDataset(
        config.data.repo_id,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )

    dataset = TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
        ],
    )


    i = 0
    while i < 200:
        example = dataset[i]

        result = policy.infer(example)
        n_jnts = result["actions"].shape[1]
        ncols = 2
        nrows = (n_jnts + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for jnt_idx in range(n_jnts):
            ax = axes_flat[jnt_idx]

            # Plot to the axis
            ax.plot(result["actions"][:, jnt_idx], label="actions")
            ax.scatter(0, example["observation/state"][jnt_idx].numpy(), label="last_obs")
            ax.set_title(f"Joint {jnt_idx}")
            ax.legend()

        plt.tight_layout()
        plt.show()

        i += 5

