# uv run python3 -m wam.training.convert_data_to_lerobot --dataset-dir ./dataset
# uv run python3 -m wam.training.convert_data_to_lerobot --dataset-dir ./dataset --json-path episode_splits.json
import json
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro


def find_nearest_index(timestamps, target_time):
    """Return index of the timestamp closest to target_time."""
    return int(np.argmin(np.abs(timestamps - target_time)))


def main(
    dataset_dir: list[str],
    save_name: str = "Breakdancingbear/wam_teleop_dataset",
    overwrite: bool = False,
):
    hf_cache_path = HF_LEROBOT_HOME / Path(save_name)
    if hf_cache_path.exists():
        if overwrite:
            print(f"Overwriting existing cached dataset at {hf_cache_path}")
            shutil.rmtree(hf_cache_path)
        else:
            response = input(f"Cached dataset exists at {hf_cache_path}. Overwrite? [y/N]: ")
            if response.strip().lower() == "y":
                shutil.rmtree(hf_cache_path)
            else:
                print("Aborting.")
                return

    dataset = LeRobotDataset.create(
        repo_id=save_name,
        # root=output_path,
        robot_type="wam",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    dataset_paths = [Path(d) for d in dataset_dir]

    for d_path in dataset_paths:
        json_path = d_path / "episode_splits.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Missing required JSON file: {json_path}")

        with open(json_path) as f:
            splits = json.load(f)

        for split in splits:
            hdf5_file_path = d_path / split["file"]
            if not hdf5_file_path.exists():
                continue

            task_prompt = (
                split["task_names"][0] if isinstance(split.get("task_names"), list) else split.get("task_name", "")
            )

            with h5py.File(hdf5_file_path, "r") as src_h5:
                timestamps = src_h5["low_dim/timestamp_ns"][:]
                front_imgs = src_h5["front_image"][:]
                wrist_imgs = src_h5["wrist_image"][:]
                jp_arr = src_h5["low_dim/follower_jp"][:]
                gripper_arr = src_h5["low_dim/gripper_pos"][:]

                t_start = timestamps[0]
                t_end = timestamps[-1]

                step_ns = 100_000_000
                target_times = np.arange(t_start, t_end, step_ns)  # 0.1s timestep for a 10fps dataset

                sampled_indices = [find_nearest_index(timestamps, t) for t in target_times]
                sampled_indices = list(dict.fromkeys(sampled_indices))
                n_sampled = len(sampled_indices)
                episode_frames = []

                for idx in range(n_sampled):
                    i = sampled_indices[idx]

                    jp = jp_arr[i]
                    gripper = gripper_arr[i]

                    state = np.concatenate([jp, [gripper]]).astype(np.float32)
                    action = np.concatenate([jp, [gripper]]).astype(np.float32)

                    episode_frames.append(
                        {
                            "image": front_imgs[i],
                            "wrist_image": wrist_imgs[i],
                            "state": state,
                            "actions": action,
                            "task": task_prompt,
                        }
                    )

                for frame in episode_frames:
                    dataset.add_frame(frame)

                dataset.save_episode()

    print(f"Pushing to HuggingFace Hub as {save_name} ...")
    # make sure to hf auth login
    dataset.push_to_hub(
        repo_id=save_name,
        private=False,
    )
    print(f"https://huggingface.co/datasets/{save_name}")


if __name__ == "__main__":
    tyro.cli(main)
