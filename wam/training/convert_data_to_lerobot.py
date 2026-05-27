# uv run python3 -m wam.training.convert_data_to_lerobot --dataset-dir ./dataset
# uv run python3 -m wam.training.convert_data_to_lerobot --dataset-dir ./dataset --json-path episode_splits.json
import shutil
import os
import json
import h5py
import numpy as np
import tyro
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main(
    dataset_dir: str,
    json_path: str = "episode_splits.json",
    save_name: str = "wxat333/wam_teleop_dataset",
    output_dir: str = "./wam_teleop_dataset"
):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON file: {json_path}")
        
    with open(json_path, 'r') as f:
        splits = json.load(f)

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=save_name,
        root=output_path,
        robot_type="wam",
        fps=10, 
        features={
            "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    dataset_path = Path(dataset_dir)
    
    for split in splits:
        hdf5_file_path = dataset_path / split['file']
        if not hdf5_file_path.exists():
            continue

        task_prompt = split['task_names'][0] if isinstance(split.get('task_names'), list) else split.get('task_name', "")

        with h5py.File(hdf5_file_path, 'r') as src_h5:
            front_imgs = src_h5["front_image"][:]
            wrist_imgs = src_h5["wrist_image"][:]

            for i in range(len(front_imgs) - 1):
                current_jp = src_h5["follower_state/jp"][i]
                next_jp = src_h5["follower_state/jp"][i + 1]
                
                delta_jp = next_jp - current_jp  # delta action
                gripper = src_h5["follower_state/gripper"][i]  # absolute, as required
                
                state = np.concatenate([current_jp, [gripper]]).astype(np.float32)
                action = np.concatenate([delta_jp, [gripper]]).astype(np.float32)

                dataset.add_frame({
                    "image": front_imgs[i],
                    "wrist_image": wrist_imgs[i],
                    "state": state,
                    "actions": action,
                    "task": task_prompt, 
                })
            
            dataset.save_episode()

if __name__ == "__main__":
    tyro.cli(main)
