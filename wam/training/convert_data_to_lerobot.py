"""
convert dataset to lerobot format.
"""

import shutil

import os
import h5py

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

TASK_PROMPT = "do something"
SAVE_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
SOURCE_FOLDER = "test"
RAW_DATASET_NAMES = [
    "episode1",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / SAVE_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=SAVE_NAME,
        robot_type="wam",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        filepath = os.path.join(data_dir, f"{raw_dataset_name}.hdf5")
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Skipping.")
            continue
            
        with h5py.File(filepath, 'r') as f:
            # Determine the episode length by checking the size of one of the datasets
            num_steps = len(f["state/joints"])
            
            for i in range(num_steps):
                dataset.add_frame(
                    {
                        "image": f["images/front"][i],
                        "wrist_image": f["images/wrist"][i],
                        "state": f["state/joints"][i],
                        "actions": f["actions"][i],
                        "task": TASK_PROMPT,  # Added a dummy string, as OpenPI/LeRobot often expects a task prompt
                    }
                )
            
            # Save the episode to disk after adding all frames
            dataset.save_episode()




if __name__ == "__main__":
    # tyro.cli(main)
    main(data_dir=SOURCE_FOLDER)
