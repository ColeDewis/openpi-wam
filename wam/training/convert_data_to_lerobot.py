import shutil
import os
import json
import h5py
import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main(
    hdf5_path: str,
    json_path: str = "episode_splits.json",
    save_name: str = "wxat333/wam_teleop_dataset"
):
    """
    Reads an original HDF5 episode and a JSON splits file, 
    extracts the sub-episodes, and converts them directly to LeRobot format.
    """
    # 1. Load the annotations
    if not os.path.exists(json_path):
        print(f"[!] Cannot find JSON file: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        splits = json.load(f)

    if not splits:
        print("[!] No splits found in the JSON file.")
        return

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / save_name
    if output_path.exists():
        print(f"[*] Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # 2. Create LeRobot dataset, define features to store
    print(f"[*] Initializing LeRobot Dataset: {save_name}")
    dataset = LeRobotDataset.create(
        repo_id=save_name,
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

    print(f"[*] Opening original dataset: {hdf5_path}")
    
    # 3. Open original HDF5 and start converting
    with h5py.File(hdf5_path, 'r') as src_h5:
        for i, split in enumerate(splits):
            start = split['start']
            end = split['end'] + 1  # Add 1 so the final frame is inclusive
            task_prompt = split['task_name']

            print(f"\n  -> Processing Split {i+1}/{len(splits)}")
            print(f"     Frames: [{start} : {end - 1}] | Task: '{task_prompt}'")

            # Load ONLY the sliced data into memory for speed
            front_imgs = src_h5["front_image"][start:end]
            wrist_imgs = src_h5["wrist_image"][start:end]
            num_steps = len(front_imgs)

            # Extract and format State (Follower JP [7] + Gripper [1] = 8D vector)
            follower_jp = src_h5["follower_state/jp"][start:end]
            follower_gripper = np.expand_dims(src_h5["follower_state/gripper"][start:end], axis=1)
            states = np.concatenate([follower_jp, follower_gripper], axis=1).astype(np.float32)

            # Extract and format Actions (Leader JP [7] + Gripper [1] = 8D vector)
            leader_jp = src_h5["leader_state/jp"][start:end]
            leader_gripper = np.expand_dims(src_h5["leader_state/gripper"][start:end], axis=1)
            actions = np.concatenate([leader_jp, leader_gripper], axis=1).astype(np.float32)

            # 4. Add frames sequentially for this sub-episode
            for step_idx in range(num_steps):
                dataset.add_frame(
                    {
                        "image": front_imgs[step_idx],
                        "wrist_image": wrist_imgs[step_idx],
                        "state": states[step_idx],
                        "actions": actions[step_idx],
                        "task": task_prompt, 
                    }
                )
            
            # Save the episode to disk after adding all frames
            dataset.save_episode()
            print("     [+] Episode saved.")

    print(f"\n[*] Conversion complete! Dataset saved to: {output_path}")

if __name__ == "__main__":
    tyro.cli(main)
