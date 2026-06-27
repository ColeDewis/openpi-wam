# uv run python3 -m wam.training.convert_data_to_lerobot --dataset-dir ./dataset
# uv run python3 -m wam.training.convert_data_to_lerobot --dataset-dir ./dataset --json-path episode_splits.json
import shutil
import os
import json
import h5py
import numpy as np
import tyro
from pathlib import Path
from scipy.spatial.transform import Rotation as R


from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

POS_THRESHOLD = 0.001
ROT_THRESHOLD = 0.01

def find_nearest_index(timestamps, target_time):
    """Return index of the timestamp closest to target_time."""
    return int(np.argmin(np.abs(timestamps - target_time)))

def quat_to_euler(quat):
    """Quaternion [x, y, z, w] → Euler (roll, pitch, yaw) in radians."""
    return R.from_quat(quat).as_euler('xyz', degrees=False)


def delta_euler(rot_current, rot_next):
    """Minimal rotation delta between two quaternions, expressed as Euler xyz."""
    r_delta = R.from_quat(rot_current).inv() * R.from_quat(rot_next)
    return r_delta.as_euler('xyz', degrees=False)


def is_significant(delta_pos, delta_rot):
    """True if any axis of pos OR rot exceeds its threshold."""
    return (np.any(np.abs(delta_pos) > POS_THRESHOLD) or
            np.any(np.abs(delta_rot) > ROT_THRESHOLD))


def main(
    dataset_dir: str,
    json_path: str = "episode_splits.json",
    save_name: str = "Breakdancingbear/wam_teleop_dataset",
    overwrite: bool = False,
):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON file: {json_path}")
        
    with open(json_path, 'r') as f:
        splits = json.load(f)

    hf_cache_path = HF_LEROBOT_HOME / Path(save_name)
    if hf_cache_path.exists():
        if overwrite:
            print(f"Overwriting existing cached dataset at {hf_cache_path}")
            shutil.rmtree(hf_cache_path)
        else:
            response = input(f"Cached dataset exists at {hf_cache_path}. Overwrite? [y/N]: ")
            if response.strip().lower() == 'y':
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
            "image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
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
            timestamps  = src_h5["timestamp"][:]
            front_imgs  = src_h5["front_image"][:]
            wrist_imgs  = src_h5["wrist_image"][:]
            cart_pos    = src_h5["follower_state/cart_pos"][:]
            cart_rot    = src_h5["follower_state/cart_rot"][:]
            gripper_arr = src_h5["follower_state/gripper"][:]

            t_start = timestamps[0]
            t_end   = timestamps[-1]

            target_times = np.arange(t_start, t_end, 0.1) # 0.1 timestep for a 10fps dataset

            sampled_indices = [find_nearest_index(timestamps, t) for t in target_times]
            sampled_indices = list(dict.fromkeys(sampled_indices))
            n_sampled = len(sampled_indices)

            WINDOW = 10
            episode_frames = []

            for w_start in range(n_sampled - WINDOW):
                window_idxs = sampled_indices[w_start: w_start + WINDOW + 1]

                significant_window = False
                for k in range(WINDOW):
                    i      = window_idxs[k]
                    i_next = window_idxs[k + 1]
                    dp = cart_pos[i_next] - cart_pos[i]
                    dr = delta_euler(cart_rot[i], cart_rot[i_next])
                    if is_significant(dp, dr):
                        significant_window = True
                        break

                # remove noop
                if not significant_window:
                    continue

                i      = window_idxs[0]
                i_next = window_idxs[1]

                delta_pos  = cart_pos[i_next] - cart_pos[i]
                delta_rot  = delta_euler(cart_rot[i], cart_rot[i_next])
                euler      = quat_to_euler(cart_rot[i])
                gripper    = gripper_arr[i]
                gripper_action = 1 if gripper < 0.1 else -1

                state  = np.concatenate([cart_pos[i], euler,     [gripper]       ]).astype(np.float32)
                action = np.concatenate([delta_pos,   delta_rot, [gripper_action]]).astype(np.float32)

                episode_frames.append({
                    "image":       front_imgs[i],
                    "wrist_image": wrist_imgs[i],
                    "state":       state,
                    "actions":     action,
                    "task":        task_prompt,
                })

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
