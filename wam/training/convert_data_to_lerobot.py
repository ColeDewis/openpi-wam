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

POS_THRESHOLD = 0.01
ROT_THRESHOLD = 0.1
GRIP_THRESHOLD = 0.1

def find_nearest_index(timestamps, target_time):
    """Return index of the timestamp closest to target_time."""
    return int(np.argmin(np.abs(timestamps - target_time)))

def quat_to_euler(quat):
    """Quaternion [w, x, y, z] → Euler (roll, pitch, yaw) in radians."""
    quat_xyzw = np.roll(quat, -1, axis=-1)
    
    return R.from_quat(quat_xyzw).as_euler('xyz', degrees=False)

def delta_euler(rot_current, rot_next):
    """Minimal rotation delta between two quaternions, expressed as Euler xyz."""
    curr_xyzw = np.roll(rot_current, -1, axis=-1)
    next_xyzw = np.roll(rot_next, -1, axis=-1)
    
    r_delta = R.from_quat(curr_xyzw).inv() * R.from_quat(next_xyzw)
    return r_delta.as_euler('xyz', degrees=False)

def is_significant(delta_pos, delta_rot, delta_grip):
    """True if any axis of pos OR rot exceeds its threshold."""
    return (np.any(np.abs(delta_pos) > POS_THRESHOLD) or
            np.any(np.abs(delta_rot) > ROT_THRESHOLD) or
            np.any(np.abs(delta_grip) > GRIP_THRESHOLD))


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
            timestamps  = src_h5["timestamp_ns"][:]
            front_imgs  = src_h5["front_image"][:]
            wrist_imgs  = src_h5["wrist_image"][:]
            cart_pos    = src_h5["cart_pos"][:]
            cart_rot    = src_h5["cart_rot"][:]
            gripper_arr = src_h5["gripper_pos"][:]

            t_start = timestamps[0]
            t_end   = timestamps[-1]

            step_ns = 100_000_000
            target_times = np.arange(t_start, t_end, step_ns) # 0.1s timestep for a 10fps dataset

            sampled_indices = [find_nearest_index(timestamps, t) for t in target_times]
            sampled_indices = list(dict.fromkeys(sampled_indices))
            n_sampled = len(sampled_indices)

            WINDOW = 10
            episode_frames = []

            total_windows = n_sampled - WINDOW
            noop_count = 0

            for w_start in range(n_sampled - WINDOW):
                window_idxs = sampled_indices[w_start: w_start + WINDOW + 1]

                significant_window = False
                for k in range(WINDOW):
                    i      = window_idxs[k]
                    i_next = window_idxs[k + 1]
                    dp = cart_pos[i_next] - cart_pos[i]
                    dr = delta_euler(cart_rot[i], cart_rot[i_next])
                    dg = gripper_arr[i_next] - gripper_arr[i]
                    if is_significant(dp, dr, dg):
                        significant_window = True
                        break

                # remove noop
                if not significant_window:
                    noop_count += 1
                    continue

                i      = window_idxs[0]
                i_next = window_idxs[1]

                delta_pos  = cart_pos[i_next] - cart_pos[i]
                delta_rot  = delta_euler(cart_rot[i], cart_rot[i_next])
                euler      = quat_to_euler(cart_rot[i])
                gripper    = gripper_arr[i]
                # NOTE: in libero close is 1 and -1 is open
                gripper_action = 1 if gripper > -0.01 else -1 # assuming gripper close is around 0 and open is around -0.04

                state  = np.concatenate([cart_pos[i], euler,     [gripper], [0]]).astype(np.float32)
                action = np.concatenate([delta_pos,   delta_rot, [gripper_action]]).astype(np.float32)
                print(action)

                episode_frames.append({
                    "image":       front_imgs[i],
                    "wrist_image": wrist_imgs[i],
                    "state":       state,
                    "actions":     action,
                    "task":        task_prompt,
                })

            if total_windows > 0:
                noop_percentage = (noop_count / total_windows) * 100
                print(f"[{split['file']}] '{task_prompt}' | Noops: {noop_count}/{total_windows} windows ({noop_percentage:.2f}%) removed.")
            else:
                print(f"[{split['file']}] '{task_prompt}' | Episode too short to form a single {WINDOW}-frame window.")

            for frame in episode_frames:
                dataset.add_frame(frame)

            dataset.save_episode()

    print(f"Pushing to HuggingFace Hub as {save_name} ...")
    # make sure to hf auth login
    # dataset.push_to_hub(
    #     repo_id=save_name,
    #     private=False,
    # )
    print(f"https://huggingface.co/datasets/{save_name}")


if __name__ == "__main__":
    tyro.cli(main)
