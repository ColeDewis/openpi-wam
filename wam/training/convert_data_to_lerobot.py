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

POS_THRESHOLD = 0.0015
ROT_THRESHOLD = 0.0005

def find_nearest_index(timestamps, target_time):
    """Return index of the timestamp closest to target_time."""
    return int(np.argmin(np.abs(timestamps - target_time)))

def quats_to_euler(quats):
    """Quaternion [w, x, y, z] → Euler (roll, pitch, yaw) in radians."""
    quats_xyzw = np.roll(quats, -1, axis=-1)
    
    euler = R.from_quat(quats_xyzw).as_euler('xyz', degrees=False)  # (N,3)
    return np.unwrap(euler, axis=0)


def delta_rotvec(rot_current, rot_next):
    """Minimal rotation delta between two quaternions, as an axis-angle (rotation) vector."""
    curr_xyzw = np.roll(rot_current, -1, axis=-1)
    next_xyzw = np.roll(rot_next, -1, axis=-1)
    r_delta = R.from_quat(curr_xyzw).inv() * R.from_quat(next_xyzw)
    return r_delta.as_rotvec()  # shape (3,), axis * angle_rad


def is_noop(delta_pos, delta_rot, gripper_action, prev_gripper_action):
    """
    port from      https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py   """
    motion_is_noop = (np.linalg.norm(delta_pos) < POS_THRESHOLD and
                       np.linalg.norm(delta_rot) < ROT_THRESHOLD)
 
    if prev_gripper_action is None:
        return motion_is_noop
 
    return motion_is_noop and (gripper_action == prev_gripper_action)


def zero_noop_axes(delta_pos, delta_rot):
    """Zero out individual axes of a delta that don't clear their threshold"""
    delta_pos = np.where(np.abs(delta_pos) > POS_THRESHOLD, delta_pos, 0.0)
    delta_rot = np.where(np.abs(delta_rot) > ROT_THRESHOLD, delta_rot, 0.0)
    return delta_pos, delta_rot


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
            timestamps  = src_h5["low_dim/timestamp_ns"][:]
            front_imgs  = src_h5["front_image"][:]
            wrist_imgs  = src_h5["wrist_image"][:]
            cart_pos    = src_h5["low_dim/follower_cart_pos"][:]
            cart_rot    = src_h5["low_dim/follower_cart_rot"][:]
            gripper_arr = src_h5["low_dim/gripper_pos"][:]

            eulers = quats_to_euler(cart_rot)

            t_start = timestamps[0]
            t_end   = timestamps[-1]

            step_ns = 100_000_000
            target_times = np.arange(t_start, t_end, step_ns) # 0.1s timestep for a 10fps dataset

            sampled_indices = [find_nearest_index(timestamps, t) for t in target_times]
            sampled_indices = list(dict.fromkeys(sampled_indices))
            n_sampled = len(sampled_indices)
            episode_frames = []

            total_steps = max(n_sampled - 1, 0)
            noop_count = 0
            prev_gripper_action = None

            for idx in range(n_sampled - 1):
                i      = sampled_indices[idx]
                i_next = sampled_indices[idx + 1]
 
                delta_pos = cart_pos[i_next] - cart_pos[i]
                delta_rot = delta_rotvec(cart_rot[i], cart_rot[i_next])
 
                gripper = gripper_arr[i]
                # NOTE: in libero close is 1 and -1 is open
                gripper_action = 1 if gripper > -0.01 else -1 # assuming gripper close is around 0 and open is around -0.04
 
                if is_noop(delta_pos, delta_rot, gripper_action, prev_gripper_action):
                    noop_count += 1
                    continue
 
                # zero out individual axes of the kept delta that are near zero
                delta_pos, delta_rot = zero_noop_axes(delta_pos, delta_rot)
 
                euler = eulers[i]
                state  = np.concatenate([cart_pos[i], euler,     [gripper], [-gripper]]).astype(np.float32)
                action = np.concatenate([delta_pos,   delta_rot, [gripper_action]]).astype(np.float32)
 
                episode_frames.append({
                    "image":       front_imgs[i],
                    "wrist_image": wrist_imgs[i],
                    "state":       state,
                    "actions":     action,
                    "task":        task_prompt,
                })
 
                prev_gripper_action = gripper_action

            if total_steps > 0:
                noop_percentage = (noop_count / total_steps) * 100
                print(f"[{split['file']}] '{task_prompt}' | Noops: {noop_count}/{total_steps} steps ({noop_percentage:.2f}%) removed.")
            else:
                print(f"[{split['file']}] '{task_prompt}' | Episode too short to form a single step.")


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
