import h5py
import time
import os
import numpy as np
from termcolor import cprint
import threading

class HDF5Recorder:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.clear()

    def clear(self):
        """Reset the data buffer for a new episode."""
        self.data_dict = {
            "images/wrist": [],
            "images/front": [],
            "state/joints": []
        }

    def add_step(self, wrist_img, front_img, joints):
        with self.lock:
            self.data_dict["images/wrist"].append(wrist_img)
            self.data_dict["images/front"].append(front_img)
            self.data_dict["state/joints"].append(joints)

    def save_episode(self, episode_name: str):
        with self.lock:
            data_to_save = self.data_dict
            self.clear()
            
        filepath = os.path.join(self.save_dir, f"{episode_name}.hdf5")
        cprint(f"\n[RECORDER] Saving episode with {len(data_to_save['state/joints'])} steps to {filepath}...", "yellow")
        
        with h5py.File(filepath, 'w') as f:
            for key, data in data_to_save.items():
                f.create_dataset(key, data=np.array(data), compression="gzip")
        
        cprint("[RECORDER] Episode saved successfully!", "green")
        self.clear()