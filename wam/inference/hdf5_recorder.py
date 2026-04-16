import h5py
import numpy as np
import os
from collections import defaultdict


class HDF5Recorder:
    def __init__(self, save_dir="./dataset"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.episode_data = defaultdict(list)

    def clear(self):
        self.episode_data.clear()

    def add_step(self, data_dict):
        """Appends the data. Handles nesting by keeping the dict structure."""
        for key, value in data_dict.items():
            self.episode_data[key].append(value)

    def _save_recursive(self, h5_group, key, data_list):
        """Recursively saves data, creating groups for nested dictionaries."""
        # Check if the first item in the list is a dictionary
        if isinstance(data_list[0], dict):
            sub_group = h5_group.create_group(key)
            # Find all keys present in the sub-dictionaries
            sub_keys = data_list[0].keys()
            for sk in sub_keys:
                # Extract the list of values for this specific sub-key
                sub_data_list = [d[sk] for d in data_list]
                self._save_recursive(sub_group, sk, sub_data_list)
        else:
            # Base case: it's a list of arrays/numbers, so save as dataset
            data = np.array(data_list)
            h5_group.create_dataset(key, data=data, compression="gzip")

    def save_episode(self, episode_name, metadata):
        if not self.episode_data:
            return

        filepath = os.path.join(self.save_dir, f"{episode_name}.hdf5")
        with h5py.File(filepath, "w") as f:
            # 1. Save data using the recursive helper
            for key, data_list in self.episode_data.items():
                self._save_recursive(f, key, data_list)

            # 2. Save metadata as top-level attributes
            for meta_key, meta_val in metadata.items():
                f.attrs[meta_key] = meta_val

        print(f"[RECORDER] Saved {episode_name} to {filepath}")
        self.clear()
