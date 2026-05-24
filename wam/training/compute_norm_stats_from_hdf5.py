import h5py
import numpy as np
import json
import argparse
from pathlib import Path

def compute_norm_stats(input_dir, output_json, lookahead_steps):
    print(f"going through all episodes in {input_dir}...")
    
    all_states = []
    all_actions = []
    
    for hdf5_file in Path(input_dir).glob("*.hdf5"):
        with h5py.File(hdf5_file, 'r') as f:
            jp = f['follower_state/jp'][:]
            gripper = f['follower_state/gripper'][:]
            
            gripper = np.expand_dims(gripper, axis=-1)
            states = np.concatenate([jp, gripper], axis=-1)
            
        # Calculate actions per-episode to avoid bad boundaries across files
        actions = states[lookahead_steps:] - states[:-lookahead_steps]
        
        all_states.append(states)
        all_actions.append(actions)
        
    # Combine all episodes into global arrays
    global_states = np.concatenate(all_states, axis=0)
    global_actions = np.concatenate(all_actions, axis=0)
        
    print(f"Combined states array with shape: {global_states.shape}")
    print(f"Combined actions array with shape: {global_actions.shape} (lookahead={lookahead_steps})")

    def get_stats(array_data):
        return {
            "mean": np.mean(array_data, axis=0).tolist(),
            "std":  np.std(array_data, axis=0).tolist(),
            "q01":  np.percentile(array_data, 1, axis=0).tolist(),
            "q99":  np.percentile(array_data, 99, axis=0).tolist(),
        }

    openpi_norm_file = {
        "norm_stats": {
            "state": get_stats(global_states),
            "actions": get_stats(global_actions)
        }
    }

    with open(output_json, "w") as f_out:
        json.dump(openpi_norm_file, f_out, indent=2)
        
    print(f"Successfully saved global normalization stats to {output_json}")

if __name__ == "__main__":
    # You can run this directly from the command line, or just edit these variables
    # Example: python compute_wam_stats.py --input wam_recording.npz --lookahead 5
    parser = argparse.ArgumentParser(description="Compute OpenPI norm stats from a continuous WAM recording.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .hdf5 file")
    parser.add_argument("--output", type=str, default="wam_norm_stats.json", help="Path to save the .json file")
    parser.add_argument("--lookahead", type=int, default=1, help="Number of timesteps in the future to define an action")
    
    args = parser.parse_args()
    
    compute_norm_stats(args.input, args.output, args.lookahead)
