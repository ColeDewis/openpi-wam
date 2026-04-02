import numpy as np
import json
import argparse

def compute_norm_stats(npz_file, output_json, lookahead_steps):
    # 1. Load the raw states from the .npz file
    print(f"Loading data from {npz_file}...")
    data = np.load(npz_file)
    # .npz files store arrays in a dictionary format. 
    # If you didn't name the array when saving, NumPy defaults to 'arr_0'.
    # We will just grab the first array in the file automatically.
    # array_key = list(data.keys())[0]
    # states = data[array_key]
    states = data
    
    print(f"Loaded states array with shape: {states.shape}")

    # 2. Calculate Actions (State_{t+k} - State_t)
    # We slice the array to subtract the current state from the future state.
    # Note: This naturally drops the very last 'k' frames because they have 
    # no future frame to calculate a delta against. This is mathematically correct!
    actions = states[lookahead_steps:] - states[:-lookahead_steps]
    
    print(f"Computed actions with shape: {actions.shape} (lookahead={lookahead_steps})")

    # 3. Helper function to compute OpenPI statistics
    def get_stats(array_data):
        return {
            "mean": np.mean(array_data, axis=0).tolist(),
            "std":  np.std(array_data, axis=0).tolist(),
            "q01":  np.percentile(array_data, 1, axis=0).tolist(),
            "q99":  np.percentile(array_data, 99, axis=0).tolist(),
        }

    # 4. Build the nested dictionary exactly how OpenPI expects it
    openpi_norm_file = {
        "norm_stats": {
            "state": get_stats(states),
            "actions": get_stats(actions)
        }
    }

    # 5. Save it to disk
    with open(output_json, "w") as f:
        json.dump(openpi_norm_file, f, indent=2)
        
    print(f"Successfully saved normalization stats to {output_json}")

if __name__ == "__main__":
    # You can run this directly from the command line, or just edit these variables
    # Example: python compute_wam_stats.py --input wam_recording.npz --lookahead 5
    parser = argparse.ArgumentParser(description="Compute OpenPI norm stats from a continuous WAM recording.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .npz file")
    parser.add_argument("--output", type=str, default="wam_norm_stats.json", help="Path to save the .json file")
    parser.add_argument("--lookahead", type=int, default=1, help="Number of timesteps in the future to define an action")
    
    args = parser.parse_args()
    
    compute_norm_stats(args.input, args.output, args.lookahead)