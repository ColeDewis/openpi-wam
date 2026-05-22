import numpy as np
import json
import argparse

def compute_norm_stats(npz_file, output_json, lookahead_steps):
    print(f"Loading data from {npz_file}...")
    data = np.load(npz_file)
    states = data
    
    print(f"Loaded states array with shape: {states.shape}")

    # We slice the array to subtract the current state from the future state.
    actions = states[lookahead_steps:] - states[:-lookahead_steps]
    
    print(f"Computed actions with shape: {actions.shape} (lookahead={lookahead_steps})")

    def get_stats(array_data):
        return {
            "mean": np.mean(array_data, axis=0).tolist(),
            "std":  np.std(array_data, axis=0).tolist(),
            "q01":  np.percentile(array_data, 1, axis=0).tolist(),
            "q99":  np.percentile(array_data, 99, axis=0).tolist(),
        }

    openpi_norm_file = {
        "norm_stats": {
            "state": get_stats(states),
            "actions": get_stats(actions)
        }
    }

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
