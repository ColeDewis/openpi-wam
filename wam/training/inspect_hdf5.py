import h5py
import numpy as np
import sys

"""
Script to inspect an hdf5 file recorded by openpi tingy, and ensure you have the correct data and shapes.

Usage: uv run python scripts/inspect_hdf5.py
"""


def inspect_hdf5(file_path):
    print(f"\n{'='*50}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*50}\n")

    # Limit float precision globally for cleaner output
    np.set_printoptions(precision=4, suppress=True)

    try:
        with h5py.File(file_path, "r") as f:
            # 1. Print Global Metadata
            print("--- Metadata (Attributes) ---")
            for attr_name, attr_value in f.attrs.items():
                print(f"  {attr_name}: {attr_value}")
            print("")

            # 2. Recursive function to walk the tree and print examples
            def print_structure(name, obj):
                indent = "  " * name.count("/")
                if isinstance(obj, h5py.Dataset):
                    # Safely extract a small sample for the example
                    try:
                        if obj.ndim == 0:
                            # Scalar value (e.g., a single integer/float/string)
                            sample = str(obj[()])
                        elif obj.size == 0:
                            sample = "Empty"
                        else:
                            # For N-dimensional arrays, grab the first element (e.g., first frame)
                            first_item = obj[0] if obj.ndim > 1 else obj[:]
                            
                            # Flatten it to stringify a clean preview of the first 4 elements
                            flat_data = np.array(first_item).flatten()
                            preview = flat_data[:10]
                            ellipsis = "..." if len(flat_data) > 4 else ""
                            
                            # Format nicely
                            sample = f"[{', '.join(map(str, preview))}{ellipsis}]"
                    except Exception:
                        sample = "Error reading data"

                    # Show shape, dtype, AND the data example
                    print(
                        f"{indent}📄 {name.split('/')[-1]} | Shape: {obj.shape} | Type: {obj.dtype} | Ex: {sample}"
                    )
                elif isinstance(obj, h5py.Group):
                    # Just show the group name
                    print(f"{indent}📂 {name.split('/')[-1]}/")

            print("--- File Structure & Data Examples ---")
            f.visititems(print_structure)

            # 3. Quick Sample (Low-Dim data check) 
            if "observations" in f:
                print("\n--- Low-Dim Sample (First Frame) ---")

                def print_leaf(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # Filter for low-dim data: 1D arrays or 2D arrays with small width (< 20)
                        # This skips images/pointclouds
                        if obj.ndim == 1 or (obj.ndim == 2 and obj.shape[1] < 20):
                            print(f"  {name}: {obj[0]}")

                # visititems walks down into 'joints', 'cartesian', etc. automatically
                f["observations"].visititems(print_leaf)

    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/episode_0.h5"
    inspect_hdf5(path)


if __name__ == "__main__":
    main()
