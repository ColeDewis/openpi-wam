import h5py
import numpy as np

"""

Script to inspect an hdf5 file recorded by openpi tingy, and ensure you have the correct data and shapes.

Usage: uv run python scripts/inspect_hdf5.py

"""


def inspect_hdf5(file_path):
    print(f"\n{'='*50}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*50}\n")

    try:
        with h5py.File(file_path, "r") as f:
            # 1. Print Global Metadata
            print("--- Metadata (Attributes) ---")
            for attr_name, attr_value in f.attrs.items():
                print(f"  {attr_name}: {attr_value}")
            print("")

            # 2. Recursive function to walk the tree
            def print_structure(name, obj):
                indent = "  " * name.count("/")
                if isinstance(obj, h5py.Dataset):
                    # Show shape and dtype for data
                    print(
                        f"{indent}📄 {name.split('/')[-1]} | Shape: {obj.shape} | Type: {obj.dtype}"
                    )
                elif isinstance(obj, h5py.Group):
                    # Just show the group name
                    print(f"{indent}📂 {name.split('/')[-1]}/")

            print("--- File Structure ---")
            f.visititems(print_structure)

            # 3. Quick Sample (Low-Dim data check)
            if "observations" in f:
                print("\n--- Low-Dim Sample (First Frame) ---")
                # Limit float precision for cleaner output
                np.set_printoptions(precision=4, suppress=True)

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
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/episode_0.h5"
    inspect_hdf5(path)


if __name__ == "__main__":
    main()
