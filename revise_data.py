#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY DATA PATCHER
--------------------------------
Functionality:
1. Iterates through specific subdirectories: training, validation, test.
2. Finds all .npz files in each directory.
3. Extracts 'states' and 'controls' arrays.
4. Revises 'controls' by dividing by dt (0.1).
5. Overwrites the original .npz file with the corrected data.
"""

import glob
import os

import numpy as np


def patch_hexapod_controls(base_dir="hexapod_data", dt=0.1):
    subdirs = ["training", "validation", "test"]
    total_files_processed = 0

    for subdir in subdirs:
        target_dir = os.path.join(base_dir, subdir)

        # Check if directory exists
        if not os.path.exists(target_dir):
            print(f"Directory not found: {target_dir}. Skipping...")
            continue

        # Find all .npz files in the current subdirectory
        files = glob.glob(os.path.join(target_dir, "*.npz"))

        if not files:
            print(f"No .npz files found in {target_dir}.")
            continue

        print(f"--- Processing {len(files)} files in {target_dir} ---")

        for file_path in files:
            try:
                # 1. Load the archive
                archive = np.load(file_path)

                # Ensure the required keys exist before modifying
                if "states" not in archive.files or "controls" not in archive.files:
                    print(
                        f"  [Warning] Missing 'states' or 'controls' in {file_path}. Skipping..."
                    )
                    continue

                states_data = archive["states"]
                controls_data = archive["controls"]

                # 2. Revise the controls data
                revised_controls = controls_data / dt

                # 3. Save the modified data back (overwriting the original file)
                # Using savez_compressed to maintain the same format as the collector
                np.savez(file_path, states=states_data, controls=revised_controls)

                total_files_processed += 1

            except Exception as e:
                print(f"  [Error] Failed to process {file_path}: {e}")

    print(
        f"\nPatching complete! Successfully revised controls in {total_files_processed} files."
    )


if __name__ == "__main__":
    # Ensure this matches your workspace directory structure
    patch_hexapod_controls(base_dir="hexapod_data", dt=0.1)
