#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY DATA PATCHER
--------------------------------
Functionality:
1. Iterates through subdirectories: training, validation, test + models/test_traj.npz
2. Corrects the off-by-one alignment bug:
   Original: (s_i, c_i) where c_i = (s_i - s_{i-1})/dt  [c arrived AT s_i]
   Fixed:    (s_i, c_{i+1}) where c_{i+1} transitions s_i -> s_{i+1}
   Implementation: new_states = states[:-1], new_controls = controls[1:]
"""

import glob
import os

import numpy as np


def patch_file(file_path):
    try:
        archive = np.load(file_path)

        if "states" not in archive.files or "controls" not in archive.files:
            print(f"  [Warning] Missing keys in {file_path}. Skipping...")
            return False

        states = archive["states"]
        controls = archive["controls"]

        # Shift: drop states[-1] (no future action) and controls[0] (pre-recording garbage)
        new_states = states[:-1]
        new_controls = controls[1:]

        assert len(new_states) == len(new_controls), "Length mismatch after patch!"

        np.savez(file_path, states=new_states, controls=new_controls)
        print(f"  [OK] Patched {file_path}: {len(states)} -> {len(new_states)} steps.")
        return True

    except Exception as e:
        print(f"  [Error] Failed to process {file_path}: {e}")
        return False


def patch_hexapod_controls(base_dir="hexapod_data"):
    total = 0

    # 1. Patch training/validation/test subdirectories
    for subdir in ["training", "validation", "test"]:
        target_dir = os.path.join(base_dir, subdir)

        if not os.path.exists(target_dir):
            print(f"Directory not found: {target_dir}. Skipping...")
            continue

        files = glob.glob(os.path.join(target_dir, "*.npz"))
        if not files:
            print(f"No .npz files found in {target_dir}.")
            continue

        print(f"--- Processing {len(files)} files in {target_dir} ---")
        for file_path in files:
            if patch_file(file_path):
                total += 1

    # 2. Patch models/test_traj.npz
    test_traj_path = os.path.join("models", "test_traj.npz")
    if os.path.exists(test_traj_path):
        print(f"--- Processing {test_traj_path} ---")
        if patch_file(test_traj_path):
            total += 1
    else:
        print(f"[Warning] {test_traj_path} not found. Skipping...")

    print(f"\nPatching complete! Successfully patched {total} files.")


if __name__ == "__main__":
    patch_hexapod_controls(base_dir="hexapod_data")
