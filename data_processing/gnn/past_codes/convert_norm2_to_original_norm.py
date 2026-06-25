#!/usr/bin/env python
"""
Convert norm2 dataset to original normalization (include zeros in stats).
Only recalculates norm_stats - raw data remains unchanged.
"""

import torch
import numpy as np
import sys
from pathlib import Path

def convert_norm_stats(input_path, output_path=None):
    """
    Convert norm2 (exclude zeros) to original (include zeros) normalization stats.

    Args:
        input_path: Path to existing norm2 dataset
        output_path: Output path. If None, creates _original_norm version
    """
    input_path = Path(input_path)

    if output_path is None:
        # Create output path with _original_norm suffix
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_original_norm.pth"
    else:
        output_path = Path(output_path)

    print(f"Loading dataset: {input_path}")
    print(f"Output will be saved to: {output_path}")

    # Load existing dataset
    train_data = torch.load(input_path, weights_only=False)
    node_features = train_data['node_features'].numpy()

    print(f"\nDataset info:")
    print(f"  node_features shape: {node_features.shape}")
    print(f"  num_tasks: {train_data.get('num_tasks', 'N/A')}")
    print(f"  num_libs: {train_data.get('num_libs', 'N/A')}")

    # Get normalize indices and names
    include_parasitic_cap = train_data.get('include_parasitic_cap', False)
    if include_parasitic_cap:
        NORMALIZE_INDICES = [4, 5, 6, 10, 11]
        NORMALIZE_NAMES = ['voltage', 'input_slew', 'output_load', 'temperature', 'parasitic_cap']
    else:
        NORMALIZE_INDICES = [4, 5, 6, 10]
        NORMALIZE_NAMES = ['voltage', 'input_slew', 'output_load', 'temperature']

    # Show old norm_stats
    print("\n" + "="*60)
    print("Old norm_stats (norm2 - exclude zeros):")
    print("="*60)
    old_stats = train_data.get('norm_stats', {}).get('node_features', {})
    for name in NORMALIZE_NAMES:
        if name in old_stats:
            print(f"  {name}: mean={old_stats[name]['mean']:.6f}, std={old_stats[name]['std']:.6f}")

    # Recalculate with original method (include zeros)
    print("\n" + "="*60)
    print("Recalculating norm_stats (original - include zeros):")
    print("="*60)

    new_norm_stats = {}
    for idx, name in zip(NORMALIZE_INDICES, NORMALIZE_NAMES):
        feature_data = node_features[:, :, idx].flatten()

        # Original method: include all values including zeros
        mean = float(np.mean(feature_data))
        std = float(np.std(feature_data))
        if std == 0:
            std = 1.0

        new_norm_stats[name] = {'mean': mean, 'std': std}

        # Calculate non-zero ratio for info
        nonzero_count = np.sum(feature_data != 0)
        total_count = len(feature_data)
        nonzero_ratio = nonzero_count / total_count * 100

        print(f"  {name}: mean={mean:.6f}, std={std:.6f} (non-zero: {nonzero_ratio:.1f}%)")

    # Show comparison
    print("\n" + "="*60)
    print("Comparison (old vs new):")
    print("="*60)
    for name in NORMALIZE_NAMES:
        if name in old_stats and name in new_norm_stats:
            old_mean = old_stats[name]['mean']
            old_std = old_stats[name]['std']
            new_mean = new_norm_stats[name]['mean']
            new_std = new_norm_stats[name]['std']
            mean_diff = new_mean - old_mean
            std_diff = new_std - old_std
            print(f"  {name}:")
            print(f"    mean: {old_mean:.6f} -> {new_mean:.6f} (diff: {mean_diff:+.6f})")
            print(f"    std:  {old_std:.6f} -> {new_std:.6f} (diff: {std_diff:+.6f})")

    # Update train_data
    train_data['norm_stats']['node_features'] = new_norm_stats
    train_data['normalize_nonzero_only'] = False
    train_data['include_zeros_in_norm'] = True

    # Save updated dataset
    print(f"\nSaving to: {output_path}")
    torch.save(train_data, output_path)

    # Verify saved file
    file_size = output_path.stat().st_size / (1024**3)  # GB
    print(f"Saved successfully! File size: {file_size:.2f} GB")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"\nNote: Test data does not need modification.")
    print(f"The training script will use norm_stats from this file for both train and test.")

if __name__ == "__main__":
    # Default paths
    default_input = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN_unified_norm2/train_cell_stage_aware.pth"

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = default_input

    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_norm_stats(input_path, output_path)
