#!/usr/bin/env python
"""
Convert vdd_only dataset with hybrid normalization:
- voltage, temperature: non-zero values only (norm2 method) - sparse features
- input_slew, output_load: include zeros (original method) - dense features
"""

import torch
import numpy as np
import sys
from pathlib import Path

def convert_hybrid_norm_stats(input_path, output_path=None):
    """
    Convert to hybrid normalization stats.

    - voltage, temperature: exclude zeros (sparse, meaningful zeros)
    - input_slew, output_load: include zeros (dense or zeros are actual values)
    """
    input_path = Path(input_path)

    if output_path is None:
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}_hybrid_norm.pth"
    else:
        output_path = Path(output_path)

    print(f"Loading dataset: {input_path}")
    print(f"Output will be saved to: {output_path}")

    # Load existing dataset
    train_data = torch.load(input_path, weights_only=False)
    node_features = train_data['node_features'].numpy()

    print(f"\nDataset info:")
    print(f"  node_features shape: {node_features.shape}")
    print(f"  voltage_mode: {train_data.get('voltage_mode', 'N/A')}")

    # Define normalization strategy per feature
    # Index: 4=voltage, 5=input_slew, 6=output_load, 10=temperature
    FEATURE_CONFIG = {
        4: {'name': 'voltage', 'include_zeros': False},      # Non-zero only (sparse in vdd_only)
        5: {'name': 'input_slew', 'include_zeros': True},    # Include zeros
        6: {'name': 'output_load', 'include_zeros': True},   # Include zeros
        10: {'name': 'temperature', 'include_zeros': False}, # Non-zero only (MOS nodes only)
    }

    # Show old norm_stats
    print("\n" + "="*60)
    print("Old norm_stats:")
    print("="*60)
    old_stats = train_data.get('norm_stats', {}).get('node_features', {})
    for idx, config in FEATURE_CONFIG.items():
        name = config['name']
        if name in old_stats:
            print(f"  {name}: mean={old_stats[name]['mean']:.6f}, std={old_stats[name]['std']:.6f}")

    # Recalculate with hybrid method
    print("\n" + "="*60)
    print("Recalculating norm_stats (hybrid):")
    print("  - voltage, temperature: non-zero only")
    print("  - input_slew, output_load: include zeros")
    print("="*60)

    new_norm_stats = {}
    for idx, config in FEATURE_CONFIG.items():
        name = config['name']
        include_zeros = config['include_zeros']
        feature_data = node_features[:, :, idx].flatten()

        if include_zeros:
            # Include zeros in stats
            mean = float(np.mean(feature_data))
            std = float(np.std(feature_data))
            method = "include zeros"
        else:
            # Non-zero only
            nonzero_data = feature_data[feature_data != 0]
            if len(nonzero_data) > 0:
                mean = float(np.mean(nonzero_data))
                std = float(np.std(nonzero_data))
            else:
                mean = 0.0
                std = 1.0
            method = "non-zero only"

        if std == 0:
            std = 1.0

        new_norm_stats[name] = {'mean': mean, 'std': std}

        # Calculate non-zero ratio for info
        nonzero_count = np.sum(feature_data != 0)
        total_count = len(feature_data)
        nonzero_ratio = nonzero_count / total_count * 100

        print(f"  {name}: mean={mean:.6f}, std={std:.6f} [{method}] (non-zero: {nonzero_ratio:.1f}%)")

    # Show comparison
    print("\n" + "="*60)
    print("Comparison (old vs new):")
    print("="*60)
    for idx, config in FEATURE_CONFIG.items():
        name = config['name']
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
    train_data['normalize_nonzero_only'] = 'hybrid'  # Mark as hybrid
    train_data['norm_method'] = {
        'voltage': 'nonzero_only',
        'input_slew': 'include_zeros',
        'output_load': 'include_zeros',
        'temperature': 'nonzero_only'
    }

    # Save updated dataset
    print(f"\nSaving to: {output_path}")
    torch.save(train_data, output_path)

    # Verify saved file
    file_size = output_path.stat().st_size / (1024**3)  # GB
    print(f"Saved successfully! File size: {file_size:.2f} GB")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    # Default path - vdd_only dataset
    default_input = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN_unified/train_cell_stage_aware_vdd_only.pth"

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = default_input

    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert_hybrid_norm_stats(input_path, output_path)
