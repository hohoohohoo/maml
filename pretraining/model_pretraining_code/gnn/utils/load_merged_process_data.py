"""
Load merged process-aware datasets

This module provides functions to load pre-merged datasets for efficient training.
"""

import os
import torch


def load_merged_process_data(data_type='cell', graph_mode='stage_aware', base_path=None):
    """
    Load pre-merged process-aware dataset

    Args:
        data_type: 'cell' or 'transition'
        graph_mode: 'stage_aware' or 'full_graph'
        base_path: Base directory (default: dataset_temp_process)

    Returns:
        minimal_data_per_file: Merged minimal data
        topology_cache: Shared topology cache
        norm_stats: None (normalization done in training script)
    """

    if base_path is None:
        base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_ASAP7"

    merged_filename = f"merged_{data_type}_{graph_mode}.pth"
    merged_path = os.path.join(base_path, merged_filename)

    print(f"\n📂 Loading merged process-aware dataset")
    print(f"   Path: {merged_path}")

    if not os.path.exists(merged_path):
        raise FileNotFoundError(
            f"Merged dataset not found: {merged_path}\n"
            f"Please run: python merge_process_datasets.py --data_type {data_type} --graph_mode {graph_mode}"
        )

    # Load merged dataset
    print(f"   📥 Loading merged file...")
    merged_data = torch.load(merged_path, weights_only=False, map_location='cpu')

    minimal_data_per_file = merged_data['minimal_data_per_file']
    cache_path = merged_data.get('cache_path', None)
    metadata = merged_data.get('metadata', {})

    print(f"   ✅ Loaded merged dataset")
    print(f"   📊 Metadata:")
    print(f"      Datasets merged: {metadata.get('num_datasets', 'N/A')}")
    print(f"      Total samples: {len(minimal_data_per_file[0])}")
    print(f"      Lib files per task: {len(minimal_data_per_file)}")

    # Load topology cache
    topology_cache = None
    if cache_path:
        # Convert to absolute path if needed
        if not os.path.isabs(cache_path):
            cache_filename = os.path.basename(cache_path)
            cache_file = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"
            if os.path.exists(cache_file):
                cache_path = cache_file

        if os.path.exists(cache_path):
            print(f"   📥 Loading topology cache: {cache_path}")
            topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu')
            print(f"   ✓ Loaded topology cache for {len(topology_cache)} cells")
        else:
            print(f"   ⚠️  Warning: Topology cache not found: {cache_path}")

    # Return None for norm_stats (will be computed in training script)
    return minimal_data_per_file, topology_cache, None
