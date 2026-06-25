#!/usr/bin/env python
"""
Split TSMC GNN dataset with unified preprocessing pipeline

TSMC dataset structure:
- 3 corners: TT, FF, SS
- 5 temperatures: 0, 25, 50, 75, 100
- 15 total configurations: TSMC_{corner}_{temperature}
  (e.g., TSMC_TT_0, TSMC_FF_25, TSMC_SS_100)
- Each folder contains: graph_data/cell_all_graph_data_{graph_mode}.pth
- Different corners/temperatures are different dataset groups (not merged)

This script:
1. Validates that each configuration folder has the expected data file
2. Processes each folder independently (not merged)
3. Creates train/test split for each folder separately

Usage:
    # Process all available TSMC folders (15 configurations):
    python split_gnn_dataset_tsmc.py --run_all --graph_mode stage_aware

    # Process a single folder using folder name:
    python split_gnn_dataset_tsmc.py --folder TSMC_FF_0 --graph_mode full_graph

    # Process using corner and temperature:
    python split_gnn_dataset_tsmc.py --corner FF --temperature 0 --graph_mode stage_aware
"""

import torch
import os
import sys
from pathlib import Path
import random
import numpy as np
import gc

# Add path for preprocessing utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pretraining', 'model_pretraining_code', 'gnn', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pretraining', 'model_test_code', 'gnn', 'utils'))

from gnn_data_preprocessing_utils import (
    preprocess_gnn_minimal_data,
    validate_gnn_data,
    calculate_norm_stats_from_minimal_data_safe
)


def check_output_continuity(data, threshold_ratio=0.18):
    """Check if output data is continuous"""
    if len(data) < 2:
        return True, 1.0, [], 0, 0

    data_flat = data.flatten()
    diffs = np.abs(np.diff(data_flat))
    data_range = data_flat.max() - data_flat.min()

    if data_range == 0:
        return True, 1.0, [], 0, 0

    threshold = threshold_ratio * data_range
    gaps = np.where(diffs > threshold)[0]
    max_jump = diffs.max() if len(diffs) > 0 else 0
    max_ratio = max_jump / data_range if data_range > 0 else 0

    score = 1.0 - len(gaps) / max(len(diffs), 1)
    is_continuous = len(gaps) == 0

    return is_continuous, score, gaps.tolist(), max_jump, max_ratio


def analyze_gnn_continuity(minimal_data_per_file, threshold_ratio=0.18, max_check_samples=100000):
    """
    Analyze output continuity of GNN data (minimal format).
    """
    continuous_task_ids = []
    discontinuous_task_ids = []
    continuity_analysis = []

    num_libs = len(minimal_data_per_file)
    num_tasks = len(minimal_data_per_file[0])
    num_check_samples = min(max_check_samples, num_tasks)

    print(f"\n   Analyzing output continuity for first {num_check_samples} tasks...")
    print(f"   Libs: {num_libs}, Total tasks: {num_tasks}")

    for task_idx in range(num_check_samples):
        if task_idx % 10000 == 0 and task_idx > 0:
            print(f"   Progress: {task_idx}/{num_check_samples}")

        try:
            task_outputs = []
            for lib_idx in range(num_libs):
                sample = minimal_data_per_file[lib_idx][task_idx]
                task_outputs.append(sample['output'])

            task_outputs = np.array(task_outputs)
            output_continuous, output_score, output_gaps, _, output_max_ratio = check_output_continuity(
                task_outputs.reshape(-1, 1), threshold_ratio=threshold_ratio
            )

            continuity_analysis.append({
                'task_id': task_idx,
                'output_continuous': output_continuous,
                'output_score': output_score,
                'output_gaps': len(output_gaps),
                'output_max_ratio': output_max_ratio
            })

            if output_continuous:
                continuous_task_ids.append(task_idx)
            else:
                discontinuous_task_ids.append(task_idx)

        except Exception as e:
            if task_idx < 10:
                print(f"   Error processing task {task_idx}: {e}")
            continue

    print(f"\n   Continuity Analysis Complete!")
    print(f"   - Analyzed tasks: {len(continuity_analysis)}")
    print(f"   - Continuous: {len(continuous_task_ids)} ({len(continuous_task_ids)/max(len(continuity_analysis),1)*100:.1f}%)")
    print(f"   - Discontinuous: {len(discontinuous_task_ids)} ({len(discontinuous_task_ids)/max(len(continuity_analysis),1)*100:.1f}%)")

    return continuous_task_ids, discontinuous_task_ids, continuity_analysis


def convert_to_tensor_format(minimal_data_per_lib):
    """Convert minimal_data_per_file to tensor-based format."""
    num_libs = len(minimal_data_per_lib)
    num_tasks = len(minimal_data_per_lib[0])

    print(f"   Converting to tensor format: {num_libs} libs, {num_tasks} tasks")

    # First pass: collect metadata from lib 0
    node_counts = []
    cell_names = []
    delay_types = []
    output_names = []

    for task_idx in range(num_tasks):
        sample = minimal_data_per_lib[0][task_idx]
        node_features = sample['node_features']

        if isinstance(node_features, torch.Tensor):
            node_counts.append(node_features.shape[0])
        else:
            node_counts.append(len(node_features))

        cell_names.append(sample.get('cell_name', f'task_{task_idx}'))
        delay_types.append(sample.get('delay_type', 'rise'))
        output_names.append(sample.get('output_name', ''))

    total_nodes = sum(node_counts)
    first_sample = minimal_data_per_lib[0][0]
    num_features = first_sample['node_features'].shape[1]

    print(f"   Total nodes: {total_nodes}, Features: {num_features}")

    # Create slices
    node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
    node_slices[1:] = np.cumsum(node_counts)

    # Allocate tensors
    all_node_features = np.zeros((num_libs, total_nodes, num_features), dtype=np.float32)
    all_outputs = np.zeros((num_libs, num_tasks), dtype=np.float32)

    # Second pass: fill tensors
    print(f"   Filling tensors...")
    mismatched_tasks = []

    for lib_idx in range(num_libs):
        if lib_idx % 10 == 0:
            print(f"   Processing lib {lib_idx}/{num_libs}...")

        for task_idx in range(num_tasks):
            sample = minimal_data_per_lib[lib_idx][task_idx]
            node_features = sample['node_features']

            if isinstance(node_features, torch.Tensor):
                node_features = node_features.cpu().numpy()

            node_start = node_slices[task_idx]
            node_end = node_slices[task_idx + 1]
            expected_nodes = node_end - node_start
            actual_nodes = node_features.shape[0]

            if actual_nodes != expected_nodes:
                if len(mismatched_tasks) < 10:
                    mismatched_tasks.append({
                        'lib_idx': lib_idx, 'task_idx': task_idx,
                        'cell_name': sample.get('cell_name', ''),
                        'expected': expected_nodes, 'actual': actual_nodes
                    })
                copy_nodes = min(actual_nodes, expected_nodes)
                all_node_features[lib_idx, node_start:node_start + copy_nodes, :] = node_features[:copy_nodes]
            else:
                all_node_features[lib_idx, node_start:node_end, :] = node_features

            output = sample['output']
            if isinstance(output, torch.Tensor):
                output = output.item()
            all_outputs[lib_idx, task_idx] = output

    if mismatched_tasks:
        print(f"\n   WARNING: Found {len(mismatched_tasks)}+ node count mismatches")

    print(f"   Conversion complete!")
    print(f"   node_features shape: {all_node_features.shape}")
    print(f"   outputs shape: {all_outputs.shape}")

    return {
        'node_features': torch.from_numpy(all_node_features),
        'outputs': torch.from_numpy(all_outputs),
        'node_slices': torch.from_numpy(node_slices),
        'node_counts': node_counts,
        'cell_names': cell_names,
        'delay_types': delay_types,
        'output_names': output_names,
        'num_libs': num_libs,
        'num_tasks': num_tasks,
        'num_features': num_features,
        'total_nodes': total_nodes,
    }


# TSMC corners and temperatures
TSMC_CORNERS = ['TT', 'FF', 'SS']
TSMC_TEMPERATURES = [0, 25, 50, 75, 100]

# Generate all 15 folder combinations
TSMC_ALL_FOLDERS = [f"TSMC_{corner}_{temp}" for corner in TSMC_CORNERS for temp in TSMC_TEMPERATURES]


def get_folder_name(corner, temperature):
    """Generate folder name from corner and temperature"""
    return f"TSMC_{corner}_{temperature}"


def validate_tsmc_datasets(base_path, graph_mode='stage_aware', data_type='cell'):
    """
    Validate TSMC dataset structure.

    Checks all 15 configurations (3 corners x 5 temperatures) for data file existence.

    Returns:
        Dict with validation results
    """
    results = {
        'valid_folders': [],
        'missing_folders': [],
        'missing_files': []
    }

    print(f"\nValidating TSMC datasets...")
    print(f"Base path: {base_path}")
    print(f"Graph mode: {graph_mode}")
    print(f"Data type: {data_type}")
    print(f"Checking {len(TSMC_ALL_FOLDERS)} configurations (3 corners x 5 temperatures)")
    print("-" * 50)

    for folder_name in TSMC_ALL_FOLDERS:
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            results['missing_folders'].append(folder_name)
            print(f"  {folder_name}: MISSING FOLDER")
            continue

        # Check for data file
        data_file = os.path.join(folder_path, "graph_data", f"{data_type}_all_graph_data_{graph_mode}.pth")

        if os.path.exists(data_file):
            file_size = os.path.getsize(data_file) / (1024**3)
            results['valid_folders'].append({
                'folder': folder_name,
                'path': folder_path,
                'data_file': data_file,
                'size_gb': file_size
            })
            print(f"  {folder_name}: OK ({file_size:.2f} GB)")
        else:
            results['missing_files'].append({
                'folder': folder_name,
                'expected_file': data_file
            })
            print(f"  {folder_name}: MISSING FILE ({graph_mode})")

    print("-" * 50)
    print(f"Valid: {len(results['valid_folders'])}/{len(TSMC_ALL_FOLDERS)}")

    return results


def split_tsmc_single_folder(folder_name, base_path=None, train_ratio=0.8, seed=42,
                              data_type='cell', graph_mode='stage_aware',
                              enable_filtering=True, min_std_threshold=1e-6,
                              enable_continuity_analysis=True,
                              continuity_threshold=0.15,
                              delete_source_files=False):
    """
    Split a single TSMC temperature folder dataset.

    Each temperature folder (TSMC_FF_0, TSMC_FF_25, etc.) is processed independently.

    Args:
        folder_name: Name of temperature folder (e.g., 'TSMC_FF_0')
        base_path: Path to TSMC dataset directory
        train_ratio: Ratio of training data (default 0.8)
        seed: Random seed
        data_type: 'cell' or 'transition'
        graph_mode: 'full_graph' or 'stage_aware'
        enable_filtering: Whether to filter invalid/low-variance tasks
        min_std_threshold: Minimum std threshold for output filtering
        enable_continuity_analysis: Whether to perform continuity analysis
        continuity_threshold: Threshold ratio for continuity check
        delete_source_files: Whether to delete source files after loading

    Returns:
        Dictionary with paths to train/test files and statistics
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Default base path
    if base_path is None:
        base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN"

    folder_path = os.path.join(base_path, folder_name)
    data_file = os.path.join(folder_path, "graph_data", f"{data_type}_all_graph_data_{graph_mode}.pth")

    print(f"\n{'='*80}")
    print(f"Split TSMC GNN Dataset: {folder_name}")
    print(f"{'='*80}")
    print(f"Folder path: {folder_path}")
    print(f"Data file: {data_file}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Train ratio: {train_ratio}")
    print(f"Filtering enabled: {enable_filtering}")
    print(f"Continuity analysis: {enable_continuity_analysis}")
    print(f"{'='*80}")

    # Validate data file exists
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        return None

    file_size = os.path.getsize(data_file) / (1024**3)
    print(f"Data file size: {file_size:.2f} GB")

    # ============================================================
    # Phase 1: Load data and split
    # ============================================================
    print(f"\n{'='*60}")
    print("Phase 1: Loading data and performing split")
    print(f"{'='*60}")

    print(f"\nLoading: {folder_name}")
    data = torch.load(data_file, weights_only=False)

    if 'minimal_data_per_file' not in data:
        print(f"ERROR: {folder_name} is not in minimal format")
        return None

    minimal_data = data['minimal_data_per_file']
    num_libs = len(minimal_data)
    num_samples = len(minimal_data[0])

    # Load topology cache
    topology_cache = None
    cache_path = data.get('cache_path', None)
    if cache_path:
        if not os.path.isabs(cache_path):
            cache_filename = os.path.basename(cache_path)
            cache_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"

        if os.path.exists(cache_path):
            print(f"Loading topology cache: {cache_path}")
            topology_cache = torch.load(cache_path, weights_only=False)
            print(f"Loaded {len(topology_cache)} cells")

    print(f"Samples: {num_samples}, Libs: {num_libs}")

    # Create sample indices and split
    sample_indices = list(range(num_samples))
    random.shuffle(sample_indices)

    train_size = int(num_samples * train_ratio)
    train_indices = sorted(sample_indices[:train_size])
    test_indices = sorted(sample_indices[train_size:])

    print(f"Split: {len(train_indices)} train, {len(test_indices)} test")

    # Extract train/test data
    train_data_per_lib = [[] for _ in range(num_libs)]
    test_data_per_lib = [[] for _ in range(num_libs)]

    for lib_idx in range(num_libs):
        lib_samples = minimal_data[lib_idx]
        for idx in train_indices:
            train_data_per_lib[lib_idx].append(lib_samples[idx])
        for idx in test_indices:
            test_data_per_lib[lib_idx].append(lib_samples[idx])

    total_train_samples = len(train_indices)
    total_test_samples = len(test_indices)

    # Clean up source data
    del data
    del minimal_data
    gc.collect()

    # Delete source file if requested
    if delete_source_files:
        try:
            os.remove(data_file)
            print(f"Deleted source file: {data_file}")
        except Exception as e:
            print(f"Warning: Failed to delete {data_file}: {e}")

    print(f"\n{'='*60}")
    print("Phase 1 Complete: Data Split")
    print(f"{'='*60}")
    print(f"Train samples: {total_train_samples}")
    print(f"Test samples: {total_test_samples}")
    print(f"Libs: {num_libs}")

    # ============================================================
    # Phase 2: Preprocessing TRAIN data
    # ============================================================
    print(f"\n{'='*60}")
    print("Phase 2: Preprocessing TRAIN data")
    print(f"{'='*60}")

    train_norm_stats = None
    train_filter_stats = None

    if enable_filtering:
        print("\nPreprocessing TRAIN data...")
        preprocessed_train, train_norm_stats, train_preprocessing_stats = preprocess_gnn_minimal_data(
            train_data_per_lib,
            min_std_threshold=min_std_threshold,
            enable_filtering=True,
            verbose=True
        )

        train_filter_stats = train_preprocessing_stats.get('filtering', {})
        print(f"  Train: {train_filter_stats.get('valid_tasks', 0)} valid tasks after filtering")

        train_data_per_lib = preprocessed_train
        total_train_samples = len(train_data_per_lib[0]) if train_data_per_lib else 0
    else:
        print("\nCalculating normalization statistics...")
        train_norm_stats = calculate_norm_stats_from_minimal_data_safe(
            train_data_per_lib, sample_rate=10
        )

    print(f"\n  Test data: {total_test_samples} samples (no filtering)")

    # ============================================================
    # Phase 3: Continuity analysis (TEST only)
    # ============================================================
    test_continuity_stats = None

    if enable_continuity_analysis and test_data_per_lib:
        print(f"\n{'='*60}")
        print("Phase 3: Continuity Analysis (TEST data)")
        print(f"{'='*60}")

        test_continuous, test_discontinuous, test_analysis = analyze_gnn_continuity(
            test_data_per_lib,
            threshold_ratio=continuity_threshold,
            max_check_samples=min(100000, total_test_samples)
        )

        test_continuity_stats = {
            'continuous_count': len(test_continuous),
            'discontinuous_count': len(test_discontinuous),
            'continuous_ratio': len(test_continuous) / max(len(test_analysis), 1) * 100,
            'continuous_task_ids': test_continuous[:1000],
            'total_analyzed': len(test_analysis)
        }

    # ============================================================
    # Phase 4: Convert to tensor format and save
    # ============================================================
    print(f"\n{'='*60}")
    print("Phase 4: Converting to tensor format and saving")
    print(f"{'='*60}")

    # Output in same folder
    output_dir = os.path.join(folder_path, "train_test_split")
    os.makedirs(output_dir, exist_ok=True)

    # Convert TRAIN data
    print("\nConverting TRAIN data...")
    train_tensor_data = convert_to_tensor_format(train_data_per_lib)
    del train_data_per_lib
    gc.collect()

    # Save train file
    train_filename = f"train_{data_type}_{graph_mode}.pth"
    train_path = os.path.join(output_dir, train_filename)

    train_meta = {
        'node_features': train_tensor_data['node_features'],
        'outputs': train_tensor_data['outputs'],
        'node_slices': train_tensor_data['node_slices'],
        'cell_names': train_tensor_data['cell_names'],
        'delay_types': train_tensor_data['delay_types'],
        'output_names': train_tensor_data['output_names'],
        'node_counts': train_tensor_data['node_counts'],
        'num_tasks': train_tensor_data['num_tasks'],
        'num_libs': train_tensor_data['num_libs'],
        'num_features': train_tensor_data['num_features'],
        'total_nodes': train_tensor_data['total_nodes'],
        'format': 'tensor',
        'process_node': 'TSMC',
        'folder_name': folder_name,
        'data_type': data_type,
        'graph_mode': graph_mode,
        'cache_path': cache_path,
        'split_type': 'train',
        'train_ratio': train_ratio,
        'seed': seed,
        'norm_stats': train_norm_stats,
        'filter_stats': train_filter_stats,
    }

    torch.save(train_meta, train_path)
    print(f"\nSaved TRAIN: {train_path}")
    print(f"  Tasks: {train_tensor_data['num_tasks']}")
    print(f"  node_features: {train_tensor_data['node_features'].shape}")

    del train_tensor_data
    gc.collect()

    # Convert TEST data
    print("\nConverting TEST data...")
    test_tensor_data = convert_to_tensor_format(test_data_per_lib)
    del test_data_per_lib
    gc.collect()

    # Save test file
    test_filename = f"test_{data_type}_{graph_mode}.pth"
    test_path = os.path.join(output_dir, test_filename)

    test_meta = {
        'node_features': test_tensor_data['node_features'],
        'outputs': test_tensor_data['outputs'],
        'node_slices': test_tensor_data['node_slices'],
        'cell_names': test_tensor_data['cell_names'],
        'delay_types': test_tensor_data['delay_types'],
        'output_names': test_tensor_data['output_names'],
        'node_counts': test_tensor_data['node_counts'],
        'num_tasks': test_tensor_data['num_tasks'],
        'num_libs': test_tensor_data['num_libs'],
        'num_features': test_tensor_data['num_features'],
        'total_nodes': test_tensor_data['total_nodes'],
        'format': 'tensor',
        'process_node': 'TSMC',
        'folder_name': folder_name,
        'data_type': data_type,
        'graph_mode': graph_mode,
        'cache_path': cache_path,
        'split_type': 'test',
        'train_ratio': train_ratio,
        'seed': seed,
        'norm_stats': train_norm_stats,
        'continuity_stats': test_continuity_stats,
    }

    torch.save(test_meta, test_path)
    print(f"\nSaved TEST: {test_path}")
    print(f"  Tasks: {test_tensor_data['num_tasks']}")
    print(f"  node_features: {test_tensor_data['node_features'].shape}")

    total_train_samples = train_meta['num_tasks']
    total_test_samples = test_meta['num_tasks']

    del test_tensor_data
    gc.collect()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {folder_name}")
    print(f"Graph mode: {graph_mode}")
    print(f"\nFinal dataset:")
    print(f"  Train: {total_train_samples} samples")
    print(f"  Test: {total_test_samples} samples")
    print(f"  Libs: {num_libs}")

    if test_continuity_stats:
        print(f"\nContinuity: {test_continuity_stats['continuous_ratio']:.1f}% continuous")

    print(f"\nOutput files:")
    print(f"  {train_path}")
    print(f"  {test_path}")
    print(f"{'='*80}")

    return {
        'train_path': train_path,
        'test_path': test_path,
        'train_samples': total_train_samples,
        'test_samples': total_test_samples,
        'num_libs': num_libs,
        'norm_stats': train_norm_stats,
        'train_filter_stats': train_filter_stats,
        'test_continuity_stats': test_continuity_stats,
    }


def run_all_tsmc(base_path=None, graph_mode='stage_aware', data_type='cell',
                 train_ratio=0.8, seed=42, enable_filtering=True,
                 enable_continuity_analysis=True, delete_source_files=False):
    """
    Run preprocessing for all available TSMC temperature folders.
    """
    if base_path is None:
        base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN"

    # First validate
    validation = validate_tsmc_datasets(base_path, graph_mode, data_type)

    if not validation['valid_folders']:
        print("\nNo valid folders found!")
        return []

    print(f"\n{'#'*80}")
    print(f"# Processing {len(validation['valid_folders'])} TSMC folders")
    print(f"{'#'*80}")

    results = []
    for folder_info in validation['valid_folders']:
        folder_name = folder_info['folder']

        print(f"\n\n{'#'*80}")
        print(f"# Processing: {folder_name}")
        print(f"{'#'*80}")

        try:
            result = split_tsmc_single_folder(
                folder_name=folder_name,
                base_path=base_path,
                train_ratio=train_ratio,
                seed=seed,
                data_type=data_type,
                graph_mode=graph_mode,
                enable_filtering=enable_filtering,
                enable_continuity_analysis=enable_continuity_analysis,
                delete_source_files=delete_source_files
            )
            if result:
                results.append((folder_name, result))
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Folder':<15} {'Train':<10} {'Test':<10} {'Libs':<6}")
    print("-" * 50)
    for name, result in results:
        print(f"{name:<15} {result['train_samples']:<10} {result['test_samples']:<10} {result['num_libs']:<6}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Split TSMC GNN datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate TSMC dataset structure:
  python split_gnn_dataset_tsmc.py --validate --graph_mode stage_aware

  # Process single folder using folder name:
  python split_gnn_dataset_tsmc.py --folder TSMC_FF_0 --graph_mode full_graph

  # Process using corner and temperature:
  python split_gnn_dataset_tsmc.py --corner FF --temperature 0 --graph_mode stage_aware

  # Process all available folders (15 configurations):
  python split_gnn_dataset_tsmc.py --run_all --graph_mode stage_aware
"""
    )

    # Mode selection
    parser.add_argument('--validate', action='store_true',
                       help='Only validate dataset structure')
    parser.add_argument('--folder', type=str, default=None,
                       help='Process single folder (e.g., TSMC_FF_0)')
    parser.add_argument('--corner', type=str,
                       choices=['TT', 'FF', 'SS'],
                       help='Process corner (shorthand, use with --temperature)')
    parser.add_argument('--temperature', type=int,
                       choices=[0, 25, 50, 75, 100],
                       help='Temperature in Celsius (shorthand, use with --corner)')
    parser.add_argument('--run_all', action='store_true',
                       help='Process all 15 configurations (3 corners x 5 temperatures)')

    # Common options
    parser.add_argument('--base_path', type=str, default=None,
                       help='Path to TSMC dataset directory')
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['full_graph', 'stage_aware'],
                       help='Graph mode (default: stage_aware)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # Preprocessing options
    parser.add_argument('--no_filtering', action='store_true',
                       help='Disable filtering')
    parser.add_argument('--no_continuity', action='store_true',
                       help='Disable continuity analysis')
    parser.add_argument('--delete_source_files', action='store_true',
                       help='Delete source .pth files after loading')

    args = parser.parse_args()

    base_path = args.base_path or "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN"

    # Convert corner + temperature to folder name
    if args.corner is not None and args.temperature is not None and args.folder is None:
        args.folder = get_folder_name(args.corner, args.temperature)
    elif args.temperature is not None and args.folder is None:
        # Default to FF corner if only temperature is specified
        args.folder = f"TSMC_FF_{args.temperature}"

    if args.validate:
        validate_tsmc_datasets(base_path, args.graph_mode, args.data_type)

    elif args.run_all:
        run_all_tsmc(
            base_path=base_path,
            graph_mode=args.graph_mode,
            data_type=args.data_type,
            train_ratio=args.train_ratio,
            seed=args.seed,
            enable_filtering=not args.no_filtering,
            enable_continuity_analysis=not args.no_continuity,
            delete_source_files=args.delete_source_files
        )

    elif args.folder:
        result = split_tsmc_single_folder(
            folder_name=args.folder,
            base_path=base_path,
            train_ratio=args.train_ratio,
            seed=args.seed,
            data_type=args.data_type,
            graph_mode=args.graph_mode,
            enable_filtering=not args.no_filtering,
            enable_continuity_analysis=not args.no_continuity,
            delete_source_files=args.delete_source_files
        )

        if result:
            print(f"\nDone!")
            print(f"  Train: {result['train_path']}")
            print(f"  Test: {result['test_path']}")

    else:
        parser.print_help()
        print("\nSpecify --validate, --folder <name>, --corner + --temperature, or --run_all")
