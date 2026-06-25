#!/usr/bin/env python
"""
Preprocess process-condition GNN datasets (11D node features)

For large datasets that cannot be merged into a single file, this script:
1. Phase 1: Compute global normalization statistics across all files
2. Phase 2: Convert each file to tensor format with filtering and continuity analysis
3. Phase 3: Save processed files and metadata for multi-file DataLoader

Usage:
    # Compute global stats first
    python preprocess_process_condition_dataset.py --phase1 --data_type cell

    # Process all files with the computed stats
    python preprocess_process_condition_dataset.py --phase2 --data_type cell

    # Run both phases
    python preprocess_process_condition_dataset.py --all --data_type cell
"""

import torch
import os
import sys
import gc
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pretraining', 'model_pretraining_code', 'gnn', 'utils'))


def find_all_dataset_files(base_path: str, data_type: str = 'cell',
                           graph_mode: str = 'full_graph') -> List[Dict]:
    """
    Find all dataset files matching the pattern.

    Args:
        base_path: Base directory containing dataset folders
        data_type: 'cell' or 'transition'
        graph_mode: 'full_graph' or 'stage_aware'

    Returns:
        List of dicts with folder info and file paths
    """
    dataset_files = []

    for item in sorted(os.listdir(base_path)):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue

        # Skip unified folders and other non-dataset folders
        if item.startswith('unified_') or item.startswith('.'):
            continue

        # Look for graph_data file
        data_file = os.path.join(item_path, "graph_data", f"{data_type}_all_graph_data_{graph_mode}.pth")

        if os.path.exists(data_file):
            # Parse folder name for metadata
            # Format: processed_invbuf_0_0_0_125 or processed_simple_0_0_0_125
            parts = item.split('_')

            cell_type = parts[1] if len(parts) > 1 else 'unknown'  # invbuf, simple, etc.

            dataset_files.append({
                'folder_name': item,
                'folder_path': item_path,
                'data_file': data_file,
                'cell_type': cell_type
            })

    return dataset_files


def validate_dataset_files(dataset_files: List[Dict],
                           expected_invbuf: int = 384,
                           expected_simple: int = 384) -> bool:
    """
    Validate that the expected number of invbuf and simple files exist.

    Args:
        dataset_files: List of dataset file info dicts
        expected_invbuf: Expected number of invbuf folders (default: 384)
        expected_simple: Expected number of simple folders (default: 384)

    Returns:
        True if validation passes, False otherwise
    """
    # Count by cell type
    cell_type_counts = {}
    for file_info in dataset_files:
        cell_type = file_info['cell_type']
        cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1

    print("\n" + "=" * 80)
    print("VALIDATION: Checking dataset file counts")
    print("=" * 80)
    print(f"Found cell types and counts:")
    for cell_type, count in sorted(cell_type_counts.items()):
        print(f"  {cell_type}: {count}")

    invbuf_count = cell_type_counts.get('invbuf', 0)
    simple_count = cell_type_counts.get('simple', 0)

    print(f"\nExpected:")
    print(f"  invbuf: {expected_invbuf}")
    print(f"  simple: {expected_simple}")

    # Validate
    errors = []
    if invbuf_count != expected_invbuf:
        errors.append(f"invbuf count mismatch: expected {expected_invbuf}, found {invbuf_count}")
    if simple_count != expected_simple:
        errors.append(f"simple count mismatch: expected {expected_simple}, found {simple_count}")

    if errors:
        print(f"\n*** VALIDATION FAILED ***")
        for error in errors:
            print(f"  ERROR: {error}")
        print("=" * 80)
        return False

    print(f"\n*** VALIDATION PASSED ***")
    print(f"Total files: {len(dataset_files)} (invbuf: {invbuf_count}, simple: {simple_count})")
    print("=" * 80)
    return True


def compute_global_norm_stats(dataset_files: List[Dict],
                              verbose: bool = True) -> Dict:
    """
    Phase 1: Compute global normalization statistics for input features across all files.

    Computes mean/std for:
    - voltage (col 4)
    - input_slew (col 5)
    - output_load (col 6)
    - temperature (col 10) - for 11D features

    Only considers non-zero values for each feature.

    Args:
        dataset_files: List of dataset file info dicts
        verbose: Print progress

    Returns:
        Dict with global normalization statistics
    """
    print("\n" + "=" * 80)
    print("PHASE 1: Computing Global Normalization Statistics (Input Features)")
    print("=" * 80)
    print(f"Total files: {len(dataset_files)}")

    # Collect non-zero values for each feature
    all_voltages = []
    all_input_slews = []
    all_output_loads = []
    all_temperatures = []

    num_features_detected = None
    skipped_nan_inf = 0

    print(f"Processing all {len(dataset_files)} files...")

    for i, file_info in enumerate(dataset_files):
        if verbose and i % 50 == 0:
            print(f"  Processing file {i+1}/{len(dataset_files)}: {file_info['folder_name']}")

        try:
            data = torch.load(file_info['data_file'], weights_only=False)

            if 'minimal_data_per_file' not in data:
                continue

            minimal_data = data['minimal_data_per_file']
            num_libs = len(minimal_data)
            num_samples = len(minimal_data[0]) if minimal_data else 0

            # Process ALL libs and ALL samples for accurate statistics
            for lib_idx in range(num_libs):
                for sample_idx in range(num_samples):
                    sample = minimal_data[lib_idx][sample_idx]

                    if 'node_features' not in sample:
                        continue

                    node_features = sample['node_features']
                    if isinstance(node_features, torch.Tensor):
                        node_features = node_features.numpy()

                    # Check for NaN/Inf
                    if np.any(np.isnan(node_features)) or np.any(np.isinf(node_features)):
                        skipped_nan_inf += 1
                        continue

                    # Detect feature dimension
                    if num_features_detected is None:
                        num_features_detected = node_features.shape[1]
                        print(f"  Detected {num_features_detected}D node features")

                    # Extract non-zero values for each feature
                    # voltage (col 4)
                    voltage_vals = node_features[:, 4]
                    voltage_nonzero = voltage_vals[voltage_vals != 0]
                    all_voltages.extend(voltage_nonzero.tolist())

                    # input_slew (col 5)
                    slew_vals = node_features[:, 5]
                    slew_nonzero = slew_vals[slew_vals != 0]
                    all_input_slews.extend(slew_nonzero.tolist())

                    # output_load (col 6)
                    load_vals = node_features[:, 6]
                    load_nonzero = load_vals[load_vals != 0]
                    all_output_loads.extend(load_nonzero.tolist())

                    # temperature (col 10) - only for 11D features
                    if node_features.shape[1] > 10:
                        temp_vals = node_features[:, 10]
                        temp_nonzero = temp_vals[temp_vals != 0]
                        all_temperatures.extend(temp_nonzero.tolist())

            del data
            del minimal_data
            gc.collect()

        except Exception as e:
            print(f"  Warning: Error processing {file_info['folder_name']}: {e}")
            continue

    # Compute statistics with safety checks
    def safe_stats(values, name):
        if len(values) == 0:
            print(f"  Warning: No {name} values found, using defaults")
            return {'mean': 0.0, 'std': 1.0, 'count': 0}

        values_array = np.array(values)

        # Remove any remaining NaN/Inf
        values_array = values_array[~np.isnan(values_array)]
        values_array = values_array[~np.isinf(values_array)]

        if len(values_array) == 0:
            print(f"  Warning: All {name} values were NaN/Inf, using defaults")
            return {'mean': 0.0, 'std': 1.0, 'count': 0}

        mean_val = float(np.mean(values_array))
        std_val = float(np.std(values_array))

        if std_val < 1e-8:
            print(f"  Warning: {name} std too small ({std_val:.2e}), using 1.0")
            std_val = 1.0

        return {
            'mean': mean_val,
            'std': std_val,
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'count': len(values_array)
        }

    print(f"\nComputing statistics...")
    if skipped_nan_inf > 0:
        print(f"  Skipped {skipped_nan_inf} samples due to NaN/Inf")

    voltage_stats = safe_stats(all_voltages, 'voltage')
    input_slew_stats = safe_stats(all_input_slews, 'input_slew')
    output_load_stats = safe_stats(all_output_loads, 'output_load')
    temperature_stats = safe_stats(all_temperatures, 'temperature') if all_temperatures else None

    global_stats = {
        'node_features': {
            'voltage': voltage_stats,
            'input_slew': input_slew_stats,
            'output_load': output_load_stats,
        },
        'num_features': num_features_detected,
        'total_files': len(dataset_files)
    }

    # Add temperature stats if 11D
    if temperature_stats:
        global_stats['node_features']['temperature'] = temperature_stats

    print(f"\nGlobal Normalization Statistics:")
    print(f"  Node features: {num_features_detected}D")
    print(f"  voltage:     mean={voltage_stats['mean']:.6f}, std={voltage_stats['std']:.6f} (n={voltage_stats['count']})")
    print(f"  input_slew:  mean={input_slew_stats['mean']:.6f}, std={input_slew_stats['std']:.6f} (n={input_slew_stats['count']})")
    print(f"  output_load: mean={output_load_stats['mean']:.6f}, std={output_load_stats['std']:.6f} (n={output_load_stats['count']})")
    if temperature_stats:
        print(f"  temperature: mean={temperature_stats['mean']:.6f}, std={temperature_stats['std']:.6f} (n={temperature_stats['count']})")

    return global_stats


def check_output_continuity(outputs: np.ndarray, threshold_ratio: float = 0.18) -> Tuple[bool, float, int]:
    """
    Check if output values are continuous across voltage levels.

    Args:
        outputs: Array of output values [num_libs]
        threshold_ratio: Threshold ratio for gap detection

    Returns:
        Tuple of (is_continuous, score, num_gaps)
    """
    if len(outputs) < 2:
        return True, 1.0, 0

    data_range = outputs.max() - outputs.min()
    if data_range == 0:
        return True, 1.0, 0

    diffs = np.abs(np.diff(outputs))
    threshold = threshold_ratio * data_range
    gaps = np.where(diffs > threshold)[0]

    score = 1.0 - len(gaps) / max(len(diffs), 1)
    is_continuous = len(gaps) == 0

    return is_continuous, score, len(gaps)


def convert_file_to_tensor_format(minimal_data_per_lib: List[List[Dict]],
                                  file_info: Dict,
                                  global_stats: Dict,
                                  min_std_threshold: float = 1e-6,
                                  continuity_threshold: float = 0.18,
                                  enable_filtering: bool = True,
                                  train_ratio: float = 0.8,
                                  seed: int = 42) -> Dict:
    """
    Convert a single file's minimal_data to tensor format with filtering.

    Note: Normalization is NOT applied here. Raw data is saved with norm_stats.
    Normalization should be applied at runtime during pretraining.

    Args:
        minimal_data_per_lib: List of lists [num_libs][num_tasks]
        file_info: Dict with folder metadata
        global_stats: Global normalization statistics (saved but not applied)
        min_std_threshold: Minimum std for output filtering
        continuity_threshold: Threshold for continuity check
        enable_filtering: Whether to filter low-variance tasks
        train_ratio: Ratio for train/test split
        seed: Random seed

    Returns:
        Dict with tensor data and metadata
    """
    random.seed(seed)
    np.random.seed(seed)

    num_libs = len(minimal_data_per_lib)
    num_tasks = len(minimal_data_per_lib[0])

    # First pass: collect metadata and filter tasks
    valid_task_indices = []
    continuity_info = []

    for task_idx in range(num_tasks):
        # Collect outputs across libs
        task_outputs = []
        for lib_idx in range(num_libs):
            sample = minimal_data_per_lib[lib_idx][task_idx]
            task_outputs.append(sample['output'])

        task_outputs = np.array(task_outputs)

        # Check for NaN/Inf
        if np.any(np.isnan(task_outputs)) or np.any(np.isinf(task_outputs)):
            continue

        # Check variance
        if enable_filtering:
            task_std = np.std(task_outputs)
            if task_std < min_std_threshold:
                continue

        # Check continuity
        is_continuous, cont_score, num_gaps = check_output_continuity(
            task_outputs, threshold_ratio=continuity_threshold
        )

        valid_task_indices.append(task_idx)
        continuity_info.append({
            'task_idx': task_idx,
            'is_continuous': is_continuous,
            'score': cont_score,
            'num_gaps': num_gaps
        })

    if not valid_task_indices:
        return None

    # Split into train/test
    shuffled_indices = valid_task_indices.copy()
    random.shuffle(shuffled_indices)

    train_size = int(len(shuffled_indices) * train_ratio)
    train_indices = sorted(shuffled_indices[:train_size])
    test_indices = sorted(shuffled_indices[train_size:])

    # Get feature dimension from first valid sample
    first_sample = minimal_data_per_lib[0][valid_task_indices[0]]
    node_features = first_sample['node_features']
    if isinstance(node_features, torch.Tensor):
        num_features = node_features.shape[1]
    else:
        num_features = node_features.shape[1]

    def build_tensor_data(task_indices: List[int]) -> Dict:
        """Build tensor data for a set of task indices."""
        if not task_indices:
            return None

        # Collect node counts and metadata
        node_counts = []
        cell_names = []
        delay_types = []
        output_names = []

        for task_idx in task_indices:
            sample = minimal_data_per_lib[0][task_idx]
            nf = sample['node_features']

            if isinstance(nf, torch.Tensor):
                node_counts.append(nf.shape[0])
            else:
                node_counts.append(len(nf))

            cell_names.append(sample.get('cell_name', f'task_{task_idx}'))
            delay_types.append(sample.get('delay_type', 'rise'))
            output_names.append(sample.get('output_name', ''))

        total_nodes = sum(node_counts)
        num_tasks_subset = len(task_indices)

        # Create slices
        node_slices = np.zeros(num_tasks_subset + 1, dtype=np.int64)
        node_slices[1:] = np.cumsum(node_counts)

        # Allocate tensors
        all_node_features = np.zeros((num_libs, total_nodes, num_features), dtype=np.float32)
        all_outputs = np.zeros((num_libs, num_tasks_subset), dtype=np.float32)

        # Fill tensors (raw data, no normalization)
        for lib_idx in range(num_libs):
            for i, task_idx in enumerate(task_indices):
                sample = minimal_data_per_lib[lib_idx][task_idx]

                # Node features (raw, not normalized)
                nf = sample['node_features']
                if isinstance(nf, torch.Tensor):
                    nf = nf.cpu().numpy()
                else:
                    nf = np.array(nf)

                node_start = node_slices[i]
                node_end = node_slices[i + 1]
                all_node_features[lib_idx, node_start:node_end, :] = nf

                # Output
                output = sample['output']
                if isinstance(output, torch.Tensor):
                    output = output.item()
                all_outputs[lib_idx, i] = output

        return {
            'node_features': torch.from_numpy(all_node_features),
            'outputs': torch.from_numpy(all_outputs),
            'node_slices': torch.from_numpy(node_slices),
            'node_counts': node_counts,
            'cell_names': cell_names,
            'delay_types': delay_types,
            'output_names': output_names,
            'num_libs': num_libs,
            'num_tasks': num_tasks_subset,
            'num_features': num_features,
            'total_nodes': total_nodes,
            'original_task_indices': task_indices
        }

    train_data = build_tensor_data(train_indices)
    test_data = build_tensor_data(test_indices)

    # Continuity stats
    continuous_count = sum(1 for c in continuity_info if c['is_continuous'])

    return {
        'train': train_data,
        'test': test_data,
        'stats': {
            'original_tasks': num_tasks,
            'valid_tasks': len(valid_task_indices),
            'train_tasks': len(train_indices),
            'test_tasks': len(test_indices),
            'filter_ratio': (1 - len(valid_task_indices) / num_tasks) * 100 if num_tasks > 0 else 0,
            'continuous_ratio': continuous_count / len(continuity_info) * 100 if continuity_info else 0
        },
        'file_info': file_info
    }


def process_single_file(file_info: Dict, global_stats: Dict, output_base_path: str,
                        data_type: str, graph_mode: str,
                        train_ratio: float = 0.8, seed: int = 42,
                        enable_filtering: bool = True,
                        delete_source: bool = True) -> Optional[Dict]:
    """
    Process a single dataset file and save train/test tensors.

    Note: Raw data is saved. norm_stats are included for runtime normalization.

    Args:
        file_info: Dict with file metadata
        global_stats: Global normalization statistics (saved for runtime use)
        output_base_path: Base path for output files
        data_type: 'cell' or 'transition'
        graph_mode: 'full_graph' or 'stage_aware'
        train_ratio: Train/test split ratio
        seed: Random seed
        enable_filtering: Whether to filter
        delete_source: Whether to delete source file after successful conversion

    Returns:
        Dict with processing stats, or None on failure
    """
    folder_name = file_info['folder_name']
    source_file = file_info['data_file']
    train_file = None
    test_file = None
    stats = {}

    try:
        # Load data
        data = torch.load(source_file, weights_only=False)

        if 'minimal_data_per_file' not in data:
            print(f"  Skipping {folder_name}: not in minimal format")
            return None

        minimal_data = data['minimal_data_per_file']
        cache_path = data.get('cache_path', None)

        # Convert to tensor format with filtering (no normalization - raw data)
        result = convert_file_to_tensor_format(
            minimal_data,
            file_info,
            global_stats,
            enable_filtering=enable_filtering,
            train_ratio=train_ratio,
            seed=seed
        )

        if result is None:
            print(f"  Skipping {folder_name}: no valid tasks after filtering")
            return None

        stats = result.get('stats', {})

        # Save train file
        output_folder = os.path.join(output_base_path, folder_name, "tensor_data")
        os.makedirs(output_folder, exist_ok=True)

        if result['train'] is not None:
            train_file = os.path.join(output_folder, f"train_{data_type}_{graph_mode}.pth")
            train_meta = {
                **result['train'],
                'format': 'tensor',
                'split_type': 'train',
                'norm_stats': global_stats,  # For runtime normalization
                'cache_path': cache_path,
                'data_type': data_type,
                'graph_mode': graph_mode,
                'folder_name': folder_name
            }
            torch.save(train_meta, train_file)

        # Save test file
        if result['test'] is not None:
            test_file = os.path.join(output_folder, f"test_{data_type}_{graph_mode}.pth")
            test_meta = {
                **result['test'],
                'format': 'tensor',
                'split_type': 'test',
                'norm_stats': global_stats,  # For runtime normalization
                'cache_path': cache_path,
                'data_type': data_type,
                'graph_mode': graph_mode,
                'folder_name': folder_name
            }
            torch.save(test_meta, test_file)

        # Clean up memory
        del data
        del minimal_data
        del result
        gc.collect()

        # Delete source file after successful conversion
        if delete_source and os.path.exists(source_file):
            try:
                os.remove(source_file)
                print(f"  Deleted source: {os.path.basename(source_file)}")
            except Exception as e:
                print(f"  Warning: Failed to delete source file: {e}")

        return {
            'folder_name': folder_name,
            'train_file': train_file,
            'test_file': test_file,
            'stats': stats
        }

    except Exception as e:
        print(f"  Error processing {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def phase2_process_all_files(dataset_files: List[Dict], global_stats: Dict,
                             output_base_path: str, data_type: str, graph_mode: str,
                             train_ratio: float = 0.8, seed: int = 42,
                             enable_filtering: bool = True,
                             delete_source: bool = True) -> Dict:
    """
    Phase 2: Process all files with global stats.

    Note: Raw data is saved. norm_stats are included in each file for runtime normalization.

    Args:
        dataset_files: List of file info dicts
        global_stats: Global normalization stats from Phase 1 (saved for runtime use)
        output_base_path: Output directory
        data_type: 'cell' or 'transition'
        graph_mode: 'full_graph' or 'stage_aware'
        train_ratio: Train/test ratio
        seed: Random seed
        enable_filtering: Whether to filter
        delete_source: Whether to delete source files after conversion

    Returns:
        Dict with processing summary
    """
    print("\n" + "=" * 80)
    print("PHASE 2: Processing Individual Files (Raw Data + norm_stats)")
    print("=" * 80)
    print(f"Total files: {len(dataset_files)}")
    print(f"Output path: {output_base_path}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Train ratio: {train_ratio}")
    print(f"Filtering: {enable_filtering}")
    print(f"Delete source after conversion: {delete_source}")

    processed_files = []
    failed_files = []
    total_train_tasks = 0
    total_test_tasks = 0

    for i, file_info in enumerate(dataset_files):
        print(f"\n[{i+1}/{len(dataset_files)}] Processing: {file_info['folder_name']}")

        result = process_single_file(
            file_info, global_stats, output_base_path,
            data_type, graph_mode, train_ratio, seed, enable_filtering,
            delete_source=delete_source
        )

        if result:
            processed_files.append(result)
            stats = result.get('stats', {})
            total_train_tasks += stats.get('train_tasks', 0)
            total_test_tasks += stats.get('test_tasks', 0)
            print(f"  Train: {stats.get('train_tasks', 0)}, Test: {stats.get('test_tasks', 0)}")
        else:
            failed_files.append(file_info['folder_name'])

    # Save manifest file
    manifest = {
        'data_type': data_type,
        'graph_mode': graph_mode,
        'norm_stats': global_stats,  # For runtime normalization
        'train_ratio': train_ratio,
        'seed': seed,
        'num_files': len(processed_files),
        'total_train_tasks': total_train_tasks,
        'total_test_tasks': total_test_tasks,
        'files': [
            {
                'folder_name': r['folder_name'],
                'train_file': r.get('train_file'),
                'test_file': r.get('test_file')
            }
            for r in processed_files
        ],
        'failed_files': failed_files
    }

    manifest_file = os.path.join(output_base_path, f"manifest_{data_type}_{graph_mode}.json")
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)
    print(f"Processed files: {len(processed_files)}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Total train tasks: {total_train_tasks}")
    print(f"Total test tasks: {total_test_tasks}")
    print(f"Manifest saved: {manifest_file}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess process-condition GNN datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all phases (validation + global stats + conversion + delete source)
    python preprocess_process_condition_dataset.py --all --data_type cell

    # Run Phase 1 only (compute global stats)
    python preprocess_process_condition_dataset.py --phase1 --data_type cell

    # Run Phase 2 only (process files, requires Phase 1 to be done first)
    python preprocess_process_condition_dataset.py --phase2 --data_type cell

    # Keep source files (don't delete after conversion)
    python preprocess_process_condition_dataset.py --all --keep_source
"""
    )

    parser.add_argument('--base_path', type=str,
                       default="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp_process",
                       help="Base path to dataset folders")
    parser.add_argument('--output_path', type=str, default=None,
                       help="Output path (default: same as base_path)")
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help="Data type")
    parser.add_argument('--graph_mode', type=str, default='full_graph',
                       choices=['full_graph', 'stage_aware'],
                       help="Graph mode")
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help="Train/test split ratio")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed")
    parser.add_argument('--no_filtering', action='store_true',
                       help="Disable filtering")

    # Phase selection
    parser.add_argument('--phase1', action='store_true',
                       help="Run Phase 1 only (compute global stats)")
    parser.add_argument('--phase2', action='store_true',
                       help="Run Phase 2 only (process files)")
    parser.add_argument('--all', action='store_true',
                       help="Run all phases")

    # Validation options
    parser.add_argument('--expected_invbuf', type=int, default=384,
                       help="Expected number of invbuf folders (default: 384)")
    parser.add_argument('--expected_simple', type=int, default=384,
                       help="Expected number of simple folders (default: 384)")
    parser.add_argument('--skip_validation', action='store_true',
                       help="Skip validation check")

    # Source file handling
    parser.add_argument('--keep_source', action='store_true',
                       help="Keep source files after conversion (default: delete)")

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.base_path

    # Find all dataset files
    print(f"\nSearching for datasets in: {args.base_path}")
    dataset_files = find_all_dataset_files(
        args.base_path, args.data_type, args.graph_mode
    )
    print(f"Found {len(dataset_files)} dataset files")

    if not dataset_files:
        print("No dataset files found!")
        return

    # Validate dataset file counts
    if not args.skip_validation:
        if not validate_dataset_files(dataset_files,
                                      expected_invbuf=args.expected_invbuf,
                                      expected_simple=args.expected_simple):
            print("\nAborting due to validation failure.")
            print("Use --skip_validation to skip this check.")
            return
    else:
        print("\nSkipping validation check (--skip_validation)")

    # Phase 1: Compute global stats
    stats_file = os.path.join(args.output_path, f"global_stats_{args.data_type}_{args.graph_mode}.json")

    if args.phase1 or args.all:
        global_stats = compute_global_norm_stats(
            dataset_files,
            verbose=True
        )

        # Save stats
        with open(stats_file, 'w') as f:
            json.dump(global_stats, f, indent=2)
        print(f"\nGlobal stats saved: {stats_file}")

    elif args.phase2:
        # Load existing stats
        if not os.path.exists(stats_file):
            print(f"Error: Global stats file not found: {stats_file}")
            print("Run --phase1 first to compute global stats")
            return

        with open(stats_file, 'r') as f:
            global_stats = json.load(f)
        print(f"Loaded global stats from: {stats_file}")

    # Phase 2: Process all files (raw data + norm_stats for runtime normalization)
    if args.phase2 or args.all:
        manifest = phase2_process_all_files(
            dataset_files, global_stats, args.output_path,
            args.data_type, args.graph_mode,
            args.train_ratio, args.seed,
            enable_filtering=not args.no_filtering,
            delete_source=not args.keep_source
        )

        print("\nProcessing complete!")
        print(f"Use manifest file for DataLoader: {os.path.join(args.output_path, f'manifest_{args.data_type}_{args.graph_mode}.json')}")


if __name__ == "__main__":
    main()
