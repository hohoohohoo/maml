#!/usr/bin/env python
"""
Split GNN dataset with unified preprocessing pipeline

Two modes:
1. Single directory mode: split_gnn_dataset_batch() - splits one dataset folder
2. Multi-directory mode: split_gnn_dataset_by_process_corner() - finds all folders
   matching process_type + corner_type (e.g., LVT_TT) and creates unified train/test files

Multi-directory mode includes:
- NaN/Inf filtering
- Low variance task filtering
- Normalization statistics calculation
- Continuity analysis (from ASAP7_GCN_voltage_validation.py)
- Unified train/test file output (one train.pth + one test.pth per process+corner)
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

try:
    from data_management_utils import check_input_continuity, check_output_continuity
except ImportError:
    # Define locally if import fails
    def check_input_continuity(data, feature_idx=0, threshold_ratio=0.18):
        """Check if input data is continuous"""
        if len(data) < 2:
            return True, 1.0, [], 0, 0

        sorted_data = np.sort(data.flatten())
        diffs = np.diff(sorted_data)
        data_range = sorted_data[-1] - sorted_data[0]

        if data_range == 0:
            return True, 1.0, [], 0, 0

        threshold = threshold_ratio * data_range / (len(data) - 1)
        gaps = np.where(diffs > threshold)[0]
        max_jump = diffs.max() if len(diffs) > 0 else 0
        max_ratio = max_jump / data_range if data_range > 0 else 0

        score = 1.0 - len(gaps) / max(len(diffs), 1)
        is_continuous = len(gaps) == 0

        return is_continuous, score, gaps.tolist(), max_jump, max_ratio

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
    Checks if output varies continuously across voltage conditions.

    Note: Lib files (40-100) are already sorted by voltage order,
    so lib_idx order = voltage order. No re-sorting needed.

    Args:
        minimal_data_per_file: List of lists [num_libs][num_samples] containing minimal samples
        threshold_ratio: Continuity threshold ratio (default: 0.18)
        max_check_samples: Maximum number of tasks to check (default: 100000)

    Returns:
        continuous_task_ids: List of continuous task IDs
        discontinuous_task_ids: List of discontinuous task IDs
        continuity_analysis: List of analysis results for each task
    """
    continuous_task_ids = []
    discontinuous_task_ids = []
    continuity_analysis = []

    num_libs = len(minimal_data_per_file)
    num_tasks = len(minimal_data_per_file[0])
    num_check_samples = min(max_check_samples, num_tasks)

    print(f"\n   Analyzing output continuity for first {num_check_samples} tasks...")
    print(f"   Libs: {num_libs} (already voltage-sorted), Total tasks: {num_tasks}")

    for task_idx in range(num_check_samples):
        if task_idx % 10000 == 0 and task_idx > 0:
            print(f"   Progress: {task_idx}/{num_check_samples}")

        try:
            # Collect outputs across all libs for this task
            # Lib order (40-100) is already voltage order, no sorting needed
            task_outputs = []

            for lib_idx in range(num_libs):
                sample = minimal_data_per_file[lib_idx][task_idx]
                task_outputs.append(sample['output'])

            task_outputs = np.array(task_outputs)

            # Check output continuity (lib order = voltage order)
            output_continuous, output_score, output_gaps, output_max_jump, output_max_ratio = check_output_continuity(
                task_outputs.reshape(-1, 1), threshold_ratio=threshold_ratio
            )

            analysis_result = {
                'task_id': task_idx,
                'output_continuous': output_continuous,
                'output_score': output_score,
                'output_gaps': len(output_gaps),
                'output_max_ratio': output_max_ratio
            }

            continuity_analysis.append(analysis_result)

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
    print(f"   - Continuous tasks: {len(continuous_task_ids)} ({len(continuous_task_ids)/max(len(continuity_analysis),1)*100:.1f}%)")
    print(f"   - Discontinuous tasks: {len(discontinuous_task_ids)} ({len(discontinuous_task_ids)/max(len(continuity_analysis),1)*100:.1f}%)")

    return continuous_task_ids, discontinuous_task_ids, continuity_analysis


def convert_to_tensor_format(minimal_data_per_lib):
    """
    Convert minimal_data_per_file (list of lists of dicts) to tensor-based format.

    This enables efficient mmap loading since tensors can be memory-mapped,
    while Python lists/dicts cannot.

    Note: edge_index is NOT included here - it's loaded from topology_cache
    during training using cell_name lookup.

    Args:
        minimal_data_per_lib: List of lists [num_libs][num_tasks] containing minimal samples
            Each sample is a dict with 'node_features', 'cell_name', 'output'

    Returns:
        Dictionary with tensor-based storage:
        {
            'node_features': tensor [num_libs, total_nodes, num_features],
            'outputs': tensor [num_libs, num_tasks],
            'node_slices': tensor [num_tasks + 1],
            'cell_names': list [num_tasks],  # for topology cache lookup
            'num_libs': int,
            'num_tasks': int,
        }
    """
    num_libs = len(minimal_data_per_lib)
    num_tasks = len(minimal_data_per_lib[0])

    print(f"   Converting to tensor format: {num_libs} libs, {num_tasks} tasks")

    # First pass: calculate total nodes, collect cell names and metadata
    # Use lib 0 as reference (all libs have same structure)
    node_counts = []
    cell_names = []
    delay_types = []
    output_names = []

    for task_idx in range(num_tasks):
        sample = minimal_data_per_lib[0][task_idx]
        node_features = sample['node_features']
        cell_name = sample.get('cell_name', f'task_{task_idx}')
        delay_type = sample.get('delay_type', 'rise')
        output_name = sample.get('output_name', '')

        if isinstance(node_features, torch.Tensor):
            node_counts.append(node_features.shape[0])
        else:
            node_counts.append(len(node_features))

        cell_names.append(cell_name)
        delay_types.append(delay_type)
        output_names.append(output_name)

    total_nodes = sum(node_counts)

    # Get feature dimension from first sample
    first_sample = minimal_data_per_lib[0][0]
    if isinstance(first_sample['node_features'], torch.Tensor):
        num_features = first_sample['node_features'].shape[1]
    else:
        num_features = first_sample['node_features'].shape[1]

    print(f"   Total nodes: {total_nodes}, Features: {num_features}")

    # Create slices (cumulative sums)
    node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
    node_slices[1:] = np.cumsum(node_counts)

    # Allocate tensors
    all_node_features = np.zeros((num_libs, total_nodes, num_features), dtype=np.float32)
    all_outputs = np.zeros((num_libs, num_tasks), dtype=np.float32)

    # Second pass: fill tensors
    print(f"   Filling tensors...")

    # Fill node_features and outputs for each lib
    mismatched_tasks = []
    for lib_idx in range(num_libs):
        if lib_idx % 10 == 0:
            print(f"   Processing lib {lib_idx}/{num_libs}...")

        for task_idx in range(num_tasks):
            sample = minimal_data_per_lib[lib_idx][task_idx]

            # Node features
            node_features = sample['node_features']
            if isinstance(node_features, torch.Tensor):
                node_features = node_features.cpu().numpy()

            node_start = node_slices[task_idx]
            node_end = node_slices[task_idx + 1]
            expected_nodes = node_end - node_start
            actual_nodes = node_features.shape[0]

            # Check for node count mismatch
            if actual_nodes != expected_nodes:
                if len(mismatched_tasks) < 10:  # Only log first 10
                    cell_name = sample.get('cell_name', f'task_{task_idx}')
                    mismatched_tasks.append({
                        'lib_idx': lib_idx,
                        'task_idx': task_idx,
                        'cell_name': cell_name,
                        'expected': expected_nodes,
                        'actual': actual_nodes
                    })
                # Use the minimum to avoid overflow, pad with zeros if needed
                copy_nodes = min(actual_nodes, expected_nodes)
                all_node_features[lib_idx, node_start:node_start + copy_nodes, :] = node_features[:copy_nodes]
            else:
                all_node_features[lib_idx, node_start:node_end, :] = node_features

            # Output
            output = sample['output']
            if isinstance(output, torch.Tensor):
                output = output.item()
            all_outputs[lib_idx, task_idx] = output

    # Report mismatches
    if mismatched_tasks:
        print(f"\n   WARNING: Found {len(mismatched_tasks)}+ node count mismatches across libs!")
        print(f"   This indicates inconsistent data - same cell has different node counts in different libs.")
        print(f"   First few mismatches:")
        for m in mismatched_tasks[:5]:
            print(f"      lib {m['lib_idx']}, task {m['task_idx']} ({m['cell_name']}): expected {m['expected']} nodes, got {m['actual']}")

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


def split_gnn_dataset_by_process_corner(process_type, corner_type, base_path=None,
                                         train_ratio=0.8, seed=42,
                                         data_type='cell', graph_mode='full_graph',
                                         enable_filtering=True, min_std_threshold=1e-6,
                                         enable_continuity_analysis=True,
                                         continuity_threshold=0.15,
                                         delete_source_files=False):
    """
    Split GNN datasets by process_type + corner_type combination.

    Finds all folders matching the pattern (e.g., *_LVT_TT for process_type=LVT, corner_type=TT),
    splits each folder's data, applies preprocessing, and creates unified train/test files.

    This combines functionality from:
    - maml_gnn_training_cached.py: load_cached_gnn_data_for_maml()
    - ASAP7_GCN_voltage_validation.py: load_test_data_from_indices(), analyze_gnn_continuity()

    Args:
        process_type: Process type (RVT, LVT, SLVT, SRAM)
        corner_type: Corner type (TT, FF, SS)
        base_path: Base path to dataset directory (default: dataset_all/dataset_temp)
        train_ratio: Ratio of training data (default 0.8)
        seed: Random seed for reproducibility
        data_type: 'cell' or 'transition' (default: 'cell')
        graph_mode: 'full_graph' or 'stage_aware' (default: 'full_graph')
        enable_filtering: Whether to filter invalid/low-variance tasks (default: True)
        min_std_threshold: Minimum std threshold for output filtering (default: 1e-6)
        enable_continuity_analysis: Whether to perform continuity analysis (default: True)
        continuity_threshold: Threshold ratio for continuity check (default: 0.18)
        delete_source_files: Whether to delete source files after loading (default: False)

    Returns:
        Dictionary with paths to unified train/test files and statistics
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Default base path
    if base_path is None:
        base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp"

    print(f"\n{'='*80}")
    print(f"Split GNN Dataset by Process-Corner")
    print(f"{'='*80}")
    print(f"Process type: {process_type}")
    print(f"Corner type: {corner_type}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Train ratio: {train_ratio}")
    print(f"Filtering enabled: {enable_filtering}")
    print(f"Continuity analysis: {enable_continuity_analysis}")
    print(f"Delete source files after loading: {delete_source_files}")
    print(f"{'='*80}")

    # Required cell type prefixes (all 4 must exist)
    REQUIRED_CELL_TYPES = ['AO', 'OA', 'simple', 'INVBUF']

    # Find all matching folders
    matching_folders = []
    found_cell_types = set()

    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue

        parts = item.split('_')
        if len(parts) < 3:
            continue

        # Extract process type from folder name
        folder_process = None
        for part in parts:
            if part in ['RVT', 'LVT', 'SLVT', 'SRAM']:
                folder_process = part
                break

        if folder_process != process_type:
            continue

        # Check corner type matching
        corner_match = False
        if corner_type == "TT" and not item.endswith(('_FF', '_SS')):
            corner_match = True
        elif corner_type == "FF" and item.endswith('_FF'):
            corner_match = True
        elif corner_type == "SS" and item.endswith('_SS'):
            corner_match = True

        if corner_match:
            # Check if data file exists
            data_file = os.path.join(item_path, "graph_data", f"{data_type}_all_graph_data_{graph_mode}.pth")
            if os.path.exists(data_file):
                matching_folders.append((item, item_path, data_file))
                # Extract cell type prefix
                cell_type = parts[0]
                found_cell_types.add(cell_type)
            else:
                # Try without graph_mode suffix
                data_file_alt = os.path.join(item_path, "graph_data", f"{data_type}_all_graph_data.pth")
                if os.path.exists(data_file_alt):
                    matching_folders.append((item, item_path, data_file_alt))
                    cell_type = parts[0]
                    found_cell_types.add(cell_type)

    if not matching_folders:
        print(f"No matching folders found for {process_type}_{corner_type}")
        return None

    # Validate all required cell types exist
    missing_cell_types = set(REQUIRED_CELL_TYPES) - found_cell_types
    if missing_cell_types:
        print(f"\n   ERROR: Missing required cell types for {process_type}_{corner_type}:")
        print(f"   Required: {REQUIRED_CELL_TYPES}")
        print(f"   Found: {sorted(found_cell_types)}")
        print(f"   Missing: {sorted(missing_cell_types)}")
        print(f"   Skipping this process-corner combination.")
        return None

    print(f"\nFound {len(matching_folders)} matching folders:")
    print(f"   Cell types found: {sorted(found_cell_types)}")
    for folder_name, _, _ in matching_folders:
        print(f"  - {folder_name}")

    # ============================================================
    # Phase 1: Load data and perform individual splits
    # ============================================================
    print(f"\n{'='*60}")
    print("Phase 1: Loading data and performing individual splits")
    print(f"{'='*60}")

    all_train_data_per_lib = None
    all_test_data_per_lib = None
    topology_cache = None
    cache_path = None

    folder_stats = []
    total_train_samples = 0
    total_test_samples = 0
    num_libs = None

    for folder_name, folder_path, data_file in matching_folders:
        print(f"\nProcessing: {folder_name}")

        # Load dataset
        data = torch.load(data_file, weights_only=False)

        if 'minimal_data_per_file' not in data:
            print(f"  Skipping {folder_name}: not in minimal format")
            continue

        minimal_data = data['minimal_data_per_file']
        folder_num_libs = len(minimal_data)
        num_samples = len(minimal_data[0])

        # Check lib count consistency
        if num_libs is None:
            num_libs = folder_num_libs
        elif num_libs != folder_num_libs:
            print(f"  Warning: {folder_name} has {folder_num_libs} libs, expected {num_libs}. Skipping.")
            continue

        # Load topology cache (only once)
        if topology_cache is None and 'cache_path' in data:
            cache_path = data['cache_path']
            # Resolve cache path
            if cache_path and not os.path.isabs(cache_path):
                cache_filename = os.path.basename(cache_path)
                cache_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"

            if cache_path and os.path.exists(cache_path):
                print(f"  Loading topology cache: {cache_path}")
                topology_cache = torch.load(cache_path, weights_only=False)
                print(f"  Loaded {len(topology_cache)} cells")

        print(f"  Samples: {num_samples}, Libs: {folder_num_libs}")

        # Create sample indices and split
        sample_indices = list(range(num_samples))
        random.shuffle(sample_indices)

        train_size = int(num_samples * train_ratio)
        train_indices = sorted(sample_indices[:train_size])
        test_indices = sorted(sample_indices[train_size:])

        print(f"  Split: {len(train_indices)} train, {len(test_indices)} test")

        # Initialize merged data structures
        if all_train_data_per_lib is None:
            all_train_data_per_lib = [[] for _ in range(num_libs)]
            all_test_data_per_lib = [[] for _ in range(num_libs)]

        # Extract train/test samples for each lib
        for lib_idx in range(num_libs):
            lib_samples = minimal_data[lib_idx]

            # Add train samples
            for idx in train_indices:
                all_train_data_per_lib[lib_idx].append(lib_samples[idx])

            # Add test samples
            for idx in test_indices:
                all_test_data_per_lib[lib_idx].append(lib_samples[idx])

        total_train_samples += len(train_indices)
        total_test_samples += len(test_indices)

        folder_stats.append({
            'folder': folder_name,
            'total_samples': num_samples,
            'train_samples': len(train_indices),
            'test_samples': len(test_indices)
        })

        # Clean up to save memory
        del data
        del minimal_data
        gc.collect()

        # Delete source file after loading to save storage
        if delete_source_files:
            try:
                os.remove(data_file)
                print(f"  Deleted source file: {data_file}")
            except Exception as e:
                print(f"  Warning: Failed to delete {data_file}: {e}")

    print(f"\n{'='*60}")
    print("Phase 1 Complete: Data Merged")
    print(f"{'='*60}")
    print(f"Total train samples: {total_train_samples}")
    print(f"Total test samples: {total_test_samples}")
    print(f"Libs per sample: {num_libs}")

    # ============================================================
    # Phase 2: Apply preprocessing to TRAIN data only
    # (filtering + normalization - same as maml_gnn_training_cached.py)
    # ============================================================
    print(f"\n{'='*60}")
    print("Phase 2: Preprocessing TRAIN data (filtering + normalization)")
    print(f"{'='*60}")

    train_norm_stats = None
    train_filter_stats = None

    if enable_filtering:
        print("\nPreprocessing TRAIN data...")
        preprocessed_train, train_norm_stats, train_preprocessing_stats = preprocess_gnn_minimal_data(
            all_train_data_per_lib,
            min_std_threshold=min_std_threshold,
            enable_filtering=True,
            verbose=True
        )

        train_filter_stats = train_preprocessing_stats.get('filtering', {})
        print(f"  Train: {train_filter_stats.get('valid_tasks', 0)} valid tasks after filtering")
        print(f"  Filter ratio: {train_filter_stats.get('filter_ratio', 0):.1f}%")

        # Replace with preprocessed data
        all_train_data_per_lib = preprocessed_train
        total_train_samples = len(all_train_data_per_lib[0]) if all_train_data_per_lib else 0
    else:
        # Just calculate norm_stats without filtering
        print("\nCalculating normalization statistics (filtering disabled)...")
        train_norm_stats = calculate_norm_stats_from_minimal_data_safe(
            all_train_data_per_lib, sample_rate=10
        )

    # NOTE: Test data is NOT filtered - kept as-is for validation
    print(f"\n  Test data: {total_test_samples} samples (no filtering applied)")

    # ============================================================
    # Phase 3: Continuity analysis for TEST data only
    # (same as ASAP7_GCN_voltage_validation.py)
    # ============================================================
    test_continuity_stats = None

    if enable_continuity_analysis and all_test_data_per_lib:
        print(f"\n{'='*60}")
        print("Phase 3: Continuity Analysis (TEST data only)")
        print(f"{'='*60}")

        print("\nAnalyzing TEST data continuity...")
        test_continuous, test_discontinuous, test_analysis = analyze_gnn_continuity(
            all_test_data_per_lib,
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

    # Create output directory
    output_dir = os.path.join(base_path, f"unified_{process_type}_{corner_type}")
    os.makedirs(output_dir, exist_ok=True)

    # Convert TRAIN data to tensor format
    print("\nConverting TRAIN data to tensor format...")
    train_tensor_data = convert_to_tensor_format(all_train_data_per_lib)

    # Free original train data
    del all_train_data_per_lib
    gc.collect()

    # Build train file
    train_filename = f"train_{data_type}_{graph_mode}.pth"
    train_path = os.path.join(output_dir, train_filename)

    train_meta = {
        # Tensor data (mmap-able)
        'node_features': train_tensor_data['node_features'],
        'outputs': train_tensor_data['outputs'],
        'node_slices': train_tensor_data['node_slices'],
        # Metadata
        'cell_names': train_tensor_data['cell_names'],  # for topology cache lookup
        'delay_types': train_tensor_data['delay_types'],  # rise/fall for stage_aware
        'output_names': train_tensor_data['output_names'],  # output pin for stage_aware
        'node_counts': train_tensor_data['node_counts'],
        'num_tasks': train_tensor_data['num_tasks'],
        'num_libs': train_tensor_data['num_libs'],
        'num_features': train_tensor_data['num_features'],
        'total_nodes': train_tensor_data['total_nodes'],
        # Dataset info
        'format': 'tensor',  # Mark as tensor format
        'process_type': process_type,
        'corner_type': corner_type,
        'data_type': data_type,
        'graph_mode': graph_mode,
        'cache_path': cache_path,
        'split_type': 'train',
        'train_ratio': train_ratio,
        'seed': seed,
        'norm_stats': train_norm_stats,
        'filter_stats': train_filter_stats,
        'source_folders': [f['folder'] for f in folder_stats],
        'folder_stats': folder_stats
    }

    torch.save(train_meta, train_path)
    print(f"\nSaved TRAIN file: {train_path}")
    print(f"  Tasks: {train_tensor_data['num_tasks']}")
    print(f"  Libs: {train_tensor_data['num_libs']}")
    print(f"  node_features: {train_tensor_data['node_features'].shape}")
    print(f"  outputs: {train_tensor_data['outputs'].shape}")

    # Free train tensor data
    del train_tensor_data
    gc.collect()

    # Convert TEST data to tensor format
    print("\nConverting TEST data to tensor format...")
    test_tensor_data = convert_to_tensor_format(all_test_data_per_lib)

    # Free original test data
    del all_test_data_per_lib
    gc.collect()

    # Build test file
    test_filename = f"test_{data_type}_{graph_mode}.pth"
    test_path = os.path.join(output_dir, test_filename)

    test_meta = {
        # Tensor data (mmap-able)
        'node_features': test_tensor_data['node_features'],
        'outputs': test_tensor_data['outputs'],
        'node_slices': test_tensor_data['node_slices'],
        # Metadata
        'cell_names': test_tensor_data['cell_names'],  # for topology cache lookup
        'delay_types': test_tensor_data['delay_types'],  # rise/fall for stage_aware
        'output_names': test_tensor_data['output_names'],  # output pin for stage_aware
        'node_counts': test_tensor_data['node_counts'],
        'num_tasks': test_tensor_data['num_tasks'],
        'num_libs': test_tensor_data['num_libs'],
        'num_features': test_tensor_data['num_features'],
        'total_nodes': test_tensor_data['total_nodes'],
        # Dataset info
        'format': 'tensor',  # Mark as tensor format
        'process_type': process_type,
        'corner_type': corner_type,
        'data_type': data_type,
        'graph_mode': graph_mode,
        'cache_path': cache_path,
        'split_type': 'test',
        'train_ratio': train_ratio,
        'seed': seed,
        'norm_stats': train_norm_stats,  # Use train norm_stats for consistency
        'continuity_stats': test_continuity_stats,
        'source_folders': [f['folder'] for f in folder_stats],
        'folder_stats': folder_stats
    }

    torch.save(test_meta, test_path)
    print(f"\nSaved TEST file: {test_path}")
    print(f"  Tasks: {test_tensor_data['num_tasks']}")
    print(f"  Libs: {test_tensor_data['num_libs']}")
    print(f"  node_features: {test_tensor_data['node_features'].shape}")
    print(f"  outputs: {test_tensor_data['outputs'].shape}")

    # Update counts for summary
    total_train_samples = train_meta['num_tasks']
    total_test_samples = test_meta['num_tasks']

    # Free test tensor data
    del test_tensor_data
    gc.collect()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Process: {process_type}, Corner: {corner_type}")
    print(f"Source folders: {len(folder_stats)}")
    for stat in folder_stats:
        print(f"  - {stat['folder']}: {stat['train_samples']} train, {stat['test_samples']} test")
    print(f"\nFinal dataset:")
    print(f"  Train: {total_train_samples} samples (filtered)")
    print(f"  Test: {total_test_samples} samples (unfiltered)")
    print(f"  Libs per sample: {num_libs}")

    if test_continuity_stats:
        print(f"\nContinuity (test only): {test_continuity_stats['continuous_ratio']:.1f}% continuous")

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
        'folder_stats': folder_stats
    }


def split_gnn_dataset_batch(dataset_dir, train_ratio=0.8, seed=42, cache_type=None, data_type='cell'):
    """
    Split GNN dataset into train/test sets (single directory mode)
    Compatible with both full_graph and stage_aware cached datasets

    NOTE: For multi-directory processing with preprocessing, use split_gnn_dataset_by_process_corner()

    Args:
        dataset_dir: Directory containing graph_data/{data_type}_all_graph_data*.pth
        train_ratio: Ratio of training data (default 0.8)
        seed: Random seed for reproducibility
        cache_type: 'full_graph' or 'stage_aware' (auto-detected if None)
        data_type: 'cell' or 'transition' (default: 'cell')

    Returns:
        Dictionary with paths to train/test indices
    """

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Construct paths
    dataset_path = Path(dataset_dir)

    # Try to find dataset file
    if cache_type:
        possible_files = [
            dataset_path / "graph_data" / f"{data_type}_all_graph_data_{cache_type}.pth",
        ]
    else:
        possible_files = [
            dataset_path / "graph_data" / f"{data_type}_all_graph_data.pth",
            dataset_path / "graph_data" / f"{data_type}_all_graph_data_full_graph.pth",
            dataset_path / "graph_data" / f"{data_type}_all_graph_data_stage_aware.pth"
        ]

    existing_files = [f for f in possible_files if f.exists()]

    if not existing_files:
        print(f"Dataset not found in: {dataset_path / 'graph_data'}")
        if cache_type:
            print(f"   Requested cache type: {cache_type}")
        return None

    results = []
    for data_file in existing_files:
        print(f"\n{'='*70}")
        print(f"Processing dataset file: {data_file.name}")
        print(f"{'='*70}")
        print(f"   Full path: {data_file}")

        # Load dataset
        data = torch.load(data_file, weights_only=False)

        # Detect cache type
        detected_cache_type = data.get('cache_type', 'unknown')
        current_cache_type = cache_type if cache_type else detected_cache_type

        print(f"   Cache type: {current_cache_type}")

        # Get data
        if 'minimal_data_per_file' in data:
            minimal_data = data['minimal_data_per_file']
            num_lib_files = data.get('num_lib_files', len(minimal_data))
            num_samples = len(minimal_data[0]) if minimal_data else 0
        else:
            graph_data = data.get('graph_data_per_file', [])
            num_lib_files = data.get('num_lib_files', len(graph_data))
            num_samples = len(graph_data[0]) if graph_data else 0

        print(f"   Lib files: {num_lib_files}")
        print(f"   Samples per lib: {num_samples}")

        # Split sample indices
        sample_indices = list(range(num_samples))
        random.shuffle(sample_indices)

        train_size = int(num_samples * train_ratio)
        train_sample_indices = sorted(sample_indices[:train_size])
        test_sample_indices = sorted(sample_indices[train_size:])

        print(f"\n   Split:")
        print(f"   Train samples: {len(train_sample_indices)} ({len(train_sample_indices)/num_samples*100:.1f}%)")
        print(f"   Test samples: {len(test_sample_indices)} ({len(test_sample_indices)/num_samples*100:.1f}%)")

        # Create output directory
        output_path = dataset_path / "train_test_split"
        output_path.mkdir(parents=True, exist_ok=True)

        # Save train indices
        train_meta = {
            'sample_indices': train_sample_indices,
            'num_samples': len(train_sample_indices),
            'dataset_dir': str(dataset_dir),
            'data_file': str(data_file.resolve()),
            'cache_type': current_cache_type,
            'cache_path': data.get('cache_path', None),
            'split_type': 'train',
            'train_ratio': train_ratio,
            'seed': seed,
            'num_lib_files': num_lib_files,
            'total_samples_per_lib': num_samples
        }

        train_filename = f"train_indices_{data_type}_{current_cache_type}.pth" if current_cache_type != 'unknown' else f"train_indices_{data_type}.pth"
        train_path = output_path / train_filename
        torch.save(train_meta, train_path)
        print(f"\n   Saved: {train_path}")

        # Save test indices
        test_meta = {
            'sample_indices': test_sample_indices,
            'num_samples': len(test_sample_indices),
            'dataset_dir': str(dataset_dir),
            'data_file': str(data_file.resolve()),
            'cache_type': current_cache_type,
            'cache_path': data.get('cache_path', None),
            'split_type': 'test',
            'train_ratio': train_ratio,
            'seed': seed,
            'num_lib_files': num_lib_files,
            'total_samples_per_lib': num_samples
        }

        test_filename = f"test_indices_{data_type}_{current_cache_type}.pth" if current_cache_type != 'unknown' else f"test_indices_{data_type}.pth"
        test_path = output_path / test_filename
        torch.save(test_meta, test_path)
        print(f"   Saved: {test_path}")

        result = {
            'train_path': str(train_path),
            'test_path': str(test_path),
            'train_samples': len(train_sample_indices),
            'test_samples': len(test_sample_indices),
            'dataset_dir': str(dataset_dir),
            'cache_type': current_cache_type
        }
        results.append(result)

    if len(results) == 1:
        return results[0]
    else:
        return results


class GNNBatchDataLoader:
    """
    DataLoader for GNN datasets (both old and new cached formats)
    Loads data on-demand using saved indices
    """
    def __init__(self, indices_path, batch_size=32, topology_cache=None):
        self.meta = torch.load(indices_path, weights_only=False)
        self.sample_indices = self.meta['sample_indices']
        self.data_file = self.meta['data_file']
        self.batch_size = batch_size
        self.num_lib_files = self.meta['num_lib_files']
        self.cache_type = self.meta.get('cache_type', 'unknown')

        # Load the full dataset once
        print(f"Loading dataset from: {self.data_file}")
        self.data = torch.load(self.data_file, weights_only=False)

        if 'minimal_data_per_file' in self.data:
            self.minimal_data = self.data['minimal_data_per_file']
            self.cache_path = self.data.get('cache_path', None)
            self.is_cached = True

            if topology_cache is None and self.cache_path:
                print(f"Loading topology cache from: {self.cache_path}")
                self.topology_cache = torch.load(self.cache_path, weights_only=False)
            else:
                self.topology_cache = topology_cache
        else:
            self.graph_data = self.data.get('graph_data_per_file', [])
            self.outputs = self.data.get('stacked_outputs', None)
            self.is_cached = False

        print(f"DataLoader initialized:")
        print(f"   Split: {self.meta['split_type']}")
        print(f"   Samples: {len(self.sample_indices)}")
        print(f"   Batch size: {batch_size}")

    def get_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.sample_indices))

        if self.is_cached:
            batch_samples = []
            for i in range(start_idx, end_idx):
                sample_idx = self.sample_indices[i]
                sample = self.minimal_data[0][sample_idx]
                batch_samples.append(sample)
            return batch_samples
        else:
            batch_graphs = []
            batch_outputs = []

            for i in range(start_idx, end_idx):
                sample_idx = self.sample_indices[i]
                output_row = self.outputs[sample_idx, :]
                graph = self.graph_data[0][sample_idx]
                batch_graphs.append(graph)
                batch_outputs.append(output_row)

            return batch_graphs, torch.stack(batch_outputs) if batch_outputs else torch.tensor([])

    def __len__(self):
        return (len(self.sample_indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for batch_idx in range(len(self)):
            yield self.get_batch(batch_idx)


class UnifiedGNNDataLoader:
    """
    DataLoader for unified train/test files created by split_gnn_dataset_by_process_corner()
    """
    def __init__(self, data_path, batch_size=32, topology_cache=None):
        print(f"Loading unified dataset from: {data_path}")
        self.data = torch.load(data_path, weights_only=False)

        self.minimal_data_per_file = self.data['minimal_data_per_file']
        self.num_samples = self.data['num_samples']
        self.num_libs = self.data['num_lib_files']
        self.norm_stats = self.data.get('norm_stats', None)
        self.batch_size = batch_size

        # Load topology cache if needed
        cache_path = self.data.get('cache_path', None)
        if topology_cache is None and cache_path and os.path.exists(cache_path):
            print(f"Loading topology cache from: {cache_path}")
            self.topology_cache = torch.load(cache_path, weights_only=False)
        else:
            self.topology_cache = topology_cache

        print(f"DataLoader initialized:")
        print(f"   Split: {self.data['split_type']}")
        print(f"   Samples: {self.num_samples}")
        print(f"   Libs: {self.num_libs}")
        print(f"   Batch size: {batch_size}")

    def get_task(self, task_idx):
        """Get all lib data for a single task"""
        if task_idx >= self.num_samples:
            raise IndexError(f"Task index {task_idx} out of range (max: {self.num_samples - 1})")

        samples = []
        outputs = []

        for lib_idx in range(self.num_libs):
            sample = self.minimal_data_per_file[lib_idx][task_idx]
            samples.append(sample)
            outputs.append(sample['output'])

        return samples, outputs

    def get_batch(self, batch_idx):
        """Get a batch of tasks"""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        batch_samples = []
        batch_outputs = []

        for task_idx in range(start_idx, end_idx):
            samples, outputs = self.get_task(task_idx)
            batch_samples.append(samples)
            batch_outputs.append(outputs)

        return batch_samples, batch_outputs

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for batch_idx in range(len(self)):
            yield self.get_batch(batch_idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Split GNN datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single directory mode (original behavior):
  python split_gnn_dataset.py --single ../../dataset_all/dataset_temp/AO_LVT_FF

  # Multi-directory mode (new - unified train/test by process+corner):
  python split_gnn_dataset.py --process LVT --corner TT
  python split_gnn_dataset.py --process RVT --corner SS --graph_mode stage_aware

  # Run all process-corner combinations:
  python split_gnn_dataset.py --run_all
"""
    )

    # Mode selection
    parser.add_argument('--single', type=str, default=None,
                       help='Single directory mode: path to dataset directory')
    parser.add_argument('--process', type=str, default=None,
                       choices=['RVT', 'LVT', 'SLVT', 'SRAM'],
                       help='Multi-directory mode: process type')
    parser.add_argument('--corner', type=str, default=None,
                       choices=['TT', 'FF', 'SS'],
                       help='Multi-directory mode: corner type')
    parser.add_argument('--run_all', action='store_true',
                       help='Run all 12 process-corner combinations')

    # Common options
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='full_graph',
                       choices=['full_graph', 'stage_aware'],
                       help='Graph mode (default: full_graph)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # Preprocessing options (multi-directory mode only)
    parser.add_argument('--no_filtering', action='store_true',
                       help='Disable filtering (multi-directory mode)')
    parser.add_argument('--no_continuity', action='store_true',
                       help='Disable continuity analysis (multi-directory mode)')
    parser.add_argument('--delete_source_files', action='store_true',
                       help='Delete source .pth files after loading to save storage')

    args = parser.parse_args()

    if args.run_all:
        # Run all 12 combinations
        processes = ['LVT', 'RVT', 'SLVT', 'SRAM']
        corners = ['FF', 'TT', 'SS']

        print(f"\n{'#'*80}")
        print(f"# Running all 12 process-corner combinations")
        print(f"{'#'*80}")

        results = []
        for process in processes:
            for corner in corners:
                print(f"\n\n{'#'*80}")
                print(f"# Processing: {process}_{corner}")
                print(f"{'#'*80}")

                try:
                    result = split_gnn_dataset_by_process_corner(
                        process_type=process,
                        corner_type=corner,
                        data_type=args.data_type,
                        graph_mode=args.graph_mode,
                        train_ratio=args.train_ratio,
                        seed=args.seed,
                        enable_filtering=not args.no_filtering,
                        enable_continuity_analysis=not args.no_continuity,
                        delete_source_files=args.delete_source_files
                    )
                    if result:
                        results.append((f"{process}_{corner}", result))
                except Exception as e:
                    print(f"Failed: {e}")

        # Print summary
        print(f"\n\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"{'Combination':<15} {'Train':<10} {'Test':<10} {'Libs':<6}")
        print("-" * 50)
        for name, result in results:
            print(f"{name:<15} {result['train_samples']:<10} {result['test_samples']:<10} {result['num_libs']:<6}")

    elif args.single:
        # Single directory mode
        result = split_gnn_dataset_batch(
            dataset_dir=args.single,
            train_ratio=args.train_ratio,
            seed=args.seed,
            data_type=args.data_type
        )

        if result:
            print(f"\nDone!")
            results_list = result if isinstance(result, list) else [result]
            for r in results_list:
                print(f"  Train: {r['train_path']}")
                print(f"  Test: {r['test_path']}")

    elif args.process and args.corner:
        # Multi-directory mode
        result = split_gnn_dataset_by_process_corner(
            process_type=args.process,
            corner_type=args.corner,
            data_type=args.data_type,
            graph_mode=args.graph_mode,
            train_ratio=args.train_ratio,
            seed=args.seed,
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
        print("\nError: Specify --single <path>, or --process + --corner, or --run_all")
