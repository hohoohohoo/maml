#!/usr/bin/env python
"""
Unified GCN Validation Script (Unified Dataset Version)

This script validates GCN models using the unified tensor dataset format:
- Uses mmap loading for memory efficiency
- Loads topology_cache from cache_path in data file
- Same validation logic as original but with new data format

Adapted from ASAP7_GCN_voltage_validation.py
"""

import os
import sys

# Parse GPU argument before importing torch
# This is necessary because CUDA_VISIBLE_DEVICES must be set before torch import
def get_gpu_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return '0'  # default

os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_from_args()

import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from torch_geometric.data import Data, Batch

# Add paths
sys.path.append('../../../model_code/')
sys.path.append('../../../data_processing/gnn/')
sys.path.append('../utils/')

from gnn_maml import create_maml_gcn_model
from gnn_functions import evaluate_model_performance_gnn


def check_output_continuity(data, threshold_ratio=0.18):
    """
    Check if output data is continuous across voltage points.

    Args:
        data: 1D or 2D array of output values
        threshold_ratio: Maximum allowed jump as ratio of data range

    Returns:
        tuple: (is_continuous, score, gaps, max_jump, max_ratio)
    """
    if len(data) < 2:
        return True, 1.0, [], 0, 0

    data_flat = data.flatten() if hasattr(data, 'flatten') else np.array(data).flatten()
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


def filter_continuous_tasks(test_dataset, task_indices, threshold_ratio=0.18, verbose=True):
    """
    Filter task indices to only include those with continuous outputs.

    Args:
        test_dataset: UnifiedTestDataset object
        task_indices: List of task indices to check
        threshold_ratio: Continuity threshold ratio
        verbose: Print progress

    Returns:
        tuple: (continuous_indices, discontinuous_indices)
    """
    continuous_indices = []
    discontinuous_indices = []

    if verbose:
        print(f"\n   Filtering tasks for continuity (threshold={threshold_ratio})...")
        print(f"   Checking {len(task_indices)} tasks...")

    for i, task_idx in enumerate(task_indices):
        if verbose and i % 10000 == 0 and i > 0:
            print(f"   Progress: {i}/{len(task_indices)} ({len(continuous_indices)} continuous)")

        try:
            # Get all outputs for this task across all libs
            task_outputs = test_dataset.get_task_outputs(task_idx).numpy()
            is_continuous, _, _, _, _ = check_output_continuity(
                task_outputs.reshape(-1, 1), threshold_ratio=threshold_ratio
            )

            if is_continuous:
                continuous_indices.append(task_idx)
            else:
                discontinuous_indices.append(task_idx)
        except Exception as e:
            if i < 5:
                print(f"   Error checking task {task_idx}: {e}")
            discontinuous_indices.append(task_idx)

    if verbose:
        total = len(continuous_indices) + len(discontinuous_indices)
        cont_ratio = len(continuous_indices) / max(total, 1) * 100
        print(f"\n   Continuity filter results:")
        print(f"   - Continuous: {len(continuous_indices)} ({cont_ratio:.1f}%)")
        print(f"   - Discontinuous: {len(discontinuous_indices)} ({100-cont_ratio:.1f}%)")

    return continuous_indices, discontinuous_indices


def normalize_node_features(node_features, norm_stats):
    """
    Normalize node features using saved statistics
    Only normalize voltage (col 4), input_slew (col 5), output_load (col 6)
    """
    if norm_stats is None:
        return node_features

    normalized = node_features.clone()

    # Normalize voltage (column 4)
    voltage_mask = normalized[:, 4] != 0
    if voltage_mask.any():
        normalized[voltage_mask, 4] = (
            normalized[voltage_mask, 4] - norm_stats['node_features']['voltage']['mean']
        ) / norm_stats['node_features']['voltage']['std']

    # Normalize input_slew (column 5)
    slew_mask = normalized[:, 5] != 0
    if slew_mask.any():
        normalized[slew_mask, 5] = (
            normalized[slew_mask, 5] - norm_stats['node_features']['input_slew']['mean']
        ) / norm_stats['node_features']['input_slew']['std']

    # Normalize output_load (column 6)
    load_mask = normalized[:, 6] != 0
    if load_mask.any():
        normalized[load_mask, 6] = (
            normalized[load_mask, 6] - norm_stats['node_features']['output_load']['mean']
        ) / norm_stats['node_features']['output_load']['std']

    return normalized




class UnifiedTestDataset:
    """
    Dataset class for loading unified tensor format test data with mmap.
    """
    def __init__(self, unified_test_path, graph_mode='full_graph'):
        self.unified_test_path = unified_test_path
        self.graph_mode = graph_mode
        self.topology_cache = None

        self._load_data()

    def _load_data(self):
        """Load unified test data with mmap"""
        print(f"   Loading unified test data from: {self.unified_test_path}")

        # Load with mmap for memory efficiency
        data = torch.load(self.unified_test_path, weights_only=False, map_location='cpu', mmap=True)

        self._node_features = data['node_features']  # [num_libs, total_nodes, num_features]
        self._outputs = data['outputs']  # [num_libs, num_tasks]
        self._node_slices = data['node_slices']  # [num_tasks + 1]
        self._cell_names = data['cell_names']  # [num_tasks]

        # Load additional metadata if available
        self._delay_types = data.get('delay_types', None)
        self._output_names = data.get('output_names', None)
        self._continuity_stats = data.get('continuity_stats', None)
        self._norm_stats = data.get('norm_stats', None)

        self.num_libs = self._node_features.shape[0]
        self.num_tasks = self._outputs.shape[1]

        print(f"   Loaded: {self.num_tasks} tasks, {self.num_libs} libs")
        print(f"   Node features shape: {self._node_features.shape}")
        print(f"   Outputs shape: {self._outputs.shape}")

        # Print continuity stats if available (pre-computed in split_gnn_dataset.py)
        if self._continuity_stats:
            print(f"   Continuity: {self._continuity_stats.get('continuous_ratio', 0):.1f}% continuous")

        # Load topology cache from cache_path in data file
        cache_path = data.get('cache_path', None)
        if cache_path:
            self._load_topology_cache(cache_path)
        else:
            print("   Warning: No cache_path found in data file")

    def _load_topology_cache(self, cache_path):
        """Load topology cache from path"""
        # Handle /mnt/home vs /home path variations
        if cache_path.startswith('/mnt/home/'):
            cache_path = cache_path.replace('/mnt/home/', '/home/')

        # Resolve relative path if needed
        if not os.path.isabs(cache_path) or not os.path.exists(cache_path):
            cache_filename = os.path.basename(cache_path)
            cache_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"

        if os.path.exists(cache_path):
            print(f"   Loading topology cache from: {cache_path}")
            self.topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu')
            print(f"   Loaded topology cache for {len(self.topology_cache)} cells")
        else:
            print(f"   Warning: Topology cache not found at {cache_path}")

    def get_task_data(self, task_idx, lib_idx, clone=True):
        """
        Get data for a specific task and lib.

        Args:
            task_idx: Task index
            lib_idx: Library index
            clone: If True, clone tensors (safe but uses memory).
                   If False, return view (memory efficient but read-only)

        Returns:
            dict with 'node_features', 'output', 'cell_name', 'delay_type', 'output_name'
        """
        node_start = self._node_slices[task_idx].item()
        node_end = self._node_slices[task_idx + 1].item()

        # Use view for mmap efficiency, clone only when needed
        node_features = self._node_features[lib_idx, node_start:node_end, :]
        if clone:
            node_features = node_features.clone()

        output = self._outputs[lib_idx, task_idx].item()
        cell_name = self._cell_names[task_idx]

        sample = {
            'node_features': node_features,
            'output': output,
            'cell_name': cell_name,
        }

        if self._delay_types is not None:
            sample['delay_type'] = self._delay_types[task_idx]
        else:
            sample['delay_type'] = 'rise'  # default

        if self._output_names is not None:
            sample['output_name'] = self._output_names[task_idx]

        return sample

    def get_all_libs_for_task(self, task_idx, clone=True):
        """
        Get data for all libs for a specific task.

        Args:
            task_idx: Task index
            clone: If True, clone tensors for modification

        Returns:
            list of dicts, one per lib
        """
        samples = []
        outputs = []
        for lib_idx in range(self.num_libs):
            sample = self.get_task_data(task_idx, lib_idx, clone=clone)
            samples.append(sample)
            outputs.append(sample['output'])
        return samples, outputs

    def get_task_outputs(self, task_idx):
        """Get outputs for all libs for a task (no clone, memory efficient)"""
        return self._outputs[:, task_idx]


def load_unified_test_data(process_type, corner_type, data_type='cell', graph_mode='stage_aware'):
    """
    Load unified test data for a process-corner combination.

    Returns:
        UnifiedTestDataset object
    """
    print(f"   Loading unified test data for: {process_type}_{corner_type}")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")

    # Base path for unified datasets (matches split_gnn_dataset.py output)
    base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp"

    # Build path: unified_{process}_{corner}/test_{data_type}_{graph_mode}.pth
    unified_dir = f"unified_{process_type}_{corner_type}"
    filename = f"test_{data_type}_{graph_mode}.pth"
    unified_path = os.path.join(base_path, unified_dir, filename)

    if not os.path.exists(unified_path):
        raise FileNotFoundError(f"Unified test data not found: {unified_path}")

    dataset = UnifiedTestDataset(unified_path, graph_mode)
    return dataset


def run_validation(args, process_type, corner_type, device):
    """
    Run validation for a single process-corner combination.
    """
    print(f"\n{'='*80}")
    print(f"   Running validation for: {process_type}_{corner_type}")
    print(f"{'='*80}")

    # Set mode-dependent default indices if not provided
    indices = args.indices
    if indices is None:
        if args.mode == 'extrapolation':
            indices = [5, 30, 55]
        else:  # interpolation
            indices = [0, 13, 30, 45, 60]

    # For interpolation mode: automatically add endpoints if not present
    if args.mode == 'interpolation':
        middle_indices = sorted(set(indices))
        if 0 not in middle_indices:
            middle_indices = [0] + middle_indices
        if args.total_points - 1 not in middle_indices:
            middle_indices = middle_indices + [args.total_points - 1]
        indices = middle_indices
        print(f"\n   Interpolation mode: Added endpoints to indices -> {indices}")

    # Calculate K, left_bound, right_bound from indices
    k = len(indices)
    left_bound = min(indices)
    right_bound = max(indices) + 1
    middle_idx = len(indices) // 2

    # Validate indices
    if k == 0:
        print(f"   Error: At least one index must be provided")
        return None

    if left_bound < 0 or right_bound > args.total_points:
        print(f"   Error: Indices must be within [0, {args.total_points - 1}]. "
              f"Got indices range [{left_bound}, {max(indices)}]")
        return None

    print(f"\n   Configuration:")
    print(f"   Process: {process_type}")
    print(f"   Corner: {corner_type}")
    print(f"   Model type: {args.model_type}")
    print(f"   Data type: {args.data_type}")
    print(f"   Graph mode: {args.graph_mode}")
    print(f"   Mode: {args.mode}")
    print(f"   Indices: {indices}")
    print(f"   -> Calculated K (support samples): {k}")
    print(f"   -> Calculated left_bound: {left_bound}")
    print(f"   -> Calculated right_bound: {right_bound}")
    print(f"   -> Total points: {args.total_points}")

    # Load unified test data
    print("\n   Loading unified TEST dataset...")
    try:
        test_dataset = load_unified_test_data(
            process_type, corner_type, args.data_type, args.graph_mode
        )
    except Exception as e:
        print(f"   Failed to load test data: {e}")
        return None

    topology_cache = test_dataset.topology_cache
    if topology_cache is None:
        print("   Error: Topology cache not loaded")
        return None

    # Get norm_stats from dataset (pre-computed in split_gnn_dataset.py)
    norm_stats = test_dataset._norm_stats

    # Load pretrained model
    print(f"\n   Loading pretrained {args.model_type.upper()} model...")

    if args.model_path is None:
        # Build model architecture suffix for filename
        arch_suffix = f"_conv{args.conv_hidden_dim}x{args.num_conv_layers}_fc{args.fc_hidden_dim}x{args.num_fc_layers}"

        # Auto-determine model path
        if args.model_type == 'baseline':
            model_dir = "../../../pretrained_models/gnn_baseline_checkpoints"
            model_filename = f"gnn_baseline_{process_type}_{corner_type}_{args.data_type}_{args.graph_mode}_iter{args.num_iterations}{arch_suffix}.pth"
        else:  # maml
            model_dir = "../../../pretrained_models/gnn_maml_final"
            model_filename = f"gnn_maml_{process_type}_{corner_type}_{args.data_type}_{args.graph_mode}_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}.pth"

        model_path = os.path.join(model_dir, model_filename)
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"   Model not found: {model_path}")
        return None

    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    # Use norm_stats from checkpoint (what model was trained with), fallback to dataset
    checkpoint_norm_stats = checkpoint.get('norm_stats', None)
    if checkpoint_norm_stats is not None:
        norm_stats = checkpoint_norm_stats
    # else: use norm_stats from dataset (already set above)

    # Get architecture params from checkpoint config or use command line args
    config = checkpoint.get('config', {})
    conv_hidden_dim = config.get('conv_hidden_dim', args.conv_hidden_dim)
    num_conv_layers = config.get('num_conv_layers', args.num_conv_layers)
    fc_hidden_dim = config.get('fc_hidden_dim', args.fc_hidden_dim)
    num_fc_layers = config.get('num_fc_layers', args.num_fc_layers)

    # Get node_features from checkpoint weight shape
    node_features = checkpoint['model_state_dict']['convs.0.lin.weight'].shape[1]
    print(f"   Detected node_features from checkpoint: {node_features}")

    # Create model
    model = create_maml_gcn_model(
        node_features=node_features,
        pooling='mean',
        output_dim=1,
        dropout=0.0,
        conv_hidden_dim=conv_hidden_dim,
        num_conv_layers=num_conv_layers,
        fc_hidden_dim=fc_hidden_dim,
        num_fc_layers=num_fc_layers
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    print(f"   Loaded model: {model_path}")
    print(f"   Architecture: conv={conv_hidden_dim}x{num_conv_layers}, fc={fc_hidden_dim}x{num_fc_layers}")

    # Process test tasks
    print(f"\n   Starting validation (mode={args.mode}, K={k})...")

    num_tasks = test_dataset.num_tasks
    num_libs = test_dataset.num_libs

    print(f"   Tasks structure: {num_tasks} tasks x {num_libs} lib files = {num_tasks * num_libs} total samples")
    print(f"   Using lazy loading (mmap) - data loaded on demand")

    # Random sampling of test tasks (lazy loading - no bulk data loading)
    num_test_samples = min(args.num_test_samples, num_tasks)
    requested_num_samples = num_test_samples  # Store original requested number for directory naming
    test_indices_list = random.sample(range(num_tasks), num_test_samples)

    print(f"\n   Selected {num_test_samples} random test tasks from {num_tasks} total tasks")

    # Apply continuity filtering if requested
    if args.filter_continuous:
        continuous_indices, discontinuous_indices = filter_continuous_tasks(
            test_dataset, test_indices_list,
            threshold_ratio=args.continuity_threshold,
            verbose=True
        )
        test_indices_list = continuous_indices
        num_test_samples = len(test_indices_list)
        print(f"\n   After filtering: {num_test_samples} continuous tasks (directory uses original: {requested_num_samples})")

    # Process test tasks
    print(f"\n   Processing {num_test_samples} test tasks...")
    adam_condition_count = 0
    total_nrmse = []
    total_extra_l = []
    total_extra_r = []
    total_inter = []
    total_l_mape = []
    total_r_mape = []
    total_in_mape = []
    total_mape = []
    total_l_mae = []
    total_r_mae = []
    total_in_mae = []
    total_mae = []

    all_predictions_global = []
    all_actuals_global = []

    print(f"   Parameter check: K:{k}, middle:{middle_idx}, left_value:{left_bound}, right_value:{right_bound}")

    for i, randomtask in enumerate(test_indices_list):
        if i % 100 == 0:
            print(f"   Processing task {i+1}/{num_test_samples} (index: {randomtask})")

        try:
            # Lazy loading: load task data on demand (mmap efficient)
            task_samples, task_outputs = test_dataset.get_all_libs_for_task(randomtask, clone=True)
            task_outputs_tensor = torch.tensor(task_outputs, dtype=torch.float32)

            # Apply normalization on-the-fly
            for sample in task_samples:
                sample['node_features'] = normalize_node_features(sample['node_features'], norm_stats)

            # Get support set samples (X) and all samples (true_samples)
            X_samples = [task_samples[idx] for idx in indices]
            y = task_outputs_tensor[indices]

            true_samples = task_samples
            true_function = task_outputs_tensor

            # Define regions based on mode
            testdata_inter_output = task_outputs_tensor[left_bound:right_bound]
            y_inter_mean = testdata_inter_output.mean()
            y1_mean = task_outputs_tensor.mean()

            # Only calculate left/right regions for extrapolation mode
            if args.mode == 'extrapolation':
                testdata_rightex_output = task_outputs_tensor[right_bound:]
                testdata_leftex_output = task_outputs_tensor[:left_bound]
                y_leftex_mean = testdata_leftex_output.mean()
                y_rightex_mean = testdata_rightex_output.mean()

            y_mean = y.mean()
            y_std = y.std()

            if y_std > 0:
                y_norm = (y - y_mean) / y_std

                # Create center input
                center_sample = task_samples[indices[0]]
                center_node_features = center_sample['node_features'].clone()
                center_node_features[:, 4] = 0.0

                center_sample_dict = {
                    'node_features': center_node_features,
                    'cell_name': center_sample['cell_name'],
                    'delay_type': center_sample['delay_type']
                }

                if args.graph_mode == 'stage_aware':
                    center_sample_dict['output_name'] = center_sample.get('output_name', '')

                # Get center prediction from model
                cell_name = center_sample_dict['cell_name']
                cell_cache = topology_cache[cell_name]

                if args.graph_mode == 'stage_aware':
                    output_name = center_sample_dict.get('output_name', '')
                    delay_type = center_sample_dict['delay_type']
                    output_topo = cell_cache['output_topologies'][output_name]
                    if 'rise' in delay_type:
                        adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
                    else:
                        adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
                else:
                    adjacency_matrix = cell_cache['adjacency_matrix']

                edge_index = adjacency_matrix.nonzero().t()
                center_data = Data(x=center_node_features, edge_index=edge_index)
                center_batch = Batch.from_data_list([center_data]).to(device)

                with torch.no_grad():
                    center = model(center_batch).item()

                y_max = y_norm[:, 0].max() if len(y_norm.shape) > 1 else y_norm.max()
                y_min = y_norm[:, 0].min() if len(y_norm.shape) > 1 else y_norm.min()

                # Get model predictions for scaling (interpolation region)
                inter_predictions = []
                for idx in range(left_bound, right_bound):
                    sample = task_samples[idx]
                    cell_name = sample['cell_name']
                    cell_cache = topology_cache[cell_name]

                    if args.graph_mode == 'stage_aware':
                        output_name = sample.get('output_name', '')
                        delay_type = sample['delay_type']
                        output_topo = cell_cache['output_topologies'][output_name]
                        if 'rise' in delay_type:
                            adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
                        else:
                            adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
                    else:
                        adjacency_matrix = cell_cache['adjacency_matrix']

                    edge_index = adjacency_matrix.nonzero().t()
                    data = Data(x=sample['node_features'], edge_index=edge_index)
                    batch = Batch.from_data_list([data]).to(device)

                    with torch.no_grad():
                        pred = model(batch).item()
                        inter_predictions.append(pred)

                inter_predictions = torch.tensor(inter_predictions)
                min_val = inter_predictions.min().item()
                max_val = inter_predictions.max().item()

                if abs(max_val - min_val) > 0:
                    grad = (y_max - y_min) / (max_val - min_val)

                    y_norm_middle = y_norm[middle_idx, 0] if len(y_norm.shape) > 1 else y_norm[middle_idx]
                    move = center - y_norm_middle / grad

                    (total_loss1, inter_loss1, leftex_loss1, rightex_loss1,
                    mape_loss1, leftex_mape1, inter_mape1, rightex_mape1,
                    predictions, actual_values, _, _, _, _, _, _, adam_used,
                    mae_loss, mae_l_loss, mae_r_loss, mae_in_loss) = evaluate_model_performance_gnn(
                        model, 'GCN', X_samples, y,
                        true_samples, true_function, grad, move,
                        topology_cache, args.graph_mode, norm_stats, normalize_node_features,
                        left_bound=left_bound, right_bound=right_bound, total_points=args.total_points,
                        mode=args.mode
                    )

                    if adam_used:
                        adam_condition_count += 1

                    all_predictions_global.extend(predictions)
                    all_actuals_global.extend(actual_values)

                    if args.mode == 'extrapolation':
                        nrmse_leftex = (leftex_loss1 ** 0.5) / (abs(y_leftex_mean) + 1e-4) * 100
                        nrmse_rightex = (rightex_loss1 ** 0.5) / (abs(y_rightex_mean) + 1e-4) * 100
                        mape_l_percent = leftex_mape1 * 100
                        mape_r_percent = rightex_mape1 * 100
                    else:
                        nrmse_leftex = 0
                        nrmse_rightex = 0
                        mape_l_percent = 0
                        mape_r_percent = 0

                    nrmse1 = (total_loss1 ** 0.5) / (abs(y1_mean) + 1e-4) * 100
                    nrmse_inter = (inter_loss1 ** 0.5) / (abs(y_inter_mean) + 1e-4) * 100
                    mape_percent = mape_loss1 * 100
                    mape_in_percent = inter_mape1 * 100

                    if not(torch.isinf(nrmse1) or torch.isnan(nrmse1)):
                        total_nrmse.append(nrmse1.item())
                        total_extra_l.append(nrmse_leftex.item() if torch.is_tensor(nrmse_leftex) else nrmse_leftex)
                        total_extra_r.append(nrmse_rightex.item() if torch.is_tensor(nrmse_rightex) else nrmse_rightex)
                        total_inter.append(nrmse_inter.item())
                        total_mape.append(mape_percent)
                        total_r_mape.append(mape_r_percent)
                        total_l_mape.append(mape_l_percent)
                        total_in_mape.append(mape_in_percent)
                        total_mae.append(mae_loss)
                        total_l_mae.append(mae_l_loss)
                        total_r_mae.append(mae_r_loss)
                        total_in_mae.append(mae_in_loss)

                    if i % 100 == 0 and i > 0:
                        print(f"     Adam usage: {(adam_condition_count/len(total_nrmse)):.2f}")
                        print(f"     Current avg NRMSE: {sum(total_nrmse)/len(total_nrmse):.2f}% - Tasks: {len(total_nrmse)}")
                        print(f"     Current avg MAPE: {sum(total_mape)/len(total_mape):.2f}% - Tasks: {len(total_mape)}")
                        print(f"     Current avg MAE: {1000*sum(total_mae)/len(total_mae):.3f}ps - Tasks: {len(total_mae)}")
        except Exception as e:
            if i < 10:
                print(f"   Error processing task {randomtask}: {e}")
            continue

    if len(total_nrmse) == 0:
        print(f"   No valid tasks processed for {process_type}_{corner_type}")
        return None

    print(f"\n   Completed processing {len(total_nrmse)} valid tasks")

    # Calculate final metrics
    results = {
        'process': process_type,
        'corner': corner_type,
        'num_valid_tasks': len(total_nrmse),
        'nrmse_total': sum(total_nrmse)/len(total_nrmse),
        'nrmse_left': sum(total_extra_l)/len(total_extra_l),
        'nrmse_right': sum(total_extra_r)/len(total_extra_r),
        'nrmse_inter': sum(total_inter)/len(total_inter),
        'mape_total': sum(total_mape)/len(total_mape),
        'mape_left': sum(total_l_mape)/len(total_l_mape),
        'mape_right': sum(total_r_mape)/len(total_r_mape),
        'mape_inter': sum(total_in_mape)/len(total_in_mape),
        'mae_total': sum(total_mae)/len(total_mae),
        'mae_left': sum(total_l_mae)/len(total_l_mae),
        'mae_right': sum(total_r_mae)/len(total_r_mae),
        'mae_inter': sum(total_in_mae)/len(total_in_mae),
    }

    # Print results
    print(f"\n   Results for {process_type}_{corner_type}:")
    print(f"   NRMSE (Total/Left/Right/Inter): [{results['nrmse_total']:.2f}, "
          f"{results['nrmse_left']:.2f}, "
          f"{results['nrmse_right']:.2f}, "
          f"{results['nrmse_inter']:.2f}]")
    print(f"   MAPE (Total/Left/Right/Inter): [{results['mape_total']:.2f}, "
          f"{results['mape_left']:.2f}, "
          f"{results['mape_right']:.2f}, "
          f"{results['mape_inter']:.2f}]")
    print(f"   MAE (Total/Left/Right/Inter): [{results['mae_total']:.6f}, "
          f"{results['mae_left']:.6f}, "
          f"{results['mae_right']:.6f}, "
          f"{results['mae_inter']:.6f}]")

    # Save results if requested
    if args.save_results:
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

        # Add _filtered suffix if continuity filtering was applied
        filter_suffix = "_filtered" if args.filter_continuous else ""

        if args.model_type == 'baseline':
            base_name = f"{args.output_prefix}_{process_type}_{corner_type}_{args.data_type}_{args.graph_mode}_{args.mode}_{args.model_type}_iter{args.num_iterations}{arch_suffix}{filter_suffix}"
        else:
            base_name = f"{args.output_prefix}_{process_type}_{corner_type}_{args.data_type}_{args.graph_mode}_{args.mode}_{args.model_type}_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}{filter_suffix}"

        if requested_num_samples == num_tasks:
            pred_filename = f"data_result_npy_directory/{base_name}_pred.npy"
            act_filename = f"data_result_npy_directory/{base_name}_act.npy"
        else:
            pred_filename = f"data_result_npy_directory/{requested_num_samples}samples/{base_name}_pred.npy"
            act_filename = f"data_result_npy_directory/{requested_num_samples}samples/{base_name}_act.npy"

        os.makedirs(os.path.dirname(pred_filename), exist_ok=True)

        np.save(pred_filename, all_predictions_global)
        np.save(act_filename, all_actuals_global)

        print(f"\n   Saved predictions to: {pred_filename}")
        print(f"   Saved actuals to: {act_filename}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Unified GCN Validation (Unified Dataset Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ASAP7_GCN_voltage_validation_unified.py --process LVT --corner FF --model_type baseline --gpu 5
  python ASAP7_GCN_voltage_validation_unified.py --process SRAM --corner TT --model_type maml --mode interpolation
  python ASAP7_GCN_voltage_validation_unified.py --run_all --model_type baseline --gpu 5

  # With continuity filtering (only use continuous tasks, saves with _filtered suffix):
  python ASAP7_GCN_voltage_validation_unified.py --process LVT --corner FF --model_type maml --filter_continuous --save_results
  python ASAP7_GCN_voltage_validation_unified.py --run_all --model_type maml --filter_continuous --continuity_threshold 0.15
        """
    )

    # Required arguments
    parser.add_argument('--process', type=str, default=None,
                       choices=['RVT', 'LVT', 'SLVT', 'SRAM'],
                       help='Process type')
    parser.add_argument('--corner', type=str, default=None,
                       choices=['TT', 'FF', 'SS'],
                       help='Process corner')
    parser.add_argument('--run_all', action='store_true',
                       help='Run validation for all 12 process-corner combinations')

    # Model configuration
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['baseline', 'maml'],
                       help='Model type: baseline or maml')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Custom model checkpoint path (optional)')

    # Model architecture parameters
    parser.add_argument('--conv_hidden_dim', type=int, default=128,
                       help='Convolution layer hidden dimension (default: 128)')
    parser.add_argument('--num_conv_layers', type=int, default=3,
                       help='Number of GCN convolutional layers (default: 3)')
    parser.add_argument('--fc_hidden_dim', type=int, default=128,
                       help='FC layer hidden dimension (default: 128)')
    parser.add_argument('--num_fc_layers', type=int, default=2, choices=[1, 2, 3],
                       help='Number of FC layers (default: 2)')

    # Mode configuration
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'], default='extrapolation',
                       help='Testing mode (default: extrapolation)')

    # Data configuration
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode (default: stage_aware)')

    # Sampling configuration
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                       help='Sampling indices for support set')
    parser.add_argument('--total_points', type=int, default=61,
                       help='Total number of data points per task (default: 61)')
    parser.add_argument('--num_test_samples', type=int, default=100000,
                       help='Number of test samples to process (default: 100000)')

    # GPU configuration
    parser.add_argument('--gpu', type=str, default='2',
                       help='GPU device ID (default: 2)')

    # MAML-specific
    parser.add_argument('--innerdiv', type=int, default=10,
                       help='Inner div for MAML (default: 10)')
    parser.add_argument('--tasks_per_meta_batch', type=int, default=16,
                       help='Tasks per meta batch for MAML (default: 16)')
    parser.add_argument('--inner_steps', type=int, default=1,
                       help='Inner steps for MAML (default: 1)')
    parser.add_argument('--num_iterations', type=int, default=100000,
                       help='Pretraining iterations (default: 100000)')

    # Results saving
    parser.add_argument('--save_results', action='store_true',
                       help='Save prediction results to .npy files')
    parser.add_argument('--output_prefix', type=str, default='GCN_unified',
                       help='Prefix for output files (default: GCN_unified)')

    # Continuity filtering
    parser.add_argument('--filter_continuous', action='store_true',
                       help='Filter test tasks to only use continuous data (adds _filtered suffix to output)')
    parser.add_argument('--continuity_threshold', type=float, default=0.18,
                       help='Threshold ratio for continuity check (default: 0.18)')

    args = parser.parse_args()

    # Validate arguments
    if not args.run_all and (args.process is None or args.corner is None):
        parser.error("--process and --corner are required unless --run_all is specified")

    # GPU is set at the top of file before torch import
    print(f"Using GPU: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # Run all combinations or single combination
    if args.run_all:
        processes = ['RVT', 'LVT', 'SLVT', 'SRAM']
        corners = ['TT', 'FF', 'SS']

        print(f"\n{'#'*80}")
        print(f"# Running ALL 12 process-corner combinations")
        print(f"# Model type: {args.model_type}")
        print(f"# Mode: {args.mode}")
        print(f"# GPU: {args.gpu}")
        print(f"{'#'*80}")

        all_results = []
        successful = 0
        failed = 0

        for process in processes:
            for corner in corners:
                result = run_validation(args, process, corner, device)
                if result is not None:
                    all_results.append(result)
                    successful += 1
                else:
                    failed += 1

        # Print summary table
        print(f"\n{'='*80}")
        print(f"   SUMMARY: All Process-Corner Combinations")
        print(f"{'='*80}")
        print(f"   Successful: {successful}/12, Failed: {failed}/12")
        print()

        if all_results:
            print(f"{'Process':<8} {'Corner':<8} {'Tasks':<8} {'NRMSE%':<10} {'MAPE%':<10} {'MAE(ns)':<12}")
            print("-" * 60)

            for result in all_results:
                print(f"{result['process']:<8} {result['corner']:<8} {result['num_valid_tasks']:<8} "
                      f"{result['nrmse_total']:<10.2f} {result['mape_total']:<10.2f} {result['mae_total']*1e9:<12.4f}")

            avg_nrmse = sum(r['nrmse_total'] for r in all_results) / len(all_results)
            avg_mape = sum(r['mape_total'] for r in all_results) / len(all_results)
            avg_mae = sum(r['mae_total'] for r in all_results) / len(all_results)

            print("-" * 60)
            print(f"{'AVERAGE':<8} {'':<8} {'':<8} {avg_nrmse:<10.2f} {avg_mape:<10.2f} {avg_mae*1e9:<12.4f}")

        return 0 if failed == 0 else 1

    else:
        result = run_validation(args, args.process, args.corner, device)
        return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())
