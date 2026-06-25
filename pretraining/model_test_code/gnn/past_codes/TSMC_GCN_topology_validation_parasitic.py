#!/usr/bin/env python
"""
TSMC GCN Topology Validation Script with Parasitic Capacitance

This script validates GCN models for TSMC Process topology experiments with parasitic cap:
- topology_agnostic: Test on unseen cell topologies
- intra_topology: Test on seen cell topologies with unseen conditions

Uses the TSMC Process dataset with 12D node features (7 base + 4 process params + parasitic_cap).
Evaluates per-cell and aggregates results.

Usage:
  python TSMC_GCN_topology_validation_parasitic.py --experiment intra_topology --model_type maml --gpu 0
  python TSMC_GCN_topology_validation_parasitic.py --experiment topology_agnostic --model_type maml --gpu 1
"""

import os
import sys

# Parse GPU argument before importing torch
def get_gpu_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return '0'

os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_from_args()

print("Importing torch...", flush=True)
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
print("Importing torch_geometric...", flush=True)
from torch_geometric.data import Data, Batch
print("Imports complete.", flush=True)

# Add paths
sys.path.append('../../../model_code/')
sys.path.append('../../../data_processing/gnn/')
sys.path.append('../utils/')

from gnn_maml import create_maml_gcn_model
from gnn_functions import evaluate_model_performance_gnn


def check_output_continuity(data, threshold_ratio=0.18):
    """
    Check if output data is continuous across voltage points.
    Returns (is_continuous, continuity_score, gap_indices, max_jump, max_ratio)
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


def filter_continuous_tasks(cell_dataset, task_indices, threshold_ratio=0.18, verbose=True):
    """
    Filter task indices to only include those with continuous outputs.
    Works with CellTestDataset which has per-cell data.
    """
    continuous_indices = []
    discontinuous_indices = []

    for i, task_idx in enumerate(task_indices):
        task_outputs = cell_dataset.get_task_outputs(task_idx).numpy()
        is_continuous, _, _, _, _ = check_output_continuity(
            task_outputs.reshape(-1, 1), threshold_ratio=threshold_ratio
        )

        if is_continuous:
            continuous_indices.append(task_idx)
        else:
            discontinuous_indices.append(task_idx)

    if verbose:
        print(f"   Continuity filter: {len(continuous_indices)}/{len(task_indices)} tasks passed "
              f"({len(discontinuous_indices)} filtered out)")

    return continuous_indices, discontinuous_indices


# Cell lists for each experiment type (matching dataset_TSMC structure)
INTRA_TOPOLOGY_CELLS = [
    'AN4D0BWP30P140',
    'ND3D0BWP30P140',
    'NR3D1BWP30P140',
    'OR4D0BWP30P140',
    'XNR3D1BWP30P140',
    'XOR3D1BWP30P140',
]

TOPOLOGY_AGNOSTIC_CELLS = ['HA1D0BWP30P140', 'FA1D0BWP30P140', 'IOA21D0BWP30P140', 'IOA21D1BWP30P140',
                          'OA21D0BWP30P140', 'OA21D1BWP30P140', 'OA211D0BWP30P140', 'OA211D1BWP30P140',
                          'IAO21D0BWP30P140', 'IAO21D1BWP30P140', 'AO21D0BWP30P140', 'AO21D1BWP30P140',
                          'AO211D0BWP30P140', 'AO211D1BWP30P140', 'SDFSNQD0BWP30P140', 'DFCNQD1BWP30P140']


def normalize_node_features(node_features, norm_stats):
    """
    Normalize node features using saved statistics.
    Supports both zscore (mean/std) and minmax (min/max/epsilon) normalization.
    For TSMC Process with Parasitic Cap (12D):
    - Columns 0-3: One-hot node type (not normalized)
    - Column 4: voltage (normalized)
    - Column 5: input_slew (normalized)
    - Column 6: output_load (normalized)
    - Columns 7-9: process params param_a, param_b, param_c (NOT normalized - same as training)
    - Column 10: temperature (normalized)
    - Column 11: parasitic_cap (NOT normalized - same as training)
    """
    if norm_stats is None:
        return node_features

    normalized = node_features.clone()

    # Get nested norm_stats if needed
    node_norm_stats = norm_stats.get('node_features', norm_stats)

    # Helper function to apply normalization based on stats structure
    def apply_norm(values, stats):
        if 'method' in stats and stats['method'] == 'minmax_positive':
            # minmax: normalized = epsilon + (x - min) / (max - min) * (1 - epsilon)
            epsilon = stats.get('epsilon', 0.01)
            feat_min, feat_max = stats['min'], stats['max']
            if feat_max > feat_min:
                return epsilon + (values - feat_min) / (feat_max - feat_min) * (1 - epsilon)
            else:
                return torch.ones_like(values) * epsilon
        else:
            # zscore: normalized = (x - mean) / std
            return (values - stats['mean']) / stats['std']

    # Normalize voltage (column 4)
    if 'voltage' in node_norm_stats:
        voltage_mask = normalized[:, 4] != 0
        if voltage_mask.any():
            normalized[voltage_mask, 4] = apply_norm(
                normalized[voltage_mask, 4], node_norm_stats['voltage']
            )

    # Normalize input_slew (column 5)
    if 'input_slew' in node_norm_stats:
        slew_mask = normalized[:, 5] != 0
        if slew_mask.any():
            normalized[slew_mask, 5] = apply_norm(
                normalized[slew_mask, 5], node_norm_stats['input_slew']
            )

    # Normalize output_load (column 6)
    if 'output_load' in node_norm_stats:
        load_mask = normalized[:, 6] != 0
        if load_mask.any():
            normalized[load_mask, 6] = apply_norm(
                normalized[load_mask, 6], node_norm_stats['output_load']
            )

    # Note: param_a (col 7), param_b (col 8), param_c (col 9) are NOT normalized
    # to match training behavior (normalize_node_features_safe doesn't normalize them)

    # Normalize temperature (column 10) if available
    # Detect temp_all vs mos_only mode:
    # - mos_only: only MOS nodes have temperature (non-MOS nodes have temp=0)
    # - temp_all: all nodes have temperature values
    if 'temperature' in node_norm_stats and normalized.shape[1] > 10:
        temp_values = normalized[:, 10]
        temp_stats = node_norm_stats['temperature']
        mosfet_mask = normalized[:, 2] != 0  # MOSFET nodes (PMOS=+1, NMOS=-1)
        non_mosfet_mask = normalized[:, 2] == 0  # Non-MOSFET nodes

        # Check if mode is stored in norm_stats, otherwise detect from data
        if 'mode' in temp_stats:
            is_temp_all = temp_stats['mode'] == 'temp_all'
        else:
            # Fallback: detect from data (check if non-MOS nodes have temp values)
            non_mos_temps = temp_values[non_mosfet_mask]
            is_temp_all = non_mos_temps.abs().max() > 1e-6 if non_mosfet_mask.any() else False

        if is_temp_all:
            # temp_all mode: normalize all nodes
            normalized[:, 10] = apply_norm(temp_values, temp_stats)
        else:
            # mos_only mode: only normalize MOS nodes
            if mosfet_mask.any():
                normalized[mosfet_mask, 10] = apply_norm(
                    normalized[mosfet_mask, 10], temp_stats
                )

    # Note: parasitic_cap (col 11) is NOT normalized
    # to match training behavior (normalize_node_features_safe doesn't normalize it)

    return normalized


class CellTestDataset:
    """
    Dataset class for loading per-cell test data with mmap.
    For TSMC Process Parasitic dataset (12D node features).
    """
    def __init__(self, cell_path, topology_cache=None):
        self.cell_path = cell_path
        self.topology_cache = topology_cache
        self._load_data()

    def _load_data(self):
        """Load cell test data with mmap"""
        data = torch.load(self.cell_path, weights_only=False, map_location='cpu', mmap=True)

        self._node_features = data['node_features']  # [num_libs, total_nodes, num_features]
        self._outputs = data['outputs']  # [num_libs, num_tasks]
        self._node_slices = data['node_slices']  # [num_tasks + 1]
        self._cell_name = data['cell_name']
        self._delay_types = data.get('delay_types', None)  # [num_tasks]
        self._output_names = data.get('output_names', None)  # [num_tasks]

        self.num_libs = data['num_libs']
        self.num_tasks = data['num_tasks']
        self.total_nodes = data['total_nodes']
        self.cell_name = data['cell_name']

    def get_task_data(self, task_idx, lib_idx, clone=True):
        """Get data for a specific task and lib."""
        node_start = self._node_slices[task_idx].item()
        node_end = self._node_slices[task_idx + 1].item()

        node_features = self._node_features[lib_idx, node_start:node_end, :]
        if clone:
            node_features = node_features.clone()

        output = self._outputs[lib_idx, task_idx].item()

        # Get delay_type and output_name for this task
        delay_type = 'rise'  # Default
        output_name = ''  # Default
        if self._delay_types is not None:
            delay_type = self._delay_types[task_idx]
        if self._output_names is not None:
            output_name = self._output_names[task_idx]

        return {
            'node_features': node_features,
            'output': output,
            'cell_name': self._cell_name,
            'delay_type': delay_type,
            'output_name': output_name,
        }

    def get_all_libs_for_task(self, task_idx, clone=True):
        """Get data for all libs for a specific task."""
        samples = []
        outputs = []
        for lib_idx in range(self.num_libs):
            sample = self.get_task_data(task_idx, lib_idx, clone=clone)
            samples.append(sample)
            outputs.append(sample['output'])
        return samples, outputs

    def get_task_outputs(self, task_idx):
        """Get all lib outputs for a specific task (for continuity checking)."""
        return self._outputs[:, task_idx]


def run_cell_validation(args, cell_name, topology_cache, norm_stats, model, device):
    """
    Run validation for a single cell.
    """
    print(f"\n   Processing cell: {cell_name}")

    # Load cell test data (directory includes data_type to separate cell vs transition)
    cell_path = os.path.join(
        args.dataset_dir,
        f"test_by_{args.data_type}_{args.graph_mode}",
        f"{cell_name}.pth"
    )

    if not os.path.exists(cell_path):
        print(f"   Cell data not found: {cell_path}")
        return None

    cell_dataset = CellTestDataset(cell_path, topology_cache)
    print(f"   Loaded: {cell_dataset.num_tasks} tasks, {cell_dataset.num_libs} libs")

    # Set mode-dependent default indices
    indices = args.indices
    if indices is None:
        if args.mode == 'extrapolation':
            indices = [5, 30, 55]
        else:  # interpolation
            indices = [0, 13, 30, 45, 60]

    # For interpolation mode: add endpoints
    if args.mode == 'interpolation':
        middle_indices = sorted(set(indices))
        if 0 not in middle_indices:
            middle_indices = [0] + middle_indices
        if args.total_points - 1 not in middle_indices:
            middle_indices = middle_indices + [args.total_points - 1]
        indices = middle_indices

    k = len(indices)
    left_bound = min(indices)
    right_bound = max(indices) + 1
    middle_idx = len(indices) // 2

    # Validate indices
    if k == 0 or left_bound < 0 or right_bound > args.total_points:
        print(f"   Invalid indices configuration")
        return None

    # Random sampling of test tasks
    num_tasks = cell_dataset.num_tasks
    num_test_samples = min(args.num_test_samples, num_tasks)
    test_indices_list = random.sample(range(num_tasks), num_test_samples)

    # Apply continuity filter if enabled
    if args.filter_continuous:
        continuous_indices, discontinuous_indices = filter_continuous_tasks(
            cell_dataset, test_indices_list,
            threshold_ratio=args.continuity_threshold, verbose=True
        )
        test_indices_list = continuous_indices
        num_test_samples = len(test_indices_list)

        if num_test_samples == 0:
            print(f"   No continuous tasks found for {cell_name}")
            return None

    # Metrics collections
    total_nrmse = []
    total_extra_l = []
    total_extra_r = []
    total_inter = []
    total_l_mape = []
    total_r_mape = []
    total_in_mape = []
    total_mape = []
    total_l_rmse = []
    total_r_rmse = []
    total_in_rmse = []
    total_rmse = []
    all_predictions = []
    all_actuals = []
    adam_condition_count = 0

    for i, randomtask in enumerate(test_indices_list):
        if i % 100 == 0:
            print(f"   Processing task {i+1}/{num_test_samples} (index: {randomtask})", flush=True)

        try:
            # Load task data
            task_samples, task_outputs = cell_dataset.get_all_libs_for_task(randomtask, clone=True)
            task_outputs_tensor = torch.tensor(task_outputs, dtype=torch.float32)

            # Apply normalization
            for sample in task_samples:
                sample['node_features'] = normalize_node_features(sample['node_features'], norm_stats)

            # Get support set samples
            X_samples = [task_samples[idx] for idx in indices]
            y = task_outputs_tensor[indices]

            true_samples = task_samples
            true_function = task_outputs_tensor

            # Define regions and calculate ranges for NRMSE normalization
            testdata_inter_output = task_outputs_tensor[left_bound:right_bound]
            y_inter_range = testdata_inter_output.max() - testdata_inter_output.min()
            y1_range = task_outputs_tensor.max() - task_outputs_tensor.min()

            if args.mode == 'extrapolation':
                testdata_rightex_output = task_outputs_tensor[right_bound:]
                testdata_leftex_output = task_outputs_tensor[:left_bound]
                y_leftex_range = testdata_leftex_output.max() - testdata_leftex_output.min()
                y_rightex_range = testdata_rightex_output.max() - testdata_rightex_output.min()

            y_mean = y.mean()
            y_std = y.std()

            if y_std > 0:
                y_norm = (y - y_mean) / y_std

                # Create center input
                center_sample = task_samples[indices[0]]
                center_node_features = center_sample['node_features'].clone()
                center_node_features[:, 4] = 0.0  # Set voltage to 0

                # Get adjacency matrix from topology cache
                cell_cache = topology_cache[cell_name]

                if args.graph_mode == 'stage_aware':
                    output_name = center_sample.get('output_name', '')
                    delay_type = center_sample.get('delay_type', 'rise')

                    # If output_name is empty, get first output from topology cache
                    if not output_name and 'output_topologies' in cell_cache:
                        available_outputs = list(cell_cache['output_topologies'].keys())
                        if available_outputs:
                            output_name = available_outputs[0]

                    if 'output_topologies' in cell_cache and output_name in cell_cache['output_topologies']:
                        output_topo = cell_cache['output_topologies'][output_name]
                        if 'rise' in delay_type:
                            adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
                        else:
                            adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
                    else:
                        adjacency_matrix = cell_cache.get('adjacency_matrix', torch.zeros((1, 1)))
                else:
                    adjacency_matrix = cell_cache['adjacency_matrix']

                edge_index = adjacency_matrix.nonzero().t()
                center_data = Data(x=center_node_features, edge_index=edge_index)
                center_batch = Batch.from_data_list([center_data]).to(device)

                with torch.no_grad():
                    center = model(center_batch).item()

                y_max = y_norm.max().item()
                y_min = y_norm.min().item()

                # Get model predictions for scaling
                inter_predictions = []
                for idx in range(left_bound, right_bound):
                    sample = task_samples[idx]
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
                    y_norm_middle = y_norm[middle_idx].item()
                    move = center - y_norm_middle / grad

                    (total_loss1, inter_loss1, leftex_loss1, rightex_loss1,
                     mape_loss1, leftex_mape1, inter_mape1, rightex_mape1,
                     predictions, actual_values, _, _, _, _, _, _, adam_used,
                     rmse_loss, rmse_l_loss, rmse_r_loss, rmse_in_loss) = evaluate_model_performance_gnn(
                        model, 'GCN', X_samples, y,
                        true_samples, true_function, grad, move,
                        topology_cache, args.graph_mode, norm_stats, normalize_node_features,
                        left_bound=left_bound, right_bound=right_bound, total_points=args.total_points,
                        mode=args.mode
                    )

                    if adam_used:
                        adam_condition_count += 1

                    all_predictions.extend(predictions)
                    all_actuals.extend(actual_values)

                    if args.mode == 'extrapolation':
                        nrmse_leftex = (leftex_loss1 ** 0.5) / (abs(y_leftex_range) + 1e-4) * 100
                        nrmse_rightex = (rightex_loss1 ** 0.5) / (abs(y_rightex_range) + 1e-4) * 100
                        mape_l_percent = leftex_mape1 * 100
                        mape_r_percent = rightex_mape1 * 100
                    else:
                        nrmse_leftex = 0
                        nrmse_rightex = 0
                        mape_l_percent = 0
                        mape_r_percent = 0

                    nrmse1 = (total_loss1 ** 0.5) / (abs(y1_range) + 1e-4) * 100
                    nrmse_inter = (inter_loss1 ** 0.5) / (abs(y_inter_range) + 1e-4) * 100
                    mape_percent = mape_loss1 * 100
                    mape_in_percent = inter_mape1 * 100

                    if not (torch.isinf(nrmse1) or torch.isnan(nrmse1)):
                        total_nrmse.append(nrmse1.item())
                        total_extra_l.append(nrmse_leftex.item() if torch.is_tensor(nrmse_leftex) else nrmse_leftex)
                        total_extra_r.append(nrmse_rightex.item() if torch.is_tensor(nrmse_rightex) else nrmse_rightex)
                        total_inter.append(nrmse_inter.item())
                        total_mape.append(mape_percent)
                        total_r_mape.append(mape_r_percent)
                        total_l_mape.append(mape_l_percent)
                        total_in_mape.append(mape_in_percent)
                        total_rmse.append(rmse_loss)
                        total_l_rmse.append(rmse_l_loss)
                        total_r_rmse.append(rmse_r_loss)
                        total_in_rmse.append(rmse_in_loss)

                    if i % 100 == 0 and i > 0:
                        print(f"     Adam usage: {(adam_condition_count/len(total_nrmse)):.2f}", flush=True)
                        print(f"     Current avg NRMSE: {sum(total_nrmse)/len(total_nrmse):.2f}% - Tasks: {len(total_nrmse)}", flush=True)
                        print(f"     Current avg MAPE: {sum(total_mape)/len(total_mape):.2f}% - Tasks: {len(total_mape)}", flush=True)
                        print(f"     Current avg RMSE: {1000*sum(total_rmse)/len(total_rmse):.3f}ps - Tasks: {len(total_rmse)}", flush=True)

        except Exception as e:
            if i < 5:
                print(f"     Error task {randomtask}: {e}")
            continue

    if len(total_nrmse) == 0:
        print(f"   No valid tasks for {cell_name}")
        return None

    # Calculate final metrics
    results = {
        'cell_name': cell_name,
        'num_valid_tasks': len(total_nrmse),
        'nrmse_total': sum(total_nrmse) / len(total_nrmse),
        'nrmse_left': sum(total_extra_l) / len(total_extra_l),
        'nrmse_right': sum(total_extra_r) / len(total_extra_r),
        'nrmse_inter': sum(total_inter) / len(total_inter),
        'mape_total': sum(total_mape) / len(total_mape),
        'mape_left': sum(total_l_mape) / len(total_l_mape),
        'mape_right': sum(total_r_mape) / len(total_r_mape),
        'mape_inter': sum(total_in_mape) / len(total_in_mape),
        'rmse_total': sum(total_rmse) / len(total_rmse),
        'rmse_left': sum(total_l_rmse) / len(total_l_rmse),
        'rmse_right': sum(total_r_rmse) / len(total_r_rmse),
        'rmse_inter': sum(total_in_rmse) / len(total_in_rmse),
        'predictions': all_predictions,
        'actuals': all_actuals,
    }

    print(f"   {cell_name}: NRMSE={results['nrmse_total']:.2f}%, MAPE={results['mape_total']:.2f}%, "
          f"RMSE={results['rmse_total']*1e9:.4f}ns ({len(total_nrmse)} tasks)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='TSMC GCN Topology Validation with Parasitic Cap (12D features)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TSMC_GCN_topology_validation_parasitic.py --experiment intra_topology --model_type maml --gpu 0
  python TSMC_GCN_topology_validation_parasitic.py --experiment topology_agnostic --model_type maml --gpu 1
  python TSMC_GCN_topology_validation_parasitic.py --experiment intra_topology --model_type maml --mode interpolation
        """
    )

    # Required arguments
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['intra_topology', 'topology_agnostic'],
                        help='Experiment type')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['baseline', 'maml'],
                        help='Model type: baseline or maml')

    # Model configuration
    parser.add_argument('--model_path', type=str, default=None,
                        help='Custom model checkpoint path')
    parser.add_argument('--conv_hidden_dim', type=int, default=32,
                        help='Convolution layer hidden dimension (default: 32)')
    parser.add_argument('--num_conv_layers', type=int, default=2,
                        help='Number of GCN convolutional layers (default: 2)')
    parser.add_argument('--fc_hidden_dim', type=int, default=512,
                        help='FC layer hidden dimension (default: 512)')
    parser.add_argument('--num_fc_layers', type=int, default=2,
                        help='Number of FC layers (default: 2)')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'add', 'output'],
                        help='Pooling method: mean, max, add, or output (output-node-only) (default: mean)')

    # Mode configuration
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'],
                        default='extrapolation', help='Testing mode (default: extrapolation)')

    # Data configuration - TSMC Process Parasitic dataset
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN_unified/with_parasitic_cap',
                        help='Dataset directory (default: TSMC Process Parasitic dataset)')
    parser.add_argument('--data_type', type=str, default='cell',
                        choices=['cell', 'transition'],
                        help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='full_graph',
                        choices=['stage_aware', 'full_graph'],
                        help='Graph mode (default: full_graph)')

    # Sampling configuration
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Sampling indices for support set')
    parser.add_argument('--total_points', type=int, default=61,
                        help='Total number of data points per task (default: 61)')
    parser.add_argument('--num_test_samples', type=int, default=100000,
                        help='Number of test samples per cell (default: 100000)')

    # GPU configuration
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID (default: 0)')

    # MAML-specific
    parser.add_argument('--innerdiv', type=int, default=10,
                        help='Inner div for MAML (default: 10)')
    parser.add_argument('--tasks_per_meta_batch', type=int, default=16,
                        help='Tasks per meta batch for MAML (default: 16)')
    parser.add_argument('--inner_steps', type=int, default=1,
                        help='Inner steps for MAML (default: 1)')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='Pretraining iterations (default: 300000)')

    # Results saving
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to .npy files')
    parser.add_argument('--output_prefix', type=str, default='TSMC_GCN_parasitic',
                        help='Prefix for output files')

    # Cell selection
    parser.add_argument('--cells', type=str, nargs='+', default=None,
                        help='Specific cells to test (default: all cells for experiment type)')

    # Continuity filtering
    parser.add_argument('--filter_continuous', action='store_true',
                        help='Filter test tasks to only use continuous data (adds _filtered suffix to output)')
    parser.add_argument('--continuity_threshold', type=float, default=0.18,
                        help='Threshold ratio for continuity check (default: 0.18)')

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {args.gpu}")
    print(f"Device: {device}")

    # Get cell list based on experiment type
    if args.cells is not None:
        cell_list = args.cells
    elif args.experiment == 'intra_topology':
        cell_list = INTRA_TOPOLOGY_CELLS
    else:
        cell_list = TOPOLOGY_AGNOSTIC_CELLS

    print(f"\n{'='*80}")
    print(f"TSMC GCN Topology Validation with Parasitic Cap")
    print(f"{'='*80}")
    print(f"Experiment: {args.experiment}")
    print(f"Model type: {args.model_type}")
    print(f"Graph mode: {args.graph_mode}")
    print(f"Data type: {args.data_type}")
    print(f"Mode: {args.mode}")
    print(f"Pooling: {args.pooling}")
    print(f"Node features: 12D (7 base + 4 process params + parasitic_cap)")
    print(f"Cells: {len(cell_list)} cells")
    print(f"{'='*80}")

    # Load train data to get norm_stats and cache_path
    train_path = os.path.join(args.dataset_dir, f"train_{args.data_type}_{args.graph_mode}.pth")
    print(f"\nLoading train data for norm_stats: {train_path}", flush=True)

    if not os.path.exists(train_path):
        print(f"Train data not found: {train_path}")
        return 1

    print("Loading train_data with mmap...", flush=True)
    train_data = torch.load(train_path, weights_only=False, map_location='cpu', mmap=True)
    print("train_data loaded.", flush=True)
    norm_stats = train_data.get('norm_stats', None)
    cache_path = train_data.get('cache_path', None)

    print(f"Norm stats loaded: {norm_stats is not None}")

    # Load topology cache
    if cache_path:
        if cache_path.startswith('/mnt/home/'):
            cache_path = cache_path.replace('/mnt/home/', '/home/')
        if not os.path.exists(cache_path):
            cache_filename = os.path.basename(cache_path)
            cache_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"

    if cache_path and os.path.exists(cache_path):
        print(f"Loading topology cache: {cache_path}", flush=True)
        topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu', mmap=True)
        print(f"Loaded topology cache for {len(topology_cache)} cells", flush=True)
    else:
        print(f"Topology cache not found: {cache_path}")
        return 1

    # Load model - TSMC Process Parasitic version
    print(f"\nLoading {args.model_type.upper()} model...")

    if args.model_path is None:
        # Add pooling suffix only if not 'mean' (default)
        pool_suffix = f"_pool{args.pooling}" if args.pooling != 'mean' else ""
        arch_suffix = f"_conv{args.conv_hidden_dim}x{args.num_conv_layers}_fc{args.fc_hidden_dim}x{args.num_fc_layers}{pool_suffix}"

        if args.model_type == 'baseline':
            model_dir = "../../../pretrained_models/gnn_baseline_tsmc_parasitic_checkpoints"
            model_filename = f"gnn_baseline_tsmc_parasitic_{args.data_type}_{args.graph_mode}_iter{args.num_iterations}{arch_suffix}.pth"
        else:
            model_dir = "../../../pretrained_models/gnn_maml_tsmc_parasitic_final"
            model_filename = f"gnn_maml_tsmc_parasitic_{args.data_type}_{args.graph_mode}_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}.pth"

        model_path = os.path.join(model_dir, model_filename)
    else:
        model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please provide --model_path or ensure the model exists")
        return 1

    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    # Use norm_stats from checkpoint if available
    checkpoint_norm_stats = checkpoint.get('norm_stats', None)
    if checkpoint_norm_stats is not None:
        norm_stats = checkpoint_norm_stats
        print("Using norm_stats from checkpoint")

    # Get architecture params from checkpoint
    config = checkpoint.get('config', {})
    conv_hidden_dim = config.get('conv_hidden_dim', args.conv_hidden_dim)
    num_conv_layers = config.get('num_conv_layers', args.num_conv_layers)
    fc_hidden_dim = config.get('fc_hidden_dim', args.fc_hidden_dim)
    num_fc_layers = config.get('num_fc_layers', args.num_fc_layers)

    # Get node_features from checkpoint weight shape
    node_features = checkpoint['model_state_dict']['convs.0.lin.weight'].shape[1]
    print(f"Detected node_features from checkpoint: {node_features}")

    # Get pooling from checkpoint config or use argument
    pooling_mode = config.get('pooling', args.pooling)
    print(f"Pooling mode: {pooling_mode}")

    # Create model
    model = create_maml_gcn_model(
        node_features=node_features,
        pooling=pooling_mode,
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

    print(f"Loaded model: {model_path}")
    print(f"Architecture: conv={conv_hidden_dim}x{num_conv_layers}, fc={fc_hidden_dim}x{num_fc_layers}, pooling={pooling_mode}")

    # Process each cell
    all_results = []
    successful = 0
    failed = 0

    for cell_name in cell_list:
        result = run_cell_validation(args, cell_name, topology_cache, norm_stats, model, device)
        if result is not None:
            all_results.append(result)
            successful += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.experiment}")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(cell_list)}, Failed: {failed}/{len(cell_list)}")
    print()

    if all_results:
        print(f"{'Cell':<25} {'Tasks':<8} {'NRMSE%':<10} {'MAPE%':<10} {'RMSE(ns)':<12}")
        print("-" * 70)

        for result in all_results:
            print(f"{result['cell_name']:<25} {result['num_valid_tasks']:<8} "
                  f"{result['nrmse_total']:<10.2f} {result['mape_total']:<10.2f} "
                  f"{result['rmse_total']*1e9:<12.4f}")

        # Calculate averages
        avg_nrmse = sum(r['nrmse_total'] for r in all_results) / len(all_results)
        avg_mape = sum(r['mape_total'] for r in all_results) / len(all_results)
        avg_rmse = sum(r['rmse_total'] for r in all_results) / len(all_results)
        total_tasks = sum(r['num_valid_tasks'] for r in all_results)

        print("-" * 70)
        print(f"{'AVERAGE':<25} {total_tasks:<8} {avg_nrmse:<10.2f} {avg_mape:<10.2f} {avg_rmse*1e9:<12.4f}")

        # Print region-specific metrics
        print(f"\nRegion-specific metrics (Average across all cells):")
        print(f"  NRMSE - Left: {sum(r['nrmse_left'] for r in all_results)/len(all_results):.2f}%, "
              f"Right: {sum(r['nrmse_right'] for r in all_results)/len(all_results):.2f}%, "
              f"Inter: {sum(r['nrmse_inter'] for r in all_results)/len(all_results):.2f}%")
        print(f"  MAPE  - Left: {sum(r['mape_left'] for r in all_results)/len(all_results):.2f}%, "
              f"Right: {sum(r['mape_right'] for r in all_results)/len(all_results):.2f}%, "
              f"Inter: {sum(r['mape_inter'] for r in all_results)/len(all_results):.2f}%")
        print(f"  RMSE  - Left: {sum(r['rmse_left'] for r in all_results)/len(all_results)*1e9:.4f}ns, "
              f"Right: {sum(r['rmse_right'] for r in all_results)/len(all_results)*1e9:.4f}ns, "
              f"Inter: {sum(r['rmse_inter'] for r in all_results)/len(all_results)*1e9:.4f}ns")

    # Save results if requested
    if args.save_results and all_results:
        pool_suffix = f"_pool{pooling_mode}" if pooling_mode != 'mean' else ""
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}{pool_suffix}"
        filter_suffix = "_filtered" if args.filter_continuous else ""

        for result in all_results:
            cell_name = result['cell_name']

            if args.model_type == 'baseline':
                base_name = f"{args.output_prefix}_{args.experiment}_{cell_name}_{args.data_type}_{args.graph_mode}_{args.mode}_{args.model_type}_iter{args.num_iterations}{arch_suffix}{filter_suffix}"
            else:
                base_name = f"{args.output_prefix}_{args.experiment}_{cell_name}_{args.data_type}_{args.graph_mode}_{args.mode}_{args.model_type}_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}{filter_suffix}"

            pred_filename = f"data_result_npy_directory/{base_name}_pred.npy"
            act_filename = f"data_result_npy_directory/{base_name}_act.npy"

            os.makedirs("data_result_npy_directory", exist_ok=True)

            np.save(pred_filename, result['predictions'])
            np.save(act_filename, result['actuals'])

        print(f"\nSaved results to data_result_npy_directory/")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
