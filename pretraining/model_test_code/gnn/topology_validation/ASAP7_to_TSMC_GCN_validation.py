#!/usr/bin/env python
"""
ASAP7 to TSMC Cross-PDK GCN Validation Script

This script validates GCN models trained on the ASAP7 Process dataset using
TSMC test cells. Cross-PDK evaluation: ASAP7 model -> TSMC test data.

Uses the TSMC dataset with 11D node features (column layout shared with ASAP7).
Loads ASAP7-trained model and evaluates per-cell on TSMC data.

Usage:
  python ASAP7_to_TSMC_GCN_validation.py --experiment intra_topology --model_type maml --gpu 0
  python ASAP7_to_TSMC_GCN_validation.py --experiment topology_agnostic --model_type maml --gpu 1
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


# Cell lists for each experiment type (TSMC dataset)
INTRA_TOPOLOGY_CELLS = [
    'AN4D0BWP30P140',
    'ND3D0BWP30P140',
    'NR3D1BWP30P140',
    'OR4D0BWP30P140',
    'XNR3D1BWP30P140',
    'XOR3D1BWP30P140',
]

TOPOLOGY_AGNOSTIC_CELLS = ['OA21D0BWP30P140', 'OA21D1BWP30P140', 'OA211D0BWP30P140', 'OA211D1BWP30P140',
                          'IOA21D0BWP30P140', 'IOA21D1BWP30P140', 'HA1D0BWP30P140', 'FA1D0BWP30P140',
                          'IAO21D0BWP30P140', 'IAO21D1BWP30P140', 'AO21D0BWP30P140', 'AO21D1BWP30P140',
                          'AO211D0BWP30P140', 'AO211D1BWP30P140', 'SDFSNQD0BWP30P140', 'DFCNQD1BWP30P140']


def apply_pdk_scale(node_features, output_value, scale_factor, voltage_shift=0.0):
    """
    Apply PDK unit scaling to convert TSMC units (ns/pf) to ASAP7 units (ps/ff).

    TSMC: time_unit = 1ns, capacitive_load_unit = 1pf, nominal_voltage = 0.9V
    ASAP7: time_unit = 1ps, capacitive_load_unit = 1ff, nominal_voltage = 0.7V

    To match ASAP7 scale (so the ASAP7-trained model sees inputs in its native range):
    - scale_factor = 1000.0 for time/cap (ns -> ps, pf -> ff)
    - voltage_shift = -0.2 to shift TSMC 0.9V -> ASAP7 0.7V scale

    Applies to:
    - Column 4: voltage (shifted by voltage_shift)
    - Column 5: input_slew (scaled by scale_factor)
    - Column 6: output_load (scaled by scale_factor)
    - output_value: delay output (scaled by scale_factor)
    """
    if scale_factor == 1.0 and voltage_shift == 0.0:
        return node_features, output_value

    scaled_features = node_features.clone()

    # Shift voltage (column 4) - TSMC 0.9V nominal -> ASAP7 0.7V nominal
    if voltage_shift != 0.0:
        voltage_mask = scaled_features[:, 4] != 0
        if voltage_mask.any():
            scaled_features[voltage_mask, 4] = scaled_features[voltage_mask, 4] + voltage_shift

    # Scale input_slew (column 5) - time unit
    if scale_factor != 1.0:
        slew_mask = scaled_features[:, 5] != 0
        if slew_mask.any():
            scaled_features[slew_mask, 5] = scaled_features[slew_mask, 5] * scale_factor

        # Scale output_load (column 6) - capacitance unit
        load_mask = scaled_features[:, 6] != 0
        if load_mask.any():
            scaled_features[load_mask, 6] = scaled_features[load_mask, 6] * scale_factor

        # Scale output value - time unit
        scaled_output = output_value * scale_factor
    else:
        scaled_output = output_value

    return scaled_features, scaled_output


def normalize_node_features(node_features, norm_stats):
    """
    Normalize node features using saved statistics.
    For ASAP7/TSMC Process (11D):
    - Columns 0-3: One-hot node type (not normalized)
    - Column 4: voltage
    - Column 5: input_slew
    - Column 6: output_load
    - Columns 7-10: process params (param_a, param_b, param_c, temperature)
    Supports both zscore (mean/std) and minmax (min/max/epsilon) normalization.
    """
    if norm_stats is None:
        return node_features

    normalized = node_features.clone()

    # Get nested norm_stats if needed
    node_norm = norm_stats.get('node_features', norm_stats)

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
    if 'voltage' in node_norm:
        voltage_mask = normalized[:, 4] != 0
        if voltage_mask.any():
            normalized[voltage_mask, 4] = apply_norm(
                normalized[voltage_mask, 4], node_norm['voltage']
            )

    # Normalize input_slew (column 5)
    if 'input_slew' in node_norm:
        slew_mask = normalized[:, 5] != 0
        if slew_mask.any():
            normalized[slew_mask, 5] = apply_norm(
                normalized[slew_mask, 5], node_norm['input_slew']
            )

    # Normalize output_load (column 6)
    if 'output_load' in node_norm:
        load_mask = normalized[:, 6] != 0
        if load_mask.any():
            normalized[load_mask, 6] = apply_norm(
                normalized[load_mask, 6], node_norm['output_load']
            )

    # Normalize process params if available (columns 7-9: param_a, param_b, param_c)
    process_params = ['param_a', 'param_b', 'param_c']
    for i, param_name in enumerate(process_params):
        col_idx = 7 + i
        if param_name in node_norm and normalized.shape[1] > col_idx:
            param_mask = normalized[:, col_idx] != 0
            if param_mask.any():
                normalized[param_mask, col_idx] = apply_norm(
                    normalized[param_mask, col_idx], node_norm[param_name]
                )

    # Normalize temperature (column 10) if available
    if 'temperature' in node_norm and normalized.shape[1] > 10:
        temp_values = normalized[:, 10]
        temp_stats = node_norm['temperature']
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

    return normalized


class CellTestDataset:
    """
    Dataset class for loading per-cell test data with mmap (TSMC format).

    Data format:
    - node_features: [num_libs, total_nodes, num_features]
    - outputs: [num_libs, num_tasks]
    - node_slices: [num_tasks + 1] - cumulative node offsets per task
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


def run_cell_validation(args, cell_name, topology_cache, norm_stats, model, device,
                        pdk_scale_factor=1.0, voltage_shift=0.0):
    """
    Run validation for a single cell.

    Args:
        pdk_scale_factor: Scale factor for TSMC units (ns/pf) to ASAP7 units (ps/ff).
                          Use 1000.0 for cross-PDK validation.
        voltage_shift: Voltage shift to align TSMC (0.9V) to ASAP7 (0.7V) scale.
                       Use -0.2 for cross-PDK validation.
    """
    print(f"\n   Processing cell: {cell_name}")

    # Load cell test data - TSMC specific path pattern
    # Build suffixes for test directory (must match dataset generation, TSMC naming)
    topology_suffix = "_inputport" if args.inputport else ""
    voltage_suffix = f"_{args.voltage_mode}" if args.voltage_mode != "all_nodes" else ""
    temp_suffix = "_temp_all" if args.temp_mode == "temp_all" else ""
    slew_suffix = "_relpin" if args.related_pin_only else ""
    test_dir = f"test_by_{args.data_type}_{args.graph_mode}{topology_suffix}{voltage_suffix}{temp_suffix}{slew_suffix}"
    cell_path = os.path.join(
        args.tsmc_dataset_dir,
        test_dir,
        f"{cell_name}.pth"
    )
    print(f"   Test directory: {test_dir}")
    print(f"   Cell path: {cell_path}")

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

            # Debug: Print info for first task
            if i < 5:
                print(f"      task debug info:")
                print(f"       Num samples: {len(task_samples)}")
                if len(task_samples) > 0:
                    sample = task_samples[0]
                    print(f"       Node features shape: {sample['node_features'].shape}")
                    if pdk_scale_factor != 1.0 or voltage_shift != 0.0:
                        print(f"       PDK scale factor: {pdk_scale_factor}, Voltage shift: {voltage_shift}")
                        voltage_nz = sample['node_features'][:, 4][sample['node_features'][:, 4] != 0]
                        if len(voltage_nz) > 0:
                            print(f"       Original voltage range: {voltage_nz.min():.4f} - {voltage_nz.max():.4f}")
                        print(f"       Original slew range: {sample['node_features'][:, 5].min():.6f} - {sample['node_features'][:, 5].max():.6f}")
                        print(f"       Original load range: {sample['node_features'][:, 6].min():.9f} - {sample['node_features'][:, 6].max():.9f}")
                        print(f"       Original output range: {task_outputs_tensor.min():.6f} - {task_outputs_tensor.max():.6f}")

            # Apply PDK unit scaling BEFORE normalization (TSMC ns/pf -> ASAP7 ps/ff scale)
            if pdk_scale_factor != 1.0 or voltage_shift != 0.0:
                for j, sample in enumerate(task_samples):
                    scaled_features, scaled_output = apply_pdk_scale(
                        sample['node_features'], task_outputs[j], pdk_scale_factor, voltage_shift
                    )
                    sample['node_features'] = scaled_features
                    task_outputs[j] = scaled_output

                # Update tensor with scaled outputs
                task_outputs_tensor = torch.tensor(task_outputs, dtype=torch.float32)

                if i < 5:
                    sample = task_samples[0]
                    voltage_nz = sample['node_features'][:, 4][sample['node_features'][:, 4] != 0]
                    if len(voltage_nz) > 0:
                        print(f"       Shifted voltage range: {voltage_nz.min():.4f} - {voltage_nz.max():.4f}")
                    print(f"       Scaled slew range: {sample['node_features'][:, 5].min():.4f} - {sample['node_features'][:, 5].max():.4f}")
                    print(f"       Scaled load range: {sample['node_features'][:, 6].min():.6f} - {sample['node_features'][:, 6].max():.6f}")
                    print(f"       Scaled output range: {task_outputs_tensor.min():.4f} - {task_outputs_tensor.max():.4f}")

            # Apply normalization (default: ASAP7 source norm_stats from checkpoint)
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

                # Create center input with nominal voltage (0.9V for TSMC native scale).
                # Mirrors past TSMC_to_ASAP7 behavior: NOMINAL_VOLTAGE is the TARGET PDK
                # nominal in its native (pre-shift) scale, normalized through the source
                # norm_stats. This anchors the center prediction at the target's nominal
                # operating point, providing a consistent reference for scaling.
                NOMINAL_VOLTAGE = 0.9  # TSMC nominal voltage (target, native scale)
                center_sample = task_samples[indices[0]]
                center_node_features = center_sample['node_features'].clone()

                # Calculate normalized nominal voltage based on normalization method
                node_norm = norm_stats.get('node_features', norm_stats)
                voltage_stats = node_norm['voltage']
                if 'method' in voltage_stats and voltage_stats['method'] == 'minmax_positive':
                    epsilon = voltage_stats.get('epsilon', 0.01)
                    v_min, v_max = voltage_stats['min'], voltage_stats['max']
                    normalized_nominal = epsilon + (NOMINAL_VOLTAGE - v_min) / (v_max - v_min) * (1 - epsilon)
                else:
                    normalized_nominal = (NOMINAL_VOLTAGE - voltage_stats['mean']) / voltage_stats['std']

                # Set voltage to normalized nominal value (for nodes that have voltage)
                voltage_mask = center_node_features[:, 4] != 0
                center_node_features[voltage_mask, 4] = normalized_nominal

                # Get adjacency matrix from TSMC topology cache
                if cell_name not in topology_cache:
                    print(f"   Cell {cell_name} not found in topology cache")
                    continue

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

                # Debug: Check for index mismatch on first few tasks
                if i < 5:
                    num_nodes = center_node_features.shape[0]
                    print(f"       Adjacency matrix shape: {adjacency_matrix.shape}")
                    print(f"       Edge index shape: {edge_index.shape}")
                    if edge_index.numel() > 0:
                        print(f"       Edge index max: {edge_index.max().item()}, Num nodes: {num_nodes}")

                # Create center batch for grad/move calculation
                center_data = Data(x=center_node_features, edge_index=edge_index)
                center_batch = Batch.from_data_list([center_data]).to(device)

                with torch.no_grad():
                    center = model(center_batch).item()

                y_max = y_norm.max().item()
                y_min = y_norm.min().item()

                # Get model predictions for scaling (use support set indices for consistency)
                support_predictions = []
                for idx in indices:
                    sample = task_samples[idx]
                    data = Data(x=sample['node_features'], edge_index=edge_index)
                    batch = Batch.from_data_list([data]).to(device)

                    with torch.no_grad():
                        pred = model(batch).item()
                        support_predictions.append(pred)

                support_predictions = torch.tensor(support_predictions)
                min_val = support_predictions.min().item()
                max_val = support_predictions.max().item()

                if abs(max_val - min_val) > 0:
                    grad = (y_max - y_min) / (max_val - min_val)
                    y_norm_middle = y_norm[middle_idx].item()
                    move = center - y_norm_middle / grad

                    # Evaluate with model using correct function signature
                    (total_loss1, inter_loss1, leftex_loss1, rightex_loss1,
                     mape_loss1, leftex_mape1, inter_mape1, rightex_mape1,
                     predictions, actual_values, _, _, _, _, _, _, adam_used,
                     rmse_loss, rmse_l_loss, rmse_r_loss, rmse_in_loss) = evaluate_model_performance_gnn(
                        model, 'GCN', X_samples, y,
                        true_samples, true_function, grad, move,
                        topology_cache, args.graph_mode, norm_stats, normalize_node_features,
                        left_bound=left_bound, right_bound=right_bound, total_points=args.total_points,
                        mode=args.mode, adaptation_method=args.adaptation_method
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

                    nrmse1_val = nrmse1.item() if hasattr(nrmse1, 'item') else float(nrmse1)
                    if not (np.isinf(nrmse1_val) or np.isnan(nrmse1_val)):
                        total_nrmse.append(nrmse1.item() if hasattr(nrmse1, 'item') else nrmse1)
                        total_extra_l.append(nrmse_leftex.item() if hasattr(nrmse_leftex, 'item') else nrmse_leftex)
                        total_extra_r.append(nrmse_rightex.item() if hasattr(nrmse_rightex, 'item') else nrmse_rightex)
                        total_inter.append(nrmse_inter.item() if hasattr(nrmse_inter, 'item') else nrmse_inter)
                        total_mape.append(mape_percent)
                        total_r_mape.append(mape_r_percent)
                        total_l_mape.append(mape_l_percent)
                        total_in_mape.append(mape_in_percent)
                        total_rmse.append(rmse_loss)
                        total_l_rmse.append(rmse_l_loss)
                        total_r_rmse.append(rmse_r_loss)
                        total_in_rmse.append(rmse_in_loss)

                    if i % 100 == 0 and i > 0 and len(total_nrmse) > 0:
                        print(f"     Adam usage: {(adam_condition_count/len(total_nrmse)):.2f}", flush=True)
                        print(f"     Current avg NRMSE: {sum(total_nrmse)/len(total_nrmse):.2f}% - Tasks: {len(total_nrmse)}", flush=True)
                        print(f"     Current avg MAPE: {sum(total_mape)/len(total_mape):.2f}% - Tasks: {len(total_mape)}", flush=True)
                        print(f"     Current avg RMSE: {1000*sum(total_rmse)/len(total_rmse):.3f}ps - Tasks: {len(total_rmse)}", flush=True)

        except Exception as e:
            if i < 5:
                print(f"   Task {randomtask} failed: {e}")
                import traceback
                traceback.print_exc()
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
        description='ASAP7 to TSMC Cross-PDK GCN Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ASAP7_to_TSMC_GCN_validation.py --experiment intra_topology --model_type maml --gpu 0
  python ASAP7_to_TSMC_GCN_validation.py --experiment topology_agnostic --model_type maml --gpu 1
  python ASAP7_to_TSMC_GCN_validation.py --experiment intra_topology --model_type maml \\
      --pdk_scale_factor 1000.0 --voltage_shift -0.2
        """
    )

    # Required arguments
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['intra_topology', 'topology_agnostic'],
                        help='Experiment type')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['baseline', 'maml'],
                        help='Model type: baseline or maml')

    # Model configuration (ASAP7 model defaults)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Custom model checkpoint path (ASAP7 model)')
    parser.add_argument('--conv_hidden_dim', type=int, default=64,
                        help='Convolution layer hidden dimension (default: 64)')
    parser.add_argument('--num_conv_layers', type=int, default=2,
                        help='Number of GCN convolutional layers (default: 2)')
    parser.add_argument('--fc_hidden_dim', type=int, default=128,
                        help='FC layer hidden dimension (default: 128)')
    parser.add_argument('--num_fc_layers', type=int, default=2,
                        help='Number of FC layers (default: 2)')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'add', 'output'],
                        help='Pooling method: mean, max, add, or output (output-node-only) (default: mean)')

    # Mode configuration
    parser.add_argument('--mode', type=str, choices=['extrapolation', 'interpolation'],
                        default='extrapolation', help='Testing mode (default: extrapolation)')

    # Data configuration
    parser.add_argument('--tsmc_dataset_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC',
                        help='TSMC test dataset directory')
    parser.add_argument('--asap7_dataset_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_ASAP7',
                        help='ASAP7 train dataset directory (for norm_stats fallback if not in checkpoint)')
    parser.add_argument('--data_type', type=str, default='cell',
                        choices=['cell', 'transition'],
                        help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                        choices=['stage_aware', 'full_graph'],
                        help='Graph mode (default: stage_aware)')
    parser.add_argument('--voltage_mode', type=str, default='all_nodes',
                        choices=['all_nodes', 'vdd_only', 'vdd_mos'],
                        help='Voltage mode (default: all_nodes)')
    parser.add_argument('--normalization', type=str, default='zscore',
                        choices=['zscore', 'minmax'],
                        help='Normalization method (default: zscore)')
    parser.add_argument('--temp_mode', type=str, default='typical',
                        choices=['typical', 'temp_all'],
                        help='Temperature mode (default: typical)')
    parser.add_argument('--asap7_cache_path', type=str, default=None,
                        help='Path to ASAP7 topology cache (for checkpoint suffix detection)')
    parser.add_argument('--tsmc_cache_path', type=str, default=None,
                        help='Path to TSMC topology cache file (for test data adjacency)')
    parser.add_argument('--use_target_norm', action='store_true',
                        help='Use TSMC (target) norm_stats instead of ASAP7 (source) for test data normalization. '
                             'Target norm_stats are derivable from the TSMC lib file alone (voltage/slew/load/PVT sweep '
                             'grids), so this option does not assume access to TSMC training characterization data.')
    parser.add_argument('--inputport', action='store_true',
                        help='Use inputport topology (adds _inputport suffix)')
    parser.add_argument('--related_pin_only', action='store_true',
                        help='Use related_pin_only slew assignment (adds _relpin suffix)')
    parser.add_argument('--adaptation_method', type=str, default='selective_adam',
                        choices=['selective_adam', 'adam'],
                        help='Adaptation method (default: selective_adam)')
    parser.add_argument('--output_dir', type=str, default='final',
                        choices=['final', 'test'],
                        help='Output directory: final or test (default: final)')
    parser.add_argument('--sample_suffix', type=str, default='_10pct',
                        help='ASAP7 train data sampling suffix used during pretraining '
                             '(e.g., _10pct, _5pct, or empty string for full data)')

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
    parser.add_argument('--output_prefix', type=str, default='ASAP7_to_TSMC_GCN',
                        help='Prefix for output files')

    # Cell selection
    parser.add_argument('--cells', type=str, nargs='+', default=None,
                        help='Specific cells to test (default: all cells for experiment type)')

    # Continuity filtering
    parser.add_argument('--filter_continuous', action='store_true',
                        help='Filter test tasks to only use continuous data (adds _filtered suffix to output)')
    parser.add_argument('--continuity_threshold', type=float, default=0.18,
                        help='Threshold ratio for continuity check (default: 0.18)')

    # Cross-PDK unit scaling
    parser.add_argument('--pdk_scale_factor', type=float, default=1.0,
                        help='Scale factor for TSMC time/cap units to match ASAP7 units. '
                             'TSMC uses ns/pf, ASAP7 uses ps/ff, so use 1000.0 to convert TSMC to ASAP7 scale. '
                             'Applies to input_slew, output_load, and output values. (default: 1.0)')
    parser.add_argument('--voltage_shift', type=float, default=0.0,
                        help='Voltage shift to align TSMC voltage to ASAP7 voltage scale. '
                             'TSMC nominal=0.9V, ASAP7 nominal=0.7V, so use -0.2 to shift TSMC voltages. '
                             '(default: 0.0)')

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
    print(f"ASAP7 to TSMC Cross-PDK GCN Validation")
    print(f"{'='*80}")
    print(f"Source: ASAP7 model")
    print(f"Target: TSMC test cells")
    print(f"Experiment: {args.experiment}")
    print(f"Model type: {args.model_type}")
    print(f"Graph mode: {args.graph_mode}")
    print(f"Voltage mode: {args.voltage_mode}")
    print(f"Normalization: {args.normalization}")
    print(f"Temp mode: {args.temp_mode}")
    print(f"Data type: {args.data_type}")
    print(f"Mode: {args.mode}")
    print(f"Pooling: {args.pooling}")
    print(f"Adaptation method: {args.adaptation_method}")
    print(f"Output dir: {args.output_dir}")
    if args.inputport:
        print(f"Inputport: enabled")
    if args.related_pin_only:
        print(f"Related pin only: enabled")
    if args.use_target_norm:
        print(f"Normalization: TSMC (target) norm_stats")
    else:
        print(f"Normalization: ASAP7 (source) norm_stats")
    print(f"Cells: {len(cell_list)} cells")
    print(f"{'='*80}")

    # ASAP7 model checkpoint dir suffixes (ASAP7 naming convention)
    asap7_topology_suffix = "_inputport" if args.inputport else ""
    asap7_voltage_suffix = "_vdd_only" if args.voltage_mode == "vdd_only" else ("_vdd_mos" if args.voltage_mode == "vdd_mos" else "")
    asap7_norm_suffix = "_minmax" if args.normalization == "minmax" else ""
    asap7_temp_suffix = "_tempall" if args.temp_mode == "temp_all" else ""
    asap7_slew_suffix = "_relpin" if args.related_pin_only else ""

    # Build checkpoint suffix from ASAP7 cache filename if provided
    checkpoint_suffix = ""
    if args.asap7_cache_path:
        cache_basename = os.path.basename(args.asap7_cache_path)
        if "_gatectrl" in cache_basename:
            checkpoint_suffix += "_gatectrl"
        if "_bidir" in cache_basename:
            checkpoint_suffix += "_bidir"
        if "_directmos" in cache_basename:
            checkpoint_suffix += "_directmos"
    checkpoint_suffix += asap7_topology_suffix

    # Load ASAP7 model
    print(f"\nLoading ASAP7 {args.model_type.upper()} model...")

    if args.model_path is None:
        # Add pooling suffix only if not 'mean' (default)
        pool_suffix = f"_pool{args.pooling}" if args.pooling != 'mean' else ""
        arch_suffix = f"_conv{args.conv_hidden_dim}x{args.num_conv_layers}_fc{args.fc_hidden_dim}x{args.num_fc_layers}{pool_suffix}"

        sample_suffix = args.sample_suffix

        if args.model_type == 'baseline':
            model_dir = f"../../../pretrained_models/gnn_baseline_asap7_process_checkpoints{checkpoint_suffix}{asap7_voltage_suffix}{asap7_norm_suffix}{asap7_temp_suffix}{asap7_slew_suffix}{sample_suffix}"
            model_filename = f"gnn_baseline_asap7_process_{args.data_type}_{args.graph_mode}_iter{args.num_iterations}{arch_suffix}.pth"
        else:
            model_dir = f"../../../pretrained_models/gnn_maml_asap7_process_checkpoints{checkpoint_suffix}{asap7_voltage_suffix}{asap7_norm_suffix}{asap7_temp_suffix}{asap7_slew_suffix}{sample_suffix}"
            model_filename = f"gnn_maml_asap7_process_{args.data_type}_{args.graph_mode}_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}.pth"

        model_path = os.path.join(model_dir, model_filename)
    else:
        model_path = args.model_path

    # Anchor relative model_path to this script's directory so external
    # sweep runners that change cwd don't break the default lookup.
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), model_path))

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please provide --model_path or ensure the ASAP7 model exists")
        return 1

    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)

    # Determine which norm_stats to use
    if args.use_target_norm:
        # Use TSMC (target) norm_stats — derivable from TSMC lib file alone
        tsmc_topology_suffix = "_inputport" if args.inputport else ""
        tsmc_voltage_suffix = f"_{args.voltage_mode}" if args.voltage_mode != "all_nodes" else ""
        tsmc_norm_suffix = "_minmax" if args.normalization == "minmax" else ""
        tsmc_temp_suffix = "_temp_all" if args.temp_mode == "temp_all" else ""
        tsmc_slew_suffix = "_relpin" if args.related_pin_only else ""
        tsmc_train_path = os.path.join(
            args.tsmc_dataset_dir,
            f"train_{args.data_type}_{args.graph_mode}{tsmc_topology_suffix}{tsmc_voltage_suffix}{tsmc_norm_suffix}{tsmc_temp_suffix}{tsmc_slew_suffix}.pth"
        )

        if os.path.exists(tsmc_train_path):
            print(f"Loading norm_stats from TSMC (target) train data: {tsmc_train_path}")
            tsmc_train_data = torch.load(tsmc_train_path, weights_only=False, map_location='cpu', mmap=True)
            norm_stats = tsmc_train_data.get('norm_stats', None)
            if norm_stats is not None:
                print("Using TSMC (target) norm_stats for test data normalization")
            else:
                print("Warning: TSMC train data has no norm_stats")
        else:
            print(f"Warning: TSMC train data not found: {tsmc_train_path}")
            norm_stats = None
    else:
        # Use ASAP7 (source) norm_stats from checkpoint
        norm_stats = checkpoint.get('norm_stats', None)
        if norm_stats is not None:
            print("Using norm_stats from ASAP7 model checkpoint (source)")
        else:
            # Fallback: load from ASAP7 train data
            sample_suffix = args.sample_suffix
            asap7_train_path = os.path.join(
                args.asap7_dataset_dir,
                f"train_{args.data_type}_{args.graph_mode}{asap7_topology_suffix}{asap7_slew_suffix}{sample_suffix}.pth"
            )
            if os.path.exists(asap7_train_path):
                print(f"Loading norm_stats from ASAP7 train data: {asap7_train_path}")
                asap7_train_data = torch.load(asap7_train_path, weights_only=False, map_location='cpu', mmap=True)
                norm_stats = asap7_train_data.get('norm_stats', None)
            else:
                print(f"Warning: Could not load norm_stats from checkpoint or ASAP7 train data")

    # Get architecture params from checkpoint
    config = checkpoint.get('config', {})
    conv_hidden_dim = config.get('conv_hidden_dim', args.conv_hidden_dim)
    num_conv_layers = config.get('num_conv_layers', args.num_conv_layers)
    fc_hidden_dim = config.get('fc_hidden_dim', args.fc_hidden_dim)
    num_fc_layers = config.get('num_fc_layers', args.num_fc_layers)
    node_features = config.get('node_features', 11)  # 11D for both ASAP7 and TSMC Process

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

    print(f"Loaded ASAP7 model: {model_path}")
    print(f"Architecture: conv={conv_hidden_dim}x{num_conv_layers}, fc={fc_hidden_dim}x{num_fc_layers}, pooling={pooling_mode}")

    # Load TSMC topology cache (for test data adjacency)
    tsmc_cache_path = args.tsmc_cache_path
    if tsmc_cache_path is None:
        # Default TSMC topology cache path
        bidir_suffix = "_bidir" if args.asap7_cache_path and "_bidir" in args.asap7_cache_path else ""
        tsmc_cache_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/stage_aware_topology_cache_tsmc{bidir_suffix}.pth"

    if tsmc_cache_path.startswith('/mnt/home/'):
        tsmc_cache_path = tsmc_cache_path.replace('/mnt/home/', '/home/')

    if not os.path.exists(tsmc_cache_path):
        print(f"TSMC topology cache not found: {tsmc_cache_path}")
        return 1

    print(f"\nLoading TSMC topology cache: {tsmc_cache_path}", flush=True)
    topology_cache = torch.load(tsmc_cache_path, weights_only=False, map_location='cpu', mmap=True)
    print(f"Loaded TSMC topology cache for {len(topology_cache)} cells", flush=True)

    # Print PDK scale factor info
    if args.pdk_scale_factor != 1.0 or args.voltage_shift != 0.0:
        print(f"\nCross-PDK Unit Conversion:")
        if args.pdk_scale_factor != 1.0:
            print(f"  PDK scale factor: {args.pdk_scale_factor}")
            print(f"    - TSMC input_slew/output_load scaled by {args.pdk_scale_factor}")
            print(f"    - TSMC output values scaled by {args.pdk_scale_factor}")
            print(f"    - (TSMC: ns/pf -> ASAP7: ps/ff)")
        if args.voltage_shift != 0.0:
            print(f"  Voltage shift: {args.voltage_shift}V")
            print(f"    - TSMC voltage shifted by {args.voltage_shift}V")
            print(f"    - (TSMC nominal: 0.9V -> ASAP7 nominal: 0.7V)")

    # Process each cell
    all_results = []
    successful = 0
    failed = 0

    for cell_name in cell_list:
        result = run_cell_validation(args, cell_name, topology_cache, norm_stats, model, device,
                                     pdk_scale_factor=args.pdk_scale_factor,
                                     voltage_shift=args.voltage_shift)
        if result is not None:
            all_results.append(result)
            successful += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: ASAP7 to TSMC - {args.experiment}")
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

        # Determine output directory based on output_dir argument
        output_dir_name = "data_result_npy_directory_final" if args.output_dir == "final" else "data_result_npy_directory"

        for result in all_results:
            cell_name = result['cell_name']

            if args.model_type == 'baseline':
                base_name = f"{args.output_prefix}_{args.experiment}_{cell_name}_{args.data_type}_{args.graph_mode}_{args.mode}_{args.model_type}_iter{args.num_iterations}{arch_suffix}{filter_suffix}"
            else:
                base_name = f"{args.output_prefix}_{args.experiment}_{cell_name}_{args.data_type}_{args.graph_mode}_{args.mode}_{args.model_type}_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}{filter_suffix}"

            pred_filename = f"{output_dir_name}/{base_name}_pred.npy"
            act_filename = f"{output_dir_name}/{base_name}_act.npy"

            os.makedirs(output_dir_name, exist_ok=True)

            np.save(pred_filename, result['predictions'])
            np.save(act_filename, result['actuals'])

        print(f"\nSaved results to {output_dir_name}/")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
