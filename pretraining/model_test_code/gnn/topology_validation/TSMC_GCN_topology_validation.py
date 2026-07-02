#!/usr/bin/env python
"""
TSMC GCN Topology Validation Script

This script validates GCN models for TSMC topology experiments:
- topology_agnostic: Test on unseen cell topologies
- intra_topology: Test on seen cell topologies with unseen conditions

Evaluates per-cell and aggregates results.

Usage:
  python TSMC_GCN_topology_validation.py --experiment intra_topology --model_type baseline --gpu 0
  python TSMC_GCN_topology_validation.py --experiment topology_agnostic --model_type maml --gpu 1
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

TOPOLOGY_AGNOSTIC_CELLS = ['OA21D0BWP30P140', 'OA21D1BWP30P140', 'OA211D0BWP30P140', 'OA211D1BWP30P140',
                            'IOA21D0BWP30P140', 'IOA21D1BWP30P140', 'HA1D0BWP30P140', 'FA1D0BWP30P140', 
                          'IAO21D0BWP30P140', 'IAO21D1BWP30P140', 'AO21D0BWP30P140', 'AO21D1BWP30P140',
                          'AO211D0BWP30P140', 'AO211D1BWP30P140', 'SDFSNQD0BWP30P140', 'DFCNQD1BWP30P140']


def normalize_node_features(node_features, norm_stats, temp_mode='typical'):
    """
    Normalize node features using saved statistics.
    Only normalize voltage (col 4), input_slew (col 5), output_load (col 6), temperature (col 10 for 11D)
    Supports both zscore (mean/std) and minmax (min/max/epsilon) normalization.

    Args:
        node_features: Node feature tensor
        norm_stats: Normalization statistics
        temp_mode: 'temp_all' to normalize all nodes, 'typical' to normalize MOS nodes only
    """
    if norm_stats is None:
        return node_features

    normalized = node_features.clone()
    node_norm = norm_stats['node_features']

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
    voltage_mask = normalized[:, 4] != 0
    if voltage_mask.any():
        normalized[voltage_mask, 4] = apply_norm(
            normalized[voltage_mask, 4], node_norm['voltage']
        )

    # Normalize input_slew (column 5)
    slew_mask = normalized[:, 5] != 0
    if slew_mask.any():
        normalized[slew_mask, 5] = apply_norm(
            normalized[slew_mask, 5], node_norm['input_slew']
        )

    # Normalize output_load (column 6)
    load_mask = normalized[:, 6] != 0
    if load_mask.any():
        normalized[load_mask, 6] = apply_norm(
            normalized[load_mask, 6], node_norm['output_load']
        )

    # Normalize temperature (column 10 for 11D features) if available
    if 'temperature' in node_norm and normalized.shape[1] > 10:
        temp_values = normalized[:, 10]
        temp_stats = node_norm['temperature']

        if temp_mode == 'temp_all':
            # temp_all mode: normalize all nodes
            normalized[:, 10] = apply_norm(temp_values, temp_stats)
        else:
            # typical/mos_only mode: only normalize MOS nodes
            mosfet_mask = normalized[:, 2] != 0  # MOSFET nodes (PMOS=+1, NMOS=-1)
            if mosfet_mask.any():
                normalized[mosfet_mask, 10] = apply_norm(
                    normalized[mosfet_mask, 10], temp_stats
                )

    return normalized


class CellTestDataset:
    """
    Dataset class for loading per-cell test data with mmap.
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
    # Add voltage_mode suffix to directory if not all_nodes (e.g., _vdd_only, _vdd_mos)
    # Add topology_suffix for inputport (inputport affects node features)
    topology_suffix = "_inputport" if args.inputport else ""
    # Add voltage_mode and temp_mode suffixes
    voltage_suffix = f"_{args.voltage_mode}" if args.voltage_mode != "all_nodes" else ""
    temp_suffix = "_temp_all" if args.temp_mode == "temp_all" else ""
    slew_suffix = "_relpin" if args.related_pin_only else ""
    # If constraint_category is set, read test PTHs from test_by_<category>_<graph_mode>/
    # but keep cell/transition model + norm_stats loading via --data_type.
    test_data_type = args.constraint_category if args.constraint_category else args.data_type
    test_dir = f"test_by_{test_data_type}_{args.graph_mode}{topology_suffix}{voltage_suffix}{temp_suffix}{slew_suffix}"
    cell_path = os.path.join(
        args.dataset_dir,
        test_dir,
        f"{cell_name}.pth"
    )
    print(f"   Test directory: {test_dir}")

    if not os.path.exists(cell_path):
        print(f"   Cell data not found: {cell_path}")
        return None

    cell_dataset = CellTestDataset(cell_path, topology_cache)
    print(f"   Test data: {os.path.basename(os.path.dirname(cell_path))}/{cell_name}.pth")
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

    # Random sampling of test tasks (optionally seeded per cell for reproducible sweeps)
    num_tasks = cell_dataset.num_tasks
    num_test_samples = min(args.num_test_samples, num_tasks)
    if args.seed is not None:
        # Per-cell seed: combine base seed with cell name so different cells get different
        # tasks (avoids accidental task overlap) while a fixed --seed makes the run reproducible.
        rng = random.Random(args.seed + (hash(cell_name) & 0xFFFFFFFF))
        test_indices_list = rng.sample(range(num_tasks), num_test_samples)
    else:
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
    total_mape = []
    total_rmse = []
    all_predictions = []
    all_actuals = []
    # Per-task scaling factor for conformal calibration in normalized units
    # (= y_std_support × grad; same scale safe_eps uses, so delta_norm here is
    # directly comparable to a safe_eps value).
    all_task_scales = []
    adam_condition_count = 0

    for i, randomtask in enumerate(test_indices_list):
        if i % 100 == 0:
            print(f"   Processing task {i+1}/{num_test_samples} (index: {randomtask})", flush=True)

        try:
            # Load task data
            task_samples, task_outputs = cell_dataset.get_all_libs_for_task(randomtask, clone=True)
            task_outputs_tensor = torch.tensor(task_outputs, dtype=torch.float32)
            # v6 per-task sign_flip for constraint LUTs: detect the curve's natural
            # direction from extreme support points. When y_high > y_low the curve
            # is "inverted" (rises with V) relative to the cell-delay model's
            # monotone-decreasing prior, so mirror it for adaptation. Predictions
            # are unflipped after evaluate_model_performance_gnn returns. Pure
            # vertical mirror — no zero-crossing kinks, no shape distortion.
            task_sign_flip = 1.0
            if args.constraint_category is not None:
                y_low  = task_outputs_tensor[indices[0]].item()
                y_high = task_outputs_tensor[indices[-1]].item()
                if y_high > y_low:
                    task_sign_flip = -1.0
                    task_outputs_tensor = -task_outputs_tensor

            # Apply normalization
            for sample in task_samples:
                sample['node_features'] = normalize_node_features(sample['node_features'], norm_stats, args.temp_mode)

            # Get support set samples
            X_samples = [task_samples[idx] for idx in indices]
            y = task_outputs_tensor[indices]

            true_samples = task_samples
            true_function = task_outputs_tensor

            # Calculate range for NRMSE normalization
            y1_range = task_outputs_tensor.max() - task_outputs_tensor.min()

            y_mean = y.mean()
            y_std = y.std()

            if y_std > 0:
                y_norm = (y - y_mean) / y_std

                # Create center input with nominal voltage (0.9V for TSMC)
                NOMINAL_VOLTAGE = 0.9  # TSMC nominal voltage
                center_sample = task_samples[indices[0]]
                center_node_features = center_sample['node_features'].clone()

                # Calculate normalized nominal voltage based on normalization method
                voltage_stats = norm_stats['node_features']['voltage']
                if 'method' in voltage_stats and voltage_stats['method'] == 'minmax_positive':
                    # minmax: normalized = epsilon + (x - min) / (max - min) * (1 - epsilon)
                    epsilon = voltage_stats.get('epsilon', 0.01)
                    v_min, v_max = voltage_stats['min'], voltage_stats['max']
                    normalized_nominal = epsilon + (NOMINAL_VOLTAGE - v_min) / (v_max - v_min) * (1 - epsilon)
                else:
                    # zscore: normalized = (x - mean) / std
                    normalized_nominal = (NOMINAL_VOLTAGE - voltage_stats['mean']) / voltage_stats['std']

                # Set voltage to normalized nominal value (for nodes that have voltage)
                voltage_mask = center_node_features[:, 4] != 0
                center_node_features[voltage_mask, 4] = normalized_nominal

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

                    (total_loss1, mape_loss1,
                     predictions, actual_values, _, adam_used,
                     rmse_loss) = evaluate_model_performance_gnn(
                        model, 'GCN', X_samples, y,
                        true_samples, true_function, grad, move,
                        topology_cache, args.graph_mode, norm_stats, normalize_node_features,
                        left_bound=left_bound, right_bound=right_bound, total_points=args.total_points,
                        mode=args.mode, adaptation_method=args.adaptation_method,
                        asym_alpha=args.asym_alpha, safe_eps=args.safe_eps,
                        pinball_tau=args.pinball_tau,
                        inner_adam_lr=args.inner_lr,
                    )

                    if adam_used:
                        adam_condition_count += 1

                    # v6: unflip predictions/actuals back to original sign space
                    if task_sign_flip < 0:
                        predictions = [-p for p in predictions]
                        actual_values = [-a for a in actual_values]

                    all_predictions.extend(predictions)
                    all_actuals.extend(actual_values)
                    # Record this task's normalized scale (y_std_support × grad);
                    # used by post-hoc conformal calibration to compute / apply a
                    # task-adaptive raw shift, mirroring how safe_eps composes.
                    all_task_scales.append(float(y_std.item() * grad))

                    # Calculate metrics
                    nrmse1 = (total_loss1 ** 0.5) / (abs(y1_range) + 1e-4) * 100
                    mape_percent = mape_loss1 * 100

                    if not (torch.isinf(nrmse1) or torch.isnan(nrmse1)):
                        total_nrmse.append(nrmse1.item())
                        total_mape.append(mape_percent)
                        total_rmse.append(rmse_loss)

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

    # ------------------------------------------------------------------
    # Post-hoc per-cell conformal calibration in NORMALIZED units (the
    # same scale safe_eps uses: per-task y_std_support × grad). Splits
    # this cell's completed tasks into a calibration half (used only to
    # compute delta_norm = (1-alpha)-quantile of residuals / scale) and
    # an evaluation half (each task's predictions shifted by
    # +delta_norm × scale_task before metrics / npy are reported).
    # Operates orthogonally to safe_eps / asym_alpha / pinball_tau (they
    # affect *adaptation*; this acts only on the final predictions).
    # Under exchangeability the eval under-prediction rate <= alpha.
    # ------------------------------------------------------------------
    if args.conformal_alpha is not None and len(all_predictions) >= 2 * args.total_points:
        gp = args.total_points
        n_tasks_done = min(len(all_task_scales), len(all_predictions) // gp)
        pred_arr  = np.array(all_predictions[:n_tasks_done * gp], dtype=np.float64).reshape(n_tasks_done, gp)
        act_arr   = np.array(all_actuals    [:n_tasks_done * gp], dtype=np.float64).reshape(n_tasks_done, gp)
        scale_arr = np.array(all_task_scales[:n_tasks_done],      dtype=np.float64)
        # Guard: zero / non-finite scale ⇒ skip task (can't normalize)
        valid_mask = np.isfinite(scale_arr) & (scale_arr > 0)
        if not valid_mask.all():
            pred_arr  = pred_arr[valid_mask]
            act_arr   = act_arr[valid_mask]
            scale_arr = scale_arr[valid_mask]
            n_tasks_done = int(valid_mask.sum())

        n_cal = max(1, int(round(args.conformal_split * n_tasks_done)))
        n_cal = min(n_cal, n_tasks_done - 1)  # keep at least 1 eval task
        # First n_cal tasks → calibration; remaining → evaluation
        pred_cal, act_cal, scale_cal = pred_arr[:n_cal], act_arr[:n_cal], scale_arr[:n_cal]
        pred_ev,  act_ev,  scale_ev  = pred_arr[n_cal:], act_arr[n_cal:], scale_arr[n_cal:]

        # Normalize residuals by per-task scale before pooling — this is what
        # makes the recovered delta directly comparable to a safe_eps value
        # and ensures the eval-time shift is *task-adaptive* (small tasks get
        # small raw shift, large tasks get large raw shift) rather than the
        # one-size-fits-all raw constant the previous formulation used.
        residuals_norm_cal = ((act_cal - pred_cal) / scale_cal[:, None]).ravel()
        delta_norm = float(np.quantile(residuals_norm_cal, 1.0 - args.conformal_alpha))

        cal_under = float((residuals_norm_cal > 0).mean())
        ev_under_before = float(((act_ev - pred_ev) > 0).mean())
        # Apply per-task raw shift
        raw_shift_per_task = delta_norm * scale_ev          # [n_eval]
        pred_ev_shifted    = pred_ev + raw_shift_per_task[:, None]
        ev_resid           = act_ev - pred_ev_shifted
        ev_under_after     = float((ev_resid > 0).mean())

        # Recompute per-task metrics on the calibrated eval split, matching
        # the conventions in analyze_safe_eps_sweep.py (per-task RMSE /
        # range for NRMSE; per-task 10% floor for MAPE).
        rmse_g   = np.sqrt(((pred_ev_shifted - act_ev) ** 2).mean(axis=1))
        y_range  = np.maximum(act_ev.max(axis=1) - act_ev.min(axis=1), 1e-12)
        nrmse_g  = (rmse_g / y_range) * 100.0
        floor    = np.maximum(np.abs(act_ev).max(axis=1, keepdims=True) * 0.10, 1e-12)
        denom    = np.maximum(np.abs(act_ev), floor)
        mape_g   = (np.abs(pred_ev_shifted - act_ev) / denom).mean(axis=1) * 100.0

        total_rmse  = rmse_g.tolist()
        total_nrmse = nrmse_g.tolist()
        total_mape  = mape_g.tolist()
        # Overwrite stored predictions / actuals with the calibrated eval
        # portion so the npy save downstream reflects what was scored.
        all_predictions = pred_ev_shifted.ravel().tolist()
        all_actuals     = act_ev.ravel().tolist()

        print(f"   [conformal alpha={args.conformal_alpha:g}, split={args.conformal_split:g}] "
              f"n_cal={n_cal}, n_eval={pred_ev.shape[0]}  delta_norm={delta_norm:+.6e}  "
              f"raw_shift_mean={raw_shift_per_task.mean():+.6e}  "
              f"under(cal)={cal_under:.3f}  under(eval)_before={ev_under_before:.3f}  "
              f"after={ev_under_after:.3f}", flush=True)

    # Calculate final metrics
    results = {
        'cell_name': cell_name,
        'num_valid_tasks': len(total_nrmse),
        'nrmse_total': sum(total_nrmse) / len(total_nrmse),
        'mape_total': sum(total_mape) / len(total_mape),
        'rmse_total': sum(total_rmse) / len(total_rmse),
        'predictions': all_predictions,
        'actuals': all_actuals,
    }

    print(f"   {cell_name}: NRMSE={results['nrmse_total']:.2f}%, MAPE={results['mape_total']:.2f}%, "
          f"RMSE={results['rmse_total']*1e9:.4f}ns ({len(total_nrmse)} tasks)")

    return results


# ---------------- main() helpers ----------------

def _build_argparser():
    """CLI parser for TSMC GCN Topology Validation."""
    parser = argparse.ArgumentParser(
        description='TSMC GCN Topology Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python TSMC_GCN_topology_validation.py --experiment intra_topology --model_type baseline --gpu 0
  python TSMC_GCN_topology_validation.py --experiment topology_agnostic --model_type maml --gpu 1
  python TSMC_GCN_topology_validation.py --experiment intra_topology --model_type maml --mode interpolation
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
                        help='Convolution layer hidden dimension (default: 128)')
    parser.add_argument('--num_conv_layers', type=int, default=2,
                        help='Number of GCN convolutional layers (default: 3)')
    parser.add_argument('--fc_hidden_dim', type=int, default=256,
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
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC',
                        help='Dataset directory')
    parser.add_argument('--data_type', type=str, default='cell',
                        choices=['cell', 'transition'],
                        help='Data type (default: cell). For constraint LUT validation '
                             '(setup/hold/etc.) keep this as cell and pass --constraint_category.')
    parser.add_argument('--constraint_category', type=str, default=None,
                        choices=['setup', 'hold', 'recovery', 'removal',
                                 'non_seq_setup', 'non_seq_hold'],
                        help=('Constraint-LUT category. When set, the script reads test '
                              'PTHs from test_by_<constraint_category>_<graph_mode>/ but '
                              'still loads the cell/transition model + norm_stats via '
                              '--data_type. Compatible only with --data_type cell + '
                              '--graph_mode stage_aware (the canonical "ours" setup).'))
    parser.add_argument('--graph_mode', type=str, default='full_graph',
                        choices=['stage_aware', 'full_graph'],
                        help='Graph mode (default: full_graph)')
    parser.add_argument('--voltage_mode', type=str, default='all_nodes',
                        choices=['all_nodes', 'vdd_only', 'vdd_mos'],
                        help='Voltage mode: all_nodes (voltage on all nodes), vdd_only (voltage only on VDD), or vdd_mos (voltage on VDD and MOS nodes) (default: all_nodes)')
    parser.add_argument('--normalization', type=str, default='zscore',
                        choices=['zscore', 'minmax'],
                        help='Normalization method: zscore (default) or minmax')
    parser.add_argument('--temp_mode', type=str, default='typical',
                        choices=['typical', 'temp_all'],
                        help='Temperature mode: typical (temp on MOS only) or temp_all (temp on all nodes) (default: typical)')
    parser.add_argument('--cache_path', type=str, default=None,
                        help='Override topology cache path (default: use cache_path stored in dataset file)')
    parser.add_argument('--inputport', action='store_true',
                        help='Use inputport topology (checkpoint dir includes _inputport suffix)')
    parser.add_argument('--related_pin_only', action='store_true',
                        help='Use related_pin_only slew assignment (adds _relpin suffix)')

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
    parser.add_argument('--num_iterations', type=int, default=100000,
                        help='Pretraining iterations (default: 100000)')

    # Results saving
    parser.add_argument('--save_results', action='store_true',
                        help='Save prediction results to .npy files')
    parser.add_argument('--output_prefix', type=str, default='TSMC_GCN',
                        help='Prefix for output files')

    # Cell selection
    parser.add_argument('--cells', type=str, nargs='+', default=None,
                        help='Specific cells to test (default: all cells for experiment type)')

    # Continuity filtering
    parser.add_argument('--filter_continuous', action='store_true',
                        help='Filter test tasks to only use continuous data (adds _filtered suffix to output)')
    parser.add_argument('--continuity_threshold', type=float, default=0.18,
                        help='Threshold ratio for continuity check (default: 0.18)')

    # Adaptation method
    parser.add_argument('--adaptation_method', type=str, default='selective_adam',
                        choices=['selective_adam', 'adam'],
                        help='Adaptation method: selective_adam (Adam if loss>1e-4) or adam (always Adam) (default: selective_adam)')
    parser.add_argument('--inner_lr', type=float, default=3e-4,
                        help=('Inner-loop Adam learning rate (default 3e-4). For constraint '
                              'categories (setup/hold/...) try 3e-3 — about 10x default — for '
                              '~50%% NRMSE improvement; cell/transition are already optimal at 3e-4.'))
    parser.add_argument('--asym_alpha', type=float, default=None,
                        help=('Asymmetric-MSE alpha for inner-loop few-shot adaptation. '
                              'None or 0.5 → standard MSE (default). >0.5 penalizes under-prediction '
                              'and biases predictions to over-estimate (safe). Try 0.7/0.8.'))
    parser.add_argument('--safe_eps', type=float, default=None,
                        help=('Safe-margin epsilon added to the support-training target during '
                              'grad/move normalization (selective_adam only). Predictions end up '
                              'shifted up by ~safe_eps * y_std * grad in raw units → biased '
                              'toward over-estimate without changing the loss. Try 0.02/0.05/0.1.'))
    parser.add_argument('--seed', type=int, default=None,
                        help=('Random seed for task sampling (so different runs hit the same task '
                              'indices for fair sweep comparison). Default: None (uses random sampling).'))
    parser.add_argument('--pinball_tau', type=float, default=None,
                        help=('Pinball / quantile loss tau (inner-loop training only). '
                              '>0.5 biases prediction toward over-estimate (target = tau-quantile). '
                              'Try 0.7/0.8/0.9. Mutually exclusive with --asym_alpha.'))
    parser.add_argument('--conformal_alpha', type=float, default=None,
                        help=('Post-hoc per-cell conformal calibration target miscoverage rate. '
                              'When set (e.g., 0.05), splits this cell\'s tasks into a calibration '
                              'half (compute (1-alpha)-quantile of (actual - pred) residuals) and '
                              'an evaluation half (apply delta_cell shift before reporting metrics / '
                              'saving npy). Statistical guarantee: under-pred rate <= alpha on eval '
                              'half (under exchangeability). Composes orthogonally with safe_eps / '
                              'asym_alpha / pinball_tau (they affect adaptation; this acts on the '
                              'final predictions).'))
    parser.add_argument('--conformal_split', type=float, default=0.5,
                        help=('Fraction of this cell\'s tasks used for conformal calibration. '
                              'Remaining fraction is the evaluation set whose predictions get '
                              'shifted by delta_cell and whose metrics / npy files are reported. '
                              'Only meaningful when --conformal_alpha is set. Default: 0.5'))

    # Output directory
    parser.add_argument('--output_dir', type=str, default='final',
                        help='Output directory mode: "final" for data_result_npy_directory_final, otherwise data_result_npy_directory (default: final)')
    return parser


def _resolve_cell_list(args):
    """Pick the cell list from --cells, experiment default, or intra/agnostic constant."""
    if args.cells is not None:
        return args.cells
    if args.experiment == 'intra_topology':
        return INTRA_TOPOLOGY_CELLS
    return TOPOLOGY_AGNOSTIC_CELLS


def _build_all_suffixes(args):
    """
    Build the collection of path suffixes that toggle features on/off.
    Returns dict with keys: topology, voltage, norm, temp, slew.
    """
    return {
        'topology': "_inputport" if args.inputport else "",
        'voltage': f"_{args.voltage_mode}" if args.voltage_mode != "all_nodes" else "",
        'norm': "_minmax" if args.normalization == "minmax" else "",
        'temp': "_temp_all" if args.temp_mode == "temp_all" else "",
        'slew': "_relpin" if args.related_pin_only else "",
    }


def _build_checkpoint_suffix(args, topology_suffix):
    """Detect cache-derived suffixes (_gatectrl, _bidir, _directmos) and append topology_suffix."""
    suffix = ""
    if args.cache_path:
        for tag in ("_gatectrl", "_bidir", "_directmos"):
            if tag in args.cache_path:
                suffix += tag
    return suffix + topology_suffix


def _print_config_banner(args, cell_list):
    """Config banner printed at the top of main()."""
    print(f"\n{'='*80}")
    print(f"TSMC GCN Topology Validation")
    print(f"{'='*80}")
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
    if args.asym_alpha is not None:
        print(f"Asymmetric-MSE alpha (inner-loop only): {args.asym_alpha}")
    if args.safe_eps is not None:
        print(f"Safe-margin epsilon (move shift, normalized units): {args.safe_eps}")
    if args.pinball_tau is not None:
        print(f"Pinball / quantile tau (inner-loop only): {args.pinball_tau}")
    print(f"Cells: {len(cell_list)} cells")
    print(f"{'='*80}")


def _load_train_data_and_norm_stats(args, suffixes):
    """
    Locate the train dataset (falling back to legacy paths), load with mmap.

    Returns (norm_stats, cache_path_from_data) on success; (None, None) if no
    candidate path exists.
    """
    top = suffixes['topology']
    volt = suffixes['voltage']
    norm = suffixes['norm']
    temp = suffixes['temp']
    slew = suffixes['slew']
    train_path = os.path.join(
        args.dataset_dir,
        f"train_{args.data_type}_{args.graph_mode}{top}{volt}{norm}{temp}{slew}.pth",
    )
    print(f"\nLoading train data for norm_stats: {train_path}", flush=True)
    if not os.path.exists(train_path):
        # Try without slew_suffix
        alt = os.path.join(
            args.dataset_dir,
            f"train_{args.data_type}_{args.graph_mode}{top}{volt}{norm}{temp}.pth",
        )
        if os.path.exists(alt):
            print(f"Using train path without slew suffix: {alt}", flush=True)
            train_path = alt
        else:
            # Legacy path without topology suffix
            legacy = os.path.join(
                args.dataset_dir,
                f"train_{args.data_type}_{args.graph_mode}.pth",
            )
            if os.path.exists(legacy):
                print(f"Using legacy train path: {legacy}", flush=True)
                train_path = legacy
            else:
                print(f"Train data not found: {train_path}")
                return None, None

    print("Loading train_data with mmap...", flush=True)
    train_data = torch.load(train_path, weights_only=False, map_location='cpu', mmap=True)
    print("train_data loaded.", flush=True)
    norm_stats = train_data.get('norm_stats', None)
    cache_path_from_data = train_data.get('cache_path', None)
    print(f"Norm stats loaded: {norm_stats is not None}")
    return norm_stats, cache_path_from_data


def _load_topology_cache(args, cache_path_from_data):
    """
    Resolve topology cache path (prefer CLI, else train_data-embedded),
    remap /mnt/home/ → /home/, load with mmap.  Returns None if not found.
    """
    if args.cache_path:
        cache_path = args.cache_path
        print(f"Using cache_path override: {cache_path}")
    else:
        cache_path = cache_path_from_data
    if cache_path:
        if cache_path.startswith('/mnt/home/'):
            cache_path = cache_path.replace('/mnt/home/', '/home/')
        if not os.path.exists(cache_path):
            cache_filename = os.path.basename(cache_path)
            cache_path = (
                f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/"
                f"gnn/topology_cache/{cache_filename}"
            )
    if not (cache_path and os.path.exists(cache_path)):
        print(f"Topology cache not found: {cache_path}")
        return None
    print(f"Loading topology cache: {cache_path}", flush=True)
    topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu', mmap=True)
    print(f"Loaded topology cache for {len(topology_cache)} cells", flush=True)
    return topology_cache


def _resolve_model_path(args, suffixes, checkpoint_suffix):
    """
    Build the default MAML/baseline model checkpoint path from CLI args + suffixes,
    or return args.model_path when the user supplied a custom path.
    """
    if args.model_path is not None:
        return args.model_path
    pool_suffix = f"_pool{args.pooling}" if args.pooling != 'mean' else ""
    arch_suffix = (f"_conv{args.conv_hidden_dim}x{args.num_conv_layers}"
                   f"_fc{args.fc_hidden_dim}x{args.num_fc_layers}{pool_suffix}")
    volt = suffixes['voltage']
    norm = suffixes['norm']
    temp = suffixes['temp']
    slew = suffixes['slew']
    if args.model_type == 'baseline':
        model_dir = (f"../../../pretrained_models/gnn_baseline_tsmc_process_checkpoints"
                     f"{checkpoint_suffix}{volt}{norm}{temp}{slew}")
        model_filename = (f"gnn_baseline_tsmc_process_{args.data_type}_{args.graph_mode}"
                          f"_iter{args.num_iterations}{arch_suffix}.pth")
    else:
        model_dir = (f"../../../pretrained_models/gnn_maml_tsmc_process_checkpoints"
                     f"{checkpoint_suffix}{volt}{norm}{temp}{slew}")
        model_filename = (f"gnn_maml_tsmc_process_{args.data_type}_{args.graph_mode}"
                          f"_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}"
                          f"_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}.pth")
    return os.path.join(model_dir, model_filename)


def _load_gnn_model_from_checkpoint(model_path, args, device):
    """
    Load a GNN checkpoint from model_path (anchored to script dir if relative),
    construct a matching model, load state.

    Returns (model, arch_kwargs, ckpt_norm_stats) on success; (None, None, None)
    if the checkpoint file is missing.
    """
    if not os.path.isabs(model_path):
        # Anchor relative paths to this script's directory so external sweep
        # runners that change cwd don't break the default lookup.
        model_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), model_path))
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please provide --model_path or ensure the model exists")
        return None, None, None

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    ckpt_norm_stats = checkpoint.get('norm_stats', None)
    if ckpt_norm_stats is not None:
        print("Using norm_stats from checkpoint")

    config = checkpoint.get('config', {})
    # TSMC detects node_features from the first conv layer weight shape (not config).
    node_features = checkpoint['model_state_dict']['convs.0.lin.weight'].shape[1]
    arch_kwargs = dict(
        conv_hidden_dim=config.get('conv_hidden_dim', args.conv_hidden_dim),
        num_conv_layers=config.get('num_conv_layers', args.num_conv_layers),
        fc_hidden_dim=config.get('fc_hidden_dim', args.fc_hidden_dim),
        num_fc_layers=config.get('num_fc_layers', args.num_fc_layers),
        node_features=node_features,
        pooling=config.get('pooling', args.pooling),
    )
    print(f"Detected node_features from checkpoint: {arch_kwargs['node_features']}")
    print(f"Pooling mode: {arch_kwargs['pooling']}")

    model = create_maml_gcn_model(
        node_features=arch_kwargs['node_features'],
        pooling=arch_kwargs['pooling'],
        output_dim=1,
        dropout=0.0,
        conv_hidden_dim=arch_kwargs['conv_hidden_dim'],
        num_conv_layers=arch_kwargs['num_conv_layers'],
        fc_hidden_dim=arch_kwargs['fc_hidden_dim'],
        num_fc_layers=arch_kwargs['num_fc_layers'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    print(f"Loaded model: {model_path}")
    print(f"Architecture: conv={arch_kwargs['conv_hidden_dim']}x{arch_kwargs['num_conv_layers']}, "
          f"fc={arch_kwargs['fc_hidden_dim']}x{arch_kwargs['num_fc_layers']}, "
          f"pooling={arch_kwargs['pooling']}")
    return model, arch_kwargs, ckpt_norm_stats


def _print_summary(args, cell_list, all_results, successful, failed):
    """Per-cell summary table + averages."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.experiment}")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(cell_list)}, Failed: {failed}/{len(cell_list)}")
    print()
    if not all_results:
        return
    print(f"{'Cell':<25} {'Tasks':<8} {'NRMSE%':<10} {'MAPE%':<10} {'RMSE(ns)':<12}")
    print("-" * 70)
    for result in all_results:
        print(f"{result['cell_name']:<25} {result['num_valid_tasks']:<8} "
              f"{result['nrmse_total']:<10.2f} {result['mape_total']:<10.2f} "
              f"{result['rmse_total']*1e9:<12.4f}")
    avg_nrmse = sum(r['nrmse_total'] for r in all_results) / len(all_results)
    avg_mape = sum(r['mape_total'] for r in all_results) / len(all_results)
    avg_rmse = sum(r['rmse_total'] for r in all_results) / len(all_results)
    total_tasks = sum(r['num_valid_tasks'] for r in all_results)
    print("-" * 70)
    print(f"{'AVERAGE':<25} {total_tasks:<8} {avg_nrmse:<10.2f} {avg_mape:<10.2f} {avg_rmse*1e9:<12.4f}")


def _save_results_npy(args, all_results, arch_kwargs):
    """Save per-cell prediction/actual .npy files with sweep-encoded filenames."""
    pooling_mode = arch_kwargs['pooling']
    pool_suffix = f"_pool{pooling_mode}" if pooling_mode != 'mean' else ""
    arch_suffix = (f"_conv{arch_kwargs['conv_hidden_dim']}x{arch_kwargs['num_conv_layers']}"
                   f"_fc{arch_kwargs['fc_hidden_dim']}x{arch_kwargs['num_fc_layers']}{pool_suffix}")
    filter_suffix = "_filtered" if args.filter_continuous else ""

    voltage_suffix = ""
    if args.voltage_mode == 'vdd_only':
        voltage_suffix = "_vddonly"
    elif args.voltage_mode == 'vdd_mos':
        voltage_suffix = "_vddmos"
    relpin_suffix = "_relpin" if args.related_pin_only else ""
    adapt_suffix = "_adam" if args.adaptation_method == 'adam' else ""
    asym_suffix = f"_asymA{args.asym_alpha:g}" if args.asym_alpha is not None else ""
    safe_suffix = f"_safeE{args.safe_eps:g}" if args.safe_eps is not None else ""
    pin_suffix  = f"_pinT{args.pinball_tau:g}" if args.pinball_tau is not None else ""
    conf_suffix = f"_confA{args.conformal_alpha:g}" if args.conformal_alpha is not None else ""

    # When evaluating constraint LUTs, reflect the category in the filename so
    # different category outputs don't clobber each other (or cell delay outputs).
    file_data_type = args.constraint_category if args.constraint_category else args.data_type

    output_dir_name = (
        "data_result_npy_directory_final" if args.output_dir == "final"
        else "data_result_npy_directory"
    )
    os.makedirs(output_dir_name, exist_ok=True)

    common_tail = (f"{arch_suffix}{filter_suffix}{voltage_suffix}{relpin_suffix}"
                   f"{adapt_suffix}{asym_suffix}{pin_suffix}{safe_suffix}{conf_suffix}")

    for result in all_results:
        cell_name = result['cell_name']
        base_head = (f"{args.output_prefix}_{args.experiment}_{cell_name}"
                     f"_{file_data_type}_{args.graph_mode}_{args.mode}_{args.model_type}")
        if args.model_type == 'baseline':
            base_name = f"{base_head}_iter{args.num_iterations}{common_tail}"
        else:
            base_name = (f"{base_head}_innerdiv{args.innerdiv}"
                         f"_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}"
                         f"_inner{args.inner_steps}{common_tail}")
        np.save(f"{output_dir_name}/{base_name}_pred.npy", result['predictions'])
        np.save(f"{output_dir_name}/{base_name}_act.npy", result['actuals'])

    print(f"\nSaved results to {output_dir_name}/")


def _maybe_alias_constraint_norm_stats(args, norm_stats):
    """
    Constraint-LUT mode: both index axes of constraint_template_3x3 are slews
    (related_pin_transition + constrained_pin_transition), not (slew + capacitance).
    The builder writes the constrained-pin slew into the output_load slot of the
    graph feature vector, so we alias output_load's normalization stats to those of
    input_slew (both are slews in ns).  This avoids the ~+20 sigma OOD that would
    otherwise occur when slew values are normalized against the pretrained model's
    output_load stats (which were learned on capacitance in pF).
    """
    if not (args.constraint_category and norm_stats is not None):
        return
    nf_stats = norm_stats.get('node_features', {})
    if 'input_slew' in nf_stats:
        nf_stats['output_load'] = dict(nf_stats['input_slew'])
        print(f"Constraint mode ({args.constraint_category}): aliased output_load "
              f"norm_stats → input_slew (both axes are slews in ns).")


def main():
    args = _build_argparser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {args.gpu}")
    print(f"Device: {device}")

    cell_list = _resolve_cell_list(args)
    _print_config_banner(args, cell_list)

    suffixes = _build_all_suffixes(args)
    checkpoint_suffix = _build_checkpoint_suffix(args, suffixes['topology'])

    # Load train data → norm_stats + optional embedded cache_path.
    norm_stats, cache_path_from_data = _load_train_data_and_norm_stats(args, suffixes)
    if norm_stats is None and cache_path_from_data is None:
        return 1

    topology_cache = _load_topology_cache(args, cache_path_from_data)
    if topology_cache is None:
        return 1

    print(f"\nLoading {args.model_type.upper()} model...")
    model_path = _resolve_model_path(args, suffixes, checkpoint_suffix)
    model, arch_kwargs, ckpt_norm_stats = _load_gnn_model_from_checkpoint(model_path, args, device)
    if model is None:
        return 1
    if ckpt_norm_stats is not None:
        norm_stats = ckpt_norm_stats

    _maybe_alias_constraint_norm_stats(args, norm_stats)

    # Per-cell validation.
    all_results = []
    successful = failed = 0
    for cell_name in cell_list:
        result = run_cell_validation(args, cell_name, topology_cache, norm_stats, model, device)
        if result is not None:
            all_results.append(result)
            successful += 1
        else:
            failed += 1

    _print_summary(args, cell_list, all_results, successful, failed)

    if args.save_results and all_results:
        _save_results_npy(args, all_results, arch_kwargs)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
