#!/usr/bin/env python
"""
ASAP7 GCN Topology Validation Script — 2-D V×T variant (option α: flat random K)

Mirrors ASAP7_GCN_topology_validation.py but reads 2-D V×T test datasets in
'unified_4d_VT' format. The (V, T) plane is flattened to a single
length-(V*T) sample list per task (default 61 * 5 = 305 test samples per
task), and support-set --indices are interpreted as flat positions in
[0, total_points). Everything else (per-task MAML adaptation, metrics,
output prefixes) is identical to the 1-D script.

Usage:
  python ASAP7_GCN_topology_validation_2d.py --experiment intra_topology --model_type maml --gpu 1
  python ASAP7_GCN_topology_validation_2d.py --experiment topology_agnostic --model_type maml --gpu 1
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


# Cell lists for each experiment type (ASAP7 Process dataset, matches 1-D validation script)
INTRA_TOPOLOGY_CELLS = [
    'AND2x6_ASAP7_75t_L',
    'NAND3x2_ASAP7_75t_L',
    'NOR2xp67_ASAP7_75t_L',
    'OR2x6_ASAP7_75t_L',
]

TOPOLOGY_AGNOSTIC_CELLS = [
    'FAx1_ASAP7_75t_L', 'HAxp5_ASAP7_75t_L',
    'XNOR2x2_ASAP7_75t_L', 'XOR2x2_ASAP7_75t_L',
    'AO21x1_ASAP7_75t_L', 'AO32x1_ASAP7_75t_L', 'OAI22x1_ASAP7_75t_L',
]


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


class CellTestDataset2D:
    """
    Dataset class for loading per-cell 2-D V×T test data with mmap.

    Storage shape: node_features [V, T, total_nodes, F], outputs [V, T, num_tasks].
    Externally exposes a flat 1-D "lib" axis of length num_libs = V*T, indexed in
    lexicographic (v, t) order: flat_idx = v * num_temps + t.
    """
    def __init__(self, cell_path, topology_cache=None):
        self.cell_path = cell_path
        self.topology_cache = topology_cache
        self._load_data()

    def _load_data(self):
        """Load 4-D V×T cell test data with mmap"""
        data = torch.load(self.cell_path, weights_only=False, map_location='cpu', mmap=True)

        data_format = data.get('format', 'legacy')
        if data_format != 'unified_4d_VT':
            raise ValueError(
                f"Expected unified_4d_VT format, got: {data_format} ({self.cell_path})"
            )

        self._node_features = data['node_features']   # [V, T, total_nodes, F]
        self._outputs = data['outputs']               # [V, T, num_tasks]
        self._node_slices = data['node_slices']       # [num_tasks + 1]
        self._cell_name = data['cell_name']
        self._delay_types = data.get('delay_types', None)   # [num_tasks]
        self._output_names = data.get('output_names', None) # [num_tasks]

        self._num_voltages = int(data['num_voltages'])
        self._num_temps = int(data['num_temps'])
        self.num_libs = self._num_voltages * self._num_temps   # flat sample count per task
        self.num_tasks = int(data['num_tasks'])
        self.total_nodes = int(data['total_nodes'])
        self.cell_name = data['cell_name']

    def _flat_to_vt(self, lib_idx):
        v = lib_idx // self._num_temps
        t = lib_idx %  self._num_temps
        return v, t

    def get_task_data(self, task_idx, lib_idx, clone=True):
        """Get data for a specific task and flat (V*T) lib index."""
        v, t = self._flat_to_vt(lib_idx)
        node_start = self._node_slices[task_idx].item()
        node_end = self._node_slices[task_idx + 1].item()

        node_features = self._node_features[v, t, node_start:node_end, :]
        if clone:
            node_features = node_features.clone()

        output = self._outputs[v, t, task_idx].item()

        delay_type = 'rise'
        output_name = ''
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
            'voltage_idx': v,
            'temp_idx': t,
        }

    def get_all_libs_for_task(self, task_idx, clone=True):
        """Get data for all (V*T) libs for a specific task (flat order)."""
        samples = []
        outputs = []
        for lib_idx in range(self.num_libs):
            sample = self.get_task_data(task_idx, lib_idx, clone=clone)
            samples.append(sample)
            outputs.append(sample['output'])
        return samples, outputs

    def get_task_outputs(self, task_idx):
        """Get all (V*T) outputs for a specific task, flat order. For continuity checking."""
        return self._outputs[:, :, task_idx].reshape(-1)


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
    test_dir = f"test_by_{args.data_type}_{args.graph_mode}{topology_suffix}{voltage_suffix}{temp_suffix}{slew_suffix}_2d"
    cell_path = os.path.join(
        args.dataset_dir,
        test_dir,
        f"{cell_name}.pth"
    )
    print(f"   Test directory: {test_dir}")

    if not os.path.exists(cell_path):
        print(f"   Cell data not found: {cell_path}")
        return None

    cell_dataset = CellTestDataset2D(cell_path, topology_cache)
    print(f"   Test data: {os.path.basename(os.path.dirname(cell_path))}/{cell_name}.pth")
    print(f"   Loaded: {cell_dataset.num_tasks} tasks, {cell_dataset.num_libs} libs")

    # Set mode-dependent default indices
    indices = args.indices
    if indices is None:
        # 2-D V×T flat defaults: spread support points across [0, total_points).
        # Extrapolation: 3 support points at ~8% / 50% / 92% of the flat space.
        # Interpolation: 5 support points evenly spaced, anchors at endpoints.
        if args.mode == 'extrapolation':
            indices = [args.total_points // 12,
                       args.total_points // 2,
                       (11 * args.total_points) // 12]
        else:  # interpolation
            indices = [0,
                       args.total_points // 4,
                       args.total_points // 2,
                       (3 * args.total_points) // 4,
                       args.total_points - 1]

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
    total_mape = []
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

                # Create center input with nominal voltage (0.7V for ASAP7)
                NOMINAL_VOLTAGE = 0.7  # ASAP7 nominal voltage
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
                        mode=args.mode, adaptation_method=args.adaptation_method
                    )

                    if adam_used:
                        adam_condition_count += 1

                    all_predictions.extend(predictions)
                    all_actuals.extend(actual_values)

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


def main():
    parser = argparse.ArgumentParser(
        description='ASAP7 GCN Topology Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ASAP7_GCN_topology_validation.py --experiment intra_topology --model_type baseline --gpu 0
  python ASAP7_GCN_topology_validation.py --experiment topology_agnostic --model_type maml --gpu 1
  python ASAP7_GCN_topology_validation.py --experiment intra_topology --model_type maml --mode interpolation
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
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/gnn_dataset_asap7_2d',
                        help='Dataset directory')
    parser.add_argument('--data_type', type=str, default='cell',
                        choices=['cell', 'transition'],
                        help='Data type (default: cell)')
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
    parser.add_argument('--sampling', type=str, default='10pct',
                        help='Train sampling ratio suffix matching training (e.g., "10pct" → _samp10pct in checkpoint dir). '
                             'Empty string disables. Default: "10pct".')

    # Sampling configuration
    parser.add_argument('--indices', type=int, nargs='+', default=None,
                        help='Sampling indices for support set')
    parser.add_argument('--total_points', type=int, default=305,
                        help='Total number of (V*T) data points per task (default: 305 = 61 V × 5 T)')
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
    parser.add_argument('--output_prefix', type=str, default='ASAP7_GCN_2d',
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
                        choices=['selective_adam', 'adam', 'bilinear_residual'],
                        help='Adaptation method: selective_adam (grad/move + Adam if loss>1e-4), '
                             'adam (always Adam), or bilinear_residual (2-D V×T: 4 corner bilinear '
                             'prior + 1 center alpha rescale, no weight update). '
                             'bilinear_residual requires support of exactly 4 V×T corners + 1 center.')

    # Output directory
    parser.add_argument('--output_dir', type=str, default='final',
                        help='Output directory mode: "final" for data_result_npy_directory_final, otherwise data_result_npy_directory (default: final)')

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
    print(f"ASAP7 GCN Topology Validation")
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
    print(f"Cells: {len(cell_list)} cells")
    print(f"{'='*80}")

    # Load train data to get norm_stats and cache_path
    # Add topology_suffix for inputport (inputport affects node features, gatectrl only affects cache)
    topology_suffix = "_inputport" if args.inputport else ""
    # Add voltage_mode suffix if not all_nodes, _minmax suffix if normalization is minmax
    voltage_suffix = f"_{args.voltage_mode}" if args.voltage_mode != "all_nodes" else ""
    norm_suffix = "_minmax" if args.normalization == "minmax" else ""
    temp_suffix = "_temp_all" if args.temp_mode == "temp_all" else ""
    slew_suffix = "_relpin" if args.related_pin_only else ""
    sampling_suffix_train = f"_samp{args.sampling}" if args.sampling else ""
    train_path = os.path.join(args.dataset_dir, f"train_{args.data_type}_{args.graph_mode}{topology_suffix}{voltage_suffix}{norm_suffix}{temp_suffix}{slew_suffix}{sampling_suffix_train}_2d.pth")
    print(f"\nLoading train data for norm_stats: {train_path}", flush=True)

    if not os.path.exists(train_path):
        print(f"Train data not found: {train_path}")
        return 1

    print("Loading train_data with mmap...", flush=True)
    train_data = torch.load(train_path, weights_only=False, map_location='cpu', mmap=True)
    print("train_data loaded.", flush=True)
    norm_stats = train_data.get('norm_stats', None)

    # Use cache_path override if provided, otherwise from dataset file
    if args.cache_path:
        cache_path = args.cache_path
        print(f"Using cache_path override: {cache_path}")
    else:
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

    # Load model
    print(f"\nLoading {args.model_type.upper()} model...")

    if args.model_path is None:
        # Add pooling suffix only if not 'mean' (default)
        pool_suffix = f"_pool{args.pooling}" if args.pooling != 'mean' else ""
        arch_suffix = f"_conv{args.conv_hidden_dim}x{args.num_conv_layers}_fc{args.fc_hidden_dim}x{args.num_fc_layers}{pool_suffix}"

        # Extract cache suffix from cache_path (gatectrl/bidir/directmos only affect adjacency, not train data)
        cache_suffix = ""
        if args.cache_path:
            if "_gatectrl" in args.cache_path:
                cache_suffix += "_gatectrl"
            if "_bidir" in args.cache_path:
                cache_suffix += "_bidir"
            if "_directmos" in args.cache_path:
                cache_suffix += "_directmos"
        # Add inputport suffix if flag is set (inputport affects node features)
        topology_suffix = "_inputport" if args.inputport else ""
        # checkpoint_suffix: combines cache_suffix (gatectrl) + topology_suffix (inputport)
        checkpoint_suffix = cache_suffix + topology_suffix

        # Add voltage_mode suffix if not all_nodes
        voltage_suffix = f"_{args.voltage_mode}" if args.voltage_mode != "all_nodes" else ""
        # Add minmax suffix if normalization is minmax
        norm_suffix = "_minmax" if args.normalization == "minmax" else ""
        # Add temp_all suffix if temp_mode is temp_all
        temp_suffix = "_temp_all" if args.temp_mode == "temp_all" else ""
        # Add relpin suffix if related_pin_only is set
        slew_suffix = "_relpin" if args.related_pin_only else ""
        # Sampling suffix (ASAP7 specific): training uses _samp{ratio}; validation must match
        sampling_suffix = f"_samp{args.sampling}" if args.sampling else ""

        if args.model_type == 'baseline':
            model_dir = f"../../../pretrained_models/gnn_baseline_asap7_process_2d_checkpoints{checkpoint_suffix}{voltage_suffix}{norm_suffix}{temp_suffix}{slew_suffix}{sampling_suffix}"
            model_filename = f"gnn_baseline_asap7_process_2d_{args.data_type}_{args.graph_mode}_iter{args.num_iterations}{arch_suffix}.pth"
        else:
            model_dir = f"../../../pretrained_models/gnn_maml_asap7_process_2d_checkpoints{checkpoint_suffix}{voltage_suffix}{norm_suffix}{temp_suffix}{slew_suffix}{sampling_suffix}"
            model_filename = f"gnn_maml_asap7_process_2d_{args.data_type}_{args.graph_mode}_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}.pth"

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

    # Save results if requested
    if args.save_results and all_results:
        pool_suffix = f"_pool{pooling_mode}" if pooling_mode != 'mean' else ""
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}{pool_suffix}"
        filter_suffix = "_filtered" if args.filter_continuous else ""

        # Add voltage_mode suffix (vdd_only -> _vddonly, vdd_mos -> _vddmos, all_nodes -> no suffix)
        voltage_suffix = ""
        if args.voltage_mode == 'vdd_only':
            voltage_suffix = "_vddonly"
        elif args.voltage_mode == 'vdd_mos':
            voltage_suffix = "_vddmos"

        # Add related_pin_only suffix
        relpin_suffix = "_relpin" if args.related_pin_only else ""

        for result in all_results:
            cell_name = result['cell_name']

            # Adaptation method suffix: selective_adam (default) has no suffix
            if args.adaptation_method == 'adam':
                adapt_suffix = "_adam"
            elif args.adaptation_method == 'bilinear_residual':
                adapt_suffix = "_bilinear"
            else:
                adapt_suffix = ""

            if args.model_type == 'baseline':
                base_name = f"{args.output_prefix}_{args.experiment}_{cell_name}_{args.data_type}_{args.graph_mode}_{args.mode}_{args.model_type}_iter{args.num_iterations}{arch_suffix}{filter_suffix}{voltage_suffix}{relpin_suffix}{adapt_suffix}"
            else:
                base_name = f"{args.output_prefix}_{args.experiment}_{cell_name}_{args.data_type}_{args.graph_mode}_{args.mode}_{args.model_type}_innerdiv{args.innerdiv}_meta{args.tasks_per_meta_batch}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}{filter_suffix}{voltage_suffix}{relpin_suffix}{adapt_suffix}"

            # Determine output directory based on output_dir argument
            if args.output_dir == "final":
                output_dir = "data_result_npy_directory_final"
            elif args.output_dir == "default":
                output_dir = "data_result_npy_directory"
            else:
                output_dir = args.output_dir  # custom path

            pred_filename = f"{output_dir}/{base_name}_pred.npy"
            act_filename = f"{output_dir}/{base_name}_act.npy"

            os.makedirs(output_dir, exist_ok=True)

            np.save(pred_filename, result['predictions'])
            np.save(act_filename, result['actuals'])

        # Determine output directory for print message
        if args.output_dir == "final":
            output_dir = "data_result_npy_directory_final"
        elif args.output_dir == "default":
            output_dir = "data_result_npy_directory"
        else:
            output_dir = args.output_dir
        print(f"\nSaved results to {output_dir}/")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
