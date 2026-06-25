#!/usr/bin/env python
"""
Task-level Fine-tuning Analysis Script

Analyzes fine-tuning effects on interpolation vs extrapolation performance
for individual tasks. Helps diagnose why interpolation is strong but
extrapolation is weak.

Usage:
  python analyze_task_finetuning.py --cell_name HA1D0BWP30P140 --num_tasks 5 --gpu 0
"""

import os
import sys

def get_gpu_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return '0'

os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_from_args()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch_geometric.data import Data, Batch
from pathlib import Path

sys.path.append('../../../model_code/')
sys.path.append('../../../data_processing/gnn/')

from gnn_maml import create_maml_gcn_model


def normalize_node_features(node_features, norm_stats):
    """
    Normalize node features using saved statistics.
    Supports both zscore (mean/std) and minmax (min/max/epsilon) normalization.
    """
    if norm_stats is None:
        return node_features

    normalized = node_features.clone()
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

    # Normalize temperature (column 10)
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

    return normalized


class TaskAnalyzer:
    """Analyzes fine-tuning effects on individual tasks."""

    def __init__(self, model, topology_cache, norm_stats, graph_mode='stage_aware', device='cuda'):
        self.model = model
        self.topology_cache = topology_cache
        self.norm_stats = norm_stats
        self.graph_mode = graph_mode
        self.device = device

    def create_pyg_data(self, sample, cell_name):
        """Create PyG Data object from sample."""
        node_features = sample['node_features']
        cell_cache = self.topology_cache[cell_name]

        if self.graph_mode == 'stage_aware':
            output_name = sample.get('output_name', '')
            delay_type = sample.get('delay_type', 'rise')

            if not output_name and 'output_topologies' in cell_cache:
                output_name = list(cell_cache['output_topologies'].keys())[0]

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
        return Data(x=node_features, edge_index=edge_index)

    def get_predictions(self, model, samples, cell_name, scale=None, offset=None):
        """Get predictions for all samples with optional linear scaling."""
        model.eval()
        predictions = []

        with torch.no_grad():
            for sample in samples:
                data = self.create_pyg_data(sample, cell_name)
                batch = Batch.from_data_list([data]).to(self.device)
                pred = model(batch).item()
                predictions.append(pred)

        predictions = np.array(predictions)

        # Apply linear scaling if provided
        if scale is not None and offset is not None:
            predictions = predictions * scale + offset

        return predictions

    def compute_linear_scaling(self, model, support_samples, support_targets, cell_name):
        """
        Compute linear scaling parameters (scale, offset) from support set.
        Maps model outputs to target range using least squares fit.
        """
        # Get model predictions for support set
        support_preds = self.get_predictions(model, support_samples, cell_name)
        support_targets_np = support_targets.numpy()

        # Compute linear fit: target = scale * pred + offset
        # Using least squares: minimize ||scale * pred + offset - target||^2
        pred_mean = support_preds.mean()
        target_mean = support_targets_np.mean()

        pred_centered = support_preds - pred_mean
        target_centered = support_targets_np - target_mean

        # scale = cov(pred, target) / var(pred)
        var_pred = np.var(support_preds)
        if var_pred > 1e-10:
            scale = np.sum(pred_centered * target_centered) / np.sum(pred_centered ** 2)
        else:
            scale = 1.0

        offset = target_mean - scale * pred_mean

        return scale, offset

    def fine_tune_and_track(self, samples, targets, cell_name,
                            support_indices, adam_steps=40, lr=3e-4,
                            track_every=5):
        """
        Fine-tune model and track predictions at each step.
        Uses linear scaling to map model outputs to target range.

        Returns:
            dict with keys:
                - 'before': predictions before fine-tuning (with linear scaling)
                - 'after': predictions after fine-tuning (with linear scaling)
                - 'trajectory': list of predictions at each tracked step
                - 'losses': training losses
        """
        # Clone model for fine-tuning
        model_ft = create_maml_gcn_model(
            node_features=self.model.convs[0].lin.weight.shape[1],
            conv_hidden_dim=self.model.conv_hidden_dim,
            num_conv_layers=self.model.num_conv_layers,
            fc_hidden_dim=self.model.fc_hidden_dim,
            num_fc_layers=self.model.num_fc_layers,
            pooling=self.model.pooling_type,
            output_dim=1,
            dropout=0.0
        ).to(self.device)
        model_ft.load_state_dict(self.model.state_dict())

        # Prepare support set
        support_samples = [samples[i] for i in support_indices]
        support_targets_cpu = targets[support_indices]

        # Compute linear scaling BEFORE fine-tuning
        scale_before, offset_before = self.compute_linear_scaling(
            model_ft, support_samples, support_targets_cpu, cell_name
        )

        # Get predictions before fine-tuning (with scaling)
        pred_before = self.get_predictions(model_ft, samples, cell_name, scale_before, offset_before)

        # Prepare for fine-tuning
        support_targets = support_targets_cpu.to(self.device).view(-1, 1)

        # Normalize support targets for training
        y_mean = support_targets.mean()
        y_std = support_targets.std()
        if y_std > 0:
            support_targets_norm = (support_targets - y_mean) / y_std
        else:
            support_targets_norm = support_targets - y_mean

        # Create support batch
        support_batch_data = []
        for s in support_samples:
            data = self.create_pyg_data(s, cell_name)
            support_batch_data.append(data)
        support_batch = Batch.from_data_list(support_batch_data).to(self.device)

        # Fine-tuning
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_ft.parameters(), lr=lr, weight_decay=1e-4)

        losses = []
        trajectory = [pred_before.copy()]

        model_ft.train()
        for step in range(adam_steps):
            loss = criterion(model_ft(support_batch), support_targets_norm)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % track_every == 0:
                # Recompute scaling at each step
                model_ft.eval()
                scale_step, offset_step = self.compute_linear_scaling(
                    model_ft, support_samples, support_targets_cpu, cell_name
                )
                pred_step = self.get_predictions(model_ft, samples, cell_name, scale_step, offset_step)
                trajectory.append(pred_step)
                model_ft.train()

        # Compute scaling AFTER fine-tuning
        model_ft.eval()
        scale_after, offset_after = self.compute_linear_scaling(
            model_ft, support_samples, support_targets_cpu, cell_name
        )

        # Get predictions after fine-tuning (with scaling)
        pred_after = self.get_predictions(model_ft, samples, cell_name, scale_after, offset_after)

        return {
            'before': pred_before,
            'after': pred_after,
            'trajectory': trajectory,
            'losses': losses,
            'y_mean': y_mean.item(),
            'y_std': y_std.item(),
            'scale_before': scale_before,
            'offset_before': offset_before,
            'scale_after': scale_after,
            'offset_after': offset_after
        }

    def analyze_task(self, samples, targets, cell_name, task_info,
                     left_bound=5, right_bound=56, adam_steps=40):
        """
        Analyze a single task's fine-tuning behavior.

        Returns analysis results and creates visualization.
        """
        total_points = len(samples)
        support_indices = list(range(left_bound, right_bound))

        # Run fine-tuning analysis
        results = self.fine_tune_and_track(
            samples, targets, cell_name, support_indices,
            adam_steps=adam_steps, track_every=10
        )

        # Calculate metrics for different regions
        targets_np = targets.numpy()

        def calc_metrics(pred, actual):
            mse = np.mean((pred - actual) ** 2)
            mae = np.mean(np.abs(pred - actual))
            # NRMSE normalized by mean
            nrmse = np.sqrt(mse) / (np.abs(actual.mean()) + 1e-8) * 100
            return {'mse': mse, 'mae': mae, 'nrmse': nrmse}

        # Before fine-tuning metrics
        metrics_before = {
            'total': calc_metrics(results['before'], targets_np),
            'left_ex': calc_metrics(results['before'][:left_bound], targets_np[:left_bound]),
            'inter': calc_metrics(results['before'][left_bound:right_bound], targets_np[left_bound:right_bound]),
            'right_ex': calc_metrics(results['before'][right_bound:], targets_np[right_bound:])
        }

        # After fine-tuning metrics
        metrics_after = {
            'total': calc_metrics(results['after'], targets_np),
            'left_ex': calc_metrics(results['after'][:left_bound], targets_np[:left_bound]),
            'inter': calc_metrics(results['after'][left_bound:right_bound], targets_np[left_bound:right_bound]),
            'right_ex': calc_metrics(results['after'][right_bound:], targets_np[right_bound:])
        }

        return {
            'task_info': task_info,
            'results': results,
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'targets': targets_np,
            'left_bound': left_bound,
            'right_bound': right_bound
        }

    def plot_task_analysis(self, analysis, save_path=None):
        """Create visualization for task analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        results = analysis['results']
        targets = analysis['targets']
        left_bound = analysis['left_bound']
        right_bound = analysis['right_bound']
        total_points = len(targets)
        x = np.arange(total_points)

        # Plot 1: Before vs After predictions
        ax1 = axes[0, 0]
        ax1.plot(x, targets, 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax1.plot(x, results['before'], 'b--', linewidth=1.5, label='Before FT', alpha=0.7)
        ax1.plot(x, results['after'], 'r-', linewidth=1.5, label='After FT', alpha=0.7)
        ax1.axvspan(0, left_bound, alpha=0.2, color='orange', label='Left Extrap')
        ax1.axvspan(left_bound, right_bound, alpha=0.2, color='green', label='Interpolation')
        ax1.axvspan(right_bound, total_points, alpha=0.2, color='orange')
        ax1.set_xlabel('Voltage Index')
        ax1.set_ylabel('Delay (ps)')
        ax1.set_title(f"Task: {analysis['task_info']}")
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error comparison
        ax2 = axes[0, 1]
        regions = ['Left Ex', 'Interp', 'Right Ex', 'Total']
        before_nrmse = [
            analysis['metrics_before']['left_ex']['nrmse'],
            analysis['metrics_before']['inter']['nrmse'],
            analysis['metrics_before']['right_ex']['nrmse'],
            analysis['metrics_before']['total']['nrmse']
        ]
        after_nrmse = [
            analysis['metrics_after']['left_ex']['nrmse'],
            analysis['metrics_after']['inter']['nrmse'],
            analysis['metrics_after']['right_ex']['nrmse'],
            analysis['metrics_after']['total']['nrmse']
        ]

        bar_width = 0.35
        x_pos = np.arange(len(regions))
        ax2.bar(x_pos - bar_width/2, before_nrmse, bar_width, label='Before FT', color='blue', alpha=0.7)
        ax2.bar(x_pos + bar_width/2, after_nrmse, bar_width, label='After FT', color='red', alpha=0.7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(regions)
        ax2.set_ylabel('NRMSE (%)')
        ax2.set_title('NRMSE by Region')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, (b, a) in enumerate(zip(before_nrmse, after_nrmse)):
            ax2.text(i - bar_width/2, b + 0.1, f'{b:.1f}', ha='center', va='bottom', fontsize=8)
            ax2.text(i + bar_width/2, a + 0.1, f'{a:.1f}', ha='center', va='bottom', fontsize=8)

        # Plot 3: Training loss curve
        ax3 = axes[1, 0]
        ax3.plot(results['losses'], 'g-', linewidth=1.5)
        ax3.set_xlabel('Fine-tuning Step')
        ax3.set_ylabel('MSE Loss')
        ax3.set_title('Fine-tuning Loss Curve')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Error trajectory
        ax4 = axes[1, 1]
        trajectory = results['trajectory']
        steps = [0] + list(range(10, len(results['losses']) + 1, 10))

        inter_errors = []
        left_errors = []
        right_errors = []

        for pred in trajectory:
            inter_errors.append(np.mean(np.abs(pred[left_bound:right_bound] - targets[left_bound:right_bound])))
            left_errors.append(np.mean(np.abs(pred[:left_bound] - targets[:left_bound])))
            right_errors.append(np.mean(np.abs(pred[right_bound:] - targets[right_bound:])))

        ax4.plot(steps, inter_errors, 'g-o', label='Interpolation MAE', markersize=4)
        ax4.plot(steps, left_errors, 'b-s', label='Left Extrap MAE', markersize=4)
        ax4.plot(steps, right_errors, 'r-^', label='Right Extrap MAE', markersize=4)
        ax4.set_xlabel('Fine-tuning Step')
        ax4.set_ylabel('MAE')
        ax4.set_title('Error Trajectory During Fine-tuning')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.close()

        return fig


def load_cell_dataset(cell_path, topology_cache=None):
    """Load cell test data."""
    data = torch.load(cell_path, weights_only=False, map_location='cpu')
    return data


def main():
    parser = argparse.ArgumentParser(description='Task-level Fine-tuning Analysis')

    # Dataset paths
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN_unified/',
                        help='Dataset directory')
    parser.add_argument('--cell_name', type=str, default='HA1D0BWP30P140',
                        help='Cell name to analyze')

    # Model settings
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                        choices=['stage_aware', 'full_graph'])
    parser.add_argument('--data_type', type=str, default='cell')

    # Architecture
    parser.add_argument('--conv_hidden_dim', type=int, default=64)
    parser.add_argument('--num_conv_layers', type=int, default=2)
    parser.add_argument('--fc_hidden_dim', type=int, default=256)
    parser.add_argument('--num_fc_layers', type=int, default=2)

    # Analysis settings
    parser.add_argument('--num_tasks', type=int, default=5,
                        help='Number of tasks to analyze')
    parser.add_argument('--adam_steps', type=int, default=40,
                        help='Number of Adam fine-tuning steps')
    parser.add_argument('--left_bound', type=int, default=5)
    parser.add_argument('--right_bound', type=int, default=56)

    # Output
    parser.add_argument('--output_dir', type=str, default='./finetuning_analysis')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load topology cache
    cache_dir = '/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache'
    cache_files = list(Path(cache_dir).glob(f'{args.graph_mode}_topology_cache_tsmc*.pth'))

    if not cache_files:
        print(f"No topology cache found in {cache_dir}")
        return

    topology_cache = torch.load(cache_files[0], weights_only=False)
    print(f"Loaded topology cache: {cache_files[0].name}")

    # Load train data for norm_stats
    train_path = Path(args.dataset_dir) / f'train_{args.data_type}_{args.graph_mode}.pth'
    train_data = torch.load(train_path, weights_only=False, map_location='cpu')
    norm_stats = train_data.get('norm_stats', None)

    # Find node_features dimension from train data
    node_features_dim = train_data['num_features']
    print(f"Node features dimension: {node_features_dim}")

    # Load model
    model_path = None
    if args.model_path and os.path.exists(args.model_path):
        model_path = args.model_path
    else:
        # Try to find a model
        model_dir = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/GCN'
        model_pattern = f'gnn_maml_tsmc_process_{args.data_type}_{args.graph_mode}*.pth'
        model_files = list(Path(model_dir).glob(model_pattern))

        if model_files:
            model_path = model_files[0]

    if model_path is None:
        print(f"No model found")
        print("Creating untrained model for demonstration...")
        model = create_maml_gcn_model(
            node_features=node_features_dim,
            conv_hidden_dim=args.conv_hidden_dim,
            num_conv_layers=args.num_conv_layers,
            fc_hidden_dim=args.fc_hidden_dim,
            num_fc_layers=args.num_fc_layers,
            pooling='output',
            output_dim=1
        ).to(device)
    else:
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)

        # Get architecture from checkpoint
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = create_maml_gcn_model(
                node_features=config.get('node_features', node_features_dim),
                conv_hidden_dim=config.get('conv_hidden_dim', args.conv_hidden_dim),
                num_conv_layers=config.get('num_conv_layers', args.num_conv_layers),
                fc_hidden_dim=config.get('fc_hidden_dim', args.fc_hidden_dim),
                num_fc_layers=config.get('num_fc_layers', args.num_fc_layers),
                pooling=config.get('pooling', 'output'),
                output_dim=1
            ).to(device)
        else:
            model = create_maml_gcn_model(
                node_features=node_features_dim,
                conv_hidden_dim=args.conv_hidden_dim,
                num_conv_layers=args.num_conv_layers,
                fc_hidden_dim=args.fc_hidden_dim,
                num_fc_layers=args.num_fc_layers,
                pooling='output',
                output_dim=1
            ).to(device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    # Load cell test data (directory includes data_type to separate cell vs transition)
    test_dir = Path(args.dataset_dir) / f'test_by_{args.data_type}_{args.graph_mode}'
    cell_path = test_dir / f'{args.cell_name}.pth'

    if not cell_path.exists():
        print(f"Cell data not found: {cell_path}")
        available = list(test_dir.glob('*.pth'))[:5]
        print(f"Available cells: {[p.stem for p in available]}")
        return

    cell_data = torch.load(cell_path, weights_only=False, map_location='cpu')
    print(f"Loaded cell: {args.cell_name}")
    print(f"  Num libs: {cell_data['num_libs']}")
    print(f"  Num tasks: {cell_data['num_tasks']}")

    # Create analyzer
    analyzer = TaskAnalyzer(model, topology_cache, norm_stats, args.graph_mode, device)

    # Analyze random tasks
    num_tasks = min(args.num_tasks, cell_data['num_tasks'])
    task_indices = np.random.choice(cell_data['num_tasks'], num_tasks, replace=False)

    print(f"\nAnalyzing {num_tasks} tasks...")

    summary_results = []

    for i, task_idx in enumerate(task_indices):
        print(f"\n{'='*60}")
        print(f"Task {i+1}/{num_tasks}: index={task_idx}")

        # Get task data for all libs (61 voltage points)
        node_slices = cell_data['node_slices']
        start_idx = node_slices[task_idx].item()
        end_idx = node_slices[task_idx + 1].item()

        samples = []
        for lib_idx in range(cell_data['num_libs']):
            node_features = cell_data['node_features'][lib_idx, start_idx:end_idx, :].clone()

            # Normalize
            if norm_stats:
                node_features = normalize_node_features(node_features, norm_stats)

            sample = {
                'node_features': node_features,
                'cell_name': args.cell_name,
                'output_name': cell_data.get('output_names', [''])[task_idx] if cell_data.get('output_names') else '',
                'delay_type': cell_data.get('delay_types', ['rise'])[task_idx] if cell_data.get('delay_types') else 'rise'
            }
            samples.append(sample)

        targets = cell_data['outputs'][:, task_idx]

        task_info = f"{args.cell_name}_task{task_idx}"
        if cell_data.get('delay_types'):
            task_info += f"_{cell_data['delay_types'][task_idx]}"
        if cell_data.get('output_names'):
            task_info += f"_{cell_data['output_names'][task_idx]}"

        # Analyze
        analysis = analyzer.analyze_task(
            samples, targets, args.cell_name, task_info,
            left_bound=args.left_bound, right_bound=args.right_bound,
            adam_steps=args.adam_steps
        )

        # Print summary
        print(f"\nMetrics Summary:")
        print(f"  {'Region':<12} {'Before NRMSE':>15} {'After NRMSE':>15} {'Change':>10}")
        print(f"  {'-'*52}")
        for region in ['left_ex', 'inter', 'right_ex', 'total']:
            before = analysis['metrics_before'][region]['nrmse']
            after = analysis['metrics_after'][region]['nrmse']
            change = after - before
            sign = '+' if change > 0 else ''
            print(f"  {region:<12} {before:>15.2f} {after:>15.2f} {sign}{change:>9.2f}")

        # Save plot
        save_path = output_dir / f'{task_info}_analysis.png'
        analyzer.plot_task_analysis(analysis, save_path)

        summary_results.append({
            'task_info': task_info,
            'before_inter': analysis['metrics_before']['inter']['nrmse'],
            'after_inter': analysis['metrics_after']['inter']['nrmse'],
            'before_left': analysis['metrics_before']['left_ex']['nrmse'],
            'after_left': analysis['metrics_after']['left_ex']['nrmse'],
            'before_right': analysis['metrics_before']['right_ex']['nrmse'],
            'after_right': analysis['metrics_after']['right_ex']['nrmse'],
        })

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    avg_before_inter = np.mean([r['before_inter'] for r in summary_results])
    avg_after_inter = np.mean([r['after_inter'] for r in summary_results])
    avg_before_left = np.mean([r['before_left'] for r in summary_results])
    avg_after_left = np.mean([r['after_left'] for r in summary_results])
    avg_before_right = np.mean([r['before_right'] for r in summary_results])
    avg_after_right = np.mean([r['after_right'] for r in summary_results])

    print(f"\nAverage NRMSE (%):")
    print(f"  {'Region':<15} {'Before FT':>12} {'After FT':>12} {'Change':>12}")
    print(f"  {'-'*51}")
    print(f"  {'Interpolation':<15} {avg_before_inter:>12.2f} {avg_after_inter:>12.2f} {avg_after_inter - avg_before_inter:>+12.2f}")
    print(f"  {'Left Extrap':<15} {avg_before_left:>12.2f} {avg_after_left:>12.2f} {avg_after_left - avg_before_left:>+12.2f}")
    print(f"  {'Right Extrap':<15} {avg_before_right:>12.2f} {avg_after_right:>12.2f} {avg_after_right - avg_before_right:>+12.2f}")

    print(f"\nConclusion:")
    inter_improved = avg_after_inter < avg_before_inter
    left_improved = avg_after_left < avg_before_left
    right_improved = avg_after_right < avg_before_right

    print(f"  - Interpolation: {'✅ Improved' if inter_improved else '❌ Degraded'} ({avg_after_inter - avg_before_inter:+.2f}%)")
    print(f"  - Left Extrap:   {'✅ Improved' if left_improved else '❌ Degraded'} ({avg_after_left - avg_before_left:+.2f}%)")
    print(f"  - Right Extrap:  {'✅ Improved' if right_improved else '❌ Degraded'} ({avg_after_right - avg_before_right:+.2f}%)")

    if inter_improved and not (left_improved and right_improved):
        print(f"\n⚠️  Fine-tuning improves interpolation but hurts extrapolation!")
        print(f"   Consider reducing adam_steps or using a smaller learning rate.")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == '__main__':
    main()
