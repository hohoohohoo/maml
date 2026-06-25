#!/usr/bin/env python
"""
Compare and visualize GCN Topology Validation Results.

Loads .npy prediction/actual files from TSMC_GCN_topology_validation.py
and creates comparison plots grouped by cell name.

File naming convention:
- TSMC_GCN_{experiment}_{cell_name}_{data_type}_{graph_mode}_{mode}_{model_type}_..._pred.npy
- Example: TSMC_GCN_intra_topology_AN4D0BWP30P140_cell_full_graph_interpolation_maml_innerdiv10_meta16_iter300000_inner1_conv32x2_fc128x2_pred.npy

Usage:
    python compare_gcn_topology_validation_results.py --results_dir 10000samples --mode interpolation
    python compare_gcn_topology_validation_results.py --results_dir 10000samples --experiment intra_topology
    python compare_gcn_topology_validation_results.py --results_dir 10000samples --cell AN4D0BWP30P140
"""

import os
import sys
import numpy as np
import argparse
import glob
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def parse_filename(filename):
    """
    Parse topology validation result filename to extract configuration.

    Example filenames:
    TSMC_GCN_intra_topology_AN4D0BWP30P140_cell_full_graph_interpolation_maml_innerdiv10_meta16_iter300000_inner1_conv32x2_fc128x2_pred.npy
    TSMC_GCN_topology_agnostic_FA1D0BWP30P140_cell_stage_aware_extrapolation_baseline_iter100000_conv128x3_fc40x2_pred.npy
    """
    basename = os.path.basename(filename)

    # Remove _pred.npy or _act.npy suffix
    name = basename.replace('_pred.npy', '').replace('_act.npy', '')

    config = {
        'filename': basename,
        'full_name': name,
    }

    # Extract architecture params using regex
    conv_match = re.search(r'conv(\d+)x(\d+)', name)
    fc_match = re.search(r'fc(\d+)x(\d+)', name)

    if conv_match:
        config['conv_hidden_dim'] = int(conv_match.group(1))
        config['num_conv_layers'] = int(conv_match.group(2))

    if fc_match:
        config['fc_hidden_dim'] = int(fc_match.group(1))
        config['num_fc_layers'] = int(fc_match.group(2))

    # Extract experiment type
    if 'intra_topology' in name:
        config['experiment'] = 'intra_topology'
    elif 'topology_agnostic' in name:
        config['experiment'] = 'topology_agnostic'

    # Extract cell name - pattern: after experiment type, before _cell_
    # TSMC_GCN_{experiment}_{cell_name}_cell_{graph_mode}...
    cell_match = re.search(r'TSMC_GCN_(?:intra_topology|topology_agnostic)_([A-Za-z0-9]+)_cell_', name)
    if cell_match:
        config['cell_name'] = cell_match.group(1)

    # Extract mode
    if 'interpolation' in name:
        config['mode'] = 'interpolation'
    elif 'extrapolation' in name:
        config['mode'] = 'extrapolation'

    # Extract model type
    if '_baseline_' in name:
        config['model_type'] = 'baseline'
    elif '_maml_' in name:
        config['model_type'] = 'maml'

    # Extract graph mode
    if 'stage_aware' in name:
        config['graph_mode'] = 'stage_aware'
    elif 'full_graph' in name:
        config['graph_mode'] = 'full_graph'

    # Check if filtered
    config['filtered'] = 'filtered' in name

    # Extract pooling mode
    pool_match = re.search(r'_pool(output|max|add|mean)', name)
    if pool_match:
        config['pooling'] = pool_match.group(1)
    else:
        config['pooling'] = 'mean'  # default

    # Create architecture label
    model_type = config.get('model_type', 'unknown')
    graph_mode = config.get('graph_mode', 'unknown')
    experiment = config.get('experiment', 'unknown')
    cell_name = config.get('cell_name', 'unknown')

    graph_short = 'SA' if graph_mode == 'stage_aware' else 'FG' if graph_mode == 'full_graph' else '?'
    model_short = 'B' if model_type == 'baseline' else 'M' if model_type == 'maml' else '?'
    exp_short = 'IT' if experiment == 'intra_topology' else 'TA' if experiment == 'topology_agnostic' else '?'

    if 'conv_hidden_dim' in config and 'fc_hidden_dim' in config:
        arch_params = f"c{config['conv_hidden_dim']}x{config['num_conv_layers']}_f{config['fc_hidden_dim']}x{config['num_fc_layers']}"
        config['arch_label'] = f"{model_short}_{graph_short}_{exp_short}_{cell_name}_{arch_params}"
        config['arch_only_label'] = f"{model_short}_{graph_short}_{arch_params}"
        config['cell_arch_label'] = f"{cell_name}_{arch_params}"
    else:
        config['arch_label'] = name
        config['arch_only_label'] = name
        config['cell_arch_label'] = name

    return config


def compute_metrics(predictions, actuals, group_size=61):
    """
    Compute various error metrics with 61-group averaging.

    Args:
        predictions: predicted values
        actuals: actual values
        group_size: number of samples per group (default: 61 for voltage variations)

    Returns:
        dict: averaged metrics across all groups
    """
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Filter out invalid values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) | np.isinf(predictions) | np.isinf(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    if len(predictions) == 0:
        return None

    # Group by 61 samples
    n_groups = len(predictions) // group_size

    if n_groups == 0:
        n_groups = 1
        group_size = len(predictions)

    # Trim to exact group multiples
    predictions = predictions[:n_groups * group_size]
    actuals = actuals[:n_groups * group_size]

    # Reshape to (n_groups, group_size)
    pred_grouped = predictions.reshape(n_groups, group_size)
    act_grouped = actuals.reshape(n_groups, group_size)

    # Calculate metrics per group
    group_metrics = []

    for i in range(n_groups):
        y_p = pred_grouped[i]
        y_t = act_grouped[i]

        # MAE
        mae = np.mean(np.abs(y_p - y_t))

        # MAPE
        mask = y_t != 0
        mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100 if np.any(mask) else 0

        # SMAPE
        denom = np.abs(y_t) + np.abs(y_p)
        mask_smape = denom != 0
        smape = np.mean(
            2.0 * np.abs(y_t[mask_smape] - y_p[mask_smape]) / denom[mask_smape]
        ) * 100 if np.any(mask_smape) else 0

        # RMSE
        mse = np.mean((y_p - y_t) ** 2)
        rmse = np.sqrt(mse)

        # NRMSE
        y_range = np.max(y_t) - np.min(y_t)
        nrmse = (rmse / y_range * 100) if y_range > 0 else 0

        # R-squared
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0

        group_metrics.append({
            'mse': mse,
            'rmse': rmse,
            'nrmse': nrmse,
            'mae': mae,
            'mape': mape,
            'smape': smape,
            'r2': r2
        })

    # Average across groups
    return {
        'mse': np.mean([g['mse'] for g in group_metrics]),
        'rmse': np.mean([g['rmse'] for g in group_metrics]),
        'nrmse': np.mean([g['nrmse'] for g in group_metrics]),
        'mae': np.mean([g['mae'] for g in group_metrics]),
        'mape': np.mean([g['mape'] for g in group_metrics]),
        'smape': np.mean([g['smape'] for g in group_metrics]),
        'r2': np.mean([g['r2'] for g in group_metrics]),
        'num_samples': len(predictions),
        'num_groups': n_groups
    }


def load_results(results_dir):
    """Load all prediction/actual pairs from results directory."""
    results = []

    pred_files = glob.glob(os.path.join(results_dir, '*_pred.npy'))

    for pred_file in pred_files:
        # Only process topology validation files
        if 'topology' not in pred_file:
            continue

        act_file = pred_file.replace('_pred.npy', '_act.npy')

        if not os.path.exists(act_file):
            print(f"Warning: Missing actual file for {pred_file}")
            continue

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)

            config = parse_filename(pred_file)
            metrics = compute_metrics(predictions, actuals)

            if metrics is None:
                print(f"Warning: No valid data in {pred_file}")
                continue

            results.append({
                'config': config,
                'metrics': metrics,
                'predictions': predictions,
                'actuals': actuals
            })

        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
            continue

    return results


def plot_metrics_by_cell(results, output_dir, metric_name='nrmse'):
    """Create bar chart comparing a metric grouped by cell name."""
    if not results:
        print("No results to plot")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by cell name
    cell_groups = defaultdict(list)
    for r in results:
        cell_name = r['config'].get('cell_name', 'unknown')
        cell_groups[cell_name].append(r)

    # Sort cells
    sorted_cells = sorted(cell_groups.keys())

    # Calculate average metrics per cell
    cell_metrics = []
    for cell_name in sorted_cells:
        cell_results = cell_groups[cell_name]
        avg_metric = np.mean([r['metrics'][metric_name] for r in cell_results])
        best_metric = min([r['metrics'][metric_name] for r in cell_results])
        cell_metrics.append({
            'cell_name': cell_name,
            'avg': avg_metric,
            'best': best_metric,
            'count': len(cell_results)
        })

    # Sort by average metric
    cell_metrics_sorted = sorted(cell_metrics, key=lambda x: x['avg'])

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(cell_metrics_sorted))
    cell_names = [c['cell_name'] for c in cell_metrics_sorted]
    avg_values = [c['avg'] for c in cell_metrics_sorted]
    best_values = [c['best'] for c in cell_metrics_sorted]

    if metric_name == 'mae':
        avg_values = [v * 1000 for v in avg_values]
        best_values = [v * 1000 for v in best_values]

    # Bar chart
    width = 0.35
    bars_avg = ax.bar([i - width/2 for i in x], avg_values, width, label='Average', color='steelblue', alpha=0.8)
    bars_best = ax.bar([i + width/2 for i in x], best_values, width, label='Best', color='seagreen', alpha=0.8)

    ax.set_xlabel('Cell Name', fontsize=12)
    if metric_name == 'mae':
        ax.set_ylabel('MAE (x1000)', fontsize=12)
        ax.set_title(f'MAE (x1000) by Cell - Topology Validation', fontsize=14)
    else:
        ax.set_ylabel(metric_name.upper(), fontsize=12)
        ax.set_title(f'{metric_name.upper()} by Cell - Topology Validation', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(cell_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{metric_name}_by_cell.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_cell_architecture_comparison(results, output_dir, metric_name='nrmse'):
    """Create heatmap showing metric for each cell x architecture combination."""
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get unique cells and architectures
    cells = sorted(set(r['config'].get('cell_name', 'unknown') for r in results))
    archs = sorted(set(r['config'].get('arch_only_label', 'unknown') for r in results))

    if len(cells) == 0 or len(archs) == 0:
        print("Not enough data for heatmap")
        return

    # Build data matrix
    data = np.full((len(cells), len(archs)), np.nan)

    for r in results:
        cell_name = r['config'].get('cell_name', 'unknown')
        arch = r['config'].get('arch_only_label', 'unknown')

        if cell_name in cells and arch in archs:
            cell_idx = cells.index(cell_name)
            arch_idx = archs.index(arch)

            val = r['metrics'][metric_name]
            if metric_name == 'mae':
                val = val * 1000

            # If already has value, take average
            if np.isnan(data[cell_idx, arch_idx]):
                data[cell_idx, arch_idx] = val
            else:
                data[cell_idx, arch_idx] = (data[cell_idx, arch_idx] + val) / 2

    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(12, len(archs) * 0.8), max(8, len(cells) * 0.4)))

    # Mask NaN values for visualization
    masked_data = np.ma.masked_invalid(data)

    im = ax.imshow(masked_data, cmap='RdYlGn_r', aspect='auto')
    plt.colorbar(im, ax=ax, label=f'{metric_name.upper()}' + (' (x1000)' if metric_name == 'mae' else ''))

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(cells, fontsize=8)

    # Add text annotations
    for i in range(len(cells)):
        for j in range(len(archs)):
            if not np.isnan(data[i, j]):
                text_color = 'white' if data[i, j] > np.nanmedian(data) else 'black'
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=6, color=text_color)

    ax.set_title(f'{metric_name.upper()} by Cell x Architecture - Topology Validation', fontsize=12)
    ax.set_xlabel('Architecture', fontsize=11)
    ax.set_ylabel('Cell Name', fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{metric_name}_cell_arch_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_experiment_comparison(results, output_dir):
    """Compare intra_topology vs topology_agnostic experiments."""
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by experiment
    exp_groups = defaultdict(list)
    for r in results:
        exp = r['config'].get('experiment', 'unknown')
        exp_groups[exp].append(r)

    if len(exp_groups) < 2:
        print("Not enough experiment types for comparison")
        return

    experiments = sorted(exp_groups.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['nrmse', 'mape', 'mae']
    metric_labels = {'nrmse': 'NRMSE (%)', 'mape': 'MAPE (%)', 'mae': 'MAE (x1000)'}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        data = []
        labels = []
        for exp in experiments:
            values = [r['metrics'][metric] for r in exp_groups[exp]]
            if metric == 'mae':
                values = [v * 1000 for v in values]
            data.append(values)
            labels.append(exp.replace('_', '\n'))

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(experiments)]):
            patch.set_facecolor(color)

        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f'{metric_labels[metric]} by Experiment')
        ax.grid(True, alpha=0.3)

        # Add mean values as text
        for i, exp in enumerate(experiments):
            mean_val = np.mean(data[i])
            ax.annotate(f'μ={mean_val:.2f}', xy=(i + 1, mean_val), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)

    plt.suptitle('Experiment Type Comparison - Topology Validation', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'experiment_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def save_cell_summary(results, output_dir):
    """Save summary table grouped by cell."""
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by cell
    cell_groups = defaultdict(list)
    for r in results:
        cell_name = r['config'].get('cell_name', 'unknown')
        cell_groups[cell_name].append(r)

    lines = []
    lines.append("=" * 140)
    lines.append("GCN TOPOLOGY VALIDATION RESULTS - CELL SUMMARY")
    lines.append("=" * 140)
    lines.append("")

    sorted_cells = sorted(cell_groups.keys())

    # Overall summary
    lines.append(f"Total cells: {len(sorted_cells)}")
    lines.append(f"Total configurations: {len(results)}")
    lines.append("")

    # Per-cell summary
    lines.append("-" * 140)
    lines.append(f"{'Cell Name':<25} {'Count':<8} {'Avg NRMSE':<12} {'Best NRMSE':<12} {'Avg MAPE':<12} {'Best MAPE':<12} {'Avg MAE(x1000)':<15} {'Best MAE(x1000)':<15}")
    lines.append("-" * 140)

    cell_summaries = []
    for cell_name in sorted_cells:
        cell_results = cell_groups[cell_name]

        nrmse_vals = [r['metrics']['nrmse'] for r in cell_results]
        mape_vals = [r['metrics']['mape'] for r in cell_results]
        mae_vals = [r['metrics']['mae'] * 1000 for r in cell_results]

        summary = {
            'cell_name': cell_name,
            'count': len(cell_results),
            'avg_nrmse': np.mean(nrmse_vals),
            'best_nrmse': np.min(nrmse_vals),
            'avg_mape': np.mean(mape_vals),
            'best_mape': np.min(mape_vals),
            'avg_mae': np.mean(mae_vals),
            'best_mae': np.min(mae_vals)
        }
        cell_summaries.append(summary)

        lines.append(f"{cell_name:<25} {summary['count']:<8} {summary['avg_nrmse']:<12.4f} {summary['best_nrmse']:<12.4f} "
                    f"{summary['avg_mape']:<12.4f} {summary['best_mape']:<12.4f} {summary['avg_mae']:<15.4f} {summary['best_mae']:<15.4f}")

    lines.append("-" * 140)

    # Overall average
    overall_avg_nrmse = np.mean([s['avg_nrmse'] for s in cell_summaries])
    overall_avg_mape = np.mean([s['avg_mape'] for s in cell_summaries])
    overall_avg_mae = np.mean([s['avg_mae'] for s in cell_summaries])

    lines.append(f"{'OVERALL AVERAGE':<25} {len(results):<8} {overall_avg_nrmse:<12.4f} {'':<12} "
                f"{overall_avg_mape:<12.4f} {'':<12} {overall_avg_mae:<15.4f}")
    lines.append("")

    # Best configurations per cell
    lines.append("=" * 140)
    lines.append("BEST CONFIGURATION PER CELL (by NRMSE)")
    lines.append("=" * 140)
    lines.append("")

    lines.append(f"{'Cell Name':<25} {'NRMSE':<10} {'MAPE':<10} {'MAE(x1000)':<12} {'Architecture':<50}")
    lines.append("-" * 120)

    for cell_name in sorted_cells:
        cell_results = cell_groups[cell_name]
        best = min(cell_results, key=lambda x: x['metrics']['nrmse'])

        arch = best['config'].get('arch_only_label', 'unknown')
        nrmse = best['metrics']['nrmse']
        mape = best['metrics']['mape']
        mae = best['metrics']['mae'] * 1000

        lines.append(f"{cell_name:<25} {nrmse:<10.4f} {mape:<10.4f} {mae:<12.4f} {arch:<50}")

    lines.append("")
    lines.append("=" * 140)
    lines.append("END OF SUMMARY")
    lines.append("=" * 140)

    # Write to file
    output_path = os.path.join(output_dir, 'cell_summary.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {output_path}")

    # Print to console
    for line in lines:
        print(line)


def plot_architecture_ranking(results, output_dir):
    """Plot architecture ranking across all cells."""
    if not results:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by architecture
    arch_groups = defaultdict(list)
    for r in results:
        arch = r['config'].get('arch_only_label', 'unknown')
        arch_groups[arch].append(r)

    # Calculate average metrics per architecture
    arch_metrics = []
    for arch, arch_results in arch_groups.items():
        arch_metrics.append({
            'arch': arch,
            'avg_nrmse': np.mean([r['metrics']['nrmse'] for r in arch_results]),
            'avg_mape': np.mean([r['metrics']['mape'] for r in arch_results]),
            'avg_mae': np.mean([r['metrics']['mae'] for r in arch_results]) * 1000,
            'count': len(arch_results)
        })

    # Sort by NRMSE
    arch_metrics = sorted(arch_metrics, key=lambda x: x['avg_nrmse'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [('avg_nrmse', 'NRMSE (%)'), ('avg_mape', 'MAPE (%)'), ('avg_mae', 'MAE (x1000)')]

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]

        # Sort by this metric
        sorted_archs = sorted(arch_metrics, key=lambda x: x[metric_key])

        x = range(len(sorted_archs))
        values = [a[metric_key] for a in sorted_archs]
        labels = [a['arch'] for a in sorted_archs]

        bars = ax.bar(x, values, color='steelblue', alpha=0.8)

        # Highlight top 3
        for i in range(min(3, len(bars))):
            bars[i].set_color('seagreen')

        ax.set_xlabel('Architecture', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{metric_label} Ranking (Best to Worst)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels for top 5
        for i, (bar, val) in enumerate(zip(bars[:5], values[:5])):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.suptitle('Architecture Ranking - Topology Validation', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'architecture_ranking.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_results_table(results, output_file=None):
    """Print results as a formatted table."""
    if not results:
        print("No results to display")
        return

    # Sort by cell name, then by NRMSE
    results = sorted(results, key=lambda x: (x['config'].get('cell_name', ''), x['metrics']['nrmse']))

    lines = []
    lines.append("\n" + "=" * 140)
    lines.append("GCN TOPOLOGY VALIDATION RESULTS")
    lines.append("=" * 140)
    lines.append(f"{'Cell Name':<25} {'Experiment':<18} {'Arch':<30} {'NRMSE%':<10} {'MAPE%':<10} {'MAE(x1000)':<12} {'Samples':<10}")
    lines.append("-" * 140)

    current_cell = None
    for r in results:
        cell_name = r['config'].get('cell_name', 'unknown')
        experiment = r['config'].get('experiment', 'unknown')
        arch = r['config'].get('arch_only_label', 'unknown')[:28]
        nrmse = r['metrics']['nrmse']
        mape = r['metrics']['mape']
        mae = r['metrics']['mae'] * 1000
        samples = r['metrics']['num_samples']

        # Add separator between cells
        if current_cell and current_cell != cell_name:
            lines.append("-" * 140)
        current_cell = cell_name

        lines.append(f"{cell_name:<25} {experiment:<18} {arch:<30} {nrmse:<10.4f} {mape:<10.4f} {mae:<12.4f} {samples:<10}")

    lines.append("=" * 140)

    # Print to console
    for line in lines:
        print(line)

    # Save to file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare and visualize GCN Topology Validation Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compare_gcn_topology_validation_results.py --results_dir 10000samples --mode interpolation
    python compare_gcn_topology_validation_results.py --results_dir 10000samples --experiment intra_topology
    python compare_gcn_topology_validation_results.py --results_dir 10000samples --cell AN4D0BWP30P140
    python compare_gcn_topology_validation_results.py --results_dir 10000samples --mode interpolation --model_type maml
"""
    )

    parser.add_argument('--results_dir', type=str, default='',
                       help='Subdirectory under data_result_npy_directory (e.g., 10000samples)')
    parser.add_argument('--output_dir', type=str,
                       default='/home/tkdgn2907/Deepsets_test/MAML/Projects/result_management/result_summary/gcn_topology_validation_analysis',
                       help='Output directory for plots')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['interpolation', 'extrapolation'],
                       help='Filter by validation mode')
    parser.add_argument('--experiment', type=str, default=None,
                       choices=['intra_topology', 'topology_agnostic'],
                       help='Filter by experiment type')
    parser.add_argument('--cell', type=str, default=None,
                       help='Filter by cell name')
    parser.add_argument('--model_type', type=str, default=None,
                       choices=['baseline', 'maml'],
                       help='Filter by model type')
    parser.add_argument('--graph_mode', type=str, default=None,
                       choices=['stage_aware', 'full_graph'],
                       help='Filter by graph mode')
    parser.add_argument('--no_plots', action='store_true',
                       help='Only print table, skip generating plots')
    parser.add_argument('--data_filter', type=str, default='all',
                       choices=['all', 'filtered', 'unfiltered'],
                       help='Filter by data type: all (default), filtered (only filtered data), unfiltered (only non-filtered data)')

    args = parser.parse_args()

    # Build results directory path
    base_results_dir = '../pretraining/model_test_code/gnn/data_result_npy_directory'
    if args.results_dir:
        full_results_dir = os.path.join(base_results_dir, args.results_dir)
    else:
        full_results_dir = base_results_dir

    # Check results directory
    if not os.path.exists(full_results_dir):
        print(f"Error: Results directory not found: {full_results_dir}")
        return 1

    print(f"Loading results from: {full_results_dir}")
    results = load_results(full_results_dir)

    if not results:
        print("No valid topology validation results found")
        return 1

    print(f"Loaded {len(results)} result files")

    # Apply filters
    if args.mode:
        results = [r for r in results if r['config'].get('mode') == args.mode]
        print(f"Filtered to {len(results)} results with mode='{args.mode}'")

    if args.experiment:
        results = [r for r in results if r['config'].get('experiment') == args.experiment]
        print(f"Filtered to {len(results)} results with experiment='{args.experiment}'")

    if args.cell:
        results = [r for r in results if r['config'].get('cell_name') == args.cell]
        print(f"Filtered to {len(results)} results with cell='{args.cell}'")

    if args.model_type:
        results = [r for r in results if r['config'].get('model_type') == args.model_type]
        print(f"Filtered to {len(results)} results with model_type='{args.model_type}'")

    if args.graph_mode:
        results = [r for r in results if r['config'].get('graph_mode') == args.graph_mode]
        print(f"Filtered to {len(results)} results with graph_mode='{args.graph_mode}'")

    if args.data_filter == 'filtered':
        results = [r for r in results if 'filtered' in r['config'].get('full_name', '')]
        print(f"Filtered to {len(results)} results with data_filter='filtered'")
    elif args.data_filter == 'unfiltered':
        results = [r for r in results if 'filtered' not in r['config'].get('full_name', '')]
        print(f"Filtered to {len(results)} results with data_filter='unfiltered'")

    if not results:
        print("No results after filtering")
        return 1

    # Create output directory with filter info
    subdir_parts = ['topology']
    if args.mode:
        subdir_parts.append(args.mode)
    if args.experiment:
        subdir_parts.append(args.experiment)
    if args.cell:
        subdir_parts.append(args.cell)
    if args.model_type:
        subdir_parts.append(args.model_type)
    if args.graph_mode:
        subdir_parts.append(args.graph_mode)
    if args.data_filter != 'all':
        subdir_parts.append(args.data_filter)

    subdir_name = '_'.join(subdir_parts)
    output_dir = os.path.join(args.output_dir, subdir_name)

    # Print table and save
    txt_output_file = os.path.join(output_dir, 'results_table.txt')
    print_results_table(results, output_file=txt_output_file)

    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating plots to: {output_dir}")

        # Cell-based analysis
        save_cell_summary(results, output_dir)
        plot_metrics_by_cell(results, output_dir, 'nrmse')
        plot_metrics_by_cell(results, output_dir, 'mape')
        plot_metrics_by_cell(results, output_dir, 'mae')

        # Cell x Architecture heatmap
        plot_cell_architecture_comparison(results, output_dir, 'nrmse')
        plot_cell_architecture_comparison(results, output_dir, 'mape')

        # Experiment comparison
        plot_experiment_comparison(results, output_dir)

        # Architecture ranking
        plot_architecture_ranking(results, output_dir)

        print(f"\nAll plots saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
