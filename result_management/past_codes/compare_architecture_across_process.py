#!/usr/bin/env python
"""
Compare a single architecture's performance across all process-corner combinations.

Shows how one architecture performs across different process corners
(e.g., LVT_FF, LVT_TT, LVT_SS, RVT_FF, etc.)

Location: Projects/result_management/compare_architecture_across_process.py

Usage:
    python compare_architecture_across_process.py --mode interpolation --model_type baseline --graph_mode full_graph --conv 64x2 --fc 128x2
    python compare_architecture_across_process.py --mode interpolation --model_type maml --graph_mode stage_aware --conv 64x3 --fc 64x2
"""

import os
import sys
import numpy as np
import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm


# All process-corner combinations
PROCESSES = ['LVT', 'RVT', 'SLVT', 'SRAM']
CORNERS = ['FF', 'TT', 'SS']


def parse_filename(filename):
    """
    Parse validation result filename to extract configuration.

    Example filename:
    GCN_unified_LVT_FF_cell_full_graph_interpolation_baseline_iter100000_conv128x3_fc128x2_pred.npy
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
        config['conv_str'] = f"{conv_match.group(1)}x{conv_match.group(2)}"

    if fc_match:
        config['fc_hidden_dim'] = int(fc_match.group(1))
        config['num_fc_layers'] = int(fc_match.group(2))
        config['fc_str'] = f"{fc_match.group(1)}x{fc_match.group(2)}"

    # Extract process and corner
    for process in PROCESSES:
        if f'_{process}_' in name:
            config['process'] = process
            break

    for corner in CORNERS:
        if f'_{corner}_' in name:
            config['corner'] = corner
            break

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

    # Create process-corner label
    if 'process' in config and 'corner' in config:
        config['process_corner'] = f"{config['process']}_{config['corner']}"

    return config


def compute_metrics(predictions, actuals):
    """Compute various error metrics."""
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Filter out invalid values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) | np.isinf(predictions) | np.isinf(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    if len(predictions) == 0:
        return None

    # MSE and RMSE
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)

    # NRMSE (normalized by mean)
    mean_actual = np.mean(np.abs(actuals))
    nrmse = (rmse / (mean_actual + 1e-10)) * 100

    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    # SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = np.mean(2 * np.abs(predictions - actuals) / (np.abs(predictions) + np.abs(actuals) + 1e-10)) * 100

    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    return {
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'mae': mae,
        'smape': smape,
        'r2': r2,
        'num_samples': len(predictions)
    }


def load_results(results_dir):
    """Load all prediction/actual pairs from results directory."""
    results = []

    pred_files = glob.glob(os.path.join(results_dir, '*_pred.npy'))

    for pred_file in pred_files:
        act_file = pred_file.replace('_pred.npy', '_act.npy')

        if not os.path.exists(act_file):
            continue

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)

            config = parse_filename(pred_file)
            metrics = compute_metrics(predictions, actuals)

            if metrics is None:
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


def filter_by_architecture(results, mode, model_type, graph_mode, conv_str, fc_str):
    """Filter results to match specific architecture."""
    filtered = []

    for r in results:
        config = r['config']

        # Check all architecture parameters match
        if config.get('mode') != mode:
            continue
        if config.get('model_type') != model_type:
            continue
        if config.get('graph_mode') != graph_mode:
            continue
        if config.get('conv_str') != conv_str:
            continue
        if config.get('fc_str') != fc_str:
            continue

        filtered.append(r)

    return filtered


def plot_process_corner_comparison(results, output_dir, arch_name):
    """Create bar charts comparing metrics across process-corners."""
    if not results:
        print("No results to plot")
        return

    # Sort by process then corner
    process_order = {p: i for i, p in enumerate(PROCESSES)}
    corner_order = {c: i for i, c in enumerate(CORNERS)}
    results = sorted(results, key=lambda x: (
        process_order.get(x['config'].get('process', ''), 99),
        corner_order.get(x['config'].get('corner', ''), 99)
    ))

    labels = [r['config'].get('process_corner', 'unknown') for r in results]

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_info = [
        ('nrmse', 'NRMSE (%)', axes[0, 0]),
        ('smape', 'SMAPE (%)', axes[0, 1]),
        ('mae', 'MAE (ns)', axes[1, 0]),
        ('r2', 'R²', axes[1, 1])
    ]

    # Color by process
    color_map = {'LVT': 'tab:blue', 'RVT': 'tab:orange', 'SLVT': 'tab:green', 'SRAM': 'tab:red'}
    colors = [color_map.get(r['config'].get('process', ''), 'tab:gray') for r in results]

    for metric_name, ylabel, ax in metrics_info:
        values = [r['metrics'][metric_name] for r in results]

        bars = ax.bar(range(len(labels)), values, color=colors)

        ax.set_xlabel('Process_Corner', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{ylabel} by Process-Corner', fontsize=12)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric_name == 'r2':
                text = f'{val:.4f}'
            elif metric_name == 'mae':
                text = f'{val:.4f}'
            else:
                text = f'{val:.2f}'
            ax.annotate(text,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

        ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=p) for p, c in color_map.items()]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))

    plt.suptitle(f'Architecture: {arch_name}\nPerformance Across Process-Corner Combinations', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'process_corner_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_heatmap_by_process_corner(results, output_dir, arch_name):
    """Create heatmap showing metrics for each process-corner."""
    if not results:
        return

    metrics_to_show = ['nrmse', 'smape', 'mae', 'r2']

    # Build matrix: rows = process-corner, cols = metrics
    # Sort by process then corner
    process_order = {p: i for i, p in enumerate(PROCESSES)}
    corner_order = {c: i for i, c in enumerate(CORNERS)}
    results = sorted(results, key=lambda x: (
        process_order.get(x['config'].get('process', ''), 99),
        corner_order.get(x['config'].get('corner', ''), 99)
    ))

    labels = [r['config'].get('process_corner', 'unknown') for r in results]

    data = np.zeros((len(results), len(metrics_to_show)))
    for i, r in enumerate(results):
        for j, m in enumerate(metrics_to_show):
            data[i, j] = r['metrics'][m]

    # Normalize for visualization
    data_normalized = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        if col.max() - col.min() > 1e-10:
            if metrics_to_show[j] == 'r2':
                data_normalized[:, j] = 1 - (col - col.min()) / (col.max() - col.min())
            else:
                data_normalized[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            data_normalized[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.5)))

    im = ax.imshow(data_normalized, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(metrics_to_show)))
    ax.set_xticklabels([m.upper() for m in metrics_to_show], fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    # Add text annotations
    for i in range(len(results)):
        for j in range(len(metrics_to_show)):
            val = data[i, j]
            if metrics_to_show[j] == 'r2':
                text = f'{val:.4f}'
            elif metrics_to_show[j] == 'mae':
                text = f'{val:.4f}'
            else:
                text = f'{val:.2f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                   color='white' if data_normalized[i, j] > 0.5 else 'black')

    ax.set_title(f'Architecture: {arch_name}\nMetrics by Process-Corner (Green=Better, Red=Worse)', fontsize=12)
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Process_Corner', fontsize=11)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'process_corner_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_results_table(results, arch_name):
    """Print results as a formatted table."""
    if not results:
        print("No results to display")
        return

    # Sort by process then corner
    process_order = {p: i for i, p in enumerate(PROCESSES)}
    corner_order = {c: i for i, c in enumerate(CORNERS)}
    results = sorted(results, key=lambda x: (
        process_order.get(x['config'].get('process', ''), 99),
        corner_order.get(x['config'].get('corner', ''), 99)
    ))

    print("\n" + "=" * 90)
    print(f"ARCHITECTURE: {arch_name}")
    print("=" * 90)
    print(f"{'Process_Corner':<15} {'NRMSE%':<10} {'SMAPE%':<10} {'MAE(ns)':<12} {'R²':<10} {'Samples':<10}")
    print("-" * 90)

    nrmse_vals = []
    smape_vals = []
    mae_vals = []
    r2_vals = []

    for r in results:
        pc = r['config'].get('process_corner', 'unknown')
        nrmse = r['metrics']['nrmse']
        smape = r['metrics']['smape']
        mae = r['metrics']['mae']
        r2 = r['metrics']['r2']
        samples = r['metrics']['num_samples']

        nrmse_vals.append(nrmse)
        smape_vals.append(smape)
        mae_vals.append(mae)
        r2_vals.append(r2)

        print(f"{pc:<15} {nrmse:<10.2f} {smape:<10.2f} {mae:<12.6f} {r2:<10.4f} {samples:<10}")

    print("-" * 90)

    # Summary statistics
    if nrmse_vals:
        print(f"{'Mean':<15} {np.mean(nrmse_vals):<10.2f} {np.mean(smape_vals):<10.2f} {np.mean(mae_vals):<12.6f} {np.mean(r2_vals):<10.4f}")
        print(f"{'Std':<15} {np.std(nrmse_vals):<10.2f} {np.std(smape_vals):<10.2f} {np.std(mae_vals):<12.6f} {np.std(r2_vals):<10.4f}")
        print(f"{'Min':<15} {np.min(nrmse_vals):<10.2f} {np.min(smape_vals):<10.2f} {np.min(mae_vals):<12.6f} {np.min(r2_vals):<10.4f}")
        print(f"{'Max':<15} {np.max(nrmse_vals):<10.2f} {np.max(smape_vals):<10.2f} {np.max(mae_vals):<12.6f} {np.max(r2_vals):<10.4f}")

    print("=" * 90)

    # Best and worst process-corner
    if results:
        best_idx = np.argmin(nrmse_vals)
        worst_idx = np.argmax(nrmse_vals)
        print(f"\nBest Process-Corner (lowest NRMSE): {results[best_idx]['config'].get('process_corner')} ({nrmse_vals[best_idx]:.2f}%)")
        print(f"Worst Process-Corner (highest NRMSE): {results[worst_idx]['config'].get('process_corner')} ({nrmse_vals[worst_idx]:.2f}%)")

    # Missing process-corners
    found_pcs = set(r['config'].get('process_corner') for r in results)
    all_pcs = set(f"{p}_{c}" for p in PROCESSES for c in CORNERS)
    missing = all_pcs - found_pcs
    if missing:
        print(f"\nMissing Process-Corners ({len(missing)}): {', '.join(sorted(missing))}")


def list_available_architectures(results):
    """List all unique architectures found in results."""
    architectures = set()

    for r in results:
        config = r['config']
        mode = config.get('mode', '?')
        model_type = config.get('model_type', '?')
        graph_mode = config.get('graph_mode', '?')
        conv_str = config.get('conv_str', '?')
        fc_str = config.get('fc_str', '?')

        arch = (mode, model_type, graph_mode, conv_str, fc_str)
        architectures.add(arch)

    return sorted(architectures)


def main():
    parser = argparse.ArgumentParser(
        description='Compare a single architecture across all process-corner combinations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available architectures
    python compare_architecture_across_process.py --list

    # Analyze specific architecture
    python compare_architecture_across_process.py --mode interpolation --model_type baseline --graph_mode full_graph --conv 64x2 --fc 128x2
    python compare_architecture_across_process.py --mode interpolation --model_type maml --graph_mode stage_aware --conv 64x3 --fc 64x2
"""
    )

    parser.add_argument('--results_dir', type=str, default='',
                       help='Subdirectory under data_result_npy_directory')
    parser.add_argument('--output_dir', type=str,
                       default='/home/tkdgn2907/Deepsets_test/MAML/Projects/result_management/result_summary/gcn_validation_analysis',
                       help='Output directory for plots')
    parser.add_argument('--list', action='store_true',
                       help='List all available architectures and exit')

    # Architecture specification
    parser.add_argument('--mode', type=str, choices=['interpolation', 'extrapolation'],
                       help='Validation mode')
    parser.add_argument('--model_type', type=str, choices=['baseline', 'maml'],
                       help='Model type')
    parser.add_argument('--graph_mode', type=str, choices=['full_graph', 'stage_aware'],
                       help='Graph mode')
    parser.add_argument('--conv', type=str,
                       help='Conv architecture (e.g., 64x2 for dim=64, layers=2)')
    parser.add_argument('--fc', type=str,
                       help='FC architecture (e.g., 128x2 for dim=128, layers=2)')

    parser.add_argument('--no_plots', action='store_true',
                       help='Only print table, skip generating plots')

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
        print("No valid results found")
        return 1

    print(f"Loaded {len(results)} total result files")

    # List mode
    if args.list:
        architectures = list_available_architectures(results)
        print("\n" + "=" * 80)
        print("AVAILABLE ARCHITECTURES")
        print("=" * 80)
        print(f"{'Mode':<15} {'Model':<10} {'Graph':<12} {'Conv':<10} {'FC':<10}")
        print("-" * 80)
        for arch in architectures:
            print(f"{arch[0]:<15} {arch[1]:<10} {arch[2]:<12} {arch[3]:<10} {arch[4]:<10}")
        print("=" * 80)
        print(f"\nTotal: {len(architectures)} unique architectures")
        return 0

    # Validate required arguments
    if not all([args.mode, args.model_type, args.graph_mode, args.conv, args.fc]):
        print("Error: Must specify --mode, --model_type, --graph_mode, --conv, and --fc")
        print("Use --list to see available architectures")
        return 1

    # Filter results
    filtered = filter_by_architecture(
        results, args.mode, args.model_type, args.graph_mode, args.conv, args.fc
    )

    if not filtered:
        print(f"No results found for specified architecture")
        print("Use --list to see available architectures")
        return 1

    print(f"Found {len(filtered)} results for specified architecture")

    # Create architecture name
    graph_short = 'SA' if args.graph_mode == 'stage_aware' else 'FG'
    model_short = 'B' if args.model_type == 'baseline' else 'M'
    arch_name = f"{model_short}_{graph_short}_conv{args.conv}_fc{args.fc}"

    # Print table
    print_results_table(filtered, arch_name)

    # Generate plots
    if not args.no_plots:
        # Create output subdirectory
        subdir_name = f"{args.mode}_{arch_name}"
        output_dir = os.path.join(args.output_dir, subdir_name)
        print(f"\nGenerating plots to: {output_dir}")

        plot_process_corner_comparison(filtered, output_dir, arch_name)
        plot_heatmap_by_process_corner(filtered, output_dir, arch_name)

        print(f"\nAll plots saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
