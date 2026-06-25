#!/usr/bin/env python
"""
Compare and visualize GNN validation sweep results.

Loads .npy prediction/actual files from validation sweeps and creates
comparison plots for different architectures.

Location: Projects/result_management/compare_validation_results.py

Usage:
    python compare_validation_results.py --results_dir ../pretraining/model_test_code/gnn/data_result_npy_directory
    python compare_validation_results.py --results_dir ../pretraining/model_test_code/gnn/data_result_npy_directory/10000samples
"""

import os
import sys
import numpy as np
import argparse
import glob
import re
import matplotlib.pyplot as plt


# MLP MAML baseline directory (ASAP7)
MLP_MAML_BASELINE_DIR = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/data_result_npy_directory'


def load_mlp_maml_baseline(process, corner, data_type='cell', group_size=61):
    """
    Load MLP MAML baseline results for ASAP7 comparison.

    Args:
        process: LVT, SLVT, RVT, SRAM
        corner: FF, SS, TT
        data_type: cell or transition
        group_size: number of samples per group (default: 61)

    Returns:
        dict: metrics or None if not found
    """
    process_lower = process.lower()
    corner_upper = corner.upper()

    # Find MAML files matching the process and corner
    # Pattern: ASAP7_voltage_variation_{process}_{corner}_{data_type}_maml_*_pred.npy
    pattern = f"ASAP7_voltage_variation_{process_lower}_{corner_upper}_{data_type}_maml_*_pred.npy"
    pred_files = glob.glob(os.path.join(MLP_MAML_BASELINE_DIR, pattern))

    if not pred_files:
        # Try lowercase corner as well
        pattern = f"ASAP7_voltage_variation_{process_lower}_{corner_upper.lower()}_{data_type}_maml_*_pred.npy"
        pred_files = glob.glob(os.path.join(MLP_MAML_BASELINE_DIR, pattern))

    if not pred_files:
        print(f"  Warning: No MLP MAML baseline found for {process}_{corner}")
        return None

    # Use the first matching file
    pred_file = pred_files[0]
    act_file = pred_file.replace('_pred.npy', '_act.npy')

    if not os.path.exists(act_file):
        print(f"  Warning: Missing actual file for MLP MAML baseline")
        return None

    try:
        predictions = np.load(pred_file).flatten()
        actuals = np.load(act_file).flatten()

        # Calculate metrics using 61-group methodology (same as GNN)
        n_groups = len(predictions) // group_size
        if n_groups == 0:
            n_groups = 1
            group_size = len(predictions)

        predictions = predictions[:n_groups * group_size]
        actuals = actuals[:n_groups * group_size]

        pred_grouped = predictions.reshape(n_groups, group_size)
        act_grouped = actuals.reshape(n_groups, group_size)

        group_metrics = []
        for i in range(n_groups):
            y_p = pred_grouped[i]
            y_t = act_grouped[i]

            mae = np.mean(np.abs(y_p - y_t))

            mask = y_t != 0
            mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100 if np.any(mask) else 0

            denom = np.abs(y_t) + np.abs(y_p)
            mask_smape = denom != 0
            smape = np.mean(2.0 * np.abs(y_t[mask_smape] - y_p[mask_smape]) / denom[mask_smape]) * 100 if np.any(mask_smape) else 0

            rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
            y_range = np.max(y_t) - np.min(y_t)
            nrmse = (rmse / y_range * 100) if y_range > 0 else 0

            group_metrics.append({'mae': mae, 'mape': mape, 'smape': smape, 'nrmse': nrmse})

        baseline_metrics = {
            'mae': np.mean([g['mae'] for g in group_metrics]),
            'mape': np.mean([g['mape'] for g in group_metrics]),
            'smape': np.mean([g['smape'] for g in group_metrics]),
            'nrmse': np.mean([g['nrmse'] for g in group_metrics]),
            'source_file': os.path.basename(pred_file)
        }

        print(f"  Loaded MLP MAML baseline: {os.path.basename(pred_file)}")
        print(f"    NRMSE: {baseline_metrics['nrmse']:.4f}%, SMAPE: {baseline_metrics['smape']:.4f}%, MAE: {baseline_metrics['mae']*1000:.4f}")

        return baseline_metrics

    except Exception as e:
        print(f"  Warning: Failed to load MLP MAML baseline: {e}")
        return None


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

    if fc_match:
        config['fc_hidden_dim'] = int(fc_match.group(1))
        config['num_fc_layers'] = int(fc_match.group(2))

    # Extract process and corner
    for process in ['LVT', 'RVT', 'SLVT', 'SRAM']:
        if f'_{process}_' in name:
            config['process'] = process
            break

    for corner in ['TT', 'FF', 'SS']:
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

    # Create architecture label with model_type and graph_mode
    model_type = config.get('model_type', 'unknown')
    graph_mode = config.get('graph_mode', 'unknown')

    # Short graph mode label
    graph_short = 'SA' if graph_mode == 'stage_aware' else 'FG' if graph_mode == 'full_graph' else '?'
    model_short = 'B' if model_type == 'baseline' else 'M' if model_type == 'maml' else '?'

    if 'conv_hidden_dim' in config and 'fc_hidden_dim' in config:
        arch_params = f"c{config['conv_hidden_dim']}x{config['num_conv_layers']}_f{config['fc_hidden_dim']}x{config['num_fc_layers']}"
        config['arch_label'] = f"{model_short}_{graph_short}_{arch_params}"
    else:
        config['arch_label'] = name

    # Full label for detailed display
    config['full_label'] = f"{model_type}_{graph_mode}_conv{config.get('conv_hidden_dim', '?')}x{config.get('num_conv_layers', '?')}_fc{config.get('fc_hidden_dim', '?')}x{config.get('num_fc_layers', '?')}"

    return config


def compute_metrics(predictions, actuals, group_size=61):
    """
    Compute various error metrics with 61-group averaging.
    Same methodology as compare_tsmc_aadam_maml.py.

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

    # Group by 61 samples (same as compare_tsmc_aadam_maml.py)
    n_groups = len(predictions) // group_size

    if n_groups == 0:
        # Fallback to global calculation if not enough samples
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

        # MAPE (with masking for zero values - same as compare_tsmc_aadam_maml.py)
        mask = y_t != 0
        mape = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100 if np.any(mask) else 0

        # SMAPE (with masking - same as compare_tsmc_aadam_maml.py)
        denom = np.abs(y_t) + np.abs(y_p)
        mask_smape = denom != 0
        smape = np.mean(
            2.0 * np.abs(y_t[mask_smape] - y_p[mask_smape]) / denom[mask_smape]
        ) * 100 if np.any(mask_smape) else 0

        # RMSE
        mse = np.mean((y_p - y_t) ** 2)
        rmse = np.sqrt(mse)

        # NRMSE (range normalization - same as compare_tsmc_aadam_maml.py)
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


def plot_metrics_comparison(results, output_dir, metric_name='nrmse', mlp_maml_baseline=None, crop_to_baseline=False):
    """Create bar chart comparing a metric across architectures.

    Args:
        crop_to_baseline: If True, set y-axis max to baseline value (show only below baseline)
    """
    if not results:
        print("No results to plot")
        return

    # Sort by architecture label for full plot
    results_sorted_by_label = sorted(results, key=lambda x: x['config'].get('arch_label', ''))

    # Sort by metric value for top5 plot
    results_sorted_by_metric = sorted(results, key=lambda x: x['metrics'][metric_name])

    # Separate by graph mode
    results_fg = [r for r in results if r['config'].get('graph_mode') == 'full_graph']
    results_sa = [r for r in results if r['config'].get('graph_mode') == 'stage_aware']
    results_fg_sorted = sorted(results_fg, key=lambda x: x['metrics'][metric_name])
    results_sa_sorted = sorted(results_sa, key=lambda x: x['metrics'][metric_name])

    os.makedirs(output_dir, exist_ok=True)

    # Plot full, top5, top5_FG, top5_SA versions
    for plot_type in ['full', 'top5', 'top5_FG', 'top5_SA']:
        if plot_type == 'full':
            plot_results = results_sorted_by_label
            suffix = ''
            title_suffix = ''
        elif plot_type == 'top5':
            plot_results = results_sorted_by_metric[:5]
            suffix = '_top5'
            title_suffix = ' (Top 5)'
        elif plot_type == 'top5_FG':
            plot_results = results_fg_sorted[:5]
            suffix = '_top5_FG'
            title_suffix = ' (Top 5 Full Graph)'
        else:  # top5_SA
            plot_results = results_sa_sorted[:5]
            suffix = '_top5_SA'
            title_suffix = ' (Top 5 Stage Aware)'

        if not plot_results:
            continue

        labels = [r['config'].get('arch_label', 'unknown') for r in plot_results]
        values = [r['metrics'][metric_name] for r in plot_results]

        # MAE는 x1000 스케일링
        if metric_name == 'mae':
            values = [v * 1000 for v in values]

        # Create figure
        if plot_type == 'full':
            _fig, ax = plt.subplots(figsize=(14, 6))
        else:  # top5 variants
            _fig, ax = plt.subplots(figsize=(10, 6))

        # Color by fc_hidden_dim
        colors = []
        color_map = {40: 'tab:purple', 64: 'tab:blue', 128: 'tab:orange', 256: 'tab:green', 512: 'tab:red'}
        for r in plot_results:
            fc_dim = r['config'].get('fc_hidden_dim', 64)
            colors.append(color_map.get(fc_dim, 'tab:gray'))

        bars = ax.bar(range(len(labels)), values, color=colors)

        ax.set_xlabel('Architecture', fontsize=12)
        if metric_name == 'mae':
            ax.set_ylabel('MAE (x1000)', fontsize=12)
            ax.set_title(f'MAE (x1000) Comparison Across Architectures (ASAP7){title_suffix}', fontsize=14)
        else:
            ax.set_ylabel(metric_name.upper(), fontsize=12)
            ax.set_title(f'{metric_name.upper()} Comparison Across Architectures (ASAP7){title_suffix}', fontsize=14)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9 if plot_type == 'full' else 10)

        # Add MLP MAML baseline horizontal line
        baseline_value = None
        if mlp_maml_baseline is not None and metric_name in mlp_maml_baseline:
            baseline_value = mlp_maml_baseline[metric_name]
            if metric_name == 'mae':
                baseline_value = baseline_value * 1000
            ax.axhline(y=baseline_value, color='red', linestyle='--', linewidth=2)

            # Crop y-axis to show only values below baseline (fit to data range)
            if crop_to_baseline:
                y_min = min(values) if values else 0
                y_margin = (baseline_value - y_min) * 0.1  # 10% margin
                ax.set_ylim(max(0, y_min - y_margin), baseline_value * 1.02)

        # Add value labels on bars
        if plot_type == 'full':
            # Only for top 5 best results in full plot
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            top5_indices = set(sorted_indices[:5])
            for idx, (bar, val) in enumerate(zip(bars, values)):
                if idx in top5_indices:
                    height = bar.get_height()
                    ax.annotate(f'{val:.4f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        else:
            # All bars in top5 plot
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        # Add legend for colors (combine with baseline if exists)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [Patch(facecolor=c, label=f'fc_dim={k}') for k, c in sorted(color_map.items())]
        if mlp_maml_baseline is not None and metric_name in mlp_maml_baseline:
            baseline_value = mlp_maml_baseline[metric_name]
            if metric_name == 'mae':
                baseline_value = baseline_value * 1000
            legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'MLP MAML: {baseline_value:.4f}'))
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        crop_suffix = '_cropped' if crop_to_baseline else ''
        output_path = os.path.join(output_dir, f'{metric_name}_comparison{suffix}{crop_suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def plot_all_metrics_heatmap(results, output_dir):
    """Create heatmap of all metrics for all architectures."""
    if not results:
        return

    metrics_to_show = ['nrmse', 'smape', 'mae', 'r2']

    # Sort results
    results = sorted(results, key=lambda x: x['config'].get('arch_label', ''))

    labels = [r['config'].get('arch_label', 'unknown') for r in results]

    # Build data matrix
    data = np.zeros((len(results), len(metrics_to_show)))
    for i, r in enumerate(results):
        for j, m in enumerate(metrics_to_show):
            data[i, j] = r['metrics'][m]

    # Normalize each column for visualization
    data_normalized = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        if metrics_to_show[j] == 'r2':
            # For R2, higher is better, so invert
            data_normalized[:, j] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-10)
        else:
            # For errors, lower is better
            data_normalized[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)

    _fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.4)))

    _im = ax.imshow(data_normalized, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(metrics_to_show)))
    ax.set_xticklabels([m.upper() for m in metrics_to_show], fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Add text annotations with actual values
    for i in range(len(results)):
        for j in range(len(metrics_to_show)):
            val = data[i, j]
            if metrics_to_show[j] == 'r2':
                text = f'{val:.4f}'
            elif metrics_to_show[j] == 'mae':
                text = f'{val:.4f}'  # Already in ns (dataset unit)
            else:
                text = f'{val:.2f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                   color='white' if data_normalized[i, j] > 0.5 else 'black')

    ax.set_title('Metrics Comparison Heatmap\n(Green=Better, Red=Worse)', fontsize=12)
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Architecture', fontsize=11)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_architecture_comparison_by_params(results, output_dir):
    """Plot metrics vs architecture parameters."""
    if not results:
        return

    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract data
    conv_dims = [r['config'].get('conv_hidden_dim', 0) for r in results]
    fc_dims = [r['config'].get('fc_hidden_dim', 0) for r in results]
    _num_conv = [r['config'].get('num_conv_layers', 0) for r in results]
    nrmse_vals = [r['metrics']['nrmse'] for r in results]
    smape_vals = [r['metrics']['smape'] for r in results]

    # Plot 1: NRMSE vs conv_hidden_dim
    ax = axes[0, 0]
    for fc in sorted(set(fc_dims)):
        mask = [f == fc for f in fc_dims]
        x = [c for c, m in zip(conv_dims, mask) if m]
        y = [n for n, m in zip(nrmse_vals, mask) if m]
        if x:
            ax.scatter(x, y, label=f'fc_dim={fc}', s=50)
    ax.set_xlabel('Conv Hidden Dim')
    ax.set_ylabel('NRMSE (%)')
    ax.set_title('NRMSE vs Conv Hidden Dim')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: NRMSE vs fc_hidden_dim
    ax = axes[0, 1]
    for conv in sorted(set(conv_dims)):
        mask = [c == conv for c in conv_dims]
        x = [f for f, m in zip(fc_dims, mask) if m]
        y = [n for n, m in zip(nrmse_vals, mask) if m]
        if x:
            ax.scatter(x, y, label=f'conv_dim={conv}', s=50)
    ax.set_xlabel('FC Hidden Dim')
    ax.set_ylabel('NRMSE (%)')
    ax.set_title('NRMSE vs FC Hidden Dim')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: SMAPE vs conv_hidden_dim
    ax = axes[1, 0]
    for fc in sorted(set(fc_dims)):
        mask = [f == fc for f in fc_dims]
        x = [c for c, m in zip(conv_dims, mask) if m]
        y = [n for n, m in zip(smape_vals, mask) if m]
        if x:
            ax.scatter(x, y, label=f'fc_dim={fc}', s=50)
    ax.set_xlabel('Conv Hidden Dim')
    ax.set_ylabel('SMAPE (%)')
    ax.set_title('SMAPE vs Conv Hidden Dim')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: SMAPE vs fc_hidden_dim
    ax = axes[1, 1]
    for conv in sorted(set(conv_dims)):
        mask = [c == conv for c in conv_dims]
        x = [f for f, m in zip(fc_dims, mask) if m]
        y = [n for n, m in zip(smape_vals, mask) if m]
        if x:
            ax.scatter(x, y, label=f'conv_dim={conv}', s=50)
    ax.set_xlabel('FC Hidden Dim')
    ax.set_ylabel('SMAPE (%)')
    ax.set_title('SMAPE vs FC Hidden Dim')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Metrics vs Architecture Parameters', fontsize=14)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'architecture_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_parameter_trend_analysis(results, output_dir, mlp_maml_baseline=None, crop_to_baseline=False):
    """
    Plot trend analysis by fixing one parameter and varying another.
    Separated by FG/SA and layer configurations.
    Only includes MAML models (excludes baseline).
    Optionally includes MLP MAML baseline as horizontal dashed line.

    Args:
        crop_to_baseline: If True, set y-axis max to baseline value (show only below baseline)
    """
    if not results:
        return

    # Filter to only include MAML models
    results = [r for r in results if r['config'].get('model_type') == 'maml']
    if not results:
        print("  No MAML results found for trend analysis")
        return

    os.makedirs(output_dir, exist_ok=True)
    metrics = ['nrmse', 'smape', 'mae']
    metric_labels = {'nrmse': 'NRMSE (%)', 'smape': 'SMAPE (%)', 'mae': 'MAE (x1000)'}

    # Prepare baseline values for plotting
    baseline_values = {}
    if mlp_maml_baseline is not None:
        for metric in metrics:
            if metric in mlp_maml_baseline:
                val = mlp_maml_baseline[metric]
                if metric == 'mae':
                    val = val * 1000  # Scale MAE
                baseline_values[metric] = val

    # Extract unique parameter values
    conv_dims = sorted(set(r['config'].get('conv_hidden_dim', 0) for r in results))
    fc_dims = sorted(set(r['config'].get('fc_hidden_dim', 0) for r in results))
    num_conv_layers_list = sorted(set(r['config'].get('num_conv_layers', 0) for r in results))
    num_fc_layers_list = sorted(set(r['config'].get('num_fc_layers', 0) for r in results))

    print(f"  Parameter values found:")
    print(f"    conv_hidden_dim: {conv_dims}")
    print(f"    fc_hidden_dim: {fc_dims}")
    print(f"    num_conv_layers: {num_conv_layers_list}")
    print(f"    num_fc_layers: {num_fc_layers_list}")

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    markers = ['o', 's', '^', 'D', 'v', '<']

    graph_mode_configs = [
        ('full_graph', 'FG'),
        ('stage_aware', 'SA')
    ]
    conv_filter_configs = [
        ('full', conv_dims, '')
    ]

    # Whether to show values at each point
    annotation_configs = [
        (True, ''),           # with annotations
        (False, '_no_values')  # without annotations
    ]

    layer_configs = sorted(set(
        (r['config'].get('num_conv_layers', 0), r['config'].get('num_fc_layers', 0))
        for r in results
    ))

    # Plot 1: Fix conv_hidden_dim, vary fc_hidden_dim
    for graph_mode, mode_label in graph_mode_configs:
        mode_results = [r for r in results if r['config'].get('graph_mode') == graph_mode]
        if not mode_results:
            continue

        for n_conv, n_fc in layer_configs:
            layer_results = [r for r in mode_results
                            if r['config'].get('num_conv_layers') == n_conv
                            and r['config'].get('num_fc_layers') == n_fc]
            if not layer_results:
                continue

            layer_suffix = f'_L{n_conv}x{n_fc}'

            for _version, conv_dims_to_plot, conv_suffix in conv_filter_configs:
                if not conv_dims_to_plot:
                    continue

                for show_annotations, annot_suffix in annotation_configs:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    title_suffix = f' [{mode_label}] (layers:{n_conv}conv,{n_fc}fc)'
                    fig.suptitle(f'Trend Analysis: Fix conv_hidden_dim, Vary fc_hidden_dim{title_suffix} (ASAP7)', fontsize=14, fontweight='bold')

                    for idx, metric in enumerate(metrics):
                        ax = axes[idx]

                        for i, conv_dim in enumerate(conv_dims_to_plot):
                            filtered = [r for r in layer_results if r['config'].get('conv_hidden_dim') == conv_dim]
                            if not filtered:
                                continue

                            fc_metric_data = {}
                            for r in filtered:
                                fc_dim = r['config'].get('fc_hidden_dim')
                                if fc_dim not in fc_metric_data:
                                    fc_metric_data[fc_dim] = []
                                val = r['metrics'][metric]
                                if metric == 'mae':
                                    val = val * 1000
                                fc_metric_data[fc_dim].append(val)

                            x_vals_raw = sorted(fc_metric_data.keys())
                            y_vals = [np.mean(fc_metric_data[fc]) for fc in x_vals_raw]

                            # Use indices for equal spacing
                            x_indices = [fc_dims.index(fc) for fc in x_vals_raw]

                            color = colors[i % len(colors)]
                            marker = markers[i % len(markers)]
                            ax.plot(x_indices, y_vals, label=f'conv_dim={conv_dim}',
                                   color=color, marker=marker, markersize=8, linewidth=2)

                            if show_annotations:
                                for x, y in zip(x_indices, y_vals):
                                    ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 8),
                                               textcoords='offset points', ha='center', fontsize=8)

                        # Add MLP MAML baseline horizontal line
                        if metric in baseline_values:
                            ax.axhline(y=baseline_values[metric], color='red', linestyle='--', linewidth=2, label='MLP MAML')
                            if crop_to_baseline:
                                y_min, _ = ax.get_ylim()
                                y_margin = (baseline_values[metric] - y_min) * 0.1
                                ax.set_ylim(max(0, y_min - y_margin), baseline_values[metric] * 1.02)

                        ax.set_xlabel('fc_hidden_dim', fontsize=11, fontweight='bold')
                        ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
                        ax.set_title(f'{metric_labels[metric]} vs fc_hidden_dim', fontsize=12)
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        ax.set_xticks(range(len(fc_dims)))
                        ax.set_xticklabels(fc_dims)

                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f'trend_fix_conv_vary_fc_{mode_label}{layer_suffix}{conv_suffix}{annot_suffix}.png')
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"Saved: {output_path}")
                    plt.close()

    # Plot 2: Fix fc_hidden_dim, vary conv_hidden_dim
    for graph_mode, mode_label in graph_mode_configs:
        mode_results = [r for r in results if r['config'].get('graph_mode') == graph_mode]
        if not mode_results:
            continue

        for n_conv, n_fc in layer_configs:
            layer_results = [r for r in mode_results
                            if r['config'].get('num_conv_layers') == n_conv
                            and r['config'].get('num_fc_layers') == n_fc]
            if not layer_results:
                continue

            layer_suffix = f'_L{n_conv}x{n_fc}'

            for _version, conv_dims_to_include, conv_suffix in conv_filter_configs:
                if not conv_dims_to_include:
                    continue

                for show_annotations, annot_suffix in annotation_configs:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    title_suffix = f' [{mode_label}] (layers:{n_conv}conv,{n_fc}fc)'
                    fig.suptitle(f'Trend Analysis: Fix fc_hidden_dim, Vary conv_hidden_dim{title_suffix} (ASAP7)', fontsize=14, fontweight='bold')

                    for idx, metric in enumerate(metrics):
                        ax = axes[idx]

                        for i, fc_dim in enumerate(fc_dims):
                            filtered = [r for r in layer_results
                                       if r['config'].get('fc_hidden_dim') == fc_dim
                                       and r['config'].get('conv_hidden_dim') in conv_dims_to_include]
                            if not filtered:
                                continue

                            conv_metric_data = {}
                            for r in filtered:
                                conv_dim = r['config'].get('conv_hidden_dim')
                                if conv_dim not in conv_metric_data:
                                    conv_metric_data[conv_dim] = []
                                val = r['metrics'][metric]
                                if metric == 'mae':
                                    val = val * 1000
                                conv_metric_data[conv_dim].append(val)

                            x_vals_raw = sorted(conv_metric_data.keys())
                            y_vals = [np.mean(conv_metric_data[conv]) for conv in x_vals_raw]

                            # Use indices for equal spacing
                            x_indices = [conv_dims_to_include.index(conv) for conv in x_vals_raw if conv in conv_dims_to_include]

                            color = colors[i % len(colors)]
                            marker = markers[i % len(markers)]
                            ax.plot(x_indices, y_vals, label=f'fc_dim={fc_dim}',
                                   color=color, marker=marker, markersize=8, linewidth=2)

                            if show_annotations:
                                for x, y in zip(x_indices, y_vals):
                                    ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 8),
                                               textcoords='offset points', ha='center', fontsize=8)

                        # Add MLP MAML baseline horizontal line
                        if metric in baseline_values:
                            ax.axhline(y=baseline_values[metric], color='red', linestyle='--', linewidth=2, label='MLP MAML')
                            if crop_to_baseline:
                                y_min, _ = ax.get_ylim()
                                y_margin = (baseline_values[metric] - y_min) * 0.1
                                ax.set_ylim(max(0, y_min - y_margin), baseline_values[metric] * 1.02)

                        ax.set_xlabel('conv_hidden_dim', fontsize=11, fontweight='bold')
                        ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
                        ax.set_title(f'{metric_labels[metric]} vs conv_hidden_dim', fontsize=12)
                        ax.legend(loc='best', fontsize=9)
                        ax.grid(True, alpha=0.3)
                        ax.set_xticks(range(len(conv_dims_to_include)))
                        ax.set_xticklabels(conv_dims_to_include)

                    plt.tight_layout()
                    output_path = os.path.join(output_dir, f'trend_fix_fc_vary_conv_{mode_label}{layer_suffix}{conv_suffix}{annot_suffix}.png')
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"Saved: {output_path}")
                    plt.close()

    # Plot 2.5: Compare layer configurations
    layer_colors = {(2, 2): 'tab:blue', (2, 3): 'tab:orange', (3, 2): 'tab:green', (3, 3): 'tab:red'}
    layer_markers = {(2, 2): 'o', (2, 3): 's', (3, 2): '^', (3, 3): 'D'}

    for graph_mode, mode_label in graph_mode_configs:
        mode_results = [r for r in results if r['config'].get('graph_mode') == graph_mode]
        if not mode_results:
            continue

        for _version, conv_dims_to_plot, conv_suffix in conv_filter_configs:
            if not conv_dims_to_plot:
                continue

            title_suffix = f' [{mode_label}]'

            for show_annotations, annot_suffix in annotation_configs:
                # Layer compare vary fc
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f'Layer Config Comparison: Vary fc_dim{title_suffix} (ASAP7)', fontsize=14, fontweight='bold')

                for idx, metric in enumerate(metrics):
                    ax = axes[idx]
                    for (n_conv, n_fc), color in layer_colors.items():
                        layer_results = [r for r in mode_results
                                        if r['config'].get('num_conv_layers') == n_conv
                                        and r['config'].get('num_fc_layers') == n_fc]
                        if not layer_results:
                            continue

                        fc_metric_data = {}
                        for r in layer_results:
                            if r['config'].get('conv_hidden_dim') not in conv_dims_to_plot:
                                continue
                            fc_dim = r['config'].get('fc_hidden_dim')
                            if fc_dim not in fc_metric_data:
                                fc_metric_data[fc_dim] = []
                            val = r['metrics'][metric]
                            if metric == 'mae':
                                val = val * 1000
                            fc_metric_data[fc_dim].append(val)

                        if not fc_metric_data:
                            continue

                        x_vals_raw = sorted(fc_metric_data.keys())
                        y_vals = [np.mean(fc_metric_data[fc]) for fc in x_vals_raw]

                        # Use indices for equal spacing
                        x_indices = [fc_dims.index(fc) for fc in x_vals_raw]

                        marker = layer_markers[(n_conv, n_fc)]
                        ax.plot(x_indices, y_vals, label=f'L{n_conv}x{n_fc}',
                               color=color, marker=marker, markersize=8, linewidth=2)

                        if show_annotations:
                            for x, y in zip(x_indices, y_vals):
                                ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 8),
                                           textcoords='offset points', ha='center', fontsize=8)

                    # Add MLP MAML baseline horizontal line
                    if metric in baseline_values:
                        ax.axhline(y=baseline_values[metric], color='red', linestyle='--', linewidth=2, label='MLP MAML')
                        if crop_to_baseline:
                            y_min, _ = ax.get_ylim()
                            y_margin = (baseline_values[metric] - y_min) * 0.1
                            ax.set_ylim(max(0, y_min - y_margin), baseline_values[metric] * 1.02)

                    ax.set_xlabel('fc_hidden_dim', fontsize=11, fontweight='bold')
                    ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.set_xticks(range(len(fc_dims)))
                    ax.set_xticklabels(fc_dims)

                plt.tight_layout()
                output_path = os.path.join(output_dir, f'trend_layer_compare_vary_fc_{mode_label}{conv_suffix}{annot_suffix}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {output_path}")
                plt.close()

                # Layer compare vary conv
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f'Layer Config Comparison: Vary conv_dim{title_suffix} (ASAP7)', fontsize=14, fontweight='bold')

                for idx, metric in enumerate(metrics):
                    ax = axes[idx]
                    for (n_conv, n_fc), color in layer_colors.items():
                        layer_results = [r for r in mode_results
                                        if r['config'].get('num_conv_layers') == n_conv
                                        and r['config'].get('num_fc_layers') == n_fc]
                        if not layer_results:
                            continue

                        conv_metric_data = {}
                        for r in layer_results:
                            conv_dim = r['config'].get('conv_hidden_dim')
                            if conv_dim not in conv_dims_to_plot:
                                continue
                            if conv_dim not in conv_metric_data:
                                conv_metric_data[conv_dim] = []
                            val = r['metrics'][metric]
                            if metric == 'mae':
                                val = val * 1000
                            conv_metric_data[conv_dim].append(val)

                        if not conv_metric_data:
                            continue

                        x_vals_raw = sorted(conv_metric_data.keys())
                        y_vals = [np.mean(conv_metric_data[conv]) for conv in x_vals_raw]

                        # Use indices for equal spacing
                        x_indices = [conv_dims_to_plot.index(conv) for conv in x_vals_raw if conv in conv_dims_to_plot]

                        marker = layer_markers[(n_conv, n_fc)]
                        ax.plot(x_indices, y_vals, label=f'L{n_conv}x{n_fc}',
                               color=color, marker=marker, markersize=8, linewidth=2)

                        if show_annotations:
                            for x, y in zip(x_indices, y_vals):
                                ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 8),
                                           textcoords='offset points', ha='center', fontsize=8)

                    # Add MLP MAML baseline horizontal line
                    if metric in baseline_values:
                        ax.axhline(y=baseline_values[metric], color='red', linestyle='--', linewidth=2, label='MLP MAML')
                        if crop_to_baseline:
                            y_min, _ = ax.get_ylim()
                            y_margin = (baseline_values[metric] - y_min) * 0.1
                            ax.set_ylim(max(0, y_min - y_margin), baseline_values[metric] * 1.02)

                    ax.set_xlabel('conv_hidden_dim', fontsize=11, fontweight='bold')
                    ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.set_xticks(range(len(conv_dims_to_plot)))
                    ax.set_xticklabels(conv_dims_to_plot)

                plt.tight_layout()
                output_path = os.path.join(output_dir, f'trend_layer_compare_vary_conv_{mode_label}{conv_suffix}{annot_suffix}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {output_path}")
                plt.close()

                # Best by layer config
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                fig.suptitle(f'Best Performance by Layer Configuration{title_suffix} (ASAP7)', fontsize=14, fontweight='bold')

                for idx, metric in enumerate(metrics):
                    ax = axes[idx]
                    layer_best = []
                    for (n_conv, n_fc) in [(2, 2), (2, 3), (3, 2), (3, 3)]:
                        layer_results = [r for r in mode_results
                                        if r['config'].get('num_conv_layers') == n_conv
                                        and r['config'].get('num_fc_layers') == n_fc
                                        and r['config'].get('conv_hidden_dim') in conv_dims_to_plot]
                        if not layer_results:
                            continue

                        best = min(layer_results, key=lambda x: x['metrics'][metric])
                        val = best['metrics'][metric]
                        if metric == 'mae':
                            val = val * 1000

                        layer_best.append({
                            'label': f'L{n_conv}x{n_fc}',
                            'val': val,
                            'color': layer_colors[(n_conv, n_fc)],
                            'conv': best['config'].get('conv_hidden_dim'),
                            'fc': best['config'].get('fc_hidden_dim')
                        })

                    if not layer_best:
                        continue

                    x_pos = range(len(layer_best))
                    bars = ax.bar(x_pos, [lb['val'] for lb in layer_best],
                                 color=[lb['color'] for lb in layer_best], edgecolor='black')

                    if show_annotations:
                        for i, (bar, lb) in enumerate(zip(bars, layer_best)):
                            height = bar.get_height()
                            ax.annotate(f'{lb["val"]:.3f}\n(c={lb["conv"]},f={lb["fc"]})',
                                       xy=(bar.get_x() + bar.get_width() / 2, height),
                                       xytext=(0, 3), textcoords='offset points',
                                       ha='center', va='bottom', fontsize=9)

                    # Add MLP MAML baseline horizontal line
                    if metric in baseline_values:
                        ax.axhline(y=baseline_values[metric], color='red', linestyle='--', linewidth=2, label='MLP MAML')
                        if crop_to_baseline:
                            y_min, _ = ax.get_ylim()
                            y_margin = (baseline_values[metric] - y_min) * 0.1
                            ax.set_ylim(max(0, y_min - y_margin), baseline_values[metric] * 1.02)

                    ax.set_xlabel('Layer Configuration', fontsize=11, fontweight='bold')
                    ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([lb['label'] for lb in layer_best])
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                output_path = os.path.join(output_dir, f'trend_layer_compare_best_{mode_label}{conv_suffix}{annot_suffix}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {output_path}")
                plt.close()

    # Plot 4: Best by conv_dim
    for show_annotations, annot_suffix in annotation_configs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Best Configuration by conv_hidden_dim (ASAP7)', fontsize=14, fontweight='bold')

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            best_configs = []
            for conv_dim in conv_dims:
                filtered = [r for r in results if r['config'].get('conv_hidden_dim') == conv_dim]
                if not filtered:
                    continue
                best = min(filtered, key=lambda x: x['metrics'][metric])
                val = best['metrics'][metric]
                if metric == 'mae':
                    val = val * 1000
                best_configs.append({
                    'conv_dim': conv_dim,
                    'best_fc': best['config'].get('fc_hidden_dim'),
                    'best_val': val
                })

            x_vals = [c['conv_dim'] for c in best_configs]
            y_vals = [c['best_val'] for c in best_configs]
            labels = [f"fc={c['best_fc']}" for c in best_configs]

            bars = ax.bar(range(len(x_vals)), y_vals, color=colors[:len(x_vals)], edgecolor='black')
            if show_annotations:
                for i, (bar, label, val) in enumerate(zip(bars, labels, y_vals)):
                    height = bar.get_height()
                    ax.annotate(f'{val:.3f}\n({label})',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=9)

            # Add MLP MAML baseline horizontal line
            if metric in baseline_values:
                ax.axhline(y=baseline_values[metric], color='red', linestyle='--', linewidth=2, label='MLP MAML')
                if crop_to_baseline:
                    y_min, _ = ax.get_ylim()
                    y_margin = (baseline_values[metric] - y_min) * 0.1
                    ax.set_ylim(max(0, y_min - y_margin), baseline_values[metric] * 1.02)

            ax.set_xlabel('conv_hidden_dim', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'trend_best_by_conv_dim{annot_suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    # Plot 5: Best by fc_dim
    for show_annotations, annot_suffix in annotation_configs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Best Configuration by fc_hidden_dim (ASAP7)', fontsize=14, fontweight='bold')

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            best_configs = []
            for fc_dim in fc_dims:
                filtered = [r for r in results if r['config'].get('fc_hidden_dim') == fc_dim]
                if not filtered:
                    continue
                best = min(filtered, key=lambda x: x['metrics'][metric])
                val = best['metrics'][metric]
                if metric == 'mae':
                    val = val * 1000
                best_configs.append({
                    'fc_dim': fc_dim,
                    'best_conv': best['config'].get('conv_hidden_dim'),
                    'best_val': val
                })

            x_vals = [c['fc_dim'] for c in best_configs]
            y_vals = [c['best_val'] for c in best_configs]
            labels = [f"conv={c['best_conv']}" for c in best_configs]

            bars = ax.bar(range(len(x_vals)), y_vals, color=colors[:len(x_vals)], edgecolor='black')
            if show_annotations:
                for i, (bar, label, val) in enumerate(zip(bars, labels, y_vals)):
                    height = bar.get_height()
                    ax.annotate(f'{val:.3f}\n({label})',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=9)

            # Add MLP MAML baseline horizontal line
            if metric in baseline_values:
                ax.axhline(y=baseline_values[metric], color='red', linestyle='--', linewidth=2, label='MLP MAML')
                if crop_to_baseline:
                    y_min, _ = ax.get_ylim()
                    y_margin = (baseline_values[metric] - y_min) * 0.1
                    ax.set_ylim(max(0, y_min - y_margin), baseline_values[metric] * 1.02)

            ax.set_xlabel('fc_hidden_dim', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric_labels[metric], fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'trend_best_by_fc_dim{annot_suffix}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def save_parameter_trend_summary(results, output_dir):
    """Save parameter trend analysis summary to text file. Only includes MAML models."""
    if not results:
        return

    # Filter to only include MAML models
    results = [r for r in results if r['config'].get('model_type') == 'maml']
    if not results:
        print("  No MAML results found for trend summary")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'parameter_trend_summary.txt')

    conv_dims = sorted(set(r['config'].get('conv_hidden_dim', 0) for r in results))
    fc_dims = sorted(set(r['config'].get('fc_hidden_dim', 0) for r in results))
    num_conv_layers_list = sorted(set(r['config'].get('num_conv_layers', 0) for r in results))
    num_fc_layers_list = sorted(set(r['config'].get('num_fc_layers', 0) for r in results))

    lines = []
    lines.append("=" * 100)
    lines.append("PARAMETER TREND ANALYSIS SUMMARY (ASAP7)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Total configurations analyzed: {len(results)}")
    lines.append(f"conv_hidden_dim values: {conv_dims}")
    lines.append(f"fc_hidden_dim values: {fc_dims}")
    lines.append(f"num_conv_layers values: {num_conv_layers_list}")
    lines.append(f"num_fc_layers values: {num_fc_layers_list}")
    lines.append("")

    # Overall ranking by each metric
    for metric, metric_label in [('nrmse', 'NRMSE (%)'), ('smape', 'SMAPE (%)'), ('mae', 'MAE (x1000)')]:
        lines.append("=" * 100)
        lines.append(f"OVERALL RANKING BY {metric_label} (Best to Worst)")
        lines.append("=" * 100)
        lines.append("")

        sorted_results = sorted(results, key=lambda x: x['metrics'][metric])
        header = f"{'Rank':<6} {'conv_dim':<10} {'fc_dim':<10} {'n_conv':<8} {'n_fc':<8} {'graph_mode':<12} {metric_label:<15} {'Architecture'}"
        lines.append(header)
        lines.append("-" * 100)

        for rank, r in enumerate(sorted_results, 1):
            conv_dim = r['config'].get('conv_hidden_dim', 'N/A')
            fc_dim = r['config'].get('fc_hidden_dim', 'N/A')
            n_conv = r['config'].get('num_conv_layers', 'N/A')
            n_fc = r['config'].get('num_fc_layers', 'N/A')
            graph_mode = r['config'].get('graph_mode', 'N/A')
            val = r['metrics'][metric]
            if metric == 'mae':
                val = val * 1000
            arch = r['config'].get('arch_label', 'N/A')
            line = f"{rank:<6} {conv_dim:<10} {fc_dim:<10} {n_conv:<8} {n_fc:<8} {graph_mode:<12} {val:<15.4f} {arch}"
            lines.append(line)
        lines.append("")

    # FG vs SA comparison
    lines.append("=" * 100)
    lines.append("FG (full_graph) vs SA (stage_aware) COMPARISON BY MODEL SIZE")
    lines.append("=" * 100)
    lines.append("")

    arch_groups = {}
    for r in results:
        conv_dim = r['config'].get('conv_hidden_dim', 0)
        fc_dim = r['config'].get('fc_hidden_dim', 0)
        n_conv = r['config'].get('num_conv_layers', 0)
        n_fc = r['config'].get('num_fc_layers', 0)
        graph_mode = r['config'].get('graph_mode', 'unknown')
        arch_key = (conv_dim, fc_dim, n_conv, n_fc)
        if arch_key not in arch_groups:
            arch_groups[arch_key] = {'FG': None, 'SA': None}
        mode_key = 'FG' if graph_mode == 'full_graph' else 'SA'
        arch_groups[arch_key][mode_key] = r

    sorted_archs = sorted(arch_groups.keys())
    header = f"{'conv_dim':<10} {'fc_dim':<10} {'n_conv':<8} {'n_fc':<8} | {'FG NRMSE':<12} {'SA NRMSE':<12} {'Winner':<8}"
    lines.append(header)
    lines.append("-" * 100)

    fg_wins = 0
    sa_wins = 0
    both_available = 0

    for arch_key in sorted_archs:
        conv_dim, fc_dim, n_conv, n_fc = arch_key
        fg_result = arch_groups[arch_key]['FG']
        sa_result = arch_groups[arch_key]['SA']
        if fg_result is None or sa_result is None:
            continue
        both_available += 1
        fg_nrmse = fg_result['metrics']['nrmse']
        sa_nrmse = sa_result['metrics']['nrmse']
        winner = 'FG' if fg_nrmse < sa_nrmse else 'SA'
        if winner == 'FG':
            fg_wins += 1
        else:
            sa_wins += 1
        line = f"{conv_dim:<10} {fc_dim:<10} {n_conv:<8} {n_fc:<8} | {fg_nrmse:<12.4f} {sa_nrmse:<12.4f} {winner:<8}"
        lines.append(line)

    lines.append("-" * 100)
    lines.append(f"Total: {both_available}, FG wins: {fg_wins}, SA wins: {sa_wins}")
    lines.append("")
    lines.append("=" * 100)
    lines.append("END OF PARAMETER TREND SUMMARY")
    lines.append("=" * 100)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {output_path}")

    for line in lines:
        print(line)


def print_results_table(results, output_file=None):
    """Print results as a formatted table and optionally save to file."""
    if not results:
        print("No results to display")
        return

    # Sort by NRMSE
    results = sorted(results, key=lambda x: x['metrics']['nrmse'])

    lines = []

    lines.append("\n" + "=" * 110)
    lines.append("VALIDATION RESULTS COMPARISON")
    lines.append("=" * 110)
    lines.append("Architecture format: [Model]_[Graph]_[ConvDim]x[ConvLayers]_[FCDim]x[FCLayers]")
    lines.append("  Model: B=Baseline, M=MAML  |  Graph: FG=full_graph, SA=stage_aware")
    lines.append("-" * 110)
    lines.append(f"{'Architecture':<35} {'NRMSE%':<10} {'SMAPE%':<10} {'MAE(ns)':<12} {'R²':<10} {'Samples':<10}")
    lines.append("-" * 110)

    for r in results:
        arch = r['config'].get('arch_label', 'unknown')[:28]
        nrmse = r['metrics']['nrmse']
        smape = r['metrics']['smape']
        mae = r['metrics']['mae']  # Already in ns (dataset unit)
        r2 = r['metrics']['r2']
        samples = r['metrics']['num_samples']

        lines.append(f"{arch:<30} {nrmse:<10.2f} {smape:<10.2f} {mae:<12.6f} {r2:<10.4f} {samples:<10}")

    lines.append("=" * 100)

    # Best architecture
    best = results[0]
    lines.append(f"\nBest Architecture: {best['config'].get('arch_label', 'unknown')}")
    lines.append(f"  NRMSE: {best['metrics']['nrmse']:.2f}%")
    lines.append(f"  SMAPE: {best['metrics']['smape']:.2f}%")
    lines.append(f"  MAE: {best['metrics']['mae']:.6f} ns")
    lines.append(f"  R²: {best['metrics']['r2']:.4f}")

    # Print to console
    for line in lines:
        print(line)

    # Save to file if output_file is specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare and visualize GNN validation sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compare_validation_results.py --results_dir 10000samples --mode interpolation
    python compare_validation_results.py --results_dir 10000samples --mode extrapolation
    python compare_validation_results.py --results_dir 10000samples --mode interpolation --filter "LVT_FF"

    # With --process and --corner: generates both normal + cropped plots automatically
    python compare_validation_results.py --results_dir 10000samples --mode interpolation --process LVT --corner FF
"""
    )

    parser.add_argument('--results_dir', type=str, default='',
                       help='Subdirectory under data_result_npy_directory (e.g., 10000samples)')
    parser.add_argument('--output_dir', type=str,
                       default='/home/tkdgn2907/Deepsets_test/MAML/Projects/result_management/result_summary/gcn_validation_analysis',
                       help='Output directory for plots')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['interpolation', 'extrapolation'],
                       help='Filter by validation mode (required)')
    parser.add_argument('--process', type=str, default=None,
                       choices=['LVT', 'SLVT', 'RVT', 'SRAM'],
                       help='Filter by process type')
    parser.add_argument('--corner', type=str, default=None,
                       choices=['FF', 'SS', 'TT'],
                       help='Filter by corner')
    parser.add_argument('--filter', type=str, default=None,
                       help='Additional filter by substring in filename')
    parser.add_argument('--data_filter', type=str, default='all',
                       choices=['all', 'filtered', 'unfiltered'],
                       help='Filter by data type: all (default), filtered (only filtered data), unfiltered (only non-filtered data)')
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
    print(f"Mode filter: {args.mode}")
    results = load_results(full_results_dir)

    if not results:
        print("No valid results found")
        return 1

    # Filter by mode (required)
    results = [r for r in results if r['config'].get('mode') == args.mode]
    print(f"Filtered to {len(results)} results with mode='{args.mode}'")

    if not results:
        print(f"No results found for mode '{args.mode}'")
        return 1

    # Filter by process if specified
    if args.process:
        results = [r for r in results if r['config'].get('process') == args.process]
        print(f"Filtered to {len(results)} results with process='{args.process}'")

    # Filter by corner if specified
    if args.corner:
        results = [r for r in results if r['config'].get('corner') == args.corner]
        print(f"Filtered to {len(results)} results with corner='{args.corner}'")

    # Apply additional filter if specified
    if args.filter:
        results = [r for r in results if args.filter in r['config'].get('full_name', '')]
        print(f"Further filtered to {len(results)} results matching '{args.filter}'")

    # Apply data filter (filtered/unfiltered)
    if args.data_filter == 'filtered':
        results = [r for r in results if 'filtered' in r['config'].get('full_name', '')]
        print(f"Filtered to {len(results)} results with 'filtered' in filename")
    elif args.data_filter == 'unfiltered':
        results = [r for r in results if 'filtered' not in r['config'].get('full_name', '')]
        print(f"Filtered to {len(results)} results without 'filtered' in filename")

    print(f"Loaded {len(results)} result files")

    # Create output directory with mode and filter subdirectory
    subdir_name = args.mode
    if args.process:
        subdir_name += f"_{args.process}"
    if args.corner:
        subdir_name += f"_{args.corner}"
    if args.filter:
        subdir_name += f"_{args.filter}"
    if args.data_filter != 'all':
        subdir_name += f"_{args.data_filter}"
    output_dir = os.path.join(args.output_dir, subdir_name)

    # Print table and save to txt file
    txt_output_file = os.path.join(output_dir, 'results_summary.txt')
    print_results_table(results, output_file=txt_output_file)

    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating plots to: {output_dir}")

        # Load MLP MAML baseline if process and corner are specified
        mlp_maml_baseline = None
        if args.process and args.corner:
            print(f"\nLoading MLP MAML baseline for {args.process}_{args.corner}...")
            mlp_maml_baseline = load_mlp_maml_baseline(args.process, args.corner, data_type='cell')

        # Helper function to generate all plots
        def generate_all_plots(results_to_use, out_dir, baseline, crop=False):
            plot_metrics_comparison(results_to_use, out_dir, 'nrmse', baseline, crop_to_baseline=crop)
            plot_metrics_comparison(results_to_use, out_dir, 'smape', baseline, crop_to_baseline=crop)
            plot_metrics_comparison(results_to_use, out_dir, 'mae', baseline, crop_to_baseline=crop)
            plot_all_metrics_heatmap(results_to_use, out_dir)
            plot_architecture_comparison_by_params(results_to_use, out_dir)
            plot_parameter_trend_analysis(results_to_use, out_dir, baseline, crop_to_baseline=crop)
            save_parameter_trend_summary(results_to_use, out_dir)

        # Generate normal plots
        if mlp_maml_baseline is not None:
            print("\n[1/2] Generating plots (full y-axis)...")
        else:
            print("\nGenerating plots...")
        generate_all_plots(results, output_dir, mlp_maml_baseline, crop=False)

        # Also generate cropped plots if baseline is available
        if mlp_maml_baseline is not None:
            print(f"\n[2/2] Generating cropped plots (y-axis limited to baseline)...")
            cropped_output_dir = os.path.join(output_dir, 'cropped_to_baseline')
            generate_all_plots(results, cropped_output_dir, mlp_maml_baseline, crop=True)
            print(f"  Cropped plots saved to: {cropped_output_dir}")

        print(f"\nAll plots saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
