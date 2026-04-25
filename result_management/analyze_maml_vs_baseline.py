#!/usr/bin/env python3
"""
MAML vs Baseline Pretraining Comparison
Generates publication-ready figures comparing MAML and Baseline training approaches.

Visual style inspired by speedup comparison charts:
- Stacked bars showing MAML (green) vs Baseline (blue)
- Improvement ratio displayed on top (e.g., 1.5x)
- Grouped by PDK and Data type
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import glob
import re
from collections import defaultdict

# ============================================================================
# CELL FILTER CONFIGURATION (ASAP7 only - TSMC uses all cells)
# Set to None to include all cells, or specify a list of cell names to filter
# ============================================================================
# Example configurations:
# CELL_FILTER = None  # Include all cells
# CELL_FILTER = ["AND2x6", "NAND3x2", "NOR2xp67", "OR2x6"]  # Intra-topology cells
#CELL_FILTER = ["AND2x6", "NAND3x2", "NOR2xp67", "OR2x6","OAI22x1", "AO21x1", "AO32x1", "FAx1","HAxp5","XNOR2x1","XNOR2x2","XNOR2xp5","XOR2x1","XOR2x2","XOR2xp5"]  # Topology-agnostic cells
CELL_FILTER = ["AND2x6", "NAND3x2", "NOR2xp67", "OR2x6", "AO21x1", "AO32x1","OAI22x1", "FAx1","HAxp5","XNOR2x2","XOR2x2"]  # Topology-agnostic cells

# ============================================================================

# ============================================================================
# DISPLAY LABEL MAPPING
# ============================================================================
EXPERIMENT_LABELS = {
    'topology_agnostic': 'Inter Topology',
    'intra_topology': 'Intra Topology'
}
# ============================================================================

# Publication-ready style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def parse_result_filename(filename):
    """Parse result filename to extract metadata"""
    basename = os.path.basename(filename)

    result = {
        'filename': basename,
        'prefix': None,
        'model_type': None,
        'training_type': None,
        'experiment': None,
        'data_type': None,
        'mode': None,
        'cell': None,
        'file_type': None,
        'graph_mode': None,
        'output_suffix': False,
        'innerdiv': None,
        'meta': None,
        'iterations': None,
        'conv_arch': None,
        'fc_arch': None
    }

    # Determine prefix (ASAP7 or TSMC)
    if basename.startswith('ASAP7'):
        result['prefix'] = 'ASAP7'
    elif basename.startswith('TSMC'):
        result['prefix'] = 'TSMC'

    # Determine file type
    if '_pred.npy' in basename:
        result['file_type'] = 'pred'
    elif '_act.npy' in basename:
        result['file_type'] = 'act'

    # Determine model type and training type
    if '_GCN_' in basename:
        result['model_type'] = 'GCN'
        if '_maml_' in basename:
            result['training_type'] = 'maml'
        elif '_baseline_' in basename:
            result['training_type'] = 'baseline'
    elif '_GAT_' in basename:
        result['model_type'] = 'GAT'
        if '_maml_' in basename:
            result['training_type'] = 'maml'
        elif '_baseline_' in basename:
            result['training_type'] = 'baseline'
    elif '_MAML_' in basename:
        result['model_type'] = 'MLP'
        result['training_type'] = 'maml'
    elif '_mlp_' in basename:
        result['model_type'] = 'MLP'
        result['training_type'] = 'baseline'

    # Determine experiment type
    if 'intra_topology' in basename:
        result['experiment'] = 'intra_topology'
    elif 'topology_agnostic' in basename:
        result['experiment'] = 'topology_agnostic'

    # Determine data type
    if '_cell_' in basename:
        result['data_type'] = 'cell'
    elif '_transition_' in basename:
        result['data_type'] = 'transition'

    # Determine mode
    if '_interpolation_' in basename:
        result['mode'] = 'interpolation'
    elif '_extrapolation_' in basename:
        result['mode'] = 'extrapolation'

    # Extract cell name
    cell_patterns = [
        r'(?:intra_topology|topology_agnostic)_(\w+)_(?:cell|transition)_',
    ]
    for pattern in cell_patterns:
        match = re.search(pattern, basename)
        if match:
            result['cell'] = match.group(1)
            break

    # Extract graph_mode
    if 'stage_aware' in basename:
        result['graph_mode'] = 'stage_aware'
    elif 'full_graph' in basename:
        result['graph_mode'] = 'full_graph'

    # Check for _pooloutput or _output suffix
    if '_pooloutput' in basename or '_output_' in basename or '_output.' in basename:
        result['output_suffix'] = True

    # Extract GCN MAML specific parameters
    innerdiv_match = re.search(r'_innerdiv(\d+)_', basename)
    if innerdiv_match:
        result['innerdiv'] = int(innerdiv_match.group(1))

    meta_match = re.search(r'_meta(\d+)_', basename)
    if meta_match:
        result['meta'] = int(meta_match.group(1))

    iter_match = re.search(r'_iter(\d+)_', basename)
    if iter_match:
        result['iterations'] = int(iter_match.group(1))

    conv_match = re.search(r'_conv(\d+x\d+)_', basename)
    if conv_match:
        result['conv_arch'] = conv_match.group(1)

    fc_match = re.search(r'_fc(\d+x\d+)_', basename)
    if fc_match:
        result['fc_arch'] = fc_match.group(1)

    return result


def parse_mlp_filename(filename):
    """Parse MLP result filename to extract metadata"""
    basename = os.path.basename(filename)

    # Pattern 1: ASAP7_intra_topology_{cell}_{data_type}_{mode}_MAML_...
    maml_pattern1 = r'(\w+)_(intra_topology|topology_agnostic)_(\w+)_(cell|transition)_(extrapolation|interpolation)_MAML_innerdiv(\d+)_meta(\d+)_layer(\d+)_(\d+)_(pred|act)\.npy'
    match = re.match(maml_pattern1, basename)

    if match:
        return {
            'prefix': match.group(1),
            'topology': match.group(2),
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': 'MLP_MAML',
            'innerdiv': int(match.group(6)),
            'meta': int(match.group(7)),
            'layer_length': int(match.group(8)),
            'iterations': int(match.group(9)),
            'file_type': match.group(10),
            'filename': basename
        }

    # Pattern for baseline MLP
    baseline_pattern = r'(\w+)_(intra_topology|topology_agnostic|intra|agnostic)_(\w+)_(cell|transition)_(extrapolation|interpolation)_(mlp)_(\d+)(_adam)?_(pred|act)\.npy'
    match = re.match(baseline_pattern, basename)

    if match:
        topology = match.group(2)
        if topology == 'agnostic':
            topology = 'topology_agnostic'
        elif topology == 'intra':
            topology = 'intra_topology'

        adapt_method = 'adam' if match.group(8) == '_adam' else 'selective_adam'

        return {
            'prefix': match.group(1),
            'topology': topology,
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': match.group(6).upper(),
            'iterations': int(match.group(7)),
            'adapt_method': adapt_method,
            'file_type': match.group(9),
            'filename': basename
        }

    # Legacy patterns with flexible topology names
    maml_pattern2 = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_MAML_innerdiv(\d+)_meta(\d+)_layer(\d+)_(\d+)_(pred|act)\.npy'
    match = re.match(maml_pattern2, basename)

    if match:
        topology = match.group(2)
        if topology == 'agnostic':
            topology = 'topology_agnostic'
        elif topology == 'intra':
            topology = 'intra_topology'

        return {
            'prefix': match.group(1),
            'topology': topology,
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': 'MLP_MAML',
            'innerdiv': int(match.group(6)),
            'meta': int(match.group(7)),
            'layer_length': int(match.group(8)),
            'iterations': int(match.group(9)),
            'file_type': match.group(10),
            'filename': basename
        }

    # Fallback pattern for baseline
    baseline_pattern2 = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_(mlp)_(\d+)(_adam)?_(pred|act)\.npy'
    match = re.match(baseline_pattern2, basename)

    if match:
        topology = match.group(2)
        if topology == 'agnostic':
            topology = 'topology_agnostic'
        elif topology == 'intra':
            topology = 'intra_topology'

        adapt_method = 'adam' if match.group(8) == '_adam' else 'selective_adam'

        return {
            'prefix': match.group(1),
            'topology': topology,
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': match.group(6).upper(),
            'iterations': int(match.group(7)),
            'adapt_method': adapt_method,
            'file_type': match.group(9),
            'filename': basename
        }

    return None


def calculate_metrics(predictions, actuals, group_size=61):
    """Calculate NRMSE and RMSE metrics"""
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) |
                   np.isinf(predictions) | np.isinf(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    if len(predictions) == 0:
        return None

    n_groups = len(predictions) // group_size
    if n_groups == 0:
        n_groups = 1
        group_size = len(predictions)

    predictions = predictions[:n_groups * group_size]
    actuals = actuals[:n_groups * group_size]

    pred_grouped = predictions.reshape(n_groups, group_size)
    act_grouped = actuals.reshape(n_groups, group_size)

    mse_groups = np.mean((pred_grouped - act_grouped) ** 2, axis=1)
    rmse_groups = np.sqrt(mse_groups)

    y_ranges = np.max(act_grouped, axis=1) - np.min(act_grouped, axis=1)
    y_ranges = np.where(y_ranges > 0, y_ranges, 1.0)
    nrmse_groups = (rmse_groups / y_ranges) * 100

    return {
        'NRMSE': float(np.mean(nrmse_groups)),
        'RMSE': float(np.mean(rmse_groups)),
        'num_samples': len(predictions),
        'num_groups': n_groups
    }


def load_results(data_dir, prefix_filter=None, model_filter=None, training_filter=None,
                 innerdiv_filter=None, meta_filter=None, iter_filter=None,
                 conv_filter=None, fc_filter=None, cells_filter=None,
                 data_type_filter=None, output_suffix_filter=None):
    """Load results from directory"""
    results = []

    search_dirs = [data_dir]
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            search_dirs.append(item_path)

    all_pred_files = []
    for search_dir in search_dirs:
        pred_files = glob.glob(os.path.join(search_dir, '*_pred.npy'))
        all_pred_files.extend(pred_files)

    print(f"Found {len(all_pred_files)} prediction files...")

    seen_filenames = set()
    for pred_file in all_pred_files:
        basename = os.path.basename(pred_file)
        if basename in seen_filenames:
            continue
        seen_filenames.add(basename)

        act_file = pred_file.replace('_pred.npy', '_act.npy')
        if not os.path.exists(act_file):
            continue

        metadata = parse_result_filename(pred_file)
        if metadata is None or metadata['file_type'] != 'pred':
            continue

        if prefix_filter and metadata['prefix'] != prefix_filter:
            continue
        if model_filter and metadata['model_type'] != model_filter:
            continue
        if training_filter and metadata['training_type'] != training_filter:
            continue
        if innerdiv_filter is not None and metadata.get('innerdiv') != innerdiv_filter:
            continue
        if meta_filter is not None and metadata.get('meta') != meta_filter:
            continue
        if iter_filter is not None and metadata.get('iterations') != iter_filter:
            continue
        if conv_filter is not None and metadata.get('conv_arch') != conv_filter:
            continue
        if fc_filter is not None and metadata.get('fc_arch') != fc_filter:
            continue
        if cells_filter is not None and metadata.get('prefix') == 'ASAP7':
            if metadata.get('cell') not in cells_filter:
                continue
        if data_type_filter is not None and metadata.get('data_type') != data_type_filter:
            continue
        if output_suffix_filter is not None and metadata.get('output_suffix') != output_suffix_filter:
            continue

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)

            if len(predictions) != len(actuals) or len(predictions) == 0:
                continue

            metrics = calculate_metrics(predictions, actuals)
            if metrics is None:
                continue

            result = {**metadata, **metrics}
            results.append(result)

        except Exception as e:
            continue

    print(f"Loaded {len(results)} results")
    return pd.DataFrame(results) if results else None


def load_mlp_results(mlp_maml_dir, mlp_baseline_dir=None, prefix_filter=None, experiment_filter=None, data_type_filter=None, cells_filter=None):
    """Load MLP results from separate directories for MLP MAML and MLP Baseline"""
    results = []

    if mlp_baseline_dir is None:
        mlp_baseline_dir = mlp_maml_dir

    pred_files_maml = glob.glob(os.path.join(mlp_maml_dir, '*_pred.npy'))
    print(f"Found {len(pred_files_maml)} MLP prediction files in mlp_maml_dir...")

    if mlp_baseline_dir != mlp_maml_dir:
        pred_files_baseline = glob.glob(os.path.join(mlp_baseline_dir, '*_pred.npy'))
        print(f"Found {len(pred_files_baseline)} MLP prediction files in mlp_baseline_dir...")
        pred_files = list(set(pred_files_maml + pred_files_baseline))
    else:
        pred_files = pred_files_maml
    print(f"Total {len(pred_files)} prediction files to process...")

    seen_filenames = set()
    for pred_file in pred_files:
        basename = os.path.basename(pred_file)
        if basename in seen_filenames:
            continue
        seen_filenames.add(basename)

        act_file = pred_file.replace('_pred.npy', '_act.npy')
        if not os.path.exists(act_file):
            continue

        metadata = parse_mlp_filename(pred_file)
        if metadata is None:
            continue

        if prefix_filter and metadata.get('prefix') != prefix_filter:
            continue
        if experiment_filter and metadata.get('topology') != experiment_filter:
            continue
        if data_type_filter and metadata.get('data_type') != data_type_filter:
            continue
        if cells_filter is not None and metadata.get('prefix') == 'ASAP7':
            if metadata.get('cell') not in cells_filter:
                continue

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)

            if len(predictions) != len(actuals) or len(predictions) == 0:
                continue

            metrics = calculate_metrics(predictions, actuals)
            if metrics is None:
                continue

            result = {**metadata, **metrics}
            results.append(result)

        except Exception as e:
            continue

    print(f"Loaded {len(results)} MLP results")
    return pd.DataFrame(results) if results else None


def plot_maml_vs_baseline_compact_pdk_combined(asap7_data, tsmc_data, output_path, metric='NRMSE'):
    """
    Generate compact PDK comparison chart combining ASAP7 and Commercial (TSMC).
    For each model_mode (MLP_Extra, MLP_Inter, GCN_Extra, GCN_Inter), averages cell and transition results.
    Layout: Commercial (MLP-Extra, MLP-Inter, GCN-Extra, GCN-Inter) | ASAP7 (same)
    Style: Based on generate_maml_fig.py for publication quality.
    """
    import matplotlib
    from matplotlib.patches import Patch

    # Publication style settings
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 7
    matplotlib.rcParams['axes.linewidth'] = 0.5
    matplotlib.rcParams['xtick.major.width'] = 0.4
    matplotlib.rcParams['ytick.major.width'] = 0.4
    matplotlib.rcParams['xtick.major.size'] = 2.5
    matplotlib.rcParams['ytick.major.size'] = 2.5

    COL_W = 3.333
    FIG_H = 2.0

    def get_averaged_data(pdk_data, model):
        cell_extra = pdk_data.get('cell_extra', {}).get(model, {})
        cell_inter = pdk_data.get('cell_inter', {}).get(model, {})
        trans_extra = pdk_data.get('trans_extra', {}).get(model, {})
        trans_inter = pdk_data.get('trans_inter', {}).get(model, {})

        result = {}
        for mode_name, cell_data, trans_data in [('Extra', cell_extra, trans_extra), ('Inter', cell_inter, trans_inter)]:
            cell_maml = cell_data.get('maml', np.nan)
            cell_baseline = cell_data.get('baseline', np.nan)
            trans_maml = trans_data.get('maml', np.nan)
            trans_baseline = trans_data.get('baseline', np.nan)

            maml_vals = [v for v in [cell_maml, trans_maml] if not np.isnan(v)]
            baseline_vals = [v for v in [cell_baseline, trans_baseline] if not np.isnan(v)]

            avg_maml = np.mean(maml_vals) if maml_vals else np.nan
            avg_baseline = np.mean(baseline_vals) if baseline_vals else np.nan

            result[mode_name] = {'maml': avg_maml, 'baseline': avg_baseline}

        return result

    # Colors by model type (MLP=Green, GCN=Blue)
    c_mlp_baseline = '#b4d4b4'
    c_mlp_maml = '#228b22'
    c_gcn_baseline = '#A8C4E0'
    c_gcn_maml = '#1B5E91'

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, FIG_H), sharey=False)
    pdks = ['Commercial', 'ASAP7']
    pdk_data_map = {'Commercial': tsmc_data, 'ASAP7': asap7_data}

    for ax_idx, pdk in enumerate(pdks):
        ax = axes[ax_idx]
        pdk_data = pdk_data_map[pdk]

        mlp_avg = get_averaged_data(pdk_data, 'MLP')
        gcn_avg = get_averaged_data(pdk_data, 'GCN')

        baselines = [
            mlp_avg['Extra'].get('baseline', np.nan),
            mlp_avg['Inter'].get('baseline', np.nan),
            gcn_avg['Extra'].get('baseline', np.nan),
            gcn_avg['Inter'].get('baseline', np.nan),
        ]
        maml_vals = [
            mlp_avg['Extra'].get('maml', np.nan),
            mlp_avg['Inter'].get('maml', np.nan),
            gcn_avg['Extra'].get('maml', np.nan),
            gcn_avg['Inter'].get('maml', np.nan),
        ]
        ratios = []
        for b, m in zip(baselines, maml_vals):
            if not np.isnan(b) and not np.isnan(m) and m > 0:
                ratios.append(b / m)
            else:
                ratios.append(np.nan)

        x = np.array([0, 0.6, 1.5, 2.1])
        w = 0.22

        # Plot bars - MLP section (green)
        ax.bar(x[:2] - w/2, baselines[:2], w, color=c_mlp_baseline, edgecolor='#555555',
               linewidth=0.4, linestyle='--', label='w/o MAML', zorder=3)
        ax.bar(x[:2] + w/2, maml_vals[:2], w, color=c_mlp_maml, edgecolor='#333333',
               linewidth=0.4, label='MAML', zorder=3)
        # Plot bars - GCN section (blue)
        ax.bar(x[2:] - w/2, baselines[2:], w, color=c_gcn_baseline, edgecolor='#555555',
               linewidth=0.4, linestyle='--', zorder=3)
        ax.bar(x[2:] + w/2, maml_vals[2:], w, color=c_gcn_maml, edgecolor='#333333',
               linewidth=0.4, zorder=3)

        # Improvement-factor arrows (downward) + labels
        ymax = max([v for v in baselines + maml_vals if not np.isnan(v)]) * 1.1 if any(not np.isnan(v) for v in baselines + maml_vals) else 1
        for i in range(4):
            bh, mh = baselines[i], maml_vals[i]
            if np.isnan(bh) or np.isnan(mh):
                continue
            arrow_x = x[i]
            ax.annotate('', xy=(arrow_x, mh + 0.02), xytext=(arrow_x, bh - 0.02),
                        arrowprops=dict(arrowstyle='->', color='#C0392B',
                                        lw=0.8, shrinkA=0, shrinkB=0))
            if not np.isnan(ratios[i]):
                text_y = bh + ymax * 0.02
                ax.text(arrow_x, text_y,
                        f'×{ratios[i]:.2f}',
                        fontsize=5.5, fontweight='bold', color='#C0392B',
                        va='bottom', ha='center')

        ax.set_xticks(x)
        ax.set_xticklabels(['Extra', 'Inter', 'Extra', 'Inter'], fontsize=6)

        ax.text(np.mean(x[:2]), -0.22, 'MLP', fontsize=7, fontweight='bold',
                ha='center', va='top', transform=ax.get_xaxis_transform())
        ax.text(np.mean(x[2:]), -0.22, 'GCN', fontsize=7, fontweight='bold',
                ha='center', va='top', transform=ax.get_xaxis_transform())

        sep_x = (x[1] + x[2]) / 2
        ax.axvline(sep_x, color='gray', ls='--', lw=0.4, alpha=0.5, zorder=1)

        ax.set_title(pdk, fontsize=7.5, fontweight='bold', pad=2)
        if ax_idx == 0:
            ax.set_ylabel(f'{metric} (%)', fontsize=6.5)
        ax.set_ylim(0, ymax)
        ax.set_xlim(x[0] - 0.35, x[-1] + 0.55)
        yticks = np.arange(0, ymax + 1, 1)
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', labelsize=4.5)
        ax.grid(axis='y', alpha=0.2, ls='-', lw=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend (4 entries)
    legend_handles = [
        Patch(facecolor=c_mlp_baseline, edgecolor='#555555', linestyle='--', linewidth=0.5, label='MLP w/o MAML'),
        Patch(facecolor=c_mlp_maml, edgecolor='#333333', linewidth=0.5, label='MLP MAML'),
        Patch(facecolor=c_gcn_baseline, edgecolor='#555555', linestyle='--', linewidth=0.5, label='GCN w/o MAML'),
        Patch(facecolor=c_gcn_maml, edgecolor='#333333', linewidth=0.5, label='GCN MAML'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=4, fontsize=6,
               frameon=True, fancybox=False, edgecolor='gray',
               bbox_to_anchor=(0.5, 1.02), handlelength=1.0, handletextpad=0.3,
               columnspacing=0.8)

    plt.tight_layout(rect=[0, 0.01, 1, 0.94], w_pad=1.2)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved PDK combined compact plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='MAML vs Baseline Pretraining Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_maml_vs_baseline.py --data_dir ./data_result_npy_directory
  python analyze_maml_vs_baseline.py --prefix ASAP7 --experiment intra_topology
        """
    )

    parser.add_argument('--gcn_dir', type=str,
                       default='../pretraining/model_test_code/gnn/data_result_npy_directory_final',
                       help='Directory containing GCN .npy result files')
    parser.add_argument('--mlp_maml_dir', type=str,
                       default='../pretraining/model_test_code/MLP/data_result_npy_directory_maml',
                       help='Directory containing MLP MAML .npy result files')
    parser.add_argument('--mlp_baseline_dir', type=str,
                       default='../pretraining/model_test_code/MLP/data_result_npy_directory_baseline',
                       help='Directory containing MLP Baseline (AADAM) .npy result files')
    parser.add_argument('--output_dir', type=str,
                       default='./result_summary/maml_vs_baseline',
                       help='Output directory')
    parser.add_argument('--prefix', type=str, default=None,
                       choices=['ASAP7', 'TSMC'],
                       help='Filter by PDK')
    parser.add_argument('--experiment', type=str, default=None,
                       choices=['intra_topology', 'topology_agnostic'],
                       help='Filter by experiment type')
    parser.add_argument('--mode', type=str, default='extrapolation',
                       choices=['interpolation', 'extrapolation'],
                       help='Filter by mode')
    parser.add_argument('--mlp_baseline_iter', type=int, default=300000,
                       help='MLP baseline iteration for comparison (default: 300000)')
    parser.add_argument('--mlp_adapt_method', type=str, default='selective_adam',
                       choices=['selective_adam', 'adam'],
                       help='MLP adaptation method (default: selective_adam)')
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['stage_aware', 'full_graph'],
                       help='Filter GCN results by graph mode (default: stage_aware)')
    parser.add_argument('--gcn_innerdiv', type=int, default=10,
                       help='Filter GCN MAML by innerdiv value')
    parser.add_argument('--gcn_meta', type=int, default=16,
                       help='Filter GCN MAML by meta value')
    parser.add_argument('--gcn_iter', type=int, default=300000,
                       help='Filter GCN MAML by iterations')
    parser.add_argument('--gcn_conv', type=str, default="64x2",
                       help='Filter GCN by conv architecture')
    parser.add_argument('--gcn_fc', type=str, default="256x2",
                       help='Filter GCN by fc architecture')

    args = parser.parse_args()

    print("=" * 80)
    print("MAML vs BASELINE COMPARISON")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    if CELL_FILTER:
        print(f"\nCell filter active (ASAP7 only): {CELL_FILTER}")

    # Load GCN results
    print("\nLoading GCN MAML results (cell)...")
    gcn_maml_cell_df = load_results(args.gcn_dir, prefix_filter=args.prefix,
                               model_filter='GCN', training_filter='maml',
                               innerdiv_filter=args.gcn_innerdiv,
                               meta_filter=args.gcn_meta,
                               iter_filter=args.gcn_iter,
                               conv_filter=args.gcn_conv,
                               fc_filter=args.gcn_fc,
                               cells_filter=CELL_FILTER,
                               data_type_filter='cell',
                               output_suffix_filter=False)

    print("\nLoading GCN MAML results (transition)...")
    gcn_maml_tran_df = load_results(args.gcn_dir, prefix_filter=args.prefix,
                               model_filter='GCN', training_filter='maml',
                               innerdiv_filter=args.gcn_innerdiv,
                               meta_filter=args.gcn_meta,
                               iter_filter=args.gcn_iter,
                               conv_filter=args.gcn_conv,
                               fc_filter=args.gcn_fc,
                               cells_filter=CELL_FILTER,
                               data_type_filter='transition',
                               output_suffix_filter=True)

    gcn_maml_dfs = [df for df in [gcn_maml_cell_df, gcn_maml_tran_df] if df is not None and len(df) > 0]
    gcn_maml_df = pd.concat(gcn_maml_dfs, ignore_index=True) if gcn_maml_dfs else None

    print("\nLoading GCN Baseline results (cell)...")
    gcn_baseline_cell_df = load_results(args.gcn_dir, prefix_filter=args.prefix,
                                    model_filter='GCN', training_filter='baseline',
                                    conv_filter=args.gcn_conv,
                                    fc_filter=args.gcn_fc,
                                    cells_filter=CELL_FILTER,
                                    data_type_filter='cell',
                                    output_suffix_filter=False)

    print("\nLoading GCN Baseline results (transition)...")
    gcn_baseline_tran_df = load_results(args.gcn_dir, prefix_filter=args.prefix,
                                    model_filter='GCN', training_filter='baseline',
                                    conv_filter=args.gcn_conv,
                                    fc_filter=args.gcn_fc,
                                    cells_filter=CELL_FILTER,
                                    data_type_filter='transition',
                                    output_suffix_filter=True)

    gcn_baseline_dfs = [df for df in [gcn_baseline_cell_df, gcn_baseline_tran_df] if df is not None and len(df) > 0]
    gcn_baseline_df = pd.concat(gcn_baseline_dfs, ignore_index=True) if gcn_baseline_dfs else None

    # Load MLP results
    print("\nLoading MLP results...")
    mlp_df = load_mlp_results(args.mlp_maml_dir, mlp_baseline_dir=args.mlp_baseline_dir,
                               prefix_filter=args.prefix,
                               experiment_filter=args.experiment,
                               cells_filter=CELL_FILTER)

    mlp_maml_df = None
    mlp_baseline_df = None

    if mlp_df is not None and len(mlp_df) > 0:
        maml_data = mlp_df[mlp_df['model_type'] == 'MLP_MAML']
        required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
        if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
            mlp_maml_df = maml_data[
                (maml_data['innerdiv'] == 100) &
                (maml_data['meta'] == 32) &
                (maml_data['layer_length'] == 40) &
                (maml_data['iterations'] == 300000)
            ].copy()
        else:
            mlp_maml_df = maml_data.copy() if len(maml_data) > 0 else None

        mlp_baseline_data = mlp_df[mlp_df['model_type'] == 'MLP']
        if len(mlp_baseline_data) > 0:
            mlp_baseline_data = mlp_baseline_data[mlp_baseline_data['iterations'] == args.mlp_baseline_iter]
            if 'adapt_method' in mlp_baseline_data.columns:
                mlp_baseline_data = mlp_baseline_data[mlp_baseline_data['adapt_method'] == args.mlp_adapt_method]
            mlp_baseline_df = mlp_baseline_data.copy()

    # Helper function to aggregate data
    def aggregate_data(mode_filter, exp_filter, graph_mode_filter):
        data = {}
        for prefix in ['ASAP7', 'TSMC']:
            for dtype in ['cell', 'transition']:
                key = f"{prefix}_{dtype}"
                data[key] = {}

                expected_output_suffix = (dtype == 'transition')

                # GCN MAML
                if gcn_maml_df is not None:
                    filtered = gcn_maml_df[
                        (gcn_maml_df['prefix'] == prefix) &
                        (gcn_maml_df['data_type'] == dtype)
                    ]
                    if 'output_suffix' in gcn_maml_df.columns:
                        filtered = filtered[filtered['output_suffix'] == expected_output_suffix]
                    if mode_filter:
                        filtered = filtered[filtered['mode'] == mode_filter]
                    if exp_filter:
                        filtered = filtered[filtered['experiment'] == exp_filter]
                    if graph_mode_filter and 'graph_mode' in gcn_maml_df.columns:
                        filtered = filtered[filtered['graph_mode'] == graph_mode_filter]
                    if len(filtered) > 0:
                        if 'GCN' not in data[key]:
                            data[key]['GCN'] = {}
                        data[key]['GCN']['maml'] = filtered['NRMSE'].mean()

                # GCN Baseline
                if gcn_baseline_df is not None:
                    filtered = gcn_baseline_df[
                        (gcn_baseline_df['prefix'] == prefix) &
                        (gcn_baseline_df['data_type'] == dtype)
                    ]
                    if 'output_suffix' in gcn_baseline_df.columns:
                        filtered = filtered[filtered['output_suffix'] == expected_output_suffix]
                    if mode_filter:
                        filtered = filtered[filtered['mode'] == mode_filter]
                    if exp_filter:
                        filtered = filtered[filtered['experiment'] == exp_filter]
                    if graph_mode_filter and 'graph_mode' in gcn_baseline_df.columns:
                        filtered = filtered[filtered['graph_mode'] == graph_mode_filter]
                    if len(filtered) > 0:
                        if 'GCN' not in data[key]:
                            data[key]['GCN'] = {}
                        data[key]['GCN']['baseline'] = filtered['NRMSE'].mean()

                # MLP MAML
                if mlp_maml_df is not None and len(mlp_maml_df) > 0:
                    filtered = mlp_maml_df[
                        (mlp_maml_df['prefix'] == prefix) &
                        (mlp_maml_df['data_type'] == dtype)
                    ]
                    if mode_filter:
                        filtered = filtered[filtered['mode'] == mode_filter]
                    if exp_filter and 'topology' in mlp_maml_df.columns:
                        filtered = filtered[filtered['topology'] == exp_filter]
                    if len(filtered) > 0:
                        if 'MLP' not in data[key]:
                            data[key]['MLP'] = {}
                        data[key]['MLP']['maml'] = filtered['NRMSE'].mean()

                # MLP Baseline
                if mlp_baseline_df is not None and len(mlp_baseline_df) > 0:
                    filtered = mlp_baseline_df[
                        (mlp_baseline_df['prefix'] == prefix) &
                        (mlp_baseline_df['data_type'] == dtype)
                    ]
                    if mode_filter:
                        filtered = filtered[filtered['mode'] == mode_filter]
                    if exp_filter and 'topology' in mlp_baseline_df.columns:
                        filtered = filtered[filtered['topology'] == exp_filter]
                    if len(filtered) > 0:
                        if 'MLP' not in data[key]:
                            data[key]['MLP'] = {}
                        data[key]['MLP']['baseline'] = filtered['NRMSE'].mean()

        return data

    # Generate PDK combined plot
    print(f"\nGenerating PDK COMBINED plot (Commercial vs ASAP7)...")

    graph_mode_suffix = f"_{args.graph_mode}" if args.graph_mode else ""

    extra_data = aggregate_data('extrapolation', args.experiment, args.graph_mode)
    inter_data = aggregate_data('interpolation', args.experiment, args.graph_mode)

    pdk_data = {}
    for prefix in ['ASAP7', 'TSMC']:
        cell_key = f"{prefix}_cell"
        trans_key = f"{prefix}_transition"

        pdk_data[prefix] = {
            'cell_extra': {},
            'cell_inter': {},
            'trans_extra': {},
            'trans_inter': {}
        }

        if cell_key in extra_data and extra_data[cell_key]:
            for model in ['MLP', 'GCN']:
                if model in extra_data[cell_key]:
                    pdk_data[prefix]['cell_extra'][model] = extra_data[cell_key][model]
        if cell_key in inter_data and inter_data[cell_key]:
            for model in ['MLP', 'GCN']:
                if model in inter_data[cell_key]:
                    pdk_data[prefix]['cell_inter'][model] = inter_data[cell_key][model]
        if trans_key in extra_data and extra_data[trans_key]:
            for model in ['MLP', 'GCN']:
                if model in extra_data[trans_key]:
                    pdk_data[prefix]['trans_extra'][model] = extra_data[trans_key][model]
        if trans_key in inter_data and inter_data[trans_key]:
            for model in ['MLP', 'GCN']:
                if model in inter_data[trans_key]:
                    pdk_data[prefix]['trans_inter'][model] = inter_data[trans_key][model]

    asap7_has_data = any(pdk_data['ASAP7'][k] for k in pdk_data['ASAP7'])
    tsmc_has_data = any(pdk_data['TSMC'][k] for k in pdk_data['TSMC'])

    if asap7_has_data and tsmc_has_data:
        output_path = os.path.join(args.output_dir,
                                  f'pdk_combined_maml_vs_baseline_compact{graph_mode_suffix}.png')
        plot_maml_vs_baseline_compact_pdk_combined(pdk_data['ASAP7'], pdk_data['TSMC'], output_path, metric='NRMSE')
    else:
        print("Skipping PDK combined plot: insufficient data for both PDKs")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_data = aggregate_data(args.mode, args.experiment, args.graph_mode)

    for key, data in all_data.items():
        if data:
            print(f"\n{key}:")
            for model, values in data.items():
                maml = values.get('maml', np.nan)
                baseline = values.get('baseline', np.nan)

                if not np.isnan(maml) and not np.isnan(baseline):
                    ratio = baseline / maml if maml > 0 else np.nan
                    print(f"  {model}: MAML={maml:.2f}%, Baseline={baseline:.2f}%, Improvement={ratio:.2f}x")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
