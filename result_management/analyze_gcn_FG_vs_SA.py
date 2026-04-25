#!/usr/bin/env python3
"""
GCN Cell-based vs Current-path based Graph Comparison (MAML only)
Generates publication-ready figures comparing full_graph and stage_aware GNN modes.

Visual style:
- Side-by-side bars showing Cell-based Graph (orange) vs Current-path based Graph (green)
- Improvement ratio displayed with arrow
- Grouped by PDK and Data type (Cell/Transition)
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
# CELL_FILTER = ["A2O1A1O1Ixp25", "AO21x1", "AO32x1", "AOI332xp5", "O2A1O1Ixp5", "OAI22x1", "FAx1"]  # Topology-agnostic cells

#CELL_FILTER = ["AND2x6", "NAND3x2", "NOR2xp67", "OR2x6", "AO21x1", "AO32x1","OAI22x1", "FAx1","HAxp5","XNOR2x1","XNOR2x2","XNOR2xp5","XOR2x1","XOR2x2","XOR2xp5"]  # Topology-agnostic cells
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
        'vdd_only': False,
        'relpin': False,
        'output_suffix': False,  # True if _output suffix exists (for transition)
        'innerdiv': None,
        'meta': None,
        'iteration': None
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

    # Check for vdd_only and relpin suffixes
    if '_vddonly' in basename or '_vdd_only' in basename:
        result['vdd_only'] = True
    if '_relpin' in basename or '_rel_pin' in basename:
        result['relpin'] = True

    # Check for _output suffix (transition output slew vs mean)
    # Also check for pooloutput which indicates output pooling mode
    if '_output_' in basename or '_output.' in basename or '_pooloutput' in basename:
        result['output_suffix'] = True

    # Extract innerdiv and meta values
    innerdiv_match = re.search(r'innerdiv(\d+)', basename)
    if innerdiv_match:
        result['innerdiv'] = int(innerdiv_match.group(1))

    meta_match = re.search(r'meta(\d+)', basename)
    if meta_match:
        result['meta'] = int(meta_match.group(1))

    # Extract iteration value (e.g., iter300000 or iteration300000)
    iter_match = re.search(r'iter(?:ation)?(\d+)', basename)
    if iter_match:
        result['iteration'] = int(iter_match.group(1))

    return result


def calculate_metrics(predictions, actuals, group_size=61):
    """Calculate NRMSE and RMSE metrics"""
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Filter invalid values
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
                 graph_mode_filter=None, vdd_only_filter=None, relpin_filter=None,
                 output_suffix_filter=None, innerdiv_filter=None, meta_filter=None,
                 iteration_filter=None, cells_filter=None, verbose=False):
    """Load results from directory

    Args:
        vdd_only_filter: None (don't filter), True (only vdd_only), False (only non-vdd_only)
        relpin_filter: None (don't filter), True (only relpin), False (only non-relpin)
        output_suffix_filter: None (don't filter), True (only _output), False (only mean/no suffix)
        innerdiv_filter: None (don't filter), or int value to match
        meta_filter: None (don't filter), or int value to match
        iteration_filter: None (don't filter), or int value to match
        cells_filter: List of cell names to include (None = all cells)
        verbose: If True, print loaded filenames
    """
    results = []
    loaded_files = []  # Track loaded filenames for verbose output

    # Search in main directory and subdirectories
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

    for pred_file in all_pred_files:
        act_file = pred_file.replace('_pred.npy', '_act.npy')
        if not os.path.exists(act_file):
            continue

        metadata = parse_result_filename(pred_file)
        if metadata is None or metadata['file_type'] != 'pred':
            continue

        # Apply filters
        if prefix_filter and metadata['prefix'] != prefix_filter:
            continue
        if model_filter and metadata['model_type'] != model_filter:
            continue
        if training_filter and metadata['training_type'] != training_filter:
            continue
        if graph_mode_filter and metadata['graph_mode'] != graph_mode_filter:
            continue
        if vdd_only_filter is not None and metadata['vdd_only'] != vdd_only_filter:
            continue
        if relpin_filter is not None and metadata['relpin'] != relpin_filter:
            continue
        if output_suffix_filter is not None and metadata['output_suffix'] != output_suffix_filter:
            continue
        if innerdiv_filter is not None and metadata['innerdiv'] != innerdiv_filter:
            continue
        if meta_filter is not None and metadata['meta'] != meta_filter:
            continue
        if iteration_filter is not None and metadata['iteration'] != iteration_filter:
            continue

        # Apply cell name filter (only for ASAP7)
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
            loaded_files.append(os.path.basename(pred_file))

        except Exception as e:
            continue

    print(f"Loaded {len(results)} results")

    # Print loaded filenames if verbose
    if verbose and loaded_files:
        print(f"  Loaded files ({len(loaded_files)}):")
        for fname in loaded_files:
            print(f"    - {fname}")

    return pd.DataFrame(results) if results else None


def plot_fg_vs_sa_pdk_combined(asap7_cell, asap7_trans, tsmc_cell, tsmc_trans, output_path, metric='NRMSE'):
    """
    Generate combined PDK comparison for FG vs SA.
    Layout: Commercial (Cell_Extra, Cell_Inter, Trans_Extra, Trans_Inter) | ASAP7 (same)
    Each group shows FG (Cell-based, dashed) vs SA (Current-path, solid) bars side by side.
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

    # Figure size for publication
    COL_W = 3.333
    FIG_H = 2.0

    # Colors (GCN uses blue family for consistency)
    c_fg = '#A8C4E0'   # Light blue for Cell-based (FG)
    c_sa = '#1B5E91'   # Dark blue for Current-path (SA)

    fig, axes = plt.subplots(1, 2, figsize=(COL_W, FIG_H), sharey=False)
    pdks = ['Commercial', 'ASAP7']
    pdk_data_map = {
        'Commercial': {'cell': tsmc_cell, 'trans': tsmc_trans},
        'ASAP7': {'cell': asap7_cell, 'trans': asap7_trans}
    }

    for ax_idx, pdk in enumerate(pdks):
        ax = axes[ax_idx]
        cell_data = pdk_data_map[pdk]['cell']
        trans_data = pdk_data_map[pdk]['trans']

        # Collect data: Cell-Ext, Cell-Int, Trans-Ext, Trans-Int
        fg_vals = [
            cell_data.get('extra', {}).get('full_graph', np.nan),
            cell_data.get('inter', {}).get('full_graph', np.nan),
            trans_data.get('extra', {}).get('full_graph', np.nan),
            trans_data.get('inter', {}).get('full_graph', np.nan),
        ]
        sa_vals = [
            cell_data.get('extra', {}).get('stage_aware', np.nan),
            cell_data.get('inter', {}).get('stage_aware', np.nan),
            trans_data.get('extra', {}).get('stage_aware', np.nan),
            trans_data.get('inter', {}).get('stage_aware', np.nan),
        ]
        ratios = []
        for fg, sa in zip(fg_vals, sa_vals):
            if not np.isnan(fg) and not np.isnan(sa) and sa > 0:
                ratios.append(fg / sa)
            else:
                ratios.append(np.nan)

        x = np.array([0, 0.6, 1.5, 2.1])
        w = 0.22

        # Plot bars
        ax.bar(x - w/2, fg_vals, w, color=c_fg, edgecolor='#555555',
               linewidth=0.4, linestyle='--', label='Cell-based', zorder=3)
        ax.bar(x + w/2, sa_vals, w, color=c_sa, edgecolor='#333333',
               linewidth=0.4, label='Current-path', zorder=3)

        # Improvement-factor arrows (downward) + labels
        all_vals = [v for v in fg_vals + sa_vals if not np.isnan(v)]
        ymax = max(all_vals) * 1.1 if all_vals else 1
        for i in range(4):
            fh, sh = fg_vals[i], sa_vals[i]
            if np.isnan(fh) or np.isnan(sh):
                continue
            # Downward arrow from FG to SA (touching bar tops)
            arrow_x = x[i]
            ax.annotate('', xy=(arrow_x, sh + 0.02), xytext=(arrow_x, fh - 0.02),
                        arrowprops=dict(arrowstyle='->', color='#C0392B',
                                        lw=0.8, shrinkA=0, shrinkB=0))
            if not np.isnan(ratios[i]):
                # Ratio text above the arrow
                text_y = fh + ymax * 0.02
                ax.text(arrow_x, text_y,
                        f'×{ratios[i]:.2f}',
                        fontsize=5.5, fontweight='bold', color='#C0392B',
                        va='bottom', ha='center')

        ax.set_xticks(x)
        ax.set_xticklabels(['Ext', 'Int', 'Ext', 'Int'], fontsize=6)

        # Data type group labels below tick labels (compact spacing)
        ax.text(np.mean(x[:2]), -0.22, 'Cell', fontsize=7, fontweight='bold',
                ha='center', va='top', transform=ax.get_xaxis_transform())
        ax.text(np.mean(x[2:]), -0.22, 'Tran', fontsize=7, fontweight='bold',
                ha='center', va='top', transform=ax.get_xaxis_transform())

        # Vertical separator between Cell and Tran
        sep_x = (x[1] + x[2]) / 2
        ax.axvline(sep_x, color='gray', ls='--', lw=0.4, alpha=0.5, zorder=1)

        ax.set_title(pdk, fontsize=7.5, fontweight='bold', pad=2)
        if ax_idx == 0:
            ax.set_ylabel(f'{metric} (%)', fontsize=6.5)
        ax.set_ylim(0, ymax)
        ax.set_xlim(x[0] - 0.35, x[-1] + 0.55)
        # Y-axis ticks at 1% intervals with smaller font than ratio text
        yticks = np.arange(0, ymax + 1, 1)
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', labelsize=4.5)
        ax.grid(axis='y', alpha=0.2, ls='-', lw=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend (with dashed edge for Cell-based)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c_fg, edgecolor='#555555', linestyle='--', linewidth=1.0, label='Cell-based'),
        Patch(facecolor=c_sa, edgecolor='#333333', label='Current-path'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=2, fontsize=6,
               frameon=True, fancybox=False, edgecolor='gray',
               bbox_to_anchor=(0.5, 1.02), handlelength=1.2, handletextpad=0.4,
               columnspacing=1.0)

    plt.tight_layout(rect=[0, 0.01, 1, 0.94], w_pad=1.2)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved PDK combined FG vs SA plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='GCN Full-Graph vs Stage-Aware Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_gcn_FG_vs_SA.py --vdd_only --relpin
  python analyze_gcn_FG_vs_SA.py --experiment topology_agnostic --tsmc_fg_innerdiv 100 --tsmc_fg_meta 32
        """
    )

    parser.add_argument('--data_dir', type=str,
                       default='/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/gnn/data_result_npy_directory_final',
                       help='Directory containing GCN .npy result files')
    parser.add_argument('--output_dir', type=str,
                       default='./result_summary/fg_vs_sa',
                       help='Output directory')
    parser.add_argument('--experiment', type=str, default=None,
                       choices=['intra_topology', 'topology_agnostic'],
                       help='Filter by experiment type')
    parser.add_argument('--mode', type=str, default='extrapolation',
                       choices=['interpolation', 'extrapolation'],
                       help='Filter by mode')
    parser.add_argument('--vdd_only', action='store_true',
                       help='Filter for vdd_only results (voltage only on VDD nodes)')
    parser.add_argument('--relpin', action='store_true',
                       help='Filter for related_pin_only results')
    # ASAP7 parameters
    parser.add_argument('--asap7_fg_cell_innerdiv', type=int, default=10,
                       help='Inner division for ASAP7 Full-Graph cell delay (default: 10)')
    parser.add_argument('--asap7_fg_cell_meta', type=int, default=16,
                       help='Meta batch size for ASAP7 Full-Graph cell delay (default: 16)')
    parser.add_argument('--asap7_fg_tran_innerdiv', type=int, default=10,
                       help='Inner division for ASAP7 Full-Graph transition (default: 10)')
    parser.add_argument('--asap7_fg_tran_meta', type=int, default=16,
                       help='Meta batch size for ASAP7 Full-Graph transition (default: 16)')
    parser.add_argument('--asap7_sa_innerdiv', type=int, default=10,
                       help='Inner division for ASAP7 Stage-Aware (default: 10)')
    parser.add_argument('--asap7_sa_meta', type=int, default=16,
                       help='Meta batch size for ASAP7 Stage-Aware (default: 16)')
    # TSMC parameters
    parser.add_argument('--tsmc_fg_cell_innerdiv', type=int, default=10,
                       help='Inner division for TSMC Full-Graph cell delay (default: 10)')
    parser.add_argument('--tsmc_fg_cell_meta', type=int, default=16,
                       help='Meta batch size for TSMC Full-Graph cell delay (default: 16)')
    parser.add_argument('--tsmc_fg_tran_innerdiv', type=int, default=100,
                       help='Inner division for TSMC Full-Graph transition (default: 100)')
    parser.add_argument('--tsmc_fg_tran_meta', type=int, default=32,
                       help='Meta batch size for TSMC Full-Graph transition (default: 32)')
    parser.add_argument('--tsmc_sa_innerdiv', type=int, default=10,
                       help='Inner division for TSMC Stage-Aware (default: 10)')
    parser.add_argument('--tsmc_sa_meta', type=int, default=16,
                       help='Meta batch size for TSMC Stage-Aware (default: 16)')
    # Iteration filter
    parser.add_argument('--iteration', type=int, default=300000,
                       help='Filter by iteration number (default: 300000)')

    args = parser.parse_args()

    print("=" * 80)
    print("GCN CELL-BASED vs CURRENT-PATH BASED GRAPH COMPARISON (MAML only)")
    print("=" * 80)

    # Determine filter values
    vdd_only_filter = True if args.vdd_only else None
    relpin_filter = True if args.relpin else None

    # Build output suffix based on filters
    output_suffix = ""
    if args.vdd_only or args.relpin:
        filter_desc = []
        if args.vdd_only:
            filter_desc.append("vdd_only")
            output_suffix += "_vddonly"
        if args.relpin:
            filter_desc.append("relpin")
            output_suffix += "_relpin"
        print(f"Filtering for: {', '.join(filter_desc)}")

    print(f"Iteration filter: {args.iteration}")
    print(f"ASAP7 FG cell: innerdiv={args.asap7_fg_cell_innerdiv}, meta={args.asap7_fg_cell_meta}")
    print(f"ASAP7 FG tran: innerdiv={args.asap7_fg_tran_innerdiv}, meta={args.asap7_fg_tran_meta}")
    print(f"ASAP7 SA: innerdiv={args.asap7_sa_innerdiv}, meta={args.asap7_sa_meta}")
    print(f"TSMC FG cell: innerdiv={args.tsmc_fg_cell_innerdiv}, meta={args.tsmc_fg_cell_meta}")
    print(f"TSMC FG tran: innerdiv={args.tsmc_fg_tran_innerdiv}, meta={args.tsmc_fg_tran_meta}")
    print(f"TSMC SA: innerdiv={args.tsmc_sa_innerdiv}, meta={args.tsmc_sa_meta}")

    # Print cell filter if set
    if CELL_FILTER:
        print(f"\nCell filter active (ASAP7 only): {CELL_FILTER}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results separately for each PDK and graph mode (MAML only)
    print("\nLoading results for each PDK/graph mode combination...")

    # ASAP7 Full-Graph (load cell and transition separately with different innerdiv/meta)
    print("  Loading ASAP7 Full-Graph MAML (cell)...")
    asap7_fg_cell_df = load_results(args.data_dir, prefix_filter='ASAP7',
                               model_filter='GCN', training_filter='maml',
                               graph_mode_filter='full_graph',
                               vdd_only_filter=vdd_only_filter, relpin_filter=relpin_filter,
                               output_suffix_filter=False,
                               innerdiv_filter=args.asap7_fg_cell_innerdiv, meta_filter=args.asap7_fg_cell_meta,
                               iteration_filter=args.iteration, cells_filter=CELL_FILTER, verbose=True)

    print("  Loading ASAP7 Full-Graph MAML (transition)...")
    asap7_fg_tran_df = load_results(args.data_dir, prefix_filter='ASAP7',
                               model_filter='GCN', training_filter='maml',
                               graph_mode_filter='full_graph',
                               vdd_only_filter=vdd_only_filter, relpin_filter=relpin_filter,
                               output_suffix_filter=False,
                               innerdiv_filter=args.asap7_fg_tran_innerdiv, meta_filter=args.asap7_fg_tran_meta,
                               iteration_filter=args.iteration, cells_filter=CELL_FILTER, verbose=True)
    if asap7_fg_tran_df is not None and len(asap7_fg_tran_df) > 0:
        asap7_fg_tran_df = asap7_fg_tran_df[asap7_fg_tran_df['data_type'] == 'transition'].copy()

    if asap7_fg_cell_df is not None and len(asap7_fg_cell_df) > 0:
        asap7_fg_cell_df = asap7_fg_cell_df[asap7_fg_cell_df['data_type'] == 'cell'].copy()

    asap7_fg_dfs = [df for df in [asap7_fg_cell_df, asap7_fg_tran_df] if df is not None and len(df) > 0]
    asap7_fg_df = pd.concat(asap7_fg_dfs, ignore_index=True) if asap7_fg_dfs else None

    # ASAP7 Stage-Aware
    print("  Loading ASAP7 Stage-Aware MAML...")
    asap7_sa_df = load_results(args.data_dir, prefix_filter='ASAP7',
                               model_filter='GCN', training_filter='maml',
                               graph_mode_filter='stage_aware',
                               vdd_only_filter=None, relpin_filter=None,
                               output_suffix_filter=None,
                               innerdiv_filter=args.asap7_sa_innerdiv, meta_filter=args.asap7_sa_meta,
                               iteration_filter=args.iteration, cells_filter=CELL_FILTER, verbose=True)

    # TSMC Full-Graph (load cell and transition separately with different innerdiv/meta)
    print("  Loading TSMC Full-Graph MAML (cell)...")
    tsmc_fg_cell_df = load_results(args.data_dir, prefix_filter='TSMC',
                              model_filter='GCN', training_filter='maml',
                              graph_mode_filter='full_graph',
                              vdd_only_filter=vdd_only_filter, relpin_filter=relpin_filter,
                              output_suffix_filter=False,
                              innerdiv_filter=args.tsmc_fg_cell_innerdiv, meta_filter=args.tsmc_fg_cell_meta,
                              iteration_filter=args.iteration, cells_filter=None)

    print("  Loading TSMC Full-Graph MAML (transition)...")
    tsmc_fg_tran_df = load_results(args.data_dir, prefix_filter='TSMC',
                              model_filter='GCN', training_filter='maml',
                              graph_mode_filter='full_graph',
                              vdd_only_filter=vdd_only_filter, relpin_filter=relpin_filter,
                              output_suffix_filter=False,
                              innerdiv_filter=args.tsmc_fg_tran_innerdiv, meta_filter=args.tsmc_fg_tran_meta,
                              iteration_filter=args.iteration, cells_filter=None)
    if tsmc_fg_tran_df is not None and len(tsmc_fg_tran_df) > 0:
        tsmc_fg_tran_df = tsmc_fg_tran_df[tsmc_fg_tran_df['data_type'] == 'transition'].copy()

    if tsmc_fg_cell_df is not None and len(tsmc_fg_cell_df) > 0:
        tsmc_fg_cell_df = tsmc_fg_cell_df[tsmc_fg_cell_df['data_type'] == 'cell'].copy()

    tsmc_fg_dfs = [df for df in [tsmc_fg_cell_df, tsmc_fg_tran_df] if df is not None and len(df) > 0]
    tsmc_fg_df = pd.concat(tsmc_fg_dfs, ignore_index=True) if tsmc_fg_dfs else None

    # TSMC Stage-Aware
    print("  Loading TSMC Stage-Aware MAML...")
    tsmc_sa_df = load_results(args.data_dir, prefix_filter='TSMC',
                              model_filter='GCN', training_filter='maml',
                              graph_mode_filter='stage_aware',
                              vdd_only_filter=None, relpin_filter=None,
                              output_suffix_filter=None,
                              innerdiv_filter=args.tsmc_sa_innerdiv, meta_filter=args.tsmc_sa_meta,
                              iteration_filter=args.iteration, cells_filter=None)

    # Print summary
    print("\n" + "=" * 60)
    print("LOADED DATA SUMMARY")
    print("=" * 60)
    print(f"ASAP7 Full-Graph (combined): {len(asap7_fg_df) if asap7_fg_df is not None else 0} entries")
    print(f"ASAP7 Stage-Aware: {len(asap7_sa_df) if asap7_sa_df is not None else 0} entries")
    print(f"TSMC Full-Graph (combined): {len(tsmc_fg_df) if tsmc_fg_df is not None else 0} entries")
    print(f"TSMC Stage-Aware: {len(tsmc_sa_df) if tsmc_sa_df is not None else 0} entries")
    print("=" * 60)

    # Helper function to aggregate data (MAML only)
    def aggregate_data(mode_filter, exp_filter):
        """Aggregate data with given filters (MAML only)"""
        data = {}

        pdk_dfs = {
            'ASAP7': {'fg': asap7_fg_df, 'sa': asap7_sa_df},
            'TSMC': {'fg': tsmc_fg_df, 'sa': tsmc_sa_df}
        }

        for prefix in ['ASAP7', 'TSMC']:
            fg_df = pdk_dfs[prefix]['fg']
            sa_df = pdk_dfs[prefix]['sa']

            for dtype in ['cell', 'transition']:
                key = f"{prefix}_{dtype}"
                data[key] = {}

                training_type = 'maml'
                fg_expected_output_suffix = False
                sa_expected_output_suffix = (dtype == 'transition')

                # Full-Graph
                fg_nrmse = np.nan
                fg_rmse = np.nan
                if fg_df is not None:
                    filtered = fg_df[
                        (fg_df['data_type'] == dtype) &
                        (fg_df['training_type'] == training_type) &
                        (fg_df['output_suffix'] == fg_expected_output_suffix)
                    ]
                    if mode_filter:
                        filtered = filtered[filtered['mode'] == mode_filter]
                    if exp_filter:
                        filtered = filtered[filtered['experiment'] == exp_filter]
                    if len(filtered) > 0:
                        fg_nrmse = filtered['NRMSE'].mean()
                        fg_rmse = filtered['RMSE'].mean()

                # Stage-Aware
                sa_nrmse = np.nan
                sa_rmse = np.nan
                if sa_df is not None:
                    filtered = sa_df[
                        (sa_df['data_type'] == dtype) &
                        (sa_df['training_type'] == training_type) &
                        (sa_df['output_suffix'] == sa_expected_output_suffix)
                    ]
                    if mode_filter:
                        filtered = filtered[filtered['mode'] == mode_filter]
                    if exp_filter:
                        filtered = filtered[filtered['experiment'] == exp_filter]
                    if len(filtered) > 0:
                        sa_nrmse = filtered['NRMSE'].mean()
                        sa_rmse = filtered['RMSE'].mean()

                if not np.isnan(fg_nrmse) or not np.isnan(sa_nrmse):
                    data[key]['maml'] = {
                        'full_graph': fg_nrmse,
                        'stage_aware': sa_nrmse,
                        'full_graph_rmse': fg_rmse,
                        'stage_aware_rmse': sa_rmse
                    }

        return data

    # Collect data for both modes
    extra_data = aggregate_data('extrapolation', args.experiment)
    inter_data = aggregate_data('interpolation', args.experiment)

    # Prepare data structure for PDK plots
    def prepare_pdk_data(prefix):
        cell_key = f"{prefix}_cell"
        trans_key = f"{prefix}_transition"

        cell_data = {
            'extra': extra_data.get(cell_key, {}).get('maml', {}),
            'inter': inter_data.get(cell_key, {}).get('maml', {})
        }
        trans_data = {
            'extra': extra_data.get(trans_key, {}).get('maml', {}),
            'inter': inter_data.get(trans_key, {}).get('maml', {})
        }
        return cell_data, trans_data

    # Generate PDK combined plot
    print(f"\nGenerating PDK combined FG vs SA plot...")

    asap7_cell, asap7_trans = prepare_pdk_data('ASAP7')
    tsmc_cell, tsmc_trans = prepare_pdk_data('TSMC')

    output_path = os.path.join(args.output_dir, f'pdk_combined_fg_vs_sa{output_suffix}.png')
    plot_fg_vs_sa_pdk_combined(asap7_cell, asap7_trans, tsmc_cell, tsmc_trans, output_path, metric='NRMSE')

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY (MAML)")
    print("=" * 80)

    all_data = aggregate_data(args.mode, args.experiment)

    for key, data in all_data.items():
        if data and 'maml' in data:
            values = data['maml']
            fg_nrmse = values.get('full_graph', np.nan)
            sa_nrmse = values.get('stage_aware', np.nan)

            print(f"\n{key}:")
            if not np.isnan(fg_nrmse) and not np.isnan(sa_nrmse) and sa_nrmse > 0:
                ratio = fg_nrmse / sa_nrmse
                print(f"  NRMSE: Cell={fg_nrmse:.2f}%, Path={sa_nrmse:.2f}%, Ratio={ratio:.2f}x")
            elif not np.isnan(fg_nrmse):
                print(f"  NRMSE: Cell={fg_nrmse:.2f}%, Path=N/A")
            elif not np.isnan(sa_nrmse):
                print(f"  NRMSE: Cell=N/A, Path={sa_nrmse:.2f}%")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
