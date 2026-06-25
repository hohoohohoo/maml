#!/usr/bin/env python3
"""
Analyze GCN results for a SINGLE architecture - compare with MLP baselines

This script is a simplified version of analyze_gcn_sweep_results.py,
designed to analyze results from a specific GCN architecture stored in
the final results directory.

Features:
- Analyze single GCN architecture results
- Compare with MLP MAML/AADAM baselines
- Generate comparison plots (per-cell and aggregated)
- Export results to CSV
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# CELL_FILTER = ["AND2x6", "NAND3x2", "NOR2xp67", "OR2x6","OAI22x1", "AO21x1", "AO32x1", "FAx1","HAxp5","XNOR2x1","XNOR2x2","XNOR2xp5","XOR2x1","XOR2x2","XOR2xp5"]  # Topology-agnostic cells
CELL_FILTER = ["AND2x6", "NAND3x2", "NOR2xp67", "OR2x6", "AO21x1", "AO32x1","OAI22x1", "FAx1","HAxp5","XNOR2x2","XOR2x2"]  # Topology-agnostic cells

# ============================================================================

# Set matplotlib style
plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def draw_broken_axis_bars(ax, x_positions, bar_data, model_order, model_colors, modes,
                          bar_width, group_gap, break_threshold=3.0, upper_range_ratio=0.3):
    """Draw bars with broken axis effect for handling outliers (like GCN_baseline).

    Args:
        ax: matplotlib axis
        x_positions: x positions for each cell
        bar_data: dict of {model: {mode: [values]}}
        model_order: list of model names
        model_colors: dict of {model: (light_color, dark_color)}
        modes: list of modes ['interpolation', 'extrapolation']
        bar_width: width of each bar
        group_gap: gap between extrapolation and interpolation groups
        break_threshold: y-value where axis breaks (default 3.0%)
        upper_range_ratio: ratio of figure height for upper broken section

    Returns:
        legend_handles, legend_labels, y_max_lower, y_max_upper
    """
    # Collect all values to determine ranges
    all_values = []
    for model in model_order:
        for mode in modes:
            all_values.extend([v for v in bar_data[model][mode] if not np.isnan(v)])

    if not all_values:
        return [], [], 0, 0

    max_val = max(all_values)

    # Check if we need broken axis (if max value significantly exceeds threshold)
    needs_break = max_val > break_threshold * 1.5

    legend_handles = []
    legend_labels = []

    extra_group_center = -group_gap / 2 - len(model_order) * bar_width / 2
    inter_group_center = group_gap / 2 + len(model_order) * bar_width / 2

    if needs_break:
        # Calculate ranges
        lower_max = break_threshold
        upper_min = max_val * 0.85  # Start upper axis slightly below max
        upper_max = max_val * 1.1

        # Draw bars with clipping for lower section
        for model_idx, model in enumerate(model_order):
            light_color, dark_color = model_colors[model]
            model_offset = (model_idx - 1.5) * bar_width

            extra_values = bar_data[model]['extrapolation']
            inter_values = bar_data[model]['interpolation']

            # Clip values for display in lower section
            extra_clipped = [min(v, lower_max) if not np.isnan(v) else np.nan for v in extra_values]
            inter_clipped = [min(v, lower_max) if not np.isnan(v) else np.nan for v in inter_values]

            bars_extra = ax.bar(x_positions + extra_group_center + model_offset, extra_clipped, bar_width,
                               color=dark_color, edgecolor='black', linewidth=0.5)
            bars_inter = ax.bar(x_positions + inter_group_center + model_offset, inter_clipped, bar_width,
                               color=light_color, edgecolor='black', linewidth=0.5)

            # Add text annotations for values exceeding threshold
            for i, (v_extra, v_inter) in enumerate(zip(extra_values, inter_values)):
                if not np.isnan(v_extra) and v_extra > lower_max:
                    ax.annotate(f'{v_extra:.1f}',
                               xy=(x_positions[i] + extra_group_center + model_offset, lower_max),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=6, fontweight='bold',
                               color=dark_color,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=dark_color, alpha=0.8))
                if not np.isnan(v_inter) and v_inter > lower_max:
                    ax.annotate(f'{v_inter:.1f}',
                               xy=(x_positions[i] + inter_group_center + model_offset, lower_max),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=6, fontweight='bold',
                               color=light_color,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=light_color, alpha=0.8))

            model_label = model.replace('_', ' ')
            legend_handles.append(bars_extra)
            legend_labels.append(f'{model_label} (Extra)')
            legend_handles.append(bars_inter)
            legend_labels.append(f'{model_label} (Inter)')

        # Set y-axis limit and add break indicator
        ax.set_ylim(0, lower_max * 1.15)

        # Add break lines at top of axis
        d = 0.015  # Size of diagonal lines
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        return legend_handles, legend_labels, lower_max, max_val

    else:
        # Normal plotting without break
        for model_idx, model in enumerate(model_order):
            light_color, dark_color = model_colors[model]
            model_offset = (model_idx - 1.5) * bar_width

            extra_values = bar_data[model]['extrapolation']
            inter_values = bar_data[model]['interpolation']

            bars_extra = ax.bar(x_positions + extra_group_center + model_offset, extra_values, bar_width,
                               color=dark_color, edgecolor='black', linewidth=0.5)
            bars_inter = ax.bar(x_positions + inter_group_center + model_offset, inter_values, bar_width,
                               color=light_color, edgecolor='black', linewidth=0.5)

            model_label = model.replace('_', ' ')
            legend_handles.append(bars_extra)
            legend_labels.append(f'{model_label} (Extra)')
            legend_handles.append(bars_inter)
            legend_labels.append(f'{model_label} (Inter)')

        y_max = max_val * 1.15
        ax.set_ylim(0, y_max)

        return legend_handles, legend_labels, y_max, max_val


def draw_broken_axis_bars_cell_trans(ax, x_positions, bar_data, model_order, model_colors, data_types,
                                     bar_width, group_gap, break_threshold=3.0):
    """Draw bars as one 8-bar group per cell, following data_types order.
    First data_type is solid, second is hatched.

    Args:
        ax: matplotlib axis
        x_positions: x positions for each cell
        bar_data: dict of {model: {data_type: [values]}}
        model_order: list of model names
        model_colors: dict of {model: color}
        data_types: list of data types (first=solid, second=hatched)
        bar_width: width of each bar
        group_gap: gap between groups (unused in this layout)
        break_threshold: y-value where axis breaks (default 3.0%)

    Returns:
        legend_handles, legend_labels, y_max_lower, y_max_upper
    """
    # Data type labels for legend
    dtype_labels = {'cell': 'Cell', 'transition': 'Trans'}

    all_values = []
    for model in model_order:
        for dtype in data_types:
            if dtype in bar_data[model]:
                all_values.extend([v for v in bar_data[model][dtype] if not np.isnan(v)])

    if not all_values:
        return [], [], 0, 0

    max_val = max(all_values)
    needs_break = max_val > break_threshold * 1.5

    legend_handles = []
    legend_labels = []

    n_bars_per_group = len(model_order) * 2   # 8
    offsets = (np.arange(n_bars_per_group) - (n_bars_per_group - 1) / 2) * bar_width

    # First data_type = solid, second data_type = hatched
    dt_first = data_types[0]   # solid bar
    dt_second = data_types[1]  # hatched bar

    if needs_break:
        lower_max = break_threshold

        for model_idx, model in enumerate(model_order):
            color = model_colors[model]

            first_values = bar_data[model].get(dt_first, [np.nan] * len(x_positions))
            second_values = bar_data[model].get(dt_second, [np.nan] * len(x_positions))

            first_clipped = [min(v, lower_max) if not np.isnan(v) else np.nan for v in first_values]
            second_clipped = [min(v, lower_max) if not np.isnan(v) else np.nan for v in second_values]

            first_offset = offsets[2 * model_idx]
            second_offset = offsets[2 * model_idx + 1]

            bars_first = ax.bar(
                x_positions + first_offset, first_clipped, bar_width,
                color=color, edgecolor='black', linewidth=0.5
            )
            bars_second = ax.bar(
                x_positions + second_offset, second_clipped, bar_width,
                color=color, edgecolor='white', linewidth=0.5, hatch='//'
            )

            # Annotate over-threshold values
            # First bar (transition): left side of bar, text extends left
            for i, v in enumerate(first_values):
                if not np.isnan(v) and v > lower_max:
                    ax.annotate(
                        f'{v:.1f}',
                        xy=(x_positions[i] + first_offset - bar_width*0.5, lower_max),
                        xytext=(-2, 3), textcoords='offset points',
                        ha='right', va='bottom', fontsize=11, fontweight='bold',
                        color='black',
                        bbox=dict(
                            boxstyle='round,pad=0.15',
                            facecolor='white',
                            edgecolor='black',
                            linewidth=0.8,
                            alpha=0.95
                        )
                    )

            # Second bar (cell): right side of bar, text extends right
            for i, v in enumerate(second_values):
                if not np.isnan(v) and v > lower_max:
                    ax.annotate(
                        f'{v:.1f}',
                        xy=(x_positions[i] + first_offset - bar_width*0.5, lower_max),
                        xytext=(2, 3), textcoords='offset points',
                        ha='right', va='bottom', fontsize=11, fontweight='bold',
                        color='black',
                        bbox=dict(
                            boxstyle='round,pad=0.15',
                            facecolor='white',
                            edgecolor='black',
                            linewidth=0.8,
                            alpha=0.95
                        )
                    )

            model_label = model.replace('_', ' ')
            legend_handles.append(bars_first)
            legend_labels.append(f'{model_label} ({dtype_labels[dt_first]})')
            legend_handles.append(bars_second)
            legend_labels.append(f'{model_label} ({dtype_labels[dt_second]})')

        ax.set_ylim(0, lower_max * 1.15)

        # Add break indicator on y-axis (diagonal lines below threshold)
        y_break = lower_max * 0.97
        d = 0.015  # size of diagonal lines
        kwargs = dict(transform=ax.get_yaxis_transform(), color='k', clip_on=False, linewidth=1.5)
        ax.plot((-d, +d), (y_break - 0.05, y_break + 0.05), **kwargs)
        ax.plot((-d, +d), (y_break - 0.1, y_break), **kwargs)

        return legend_handles, legend_labels, lower_max, max_val

    else:
        for model_idx, model in enumerate(model_order):
            color = model_colors[model]

            first_values = bar_data[model].get(dt_first, [np.nan] * len(x_positions))
            second_values = bar_data[model].get(dt_second, [np.nan] * len(x_positions))

            first_offset = offsets[2 * model_idx]
            second_offset = offsets[2 * model_idx + 1]

            bars_first = ax.bar(
                x_positions + first_offset, first_values, bar_width,
                color=color, edgecolor='black', linewidth=0.5
            )
            bars_second = ax.bar(
                x_positions + second_offset, second_values, bar_width,
                color=color, edgecolor='white', linewidth=0.5, hatch='//'
            )

            model_label = model.replace('_', ' ')
            legend_handles.append(bars_first)
            legend_labels.append(f'{model_label} ({dtype_labels[dt_first]})')
            legend_handles.append(bars_second)
            legend_labels.append(f'{model_label} ({dtype_labels[dt_second]})')

        y_max = max_val * 1.15
        ax.set_ylim(0, y_max)
        return legend_handles, legend_labels, y_max, max_val


def parse_gcn_filename(filename, gcn_adapt_suffix=''):
    """Parse GCN result filename to extract metadata

    Args:
        filename: Path to the file
        gcn_adapt_suffix: Optional suffix for GCN files (e.g., '_adam' for adam adaptation method)
    """
    basename = os.path.basename(filename)

    # Extract architecture: convXxY_fcAxB
    arch_pattern = r'conv(\d+)x(\d+)_fc(\d+)x(\d+)'
    arch_match = re.search(arch_pattern, basename)

    if not arch_match:
        return None

    conv_hidden_dim = int(arch_match.group(1))
    num_conv_layers = int(arch_match.group(2))
    fc_hidden_dim = int(arch_match.group(3))
    num_fc_layers = int(arch_match.group(4))

    # Determine file type (pred or act) with optional adapt suffix
    # When gcn_adapt_suffix is specified, check that suffix exists anywhere in filename
    # and that file ends with _pred.npy or _act.npy
    if gcn_adapt_suffix:
        # Check suffix exists in filename AND file ends with _pred.npy or _act.npy
        if gcn_adapt_suffix not in basename:
            return None
        if basename.endswith('_pred.npy'):
            file_type = 'pred'
        elif basename.endswith('_act.npy'):
            file_type = 'act'
        else:
            return None
    else:
        if basename.endswith('_pred.npy'):
            file_type = 'pred'
        elif basename.endswith('_act.npy'):
            file_type = 'act'
        else:
            return None

    result = {
        'conv_hidden_dim': conv_hidden_dim,
        'num_conv_layers': num_conv_layers,
        'fc_hidden_dim': fc_hidden_dim,
        'num_fc_layers': num_fc_layers,
        'file_type': file_type,
        'filename': basename,
        'arch_string': f'conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}'
    }

    # Determine prefix (ASAP7 or TSMC)
    if basename.startswith('ASAP7'):
        result['prefix'] = 'ASAP7'
    elif basename.startswith('TSMC'):
        result['prefix'] = 'TSMC'
    else:
        result['prefix'] = 'unknown'

    # Extract model type (GCN or GAT)
    if '_GCN_' in basename:
        result['gnn_model_type'] = 'GCN'
    elif '_GAT_' in basename:
        result['gnn_model_type'] = 'GAT'
    else:
        result['gnn_model_type'] = 'unknown'

    # Extract training type (maml or baseline)
    if '_maml_' in basename:
        result['training_type'] = 'maml'
    elif '_baseline_' in basename:
        result['training_type'] = 'baseline'
    else:
        result['training_type'] = 'unknown'

    # Extract experiment type
    if 'intra_topology' in basename:
        result['experiment'] = 'intra_topology'
    elif 'topology_agnostic' in basename:
        result['experiment'] = 'topology_agnostic'

    # Extract data type (cell or transition)
    if '_cell_' in basename:
        result['data_type'] = 'cell'
        # Extract cell name
        cell_pattern = r'(?:intra_topology|topology_agnostic)_(\w+)_cell_'
        cell_match = re.search(cell_pattern, basename)
        if cell_match:
            result['cell'] = cell_match.group(1)
    elif '_transition_' in basename:
        result['data_type'] = 'transition'
        # Extract cell name
        cell_pattern = r'(?:intra_topology|topology_agnostic)_(\w+)_transition_'
        cell_match = re.search(cell_pattern, basename)
        if cell_match:
            result['cell'] = cell_match.group(1)

    # Extract graph mode
    if 'stage_aware' in basename:
        result['graph_mode'] = 'stage_aware'
    elif 'full_graph' in basename:
        result['graph_mode'] = 'full_graph'

    # Extract mode
    if '_interpolation_' in basename:
        result['mode'] = 'interpolation'
    elif '_extrapolation_' in basename:
        result['mode'] = 'extrapolation'

    # Extract iteration
    iter_match = re.search(r'iter(\d+)', basename)
    if iter_match:
        result['iterations'] = int(iter_match.group(1))

    # Extract innerdiv
    innerdiv_match = re.search(r'innerdiv(\d+)', basename)
    if innerdiv_match:
        result['innerdiv'] = int(innerdiv_match.group(1))

    # Extract meta
    meta_match = re.search(r'meta(\d+)', basename)
    if meta_match:
        result['meta'] = int(meta_match.group(1))

    # Extract pooling mode
    pool_match = re.search(r'_pool(output|max|add|mean)', basename)
    if pool_match:
        result['pooling'] = pool_match.group(1)
    else:
        result['pooling'] = 'mean'  # default

    # Check if parasitic file
    result['is_parasitic'] = 'parasitic' in basename.lower()

    # Check for vdd_only and relpin suffixes
    result['vdd_only'] = '_vddonly' in basename.lower() or '_vdd_only' in basename.lower()
    result['relpin'] = '_relpin' in basename.lower() or '_rel_pin' in basename.lower()

    return result


def parse_mlp_filename(filename, aadam_adapt_suffix=''):
    """Parse MLP result filename to extract metadata

    Args:
        filename: Path to the file
        aadam_adapt_suffix: Optional suffix for aadam files (e.g., '_adam' for adam adaptation method)
    """
    basename = os.path.basename(filename)

    # Pattern 1: ASAP7_intra_topology_{cell}_{data_type}_{mode}_MAML_...
    # Pattern 2: ASAP7_topology_agnostic_{cell}_{data_type}_{mode}_MAML_...
    # Note: _layer(\d+)_ is REQUIRED to match analyze_gcn_sweep_results.py behavior
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

    # Pattern 3: ASAP7_intra_topology_{cell}_{data_type}_{mode}_aadam_{iter}[_adam]_{pred|act}.npy
    # aadam_adapt_suffix allows matching files with _adam suffix (for adam adaptation method)
    aadam_pattern1 = r'(\w+)_(intra_topology|topology_agnostic)_(\w+)_(cell|transition)_(extrapolation|interpolation)_(aadam|mlp)_(\d+)' + aadam_adapt_suffix + r'_(pred|act)\.npy'
    match = re.match(aadam_pattern1, basename)

    if match:
        return {
            'prefix': match.group(1),
            'topology': match.group(2),
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': match.group(6).upper(),
            'iterations': int(match.group(7)),
            'file_type': match.group(8),
            'filename': basename
        }

    # Pattern 3b: Shortened naming - TSMC_agnostic_{cell}_{data_type}_{mode}_aadam_{iter}[_adam]_{pred|act}.npy
    # TSMC_intra_{cell}_{data_type}_{mode}_aadam_{iter}[_adam]_{pred|act}.npy
    aadam_pattern1_short = r'(\w+)_(intra|agnostic)_(\w+)_(cell|transition)_(extrapolation|interpolation)_(aadam|mlp)_(\d+)' + aadam_adapt_suffix + r'_(pred|act)\.npy'
    match = re.match(aadam_pattern1_short, basename)

    if match:
        # Map shortened names to full names
        topo_map = {'intra': 'intra_topology', 'agnostic': 'topology_agnostic'}
        return {
            'prefix': match.group(1),
            'topology': topo_map.get(match.group(2), match.group(2)),
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': match.group(6).upper(),
            'iterations': int(match.group(7)),
            'file_type': match.group(8),
            'filename': basename
        }

    # Legacy patterns with data_type in middle
    # Note: _layer(\d+)_ is REQUIRED to match analyze_gcn_sweep_results.py behavior
    maml_pattern2 = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_MAML_innerdiv(\d+)_meta(\d+)_layer(\d+)_(\d+)_(pred|act)\.npy'
    match = re.match(maml_pattern2, basename)

    if match:
        # Map shortened topology names to full names
        topo_map = {'intra': 'intra_topology', 'agnostic': 'topology_agnostic'}
        topology_raw = match.group(2)
        return {
            'prefix': match.group(1),
            'topology': topo_map.get(topology_raw, topology_raw),
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

    aadam_pattern2 = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_(aadam|mlp)_(\d+)' + aadam_adapt_suffix + r'_(pred|act)\.npy'
    match = re.match(aadam_pattern2, basename)

    if match:
        # Map shortened topology names to full names
        topo_map = {'intra': 'intra_topology', 'agnostic': 'topology_agnostic'}
        topology_raw = match.group(2)
        return {
            'prefix': match.group(1),
            'topology': topo_map.get(topology_raw, topology_raw),
            'cell': match.group(3),
            'data_type': match.group(4),
            'mode': match.group(5),
            'model_type': match.group(6).upper(),
            'iterations': int(match.group(7)),
            'file_type': match.group(8),
            'filename': basename
        }

    return None


def calculate_metrics(predictions, actuals, group_size=61):
    """Calculate NRMSE, RMSE, SMAPE, and MAE metrics with group averaging"""
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Filter out invalid values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals) |
                   np.isinf(predictions) | np.isinf(actuals))
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]

    if len(predictions) == 0:
        return None

    # Group by group_size samples
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

    # RMSE per group
    mse_groups = np.mean((pred_grouped - act_grouped) ** 2, axis=1)
    rmse_groups = np.sqrt(mse_groups)

    # NRMSE per group (range normalization)
    y_ranges = np.max(act_grouped, axis=1) - np.min(act_grouped, axis=1)
    y_ranges = np.where(y_ranges > 0, y_ranges, 1.0)
    nrmse_groups = (rmse_groups / y_ranges) * 100

    # MAE per group
    mae_groups = np.mean(np.abs(pred_grouped - act_grouped), axis=1)

    # SMAPE per group (Symmetric MAPE - handles negative values better)
    # SMAPE = 100 * mean(2 * |pred - actual| / (|pred| + |actual| + epsilon))
    abs_diff = np.abs(pred_grouped - act_grouped)
    abs_sum = np.abs(pred_grouped) + np.abs(act_grouped)
    abs_sum = np.where(abs_sum > 1e-8, abs_sum, 1e-8)  # avoid division by zero
    smape_groups = np.mean(2 * abs_diff / abs_sum, axis=1) * 100

    return {
        'NRMSE': float(np.mean(nrmse_groups)),
        'RMSE': float(np.mean(rmse_groups)),
        'SMAPE': float(np.mean(smape_groups)),
        'MAE': float(np.mean(mae_groups)),
        'num_samples': len(predictions),
        'num_groups': n_groups,
        'actuals_mean': float(np.mean(actuals)),
        'actuals_std': float(np.std(actuals)),
        'actuals_range': float(np.max(actuals) - np.min(actuals))
    }


def load_gcn_results(data_dir, arch_filter=None, gcn_adapt_suffix='', vdd_only_filter=None, relpin_filter=None, cells_filter=None):
    """Load GCN results from directory and subdirectories, optionally filtering by architecture

    Args:
        data_dir: Directory containing GCN result files
        arch_filter: Filter by architecture string (e.g., 'conv64x2_fc256x2')
        gcn_adapt_suffix: Suffix for GCN files (e.g., '_adam' for adam adaptation method)
        vdd_only_filter: If True, only load files with vdd_only. If False, only load files without. If None, load all.
        relpin_filter: If True, only load files with relpin. If False, only load files without. If None, load all.
        cells_filter: List of cell names to include (None = all cells, ASAP7 only)
    """
    results = []

    # Search in main directory and subdirectories (same as analyze_gcn_sweep_results.py)
    search_dirs = [data_dir]
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            search_dirs.append(item_path)

    # Collect all pred files with appropriate suffix
    # Note: When gcn_adapt_suffix is specified (e.g., '_adam'), we need to find files that contain
    # the suffix anywhere before _pred.npy, not just immediately before it.
    # This handles patterns like: ..._adam_vddonly_relpin_pred.npy or ..._vddonly_relpin_adam_pred.npy
    if gcn_adapt_suffix:
        pred_glob_pattern = f'*{gcn_adapt_suffix}*_pred.npy'
    else:
        pred_glob_pattern = '*_pred.npy'

    all_pred_files = []
    for search_dir in search_dirs:
        pred_files = glob.glob(os.path.join(search_dir, pred_glob_pattern))
        all_pred_files.extend(pred_files)

    total_files = len(all_pred_files)
    print(f"Found {total_files} prediction files (suffix: '{gcn_adapt_suffix if gcn_adapt_suffix else 'none'}')...")

    for i, pred_file in enumerate(all_pred_files):
        if (i + 1) % 50 == 0:
            print(f"  Processing: {i + 1}/{total_files}", end='\r')

        # Replace _pred.npy with _act.npy to find corresponding act file
        # This handles all suffix patterns (e.g., _adam_vddonly_relpin_pred.npy -> _adam_vddonly_relpin_act.npy)
        act_file = pred_file.replace('_pred.npy', '_act.npy')

        if not os.path.exists(act_file):
            continue

        metadata = parse_gcn_filename(pred_file, gcn_adapt_suffix=gcn_adapt_suffix)
        if metadata is None:
            continue

        # Filter by architecture if specified
        if arch_filter and metadata['arch_string'] != arch_filter:
            continue

        # Filter by vdd_only if specified
        if vdd_only_filter is not None and metadata.get('vdd_only', False) != vdd_only_filter:
            continue

        # Filter by relpin if specified
        if relpin_filter is not None and metadata.get('relpin', False) != relpin_filter:
            continue

        # Apply cell name filter (only for ASAP7)
        if cells_filter is not None and metadata.get('prefix') == 'ASAP7':
            if metadata.get('cell') not in cells_filter:
                continue

        # Skip empty files
        if os.path.getsize(pred_file) == 0:
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

    print(f"\nLoaded {len(results)} GCN results")
    return pd.DataFrame(results) if results else None


def load_mlp_results(mlp_maml_dir, aadam_dir=None, prefix_filter=None, experiment_filter=None, data_type_filter=None, aadam_adapt_suffix='', cells_filter=None):
    """Load MLP results from separate directories for MLP MAML and AADAM

    Args:
        mlp_maml_dir: Directory containing MLP MAML .npy result files
        aadam_dir: Directory containing AADAM .npy result files (if None, uses mlp_maml_dir)
        prefix_filter: Filter by prefix (e.g., 'ASAP7', 'TSMC')
        experiment_filter: Filter by experiment type
        data_type_filter: Filter by data type
        aadam_adapt_suffix: Suffix for aadam files (e.g., '_adam' for adam adaptation method)
        cells_filter: List of cell names to include (None = all cells, ASAP7 only)
    """
    results = []

    # If aadam_dir is not specified, use mlp_maml_dir for both
    if aadam_dir is None:
        aadam_dir = mlp_maml_dir

    # Load from MLP MAML directory - only MLP_MAML files (exclude aadam files)
    all_files_maml = glob.glob(os.path.join(mlp_maml_dir, '*_pred.npy'))
    pred_files_maml = [f for f in all_files_maml if 'aadam_' not in os.path.basename(f).lower()]
    print(f"Found {len(pred_files_maml)} MLP_MAML prediction files in mlp_maml_dir (excluded {len(all_files_maml) - len(pred_files_maml)} aadam files)...")

    # Load from AADAM directory - only AADAM files
    if aadam_dir != mlp_maml_dir:
        all_files_aadam = glob.glob(os.path.join(aadam_dir, '*_pred.npy'))
        pred_files_aadam = [f for f in all_files_aadam if 'aadam_' in os.path.basename(f).lower()]
        print(f"Found {len(pred_files_aadam)} AADAM prediction files in aadam_dir...")
        pred_files = pred_files_maml + pred_files_aadam
    else:
        pred_files = pred_files_maml
    print(f"Total {len(pred_files)} prediction files to process...")

    for pred_file in pred_files:
        act_file = pred_file.replace('_pred.npy', '_act.npy')
        if not os.path.exists(act_file):
            continue

        metadata = parse_mlp_filename(pred_file, aadam_adapt_suffix=aadam_adapt_suffix)
        if metadata is None:
            continue

        # Filter by prefix if specified
        if prefix_filter and metadata.get('prefix') != prefix_filter:
            continue

        # Filter by experiment (topology) if specified
        if experiment_filter and metadata.get('topology') != experiment_filter:
            continue

        # Filter by data_type if specified
        if data_type_filter and metadata.get('data_type') != data_type_filter:
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

        except Exception as e:
            continue

    print(f"Loaded {len(results)} MLP results")
    return pd.DataFrame(results) if results else None


def plot_comparison(gcn_df, mlp_df, output_dir, title_prefix, aadam_iter=300000, data_type='cell', scale_rmse=False, gcn_baseline_df=None, arch='', gcn_innerdiv=10, gcn_meta=16):
    """Generate comparison plots between GCN and MLP baselines"""

    os.makedirs(output_dir, exist_ok=True)

    # Check if we have any data to plot
    has_gcn = gcn_df is not None and len(gcn_df) > 0
    has_mlp = mlp_df is not None and len(mlp_df) > 0
    has_gcn_baseline = gcn_baseline_df is not None and len(gcn_baseline_df) > 0

    if not has_gcn and not has_mlp:
        print("No data to plot (neither GCN nor MLP)")
        return

    if not has_gcn:
        print("No GCN data - plotting MLP baselines only")
        # Use MLP data to determine combinations
        source_df = mlp_df.copy()
        source_df['experiment'] = source_df['topology']
        source_df['graph_mode'] = 'N/A'
    else:
        source_df = gcn_df

    # Data type label for title
    data_type_label = 'Cell Delay' if data_type == 'cell' else 'Transition (Slew)'

    for experiment in source_df['experiment'].dropna().unique():
        for mode in source_df['mode'].dropna().unique():
            # GCN_MAML: always stage_aware, GCN_baseline: always full_graph
            # No separate loop for graph_mode

            # Get filtered data based on what's available
            # GCN MAML always uses stage_aware
            if has_gcn:
                filtered_gcn = gcn_df[
                    (gcn_df['experiment'] == experiment) &
                    (gcn_df['mode'] == mode) &
                    (gcn_df['graph_mode'] == 'stage_aware') &
                    (gcn_df['data_type'] == data_type)
                ]
            else:
                filtered_gcn = pd.DataFrame()

            # GCN baseline always uses full_graph
            filtered_gcn_baseline = None
            if has_gcn_baseline:
                filtered_gcn_baseline = gcn_baseline_df[
                    (gcn_baseline_df['experiment'] == experiment) &
                    (gcn_baseline_df['mode'] == mode) &
                    (gcn_baseline_df['data_type'] == data_type)
                ]
                if len(filtered_gcn_baseline) == 0:
                    filtered_gcn_baseline = None

            # For MLP-only case, get cells from MLP data
            if has_gcn and len(filtered_gcn) > 0:
                cells = sorted(filtered_gcn['cell'].dropna().unique())
                arch = filtered_gcn['arch_string'].iloc[0]
            elif has_gcn_baseline and filtered_gcn_baseline is not None and len(filtered_gcn_baseline) > 0:
                cells = sorted(filtered_gcn_baseline['cell'].dropna().unique())
                arch = filtered_gcn_baseline['arch_string'].iloc[0] if 'arch_string' in filtered_gcn_baseline.columns else 'baseline'
            elif has_mlp:
                # Get cells from MLP data for this experiment/mode
                mlp_filtered = mlp_df[
                    (mlp_df['topology'] == experiment) &
                    (mlp_df['mode'] == mode)
                ]
                if len(mlp_filtered) == 0:
                    continue
                cells = sorted(mlp_filtered['cell'].dropna().unique())
                arch = 'N/A'
            else:
                continue

            if len(cells) == 0:
                continue

            # Prepare data
            gcn_nrmse = []
            gcn_rmse = []
            gcn_baseline_nrmse = []
            gcn_baseline_rmse = []
            mlp_maml_nrmse = []
            mlp_maml_rmse = []
            mlp_aadam_nrmse = []
            mlp_aadam_rmse = []

            for cell in cells:
                # GCN MAML per cell - handle empty DataFrame
                if has_gcn and len(filtered_gcn) > 0:
                    cell_gcn = filtered_gcn[filtered_gcn['cell'] == cell]
                    if len(cell_gcn) > 0:
                        gcn_nrmse.append(cell_gcn['NRMSE'].mean())
                        gcn_rmse.append(cell_gcn['RMSE'].mean())
                    else:
                        gcn_nrmse.append(np.nan)
                        gcn_rmse.append(np.nan)
                else:
                    gcn_nrmse.append(np.nan)
                    gcn_rmse.append(np.nan)

                # GCN baseline per cell
                if filtered_gcn_baseline is not None and len(filtered_gcn_baseline) > 0:
                    cell_gcn_baseline = filtered_gcn_baseline[filtered_gcn_baseline['cell'] == cell]
                    if len(cell_gcn_baseline) > 0:
                        gcn_baseline_nrmse.append(cell_gcn_baseline['NRMSE'].mean())
                        gcn_baseline_rmse.append(cell_gcn_baseline['RMSE'].mean())
                    else:
                        gcn_baseline_nrmse.append(np.nan)
                        gcn_baseline_rmse.append(np.nan)
                else:
                    gcn_baseline_nrmse.append(np.nan)
                    gcn_baseline_rmse.append(np.nan)

                # MLP baselines
                if mlp_df is not None:
                    mlp_cell = mlp_df[
                        (mlp_df['cell'] == cell) &
                        (mlp_df['mode'] == mode)
                    ]

                    # MAML - filter by specific parameters (innerdiv=100, meta=32, layer_length=40, iterations=300000)
                    maml_data = mlp_cell[mlp_cell['model_type'] == 'MLP_MAML']
                    required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                    if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                        maml_specific = maml_data[
                            (maml_data['innerdiv'] == 100) &
                            (maml_data['meta'] == 32) &
                            (maml_data['layer_length'] == 40) &
                            (maml_data['iterations'] == 300000)
                        ]
                        if len(maml_specific) > 0:
                            mlp_maml_nrmse.append(maml_specific['NRMSE'].mean())
                            mlp_maml_rmse.append(maml_specific['RMSE'].mean())
                        else:
                            mlp_maml_nrmse.append(np.nan)
                            mlp_maml_rmse.append(np.nan)
                    else:
                        mlp_maml_nrmse.append(np.nan)
                        mlp_maml_rmse.append(np.nan)

                    # AADAM
                    aadam_data = mlp_cell[mlp_cell['model_type'] == 'AADAM']
                    if aadam_iter:
                        aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
                    if len(aadam_data) > 0:
                        mlp_aadam_nrmse.append(aadam_data['NRMSE'].mean())
                        mlp_aadam_rmse.append(aadam_data['RMSE'].mean())
                    else:
                        mlp_aadam_nrmse.append(np.nan)
                        mlp_aadam_rmse.append(np.nan)
                else:
                    mlp_maml_nrmse.append(np.nan)
                    mlp_maml_rmse.append(np.nan)
                    mlp_aadam_nrmse.append(np.nan)
                    mlp_aadam_rmse.append(np.nan)

            # Create plot
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            if has_gcn and len(filtered_gcn) > 0:
                title_str = f'{title_prefix} GCN_MAML ({arch}) vs GCN_baseline vs MLP - {data_type_label} - {experiment} - {mode.upper()}'
            else:
                title_str = f'{title_prefix} MLP Baselines - {data_type_label} - {experiment} - {mode.upper()}'
            fig.suptitle(title_str, fontsize=14, fontweight='bold')

            x = np.arange(len(cells))
            # Determine number of bars and width
            has_baseline_data = any(not np.isnan(v) for v in gcn_baseline_nrmse)
            if has_baseline_data:
                width = 0.2
                offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
            else:
                width = 0.25
                offsets = [-width, 0, width, None]

            # NRMSE plot
            ax1 = axes[0]
            all_bars_nrmse = []
            bars1 = ax1.bar(x + offsets[0], gcn_nrmse, width, label=f'GCN MAML ({arch})', color='#2ecc71')
            all_bars_nrmse.append(bars1)
            bars2 = ax1.bar(x + offsets[1], mlp_maml_nrmse, width, label='MLP MAML (id100_m32_l40_i300k)', color='#3498db')
            all_bars_nrmse.append(bars2)
            bars3 = ax1.bar(x + offsets[2], mlp_aadam_nrmse, width, label=f'MLP AADAM (iter={aadam_iter})', color='#e74c3c')
            all_bars_nrmse.append(bars3)
            if has_baseline_data:
                bars4 = ax1.bar(x + offsets[3], gcn_baseline_nrmse, width, label='GCN Baseline', color='#9b59b6')
                all_bars_nrmse.append(bars4)

            ax1.set_xlabel('Cell', fontsize=11)
            ax1.set_ylabel('NRMSE (%)', fontsize=11, fontweight='bold')
            ax1.set_title('NRMSE Comparison', fontsize=12)
            ax1.set_xticks(x)
            ax1.set_xticklabels(cells, rotation=45, ha='right', fontsize=9)
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bars in all_bars_nrmse:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax1.annotate(f'{height:.3f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=6, rotation=90)

            # RMSE plot
            ax2 = axes[1]
            # Apply scaling if enabled
            if scale_rmse:
                gcn_rmse_plot = [v * 1000 if not np.isnan(v) else v for v in gcn_rmse]
                gcn_baseline_rmse_plot = [v * 1000 if not np.isnan(v) else v for v in gcn_baseline_rmse]
                mlp_maml_rmse_plot = [v * 1000 if not np.isnan(v) else v for v in mlp_maml_rmse]
                mlp_aadam_rmse_plot = [v * 1000 if not np.isnan(v) else v for v in mlp_aadam_rmse]
                rmse_ylabel = 'RMSE (x1000)'
            else:
                gcn_rmse_plot = gcn_rmse
                gcn_baseline_rmse_plot = gcn_baseline_rmse
                mlp_maml_rmse_plot = mlp_maml_rmse
                mlp_aadam_rmse_plot = mlp_aadam_rmse
                rmse_ylabel = 'RMSE (ps)'

            all_bars_rmse = []
            bars1 = ax2.bar(x + offsets[0], gcn_rmse_plot, width, label=f'GCN MAML ({arch})', color='#2ecc71')
            all_bars_rmse.append(bars1)
            bars2 = ax2.bar(x + offsets[1], mlp_maml_rmse_plot, width, label='MLP MAML (id100_m32_l40_i300k)', color='#3498db')
            all_bars_rmse.append(bars2)
            bars3 = ax2.bar(x + offsets[2], mlp_aadam_rmse_plot, width, label=f'MLP AADAM (iter={aadam_iter})', color='#e74c3c')
            all_bars_rmse.append(bars3)
            if has_baseline_data:
                bars4 = ax2.bar(x + offsets[3], gcn_baseline_rmse_plot, width, label='GCN Baseline', color='#9b59b6')
                all_bars_rmse.append(bars4)

            ax2.set_xlabel('Cell', fontsize=11)
            ax2.set_ylabel(rmse_ylabel, fontsize=11, fontweight='bold')
            ax2.set_title('RMSE Comparison', fontsize=12)
            ax2.set_xticks(x)
            ax2.set_xticklabels(cells, rotation=45, ha='right', fontsize=9)
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bars in all_bars_rmse:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax2.annotate(f'{height:.3f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=6, rotation=90)

            plt.tight_layout()

            # Save
            arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
            plot_name = f'{title_prefix.lower()}_{data_type}_gcn_vs_mlp_{experiment}_{mode}{arch_suffix}.png'
            plot_path = os.path.join(output_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved: {plot_path}")

            # Print summary - use direct average from DataFrame (same as analyze_gcn_sweep_results.py)
            print(f"\n{'='*70}")
            print(f"Summary: {data_type_label} - {experiment} - {mode}")
            print(f"{'='*70}")
            rmse_header = 'RMSE (x1000)' if scale_rmse else 'RMSE (avg)'
            print(f"{'Model':<25} {'NRMSE (avg)':<15} {rmse_header:<15} {'#Cells':<10}")
            print(f"{'-'*70}")

            # GCN MAML (stage_aware)
            if has_gcn and len(filtered_gcn) > 0:
                gcn_nrmse_avg = filtered_gcn['NRMSE'].mean()
                gcn_rmse_avg = filtered_gcn['RMSE'].mean()
                gcn_rmse_display = gcn_rmse_avg * 1000 if scale_rmse else gcn_rmse_avg
                gcn_cell_count = len(filtered_gcn['cell'].unique())
                print(f"GCN_MAML ({arch}){'':<6} {gcn_nrmse_avg:<15.4f} {gcn_rmse_display:<15.4f} {gcn_cell_count:<10}")
            else:
                print(f"GCN_MAML{'':<17} {'N/A':<15} {'N/A':<15} {'N/A':<10}")

            # GCN baseline (full_graph) - always show if available
            if has_gcn_baseline and filtered_gcn_baseline is not None and len(filtered_gcn_baseline) > 0:
                gcn_baseline_nrmse_avg = filtered_gcn_baseline['NRMSE'].mean()
                gcn_baseline_rmse_avg = filtered_gcn_baseline['RMSE'].mean()
                gcn_baseline_rmse_display = gcn_baseline_rmse_avg * 1000 if scale_rmse else gcn_baseline_rmse_avg
                gcn_baseline_cell_count = len(filtered_gcn_baseline['cell'].unique())
                print(f"GCN_baseline{'':<13} {gcn_baseline_nrmse_avg:<15.4f} {gcn_baseline_rmse_display:<15.4f} {gcn_baseline_cell_count:<10}")

            # MLP baseline - filter to cells and compute direct average
            if mlp_df is not None:
                mlp_filtered = mlp_df[
                    (mlp_df['mode'] == mode) &
                    (mlp_df['cell'].isin(cells))
                ]

                # MAML with specific params
                maml_data = mlp_filtered[mlp_filtered['model_type'] == 'MLP_MAML']
                required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                    maml_specific = maml_data[
                        (maml_data['innerdiv'] == 100) &
                        (maml_data['meta'] == 32) &
                        (maml_data['layer_length'] == 40) &
                        (maml_data['iterations'] == 300000)
                    ]
                    if len(maml_specific) > 0:
                        maml_rmse_display = maml_specific['RMSE'].mean() * 1000 if scale_rmse else maml_specific['RMSE'].mean()
                        maml_cell_count = len(maml_specific['cell'].unique())
                        print(f"MLP MAML{'':<17} {maml_specific['NRMSE'].mean():<15.4f} {maml_rmse_display:<15.4f} {maml_cell_count:<10}")

                # AADAM
                aadam_data = mlp_filtered[mlp_filtered['model_type'] == 'AADAM']
                if aadam_iter:
                    aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
                if len(aadam_data) > 0:
                    aadam_rmse_display = aadam_data['RMSE'].mean() * 1000 if scale_rmse else aadam_data['RMSE'].mean()
                    aadam_cell_count = len(aadam_data['cell'].unique())
                    print(f"MLP AADAM (iter={aadam_iter}){'':<4} {aadam_data['NRMSE'].mean():<15.4f} {aadam_rmse_display:<15.4f} {aadam_cell_count:<10}")

            print(f"Cells: {cells}")


def plot_average_summary(gcn_df, mlp_df, output_dir, title_prefix, aadam_iter=300000, data_type='cell', scale_rmse=False, gcn_baseline_df=None, arch='', gcn_innerdiv=10, gcn_meta=16):
    """Generate average summary plot with average + vertical separator + per-cell results"""

    os.makedirs(output_dir, exist_ok=True)

    # Check if we have any data to plot
    has_gcn = gcn_df is not None and len(gcn_df) > 0
    has_mlp = mlp_df is not None and len(mlp_df) > 0
    has_gcn_baseline = gcn_baseline_df is not None and len(gcn_baseline_df) > 0

    if not has_gcn and not has_mlp:
        print("No data to plot (neither GCN nor MLP)")
        return

    if not has_gcn:
        print("No GCN data - plotting MLP baselines only in average summary")
        source_df = mlp_df.copy()
        source_df['experiment'] = source_df['topology']
        source_df['graph_mode'] = 'N/A'
    else:
        source_df = gcn_df

    # Data type label for title
    data_type_label = 'Cell Delay' if data_type == 'cell' else 'Transition (Slew)'

    for experiment in source_df['experiment'].dropna().unique():
        for mode in source_df['mode'].dropna().unique():
            for graph_mode in source_df['graph_mode'].dropna().unique():

                # Get filtered data based on what's available
                if has_gcn:
                    filtered_gcn = gcn_df[
                        (gcn_df['experiment'] == experiment) &
                        (gcn_df['mode'] == mode) &
                        (gcn_df['graph_mode'] == graph_mode) &
                        (gcn_df['data_type'] == data_type)
                    ]
                else:
                    filtered_gcn = pd.DataFrame()

                # For MLP-only case, get cells from MLP data
                if has_gcn and len(filtered_gcn) > 0:
                    cells = sorted(filtered_gcn['cell'].dropna().unique())
                    arch = filtered_gcn['arch_string'].iloc[0]
                elif has_mlp:
                    mlp_filtered = mlp_df[
                        (mlp_df['topology'] == experiment) &
                        (mlp_df['mode'] == mode)
                    ]
                    if len(mlp_filtered) == 0:
                        continue
                    cells = sorted(mlp_filtered['cell'].dropna().unique())
                    arch = 'N/A'
                else:
                    continue

                if len(cells) == 0:
                    continue

                # Calculate averages
                if has_gcn and len(filtered_gcn) > 0:
                    gcn_nrmse_avg = filtered_gcn['NRMSE'].mean()
                    gcn_rmse_avg = filtered_gcn['RMSE'].mean()
                else:
                    gcn_nrmse_avg = np.nan
                    gcn_rmse_avg = np.nan

                # MLP baselines averages
                mlp_maml_nrmse_avg = np.nan
                mlp_maml_rmse_avg = np.nan
                mlp_aadam_nrmse_avg = np.nan
                mlp_aadam_rmse_avg = np.nan

                # Per-cell data
                gcn_nrmse_per_cell = []
                mlp_maml_nrmse_per_cell = []
                mlp_aadam_nrmse_per_cell = []

                if mlp_df is not None:
                    mlp_filtered = mlp_df[
                        (mlp_df['mode'] == mode) &
                        (mlp_df['cell'].isin(cells))
                    ]

                    # MAML with specific params
                    maml_data = mlp_filtered[mlp_filtered['model_type'] == 'MLP_MAML']
                    required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                    if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                        maml_specific = maml_data[
                            (maml_data['innerdiv'] == 100) &
                            (maml_data['meta'] == 32) &
                            (maml_data['layer_length'] == 40) &
                            (maml_data['iterations'] == 300000)
                        ]
                        if len(maml_specific) > 0:
                            mlp_maml_nrmse_avg = maml_specific['NRMSE'].mean()
                            mlp_maml_rmse_avg = maml_specific['RMSE'].mean()
                    else:
                        maml_specific = pd.DataFrame()

                    # AADAM
                    aadam_data = mlp_filtered[mlp_filtered['model_type'] == 'AADAM']
                    if aadam_iter:
                        aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
                    if len(aadam_data) > 0:
                        mlp_aadam_nrmse_avg = aadam_data['NRMSE'].mean()
                        mlp_aadam_rmse_avg = aadam_data['RMSE'].mean()

                    # Per-cell data collection
                    for cell in cells:
                        # GCN per cell - handle empty DataFrame
                        if has_gcn and len(filtered_gcn) > 0:
                            cell_gcn = filtered_gcn[filtered_gcn['cell'] == cell]
                            gcn_nrmse_per_cell.append(cell_gcn['NRMSE'].mean() if len(cell_gcn) > 0 else np.nan)
                        else:
                            gcn_nrmse_per_cell.append(np.nan)

                        # MAML per cell
                        if len(maml_specific) > 0:
                            cell_maml = maml_specific[maml_specific['cell'] == cell]
                            mlp_maml_nrmse_per_cell.append(cell_maml['NRMSE'].mean() if len(cell_maml) > 0 else np.nan)
                        else:
                            mlp_maml_nrmse_per_cell.append(np.nan)

                        # AADAM per cell
                        if len(aadam_data) > 0:
                            cell_aadam = aadam_data[aadam_data['cell'] == cell]
                            mlp_aadam_nrmse_per_cell.append(cell_aadam['NRMSE'].mean() if len(cell_aadam) > 0 else np.nan)
                        else:
                            mlp_aadam_nrmse_per_cell.append(np.nan)
                else:
                    for cell in cells:
                        # GCN per cell - handle empty DataFrame
                        if has_gcn and len(filtered_gcn) > 0:
                            cell_gcn = filtered_gcn[filtered_gcn['cell'] == cell]
                            gcn_nrmse_per_cell.append(cell_gcn['NRMSE'].mean() if len(cell_gcn) > 0 else np.nan)
                        else:
                            gcn_nrmse_per_cell.append(np.nan)
                        mlp_maml_nrmse_per_cell.append(np.nan)
                        mlp_aadam_nrmse_per_cell.append(np.nan)

                # Prepare data for plotting: Average + per-cell
                # X-axis: [Average, cell1, cell2, ...]
                n_cells = len(cells)
                n_models = 3  # GCN, MAML, AADAM

                # Labels: Average followed by cell names
                x_labels = ['Average'] + cells

                # Create figure with wider width for cells
                fig_width = max(10, 4 + n_cells * 2)
                fig, ax = plt.subplots(figsize=(fig_width, 7))

                # X positions
                x = np.arange(len(x_labels))
                width = 0.25

                # Prepare data arrays: [avg, cell1, cell2, ...]
                gcn_values = [gcn_nrmse_avg] + gcn_nrmse_per_cell
                maml_values = [mlp_maml_nrmse_avg] + mlp_maml_nrmse_per_cell
                aadam_values = [mlp_aadam_nrmse_avg] + mlp_aadam_nrmse_per_cell

                # Calculate standard deviations for Average (first element only)
                # Per-cell bars have no error bars (individual measurements)
                gcn_valid = [v for v in gcn_nrmse_per_cell if not np.isnan(v)]
                maml_valid = [v for v in mlp_maml_nrmse_per_cell if not np.isnan(v)]
                aadam_valid = [v for v in mlp_aadam_nrmse_per_cell if not np.isnan(v)]

                gcn_std = np.std(gcn_valid) if len(gcn_valid) > 1 else 0
                maml_std = np.std(maml_valid) if len(maml_valid) > 1 else 0
                aadam_std = np.std(aadam_valid) if len(aadam_valid) > 1 else 0

                # Error bars: only for Average (index 0), zeros for per-cell
                gcn_errors = [gcn_std] + [0] * len(gcn_nrmse_per_cell)
                maml_errors = [maml_std] + [0] * len(mlp_maml_nrmse_per_cell)
                aadam_errors = [aadam_std] + [0] * len(mlp_aadam_nrmse_per_cell)

                # Colors: AADAM (red) → MLP_MAML (blue) → GCN (green) - showing development progression
                colors_aadam = '#e74c3c'  # Red (baseline)
                colors_maml = '#3498db'   # Blue (improved)
                colors_gcn = '#2ecc71'    # Green (best)

                # Plot bars in order: AADAM → MLP_MAML → GCN (left to right)
                bars_aadam = ax.bar(x - width, aadam_values, width, yerr=aadam_errors, capsize=3,
                              label=f'MLP AADAM (iter={aadam_iter//1000}k)', color=colors_aadam, edgecolor='black', linewidth=0.5,
                              error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})
                bars_maml = ax.bar(x, maml_values, width, yerr=maml_errors, capsize=3,
                              label='MLP MAML (id100_m32_l40)', color=colors_maml, edgecolor='black', linewidth=0.5,
                              error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})
                bars_gcn = ax.bar(x + width, gcn_values, width, yerr=gcn_errors, capsize=3,
                              label=f'GCN ({arch})', color=colors_gcn, edgecolor='black', linewidth=0.5,
                              error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'})

                # Add vertical dashed line separator after Average
                separator_x = 0.5  # Between Average (x=0) and first cell (x=1)
                ax.axvline(x=separator_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

                # Add value labels on bars
                all_values = gcn_values + maml_values + aadam_values
                valid_values = [v for v in all_values if not np.isnan(v)]
                max_val = max(valid_values) if valid_values else 1
                base_offset = max_val * 0.02

                # Add value labels with error bar offset for Average bars
                for bars, errors in [(bars_aadam, aadam_errors), (bars_maml, maml_errors), (bars_gcn, gcn_errors)]:
                    for idx, bar in enumerate(bars):
                        height = bar.get_height()
                        if not np.isnan(height):
                            # Add error bar height to offset for bars with error bars
                            error_offset = errors[idx] if idx < len(errors) else 0
                            total_offset = base_offset + error_offset
                            ax.text(bar.get_x() + bar.get_width()/2, height + total_offset,
                                   f'{height:.3f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=0)

                # Add total improvement (AADAM → GCN) as a curved arrow on top
                avg_aadam = aadam_values[0]
                avg_gcn = gcn_values[0]

                if not np.isnan(avg_aadam) and not np.isnan(avg_gcn) and avg_aadam > 0:
                    total_improvement = ((avg_aadam - avg_gcn) / avg_aadam) * 100
                    # Position arrow above error bars (aadam_std is the error bar height)
                    top_arrow_y = avg_aadam + aadam_std + max_val * 0.08
                    ax.annotate('', xy=(0 + width, top_arrow_y), xytext=(0 - width, top_arrow_y),
                               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5,
                                             connectionstyle='arc3,rad=-0.2'))
                    ax.text(0, top_arrow_y + max_val * 0.05, f'Total: ↓{total_improvement:.1f}%',
                           ha='center', va='bottom', fontsize=11, fontweight='bold', color='#27ae60',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#27ae60', alpha=0.9))

                ax.set_ylabel('NRMSE (%)', fontsize=14)
                ax.set_title(f'{title_prefix} - {data_type_label} - {experiment} - {mode.upper()} - {graph_mode}\nNRMSE: Average and Per-Cell ({len(cells)} Cells)',
                            fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=10, rotation=45, ha='right')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')

                # Expand y-axis to accommodate improvement arrow
                ax.set_ylim(0, max_val * 1.20)

                # Highlight Average section with light background
                ax.axvspan(-0.5, separator_x, alpha=0.1, color='yellow')

                plt.tight_layout()

                # Save - sanitize graph_mode for filename
                safe_graph_mode = graph_mode.replace('/', '_').replace('N/A', 'mlp_only')
                arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
                plot_name = f'{title_prefix.lower()}_{data_type}_average_summary_{experiment}_{mode}_{safe_graph_mode}{arch_suffix}.png'
                plot_path = os.path.join(output_dir, plot_name)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Saved: {plot_path}")


def plot_combined_average_summary(gcn_df, mlp_df, output_dir, title_prefix, aadam_iter=300000, data_type='cell', scale_rmse=False, gcn_baseline_df=None, arch='', gcn_innerdiv=10, gcn_meta=16):
    """Generate combined 2x2 grid plot showing all 4 cases (experiment x mode) in one PNG"""

    os.makedirs(output_dir, exist_ok=True)

    # Check if we have any data to plot
    has_gcn = gcn_df is not None and len(gcn_df) > 0
    has_mlp = mlp_df is not None and len(mlp_df) > 0
    has_gcn_baseline = gcn_baseline_df is not None and len(gcn_baseline_df) > 0

    if not has_gcn and not has_mlp:
        print("No data to plot (neither GCN nor MLP)")
        return

    if not has_gcn:
        print("No GCN data - plotting MLP baselines only in combined summary")
        source_df = mlp_df.copy()
        source_df['experiment'] = source_df['topology']
        source_df['graph_mode'] = 'N/A'
    else:
        source_df = gcn_df

    # Data type label for title
    data_type_label = 'Cell Delay' if data_type == 'cell' else 'Transition (Slew)'

    # Get unique graph_modes
    for graph_mode in source_df['graph_mode'].dropna().unique():
        # Collect data for all 4 cases
        cases = [
            ('intra_topology', 'extrapolation'),
            ('intra_topology', 'interpolation'),
            ('topology_agnostic', 'extrapolation'),
            ('topology_agnostic', 'interpolation'),
        ]

        case_data = []
        arch = None

        for experiment, mode in cases:
            # Get filtered data based on what's available
            if has_gcn:
                filtered_gcn = gcn_df[
                    (gcn_df['experiment'] == experiment) &
                    (gcn_df['mode'] == mode) &
                    (gcn_df['graph_mode'] == graph_mode)
                ]
            else:
                filtered_gcn = pd.DataFrame()

            # Determine cells based on available data
            if has_gcn and len(filtered_gcn) > 0:
                cells = sorted(filtered_gcn['cell'].dropna().unique())
                if arch is None:
                    arch = filtered_gcn['arch_string'].iloc[0]
            elif has_mlp:
                mlp_filtered = mlp_df[
                    (mlp_df['topology'] == experiment) &
                    (mlp_df['mode'] == mode)
                ]
                if len(mlp_filtered) == 0:
                    case_data.append(None)
                    continue
                cells = sorted(mlp_filtered['cell'].dropna().unique())
                if arch is None:
                    arch = 'N/A'
            else:
                case_data.append(None)
                continue

            if len(cells) == 0:
                case_data.append(None)
                continue

            # Calculate averages
            if has_gcn and len(filtered_gcn) > 0:
                gcn_nrmse_avg = filtered_gcn['NRMSE'].mean()
            else:
                gcn_nrmse_avg = np.nan

            # MLP baselines
            mlp_maml_nrmse_avg = np.nan
            mlp_aadam_nrmse_avg = np.nan

            if mlp_df is not None:
                mlp_filtered = mlp_df[
                    (mlp_df['mode'] == mode) &
                    (mlp_df['cell'].isin(cells))
                ]

                # MAML with specific params
                maml_data = mlp_filtered[mlp_filtered['model_type'] == 'MLP_MAML']
                required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                    maml_specific = maml_data[
                        (maml_data['innerdiv'] == 100) &
                        (maml_data['meta'] == 32) &
                        (maml_data['layer_length'] == 40) &
                        (maml_data['iterations'] == 300000)
                    ]
                    if len(maml_specific) > 0:
                        mlp_maml_nrmse_avg = maml_specific['NRMSE'].mean()

                # AADAM
                aadam_data = mlp_filtered[mlp_filtered['model_type'] == 'AADAM']
                if aadam_iter:
                    aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
                if len(aadam_data) > 0:
                    mlp_aadam_nrmse_avg = aadam_data['NRMSE'].mean()

            case_data.append({
                'experiment': experiment,
                'mode': mode,
                'n_cells': len(cells),
                'gcn': gcn_nrmse_avg,
                'maml': mlp_maml_nrmse_avg,
                'aadam': mlp_aadam_nrmse_avg
            })

        # Skip if no data
        if all(d is None for d in case_data):
            continue

        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{title_prefix} - {data_type_label} - {graph_mode}\nGCN ({arch}) vs MLP Comparison',
                    fontsize=16, fontweight='bold')

        model_names = ['GCN', 'MLP MAML', 'MLP AADAM']
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        subplot_titles = [
            'Intra-Topology - Extrapolation',
            'Intra-Topology - Interpolation',
            'Topology-Agnostic - Extrapolation',
            'Topology-Agnostic - Interpolation',
        ]

        for idx, (data, title) in enumerate(zip(case_data, subplot_titles)):
            ax = axes[idx // 2, idx % 2]

            if data is None:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                continue

            nrmse_values = [data['gcn'], data['maml'], data['aadam']]

            x = np.arange(len(model_names))
            bars = ax.bar(x, nrmse_values, color=colors, edgecolor='black', linewidth=0.5)

            # Add value labels on bars
            valid_values = [v for v in nrmse_values if not np.isnan(v)]
            max_val = max(valid_values) if valid_values else 1
            offset = max_val * 0.02
            for bar, val in zip(bars, nrmse_values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                           f'{val:.3f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax.set_ylabel('Average NRMSE (%)', fontsize=11)
            ax.set_title(f'{title}\n({data["n_cells"]} cells)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save - sanitize graph_mode for filename
        safe_graph_mode = graph_mode.replace('/', '_').replace('N/A', 'mlp_only')
        arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
        plot_name = f'{title_prefix.lower()}_{data_type}_combined_average_summary_{safe_graph_mode}{arch_suffix}.png'
        plot_path = os.path.join(output_dir, plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {plot_path}")


def _is_tsmc_cell(cell_name):
    """Check if cell name belongs to TSMC PDK (has BWP30P140 suffix or similar patterns)"""
    tsmc_patterns = ['BWP30P140', 'BWP30P', 'BWP']
    return any(pattern in cell_name for pattern in tsmc_patterns)


def _clean_tsmc_cell_name(cell_name):
    """Clean TSMC cell names for display:
    - Remove BWP suffix
    - Remove drive strength (D0, D1, etc.)
    - Expand abbreviated names (AN→AND, ND→NAND, NR→NOR)
    """
    import re
    # Remove BWP30P140, BWP30P, or BWP followed by anything
    cleaned = re.sub(r'BWP.*$', '', cell_name)
    # Remove drive strength (D followed by digit at the end)
    cleaned = re.sub(r'D\d+$', '', cleaned)
    # Expand abbreviated cell names (must be at the start)
    if cleaned.startswith('AN') and not cleaned.startswith('AND'):
        cleaned = 'AND' + cleaned[2:]
    elif cleaned.startswith('ND') and not cleaned.startswith('NAND'):
        cleaned = 'NAND' + cleaned[2:]
    elif cleaned.startswith('NR') and not cleaned.startswith('NOR'):
        cleaned = 'NOR' + cleaned[2:]
    return cleaned


def _aggregate_tsmc_cells_by_logic(cells, bar_data, pdk):
    """Aggregate TSMC cells by logic function (average across drive strengths).

    For TSMC, cells like AO21D0 and AO21D1 have the same logic but different drive strengths.
    This function groups them by cleaned name and averages their values.

    Args:
        cells: List of original cell names
        bar_data: Dict of {model: {mode: [values]}} where values align with cells
        pdk: PDK name ('TSMC' or 'ASAP7')

    Returns:
        (aggregated_cells, aggregated_bar_data): Tuple of unique cleaned cells and averaged data
    """
    if pdk != 'TSMC':
        return cells, bar_data

    from collections import defaultdict

    # Group cells by cleaned name
    cleaned_to_indices = defaultdict(list)
    for i, cell in enumerate(cells):
        cleaned = _clean_tsmc_cell_name(cell)
        cleaned_to_indices[cleaned].append(i)

    # Get unique cleaned names in original order (first occurrence)
    seen = set()
    unique_cleaned = []
    for cell in cells:
        cleaned = _clean_tsmc_cell_name(cell)
        if cleaned not in seen:
            seen.add(cleaned)
            unique_cleaned.append(cleaned)

    # Aggregate bar_data by averaging
    aggregated_bar_data = {}
    for model, mode_data in bar_data.items():
        aggregated_bar_data[model] = {}
        for mode, values in mode_data.items():
            aggregated_values = []
            for cleaned in unique_cleaned:
                indices = cleaned_to_indices[cleaned]
                cell_values = [values[i] for i in indices if i < len(values) and not np.isnan(values[i])]
                if cell_values:
                    aggregated_values.append(np.mean(cell_values))
                else:
                    aggregated_values.append(np.nan)
            aggregated_bar_data[model][mode] = aggregated_values

    return unique_cleaned, aggregated_bar_data


def _aggregate_tsmc_cell_data_by_logic(cells, cell_data, pdk):
    """Aggregate TSMC cell_data by logic function (average across drive strengths).

    Similar to _aggregate_tsmc_cells_by_logic but for cell_data dict structure
    used in LaTeX table generation.

    Args:
        cells: List of original cell names
        cell_data: Dict of {cell: {metric: {model: value}}}
        pdk: PDK name ('TSMC' or 'ASAP7')

    Returns:
        (aggregated_cells, aggregated_cell_data): Tuple of unique cleaned cells and averaged data
    """
    if pdk != 'TSMC':
        return cells, cell_data

    from collections import defaultdict

    # Group cells by cleaned name
    cleaned_to_cells = defaultdict(list)
    for cell in cells:
        cleaned = _clean_tsmc_cell_name(cell)
        cleaned_to_cells[cleaned].append(cell)

    # Get unique cleaned names in original order (first occurrence)
    seen = set()
    unique_cleaned = []
    for cell in cells:
        cleaned = _clean_tsmc_cell_name(cell)
        if cleaned not in seen:
            seen.add(cleaned)
            unique_cleaned.append(cleaned)

    # Aggregate cell_data by averaging
    aggregated_cell_data = {}
    for cleaned in unique_cleaned:
        original_cells = cleaned_to_cells[cleaned]
        aggregated_cell_data[cleaned] = {'NRMSE': {}, 'RMSE': {}}

        # Get all models from the first cell
        sample_cell = original_cells[0]
        all_models = set()
        for metric in ['NRMSE', 'RMSE']:
            if sample_cell in cell_data and metric in cell_data[sample_cell]:
                all_models.update(cell_data[sample_cell][metric].keys())

        # Average each metric for each model
        for metric in ['NRMSE', 'RMSE']:
            for model in all_models:
                values = []
                for orig_cell in original_cells:
                    if orig_cell in cell_data and metric in cell_data[orig_cell]:
                        val = cell_data[orig_cell][metric].get(model)
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            values.append(val)
                if values:
                    aggregated_cell_data[cleaned][metric][model] = np.mean(values)

    return unique_cleaned, aggregated_cell_data


def _filter_cells_by_pdk(cells, pdk):
    """Filter cell names based on PDK naming conventions"""
    if pdk == 'TSMC':
        return [c for c in cells if _is_tsmc_cell(c)]
    elif pdk == 'ASAP7':
        return [c for c in cells if not _is_tsmc_cell(c)]
    return cells


def _filter_df_by_pdk_cell(df, pdk):
    """Filter DataFrame rows based on PDK cell naming conventions"""
    if df is None or len(df) == 0 or 'cell' not in df.columns:
        return df
    if pdk == 'TSMC':
        return df[df['cell'].apply(_is_tsmc_cell)]
    elif pdk == 'ASAP7':
        return df[~df['cell'].apply(_is_tsmc_cell)]
    return df


def _get_aggregated_cell_name(cell_name):
    """
    Map cell names with different strengths to aggregated names for ASAP7.
    e.g., XOR2x1, XOR2x2, XOR2xp5 -> XOR2
          XNOR2x1, XNOR2x2, XNOR2xp5 -> XNOR2
          MAJx2, MAJx3, MAJIxp5 -> MAJ
    """
    import re

    # XNOR2 variants: XNOR2x1, XNOR2x2, XNOR2xp5
    if re.match(r'^XNOR2x[0-9p]+$', cell_name):
        return 'XNOR2'

    # XOR2 variants: XOR2x1, XOR2x2, XOR2xp5
    if re.match(r'^XOR2x[0-9p]+$', cell_name):
        return 'XOR2'

    # MAJ variants: MAJx2, MAJx3, MAJIxp5 (note: MAJI is also MAJ)
    if re.match(r'^MAJ[I]?x[0-9p]+$', cell_name):
        return 'MAJ'

    # No aggregation for other cells
    return cell_name


def _aggregate_cells_for_plotting(cells):
    """
    Aggregate cell list by grouping XOR2, XNOR2, MAJ variants.
    Returns: (aggregated_cells, cell_mapping)
        - aggregated_cells: sorted list of unique aggregated cell names
        - cell_mapping: dict mapping aggregated_name -> [original_cell_names]
    """
    cell_mapping = {}
    for cell in cells:
        agg_name = _get_aggregated_cell_name(cell)
        if agg_name not in cell_mapping:
            cell_mapping[agg_name] = []
        cell_mapping[agg_name].append(cell)

    aggregated_cells = sorted(cell_mapping.keys())
    return aggregated_cells, cell_mapping


def _get_aggregated_nrmse(df, cell_mapping, agg_cell, filter_conditions):
    """
    Get averaged NRMSE for aggregated cell across all its variants.

    Args:
        df: DataFrame with NRMSE values
        cell_mapping: dict mapping aggregated_name -> [original_cell_names]
        agg_cell: aggregated cell name
        filter_conditions: dict of column -> value for filtering (e.g., {'mode': 'extrapolation'})

    Returns:
        float: averaged NRMSE across all variants, or np.nan if no data
    """
    if df is None or len(df) == 0:
        return np.nan

    original_cells = cell_mapping.get(agg_cell, [agg_cell])

    # Build filter mask
    mask = df['cell'].isin(original_cells)
    for col, val in filter_conditions.items():
        if col in df.columns:
            mask = mask & (df[col] == val)

    filtered = df[mask]
    if len(filtered) > 0 and 'NRMSE' in filtered.columns:
        return filtered['NRMSE'].mean()
    return np.nan


def plot_pdk_comparison_grid(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict, output_dir,
                              aadam_iter=300000, mlp_maml_iter=300000, experiment='topology_agnostic', arch='', gcn_innerdiv=10, gcn_meta=16):
    """Generate 2x2 grid plot: rows=PDK, columns=data_type

    Each cell shows 4 overlapping bars per cell: Extrapolation (wider, darker) behind Interpolation (narrower, lighter)

    Args:
        gcn_maml_df_dict: Dict with keys 'ASAP7', 'TSMC' containing GCN MAML DataFrames
        gcn_baseline_df_dict: Dict with keys 'ASAP7', 'TSMC' containing GCN baseline DataFrames
        mlp_df_dict: Dict with keys 'ASAP7', 'TSMC' containing MLP DataFrames
        output_dir: Output directory for plots
        aadam_iter: AADAM iteration to filter
        mlp_maml_iter: MLP MAML iteration to filter
        experiment: Experiment type ('topology_agnostic' or 'intra_topology')
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define colors: (light_for_inter, dark_for_extra)
    # Extrapolation: darker, wider (background) - usually worse/higher values
    # Interpolation: lighter, narrower (foreground) - usually better/lower values
    # MLP family: Green, GCN family: Blue
    model_colors = {
        'AADAM': ('#d4e8d4', '#b4d4b4'),           # Green family (MLP baseline)
        'GCN_baseline': ('#c8ddf0', '#A8C4E0'),    # Blue family (GCN baseline)
        'MLP_MAML': ('#90ee90', '#228b22'),        # Green family (MLP MAML)
        'GCN_MAML': ('#7eb8e0', '#1B5E91'),        # Blue family (GCN MAML)
    }

    pdk_labels = {
        'TSMC': 'Commercial 28nm',
        'ASAP7': 'Open-source 7nm'
    }

    data_type_labels = {
        'cell': 'Cell delay',
        'transition': 'Transition time'
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    pdks = ['TSMC', 'ASAP7']
    data_types = ['cell', 'transition']
    modes = ['interpolation', 'extrapolation']
    model_order = ['AADAM', 'GCN_baseline', 'MLP_MAML', 'GCN_MAML']

    for row_idx, pdk in enumerate(pdks):
        for col_idx, data_type in enumerate(data_types):
            ax = axes[row_idx, col_idx]

            gcn_maml_df = gcn_maml_df_dict.get(pdk)
            gcn_baseline_df = gcn_baseline_df_dict.get(pdk)
            mlp_df = mlp_df_dict.get(pdk)

            # Filter DataFrames by PDK cell naming convention
            gcn_maml_df = _filter_df_by_pdk_cell(gcn_maml_df, pdk)
            gcn_baseline_df = _filter_df_by_pdk_cell(gcn_baseline_df, pdk)
            mlp_df = _filter_df_by_pdk_cell(mlp_df, pdk)

            # Find cells
            cells_set = set()

            if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                filtered = gcn_maml_df[
                    (gcn_maml_df['experiment'] == experiment) &
                    (gcn_maml_df['data_type'] == data_type)
                ]
                if len(filtered) > 0:
                    cells_set.update(filtered['cell'].dropna().unique())

            if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                filtered = gcn_baseline_df[
                    (gcn_baseline_df['experiment'] == experiment) &
                    (gcn_baseline_df['data_type'] == data_type)
                ]
                if len(filtered) > 0:
                    cells_set.update(filtered['cell'].dropna().unique())

            if mlp_df is not None and len(mlp_df) > 0:
                filtered = mlp_df[
                    (mlp_df['topology'] == experiment) &
                    (mlp_df['data_type'] == data_type)
                ]
                if len(filtered) > 0:
                    cells_set.update(filtered['cell'].dropna().unique())

            cells = sorted(cells_set)
            cells = _filter_cells_by_pdk(cells, pdk)

            if len(cells) == 0:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{pdk_labels.get(pdk, pdk)} - {data_type_labels.get(data_type, data_type)}',
                            fontsize=12, fontweight='bold')
                continue

            n_cells = len(cells)
            n_models = len(model_order)
            bar_width = 0.08  # Width of each bar
            group_gap = 0.15  # Gap between extra and inter groups within a cell
            cell_width = n_models * bar_width * 2 + group_gap + 0.3  # Total width per cell

            x = np.arange(n_cells) * cell_width  # Cell positions

            # Collect NRMSE values
            bar_data = {model: {mode: [] for mode in modes} for model in model_order}

            for cell in cells:
                # AADAM
                for mode in modes:
                    if mlp_df is not None and len(mlp_df) > 0:
                        aadam_data = mlp_df[
                            (mlp_df['cell'] == cell) &
                            (mlp_df['mode'] == mode) &
                            (mlp_df['model_type'] == 'AADAM') &
                            (mlp_df['iterations'] == aadam_iter) &
                            (mlp_df['topology'] == experiment) &
                            (mlp_df['data_type'] == data_type)
                        ]
                        if len(aadam_data) > 0:
                            bar_data['AADAM'][mode].append(aadam_data['NRMSE'].mean())
                        else:
                            bar_data['AADAM'][mode].append(np.nan)
                    else:
                        bar_data['AADAM'][mode].append(np.nan)

                # GCN baseline
                for mode in modes:
                    if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                        baseline_data = gcn_baseline_df[
                            (gcn_baseline_df['cell'] == cell) &
                            (gcn_baseline_df['mode'] == mode) &
                            (gcn_baseline_df['experiment'] == experiment) &
                            (gcn_baseline_df['data_type'] == data_type)
                        ]
                        if len(baseline_data) > 0:
                            bar_data['GCN_baseline'][mode].append(baseline_data['NRMSE'].mean())
                        else:
                            bar_data['GCN_baseline'][mode].append(np.nan)
                    else:
                        bar_data['GCN_baseline'][mode].append(np.nan)

                # MLP MAML
                for mode in modes:
                    if mlp_df is not None and len(mlp_df) > 0:
                        maml_data = mlp_df[
                            (mlp_df['cell'] == cell) &
                            (mlp_df['mode'] == mode) &
                            (mlp_df['model_type'] == 'MLP_MAML') &
                            (mlp_df['topology'] == experiment) &
                            (mlp_df['data_type'] == data_type)
                        ]
                        required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                        if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                            maml_specific = maml_data[
                                (maml_data['innerdiv'] == 100) &
                                (maml_data['meta'] == 32) &
                                (maml_data['layer_length'] == 40) &
                                (maml_data['iterations'] == mlp_maml_iter)
                            ]
                            if len(maml_specific) > 0:
                                bar_data['MLP_MAML'][mode].append(maml_specific['NRMSE'].mean())
                            else:
                                bar_data['MLP_MAML'][mode].append(np.nan)
                        else:
                            bar_data['MLP_MAML'][mode].append(np.nan)
                    else:
                        bar_data['MLP_MAML'][mode].append(np.nan)

                # GCN MAML
                for mode in modes:
                    if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                        gcn_data = gcn_maml_df[
                            (gcn_maml_df['cell'] == cell) &
                            (gcn_maml_df['mode'] == mode) &
                            (gcn_maml_df['experiment'] == experiment) &
                            (gcn_maml_df['data_type'] == data_type)
                        ]
                        if len(gcn_data) > 0:
                            bar_data['GCN_MAML'][mode].append(gcn_data['NRMSE'].mean())
                        else:
                            bar_data['GCN_MAML'][mode].append(np.nan)
                    else:
                        bar_data['GCN_MAML'][mode].append(np.nan)

            # Aggregate TSMC cells by logic function (average across drive strengths)
            display_cells, bar_data = _aggregate_tsmc_cells_by_logic(cells, bar_data, pdk)
            n_cells = len(display_cells)

            x = np.arange(n_cells) * cell_width  # Recalculate cell positions

            # Plot separated bars: each cell has 2 groups (Extra | Inter), each with 4 model bars
            legend_handles = []
            legend_labels = []

            # Calculate group offsets within each cell
            # Extra group: left side, Inter group: right side
            extra_group_center = -group_gap / 2 - n_models * bar_width / 2
            inter_group_center = group_gap / 2 + n_models * bar_width / 2

            for model_idx, model in enumerate(model_order):
                light_color, dark_color = model_colors[model]

                # Position within the 4-bar group
                model_offset = (model_idx - 1.5) * bar_width

                extra_values = bar_data[model]['extrapolation']
                inter_values = bar_data[model]['interpolation']

                # Draw extrapolation bars (left group, darker color)
                bars_extra = ax.bar(x + extra_group_center + model_offset, extra_values, bar_width,
                                   color=dark_color, edgecolor='black', linewidth=0.5)

                # Draw interpolation bars (right group, lighter color)
                bars_inter = ax.bar(x + inter_group_center + model_offset, inter_values, bar_width,
                                   color=light_color, edgecolor='black', linewidth=0.5)

                if row_idx == 0 and col_idx == 0:
                    model_label = model.replace('_', ' ')
                    legend_handles.append(bars_extra)
                    legend_labels.append(f'{model_label} (Extra)')
                    legend_handles.append(bars_inter)
                    legend_labels.append(f'{model_label} (Inter)')

            ax.set_xlabel('Cell', fontsize=11)
            ax.set_ylabel('NRMSE (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{pdk_labels.get(pdk, pdk)} - {data_type_labels.get(data_type, data_type)}\n({n_cells} cells)',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            # display_cells already cleaned and aggregated by _aggregate_tsmc_cells_by_logic
            ax.set_xticklabels(display_cells, rotation=45, ha='right', fontsize=12)

            all_values = []
            for model in model_order:
                for mode in modes:
                    all_values.extend([v for v in bar_data[model][mode] if not np.isnan(v)])
            if all_values:
                y_max = max(all_values) * 1.15
                ax.set_ylim(0, y_max)

                # Add horizontal dashed lines at every 0.5% interval
                y_ticks = np.arange(0, y_max + 0.5, 0.5)
                ax.set_yticks(y_ticks)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                for y_val in y_ticks:
                    ax.axhline(y=y_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    if legend_handles:
        axes[0, 0].legend(legend_handles, legend_labels, loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()

    arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
    plot_name = f'pdk_comparison_grid_{experiment}{arch_suffix}.png'
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {plot_path}")


def plot_per_pdk_all_experiments(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict, output_dir,
                                  aadam_iter=300000, mlp_maml_iter=300000, arch='', gcn_innerdiv=10, gcn_meta=16):
    """Generate per-PDK 2x2 grid plot: rows=mode (extrapolation/interpolation), columns=experiment

    Each PDK gets its own figure with all 4 combinations of mode × experiment.
    Each subplot shows cell and transition data together (solid=cell, hatched=transition).

    GCN_MAML uses stage_aware, GCN_baseline uses full_graph (fixed).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define single color per model (for cell_trans combined visualization)
    # MLP family: Green, GCN family: Blue
    model_colors = {
        'AADAM': '#b4d4b4',           # Green family (MLP baseline)
        'GCN_baseline': '#A8C4E0',    # Blue family (GCN baseline)
        'MLP_MAML': '#228b22',        # Green family (MLP MAML)
        'GCN_MAML': '#1B5E91',        # Blue family (GCN MAML)
    }

    pdk_labels = {
        'TSMC': 'Commercial 28nm',
        'ASAP7': 'Open-source 7nm'
    }

    experiment_labels = {
        'topology_agnostic': 'Cross-Topology',
        'intra_topology': 'Intra Topology'
    }

    mode_labels = {
        'extrapolation': 'Extrapolation',
        'interpolation': 'Interpolation'
    }

    pdks = ['TSMC', 'ASAP7']
    experiments = ['topology_agnostic', 'intra_topology']
    data_types = ['transition', 'cell']  # transition first, then cell
    modes = ['extrapolation', 'interpolation']
    model_order = ['AADAM', 'GCN_baseline', 'MLP_MAML', 'GCN_MAML']

    for pdk in pdks:
        fig, axes = plt.subplots(2, 2, figsize=(20, 9))
        plt.subplots_adjust(hspace=0.25, wspace=0.15)

        gcn_maml_df = gcn_maml_df_dict.get(pdk)
        gcn_baseline_df = gcn_baseline_df_dict.get(pdk)
        mlp_df = mlp_df_dict.get(pdk)

        # Filter DataFrames by PDK cell naming convention
        gcn_maml_df = _filter_df_by_pdk_cell(gcn_maml_df, pdk)
        gcn_baseline_df = _filter_df_by_pdk_cell(gcn_baseline_df, pdk)
        mlp_df = _filter_df_by_pdk_cell(mlp_df, pdk)

        legend_handles = []
        legend_labels = []

        # rows = experiments (cross-topology, intra-topology), columns = modes (extrapolation, interpolation)
        for row_idx, experiment in enumerate(experiments):
            for col_idx, mode in enumerate(modes):
                ax = axes[row_idx, col_idx]

                # Find cells for this experiment (across both data types)
                cells_set = set()

                for data_type in data_types:
                    expected_pooling = 'output' if data_type == 'transition' else 'mean'

                    # GCN_MAML: always use stage_aware
                    if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                        filtered = gcn_maml_df[
                            (gcn_maml_df['experiment'] == experiment) &
                            (gcn_maml_df['data_type'] == data_type) &
                            (gcn_maml_df['pooling'] == expected_pooling) &
                            (gcn_maml_df['graph_mode'] == 'stage_aware')
                        ]
                        if len(filtered) > 0:
                            cells_set.update(filtered['cell'].dropna().unique())

                    if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                        filtered = gcn_baseline_df[
                            (gcn_baseline_df['experiment'] == experiment) &
                            (gcn_baseline_df['data_type'] == data_type) &
                            (gcn_baseline_df['pooling'] == 'mean')
                        ]
                        if len(filtered) > 0:
                            cells_set.update(filtered['cell'].dropna().unique())

                    if mlp_df is not None and len(mlp_df) > 0:
                        filtered = mlp_df[
                            (mlp_df['topology'] == experiment) &
                            (mlp_df['data_type'] == data_type)
                        ]
                        if len(filtered) > 0:
                            cells_set.update(filtered['cell'].dropna().unique())

                cells = sorted(cells_set)
                cells = _filter_cells_by_pdk(cells, pdk)

                # For ASAP7: aggregate XOR2, XNOR2, MAJ variants
                if pdk == 'ASAP7':
                    cells, cell_mapping = _aggregate_cells_for_plotting(cells)
                else:
                    cell_mapping = {c: [c] for c in cells}

                if len(cells) == 0:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{experiment_labels.get(experiment, experiment)} - {mode_labels.get(mode, mode)}',
                                fontsize=12, fontweight='bold')
                    continue

                n_cells = len(cells)
                n_models = len(model_order)
                bar_width = 0.08
                group_gap = 0.15
                cell_width = n_models * bar_width * 2 + group_gap + 0.3

                x = np.arange(n_cells) * cell_width

                # Collect NRMSE values for both data types (cell and transition)
                bar_data = {model: {dt: [] for dt in data_types} for model in model_order}

                for cell in cells:
                    original_cells = cell_mapping.get(cell, [cell])

                    for data_type in data_types:
                        expected_pooling = 'output' if data_type == 'transition' else 'mean'

                        # AADAM
                        if mlp_df is not None and len(mlp_df) > 0:
                            aadam_data = mlp_df[
                                (mlp_df['cell'].isin(original_cells)) &
                                (mlp_df['mode'] == mode) &
                                (mlp_df['model_type'] == 'AADAM') &
                                (mlp_df['iterations'] == aadam_iter) &
                                (mlp_df['topology'] == experiment) &
                                (mlp_df['data_type'] == data_type)
                            ]
                            if len(aadam_data) > 0:
                                bar_data['AADAM'][data_type].append(aadam_data['NRMSE'].mean())
                            else:
                                bar_data['AADAM'][data_type].append(np.nan)
                        else:
                            bar_data['AADAM'][data_type].append(np.nan)

                        # GCN baseline
                        if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                            baseline_data = gcn_baseline_df[
                                (gcn_baseline_df['cell'].isin(original_cells)) &
                                (gcn_baseline_df['mode'] == mode) &
                                (gcn_baseline_df['experiment'] == experiment) &
                                (gcn_baseline_df['data_type'] == data_type) &
                                (gcn_baseline_df['pooling'] == 'mean')
                            ]
                            if len(baseline_data) > 0:
                                bar_data['GCN_baseline'][data_type].append(baseline_data['NRMSE'].mean())
                            else:
                                bar_data['GCN_baseline'][data_type].append(np.nan)
                        else:
                            bar_data['GCN_baseline'][data_type].append(np.nan)

                        # MLP MAML
                        if mlp_df is not None and len(mlp_df) > 0:
                            maml_data = mlp_df[
                                (mlp_df['cell'].isin(original_cells)) &
                                (mlp_df['mode'] == mode) &
                                (mlp_df['model_type'] == 'MLP_MAML') &
                                (mlp_df['topology'] == experiment) &
                                (mlp_df['data_type'] == data_type)
                            ]
                            required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                            if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                                maml_specific = maml_data[
                                    (maml_data['innerdiv'] == 100) &
                                    (maml_data['meta'] == 32) &
                                    (maml_data['layer_length'] == 40) &
                                    (maml_data['iterations'] == mlp_maml_iter)
                                ]
                                if len(maml_specific) > 0:
                                    bar_data['MLP_MAML'][data_type].append(maml_specific['NRMSE'].mean())
                                else:
                                    bar_data['MLP_MAML'][data_type].append(np.nan)
                            else:
                                bar_data['MLP_MAML'][data_type].append(np.nan)
                        else:
                            bar_data['MLP_MAML'][data_type].append(np.nan)

                        # GCN MAML: always use stage_aware
                        if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                            gcn_data = gcn_maml_df[
                                (gcn_maml_df['cell'].isin(original_cells)) &
                                (gcn_maml_df['mode'] == mode) &
                                (gcn_maml_df['experiment'] == experiment) &
                                (gcn_maml_df['data_type'] == data_type) &
                                (gcn_maml_df['pooling'] == expected_pooling) &
                                (gcn_maml_df['graph_mode'] == 'stage_aware')
                            ]
                            if len(gcn_data) > 0:
                                bar_data['GCN_MAML'][data_type].append(gcn_data['NRMSE'].mean())
                            else:
                                bar_data['GCN_MAML'][data_type].append(np.nan)
                        else:
                            bar_data['GCN_MAML'][data_type].append(np.nan)

                # Aggregate TSMC cells by logic function (average across drive strengths)
                display_cells, bar_data = _aggregate_tsmc_cells_by_logic(cells, bar_data, pdk)
                n_cells = len(display_cells)

                x = np.arange(n_cells) * cell_width

                # Plot combined cell/transition bars
                handles, labels, y_lower, y_upper = draw_broken_axis_bars_cell_trans(
                    ax, x, bar_data, model_order, model_colors, data_types,
                    bar_width, group_gap, break_threshold=3.0
                )

                if row_idx == 0 and col_idx == 0:
                    legend_handles = handles
                    legend_labels = labels

                ax.set_ylabel('NRMSE (%)', fontsize=6.5, fontweight='bold')
                ax.set_title(f'{experiment_labels.get(experiment, experiment)} - {mode_labels.get(mode, mode)}',
                            fontsize=7.5, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(display_cells, rotation=45, ha='right', fontsize=6)

                # x, y축 실선
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_color('black')
                ax.spines['bottom'].set_color('black')
                ax.spines['left'].set_linewidth(0.8)
                ax.spines['bottom'].set_linewidth(0.8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                ax.tick_params(axis='both', colors='black', width=0.6, labelsize=6)
                # Add horizontal dotted lines at every 1% interval
                y_max = y_lower if y_lower > 0 else 2.5
                y_ticks = np.arange(0, y_max + 1.0, 1.0)
                ax.set_yticks(y_ticks)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
                for y_val in y_ticks:
                    ax.axhline(y=y_val, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc='upper center',
                      fontsize=6, ncol=8, bbox_to_anchor=(0.5, 0.99), frameon=True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for legend at top

        arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
        plot_name = f'per_pdk_{pdk}_all_experiments{arch_suffix}.png'
        plot_path = os.path.join(output_dir, plot_name)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {plot_path}")

    # Print data counts summary
    print(f"\n{'='*60}")
    print(f"Per-PDK Plot Data Counts Summary (GCN_MAML=stage_aware, GCN_baseline=full_graph)")
    print(f"{'='*60}")

    for pdk in pdks:
        gcn_maml_df = gcn_maml_df_dict.get(pdk)
        gcn_baseline_df = gcn_baseline_df_dict.get(pdk)
        mlp_df = mlp_df_dict.get(pdk)

        gcn_maml_df = _filter_df_by_pdk_cell(gcn_maml_df, pdk)
        gcn_baseline_df = _filter_df_by_pdk_cell(gcn_baseline_df, pdk)
        mlp_df = _filter_df_by_pdk_cell(mlp_df, pdk)

        print(f"\n[{pdk}]")
        for experiment in experiments:
            for data_type in data_types:
                expected_pooling = 'output' if data_type == 'transition' else 'mean'
                counts = {}

                # Count AADAM
                if mlp_df is not None and len(mlp_df) > 0:
                    aadam_count = len(mlp_df[
                        (mlp_df['topology'] == experiment) &
                        (mlp_df['data_type'] == data_type) &
                        (mlp_df['model_type'] == 'AADAM') &
                        (mlp_df['iterations'] == aadam_iter)
                    ]['cell'].unique())
                    counts['AADAM'] = aadam_count
                else:
                    counts['AADAM'] = 0

                # Count GCN_baseline
                if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                    baseline_count = len(gcn_baseline_df[
                        (gcn_baseline_df['experiment'] == experiment) &
                        (gcn_baseline_df['data_type'] == data_type) &
                        (gcn_baseline_df['pooling'] == 'mean')  # GCN baseline always uses mean pooling
                    ]['cell'].unique())
                    counts['GCN_baseline'] = baseline_count
                else:
                    counts['GCN_baseline'] = 0

                # Count MLP_MAML
                if mlp_df is not None and len(mlp_df) > 0:
                    maml_data = mlp_df[
                        (mlp_df['topology'] == experiment) &
                        (mlp_df['data_type'] == data_type) &
                        (mlp_df['model_type'] == 'MLP_MAML')
                    ]
                    if len(maml_data) > 0 and all(col in maml_data.columns for col in ['innerdiv', 'meta', 'layer_length', 'iterations']):
                        maml_specific = maml_data[
                            (maml_data['innerdiv'] == 100) &
                            (maml_data['meta'] == 32) &
                            (maml_data['layer_length'] == 40) &
                            (maml_data['iterations'] == mlp_maml_iter)
                        ]
                        counts['MLP_MAML'] = len(maml_specific['cell'].unique())
                    else:
                        counts['MLP_MAML'] = 0
                else:
                    counts['MLP_MAML'] = 0

                # Count GCN_MAML (always stage_aware)
                if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                    gcn_count = len(gcn_maml_df[
                        (gcn_maml_df['experiment'] == experiment) &
                        (gcn_maml_df['data_type'] == data_type) &
                        (gcn_maml_df['pooling'] == expected_pooling) &
                        (gcn_maml_df['graph_mode'] == 'stage_aware')
                    ]['cell'].unique())
                    counts['GCN_MAML'] = gcn_count
                else:
                    counts['GCN_MAML'] = 0

                exp_short = 'Agnostic' if experiment == 'topology_agnostic' else 'Intra'
                dtype_short = 'cell' if data_type == 'cell' else 'trans'
                count_str = ', '.join([f"{k}:{v}" for k, v in counts.items()])
                print(f"  {exp_short}/{dtype_short}: {count_str}")


def generate_category_averaged_latex_table(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict, output_dir,
                                           aadam_iter=300000, mlp_maml_iter=300000, arch='', gcn_innerdiv=10, gcn_meta=16,
                                           data_type='cell'):
    """Generate LaTeX table with category-averaged NRMSE values (Interpolation/Extrapolation format)

    Args:
        data_type: 'cell' for cell delay, 'transition' for output transition

    GCN_MAML uses stage_aware, GCN_baseline uses full_graph (fixed).
    """
    os.makedirs(output_dir, exist_ok=True)

    pdks = ['TSMC', 'ASAP7']
    experiments = ['intra_topology', 'topology_agnostic']
    modes = ['interpolation', 'extrapolation']
    model_order = ['AADAM', 'GCN_baseline', 'MLP_MAML', 'GCN_MAML']

    pdk_labels = {'TSMC': 'Commercial', 'ASAP7': 'ASAP7'}
    exp_labels = {'intra_topology': 'Intra-topology', 'topology_agnostic': 'Topology-agnostic'}

    # Collect averaged NRMSE and RMSE values
    # Structure: results[pdk][experiment][model][mode] = {'nrmse': avg_nrmse, 'rmse': avg_rmse}
    results = {}
    results_rmse = {}

    for pdk in pdks:
        results[pdk] = {}
        results_rmse[pdk] = {}
        gcn_maml_df = gcn_maml_df_dict.get(pdk)
        gcn_baseline_df = gcn_baseline_df_dict.get(pdk)
        mlp_df = mlp_df_dict.get(pdk)

        # Filter by PDK cell naming convention
        gcn_maml_df = _filter_df_by_pdk_cell(gcn_maml_df, pdk)
        gcn_baseline_df = _filter_df_by_pdk_cell(gcn_baseline_df, pdk)
        mlp_df = _filter_df_by_pdk_cell(mlp_df, pdk)

        expected_pooling = 'output' if data_type == 'transition' else 'mean'

        for experiment in experiments:
            results[pdk][experiment] = {}
            results_rmse[pdk][experiment] = {}

            for model in model_order:
                results[pdk][experiment][model] = {}
                results_rmse[pdk][experiment][model] = {}

                for mode in modes:
                    avg_nrmse = np.nan
                    avg_rmse = np.nan

                    if model == 'AADAM':
                        if mlp_df is not None and len(mlp_df) > 0:
                            filtered = mlp_df[
                                (mlp_df['topology'] == experiment) &
                                (mlp_df['data_type'] == data_type) &
                                (mlp_df['mode'] == mode) &
                                (mlp_df['model_type'] == 'AADAM') &
                                (mlp_df['iterations'] == aadam_iter)
                            ]
                            if len(filtered) > 0:
                                avg_nrmse = filtered['NRMSE'].mean()
                                if 'RMSE' in filtered.columns:
                                    avg_rmse = filtered['RMSE'].mean()

                    elif model == 'GCN_baseline':
                        if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                            filtered = gcn_baseline_df[
                                (gcn_baseline_df['experiment'] == experiment) &
                                (gcn_baseline_df['data_type'] == data_type) &
                                (gcn_baseline_df['mode'] == mode) &
                                (gcn_baseline_df['pooling'] == 'mean')  # GCN baseline always uses mean pooling
                            ]
                            if len(filtered) > 0:
                                avg_nrmse = filtered['NRMSE'].mean()
                                if 'RMSE' in filtered.columns:
                                    avg_rmse = filtered['RMSE'].mean()

                    elif model == 'MLP_MAML':
                        if mlp_df is not None and len(mlp_df) > 0:
                            filtered = mlp_df[
                                (mlp_df['topology'] == experiment) &
                                (mlp_df['data_type'] == data_type) &
                                (mlp_df['mode'] == mode) &
                                (mlp_df['model_type'] == 'MLP_MAML') &
                                (mlp_df['innerdiv'] == 100) &
                                (mlp_df['meta'] == 32) &
                                (mlp_df['layer_length'] == 40) &
                                (mlp_df['iterations'] == mlp_maml_iter)
                            ]
                            if len(filtered) > 0:
                                avg_nrmse = filtered['NRMSE'].mean()
                                if 'RMSE' in filtered.columns:
                                    avg_rmse = filtered['RMSE'].mean()

                    elif model == 'GCN_MAML':
                        # GCN_MAML: always use stage_aware
                        if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                            filtered = gcn_maml_df[
                                (gcn_maml_df['experiment'] == experiment) &
                                (gcn_maml_df['data_type'] == data_type) &
                                (gcn_maml_df['mode'] == mode) &
                                (gcn_maml_df['pooling'] == expected_pooling) &
                                (gcn_maml_df['graph_mode'] == 'stage_aware')
                            ]
                            if len(filtered) > 0:
                                avg_nrmse = filtered['NRMSE'].mean()
                                if 'RMSE' in filtered.columns:
                                    avg_rmse = filtered['RMSE'].mean()

                    results[pdk][experiment][model][mode] = avg_nrmse
                    results_rmse[pdk][experiment][model][mode] = avg_rmse

    # Calculate overall averages for NRMSE
    overall = {model: {mode: [] for mode in modes} for model in model_order}
    for pdk in pdks:
        for experiment in experiments:
            for model in model_order:
                for mode in modes:
                    val = results[pdk][experiment][model][mode]
                    if not np.isnan(val):
                        overall[model][mode].append(val)

    overall_avg = {}
    for model in model_order:
        overall_avg[model] = {}
        for mode in modes:
            vals = overall[model][mode]
            overall_avg[model][mode] = np.mean(vals) if vals else np.nan

    # Calculate overall averages for RMSE (apply scaling before averaging: TSMC x1000, ASAP7 x1)
    overall_rmse_scaled = {model: {mode: [] for mode in modes} for model in model_order}
    for pdk in pdks:
        scale = 1000 if pdk == 'TSMC' else 1
        for experiment in experiments:
            for model in model_order:
                for mode in modes:
                    val = results_rmse[pdk][experiment][model][mode]
                    if not np.isnan(val):
                        overall_rmse_scaled[model][mode].append(val * scale)

    overall_avg_rmse = {}
    for model in model_order:
        overall_avg_rmse[model] = {}
        for mode in modes:
            vals = overall_rmse_scaled[model][mode]
            overall_avg_rmse[model][mode] = np.mean(vals) if vals else np.nan

    # Format function for NRMSE (%)
    def fmt_combined(inter_val, extra_val):
        inter_str = f"{inter_val:.2f}" if not np.isnan(inter_val) else "XX.XX"
        extra_str = f"{extra_val:.2f}" if not np.isnan(extra_val) else "XX.XX"
        return f"{inter_str}/{extra_str}"

    # Format function for RMSE with PDK-specific scaling (TSMC: x1000, ASAP7: no scaling)
    def fmt_combined_rmse(inter_val, extra_val, pdk='TSMC'):
        scale = 1000 if pdk == 'TSMC' else 1
        inter_str = f"{inter_val*scale:.2f}" if not np.isnan(inter_val) else "XX.XX"
        extra_str = f"{extra_val*scale:.2f}" if not np.isnan(extra_val) else "XX.XX"
        return f"{inter_str}/{extra_str}"

    # Generate LaTeX table
    data_type_label = "Cell delay" if data_type == 'cell' else "output transition"
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Category-averaged NRMSE (\%) across both PDKs (Interpolation / Extrapolation). " + data_type_label + r" results are shown.}")
    lines.append(r"\label{tab:category_averaged_" + data_type + "}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\footnotesize")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{ll|cccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{PDK} & \textbf{Scenario}")
    lines.append(r"& \textbf{Aadam} & \textbf{GCN Base} & \textbf{MLP\_MAML} & \textbf{GCN\_MAML} \\")
    lines.append(r"\midrule")

    for pdk_idx, pdk in enumerate(pdks):
        pdk_display = pdk_labels[pdk]

        for exp_idx, experiment in enumerate(experiments):
            exp_display = exp_labels[experiment]

            # Get values
            aadam_str = fmt_combined(results[pdk][experiment]['AADAM']['interpolation'],
                                     results[pdk][experiment]['AADAM']['extrapolation'])
            gcn_base_str = fmt_combined(results[pdk][experiment]['GCN_baseline']['interpolation'],
                                        results[pdk][experiment]['GCN_baseline']['extrapolation'])
            mlp_maml_str = fmt_combined(results[pdk][experiment]['MLP_MAML']['interpolation'],
                                        results[pdk][experiment]['MLP_MAML']['extrapolation'])
            gcn_maml_str = fmt_combined(results[pdk][experiment]['GCN_MAML']['interpolation'],
                                        results[pdk][experiment]['GCN_MAML']['extrapolation'])

            # First row of PDK gets multirow
            if exp_idx == 0:
                pdk_col = r"\multirow{2}{*}{" + pdk_display + "}"
            else:
                pdk_col = ""

            line = f"{pdk_col} & {exp_display} & {aadam_str} & {gcn_base_str} & {mlp_maml_str} & {gcn_maml_str} \\\\"
            lines.append(line)

        # Add midrule between PDKs
        if pdk_idx < len(pdks) - 1:
            lines.append(r"\midrule")

    # Overall average row
    lines.append(r"\midrule")
    overall_aadam = fmt_combined(overall_avg['AADAM']['interpolation'], overall_avg['AADAM']['extrapolation'])
    overall_gcn_base = fmt_combined(overall_avg['GCN_baseline']['interpolation'], overall_avg['GCN_baseline']['extrapolation'])
    overall_mlp_maml = fmt_combined(overall_avg['MLP_MAML']['interpolation'], overall_avg['MLP_MAML']['extrapolation'])
    overall_gcn_maml = fmt_combined(overall_avg['GCN_MAML']['interpolation'], overall_avg['GCN_MAML']['extrapolation'])
    lines.append(r"\multicolumn{2}{l|}{\textbf{Overall Average}}")
    lines.append(f"& {overall_aadam} & {overall_gcn_base} & {overall_mlp_maml} & {overall_gcn_maml} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    # Save table
    arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
    table_name = f'category_averaged_{data_type}{arch_suffix}.tex'
    table_path = os.path.join(output_dir, table_name)

    with open(table_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved LaTeX table: {table_path}")

    # Generate RMSE LaTeX table (ps units)
    # Format function for RMSE LaTeX (no scaling, raw ps values)
    def fmt_combined_rmse_latex(inter_val, extra_val):
        inter_str = f"{inter_val:.2f}" if not np.isnan(inter_val) else "N/A"
        extra_str = f"{extra_val:.2f}" if not np.isnan(extra_val) else "N/A"
        return f"{inter_str}/{extra_str}"

    rmse_lines = []
    rmse_lines.append(r"\begin{table}[t]")
    rmse_lines.append(r"\centering")
    if data_type == 'cell':
        rmse_lines.append(r"\caption{Category-averaged RMSE (ps) across both PDKs (Interpolation / Extrapolation). Cell delay results are shown; output transition results follow the same trend.}")
    else:
        rmse_lines.append(r"\caption{Category-averaged RMSE (ps) across both PDKs (Interpolation / Extrapolation). Output transition results are shown.}")
    rmse_lines.append(r"\label{tab:rmse_combined_" + data_type + "}")
    rmse_lines.append(r"\setlength{\tabcolsep}{3pt}")
    rmse_lines.append(r"\footnotesize")
    rmse_lines.append(r"\resizebox{\columnwidth}{!}{%")
    rmse_lines.append(r"\begin{tabular}{ll|cccc}")
    rmse_lines.append(r"\toprule")
    rmse_lines.append(r"\textbf{PDK} & \textbf{Scenario}")
    rmse_lines.append(r"& \textbf{Aadam} & \textbf{GCN Base} & \textbf{MLP\_MAML} & \textbf{GCN\_MAML} \\")
    rmse_lines.append(r"\midrule")

    for pdk_idx, pdk in enumerate(pdks):
        pdk_display = pdk_labels[pdk]

        for exp_idx, experiment in enumerate(experiments):
            exp_display = exp_labels[experiment]

            # Get RMSE values (raw ps, no scaling)
            aadam_str = fmt_combined_rmse_latex(results_rmse[pdk][experiment]['AADAM']['interpolation'],
                                                 results_rmse[pdk][experiment]['AADAM']['extrapolation'])
            gcn_base_str = fmt_combined_rmse_latex(results_rmse[pdk][experiment]['GCN_baseline']['interpolation'],
                                                    results_rmse[pdk][experiment]['GCN_baseline']['extrapolation'])
            mlp_maml_str = fmt_combined_rmse_latex(results_rmse[pdk][experiment]['MLP_MAML']['interpolation'],
                                                    results_rmse[pdk][experiment]['MLP_MAML']['extrapolation'])
            gcn_maml_str = fmt_combined_rmse_latex(results_rmse[pdk][experiment]['GCN_MAML']['interpolation'],
                                                    results_rmse[pdk][experiment]['GCN_MAML']['extrapolation'])

            # First row of PDK gets multirow
            if exp_idx == 0:
                pdk_col = r"\multirow{2}{*}{" + pdk_display + "}"
            else:
                pdk_col = ""

            line = f"{pdk_col} & {exp_display} & {aadam_str} & {gcn_base_str} & {mlp_maml_str} & {gcn_maml_str} \\\\"
            rmse_lines.append(line)

        # Add midrule between PDKs
        if pdk_idx < len(pdks) - 1:
            rmse_lines.append(r"\midrule")

    # Overall average row for RMSE
    rmse_lines.append(r"\midrule")
    overall_aadam_rmse_latex = fmt_combined_rmse_latex(overall_avg_rmse['AADAM']['interpolation'], overall_avg_rmse['AADAM']['extrapolation'])
    overall_gcn_base_rmse_latex = fmt_combined_rmse_latex(overall_avg_rmse['GCN_baseline']['interpolation'], overall_avg_rmse['GCN_baseline']['extrapolation'])
    overall_mlp_maml_rmse_latex = fmt_combined_rmse_latex(overall_avg_rmse['MLP_MAML']['interpolation'], overall_avg_rmse['MLP_MAML']['extrapolation'])
    overall_gcn_maml_rmse_latex = fmt_combined_rmse_latex(overall_avg_rmse['GCN_MAML']['interpolation'], overall_avg_rmse['GCN_MAML']['extrapolation'])
    rmse_lines.append(r"\multicolumn{2}{l|}{\textbf{Overall Average}}")
    rmse_lines.append(f"& {overall_aadam_rmse_latex} & {overall_gcn_base_rmse_latex} & {overall_mlp_maml_rmse_latex} & {overall_gcn_maml_rmse_latex} \\\\")

    rmse_lines.append(r"\bottomrule")
    rmse_lines.append(r"\end{tabular}%")
    rmse_lines.append(r"}")
    rmse_lines.append(r"\end{table}")

    # Save RMSE table
    rmse_table_name = f'category_averaged_rmse_{data_type}{arch_suffix}.tex'
    rmse_table_path = os.path.join(output_dir, rmse_table_name)

    with open(rmse_table_path, 'w') as f:
        f.write('\n'.join(rmse_lines))

    print(f"Saved RMSE LaTeX table: {rmse_table_path}")

    # Also print RMSE LaTeX to console
    print(f"\n--- RMSE LaTeX Table ({data_type_label}) ---")
    print('\n'.join(rmse_lines))
    print("--- End of RMSE LaTeX Table ---\n")

    # Also print to console
    print(f"\n{'='*80}")
    print(f"Category-Averaged NRMSE (%) - {data_type_label}")
    print(f"{'='*80}")
    print(f"{'PDK':<12} {'Scenario':<20} {'Aadam':<12} {'GCN Base':<12} {'MLP_MAML':<12} {'GCN_MAML':<12}")
    print("-" * 80)
    for pdk in pdks:
        for experiment in experiments:
            aadam_str = fmt_combined(results[pdk][experiment]['AADAM']['interpolation'],
                                     results[pdk][experiment]['AADAM']['extrapolation'])
            gcn_base_str = fmt_combined(results[pdk][experiment]['GCN_baseline']['interpolation'],
                                        results[pdk][experiment]['GCN_baseline']['extrapolation'])
            mlp_maml_str = fmt_combined(results[pdk][experiment]['MLP_MAML']['interpolation'],
                                        results[pdk][experiment]['MLP_MAML']['extrapolation'])
            gcn_maml_str = fmt_combined(results[pdk][experiment]['GCN_MAML']['interpolation'],
                                        results[pdk][experiment]['GCN_MAML']['extrapolation'])
            print(f"{pdk_labels[pdk]:<12} {exp_labels[experiment]:<20} {aadam_str:<12} {gcn_base_str:<12} {mlp_maml_str:<12} {gcn_maml_str:<12}")
    print("-" * 80)
    print(f"{'Overall':<12} {'':<20} {overall_aadam:<12} {overall_gcn_base:<12} {overall_mlp_maml:<12} {overall_gcn_maml:<12}")
    print("=" * 80)

    # Print RMSE table to console (PDK-specific scaling: TSMC x1000, ASAP7 no scaling)
    print(f"\n{'='*80}")
    print(f"Category-Averaged RMSE (TSMC: x1000, ASAP7: raw) - {data_type_label}")
    print(f"{'='*80}")
    print(f"{'PDK':<12} {'Scenario':<20} {'Aadam':<12} {'GCN Base':<12} {'MLP_MAML':<12} {'GCN_MAML':<12}")
    print("-" * 80)
    for pdk in pdks:
        for experiment in experiments:
            aadam_str = fmt_combined_rmse(results_rmse[pdk][experiment]['AADAM']['interpolation'],
                                          results_rmse[pdk][experiment]['AADAM']['extrapolation'], pdk=pdk)
            gcn_base_str = fmt_combined_rmse(results_rmse[pdk][experiment]['GCN_baseline']['interpolation'],
                                             results_rmse[pdk][experiment]['GCN_baseline']['extrapolation'], pdk=pdk)
            mlp_maml_str = fmt_combined_rmse(results_rmse[pdk][experiment]['MLP_MAML']['interpolation'],
                                             results_rmse[pdk][experiment]['MLP_MAML']['extrapolation'], pdk=pdk)
            gcn_maml_str = fmt_combined_rmse(results_rmse[pdk][experiment]['GCN_MAML']['interpolation'],
                                             results_rmse[pdk][experiment]['GCN_MAML']['extrapolation'], pdk=pdk)
            print(f"{pdk_labels[pdk]:<12} {exp_labels[experiment]:<20} {aadam_str:<12} {gcn_base_str:<12} {mlp_maml_str:<12} {gcn_maml_str:<12}")
    print("-" * 80)
    # Overall uses scaled average (TSMC x1000, ASAP7 x1 applied before averaging)
    def fmt_overall_rmse(inter_val, extra_val):
        inter_str = f"{inter_val:.2f}" if not np.isnan(inter_val) else "XX.XX"
        extra_str = f"{extra_val:.2f}" if not np.isnan(extra_val) else "XX.XX"
        return f"{inter_str}/{extra_str}"
    overall_aadam_rmse = fmt_overall_rmse(overall_avg_rmse['AADAM']['interpolation'], overall_avg_rmse['AADAM']['extrapolation'])
    overall_gcn_base_rmse = fmt_overall_rmse(overall_avg_rmse['GCN_baseline']['interpolation'], overall_avg_rmse['GCN_baseline']['extrapolation'])
    overall_mlp_maml_rmse = fmt_overall_rmse(overall_avg_rmse['MLP_MAML']['interpolation'], overall_avg_rmse['MLP_MAML']['extrapolation'])
    overall_gcn_maml_rmse = fmt_overall_rmse(overall_avg_rmse['GCN_MAML']['interpolation'], overall_avg_rmse['GCN_MAML']['extrapolation'])
    print(f"{'Overall':<12} {'(scaled avg)':<20} {overall_aadam_rmse:<12} {overall_gcn_base_rmse:<12} {overall_mlp_maml_rmse:<12} {overall_gcn_maml_rmse:<12}")
    print("=" * 80)

    return results, overall_avg


def generate_latex_table_per_pdk(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict, output_dir,
                                  aadam_iter=300000, mlp_maml_iter=300000, arch='', gcn_innerdiv=10, gcn_meta=16):
    """Generate per-PDK LaTeX tables with NRMSE and RMSE values per cell

    Creates one table per PDK/experiment/mode combination.
    Rows: Cell names
    Columns: NRMSE (4 models) | RMSE (4 models)
    """
    os.makedirs(output_dir, exist_ok=True)

    pdks = ['TSMC', 'ASAP7']
    experiments = ['topology_agnostic', 'intra_topology']
    data_types = ['cell', 'transition']
    modes = ['interpolation', 'extrapolation']
    model_order = ['AADAM', 'GCN_baseline', 'MLP_MAML', 'GCN_MAML']
    model_display = ['AADAM', 'GCN Base', 'MLP MAML', 'GCN MAML']

    pdk_labels = {
        'TSMC': 'Commercial 28nm',
        'ASAP7': 'Open-source 7nm'
    }

    experiment_labels = {
        'topology_agnostic': 'Cross-Topology',
        'intra_topology': 'Intra Topology'
    }

    data_type_labels = {
        'cell': 'Cell Delay',
        'transition': 'Transition Time'
    }

    for pdk in pdks:
        gcn_maml_df = gcn_maml_df_dict.get(pdk)
        gcn_baseline_df = gcn_baseline_df_dict.get(pdk)
        mlp_df = mlp_df_dict.get(pdk)

        # Filter DataFrames by PDK cell naming convention
        gcn_maml_df = _filter_df_by_pdk_cell(gcn_maml_df, pdk)
        gcn_baseline_df = _filter_df_by_pdk_cell(gcn_baseline_df, pdk)
        mlp_df = _filter_df_by_pdk_cell(mlp_df, pdk)

        for experiment in experiments:
            for data_type in data_types:
                for mode in modes:
                    # Find cells
                    cells_set = set()

                    if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                        filtered = gcn_maml_df[
                            (gcn_maml_df['experiment'] == experiment) &
                            (gcn_maml_df['data_type'] == data_type) &
                            (gcn_maml_df['mode'] == mode)
                        ]
                        if len(filtered) > 0:
                            cells_set.update(filtered['cell'].dropna().unique())

                    if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                        filtered = gcn_baseline_df[
                            (gcn_baseline_df['experiment'] == experiment) &
                            (gcn_baseline_df['data_type'] == data_type) &
                            (gcn_baseline_df['mode'] == mode)
                        ]
                        if len(filtered) > 0:
                            cells_set.update(filtered['cell'].dropna().unique())

                    if mlp_df is not None and len(mlp_df) > 0:
                        filtered = mlp_df[
                            (mlp_df['topology'] == experiment) &
                            (mlp_df['data_type'] == data_type) &
                            (mlp_df['mode'] == mode)
                        ]
                        if len(filtered) > 0:
                            cells_set.update(filtered['cell'].dropna().unique())

                    cells = sorted(cells_set)
                    cells = _filter_cells_by_pdk(cells, pdk)

                    if len(cells) == 0:
                        continue

                    # Collect data for each cell
                    cell_data = {}
                    for cell in cells:
                        cell_data[cell] = {'NRMSE': {}, 'RMSE': {}}

                        # AADAM
                        if mlp_df is not None and len(mlp_df) > 0:
                            aadam_data = mlp_df[
                                (mlp_df['cell'] == cell) &
                                (mlp_df['mode'] == mode) &
                                (mlp_df['model_type'] == 'AADAM') &
                                (mlp_df['iterations'] == aadam_iter) &
                                (mlp_df['topology'] == experiment) &
                                (mlp_df['data_type'] == data_type)
                            ]
                            if len(aadam_data) > 0:
                                cell_data[cell]['NRMSE']['AADAM'] = aadam_data['NRMSE'].mean()
                                if 'RMSE' in aadam_data.columns:
                                    cell_data[cell]['RMSE']['AADAM'] = aadam_data['RMSE'].mean()

                        # GCN baseline
                        if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                            baseline_data = gcn_baseline_df[
                                (gcn_baseline_df['cell'] == cell) &
                                (gcn_baseline_df['mode'] == mode) &
                                (gcn_baseline_df['experiment'] == experiment) &
                                (gcn_baseline_df['data_type'] == data_type)
                            ]
                            if len(baseline_data) > 0:
                                cell_data[cell]['NRMSE']['GCN_baseline'] = baseline_data['NRMSE'].mean()
                                if 'RMSE' in baseline_data.columns:
                                    cell_data[cell]['RMSE']['GCN_baseline'] = baseline_data['RMSE'].mean()

                        # MLP MAML
                        if mlp_df is not None and len(mlp_df) > 0:
                            maml_data = mlp_df[
                                (mlp_df['cell'] == cell) &
                                (mlp_df['mode'] == mode) &
                                (mlp_df['model_type'] == 'MLP_MAML') &
                                (mlp_df['topology'] == experiment) &
                                (mlp_df['data_type'] == data_type)
                            ]
                            required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                            if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                                maml_specific = maml_data[
                                    (maml_data['innerdiv'] == 100) &
                                    (maml_data['meta'] == 32) &
                                    (maml_data['layer_length'] == 40) &
                                    (maml_data['iterations'] == mlp_maml_iter)
                                ]
                                if len(maml_specific) > 0:
                                    cell_data[cell]['NRMSE']['MLP_MAML'] = maml_specific['NRMSE'].mean()
                                    if 'RMSE' in maml_specific.columns:
                                        cell_data[cell]['RMSE']['MLP_MAML'] = maml_specific['RMSE'].mean()

                        # GCN MAML
                        if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                            gcn_data = gcn_maml_df[
                                (gcn_maml_df['cell'] == cell) &
                                (gcn_maml_df['mode'] == mode) &
                                (gcn_maml_df['experiment'] == experiment) &
                                (gcn_maml_df['data_type'] == data_type)
                            ]
                            if len(gcn_data) > 0:
                                cell_data[cell]['NRMSE']['GCN_MAML'] = gcn_data['NRMSE'].mean()
                                if 'RMSE' in gcn_data.columns:
                                    cell_data[cell]['RMSE']['GCN_MAML'] = gcn_data['RMSE'].mean()

                    # Aggregate TSMC cells by logic function (average across drive strengths)
                    display_cells, cell_data = _aggregate_tsmc_cell_data_by_logic(cells, cell_data, pdk)

                    # Generate LaTeX table
                    lines = []
                    lines.append(r"\begin{table}[htbp]")
                    lines.append(r"\centering")
                    lines.append(r"\footnotesize")

                    mode_label = 'Interpolation' if mode == 'interpolation' else 'Extrapolation'
                    caption = f"{pdk_labels.get(pdk, pdk)} - {experiment_labels.get(experiment, experiment)} - {data_type_labels.get(data_type, data_type)} ({mode_label})"
                    lines.append(r"\caption{" + caption + r"}")
                    lines.append(r"\label{tab:" + f"{pdk.lower()}_{experiment}_{data_type}_{mode}" + r"}")

                    # Table structure: Cell | NRMSE (4 models) | RMSE (4 models)
                    # RMSE scaling: TSMC uses x1000, ASAP7 uses no scaling
                    rmse_scale = 1000 if pdk == 'TSMC' else 1
                    rmse_header = r"RMSE ($\times$1000)" if pdk == 'TSMC' else r"RMSE"

                    lines.append(r"\begin{tabular}{l|cccc|cccc}")
                    lines.append(r"\toprule")
                    lines.append(r" & \multicolumn{4}{c|}{\textbf{NRMSE (\%)}} & \multicolumn{4}{c}{\textbf{" + rmse_header + r"}} \\")
                    header = r"\textbf{Cell} & " + " & ".join([f"\\textbf{{{m}}}" for m in model_display]) + " & " + " & ".join([f"\\textbf{{{m}}}" for m in model_display]) + r" \\"
                    lines.append(header)
                    lines.append(r"\midrule")

                    def fmt(val):
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            return "-"
                        return f"{val:.2f}"

                    def fmt_rmse(val, scale=rmse_scale):
                        """Format RMSE with PDK-specific scaling (TSMC: x1000, ASAP7: no scaling)"""
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            return "-"
                        return f"{val * scale:.2f}"

                    for display_cell in display_cells:
                        # display_cells already cleaned and aggregated by _aggregate_tsmc_cell_data_by_logic
                        nrmse_vals = [fmt(cell_data[display_cell]['NRMSE'].get(m)) for m in model_order]
                        rmse_vals = [fmt_rmse(cell_data[display_cell]['RMSE'].get(m)) for m in model_order]

                        line = f"{display_cell} & " + " & ".join(nrmse_vals) + " & " + " & ".join(rmse_vals) + r" \\"
                        lines.append(line)

                    lines.append(r"\bottomrule")
                    lines.append(r"\end{tabular}")
                    lines.append(r"\end{table}")

                    # Save to file
                    arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
                    filename = f"latex_table_{pdk.lower()}_{experiment}_{data_type}_{mode}{arch_suffix}.tex"
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'w') as f:
                        f.write('\n'.join(lines))
                    print(f"Saved LaTeX table: {output_path}")


def generate_tsmc_combined_latex_table(gcn_maml_df_dict, mlp_df_dict, output_dir,
                                        mlp_maml_iter=300000, data_type='cell', arch='', gcn_innerdiv=10, gcn_meta=16,
                                        graph_mode='stage_aware'):
    """Generate TSMC combined LaTeX table with Intra + Agnostic cells

    Format:
    - Horizontal: Cell Delay (NRMSE MLP GCN, RMSE MLP GCN) | Output Transition (NRMSE MLP GCN, RMSE MLP GCN)
    - Each value: inter/extra format
    - Vertical: Intra-topology cells first, then topology-agnostic cells, then Total Avg

    Args:
        graph_mode: 'stage_aware' or 'full_graph' (default: 'stage_aware')
    """
    os.makedirs(output_dir, exist_ok=True)

    pdk = 'TSMC'
    gcn_maml_df = gcn_maml_df_dict.get(pdk)
    mlp_df = mlp_df_dict.get(pdk)

    if gcn_maml_df is None or len(gcn_maml_df) == 0:
        print(f"No GCN MAML data for {pdk}")
        return

    # Collect cells for each experiment (filter by graph_mode) - for both data types
    intra_cells_set = set()
    agnostic_cells_set = set()

    for experiment in ['intra_topology', 'topology_agnostic']:
        for dt in ['cell', 'transition']:
            filtered = gcn_maml_df[
                (gcn_maml_df['experiment'] == experiment) &
                (gcn_maml_df['data_type'] == dt) &
                (gcn_maml_df['graph_mode'] == graph_mode)
            ]
            if len(filtered) > 0:
                cells = filtered['cell'].dropna().unique()
                cells = [c for c in cells if _is_tsmc_cell(c)]
                if experiment == 'intra_topology':
                    intra_cells_set.update(cells)
                else:
                    agnostic_cells_set.update(cells)

    # Sort and clean cells
    intra_cells = sorted(intra_cells_set)
    agnostic_cells = sorted(agnostic_cells_set)


    if len(intra_cells) == 0 and len(agnostic_cells) == 0:
        print(f"No cells found for TSMC {graph_mode}")
        return

    # Collect data for each cell - for both data types
    modes = ['interpolation', 'extrapolation']
    data_types = ['cell', 'transition']
    model_order = ['MLP_MAML', 'GCN_MAML']

    def collect_cell_data(cells, experiment):
        cell_data = {}
        for cell in cells:
            cell_data[cell] = {dt: {mode: {'NRMSE': {}, 'RMSE': {}} for mode in modes} for dt in data_types}

            for dt in data_types:
                for mode in modes:
                    # MLP MAML
                    if mlp_df is not None and len(mlp_df) > 0:
                        maml_data = mlp_df[
                            (mlp_df['cell'] == cell) &
                            (mlp_df['mode'] == mode) &
                            (mlp_df['model_type'] == 'MLP_MAML') &
                            (mlp_df['data_type'] == dt)
                        ]
                        required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                        if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                            maml_specific = maml_data[
                                (maml_data['innerdiv'] == 100) &
                                (maml_data['meta'] == 32) &
                                (maml_data['layer_length'] == 40) &
                                (maml_data['iterations'] == mlp_maml_iter)
                            ]
                            if len(maml_specific) > 0:
                                cell_data[cell][dt][mode]['NRMSE']['MLP_MAML'] = maml_specific['NRMSE'].mean()
                                if 'RMSE' in maml_specific.columns:
                                    cell_data[cell][dt][mode]['RMSE']['MLP_MAML'] = maml_specific['RMSE'].mean()

                    # GCN MAML (filter by graph_mode)
                    if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                        gcn_data = gcn_maml_df[
                            (gcn_maml_df['cell'] == cell) &
                            (gcn_maml_df['mode'] == mode) &
                            (gcn_maml_df['experiment'] == experiment) &
                            (gcn_maml_df['data_type'] == dt) &
                            (gcn_maml_df['graph_mode'] == graph_mode)
                        ]
                        if len(gcn_data) > 0:
                            cell_data[cell][dt][mode]['NRMSE']['GCN_MAML'] = gcn_data['NRMSE'].mean()
                            if 'RMSE' in gcn_data.columns:
                                cell_data[cell][dt][mode]['RMSE']['GCN_MAML'] = gcn_data['RMSE'].mean()

        return cell_data

    intra_data = collect_cell_data(intra_cells, 'intra_topology')
    agnostic_data = collect_cell_data(agnostic_cells, 'topology_agnostic')

    # Aggregate by logic function (average across drive strengths)
    def aggregate_cell_data(cells, cell_data):
        from collections import defaultdict
        cleaned_to_cells = defaultdict(list)
        for cell in cells:
            cleaned = _clean_tsmc_cell_name(cell)
            cleaned_to_cells[cleaned].append(cell)

        seen = set()
        unique_cleaned = []
        for cell in cells:
            cleaned = _clean_tsmc_cell_name(cell)
            if cleaned not in seen:
                seen.add(cleaned)
                unique_cleaned.append(cleaned)

        aggregated_data = {}
        for cleaned in unique_cleaned:
            original_cells = cleaned_to_cells[cleaned]
            aggregated_data[cleaned] = {dt: {mode: {'NRMSE': {}, 'RMSE': {}} for mode in modes} for dt in data_types}

            for dt in data_types:
                for mode in modes:
                    for metric in ['NRMSE', 'RMSE']:
                        for model in model_order:
                            values = []
                            for orig_cell in original_cells:
                                if orig_cell in cell_data:
                                    val = cell_data[orig_cell][dt][mode][metric].get(model)
                                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                        values.append(val)
                            if values:
                                aggregated_data[cleaned][dt][mode][metric][model] = np.mean(values)

        return unique_cleaned, aggregated_data

    intra_cleaned, intra_agg = aggregate_cell_data(intra_cells, intra_data)
    agnostic_cleaned, agnostic_agg = aggregate_cell_data(agnostic_cells, agnostic_data)

    # Generate LaTeX table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")

    graph_mode_label = 'Stage-Aware' if graph_mode == 'stage_aware' else 'Full Graph'
    caption = f"Commercial 28nm ({graph_mode_label}) - MAML Results (Interpolation/Extrapolation)"
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{tab:tsmc_combined_" + graph_mode + r"}")

    # Table structure:
    # Cell | Cell Delay (NRMSE MLP GCN, RMSE MLP GCN) | Transition (NRMSE MLP GCN, RMSE MLP GCN)
    # Each value: inter/extra
    lines.append(r"\begin{tabular}{l|cc|cc|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c|}{\textbf{Cell Delay}} & \multicolumn{4}{c}{\textbf{Output Transition}} \\")
    lines.append(r" & \multicolumn{2}{c|}{NRMSE (\%)} & \multicolumn{2}{c|}{RMSE ($\times$1000)} & \multicolumn{2}{c|}{NRMSE (\%)} & \multicolumn{2}{c}{RMSE ($\times$1000)} \\")
    lines.append(r"\textbf{Cell} & MLP & GCN & MLP & GCN & MLP & GCN & MLP & GCN \\")
    lines.append(r"\midrule")

    def fmt(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        return f"{val:.2f}"

    def fmt_rmse(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        return f"{val * 1000:.2f}"

    def fmt_pair(inter_val, extra_val, is_rmse=False):
        """Format as inter/extra pair"""
        if is_rmse:
            inter_str = fmt_rmse(inter_val)
            extra_str = fmt_rmse(extra_val)
        else:
            inter_str = fmt(inter_val)
            extra_str = fmt(extra_val)
        return f"{inter_str}/{extra_str}"

    def write_cell_row(cell_name, data):
        # Cell Delay
        cell_nrmse_mlp = fmt_pair(data['cell']['interpolation']['NRMSE'].get('MLP_MAML'),
                                   data['cell']['extrapolation']['NRMSE'].get('MLP_MAML'))
        cell_nrmse_gcn = fmt_pair(data['cell']['interpolation']['NRMSE'].get('GCN_MAML'),
                                   data['cell']['extrapolation']['NRMSE'].get('GCN_MAML'))
        cell_rmse_mlp = fmt_pair(data['cell']['interpolation']['RMSE'].get('MLP_MAML'),
                                  data['cell']['extrapolation']['RMSE'].get('MLP_MAML'), is_rmse=True)
        cell_rmse_gcn = fmt_pair(data['cell']['interpolation']['RMSE'].get('GCN_MAML'),
                                  data['cell']['extrapolation']['RMSE'].get('GCN_MAML'), is_rmse=True)
        # Transition
        trans_nrmse_mlp = fmt_pair(data['transition']['interpolation']['NRMSE'].get('MLP_MAML'),
                                    data['transition']['extrapolation']['NRMSE'].get('MLP_MAML'))
        trans_nrmse_gcn = fmt_pair(data['transition']['interpolation']['NRMSE'].get('GCN_MAML'),
                                    data['transition']['extrapolation']['NRMSE'].get('GCN_MAML'))
        trans_rmse_mlp = fmt_pair(data['transition']['interpolation']['RMSE'].get('MLP_MAML'),
                                   data['transition']['extrapolation']['RMSE'].get('MLP_MAML'), is_rmse=True)
        trans_rmse_gcn = fmt_pair(data['transition']['interpolation']['RMSE'].get('GCN_MAML'),
                                   data['transition']['extrapolation']['RMSE'].get('GCN_MAML'), is_rmse=True)

        return f"{cell_name} & {cell_nrmse_mlp} & {cell_nrmse_gcn} & {cell_rmse_mlp} & {cell_rmse_gcn} & {trans_nrmse_mlp} & {trans_nrmse_gcn} & {trans_rmse_mlp} & {trans_rmse_gcn}" + r" \\"

    # Collect values for total average (separate by group)
    all_values = {dt: {mode: {metric: {model: [] for model in model_order} for metric in ['NRMSE', 'RMSE']} for mode in modes} for dt in data_types}
    intra_values = {dt: {mode: {metric: {model: [] for model in model_order} for metric in ['NRMSE', 'RMSE']} for mode in modes} for dt in data_types}
    agnostic_values = {dt: {mode: {metric: {model: [] for model in model_order} for metric in ['NRMSE', 'RMSE']} for mode in modes} for dt in data_types}

    def collect_values_for_avg(data_dict, target_values):
        for cell, data in data_dict.items():
            for dt in data_types:
                for mode in modes:
                    for metric in ['NRMSE', 'RMSE']:
                        for model in model_order:
                            val = data[dt][mode][metric].get(model)
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                target_values[dt][mode][metric][model].append(val)
                                all_values[dt][mode][metric][model].append(val)

    def calc_avg(values_list):
        if values_list:
            return np.mean(values_list)
        return None

    def calc_avg_data(values_dict):
        avg_data = {dt: {mode: {'NRMSE': {}, 'RMSE': {}} for mode in modes} for dt in data_types}
        for dt in data_types:
            for mode in modes:
                for metric in ['NRMSE', 'RMSE']:
                    for model in model_order:
                        avg_data[dt][mode][metric][model] = calc_avg(values_dict[dt][mode][metric][model])
        return avg_data

    def write_avg_row(label, avg_data):
        cell_nrmse_mlp = fmt_pair(avg_data['cell']['interpolation']['NRMSE'].get('MLP_MAML'),
                                   avg_data['cell']['extrapolation']['NRMSE'].get('MLP_MAML'))
        cell_nrmse_gcn = fmt_pair(avg_data['cell']['interpolation']['NRMSE'].get('GCN_MAML'),
                                   avg_data['cell']['extrapolation']['NRMSE'].get('GCN_MAML'))
        cell_rmse_mlp = fmt_pair(avg_data['cell']['interpolation']['RMSE'].get('MLP_MAML'),
                                  avg_data['cell']['extrapolation']['RMSE'].get('MLP_MAML'), is_rmse=True)
        cell_rmse_gcn = fmt_pair(avg_data['cell']['interpolation']['RMSE'].get('GCN_MAML'),
                                  avg_data['cell']['extrapolation']['RMSE'].get('GCN_MAML'), is_rmse=True)
        trans_nrmse_mlp = fmt_pair(avg_data['transition']['interpolation']['NRMSE'].get('MLP_MAML'),
                                    avg_data['transition']['extrapolation']['NRMSE'].get('MLP_MAML'))
        trans_nrmse_gcn = fmt_pair(avg_data['transition']['interpolation']['NRMSE'].get('GCN_MAML'),
                                    avg_data['transition']['extrapolation']['NRMSE'].get('GCN_MAML'))
        trans_rmse_mlp = fmt_pair(avg_data['transition']['interpolation']['RMSE'].get('MLP_MAML'),
                                   avg_data['transition']['extrapolation']['RMSE'].get('MLP_MAML'), is_rmse=True)
        trans_rmse_gcn = fmt_pair(avg_data['transition']['interpolation']['RMSE'].get('GCN_MAML'),
                                   avg_data['transition']['extrapolation']['RMSE'].get('GCN_MAML'), is_rmse=True)
        return r"\textbf{" + label + r"} & " + f"{cell_nrmse_mlp} & {cell_nrmse_gcn} & {cell_rmse_mlp} & {cell_rmse_gcn} & {trans_nrmse_mlp} & {trans_nrmse_gcn} & {trans_rmse_mlp} & {trans_rmse_gcn}" + r" \\"

    # Intra-topology cells
    if len(intra_cleaned) > 0:
        lines.append(r"\multicolumn{9}{l}{\textit{Intra-Topology Cells}} \\")
        lines.append(r"\midrule")
        for cell in intra_cleaned:
            if cell in intra_agg:
                lines.append(write_cell_row(cell, intra_agg[cell]))
                collect_values_for_avg({cell: intra_agg[cell]}, intra_values)
        # Intra Avg row
        lines.append(r"\cmidrule{1-9}")
        intra_avg_data = calc_avg_data(intra_values)
        lines.append(write_avg_row("Intra Avg", intra_avg_data))

    # Separator between groups
    if len(intra_cleaned) > 0 and len(agnostic_cleaned) > 0:
        lines.append(r"\midrule")

    # Topology-agnostic cells
    if len(agnostic_cleaned) > 0:
        lines.append(r"\multicolumn{9}{l}{\textit{Topology-Agnostic Cells}} \\")
        lines.append(r"\midrule")
        for cell in agnostic_cleaned:
            if cell in agnostic_agg:
                lines.append(write_cell_row(cell, agnostic_agg[cell]))
                collect_values_for_avg({cell: agnostic_agg[cell]}, agnostic_values)
        # Agnostic Avg row
        lines.append(r"\cmidrule{1-9}")
        agnostic_avg_data = calc_avg_data(agnostic_values)
        lines.append(write_avg_row("Agnostic Avg", agnostic_avg_data))

    # Total Average row
    lines.append(r"\midrule")
    total_avg_data = calc_avg_data(all_values)
    lines.append(write_avg_row("Total Avg", total_avg_data))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # Save to file
    arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
    filename = f"latex_table_tsmc_combined_{graph_mode}{arch_suffix}.tex"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved TSMC combined LaTeX table: {output_path}")


def generate_improvement_summary_table(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict, output_dir,
                                        aadam_iter=300000, mlp_maml_iter=300000, arch='', gcn_innerdiv=10, gcn_meta=16,
                                        graph_mode='stage_aware'):
    """Generate improvement summary table comparing MAML methods vs baselines.

    Shows:
    - GCN_MAML and MLP_MAML absolute NRMSE values (inter/extra)
    - Improvement ratios vs AADAM and GCN_baseline (×)
    - Separate summaries for each PDK (ASAP7 and TSMC)

    Format example from paper:
    "0.36%/0.68% (interpolation/extrapolation) for cell delay,
     outperforming the GCN baseline by 2.1×/2.6× and MLP_MAML by 1.3×/1.2×"
    """
    os.makedirs(output_dir, exist_ok=True)

    modes = ['interpolation', 'extrapolation']
    data_types = ['cell', 'transition']
    experiments = ['topology_agnostic', 'intra_topology']
    pdks = ['ASAP7', 'TSMC']
    pdk_labels = {'ASAP7': 'Open-source 7nm', 'TSMC': 'Commercial 28nm'}

    # Helper functions
    def fmt(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        return f"{val:.2f}"

    def fmt_ratio(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        return f"{val:.1f}$\\times$"

    def fmt_pair(inter, extra):
        return f"{fmt(inter)}/{fmt(extra)}"

    def fmt_ratio_pair(inter, extra):
        return f"{fmt_ratio(inter)}/{fmt_ratio(extra)}"

    def calc_improvement(baseline_val, our_val):
        if baseline_val is not None and our_val is not None and our_val > 0:
            return baseline_val / our_val
        return None

    # Store per-PDK results
    pdk_final_avg = {}

    for pdk in pdks:
        gcn_maml_df = gcn_maml_df_dict.get(pdk)
        gcn_baseline_df = gcn_baseline_df_dict.get(pdk) if gcn_baseline_df_dict else None
        mlp_df = mlp_df_dict.get(pdk)

        if gcn_maml_df is None or len(gcn_maml_df) == 0:
            print(f"No GCN MAML data for {pdk}")
            continue

        # Collect average values for each model/data_type/mode combination
        avg_values = {model: {dt: {mode: {'NRMSE': [], 'RMSE': []} for mode in modes} for dt in data_types}
                      for model in ['AADAM', 'GCN_baseline', 'MLP_MAML', 'GCN_MAML']}

        # Collect cells from all experiments
        all_cells = set()
        for exp in experiments:
            for dt in data_types:
                expected_pooling = 'output' if dt == 'transition' else 'mean'
                if gcn_maml_df is not None:
                    filtered = gcn_maml_df[
                        (gcn_maml_df['experiment'] == exp) &
                        (gcn_maml_df['data_type'] == dt) &
                        (gcn_maml_df['pooling'] == expected_pooling) &
                        (gcn_maml_df['graph_mode'] == graph_mode)
                    ]
                    cells = filtered['cell'].dropna().unique()
                    cells = _filter_cells_by_pdk(cells, pdk)
                    all_cells.update(cells)

        # Collect values for each cell
        for cell in all_cells:
            for dt in data_types:
                expected_pooling = 'output' if dt == 'transition' else 'mean'
                for mode in modes:
                    # AADAM
                    if mlp_df is not None and len(mlp_df) > 0:
                        aadam_data = mlp_df[
                            (mlp_df['cell'] == cell) &
                            (mlp_df['mode'] == mode) &
                            (mlp_df['model_type'] == 'AADAM') &
                            (mlp_df['iterations'] == aadam_iter) &
                            (mlp_df['data_type'] == dt)
                        ]
                        if len(aadam_data) > 0:
                            avg_values['AADAM'][dt][mode]['NRMSE'].append(aadam_data['NRMSE'].mean())
                            if 'RMSE' in aadam_data.columns:
                                avg_values['AADAM'][dt][mode]['RMSE'].append(aadam_data['RMSE'].mean())

                    # GCN_baseline
                    if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                        baseline_data = gcn_baseline_df[
                            (gcn_baseline_df['cell'] == cell) &
                            (gcn_baseline_df['mode'] == mode) &
                            (gcn_baseline_df['data_type'] == dt) &
                            (gcn_baseline_df['pooling'] == 'mean')  # GCN baseline always uses mean pooling
                        ]
                        if len(baseline_data) > 0:
                            avg_values['GCN_baseline'][dt][mode]['NRMSE'].append(baseline_data['NRMSE'].mean())
                            if 'RMSE' in baseline_data.columns:
                                avg_values['GCN_baseline'][dt][mode]['RMSE'].append(baseline_data['RMSE'].mean())

                    # MLP_MAML
                    if mlp_df is not None and len(mlp_df) > 0:
                        maml_data = mlp_df[
                            (mlp_df['cell'] == cell) &
                            (mlp_df['mode'] == mode) &
                            (mlp_df['model_type'] == 'MLP_MAML') &
                            (mlp_df['data_type'] == dt)
                        ]
                        required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                        if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                            maml_specific = maml_data[
                                (maml_data['innerdiv'] == 100) &
                                (maml_data['meta'] == 32) &
                                (maml_data['layer_length'] == 40) &
                                (maml_data['iterations'] == mlp_maml_iter)
                            ]
                            if len(maml_specific) > 0:
                                avg_values['MLP_MAML'][dt][mode]['NRMSE'].append(maml_specific['NRMSE'].mean())
                                if 'RMSE' in maml_specific.columns:
                                    avg_values['MLP_MAML'][dt][mode]['RMSE'].append(maml_specific['RMSE'].mean())

                    # GCN_MAML
                    if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                        gcn_data = gcn_maml_df[
                            (gcn_maml_df['cell'] == cell) &
                            (gcn_maml_df['mode'] == mode) &
                            (gcn_maml_df['data_type'] == dt) &
                            (gcn_maml_df['pooling'] == expected_pooling) &
                            (gcn_maml_df['graph_mode'] == graph_mode)
                        ]
                        if len(gcn_data) > 0:
                            avg_values['GCN_MAML'][dt][mode]['NRMSE'].append(gcn_data['NRMSE'].mean())
                            if 'RMSE' in gcn_data.columns:
                                avg_values['GCN_MAML'][dt][mode]['RMSE'].append(gcn_data['RMSE'].mean())

        # Calculate overall averages for this PDK
        final_avg = {model: {dt: {mode: {} for mode in modes} for dt in data_types}
                     for model in ['AADAM', 'GCN_baseline', 'MLP_MAML', 'GCN_MAML']}

        for model in ['AADAM', 'GCN_baseline', 'MLP_MAML', 'GCN_MAML']:
            for dt in data_types:
                for mode in modes:
                    for metric in ['NRMSE', 'RMSE']:
                        values = avg_values[model][dt][mode][metric]
                        if values:
                            final_avg[model][dt][mode][metric] = np.mean(values)
                        else:
                            final_avg[model][dt][mode][metric] = None

        pdk_final_avg[pdk] = final_avg

    # Generate LaTeX table (for TSMC only, as before)
    if 'TSMC' in pdk_final_avg:
        final_avg = pdk_final_avg['TSMC']
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\footnotesize")

        graph_mode_label = 'Stage-Aware' if graph_mode == 'stage_aware' else 'Full Graph'
        caption = f"Commercial 28nm ({graph_mode_label}) - MAML Improvement Summary (Inter/Extra)"
        lines.append(r"\caption{" + caption + r"}")
        lines.append(r"\label{tab:tsmc_improvement_" + graph_mode + r"}")

        # Table structure: Model | Cell Delay NRMSE | vs AADAM | vs GCN_base | Transition NRMSE | vs AADAM | vs GCN_base
        lines.append(r"\begin{tabular}{l|c|cc|c|cc}")
        lines.append(r"\toprule")
        lines.append(r" & \multicolumn{3}{c|}{\textbf{Cell Delay NRMSE (\%)}} & \multicolumn{3}{c}{\textbf{Output Transition NRMSE (\%)}} \\")
        lines.append(r"\textbf{Model} & Value & vs AADAM & vs GCN\_base & Value & vs AADAM & vs GCN\_base \\")
        lines.append(r"\midrule")

        # Write rows for each model
        for model, model_label in [('AADAM', 'AADAM'), ('GCN_baseline', 'GCN baseline'),
                                    ('MLP_MAML', 'MLP MAML'), ('GCN_MAML', 'GCN MAML')]:
            row_data = []
            row_data.append(f"\\textbf{{{model_label}}}")

            for dt in ['cell', 'transition']:
                inter_val = final_avg[model][dt]['interpolation'].get('NRMSE')
                extra_val = final_avg[model][dt]['extrapolation'].get('NRMSE')
                row_data.append(fmt_pair(inter_val, extra_val))

                # Improvement vs AADAM
                aadam_inter = final_avg['AADAM'][dt]['interpolation'].get('NRMSE')
                aadam_extra = final_avg['AADAM'][dt]['extrapolation'].get('NRMSE')
                if model == 'AADAM':
                    row_data.append("-")
                else:
                    imp_inter = calc_improvement(aadam_inter, inter_val)
                    imp_extra = calc_improvement(aadam_extra, extra_val)
                    row_data.append(fmt_ratio_pair(imp_inter, imp_extra))

                # Improvement vs GCN_baseline
                gcn_base_inter = final_avg['GCN_baseline'][dt]['interpolation'].get('NRMSE')
                gcn_base_extra = final_avg['GCN_baseline'][dt]['extrapolation'].get('NRMSE')
                if model == 'GCN_baseline':
                    row_data.append("-")
                else:
                    imp_inter = calc_improvement(gcn_base_inter, inter_val)
                    imp_extra = calc_improvement(gcn_base_extra, extra_val)
                    row_data.append(fmt_ratio_pair(imp_inter, imp_extra))

            lines.append(" & ".join(row_data) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        # Save to file
        arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
        filename = f"latex_table_tsmc_improvement_{graph_mode}{arch_suffix}.tex"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Saved improvement summary table: {output_path}")

    # Print per-PDK improvement summaries
    graph_mode_label = 'Stage-Aware' if graph_mode == 'stage_aware' else 'Full Graph'
    print(f"\n=== Improvement Summary ({graph_mode_label}) ===")

    for pdk in pdks:
        if pdk not in pdk_final_avg:
            continue

        final_avg = pdk_final_avg[pdk]
        print(f"\n--- {pdk} ({pdk_labels[pdk]}) ---")

        for dt in ['cell', 'transition']:
            dt_label = 'Cell Delay' if dt == 'cell' else 'Output Transition'
            print(f"\n{dt_label}:")
            for model in ['MLP_MAML', 'GCN_MAML']:
                inter_val = final_avg[model][dt]['interpolation'].get('NRMSE')
                extra_val = final_avg[model][dt]['extrapolation'].get('NRMSE')
                print(f"  {model}: {fmt(inter_val)}%/{fmt(extra_val)}% (inter/extra)")

                # vs AADAM
                aadam_inter = final_avg['AADAM'][dt]['interpolation'].get('NRMSE')
                aadam_extra = final_avg['AADAM'][dt]['extrapolation'].get('NRMSE')
                imp_inter = calc_improvement(aadam_inter, inter_val)
                imp_extra = calc_improvement(aadam_extra, extra_val)
                print(f"    vs AADAM: {fmt_ratio(imp_inter)}/{fmt_ratio(imp_extra)}")

                # vs GCN_baseline
                gcn_base_inter = final_avg['GCN_baseline'][dt]['interpolation'].get('NRMSE')
                gcn_base_extra = final_avg['GCN_baseline'][dt]['extrapolation'].get('NRMSE')
                imp_inter = calc_improvement(gcn_base_inter, inter_val)
                imp_extra = calc_improvement(gcn_base_extra, extra_val)
                print(f"    vs GCN_baseline: {fmt_ratio(imp_inter)}/{fmt_ratio(imp_extra)}")


def plot_pdk_average_comparison(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict, output_dir,
                                 aadam_iter=300000, mlp_maml_iter=300000, experiment='topology_agnostic', arch='', gcn_innerdiv=10, gcn_meta=16):
    """Generate 2x2 grid plot showing AVERAGE NRMSE: rows=PDK, columns=data_type

    Each subplot shows 4 overlapping bars: Extra (wider/dark) + Inter (narrower/light)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define colors: (light_for_inter, dark_for_extra)
    # Extrapolation: darker, wider (background) - usually worse/higher values
    # Interpolation: lighter, narrower (foreground) - usually better/lower values
    # MLP family: Green, GCN family: Blue
    model_colors = {
        'AADAM': ('#d4e8d4', '#b4d4b4'),           # Green family (MLP baseline)
        'GCN_baseline': ('#c8ddf0', '#A8C4E0'),    # Blue family (GCN baseline)
        'MLP_MAML': ('#90ee90', '#228b22'),        # Green family (MLP MAML)
        'GCN_MAML': ('#7eb8e0', '#1B5E91'),        # Blue family (GCN MAML)
    }

    pdk_labels = {
        'TSMC': 'Commercial 28nm',
        'ASAP7': 'Open-source 7nm'
    }

    data_type_labels = {
        'cell': 'Cell delay',
        'transition': 'Transition time'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pdks = ['TSMC', 'ASAP7']
    data_types = ['cell', 'transition']
    modes = ['interpolation', 'extrapolation']
    model_order = ['AADAM', 'GCN_baseline', 'MLP_MAML', 'GCN_MAML']

    for row_idx, pdk in enumerate(pdks):
        for col_idx, data_type in enumerate(data_types):
            ax = axes[row_idx, col_idx]

            gcn_maml_df = gcn_maml_df_dict.get(pdk)
            gcn_baseline_df = gcn_baseline_df_dict.get(pdk)
            mlp_df = mlp_df_dict.get(pdk)

            # Filter DataFrames by PDK cell naming convention
            gcn_maml_df = _filter_df_by_pdk_cell(gcn_maml_df, pdk)
            gcn_baseline_df = _filter_df_by_pdk_cell(gcn_baseline_df, pdk)
            mlp_df = _filter_df_by_pdk_cell(mlp_df, pdk)

            # Calculate averages for each model/mode
            avg_data = {model: {mode: np.nan for mode in modes} for model in model_order}
            n_cells_info = {}

            for model in model_order:
                for mode in modes:
                    if model == 'AADAM':
                        if mlp_df is not None and len(mlp_df) > 0:
                            aadam_data = mlp_df[
                                (mlp_df['topology'] == experiment) &
                                (mlp_df['data_type'] == data_type) &
                                (mlp_df['mode'] == mode) &
                                (mlp_df['model_type'] == 'AADAM') &
                                (mlp_df['iterations'] == aadam_iter)
                            ]
                            if len(aadam_data) > 0:
                                avg_data[model][mode] = aadam_data['NRMSE'].mean()
                                n_cells_info[f'{model}_{mode}'] = len(aadam_data['cell'].unique())

                    elif model == 'GCN_baseline':
                        if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
                            baseline_data = gcn_baseline_df[
                                (gcn_baseline_df['experiment'] == experiment) &
                                (gcn_baseline_df['data_type'] == data_type) &
                                (gcn_baseline_df['mode'] == mode)
                            ]
                            if len(baseline_data) > 0:
                                avg_data[model][mode] = baseline_data['NRMSE'].mean()
                                n_cells_info[f'{model}_{mode}'] = len(baseline_data['cell'].unique())

                    elif model == 'MLP_MAML':
                        if mlp_df is not None and len(mlp_df) > 0:
                            maml_data = mlp_df[
                                (mlp_df['topology'] == experiment) &
                                (mlp_df['data_type'] == data_type) &
                                (mlp_df['mode'] == mode) &
                                (mlp_df['model_type'] == 'MLP_MAML')
                            ]
                            required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                            if len(maml_data) > 0 and all(col in maml_data.columns for col in required_cols):
                                maml_specific = maml_data[
                                    (maml_data['innerdiv'] == 100) &
                                    (maml_data['meta'] == 32) &
                                    (maml_data['layer_length'] == 40) &
                                    (maml_data['iterations'] == mlp_maml_iter)
                                ]
                                if len(maml_specific) > 0:
                                    avg_data[model][mode] = maml_specific['NRMSE'].mean()
                                    n_cells_info[f'{model}_{mode}'] = len(maml_specific['cell'].unique())

                    elif model == 'GCN_MAML':
                        if gcn_maml_df is not None and len(gcn_maml_df) > 0:
                            gcn_data = gcn_maml_df[
                                (gcn_maml_df['experiment'] == experiment) &
                                (gcn_maml_df['data_type'] == data_type) &
                                (gcn_maml_df['mode'] == mode)
                            ]
                            if len(gcn_data) > 0:
                                avg_data[model][mode] = gcn_data['NRMSE'].mean()
                                n_cells_info[f'{model}_{mode}'] = len(gcn_data['cell'].unique())

            # Plot 4 overlapping bars
            n_models = len(model_order)
            bar_width_extra = 0.7  # Wider for extrapolation
            bar_width_inter = 0.45  # Narrower for interpolation
            x = np.arange(n_models)

            legend_handles = []
            legend_labels = []

            for model_idx, model in enumerate(model_order):
                light_color, dark_color = model_colors[model]

                extra_val = avg_data[model]['extrapolation']
                inter_val = avg_data[model]['interpolation']

                # Draw extrapolation first (background, wider, darker) - usually worse/higher
                bars_extra = ax.bar(x[model_idx], extra_val, bar_width_extra,
                                   color=dark_color, edgecolor='black', linewidth=0.5)

                # Draw interpolation on top (foreground, narrower, lighter) - usually better/lower
                bars_inter = ax.bar(x[model_idx], inter_val, bar_width_inter,
                                   color=light_color, edgecolor='black', linewidth=0.5)

                if row_idx == 0 and col_idx == 0:
                    model_label = model.replace('_', ' ')
                    legend_handles.append(bars_extra)
                    legend_labels.append(f'{model_label} (Extra)')
                    legend_handles.append(bars_inter)
                    legend_labels.append(f'{model_label} (Inter)')

                # Add value labels
                all_vals = [extra_val, inter_val]
                valid_vals = [v for v in all_vals if not np.isnan(v)]
                if valid_vals:
                    max_val = max(valid_vals)
                    offset = max_val * 0.03

                    if not np.isnan(extra_val):
                        ax.text(x[model_idx], extra_val + offset, f'{extra_val:.2f}%',
                               ha='center', va='bottom', fontsize=8, fontweight='bold', color=dark_color)
                    if not np.isnan(inter_val):
                        ax.text(x[model_idx], inter_val - offset * 2, f'{inter_val:.2f}%',
                               ha='center', va='top', fontsize=8, fontweight='bold', color='black')

            # Get n_cells (use first available)
            n_cells = list(n_cells_info.values())[0] if n_cells_info else 0

            ax.set_ylabel('Average NRMSE (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{pdk_labels.get(pdk, pdk)} - {data_type_labels.get(data_type, data_type)}\n({n_cells} cells)',
                        fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('_', '\n') for m in model_order], fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

            # Set y limit
            all_values = []
            for model in model_order:
                for mode in modes:
                    val = avg_data[model][mode]
                    if not np.isnan(val):
                        all_values.append(val)
            if all_values:
                ax.set_ylim(0, max(all_values) * 1.25)

    if legend_handles:
        axes[0, 0].legend(legend_handles, legend_labels, loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()

    arch_suffix = f'_{arch}_innerdiv{gcn_innerdiv}_meta{gcn_meta}' if arch else ''
    plot_name = f'pdk_average_comparison_{experiment}{arch_suffix}.png'
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {plot_path}")


def export_results(gcn_df, mlp_df, output_dir, filename_prefix):
    """Export results to CSV"""
    os.makedirs(output_dir, exist_ok=True)

    if gcn_df is not None and len(gcn_df) > 0:
        gcn_path = os.path.join(output_dir, f'{filename_prefix}_gcn_results.csv')
        gcn_df.to_csv(gcn_path, index=False)
        print(f"Exported GCN results: {gcn_path}")

    if mlp_df is not None and len(mlp_df) > 0:
        mlp_path = os.path.join(output_dir, f'{filename_prefix}_mlp_results.csv')
        mlp_df.to_csv(mlp_path, index=False)
        print(f"Exported MLP results: {mlp_path}")


def plot_pdk_geomean_summary(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                              output_dir, aadam_iter, mlp_maml_iter, arch_string,
                              sa_innerdiv, sa_meta):
    """
    Generate 2x1 subplot figure showing geometric mean NRMSE for all models.

    Layout:
    - 2x1 subplots: Commercial (TSMC) | ASAP7
    - Each subplot divided into 2 sections: Cross-Topology | Intra-Topology (separated by dashed line)
    - Each section has 16 bars: 4 groups x 4 models
    - Groups: Cell-Inter, Cell-Extra, Trans-Inter, Trans-Extra
    - Models: AADAM, MLP_MAML, GCN_Baseline, GCN_MAML (solid bars, color-coded)

    Args:
        gcn_maml_df_dict: dict of {pdk: DataFrame} for GCN MAML results
        gcn_baseline_df_dict: dict of {pdk: DataFrame} for GCN Baseline results
        mlp_df_dict: dict of {pdk: DataFrame} for MLP results
        output_dir: output directory
        aadam_iter: iteration count for AADAM baseline
        mlp_maml_iter: iteration count for MLP MAML
        arch_string: architecture string for filtering
        sa_innerdiv: innerdiv for stage_aware
        sa_meta: meta for stage_aware
    """
    from scipy.stats import gmean
    import matplotlib.gridspec as gridspec

    # Broken axis parameters
    break_point = 4.5  # Break between 4 and 5
    lower_max = 4.5    # Lower subplot: 0 to 4.5 (shows tick at 4)
    upper_min = 4.5    # Upper subplot: 4.5 to max (shows tick at 5)
    compression = 1.5  # How much to compress the upper region

    # Create figure with GridSpec: 2 rows (upper/lower) x 2 columns (PDKs)
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2.5], hspace=0)  # No gap between axes

    # Model configuration - ordered for ratio comparison (base, MAML pairs)
    # MLP family: Green, GCN family: Blue
    model_order = ['AADAM', 'MLP_MAML', 'GCN_Baseline', 'GCN_MAML']
    model_colors = {
        'AADAM': '#b4d4b4',        # Green family (MLP baseline)
        'MLP_MAML': '#228b22',     # Green family (MLP MAML)
        'GCN_Baseline': '#A8C4E0', # Blue family (GCN baseline)
        'GCN_MAML': '#1B5E91'      # Blue family (GCN MAML)
    }
    model_display_names = {
        'AADAM': 'MLP w/o MAML',
        'MLP_MAML': 'MLP_MAML',
        'GCN_Baseline': 'GCN w/o MAML',
        'GCN_MAML': 'GCN_MAML'
    }
    # Model pairs for ratio arrows (baseline_idx, maml_idx)
    ratio_pairs = [(0, 1), (2, 3)]  # (AADAM→MLP_MAML), (GCN_Baseline→GCN_MAML)

    # Experiment configuration
    experiments = [
        ('topology_agnostic', 'Cross-Topology'),
        ('intra_topology', 'Intra-Topology')
    ]

    # Group configuration
    groups = [
        ('Cell', 'interpolation', 'Cell-Inter'),
        ('Cell', 'extrapolation', 'Cell-Extra'),
        ('Trans', 'interpolation', 'Trans-Inter'),
        ('Trans', 'extrapolation', 'Trans-Extra')
    ]

    pdk_list = [('TSMC', 'Commercial'), ('ASAP7', 'ASAP7')]

    # Store all geomean values for determining upper y-limit
    all_geomean_data = {pdk: [] for pdk, _ in pdk_list}

    # Store geomean NRMSE values for summary output: {pdk: {experiment: {data_type: {mode: {model: value}}}}}
    geomean_summary = {pdk: {} for pdk, _ in pdk_list}

    for ax_idx, (pdk, pdk_label) in enumerate(pdk_list):
        # Create upper and lower axes for broken axis
        ax_upper = fig.add_subplot(gs[0, ax_idx])
        ax_lower = fig.add_subplot(gs[1, ax_idx], sharex=ax_upper)

        # Get data for this PDK
        gcn_maml_df = gcn_maml_df_dict.get(pdk)
        gcn_baseline_df = gcn_baseline_df_dict.get(pdk) if gcn_baseline_df_dict else None
        mlp_df = mlp_df_dict.get(pdk)

        # Layout parameters
        n_groups = len(groups)
        n_models = len(model_order)
        n_experiments = len(experiments)
        bar_width = 0.12
        group_width = n_models * bar_width + 0.08
        section_width = n_groups * group_width
        section_gap = 0.5  # Gap between Cross and Intra sections

        max_val = 0
        section_centers = []
        geomean_positions = {}  # Store (x_pos, geomean_val) for ratio arrows

        for e_idx, (experiment, exp_label) in enumerate(experiments):
            section_start = e_idx * (section_width + section_gap)
            section_centers.append(section_start + section_width / 2)

            for g_idx, (data_type_short, mode, group_label) in enumerate(groups):
                data_type = 'cell' if data_type_short == 'Cell' else 'transition'

                for m_idx, model in enumerate(model_order):
                    nrmse_values = []

                    if model == 'AADAM' and mlp_df is not None:
                        # AADAM (MLP Baseline)
                        filtered = mlp_df[
                            (mlp_df['model_type'] == 'AADAM') &
                            (mlp_df['data_type'] == data_type) &
                            (mlp_df['mode'] == mode) &
                            (mlp_df['iterations'] == aadam_iter)
                        ]
                        if 'topology' in mlp_df.columns:
                            filtered = filtered[filtered['topology'] == experiment]
                        elif 'experiment' in mlp_df.columns:
                            filtered = filtered[filtered['experiment'] == experiment]
                        nrmse_values = filtered['NRMSE'].dropna().tolist()

                    elif model == 'MLP_MAML' and mlp_df is not None:
                        # MLP MAML
                        filtered = mlp_df[
                            (mlp_df['model_type'] == 'MLP_MAML') &
                            (mlp_df['data_type'] == data_type) &
                            (mlp_df['mode'] == mode)
                        ]
                        if 'topology' in mlp_df.columns:
                            filtered = filtered[filtered['topology'] == experiment]
                        elif 'experiment' in mlp_df.columns:
                            filtered = filtered[filtered['experiment'] == experiment]
                        if 'iterations' in mlp_df.columns:
                            filtered = filtered[filtered['iterations'] == mlp_maml_iter]
                        nrmse_values = filtered['NRMSE'].dropna().tolist()

                    elif model == 'GCN_Baseline' and gcn_baseline_df is not None:
                        # GCN Baseline (full_graph) - always uses mean pooling
                        filtered = gcn_baseline_df[
                            (gcn_baseline_df['data_type'] == data_type) &
                            (gcn_baseline_df['mode'] == mode) &
                            (gcn_baseline_df['training_type'] == 'baseline')
                        ]
                        if 'pooling' in gcn_baseline_df.columns:
                            filtered = filtered[filtered['pooling'] == 'mean']
                        if 'experiment' in gcn_baseline_df.columns:
                            filtered = filtered[filtered['experiment'] == experiment]
                        if 'graph_mode' in gcn_baseline_df.columns:
                            filtered = filtered[filtered['graph_mode'] == 'full_graph']
                        nrmse_values = filtered['NRMSE'].dropna().tolist()

                    elif model == 'GCN_MAML' and gcn_maml_df is not None:
                        # GCN MAML (stage_aware)
                        expected_pooling = 'output' if data_type == 'transition' else 'mean'
                        filtered = gcn_maml_df[
                            (gcn_maml_df['data_type'] == data_type) &
                            (gcn_maml_df['mode'] == mode) &
                            (gcn_maml_df['training_type'] == 'maml') &
                            (gcn_maml_df['pooling'] == expected_pooling)
                        ]
                        if 'experiment' in gcn_maml_df.columns:
                            filtered = filtered[filtered['experiment'] == experiment]
                        if 'graph_mode' in gcn_maml_df.columns:
                            filtered = filtered[filtered['graph_mode'] == 'stage_aware']
                        if 'innerdiv' in gcn_maml_df.columns:
                            filtered = filtered[filtered['innerdiv'] == sa_innerdiv]
                        if 'meta' in gcn_maml_df.columns:
                            filtered = filtered[filtered['meta'] == sa_meta]
                        nrmse_values = filtered['NRMSE'].dropna().tolist()

                    # Debug: print entry count for each combination
                    print(f"  [{pdk}] {exp_label} | {group_label} | {model}: {len(nrmse_values)} entries")

                    # Calculate geometric mean
                    if len(nrmse_values) > 0:
                        geomean_val = gmean(nrmse_values)
                    else:
                        geomean_val = np.nan

                    # Calculate x position
                    x_pos = section_start + g_idx * group_width + m_idx * bar_width

                    # Store geomean data for ratio arrows: key = (e_idx, g_idx, m_idx)
                    geomean_positions[(e_idx, g_idx, m_idx)] = (x_pos, geomean_val)

                    # Store in summary dictionary for final output
                    if experiment not in geomean_summary[pdk]:
                        geomean_summary[pdk][experiment] = {}
                    if data_type not in geomean_summary[pdk][experiment]:
                        geomean_summary[pdk][experiment][data_type] = {}
                    if mode not in geomean_summary[pdk][experiment][data_type]:
                        geomean_summary[pdk][experiment][data_type][mode] = {}
                    geomean_summary[pdk][experiment][data_type][mode][model] = geomean_val

                    # Draw solid bar on BOTH axes (for broken axis effect)
                    if not np.isnan(geomean_val):
                        # Draw on lower axis (0 to break_point)
                        ax_lower.bar(x_pos, geomean_val, width=bar_width * 0.85,
                               color=model_colors[model], edgecolor='black', linewidth=0.8)
                        # Draw on upper axis (break_point to max) - only if value exceeds break point
                        if geomean_val > break_point:
                            ax_upper.bar(x_pos, geomean_val, width=bar_width * 0.85,
                                   color=model_colors[model], edgecolor='black', linewidth=0.8)
                        max_val = max(max_val, geomean_val)
                        all_geomean_data[pdk].append((x_pos, geomean_val, model))

        # Draw ratio arrows between model pairs (baseline → MAML)
        for e_idx in range(n_experiments):
            for g_idx in range(n_groups):
                for base_idx, maml_idx in ratio_pairs:
                    base_key = (e_idx, g_idx, base_idx)
                    maml_key = (e_idx, g_idx, maml_idx)

                    if base_key in geomean_positions and maml_key in geomean_positions:
                        base_x, base_val = geomean_positions[base_key]
                        maml_x, maml_val = geomean_positions[maml_key]

                        if not np.isnan(base_val) and not np.isnan(maml_val) and maml_val > 0:
                            ratio = base_val / maml_val

                            # Arrow position (between the two bars)
                            arrow_x = (base_x + maml_x) / 2

                            # Draw arrow based on where values are relative to break point
                            if base_val <= break_point and maml_val <= break_point:
                                # Both values in lower region - draw complete arrow
                                ax_lower.annotate('',
                                    xy=(arrow_x, maml_val + 0.1),
                                    xytext=(arrow_x, base_val - 0.1),
                                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5, shrinkA=0, shrinkB=0))
                                # Ratio text above the arrow
                                text_y = base_val + 0.2
                                ax_lower.text(arrow_x, text_y, f'×{ratio:.1f}',
                                    fontsize=8, fontweight='bold', color='#C0392B',
                                    va='bottom', ha='center')
                            elif base_val > break_point and maml_val <= break_point:
                                # Arrow spans across break - draw in two parts
                                # Upper part: from base to break point (line only, no arrow head)
                                ax_upper.plot([arrow_x, arrow_x], [base_val - 0.1, break_point],
                                    color='#C0392B', lw=1.5, solid_capstyle='butt')
                                # Lower part: from break point to maml (with arrow head)
                                ax_lower.annotate('',
                                    xy=(arrow_x, maml_val + 0.1),
                                    xytext=(arrow_x, break_point),
                                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5, shrinkA=0, shrinkB=0))
                                # Ratio text above the arrow (in upper region)
                                text_y = base_val + 0.3
                                ax_upper.text(arrow_x, text_y, f'×{ratio:.1f}',
                                    fontsize=8, fontweight='bold', color='#C0392B',
                                    va='bottom', ha='center')
                            else:
                                # Both values in upper region
                                ax_upper.annotate('',
                                    xy=(arrow_x, maml_val + 0.1),
                                    xytext=(arrow_x, base_val - 0.1),
                                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5, shrinkA=0, shrinkB=0))
                                # Ratio text above the arrow
                                text_y = base_val + 0.3
                                ax_upper.text(arrow_x, text_y, f'×{ratio:.1f}',
                                    fontsize=8, fontweight='bold', color='#C0392B',
                                    va='bottom', ha='center')

        # Add vertical dashed line between Cross and Intra sections on both axes
        actual_bar_area_end = (n_groups - 1) * group_width + n_models * bar_width
        second_section_start = section_width + section_gap
        divider_x = (actual_bar_area_end + second_section_start) / 2
        # PDK-specific left shift for better centering
        if pdk == 'ASAP7':
            divider_x -= 0.08
        elif pdk == 'TSMC':
            divider_x -= 0.05
        ax_lower.axvline(x=divider_x, color='black', linestyle='--', linewidth=2.0, alpha=0.8)
        ax_upper.axvline(x=divider_x, color='black', linestyle='--', linewidth=2.0, alpha=0.8)

        # Set x-axis labels on lower axis only
        all_group_centers = []
        all_group_labels = []
        for e_idx in range(n_experiments):
            section_start = e_idx * (section_width + section_gap)
            for g_idx in range(n_groups):
                center = section_start + g_idx * group_width + (n_models - 1) * bar_width / 2
                all_group_centers.append(center)
                all_group_labels.append(groups[g_idx][2])

        ax_lower.set_xticks(all_group_centers)
        ax_lower.set_xticklabels(all_group_labels, fontsize=12, fontweight='bold', rotation=20, ha='right')
        plt.setp(ax_upper.get_xticklabels(), visible=False)

        # Set y-axis limits for broken axis
        upper_max = max_val + compression if max_val > break_point else break_point + 2
        ax_lower.set_ylim(0, break_point)
        ax_upper.set_ylim(break_point, upper_max)

        # Styling for both axes
        from matplotlib.ticker import MultipleLocator, FixedLocator

        # Lower axis styling - show 0, 1, 2, 3, 4
        ax_lower.yaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4]))
        ax_lower.set_axisbelow(True)
        ax_lower.grid(True, alpha=0.5, axis='y', linestyle='--', linewidth=0.8, color='gray')
        ax_lower.tick_params(axis='y', labelsize=13)
        ax_lower.spines['bottom'].set_visible(True)
        ax_lower.spines['bottom'].set_linewidth(1.5)
        ax_lower.spines['left'].set_visible(True)
        ax_lower.spines['left'].set_linewidth(1.5)
        ax_lower.spines['top'].set_visible(False)
        ax_lower.spines['right'].set_visible(False)

        # Upper axis styling - show only odd numbers (5, 7, 9, ...)
        upper_ticks = [i for i in range(5, int(upper_max) + 1) if i % 2 == 1]
        ax_upper.yaxis.set_major_locator(FixedLocator(upper_ticks))
        ax_upper.set_axisbelow(True)
        ax_upper.grid(True, alpha=0.5, axis='y', linestyle='--', linewidth=0.8, color='gray')
        ax_upper.tick_params(axis='y', labelsize=13)
        ax_upper.spines['bottom'].set_visible(False)
        ax_upper.spines['left'].set_visible(True)
        ax_upper.spines['left'].set_linewidth(1.5)
        ax_upper.spines['top'].set_visible(False)
        ax_upper.spines['right'].set_visible(False)
        ax_upper.tick_params(axis='x', bottom=False)

        # Y-axis label only on left graphs
        if ax_idx == 0:
            ax_lower.set_ylabel('NRMSE(%)', fontsize=16, fontweight='bold')

        # Title on upper axis
        ax_upper.set_title(pdk_label, fontsize=22, fontweight='bold')

        # Add section labels at the top of upper axis
        for e_idx, (experiment, exp_label) in enumerate(experiments):
            y_label_pos = upper_max - 0.3
            ax_upper.text(section_centers[e_idx], y_label_pos, exp_label,
                   ha='center', va='top', fontsize=18, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

        # Add break indicators (proper sine wave squiggle) on Y-axis between upper and lower axes
        # Create smooth sine wave pattern
        wave_t = np.linspace(0, 2 * np.pi, 30)  # More points for smooth curve
        wave_amplitude = 0.008  # X amplitude (horizontal wiggle)
        wave_height = 0.05     # Y span of the wave

        # Wave on Y-axis at the break point
        wave_x = -0.012 + wave_amplitude * np.sin(wave_t)  # Centered on Y-axis left edge

        # Upper axis wave (at bottom)
        wave_y_upper = -0.02 + wave_height * (wave_t / (2 * np.pi))
        ax_upper.plot(wave_x, wave_y_upper, transform=ax_upper.transAxes,
                     color='black', clip_on=False, linewidth=1.5)

        # Lower axis wave (at top)
        wave_y_lower = 0.98 + wave_height * (wave_t / (2 * np.pi))
        ax_lower.plot(wave_x, wave_y_lower, transform=ax_lower.transAxes,
                     color='black', clip_on=False, linewidth=1.5)

    # Add legend (model colors only)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=model_colors[m], edgecolor='black',
                             label=model_display_names[m]) for m in model_order]

    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=13,
               bbox_to_anchor=(0.5, 1.06), frameon=False)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pdk_geomean_summary_combined.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    # ============================================================
    # Print comprehensive summary of all numerical values for paper
    # ============================================================
    print("\n" + "="*80)
    print("COMPREHENSIVE NUMERICAL SUMMARY FOR PAPER")
    print("="*80)

    # Define display names for clarity
    exp_display = {'topology_agnostic': 'Cross-Topology', 'intra_topology': 'Intra-Topology'}
    pdk_display = {'TSMC': 'Commercial', 'ASAP7': 'ASAP7'}

    # 1. GeoMean NRMSE Table
    print("\n" + "-"*80)
    print("1. GeoMean NRMSE (%) - All Categories")
    print("-"*80)
    print(f"{'PDK':<12} {'Scenario':<18} {'DataType':<12} {'Mode':<15} {'AADAM':>8} {'MLP_MAML':>10} {'GCN_Base':>10} {'GCN_MAML':>10}")
    print("-"*80)

    for pdk in ['TSMC', 'ASAP7']:
        for exp in ['topology_agnostic', 'intra_topology']:
            for dtype in ['cell', 'transition']:
                for mode in ['interpolation', 'extrapolation']:
                    if pdk in geomean_summary and exp in geomean_summary[pdk]:
                        if dtype in geomean_summary[pdk][exp] and mode in geomean_summary[pdk][exp][dtype]:
                            vals = geomean_summary[pdk][exp][dtype][mode]
                            aadam = vals.get('AADAM', np.nan)
                            mlp_maml = vals.get('MLP_MAML', np.nan)
                            gcn_base = vals.get('GCN_Baseline', np.nan)
                            gcn_maml = vals.get('GCN_MAML', np.nan)
                            print(f"{pdk_display[pdk]:<12} {exp_display[exp]:<18} {dtype:<12} {mode:<15} {aadam:>8.2f} {mlp_maml:>10.2f} {gcn_base:>10.2f} {gcn_maml:>10.2f}")

    # 2. Improvement Ratios
    print("\n" + "-"*80)
    print("2. Improvement Ratios (baseline / MAML)")
    print("-"*80)
    print(f"{'PDK':<12} {'Scenario':<18} {'DataType':<12} {'Mode':<15} {'MLP Ratio':>10} {'GCN Ratio':>10}")
    print("-"*80)

    for pdk in ['TSMC', 'ASAP7']:
        for exp in ['topology_agnostic', 'intra_topology']:
            for dtype in ['cell', 'transition']:
                for mode in ['interpolation', 'extrapolation']:
                    if pdk in geomean_summary and exp in geomean_summary[pdk]:
                        if dtype in geomean_summary[pdk][exp] and mode in geomean_summary[pdk][exp][dtype]:
                            vals = geomean_summary[pdk][exp][dtype][mode]
                            aadam = vals.get('AADAM', np.nan)
                            mlp_maml = vals.get('MLP_MAML', np.nan)
                            gcn_base = vals.get('GCN_Baseline', np.nan)
                            gcn_maml = vals.get('GCN_MAML', np.nan)
                            mlp_ratio = aadam / mlp_maml if mlp_maml > 0 else np.nan
                            gcn_ratio = gcn_base / gcn_maml if gcn_maml > 0 else np.nan
                            print(f"{pdk_display[pdk]:<12} {exp_display[exp]:<18} {dtype:<12} {mode:<15} {mlp_ratio:>10.2f}x {gcn_ratio:>10.2f}x")

    # 3. Paper text fill-in values (Cell delay focused)
    print("\n" + "-"*80)
    print("3. VALUES FOR PAPER TEXT (Cell Delay)")
    print("-"*80)

    for pdk in ['TSMC', 'ASAP7']:
        print(f"\n*** {pdk_display[pdk]} PDK ***")
        for exp in ['topology_agnostic', 'intra_topology']:
            print(f"\n  [{exp_display[exp]}]")
            if pdk in geomean_summary and exp in geomean_summary[pdk]:
                if 'cell' in geomean_summary[pdk][exp]:
                    inter_vals = geomean_summary[pdk][exp]['cell'].get('interpolation', {})
                    extra_vals = geomean_summary[pdk][exp]['cell'].get('extrapolation', {})

                    aadam_inter = inter_vals.get('AADAM', np.nan)
                    aadam_extra = extra_vals.get('AADAM', np.nan)
                    mlp_maml_inter = inter_vals.get('MLP_MAML', np.nan)
                    mlp_maml_extra = extra_vals.get('MLP_MAML', np.nan)
                    gcn_base_inter = inter_vals.get('GCN_Baseline', np.nan)
                    gcn_base_extra = extra_vals.get('GCN_Baseline', np.nan)
                    gcn_maml_inter = inter_vals.get('GCN_MAML', np.nan)
                    gcn_maml_extra = extra_vals.get('GCN_MAML', np.nan)

                    # MLP improvement
                    mlp_ratio_inter = aadam_inter / mlp_maml_inter if mlp_maml_inter > 0 else np.nan
                    mlp_ratio_extra = aadam_extra / mlp_maml_extra if mlp_maml_extra > 0 else np.nan

                    # GCN improvement
                    gcn_ratio_inter = gcn_base_inter / gcn_maml_inter if gcn_maml_inter > 0 else np.nan
                    gcn_ratio_extra = gcn_base_extra / gcn_maml_extra if gcn_maml_extra > 0 else np.nan

                    print(f"    MLP (AADAM):     {aadam_inter:.2f}% / {aadam_extra:.2f}% (inter/extra)")
                    print(f"    MLP_MAML:        {mlp_maml_inter:.2f}% / {mlp_maml_extra:.2f}%")
                    print(f"    MLP Improvement: {mlp_ratio_inter:.2f}x / {mlp_ratio_extra:.2f}x")
                    print(f"    GCN Baseline:    {gcn_base_inter:.2f}% / {gcn_base_extra:.2f}%")
                    print(f"    GCN_MAML:        {gcn_maml_inter:.2f}% / {gcn_maml_extra:.2f}%")
                    print(f"    GCN Improvement: {gcn_ratio_inter:.2f}x / {gcn_ratio_extra:.2f}x")

    # 4. Summary for specific paper sentences
    print("\n" + "-"*80)
    print("4. SPECIFIC VALUES FOR PAPER SENTENCES")
    print("-"*80)

    # Cross-topology cell delay for both PDKs combined summary
    print("\n[For Fig description - Cross-Topology Cell Delay]")
    for pdk in ['TSMC', 'ASAP7']:
        if pdk in geomean_summary and 'topology_agnostic' in geomean_summary[pdk]:
            if 'cell' in geomean_summary[pdk]['topology_agnostic']:
                inter = geomean_summary[pdk]['topology_agnostic']['cell'].get('interpolation', {})
                extra = geomean_summary[pdk]['topology_agnostic']['cell'].get('extrapolation', {})

                print(f"\n  {pdk_display[pdk]} Cross-Topology Cell Delay:")
                print(f"    AADAM: {inter.get('AADAM', np.nan):.2f}% → MLP_MAML: {inter.get('MLP_MAML', np.nan):.2f}% (×{inter.get('AADAM', 0)/inter.get('MLP_MAML', 1):.1f})")
                print(f"    GCN_Base: {inter.get('GCN_Baseline', np.nan):.2f}% → GCN_MAML: {inter.get('GCN_MAML', np.nan):.2f}% (×{inter.get('GCN_Baseline', 0)/inter.get('GCN_MAML', 1):.1f})")

    print("\n[For Fig description - Intra-Topology Cell Delay]")
    for pdk in ['TSMC', 'ASAP7']:
        if pdk in geomean_summary and 'intra_topology' in geomean_summary[pdk]:
            if 'cell' in geomean_summary[pdk]['intra_topology']:
                inter = geomean_summary[pdk]['intra_topology']['cell'].get('interpolation', {})
                extra = geomean_summary[pdk]['intra_topology']['cell'].get('extrapolation', {})

                print(f"\n  {pdk_display[pdk]} Intra-Topology Cell Delay:")
                print(f"    AADAM: {inter.get('AADAM', np.nan):.2f}% → MLP_MAML: {inter.get('MLP_MAML', np.nan):.2f}% (×{inter.get('AADAM', 0)/inter.get('MLP_MAML', 1):.1f})")
                print(f"    GCN_Base: {inter.get('GCN_Baseline', np.nan):.2f}% → GCN_MAML: {inter.get('GCN_MAML', np.nan):.2f}% (×{inter.get('GCN_Baseline', 0)/inter.get('GCN_MAML', 1):.1f})")

    # Average MLP improvement
    print("\n[Average MLP Improvement Across All Categories]")
    mlp_ratios = []
    gcn_ratios = []
    for pdk in ['TSMC', 'ASAP7']:
        for exp in ['topology_agnostic', 'intra_topology']:
            for dtype in ['cell', 'transition']:
                for mode in ['interpolation', 'extrapolation']:
                    if pdk in geomean_summary and exp in geomean_summary[pdk]:
                        if dtype in geomean_summary[pdk][exp] and mode in geomean_summary[pdk][exp][dtype]:
                            vals = geomean_summary[pdk][exp][dtype][mode]
                            aadam = vals.get('AADAM', np.nan)
                            mlp_maml = vals.get('MLP_MAML', np.nan)
                            gcn_base = vals.get('GCN_Baseline', np.nan)
                            gcn_maml = vals.get('GCN_MAML', np.nan)
                            if not np.isnan(aadam) and not np.isnan(mlp_maml) and mlp_maml > 0:
                                mlp_ratios.append(aadam / mlp_maml)
                            if not np.isnan(gcn_base) and not np.isnan(gcn_maml) and gcn_maml > 0:
                                gcn_ratios.append(gcn_base / gcn_maml)

    if mlp_ratios:
        print(f"  Average MLP improvement (AADAM → MLP_MAML): ×{np.mean(mlp_ratios):.2f}")
    if gcn_ratios:
        print(f"  Average GCN improvement (GCN_Base → GCN_MAML): ×{np.mean(gcn_ratios):.2f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze single GCN architecture results and compare with MLP baselines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ASAP7 cell delay with conv64x2_fc256x2 architecture
  python analyze_gcn_single_arch.py --prefix ASAP7 --arch conv64x2_fc256x2 --data_type cell

  # Analyze TSMC transition with custom SA MAML parameters
  python analyze_gcn_single_arch.py --prefix TSMC --arch conv64x2_fc256x2 --data_type transition \\
      --sa_innerdiv 10 --sa_meta 16

  # Filter by experiment type
  python analyze_gcn_single_arch.py --prefix ASAP7 --arch conv64x2_fc256x2 --experiment topology_agnostic

  # Scale RMSE by 1000 (useful for TSMC data)
  python analyze_gcn_single_arch.py --prefix TSMC --arch conv64x2_fc256x2 --scale_rmse

  # Generate only PDK geomean summary plot (skip all other analysis)
  python analyze_gcn_single_arch.py --prefix ASAP7 --arch conv64x2_fc256x2 --pdk_geomean_only
        """
    )

    parser.add_argument('--gcn_dir', type=str,
                       default='../pretraining/model_test_code/gnn/data_result_npy_directory_final',
                       help='Directory containing GCN .npy result files')
    parser.add_argument('--mlp_maml_dir', type=str,
                       default='../pretraining/model_test_code/data_result_npy_directory_origin',
                       help='Directory containing MLP MAML .npy result files')
    parser.add_argument('--aadam_dir', type=str,
                       default='../pretraining/model_test_code/data_result_npy_directory',
                       help='Directory containing AADAM .npy result files')
    parser.add_argument('--output_dir', type=str, default='./result_summary/gcn_single_arch_analysis',
                       help='Output directory for plots and CSV')
    parser.add_argument('--prefix', type=str, required=True,
                       choices=['ASAP7', 'TSMC'],
                       help='Data prefix to filter (ASAP7 or TSMC)')
    parser.add_argument('--arch', type=str, required=True,
                       help='GCN architecture to analyze (e.g., conv64x2_fc256x2, conv128x2_fc128x2)')
    parser.add_argument('--experiment', type=str, default=None,
                       choices=['intra_topology', 'topology_agnostic'],
                       help='Filter by experiment type')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['interpolation', 'extrapolation'],
                       help='Filter by mode')
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Filter by data type: cell (delay) or transition (slew)')
    parser.add_argument('--aadam_iter', type=int, default=300000,
                       help='AADAM iteration for baseline comparison (default: 300000)')
    parser.add_argument('--aadam_adapt_method', type=str, default='adam',
                       choices=['selective_adam', 'adam'],
                       help='AADAM adaptation method: selective_adam (no suffix) or adam (_adam suffix) (default: selective_adam)')
    parser.add_argument('--mlp_maml_iter', type=int, default=300000,
                       help='MLP MAML iteration for comparison (default: 300000)')
    parser.add_argument('--gcn_iter', type=int, default=300000,
                       help='GCN iteration to filter (default: None = all iterations)')
    # innerdiv/meta only applies to Stage-Aware MAML (FG only has baseline without these)
    parser.add_argument('--sa_innerdiv', type=int, default=10,
                       help='Stage-Aware MAML innerdiv to filter (default: 10)')
    parser.add_argument('--sa_meta', type=int, default=16,
                       help='Stage-Aware MAML meta to filter (default: 16)')
    parser.add_argument('--gcn_adapt_method', type=str, default='selective_adam',
                       choices=['selective_adam', 'adam'],
                       help='GCN adaptation method: selective_adam (no suffix) or adam (_adam suffix) (default: selective_adam)')
    # Pooling is auto-determined: MAML+cell=mean, MAML+transition=output, baseline=mean
    parser.add_argument('--gnn_model_type', type=str, default='GCN',
                       choices=['GCN', 'GAT'],
                       help='Filter by GNN model type: GCN or GAT (default: all)')
    parser.add_argument('--training_type', type=str, default='maml',
                       choices=['maml', 'baseline'],
                       help='Filter by training type: maml or baseline (default: all)')
    parser.add_argument('--exclude_parasitic', action='store_true',
                       help='Exclude parasitic files from analysis')
    parser.add_argument('--no_mlp', action='store_true',
                       help='Disable MLP baseline comparison')
    parser.add_argument('--scale_rmse', action='store_true', default=False,
                       help='Scale RMSE by 1000 (for TSMC results)')
    parser.add_argument('--no_scale_rmse', action='store_true',
                       help='Do not scale RMSE by 1000 (default behavior)')
    parser.add_argument('--include_gcn_baseline', action='store_true',
                       help='Include GCN baseline (non-MAML) model in comparison')
    parser.add_argument('--gcn_baseline_iter', type=int, default=300000,
                       help='GCN baseline iteration to filter (default: 300000)')
    parser.add_argument('--vdd_only', action='store_true',
                       help='Filter for files with vdd_only suffix')
    parser.add_argument('--relpin', action='store_true',
                       help='Filter for files with relpin suffix')
    parser.add_argument('--pdk_compare', action='store_true',
                       help='Generate PDK comparison grid (2x2: rows=PDK, cols=data_type, 8 bars per cell)')
    parser.add_argument('--per_pdk_only', action='store_true',
                       help='Only generate per-PDK plots (skip single-prefix plots). Implies --pdk_compare')
    parser.add_argument('--pdk_geomean_only', action='store_true',
                       help='Only generate PDK geomean summary plot (skip all other analysis). Implies --pdk_compare and --include_gcn_baseline')

    args = parser.parse_args()

    # --per_pdk_only implies --pdk_compare
    if args.per_pdk_only:
        args.pdk_compare = True

    # --pdk_geomean_only implies --pdk_compare and --include_gcn_baseline
    if args.pdk_geomean_only:
        args.pdk_compare = True
        args.include_gcn_baseline = True

    # Determine scale_rmse setting
    scale_rmse = args.scale_rmse and not args.no_scale_rmse

    print("=" * 80)
    print("GCN SINGLE ARCHITECTURE ANALYSIS")
    print("=" * 80)
    print(f"GCN data directory: {args.gcn_dir}")
    print(f"MLP MAML data directory: {args.mlp_maml_dir}")
    print(f"AADAM data directory: {args.aadam_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Prefix filter: {args.prefix}")
    print(f"Architecture: {args.arch}")
    print(f"Data type: {args.data_type}")
    print(f"AADAM iteration: {args.aadam_iter}")
    print(f"AADAM adapt method: {args.aadam_adapt_method}")
    print(f"MLP MAML iteration: {args.mlp_maml_iter}")
    print(f"GCN adapt method: {args.gcn_adapt_method}")
    print(f"Scale RMSE (x1000): {scale_rmse}")
    if args.experiment:
        print(f"Experiment filter: {args.experiment}")
    if args.mode:
        print(f"Mode filter: {args.mode}")
    if args.gnn_model_type:
        print(f"GNN model type filter: {args.gnn_model_type}")
    if args.training_type:
        print(f"Training type filter: {args.training_type}")
    if args.gcn_iter:
        print(f"GCN iteration filter: {args.gcn_iter}")
    print(f"Stage-Aware MAML: innerdiv={args.sa_innerdiv}, meta={args.sa_meta}")
    print(f"Full-Graph: baseline only (no innerdiv/meta)")
    # Auto-determine pooling based on data_type and training_type
    if args.training_type == 'maml':
        expected_pooling = 'output' if args.data_type == 'transition' else 'mean'
    else:
        expected_pooling = 'mean'  # baseline always uses mean
    print(f"Pooling (auto): {expected_pooling} (based on data_type={args.data_type}, training_type={args.training_type})")
    if args.vdd_only:
        print(f"VDD only filter: True")
    if args.relpin:
        print(f"Related pin filter: True")
    if args.include_gcn_baseline:
        print(f"Include GCN baseline: True")
        print(f"GCN baseline iteration: {args.gcn_baseline_iter}")
    if args.pdk_geomean_only:
        print(f"PDK Geomean Only: True (skipping all other analysis)")
    print("=" * 80)

    # Print cell filter if set
    if CELL_FILTER:
        print(f"\nCell filter active (ASAP7 only): {CELL_FILTER}")

    # Initialize variables
    gcn_df = None
    mlp_df = None

    # Regular mode processing (skip if --pdk_geomean_only)
    if not args.pdk_geomean_only:
        # Load GCN results
        print("\nLoading GCN results...")
        # Determine gcn adapt suffix based on adaptation method
        gcn_adapt_suffix = '_adam' if args.gcn_adapt_method == 'adam' else ''
        gcn_df = load_gcn_results(args.gcn_dir, arch_filter=args.arch, gcn_adapt_suffix=gcn_adapt_suffix,
                                  cells_filter=CELL_FILTER)

    if gcn_df is not None and len(gcn_df) > 0:
        # Filter by prefix
        gcn_df = gcn_df[gcn_df['prefix'] == args.prefix]

        # Apply filters
        if args.experiment:
            gcn_df = gcn_df[gcn_df['experiment'] == args.experiment]
        if args.mode:
            gcn_df = gcn_df[gcn_df['mode'] == args.mode]
        if args.data_type:
            gcn_df = gcn_df[gcn_df['data_type'] == args.data_type]

        # Auto-determine pooling based on data_type and training_type
        # MAML: cell=mean, transition=output
        # baseline: always mean
        if args.training_type == 'maml':
            expected_pooling = 'output' if args.data_type == 'transition' else 'mean'
        else:
            expected_pooling = 'mean'
        gcn_df = gcn_df[gcn_df['pooling'] == expected_pooling]

        if args.gnn_model_type:
            gcn_df = gcn_df[gcn_df['gnn_model_type'] == args.gnn_model_type]
        if args.training_type:
            gcn_df = gcn_df[gcn_df['training_type'] == args.training_type]
        if args.exclude_parasitic:
            gcn_df = gcn_df[gcn_df['is_parasitic'] == False]
        if args.gcn_iter:
            gcn_df = gcn_df[gcn_df['iterations'] == args.gcn_iter]

        # Apply innerdiv/meta filter only for SA (MAML)
        # FG only has baseline which doesn't have innerdiv/meta
        fg_mask = (gcn_df['graph_mode'] == 'full_graph')
        sa_mask = (gcn_df['graph_mode'] == 'stage_aware') & \
                  (gcn_df['innerdiv'] == args.sa_innerdiv) & \
                  (gcn_df['meta'] == args.sa_meta)
        gcn_df = gcn_df[fg_mask | sa_mask]

        print(f"Filtered GCN results: {len(gcn_df)} entries")

        # Print mapping counts for each combination
        print("\n" + "=" * 80)
        print("GCN MAPPING COUNTS BY COMBINATION (DETAILED)")
        print("=" * 80)

        # Group by graph_mode, mode, experiment and count
        combination_counts = gcn_df.groupby(['graph_mode', 'mode', 'experiment']).size().reset_index(name='count')

        # Print header
        print(f"{'Graph Mode':<15} {'Mode':<15} {'Experiment':<20} {'#Cells':<10} {'Cells'}")
        print("-" * 80)

        # Sort for consistent output
        combination_counts = combination_counts.sort_values(['graph_mode', 'mode', 'experiment'])

        total_cells_all = set()
        for _, row in combination_counts.iterrows():
            # Get cells for this combination
            mask = (gcn_df['graph_mode'] == row['graph_mode']) & \
                   (gcn_df['mode'] == row['mode']) & \
                   (gcn_df['experiment'] == row['experiment'])
            cells_in_combo = sorted(gcn_df[mask]['cell'].dropna().unique())
            total_cells_all.update(cells_in_combo)
            cells_str = ', '.join(cells_in_combo) if len(cells_in_combo) <= 10 else f"{', '.join(cells_in_combo[:10])}... (+{len(cells_in_combo)-10} more)"
            print(f"{row['graph_mode']:<15} {row['mode']:<15} {row['experiment']:<20} {len(cells_in_combo):<10} {cells_str}")

        print("-" * 80)
        print(f"{'Total combinations:':<50} {len(combination_counts)}")
        print(f"{'Total entries:':<50} {len(gcn_df)}")
        print(f"{'Total unique cells:':<50} {len(total_cells_all)}")
        print(f"All cells: {sorted(total_cells_all)}")
        print("=" * 80 + "\n")
    else:
        print("No GCN results found - will plot MLP baselines only if available")
        gcn_df = None
        total_cells_all = set()

    # Load MLP results (skip if --pdk_geomean_only)
    if not args.pdk_geomean_only and not args.no_mlp:
        print("\nLoading MLP results...")
        # Determine aadam adapt suffix based on adaptation method
        aadam_adapt_suffix = '_adam' if args.aadam_adapt_method == 'adam' else ''
        mlp_df = load_mlp_results(args.mlp_maml_dir, aadam_dir=args.aadam_dir,
                                   prefix_filter=args.prefix,
                                   experiment_filter=args.experiment,
                                   data_type_filter=args.data_type,
                                   aadam_adapt_suffix=aadam_adapt_suffix,
                                   cells_filter=CELL_FILTER)

        if mlp_df is not None and len(mlp_df) > 0:
            if args.mode:
                mlp_df = mlp_df[mlp_df['mode'] == args.mode]
            print(f"Filtered MLP results: {len(mlp_df)} entries")

            # Print unique iterations for debugging
            print("\n" + "=" * 60)
            print("MLP UNIQUE ITERATIONS (BEFORE FILTERING)")
            print("=" * 60)
            for model_type in mlp_df['model_type'].unique():
                model_data = mlp_df[mlp_df['model_type'] == model_type]
                unique_iters = sorted(model_data['iterations'].unique())
                print(f"{model_type}: {unique_iters}")
            print("=" * 60)

            # Filter AADAM by aadam_iter
            if args.aadam_iter:
                aadam_mask = (mlp_df['model_type'] == 'AADAM') & (mlp_df['iterations'] == args.aadam_iter)
                non_aadam_mask = mlp_df['model_type'] != 'AADAM'
                mlp_df = mlp_df[aadam_mask | non_aadam_mask]
                print(f"Filtered AADAM by iterations={args.aadam_iter}")

            # Filter MLP_MAML by specific parameters (innerdiv=100, meta=32, layer_length=40, iterations=300000)
            maml_mask = mlp_df['model_type'] == 'MLP_MAML'
            if maml_mask.any():
                required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                if all(col in mlp_df.columns for col in required_cols):
                    maml_specific_mask = (
                        (mlp_df['model_type'] == 'MLP_MAML') &
                        (mlp_df['innerdiv'] == 100) &
                        (mlp_df['meta'] == 32) &
                        (mlp_df['layer_length'] == 40) &
                        (mlp_df['iterations'] == 300000)
                    )
                    non_maml_mask = mlp_df['model_type'] != 'MLP_MAML'
                    mlp_df = mlp_df[maml_specific_mask | non_maml_mask]
                    print(f"Filtered MLP_MAML by innerdiv=100, meta=32, layer_length=40, iterations=300000")

            print(f"Filtered MLP results after iteration filtering: {len(mlp_df)} entries")

            # Print mapping counts for MLP models
            print("\n" + "=" * 80)
            print("MLP MAPPING COUNTS BY MODEL TYPE (DETAILED)")
            print("=" * 80)

            # Group by model_type, mode, topology and count
            mlp_combination_counts = mlp_df.groupby(['model_type', 'mode', 'topology']).size().reset_index(name='count')

            # Print header
            print(f"{'Model Type':<15} {'Mode':<15} {'Experiment':<20} {'#Cells':<10} {'Cells'}")
            print("-" * 80)

            # Sort for consistent output
            mlp_combination_counts = mlp_combination_counts.sort_values(['model_type', 'mode', 'topology'])

            mlp_total_cells_all = set()
            for _, row in mlp_combination_counts.iterrows():
                # Get cells for this combination
                mask = (mlp_df['model_type'] == row['model_type']) & \
                       (mlp_df['mode'] == row['mode']) & \
                       (mlp_df['topology'] == row['topology'])
                cells_in_combo = sorted(mlp_df[mask]['cell'].dropna().unique())
                mlp_total_cells_all.update(cells_in_combo)
                cells_str = ', '.join(cells_in_combo) if len(cells_in_combo) <= 10 else f"{', '.join(cells_in_combo[:10])}... (+{len(cells_in_combo)-10} more)"
                print(f"{row['model_type']:<15} {row['mode']:<15} {row['topology']:<20} {len(cells_in_combo):<10} {cells_str}")

            print("-" * 80)
            print(f"{'Total combinations:':<50} {len(mlp_combination_counts)}")
            print(f"{'Total entries:':<50} {len(mlp_df)}")
            print(f"{'Total unique cells:':<50} {len(mlp_total_cells_all)}")
            print(f"All MLP cells: {sorted(mlp_total_cells_all)}")

            # Compare with GCN cells to find missing
            if len(total_cells_all) > 0:
                missing_in_mlp = total_cells_all - mlp_total_cells_all
                missing_in_gcn = mlp_total_cells_all - total_cells_all
                if missing_in_mlp:
                    print(f"\n⚠️  Cells in GCN but missing in MLP: {sorted(missing_in_mlp)}")
                if missing_in_gcn:
                    print(f"⚠️  Cells in MLP but missing in GCN: {sorted(missing_in_gcn)}")
                if not missing_in_mlp and not missing_in_gcn:
                    print(f"\n✅ All cells match between GCN and MLP")

            print("=" * 80 + "\n")

    # Load GCN baseline results (if requested, skip if --pdk_geomean_only)
    gcn_baseline_df = None
    if not args.pdk_geomean_only and args.include_gcn_baseline:
        print("\nLoading GCN baseline results...")
        # GCN baseline - no _adam suffix required (ASAP7 baseline files don't have _adam suffix)
        gcn_baseline_df = load_gcn_results(args.gcn_dir, arch_filter=args.arch, gcn_adapt_suffix='',
                                           vdd_only_filter=args.vdd_only if args.vdd_only else None,
                                           relpin_filter=args.relpin if args.relpin else None,
                                           cells_filter=CELL_FILTER)

        if gcn_baseline_df is not None and len(gcn_baseline_df) > 0:
            # Filter by prefix
            gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['prefix'] == args.prefix]
            # Filter by training_type = baseline
            gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['training_type'] == 'baseline']
            # Filter by graph_mode = full_graph (baseline is only for full_graph)
            gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['graph_mode'] == 'full_graph']

            # Apply same filters as GCN MAML
            if args.experiment:
                gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['experiment'] == args.experiment]
            if args.mode:
                gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['mode'] == args.mode]
            if args.data_type:
                gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['data_type'] == args.data_type]
            # Baseline always uses mean pooling
            gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['pooling'] == 'mean']
            if args.gnn_model_type:
                gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['gnn_model_type'] == args.gnn_model_type]
            if args.exclude_parasitic:
                gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['is_parasitic'] == False]
            if args.gcn_baseline_iter:
                gcn_baseline_df = gcn_baseline_df[gcn_baseline_df['iterations'] == args.gcn_baseline_iter]
            # Baseline doesn't have innerdiv/meta in filename - no filter needed

            print(f"Filtered GCN baseline results: {len(gcn_baseline_df)} entries")

            if len(gcn_baseline_df) > 0:
                # Print loaded files
                print("\n" + "=" * 80)
                print("GCN BASELINE LOADED FILES")
                print("=" * 80)
                loaded_files = sorted(gcn_baseline_df['filename'].unique())
                for f in loaded_files:
                    print(f"  {f}")
                print(f"\nTotal: {len(loaded_files)} unique files")

                # Print mapping counts
                print("\n" + "=" * 80)
                print("GCN BASELINE MAPPING COUNTS")
                print("=" * 80)
                baseline_combination_counts = gcn_baseline_df.groupby(['graph_mode', 'mode', 'experiment']).size().reset_index(name='count')
                print(f"{'Graph Mode':<15} {'Mode':<15} {'Experiment':<20} {'#Cells':<10}")
                print("-" * 80)
                for _, row in baseline_combination_counts.iterrows():
                    mask = (gcn_baseline_df['graph_mode'] == row['graph_mode']) & \
                           (gcn_baseline_df['mode'] == row['mode']) & \
                           (gcn_baseline_df['experiment'] == row['experiment'])
                    cells_in_combo = sorted(gcn_baseline_df[mask]['cell'].dropna().unique())
                    print(f"{row['graph_mode']:<15} {row['mode']:<15} {row['experiment']:<20} {len(cells_in_combo):<10}")
                print("=" * 80 + "\n")
            else:
                gcn_baseline_df = None
                print("No GCN baseline results found after filtering")
        else:
            print("No GCN baseline results found")

    # Check if we have any data to plot (skip check if --pdk_geomean_only)
    has_gcn = gcn_df is not None and len(gcn_df) > 0
    has_mlp = mlp_df is not None and len(mlp_df) > 0
    has_gcn_baseline = gcn_baseline_df is not None and len(gcn_baseline_df) > 0

    if not args.pdk_geomean_only and not has_gcn and not has_mlp:
        print("\n" + "=" * 80)
        print("ERROR: No data available to plot (neither GCN nor MLP)")
        print("=" * 80)
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate single-prefix plots (skip if --per_pdk_only or --pdk_geomean_only)
    if not args.per_pdk_only and not args.pdk_geomean_only:
        # Generate plots
        print("\nGenerating comparison plots...")
        plot_comparison(gcn_df, mlp_df, args.output_dir, args.prefix, args.aadam_iter, args.data_type, scale_rmse, gcn_baseline_df, args.arch, args.sa_innerdiv, args.sa_meta)


        # Generate average summary plots
        print("\nGenerating average summary plots...")
        plot_average_summary(gcn_df, mlp_df, args.output_dir, args.prefix, args.aadam_iter, args.data_type, scale_rmse, gcn_baseline_df, args.arch, args.sa_innerdiv, args.sa_meta)

        # Generate combined 2x2 summary plot
        print("\nGenerating combined 2x2 summary plot...")
        plot_combined_average_summary(gcn_df, mlp_df, args.output_dir, args.prefix, args.aadam_iter, args.data_type, scale_rmse, gcn_baseline_df, args.arch, args.sa_innerdiv, args.sa_meta)

    # Generate TSMC combined LaTeX table (for single TSMC prefix)
    # Reload data without filters to include both cell and transition data
    if not args.per_pdk_only and args.prefix == 'TSMC' and has_gcn:
        print("\nGenerating TSMC combined LaTeX tables...")
        # Reload GCN data without pooling filter for combined table
        gcn_combined_df = load_gcn_results(args.gcn_dir, arch_filter=args.arch, gcn_adapt_suffix=gcn_adapt_suffix,
                                           cells_filter=CELL_FILTER)
        if gcn_combined_df is not None and len(gcn_combined_df) > 0:
            gcn_combined_df = gcn_combined_df[gcn_combined_df['prefix'] == 'TSMC']
            if args.gnn_model_type:
                gcn_combined_df = gcn_combined_df[gcn_combined_df['gnn_model_type'] == args.gnn_model_type]
            if args.training_type:
                gcn_combined_df = gcn_combined_df[gcn_combined_df['training_type'] == args.training_type]
            if args.exclude_parasitic:
                gcn_combined_df = gcn_combined_df[gcn_combined_df['is_parasitic'] == False]
            if args.gcn_iter:
                gcn_combined_df = gcn_combined_df[gcn_combined_df['iterations'] == args.gcn_iter]
            # Apply innerdiv/meta filter only for SA (MAML)
            # FG only has baseline which doesn't have innerdiv/meta
            fg_mask = (gcn_combined_df['graph_mode'] == 'full_graph')
            sa_mask = (gcn_combined_df['graph_mode'] == 'stage_aware') & \
                      (gcn_combined_df['innerdiv'] == args.sa_innerdiv) & \
                      (gcn_combined_df['meta'] == args.sa_meta)
            gcn_combined_df = gcn_combined_df[fg_mask | sa_mask]
            # Don't filter by pooling - need both mean (cell) and output (transition)
            gcn_maml_df_dict_local = {'TSMC': gcn_combined_df}
        else:
            gcn_maml_df_dict_local = {'TSMC': gcn_df}

        # Reload MLP data without data_type filter for combined table
        aadam_adapt_suffix_local = '_adam' if args.aadam_adapt_method == 'adam' else ''
        mlp_combined_df = load_mlp_results(args.mlp_maml_dir, aadam_dir=args.aadam_dir,
                                           prefix_filter='TSMC',
                                           experiment_filter=args.experiment,
                                           data_type_filter=None,  # Load all data types
                                           aadam_adapt_suffix=aadam_adapt_suffix_local,
                                           cells_filter=CELL_FILTER)
        if mlp_combined_df is not None and len(mlp_combined_df) > 0:
            if args.mode:
                mlp_combined_df = mlp_combined_df[mlp_combined_df['mode'] == args.mode]
            mlp_df_dict_local = {'TSMC': mlp_combined_df}
        else:
            mlp_df_dict_local = {'TSMC': mlp_df} if has_mlp else {}

        for graph_mode in ['stage_aware', 'full_graph']:
            generate_tsmc_combined_latex_table(gcn_maml_df_dict_local, mlp_df_dict_local,
                                               args.output_dir, args.mlp_maml_iter, 'cell', args.arch, args.sa_innerdiv, args.sa_meta,
                                               graph_mode=graph_mode)

    # Generate PDK comparison grid if requested
    if args.pdk_compare:
        print("\n" + "=" * 80)
        print("PDK COMPARISON MODE - Loading data for both ASAP7 and TSMC")
        print("=" * 80)

        # Load data for both PDKs
        gcn_maml_df_dict = {}
        gcn_baseline_df_dict = {}
        mlp_df_dict = {}

        gcn_adapt_suffix = '_adam' if args.gcn_adapt_method == 'adam' else ''
        aadam_adapt_suffix = '_adam' if args.aadam_adapt_method == 'adam' else ''

        for pdk in ['ASAP7', 'TSMC']:
            print(f"\nLoading {pdk} data...")

            # GCN MAML
            pdk_gcn_df = load_gcn_results(args.gcn_dir, arch_filter=args.arch, gcn_adapt_suffix=gcn_adapt_suffix,
                                          cells_filter=CELL_FILTER)
            if pdk_gcn_df is not None and len(pdk_gcn_df) > 0:
                pdk_gcn_df = pdk_gcn_df[pdk_gcn_df['prefix'] == pdk]
                if args.gnn_model_type:
                    pdk_gcn_df = pdk_gcn_df[pdk_gcn_df['gnn_model_type'] == args.gnn_model_type]
                if args.training_type:
                    pdk_gcn_df = pdk_gcn_df[pdk_gcn_df['training_type'] == args.training_type]
                if args.gcn_iter:
                    pdk_gcn_df = pdk_gcn_df[pdk_gcn_df['iterations'] == args.gcn_iter]
                # Apply innerdiv/meta filter only for SA (MAML)
                # FG only has baseline which doesn't have innerdiv/meta
                fg_mask = (pdk_gcn_df['graph_mode'] == 'full_graph')
                sa_mask = (pdk_gcn_df['graph_mode'] == 'stage_aware') & \
                          (pdk_gcn_df['innerdiv'] == args.sa_innerdiv) & \
                          (pdk_gcn_df['meta'] == args.sa_meta)
                pdk_gcn_df = pdk_gcn_df[fg_mask | sa_mask]
                # Skip pooling filter for pdk_compare mode - will be applied based on data_type
                if args.exclude_parasitic:
                    pdk_gcn_df = pdk_gcn_df[pdk_gcn_df['is_parasitic'] == False]
                if len(pdk_gcn_df) > 0:
                    gcn_maml_df_dict[pdk] = pdk_gcn_df
                    print(f"  GCN MAML: {len(pdk_gcn_df)} entries")

            # GCN baseline (if requested)
            if args.include_gcn_baseline:
                pdk_gcn_baseline_df = load_gcn_results(args.gcn_dir, arch_filter=args.arch, gcn_adapt_suffix='',
                                                      vdd_only_filter=args.vdd_only if args.vdd_only else None,
                                                      relpin_filter=args.relpin if args.relpin else None,
                                                      cells_filter=CELL_FILTER)
                if pdk_gcn_baseline_df is not None and len(pdk_gcn_baseline_df) > 0:
                    pdk_gcn_baseline_df = pdk_gcn_baseline_df[pdk_gcn_baseline_df['prefix'] == pdk]
                    pdk_gcn_baseline_df = pdk_gcn_baseline_df[pdk_gcn_baseline_df['training_type'] == 'baseline']
                    pdk_gcn_baseline_df = pdk_gcn_baseline_df[pdk_gcn_baseline_df['graph_mode'] == 'full_graph']
                    if args.gcn_baseline_iter:
                        pdk_gcn_baseline_df = pdk_gcn_baseline_df[pdk_gcn_baseline_df['iterations'] == args.gcn_baseline_iter]
                    # Baseline doesn't have innerdiv/meta in filename - no filter needed
                    # Skip pooling filter for pdk_compare mode - baseline always uses mean
                    if args.exclude_parasitic:
                        pdk_gcn_baseline_df = pdk_gcn_baseline_df[pdk_gcn_baseline_df['is_parasitic'] == False]
                    if len(pdk_gcn_baseline_df) > 0:
                        gcn_baseline_df_dict[pdk] = pdk_gcn_baseline_df
                        print(f"  GCN baseline: {len(pdk_gcn_baseline_df)} entries")

            # MLP
            if not args.no_mlp:
                pdk_mlp_df = load_mlp_results(args.mlp_maml_dir, aadam_dir=args.aadam_dir,
                                              prefix_filter=pdk,
                                              experiment_filter=args.experiment,
                                              data_type_filter=None,  # Load all data types
                                              aadam_adapt_suffix=aadam_adapt_suffix,
                                              cells_filter=CELL_FILTER)
                if pdk_mlp_df is not None and len(pdk_mlp_df) > 0:
                    if args.mode:
                        pdk_mlp_df = pdk_mlp_df[pdk_mlp_df['mode'] == args.mode]

                    # Filter AADAM by aadam_iter (same as regular mode)
                    if args.aadam_iter:
                        aadam_mask = (pdk_mlp_df['model_type'] == 'AADAM') & (pdk_mlp_df['iterations'] == args.aadam_iter)
                        non_aadam_mask = pdk_mlp_df['model_type'] != 'AADAM'
                        pdk_mlp_df = pdk_mlp_df[aadam_mask | non_aadam_mask]

                    # Filter MLP_MAML by specific parameters (same as regular mode)
                    maml_mask = pdk_mlp_df['model_type'] == 'MLP_MAML'
                    if maml_mask.any():
                        required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                        if all(col in pdk_mlp_df.columns for col in required_cols):
                            maml_specific_mask = (
                                (pdk_mlp_df['model_type'] == 'MLP_MAML') &
                                (pdk_mlp_df['innerdiv'] == 100) &
                                (pdk_mlp_df['meta'] == 32) &
                                (pdk_mlp_df['layer_length'] == 40) &
                                (pdk_mlp_df['iterations'] == 300000)
                            )
                            non_maml_mask = pdk_mlp_df['model_type'] != 'MLP_MAML'
                            pdk_mlp_df = pdk_mlp_df[maml_specific_mask | non_maml_mask]

                    mlp_df_dict[pdk] = pdk_mlp_df
                    print(f"  MLP: {len(pdk_mlp_df)} entries")

        # Generate PDK comparison plots for each experiment type
        experiments_to_plot = [args.experiment] if args.experiment else ['topology_agnostic', 'intra_topology']

        # Skip other plots if --pdk_geomean_only
        if not args.pdk_geomean_only:
            for exp in experiments_to_plot:
                print(f"\nGenerating PDK comparison grid for {exp}...")
                plot_pdk_comparison_grid(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                         args.output_dir, args.aadam_iter, args.mlp_maml_iter, exp, args.arch, args.sa_innerdiv, args.sa_meta)

                print(f"Generating PDK average comparison for {exp}...")
                plot_pdk_average_comparison(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                            args.output_dir, args.aadam_iter, args.mlp_maml_iter, exp, args.arch, args.sa_innerdiv, args.sa_meta)

            # Generate per-PDK plots with all experiments (2x2: experiments × data_types)
            # GCN_MAML uses stage_aware, GCN_baseline uses full_graph (fixed)
            print(f"\nGenerating per-PDK all experiments plots...")
            plot_per_pdk_all_experiments(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                         args.output_dir, args.aadam_iter, args.mlp_maml_iter, args.arch, args.sa_innerdiv, args.sa_meta)

            # Generate category-averaged LaTeX tables (Interpolation/Extrapolation format)
            print(f"Generating category-averaged LaTeX tables...")
            for data_type in ['cell', 'transition']:
                generate_category_averaged_latex_table(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                                       args.output_dir, args.aadam_iter, args.mlp_maml_iter, args.arch, args.sa_innerdiv, args.sa_meta,
                                                       data_type=data_type)


            # Generate per-PDK LaTeX tables with NRMSE and RMSE per cell
            print("\nGenerating per-PDK LaTeX tables...")
            generate_latex_table_per_pdk(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                         args.output_dir, args.aadam_iter, args.mlp_maml_iter, args.arch, args.sa_innerdiv, args.sa_meta)

            # Generate TSMC combined LaTeX table (Intra + Agnostic, MLP vs GCN only)
            print("\nGenerating TSMC combined LaTeX tables...")
            for data_type in ['cell', 'transition']:
                for graph_mode in ['stage_aware', 'full_graph']:
                    generate_tsmc_combined_latex_table(gcn_maml_df_dict, mlp_df_dict,
                                                       args.output_dir, args.mlp_maml_iter, data_type, args.arch, args.sa_innerdiv, args.sa_meta,
                                                       graph_mode=graph_mode)

            # Generate improvement summary table (MAML vs baselines)
            print("\nGenerating improvement summary tables...")
            for graph_mode in ['stage_aware', 'full_graph']:
                generate_improvement_summary_table(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                                   args.output_dir, args.aadam_iter, args.mlp_maml_iter, args.arch, args.sa_innerdiv, args.sa_meta,
                                                   graph_mode=graph_mode)

        # Generate PDK geomean summary plot (includes both Cross & Intra topology)
        print("\nGenerating PDK geomean summary plot...")
        plot_pdk_geomean_summary(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                 args.output_dir, args.aadam_iter, args.mlp_maml_iter, args.arch, args.sa_innerdiv, args.sa_meta)

        # Generate RMSE tables (also for --pdk_geomean_only mode)
        if args.pdk_geomean_only:
            print("\nGenerating category-averaged LaTeX tables...")
            for data_type in ['cell', 'transition']:
                generate_category_averaged_latex_table(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                                       args.output_dir, args.aadam_iter, args.mlp_maml_iter, args.arch, args.sa_innerdiv, args.sa_meta,
                                                       data_type=data_type)
            print("\nGenerating per-PDK LaTeX tables...")
            generate_latex_table_per_pdk(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                                         args.output_dir, args.aadam_iter, args.mlp_maml_iter, args.arch, args.sa_innerdiv, args.sa_meta)

    # Export results (skip if --pdk_geomean_only since no regular data loaded)
    if not args.pdk_geomean_only:
        print("\nExporting results...")
        export_results(gcn_df, mlp_df, args.output_dir, f'{args.prefix.lower()}_{args.data_type}_{args.arch}_sa{args.sa_innerdiv}m{args.sa_meta}')

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
