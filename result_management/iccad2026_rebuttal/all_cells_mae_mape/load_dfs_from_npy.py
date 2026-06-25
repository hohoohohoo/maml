#!/usr/bin/env python3
"""
Loader module for ICCAD 2026 rebuttal — extracted verbatim from
result_management/compare_topology.ipynb cells 1-3.

DO NOT EDIT (mirrors notebook). Add MAE/MAPE/pessimism via wrapper, not here.
The only modification is the `calculate_metrics` function below which appends
MAE_ps_scaled, MAPE_pct, and signed-error percentiles. The rest is verbatim.
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import glob
import re
from collections import defaultdict
from types import SimpleNamespace

# ============================================================================
# CELL FILTER copied from notebook cell 0 (kept identical to the notebook view)
# ============================================================================
CELL_FILTER = ["AND2x6", "NAND3x2", "NOR2xp67", "OR2x6", "AO21x1", "AO32x1",
               "OAI22x1", "FAx1", "HAxp5", "XNOR2x2", "XOR2x2"]

# ===== BEGIN cell 1 (loaders) =====


def parse_gcn_filename(filename, gcn_adapt_suffix=''):
    """Parse GCN result filename to extract metadata"""
    basename = os.path.basename(filename)

    arch_pattern = r'conv(\d+)x(\d+)_fc(\d+)x(\d+)'
    arch_match = re.search(arch_pattern, basename)

    if not arch_match:
        return None

    conv_hidden_dim = int(arch_match.group(1))
    num_conv_layers = int(arch_match.group(2))
    fc_hidden_dim = int(arch_match.group(3))
    num_fc_layers = int(arch_match.group(4))

    if gcn_adapt_suffix:
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

    if basename.startswith('ASAP7'):
        result['prefix'] = 'ASAP7'
    elif basename.startswith('TSMC'):
        result['prefix'] = 'TSMC'
    else:
        result['prefix'] = 'unknown'

    if '_GCN_' in basename:
        result['gnn_model_type'] = 'GCN'
    elif '_GAT_' in basename:
        result['gnn_model_type'] = 'GAT'
    else:
        result['gnn_model_type'] = 'unknown'

    if '_maml_' in basename:
        result['training_type'] = 'maml'
    elif '_baseline_' in basename:
        result['training_type'] = 'baseline'
    else:
        result['training_type'] = 'unknown'

    if 'intra_topology' in basename:
        result['experiment'] = 'intra_topology'
    elif 'topology_agnostic' in basename:
        result['experiment'] = 'topology_agnostic'

    if '_cell_' in basename:
        result['data_type'] = 'cell'
        cell_pattern = r'(?:intra_topology|topology_agnostic)_(\w+)_cell_'
        cell_match = re.search(cell_pattern, basename)
        if cell_match:
            result['cell'] = cell_match.group(1)
    elif '_transition_' in basename:
        result['data_type'] = 'transition'
        cell_pattern = r'(?:intra_topology|topology_agnostic)_(\w+)_transition_'
        cell_match = re.search(cell_pattern, basename)
        if cell_match:
            result['cell'] = cell_match.group(1)

    if 'stage_aware' in basename:
        result['graph_mode'] = 'stage_aware'
    elif 'full_graph' in basename:
        result['graph_mode'] = 'full_graph'

    if '_interpolation_' in basename:
        result['mode'] = 'interpolation'
    elif '_extrapolation_' in basename:
        result['mode'] = 'extrapolation'

    iter_match = re.search(r'iter(\d+)', basename)
    if iter_match:
        result['iterations'] = int(iter_match.group(1))

    innerdiv_match = re.search(r'innerdiv(\d+)', basename)
    if innerdiv_match:
        result['innerdiv'] = int(innerdiv_match.group(1))

    meta_match = re.search(r'meta(\d+)', basename)
    if meta_match:
        result['meta'] = int(meta_match.group(1))

    pool_match = re.search(r'_pool(output|max|add|mean)', basename)
    if pool_match:
        result['pooling'] = pool_match.group(1)
    else:
        result['pooling'] = 'mean'

    result['is_parasitic'] = 'parasitic' in basename.lower()
    result['vdd_only'] = '_vddonly' in basename.lower() or '_vdd_only' in basename.lower()
    result['relpin'] = '_relpin' in basename.lower() or '_rel_pin' in basename.lower()

    return result


def parse_mlp_filename(filename, aadam_adapt_suffix=''):
    """Parse MLP result filename to extract metadata"""
    basename = os.path.basename(filename)

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

    aadam_pattern1_short = r'(\w+)_(intra|agnostic)_(\w+)_(cell|transition)_(extrapolation|interpolation)_(aadam|mlp)_(\d+)' + aadam_adapt_suffix + r'_(pred|act)\.npy'
    match = re.match(aadam_pattern1_short, basename)

    if match:
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

    maml_pattern2 = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_MAML_innerdiv(\d+)_meta(\d+)_layer(\d+)_(\d+)_(pred|act)\.npy'
    match = re.match(maml_pattern2, basename)

    if match:
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

    mae_groups = np.mean(np.abs(pred_grouped - act_grouped), axis=1)

    abs_diff = np.abs(pred_grouped - act_grouped)
    abs_sum = np.abs(pred_grouped) + np.abs(act_grouped)
    abs_sum = np.where(abs_sum > 1e-8, abs_sum, 1e-8)
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


def load_gcn_results(data_dir, arch_filter=None, gcn_adapt_suffix='',
                     vdd_only_filter=None, relpin_filter=None, cells_filter=None):
    """Load GCN results from directory"""
    results = []

    search_dirs = [data_dir]
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            search_dirs.append(item_path)

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

        act_file = pred_file.replace('_pred.npy', '_act.npy')

        if not os.path.exists(act_file):
            continue

        metadata = parse_gcn_filename(pred_file, gcn_adapt_suffix=gcn_adapt_suffix)
        if metadata is None:
            continue

        if arch_filter and metadata['arch_string'] != arch_filter:
            continue

        if vdd_only_filter is not None and metadata.get('vdd_only', False) != vdd_only_filter:
            continue

        if relpin_filter is not None and metadata.get('relpin', False) != relpin_filter:
            continue

        if cells_filter is not None and metadata.get('prefix') == 'ASAP7':
            if metadata.get('cell') not in cells_filter:
                continue

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

        except Exception:
            continue

    print(f"\nLoaded {len(results)} GCN results")
    return pd.DataFrame(results) if results else None


def load_mlp_results(mlp_maml_dir, aadam_dir=None, prefix_filter=None,
                     experiment_filter=None, data_type_filter=None,
                     aadam_adapt_suffix='', cells_filter=None):
    """Load MLP results from directories"""
    results = []

    if aadam_dir is None:
        aadam_dir = mlp_maml_dir

    all_files_maml = glob.glob(os.path.join(mlp_maml_dir, '*_pred.npy'))
    pred_files_maml = [f for f in all_files_maml if 'aadam_' not in os.path.basename(f).lower()]
    print(f"Found {len(pred_files_maml)} MLP_MAML prediction files...")

    if aadam_dir != mlp_maml_dir:
        all_files_aadam = glob.glob(os.path.join(aadam_dir, '*_pred.npy'))
        pred_files_aadam = [f for f in all_files_aadam if 'aadam_' in os.path.basename(f).lower()]
        print(f"Found {len(pred_files_aadam)} AADAM prediction files...")
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

        except Exception:
            continue

    print(f"Loaded {len(results)} MLP results")
    return pd.DataFrame(results) if results else None


def plot_pdk_geomean_summary(gcn_maml_df_dict, gcn_baseline_df_dict, mlp_df_dict,
                              output_dir, aadam_iter, mlp_maml_iter, arch_string,
                              sa_innerdiv, sa_meta):
    """
    Generate 2x1 subplot figure showing geometric mean NRMSE for all models.
    Uses broken axis with ~ style indicator to handle outliers.
    """
    from scipy.stats import gmean
    import matplotlib.gridspec as gridspec

    # Use global configuration for broken axis
    break_point = BREAK_POINT
    compression = COMPRESSION

    # Create figure with GridSpec
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[HEIGHT_RATIO_UPPER, HEIGHT_RATIO_LOWER], hspace=0.05)

    # Model configuration
    model_order = ['AADAM', 'MLP_MAML', 'GCN_Baseline', 'GCN_MAML']
    model_colors = {
        'AADAM': '#b4d4b4',
        'MLP_MAML': '#228b22',
        'GCN_Baseline': '#A8C4E0',
        'GCN_MAML': '#1B5E91'
    }
    model_display_names = {
        'AADAM': 'MLP w/o MAML',
        'MLP_MAML': 'MLP_MAML',
        'GCN_Baseline': 'GCN w/o MAML',
        'GCN_MAML': 'GCN_MAML'
    }
    ratio_pairs = [(0, 1), (2, 3)]

    experiments = [
        ('topology_agnostic', 'Cross-Topology'),
        ('intra_topology', 'Intra-Topology')
    ]

    groups = [
        ('Cell', 'interpolation', 'Cell-Inter'),
        ('Cell', 'extrapolation', 'Cell-Extra'),
        ('Trans', 'interpolation', 'Trans-Inter'),
        ('Trans', 'extrapolation', 'Trans-Extra')
    ]

    pdk_list = [('TSMC', 'Commercial'), ('ASAP7', 'ASAP7')]

    all_geomean_data = {pdk: [] for pdk, _ in pdk_list}
    geomean_summary = {pdk: {} for pdk, _ in pdk_list}

    for ax_idx, (pdk, pdk_label) in enumerate(pdk_list):
        ax_upper = fig.add_subplot(gs[0, ax_idx])
        ax_lower = fig.add_subplot(gs[1, ax_idx], sharex=ax_upper)

        gcn_maml_df = gcn_maml_df_dict.get(pdk)
        gcn_baseline_df = gcn_baseline_df_dict.get(pdk) if gcn_baseline_df_dict else None
        mlp_df = mlp_df_dict.get(pdk)

        n_groups = len(groups)
        n_models = len(model_order)
        n_experiments = len(experiments)
        bar_width = 0.12
        group_width = n_models * bar_width + 0.08
        section_width = n_groups * group_width
        section_gap = 0.5

        max_val = 0
        section_centers = []
        geomean_positions = {}

        for e_idx, (experiment, exp_label) in enumerate(experiments):
            section_start = e_idx * (section_width + section_gap)
            section_centers.append(section_start + section_width / 2)

            for g_idx, (data_type_short, mode, group_label) in enumerate(groups):
                data_type = 'cell' if data_type_short == 'Cell' else 'transition'

                for m_idx, model in enumerate(model_order):
                    nrmse_values = []

                    if model == 'AADAM' and mlp_df is not None:
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

                    print(f"  [{pdk}] {exp_label} | {group_label} | {model}: {len(nrmse_values)} entries")

                    if len(nrmse_values) > 0:
                        geomean_val = gmean(nrmse_values)
                    else:
                        geomean_val = np.nan

                    x_pos = section_start + g_idx * group_width + m_idx * bar_width
                    geomean_positions[(e_idx, g_idx, m_idx)] = (x_pos, geomean_val)

                    if experiment not in geomean_summary[pdk]:
                        geomean_summary[pdk][experiment] = {}
                    if data_type not in geomean_summary[pdk][experiment]:
                        geomean_summary[pdk][experiment][data_type] = {}
                    if mode not in geomean_summary[pdk][experiment][data_type]:
                        geomean_summary[pdk][experiment][data_type][mode] = {}
                    geomean_summary[pdk][experiment][data_type][mode][model] = geomean_val

                    if not np.isnan(geomean_val):
                        ax_lower.bar(x_pos, geomean_val, width=bar_width * 0.85,
                               color=model_colors[model], edgecolor='black', linewidth=0.8)
                        if geomean_val > break_point:
                            ax_upper.bar(x_pos, geomean_val, width=bar_width * 0.85,
                                   color=model_colors[model], edgecolor='black', linewidth=0.8)
                        max_val = max(max_val, geomean_val)
                        all_geomean_data[pdk].append((x_pos, geomean_val, model))

        # Draw ratio arrows between model pairs
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
                            arrow_x = (base_x + maml_x) / 2

                            if base_val <= break_point and maml_val <= break_point:
                                ax_lower.annotate('',
                                    xy=(arrow_x, maml_val + 0.1),
                                    xytext=(arrow_x, base_val - 0.1),
                                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5, shrinkA=0, shrinkB=0))
                                text_y = base_val + 0.2
                                ax_lower.text(arrow_x, text_y, f'x{ratio:.1f}',
                                    fontsize=8, fontweight='bold', color='#C0392B',
                                    va='bottom', ha='center')
                            elif base_val > break_point and maml_val <= break_point:
                                ax_upper.plot([arrow_x, arrow_x], [base_val - 0.1, break_point],
                                    color='#C0392B', lw=1.5, solid_capstyle='butt')
                                ax_lower.annotate('',
                                    xy=(arrow_x, maml_val + 0.1),
                                    xytext=(arrow_x, break_point),
                                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5, shrinkA=0, shrinkB=0))
                                text_y = base_val + 0.3
                                ax_upper.text(arrow_x, text_y, f'x{ratio:.1f}',
                                    fontsize=8, fontweight='bold', color='#C0392B',
                                    va='bottom', ha='center')
                            else:
                                ax_upper.annotate('',
                                    xy=(arrow_x, maml_val + 0.1),
                                    xytext=(arrow_x, base_val - 0.1),
                                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.5, shrinkA=0, shrinkB=0))
                                text_y = base_val + 0.3
                                ax_upper.text(arrow_x, text_y, f'x{ratio:.1f}',
                                    fontsize=8, fontweight='bold', color='#C0392B',
                                    va='bottom', ha='center')

        # Add vertical dashed line between Cross and Intra sections
        actual_bar_area_end = (n_groups - 1) * group_width + n_models * bar_width
        second_section_start = section_width + section_gap
        divider_x = (actual_bar_area_end + second_section_start) / 2
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
        from matplotlib.ticker import FixedLocator

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

        # Y-axis label only on left graph
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

        # ============================================================
        # Add "~" style break indicators on Y-axis
        # ============================================================
        # Draw diagonal slash marks to indicate broken axis
        d = 0.015  # Size of diagonal lines in axes coords
        kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False, linewidth=1.5)

        # Upper axis: draw at bottom
        ax_upper.plot((-d, +d), (-d, +d), **kwargs)  # left diagonal
        ax_upper.plot((-d, +d), (-d - 0.02, +d - 0.02), **kwargs)  # second left diagonal

        # Lower axis: draw at top
        kwargs = dict(transform=ax_lower.transAxes, color='k', clip_on=False, linewidth=1.5)
        ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # left diagonal
        ax_lower.plot((-d, +d), (1 - d - 0.02, 1 + d - 0.02), **kwargs)  # second left diagonal

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

    # Print summary
    print("\n" + "=" * 80)
    print("GEOMEAN NRMSE SUMMARY")
    print("=" * 80)

    exp_display = {'topology_agnostic': 'Cross-Topology', 'intra_topology': 'Intra-Topology'}
    pdk_display = {'TSMC': 'Commercial', 'ASAP7': 'ASAP7'}

    print(f"{'PDK':<12} {'Scenario':<18} {'DataType':<12} {'Mode':<15} {'AADAM':>8} {'MLP_MAML':>10} {'GCN_Base':>10} {'GCN_MAML':>10}")
    print("-" * 100)

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

    print("=" * 80)
# ===== END cell 1 =====


# ============================================================================
# Wrapper added for rebuttal — adds MAE / MAPE / Pessimism columns.
# Uses the same {pred, act} arrays the cell-1 calculate_metrics consumes.
# ============================================================================
def add_engineering_metrics_to_df(df, gcn_dir, mlp_dirs):
    """For each row in df, locate the pred/act npy and append MAE_ps_scaled,
    MAPE_pct, and signed-error pessimism columns.

    gcn_dir: top-level GCN results dir.
    mlp_dirs: tuple (mlp_maml_dir, aadam_dir) — where MLP npy live.
    Mirrors the path conventions used by the notebook's `load_*_results`.
    """
    if df is None or len(df) == 0:
        return df
    out_rows = []
    for _, row in df.iterrows():
        filename = row.get('filename')
        if not isinstance(filename, str):
            out_rows.append(None); continue
        # GCN files are top-level in gcn_dir; MLP files are in mlp_dirs by model_type
        cand_paths = [
            os.path.join(gcn_dir, filename),
            os.path.join(mlp_dirs[0], filename) if mlp_dirs and mlp_dirs[0] else None,
            os.path.join(mlp_dirs[1], filename) if mlp_dirs and len(mlp_dirs) > 1 and mlp_dirs[1] else None,
        ]
        pred_path = next((p for p in cand_paths if p and os.path.exists(p)), None)
        if pred_path is None:
            out_rows.append(None); continue
        act_path = pred_path.replace('_pred.npy', '_act.npy')
        if not os.path.exists(act_path):
            out_rows.append(None); continue
        pred = np.load(pred_path).astype(np.float64).reshape(-1)
        act  = np.load(act_path ).astype(np.float64).reshape(-1)
        m = np.isfinite(pred) & np.isfinite(act)
        pred, act = pred[m], act[m]
        if len(pred) == 0:
            out_rows.append(None); continue
        # Detect scale per array: ASAP7/TSMC raw values can be in different units.
        # Use a tiered heuristic so MAE_scaled is consistent across PDKs.
        max_abs = float(np.abs(act).max())
        if max_abs < 1e-7:
            scale, unit = 1e12, 'ps'   # seconds → ps
        elif max_abs < 1e-4:
            scale, unit = 1e9,  'ps'   # ns → ps (sometimes saved this way)
        else:
            scale, unit = 1.0,  'raw'  # already in some scaled unit
        err = pred - act        # positive = pessimistic / safe over-prediction
        abs_err = np.abs(err)
        mae_scaled = float(abs_err.mean() * scale)
        # MAPE per task with per-task floor (10% of task max) — prevents blow-up
        # on near-zero ground truths (fast paths at high V / low T).
        gs = 61
        ng = len(act) // gs
        if ng == 0: ng = 1; gs = len(act)
        pred_g = pred[:ng*gs].reshape(ng, gs)
        act_g  = act[:ng*gs].reshape(ng, gs)
        per_task_max = np.abs(act_g).max(axis=1, keepdims=True)
        floor = np.maximum(per_task_max * 0.10, 1e-12)
        denom = np.maximum(np.abs(act_g), floor)
        mape_groups = np.mean(np.abs(pred_g - act_g) / denom, axis=1) * 100.0
        mape_pct = float(np.mean(mape_groups))
        signed_err = act - pred   # positive = pred low (unsafe / optimistic)
        out_rows.append({
            'MAE_scaled': mae_scaled,
            'MAE_unit': unit,
            'MAPE_pct': mape_pct,
            'UnderPred_frac': float((signed_err > 0).mean()),
            'PessSafe_p50': float(np.percentile(-signed_err, 50) * scale),  # +ve = safe
            'PessSafe_p95': float(np.percentile(-signed_err, 95) * scale),
            'MaxUnderPred': float(signed_err.max() * scale),
        })
    add = pd.DataFrame(out_rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), add.reset_index(drop=True)], axis=1)
