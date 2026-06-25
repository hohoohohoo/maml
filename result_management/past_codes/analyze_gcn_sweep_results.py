#!/usr/bin/env python3
"""
Analyze GCN sweep results - compare GCN architectures and optionally with MLP baselines

Features:
- Parse GCN result files with architecture info (conv_hidden_dim, num_conv_layers, fc_hidden_dim, num_fc_layers)
- Compare different GCN architectures
- Optionally include MLP MAML/AADAM baselines for comparison
- Generate comparison plots and CSV exports
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import re

# Set matplotlib style
plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Expected cells for each experiment type (for validation)
EXPECTED_CELLS = {
    'ASAP7': {
        'intra_topology': ['NAND3x2', 'OR2x6', 'NOR2xp67', 'AND2x6'],
        'topology_agnostic': ['MAJIxp5', 'MAJx2', 'MAJx3', 'HAxp5', 'FAx1',
                              'XOR2xp5', 'XOR2x2', 'XOR2x1', 'XNOR2xp5', 'XNOR2x2', 'XNOR2x1'],
    },
    'TSMC': {
        'intra_topology': ['NR3D1BWP30P140', 'OR4D0BWP30P140', 'ND3D0BWP30P140',
                           'AN4D0BWP30P140', 'XOR3D1BWP30P140', 'XNR3D1BWP30P140'],
        'topology_agnostic': ['HA1D0BWP30P140', 'FA1D0BWP30P140', 'IOA21D0BWP30P140', 'IOA21D1BWP30P140',
                              'OA21D0BWP30P140', 'OA21D1BWP30P140', 'OA211D0BWP30P140', 'OA211D1BWP30P140',
                              'IAO21D0BWP30P140', 'IAO21D1BWP30P140', 'AO21D0BWP30P140', 'AO21D1BWP30P140',
                              'AO211D0BWP30P140', 'AO211D1BWP30P140', 'SDFSNQD0BWP30P140', 'DFCNQD1BWP30P140'],
    }
}


def validate_cell_for_experiment(filename, cell, experiment, prefix):
    """
    Validate that a cell belongs to the correct experiment type.

    Args:
        filename: Original filename for error reporting
        cell: Cell name from filename
        experiment: Experiment type (intra_topology or topology_agnostic)
        prefix: Tech prefix (ASAP7 or TSMC)

    Returns:
        tuple: (is_valid, error_message)
    """
    if prefix not in EXPECTED_CELLS:
        return True, None  # Unknown prefix, skip validation

    if experiment not in EXPECTED_CELLS[prefix]:
        return True, None  # Unknown experiment, skip validation

    expected_cells = EXPECTED_CELLS[prefix][experiment]

    if cell not in expected_cells:
        # Check if cell belongs to the other experiment type
        other_exp = 'topology_agnostic' if experiment == 'intra_topology' else 'intra_topology'
        if other_exp in EXPECTED_CELLS[prefix] and cell in EXPECTED_CELLS[prefix][other_exp]:
            return False, f"Cell '{cell}' belongs to {other_exp}, not {experiment}"
        else:
            return False, f"Cell '{cell}' not in expected cells for {experiment}"

    return True, None


def parse_gcn_filename(filename):
    """
    Parse GCN result filename to extract metadata

    Patterns:
    1. GCN_unified_{process}_{corner}_{data_type}_{graph_mode}_{mode}_{model_type}_..._conv{X}x{Y}_fc{A}x{B}_{pred|act}.npy
    2. TSMC_GCN_intra_topology_{cell}_{data_type}_{graph_mode}_{mode}_{model_type}_..._conv{X}x{Y}_fc{A}x{B}_{pred|act}.npy
    3. GCN_{process}_{corner}_{data_type}_{graph_mode}_{mode}_{model_type}_..._conv{X}x{Y}_fc{A}x{B}_{pred|act}.npy

    Where data_type is 'cell' or 'transition'

    Returns:
        dict with metadata or None if not parseable
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

    # Determine file type (pred or act)
    if '_pred.npy' in basename:
        file_type = 'pred'
    elif '_act.npy' in basename:
        file_type = 'act'
    else:
        return None

    # Check for filtered flag
    is_filtered = '_filtered_' in basename or basename.endswith('_filtered_pred.npy') or basename.endswith('_filtered_act.npy')

    # Extract pooling mode
    pool_match = re.search(r'_pool(output|max|add|mean)', basename)
    if pool_match:
        pooling = pool_match.group(1)
    else:
        pooling = 'mean'  # default

    # Parse different filename patterns
    result = {
        'conv_hidden_dim': conv_hidden_dim,
        'num_conv_layers': num_conv_layers,
        'fc_hidden_dim': fc_hidden_dim,
        'num_fc_layers': num_fc_layers,
        'file_type': file_type,
        'is_filtered': is_filtered,
        'pooling': pooling,
        'filename': basename,
        'arch_string': f'conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}'
    }

    # Pattern 0: ASAP7_GCN_{experiment}_{cell}_{data_type}_{graph_mode}_{mode}_{model_type}_...
    asap7_pattern = r'ASAP7_GCN_(intra_topology|topology_agnostic)_(\w+)_(cell|transition)_(full_graph|stage_aware)_(interpolation|extrapolation)_(baseline|maml)'
    match = re.match(asap7_pattern, basename)
    if match:
        result['prefix'] = 'ASAP7'
        result['experiment'] = match.group(1)
        result['cell'] = match.group(2)
        result['data_type'] = match.group(3)
        result['graph_mode'] = match.group(4)
        result['mode'] = match.group(5)
        result['model_type'] = 'GCN_' + match.group(6).upper()

        # Extract MAML params if present
        if 'maml' in basename.lower():
            maml_params = re.search(r'innerdiv(\d+)_meta(\d+)_iter(\d+)_inner(\d+)', basename)
            if maml_params:
                result['innerdiv'] = int(maml_params.group(1))
                result['meta'] = int(maml_params.group(2))
                result['iterations'] = int(maml_params.group(3))
                result['inner_steps'] = int(maml_params.group(4))
        else:
            # Baseline
            iter_match = re.search(r'iter(\d+)', basename)
            if iter_match:
                result['iterations'] = int(iter_match.group(1))

        return result

    # Pattern 1: TSMC_GCN_{experiment}_{cell}_{data_type}_{graph_mode}_{mode}_...
    tsmc_pattern = r'TSMC_GCN_(intra_topology|topology_agnostic)_(\w+)_(cell|transition)_(\w+)_(interpolation|extrapolation)_(baseline|maml)'
    match = re.match(tsmc_pattern, basename)
    if match:
        result['prefix'] = 'TSMC_GCN'
        result['experiment'] = match.group(1)
        result['cell'] = match.group(2)
        result['data_type'] = match.group(3)
        result['graph_mode'] = match.group(4)
        result['mode'] = match.group(5)
        result['model_type'] = 'GCN_' + match.group(6).upper()

        # Extract MAML params
        if 'maml' in basename.lower():
            maml_params = re.search(r'innerdiv(\d+)_meta(\d+)_iter(\d+)_inner(\d+)', basename)
            if maml_params:
                result['innerdiv'] = int(maml_params.group(1))
                result['meta'] = int(maml_params.group(2))
                result['iterations'] = int(maml_params.group(3))
                result['inner_steps'] = int(maml_params.group(4))
        else:
            iter_match = re.search(r'iter(\d+)', basename)
            if iter_match:
                result['iterations'] = int(iter_match.group(1))

        return result

    return None


def parse_mlp_filename(filename):
    """
    Parse MLP result filename to extract metadata (for comparison)

    Patterns:
    - ASAP7_intra_topology_{cell}_cell_{mode}_MAML_innerdiv{X}_meta{Y}_layer{Z}_{iter}_{pred|act}.npy
    - ASAP7_intra_topology_{cell}_cell_{mode}_aadam_{iter}_{pred|act}.npy
    """
    basename = os.path.basename(filename)

    # MAML pattern
    maml_pattern = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_MAML_innerdiv(\d+)_meta(\d+)_layer(\d+)_(\d+)_(pred|act)\.npy'
    match = re.match(maml_pattern, basename)

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

    # AADAM pattern
    aadam_pattern = r'(\w+)_([\w_]+)_(\w+)_(cell|transition)_(extrapolation|interpolation)_(aadam|mlp)_(\d+)_(pred|act)\.npy'
    match = re.match(aadam_pattern, basename)

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

    return None


def filter_by_extrapolation_region(predictions, actuals, ex_region='all',
                                   group_size=61, left_bound=5, right_bound=56):
    """
    Filter predictions and actuals by extrapolation region.

    For extrapolation mode with support indices [5, 30, 55]:
    - left_bound = 5 (min support index)
    - right_bound = 56 (max support index + 1)
    - left_ex: indices 0 to left_bound-1 (0-4, 5 points per task)
    - inter: indices left_bound to right_bound-1 (5-55, 51 points per task)
    - right_ex: indices right_bound to 60 (56-60, 5 points per task)

    Args:
        predictions: flattened array of predictions
        actuals: flattened array of actual values
        ex_region: 'all', 'left_ex', 'right_ex', 'ex_only' (left + right combined), 'inter'
        group_size: number of points per task (default 61)
        left_bound: left boundary index (default 5)
        right_bound: right boundary index (default 56)

    Returns:
        filtered_predictions, filtered_actuals, new_group_size
    """
    if ex_region == 'all':
        return predictions, actuals, group_size

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    n_groups = len(predictions) // group_size
    if n_groups == 0:
        return predictions, actuals, group_size

    # Trim to exact multiples
    predictions = predictions[:n_groups * group_size]
    actuals = actuals[:n_groups * group_size]

    # Reshape to (n_groups, group_size)
    pred_grouped = predictions.reshape(n_groups, group_size)
    act_grouped = actuals.reshape(n_groups, group_size)

    if ex_region == 'left_ex':
        # Left extrapolation: indices 0 to left_bound-1
        pred_filtered = pred_grouped[:, :left_bound]
        act_filtered = act_grouped[:, :left_bound]
        new_group_size = left_bound
    elif ex_region == 'right_ex':
        # Right extrapolation: indices right_bound to end
        pred_filtered = pred_grouped[:, right_bound:]
        act_filtered = act_grouped[:, right_bound:]
        new_group_size = group_size - right_bound
    elif ex_region == 'ex_only':
        # Both extrapolation regions combined (left + right)
        pred_left = pred_grouped[:, :left_bound]
        pred_right = pred_grouped[:, right_bound:]
        act_left = act_grouped[:, :left_bound]
        act_right = act_grouped[:, right_bound:]
        pred_filtered = np.concatenate([pred_left, pred_right], axis=1)
        act_filtered = np.concatenate([act_left, act_right], axis=1)
        new_group_size = left_bound + (group_size - right_bound)
    elif ex_region == 'inter':
        # Interpolation region only: indices left_bound to right_bound-1
        pred_filtered = pred_grouped[:, left_bound:right_bound]
        act_filtered = act_grouped[:, left_bound:right_bound]
        new_group_size = right_bound - left_bound
    else:
        return predictions, actuals, group_size

    return pred_filtered.flatten(), act_filtered.flatten(), new_group_size


def calculate_metrics(predictions, actuals, group_size=61):
    """
    Calculate NRMSE, RMSE metrics with 61-group averaging.
    Vectorized implementation for speed.
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

    # Vectorized metrics calculation (no Python loops)
    # MSE and RMSE per group
    mse_groups = np.mean((pred_grouped - act_grouped) ** 2, axis=1)
    rmse_groups = np.sqrt(mse_groups)

    # NRMSE per group (range normalization)
    y_ranges = np.max(act_grouped, axis=1) - np.min(act_grouped, axis=1)
    y_ranges = np.where(y_ranges > 0, y_ranges, 1.0)  # Avoid division by zero
    nrmse_groups = (rmse_groups / y_ranges) * 100

    # Average across groups
    return {
        'NRMSE': float(np.mean(nrmse_groups)),
        'RMSE': float(np.mean(rmse_groups)),
        'num_samples': len(predictions),
        'num_groups': n_groups
    }


def load_gcn_results(data_dir, subdirs=None, ex_region='all'):
    """Load all GCN results from directory and optional subdirectories

    Args:
        data_dir: Directory containing GCN result files
        subdirs: List of subdirectories to search (default: auto-detect)
        ex_region: extrapolation region filter ('all', 'left_ex', 'right_ex', 'ex_only', 'inter')
    """
    results = []

    # Search in main directory
    search_dirs = [data_dir]

    # Add subdirectories if specified
    if subdirs:
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                search_dirs.append(subdir_path)
    else:
        # Auto-detect subdirectories
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                search_dirs.append(item_path)

    # Collect all pred files first for progress tracking
    all_pred_files = []
    for search_dir in search_dirs:
        pred_files = glob.glob(os.path.join(search_dir, '*_pred.npy'))
        all_pred_files.extend([(search_dir, f) for f in pred_files])

    total_files = len(all_pred_files)
    print(f"Found {total_files} prediction files to process...")

    # Collect validation errors
    validation_errors = []

    for i, (search_dir, pred_file) in enumerate(all_pred_files):
        if (i + 1) % 100 == 0 or i == total_files - 1:
            print(f"  Processing: {i + 1}/{total_files} ({100*(i+1)//total_files}%)", end='\r')

        act_file = pred_file.replace('_pred.npy', '_act.npy')
        if not os.path.exists(act_file):
            continue

        metadata = parse_gcn_filename(pred_file)
        if metadata is None:
            continue

        # Validate cell belongs to correct experiment type
        cell = metadata.get('cell')
        experiment = metadata.get('experiment')
        prefix = metadata.get('prefix')
        if cell and experiment and prefix:
            is_valid, error_msg = validate_cell_for_experiment(
                os.path.basename(pred_file), cell, experiment, prefix
            )
            if not is_valid:
                validation_errors.append({
                    'filename': os.path.basename(pred_file),
                    'cell': cell,
                    'experiment': experiment,
                    'prefix': prefix,
                    'error': error_msg
                })
                continue  # Skip this file

        # Skip empty files
        if os.path.getsize(pred_file) == 0:
            continue

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)

            if len(predictions) != len(actuals) or len(predictions) == 0:
                continue

            # Apply extrapolation region filter
            predictions, actuals, group_size = filter_by_extrapolation_region(
                predictions, actuals, ex_region=ex_region
            )

            metrics = calculate_metrics(predictions, actuals, group_size=group_size)
            if metrics is None:
                continue

            result = {**metadata, **metrics}
            result['source_dir'] = os.path.basename(search_dir)
            result['ex_region'] = ex_region
            results.append(result)

        except Exception as e:
            print(f"Error loading {pred_file}: {e}")
            continue

    print()  # New line after progress

    # Check for validation errors and exit if found
    if validation_errors:
        print("\n" + "=" * 80)
        print("ERROR: Cell-Experiment mismatch detected!")
        print("The following files have cells that don't belong to their experiment type:")
        print("=" * 80)
        for err in validation_errors:
            print(f"  File: {err['filename']}")
            print(f"    Cell: {err['cell']}, Experiment: {err['experiment']}, Prefix: {err['prefix']}")
            print(f"    Error: {err['error']}")
            print()
        print(f"Total mismatched files: {len(validation_errors)}")
        print("Please fix or remove these files before running analysis.")
        print("=" * 80)
        sys.exit(1)

    if not results:
        if ex_region != 'all':
            print(f"  Extrapolation region filter: {ex_region}")
        return None

    # Check for duplicate cell entries (same cell should have only one file per config)
    df = pd.DataFrame(results)
    key_cols = ['prefix', 'experiment', 'cell', 'data_type', 'mode', 'graph_mode', 'arch_string', 'iterations', 'pooling', 'model_type']
    existing_cols = [c for c in key_cols if c in df.columns]

    if existing_cols:
        duplicates = df[df.duplicated(subset=existing_cols, keep=False)]
        if len(duplicates) > 0:
            print("\n" + "=" * 80)
            print("ERROR: Duplicate cell entries detected!")
            print("Each cell should have exactly ONE file per (experiment, mode, arch, iteration, pooling) combination.")
            print("=" * 80)

            # Group by key columns and show duplicates
            dup_groups = duplicates.groupby(existing_cols)
            for keys, group in dup_groups:
                if len(group) > 1:
                    print(f"\n  Duplicate found ({len(group)} files):")
                    key_dict = dict(zip(existing_cols, keys if isinstance(keys, tuple) else [keys]))
                    for k, v in key_dict.items():
                        print(f"    {k}: {v}")
                    print(f"  Files:")
                    for _, row in group.iterrows():
                        print(f"    - {row['filename']}")

            print(f"\nTotal duplicate entries: {len(duplicates)}")
            print("Please remove duplicate files before running analysis.")
            print("=" * 80)
            sys.exit(1)

    if ex_region != 'all':
        print(f"  Extrapolation region filter: {ex_region}")

    return df


def load_mlp_results(data_dir, ex_region='all'):
    """Load MLP results for comparison

    Args:
        data_dir: Directory containing MLP result files
        ex_region: extrapolation region filter ('all', 'left_ex', 'right_ex', 'ex_only', 'inter')
    """
    results = []
    pred_files = glob.glob(os.path.join(data_dir, '*_pred.npy'))

    total_files = len(pred_files)
    if total_files > 0:
        print(f"Found {total_files} MLP prediction files...")

    # Collect validation errors
    validation_errors = []

    for i, pred_file in enumerate(pred_files):
        if total_files > 50 and ((i + 1) % 50 == 0 or i == total_files - 1):
            print(f"  Processing MLP: {i + 1}/{total_files} ({100*(i+1)//total_files}%)", end='\r')

        act_file = pred_file.replace('_pred.npy', '_act.npy')
        if not os.path.exists(act_file):
            continue

        metadata = parse_mlp_filename(pred_file)
        if metadata is None:
            continue

        # Validate cell belongs to correct experiment type
        cell = metadata.get('cell')
        topology = metadata.get('topology')  # e.g., 'intra_topology' or 'topology_agnostic'
        prefix = metadata.get('prefix')
        if cell and topology and prefix:
            is_valid, error_msg = validate_cell_for_experiment(
                os.path.basename(pred_file), cell, topology, prefix
            )
            if not is_valid:
                validation_errors.append({
                    'filename': os.path.basename(pred_file),
                    'cell': cell,
                    'experiment': topology,
                    'prefix': prefix,
                    'error': error_msg
                })
                continue  # Skip this file

        try:
            predictions = np.load(pred_file)
            actuals = np.load(act_file)

            if len(predictions) != len(actuals):
                continue

            # Apply extrapolation region filter
            predictions, actuals, group_size = filter_by_extrapolation_region(
                predictions, actuals, ex_region=ex_region
            )

            metrics = calculate_metrics(predictions, actuals, group_size=group_size)
            if metrics is None:
                continue

            result = {**metadata, **metrics}
            result['ex_region'] = ex_region
            results.append(result)

        except Exception:
            continue

    if total_files > 50:
        print()  # New line after progress

    # Check for validation errors and exit if found
    if validation_errors:
        print("\n" + "=" * 80)
        print("ERROR: Cell-Experiment mismatch detected in MLP files!")
        print("The following files have cells that don't belong to their experiment type:")
        print("=" * 80)
        for err in validation_errors:
            print(f"  File: {err['filename']}")
            print(f"    Cell: {err['cell']}, Experiment: {err['experiment']}, Prefix: {err['prefix']}")
            print(f"    Error: {err['error']}")
            print()
        print(f"Total mismatched files: {len(validation_errors)}")
        print("Please fix or remove these files before running analysis.")
        print("=" * 80)
        sys.exit(1)

    if not results:
        if ex_region != 'all':
            print(f"  MLP extrapolation region filter: {ex_region}")
        return None

    # Check for duplicate cell entries (same cell should have only one file per config)
    df = pd.DataFrame(results)
    key_cols = ['prefix', 'topology', 'cell', 'data_type', 'mode', 'model_type', 'iterations']
    # Add innerdiv, meta, layer_length for MAML
    if 'innerdiv' in df.columns:
        key_cols.extend(['innerdiv', 'meta', 'layer_length'])
    existing_cols = [c for c in key_cols if c in df.columns]

    if existing_cols:
        duplicates = df[df.duplicated(subset=existing_cols, keep=False)]
        if len(duplicates) > 0:
            print("\n" + "=" * 80)
            print("ERROR: Duplicate cell entries detected in MLP files!")
            print("Each cell should have exactly ONE file per (topology, mode, model_type, iteration) combination.")
            print("=" * 80)

            # Group by key columns and show duplicates
            dup_groups = duplicates.groupby(existing_cols)
            for keys, group in dup_groups:
                if len(group) > 1:
                    print(f"\n  Duplicate found ({len(group)} files):")
                    key_dict = dict(zip(existing_cols, keys if isinstance(keys, tuple) else [keys]))
                    for k, v in key_dict.items():
                        print(f"    {k}: {v}")
                    print(f"  Files:")
                    for _, row in group.iterrows():
                        print(f"    - {row['filename']}")

            print(f"\nTotal duplicate entries: {len(duplicates)}")
            print("Please remove duplicate files before running analysis.")
            print("=" * 80)
            sys.exit(1)

    if ex_region != 'all':
        print(f"  MLP extrapolation region filter: {ex_region}")

    return df


def plot_architecture_comparison(df, output_dir, metric='NRMSE',
                                 filter_mode=None, filter_graph_mode=None,
                                 filter_experiment=None,
                                 include_mlp=False, mlp_df=None,
                                 scale_rmse=True, aadam_iter=None,
                                 maml_iter=None, gcn_iter=None):
    """
    Plot comparison of GCN architectures

    Args:
        df: GCN results DataFrame
        output_dir: Output directory for plots
        metric: Metric to plot (NRMSE, RMSE)
        filter_mode: Filter by mode (interpolation/extrapolation)
        filter_graph_mode: Filter by graph_mode (full_graph/stage_aware)
        filter_experiment: Filter by experiment (intra_topology/topology_agnostic)
        include_mlp: Include MLP baselines
        mlp_df: MLP results DataFrame
        scale_rmse: Scale RMSE by 1000 (for TSMC results)
    """
    filtered_df = df.copy()

    if filter_mode:
        filtered_df = filtered_df[filtered_df['mode'] == filter_mode]
    if filter_graph_mode:
        filtered_df = filtered_df[filtered_df['graph_mode'] == filter_graph_mode]
    if filter_experiment and 'experiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['experiment'] == filter_experiment]
    if gcn_iter is not None and 'iterations' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['iterations'] == gcn_iter]

    if len(filtered_df) == 0:
        print(f"No data after filtering (mode={filter_mode}, graph_mode={filter_graph_mode}, experiment={filter_experiment}, gcn_iter={gcn_iter})")
        return

    # Group by architecture
    arch_groups = filtered_df.groupby('arch_string').agg({
        metric: 'mean',
        'num_samples': 'sum'
    }).reset_index()

    arch_groups = arch_groups.sort_values(metric)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x_vals = range(len(arch_groups))
    y_vals = arch_groups[metric].values

    if metric == 'RMSE' and scale_rmse:
        y_vals = y_vals * 1000
        ylabel = 'RMSE (x1000)'
    else:
        ylabel = metric

    # Bar plot
    bars = ax.bar(x_vals, y_vals, alpha=0.7, edgecolor='black')
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Value labels
    for i, y in enumerate(y_vals):
        ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_vals)
    ax.set_xticklabels(arch_groups['arch_string'], rotation=45, ha='right')

    # MLP baseline if provided - Aadam as primary reference
    if include_mlp and mlp_df is not None:
        mlp_filtered = mlp_df.copy()
        if filter_mode:
            mlp_filtered = mlp_filtered[mlp_filtered['mode'] == filter_mode]
        # Filter MLP to only cells that exist in GCN data
        if 'cell' in filtered_df.columns and 'cell' in mlp_filtered.columns:
            gcn_cells = filtered_df['cell'].dropna().unique()
            mlp_filtered = mlp_filtered[mlp_filtered['cell'].isin(gcn_cells)]

        # Filter MLP by data_type to match GCN data
        if 'data_type' in filtered_df.columns and 'data_type' in mlp_filtered.columns:
            gcn_data_type = filtered_df['data_type'].iloc[0] if len(filtered_df) > 0 else 'cell'
            mlp_filtered = mlp_filtered[mlp_filtered['data_type'] == gcn_data_type]

        if len(mlp_filtered) > 0:
            # Aadam baseline (primary reference - red dashed line)
            aadam_data = mlp_filtered[mlp_filtered['model_type'] == 'AADAM']
            if aadam_iter is not None and 'iterations' in aadam_data.columns:
                aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
            if len(aadam_data) > 0:
                aadam_val = aadam_data[metric].mean()
                if metric == 'RMSE' and scale_rmse:
                    aadam_val = aadam_val * 1000
                iter_label = f' (iter={aadam_iter})' if aadam_iter else ''
                ax.axhline(y=aadam_val, color='red', linestyle='--', linewidth=2.5,
                          label=f'MLP Aadam{iter_label}: {aadam_val:.3f}', alpha=0.8)

            # MLP MAML (secondary reference - blue dotted line)
            maml_data = mlp_filtered[mlp_filtered['model_type'] == 'MLP_MAML']
            if maml_iter is not None and 'iterations' in maml_data.columns:
                maml_data = maml_data[maml_data['iterations'] == maml_iter]
            if len(maml_data) > 0:
                maml_iter_label = f' (iter={maml_iter})' if maml_iter else ''
                maml_val = maml_data[metric].mean()
                if metric == 'RMSE' and scale_rmse:
                    maml_val = maml_val * 1000
                ax.axhline(y=maml_val, color='blue', linestyle=':', linewidth=2,
                          label=f'MLP MAML{maml_iter_label}: {maml_val:.3f}', alpha=0.7)

    # Title and labels
    title_parts = ['GCN Architecture Comparison']
    if filter_experiment:
        title_parts.append(f'Exp: {filter_experiment}')
    if filter_mode:
        title_parts.append(f'Mode: {filter_mode}')
    if filter_graph_mode:
        title_parts.append(f'Graph: {filter_graph_mode}')

    ax.set_title(' - '.join(title_parts), fontsize=14, fontweight='bold')
    ax.set_xlabel('Architecture (conv_hidden x num_layers _ fc_hidden x num_layers)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()

    # Save
    filename_parts = ['gcn_arch_comparison', metric.lower()]
    if filter_experiment:
        filename_parts.append(filter_experiment)
    if filter_mode:
        filename_parts.append(filter_mode)
    if filter_graph_mode:
        filename_parts.append(filter_graph_mode)

    plot_path = os.path.join(output_dir, '_'.join(filename_parts) + '.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_multi_metric_comparison(df, output_dir,
                                  filter_mode=None, filter_graph_mode=None,
                                  filter_experiment=None,
                                  include_mlp=False, mlp_df=None,
                                  scale_rmse=True, top_n=10, aadam_iter=None,
                                  maml_iter=None, gcn_iter=None):
    """
    Plot multiple metrics comparison (NRMSE, RMSE) in subplots
    """
    metrics = ['NRMSE', 'RMSE']

    filtered_df = df.copy()
    if filter_mode:
        filtered_df = filtered_df[filtered_df['mode'] == filter_mode]
    if filter_graph_mode:
        filtered_df = filtered_df[filtered_df['graph_mode'] == filter_graph_mode]
    if filter_experiment and 'experiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['experiment'] == filter_experiment]
    if gcn_iter is not None and 'iterations' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['iterations'] == gcn_iter]

    if len(filtered_df) == 0:
        print(f"No data after filtering")
        return

    # Get top N architectures by NRMSE
    arch_groups = filtered_df.groupby('arch_string').agg({
        'NRMSE': 'mean',
        'RMSE': 'mean',
        'num_samples': 'sum'
    }).reset_index()

    arch_groups = arch_groups.sort_values('NRMSE').head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Title
    title_parts = ['GCN Architecture Comparison']
    if filter_experiment:
        title_parts.append(f'{filter_experiment}')
    if filter_mode:
        title_parts.append(f'{filter_mode.upper()}')
    if filter_graph_mode:
        title_parts.append(f'{filter_graph_mode}')
    fig.suptitle(' - '.join(title_parts), fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Prepare GCN data
        gcn_y_vals = arch_groups[metric].values.copy()
        gcn_labels = arch_groups['arch_string'].tolist()

        if metric == 'RMSE' and scale_rmse:
            gcn_y_vals = gcn_y_vals * 1000
            ylabel = 'RMSE (x1000)'
        else:
            ylabel = metric

        # Check for MLP MAML with specific params (innerdiv=100, meta=32, layer_length=40)
        mlp_maml_val = None
        aadam_val = None

        if include_mlp and mlp_df is not None:
            mlp_filtered = mlp_df.copy()
            if filter_mode:
                mlp_filtered = mlp_filtered[mlp_filtered['mode'] == filter_mode]
            # Filter MLP to only cells that exist in GCN data
            if 'cell' in filtered_df.columns and 'cell' in mlp_filtered.columns:
                gcn_cells = filtered_df['cell'].dropna().unique()
                mlp_filtered = mlp_filtered[mlp_filtered['cell'].isin(gcn_cells)]

            # Filter MLP by data_type to match GCN data
            if 'data_type' in filtered_df.columns and 'data_type' in mlp_filtered.columns:
                gcn_data_type = filtered_df['data_type'].iloc[0] if len(filtered_df) > 0 else 'cell'
                mlp_filtered = mlp_filtered[mlp_filtered['data_type'] == gcn_data_type]

            if len(mlp_filtered) > 0:
                # Get MLP MAML with specific parameters (innerdiv=100, meta=32, layer_length=40, iterations=maml_iter)
                maml_data = mlp_filtered[mlp_filtered['model_type'] == 'MLP_MAML']
                required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                maml_iter_val = maml_iter if maml_iter is not None else 300000
                if all(col in maml_data.columns for col in required_cols):
                    maml_specific = maml_data[
                        (maml_data['innerdiv'] == 100) &
                        (maml_data['meta'] == 32) &
                        (maml_data['layer_length'] == 40) &
                        (maml_data['iterations'] == maml_iter_val)
                    ]
                    if len(maml_specific) > 0:
                        mlp_maml_val = maml_specific[metric].mean()
                        if metric == 'RMSE' and scale_rmse:
                            mlp_maml_val = mlp_maml_val * 1000

                # Get Aadam baseline (filter by iteration if specified)
                aadam_data = mlp_filtered[mlp_filtered['model_type'] == 'AADAM']
                if aadam_iter is not None and 'iterations' in aadam_data.columns:
                    aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
                if len(aadam_data) > 0:
                    aadam_val = aadam_data[metric].mean()
                    if metric == 'RMSE' and scale_rmse:
                        aadam_val = aadam_val * 1000

        # Prepend MLP MAML to the left if available
        maml_iter_label = f'i{maml_iter_val//1000}k' if maml_iter_val else 'i300k'
        if mlp_maml_val is not None:
            all_labels = [f'MLP MAML\n(id100_m32_l40_{maml_iter_label})'] + gcn_labels
            all_y_vals = np.concatenate([[mlp_maml_val], gcn_y_vals])
            bar_colors = ['#1f77b4'] + [plt.cm.viridis(i / len(gcn_labels)) for i in range(len(gcn_labels))]
        else:
            all_labels = gcn_labels
            all_y_vals = gcn_y_vals
            bar_colors = [plt.cm.viridis(i / len(gcn_labels)) for i in range(len(gcn_labels))]

        x_vals = range(len(all_labels))

        # Bar plot
        bars = ax.bar(x_vals, all_y_vals, alpha=0.7, edgecolor='black')
        for bar, color in zip(bars, bar_colors):
            bar.set_color(color)

        # Value labels
        for i, y in enumerate(all_y_vals):
            ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x_vals)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)

        # Aadam baseline as dashed line
        if aadam_val is not None:
            iter_label = f' (iter={aadam_iter})' if aadam_iter else ''
            ax.axhline(y=aadam_val, color='red', linestyle='--', linewidth=2.5,
                      label=f'MLP Aadam{iter_label}: {aadam_val:.3f}', alpha=0.8)
            # Add text annotation on the right side
            ax.text(len(all_labels) - 0.5, aadam_val, f'Aadam: {aadam_val:.3f}',
                   fontsize=9, color='red', va='bottom', ha='right', fontweight='bold')

        # Calculate and show improvement vs Aadam for best GCN
        if aadam_val is not None and len(gcn_y_vals) > 0:
            best_gcn_val = gcn_y_vals.min()
            if aadam_val > 0:
                improvement = ((aadam_val - best_gcn_val) / aadam_val) * 100
                if improvement > 0:
                    ax.text(0.02, 0.98, f'Best GCN: {improvement:.1f}% better than Aadam',
                           transform=ax.transAxes, fontsize=9, color='green',
                           va='top', ha='left', fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        ax.set_xlabel('Architecture', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{ylabel} Comparison', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    # Save - include data_type in filename if single type
    filename_parts = ['gcn_multi_metric']
    if 'data_type' in filtered_df.columns:
        data_types = filtered_df['data_type'].unique()
        if len(data_types) == 1 and data_types[0] != 'cell':
            filename_parts.append(data_types[0])
    if filter_experiment:
        filename_parts.append(filter_experiment)
    if filter_mode:
        filename_parts.append(filter_mode)
    if filter_graph_mode:
        filename_parts.append(filter_graph_mode)

    plot_path = os.path.join(output_dir, '_'.join(filename_parts) + '.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_selected_architecture(df, output_dir, arch_string,
                               include_mlp=False, mlp_df=None,
                               scale_rmse=True, aadam_iter=None):
    """
    Plot results for a specific GCN architecture across different conditions
    """
    arch_df = df[df['arch_string'] == arch_string]

    if len(arch_df) == 0:
        print(f"No data found for architecture: {arch_string}")
        return

    metrics = ['NRMSE', 'RMSE']

    # Group by mode and graph_mode
    groups = arch_df.groupby(['mode', 'graph_mode']).agg({
        'NRMSE': 'mean',
        'RMSE': 'mean',
        'num_samples': 'sum'
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'GCN Architecture: {arch_string}', fontsize=16, fontweight='bold')

    # Create labels
    groups['label'] = groups['mode'] + '\n' + groups['graph_mode']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        x_vals = range(len(groups))
        y_vals = groups[metric].values

        if metric == 'RMSE' and scale_rmse:
            y_vals = y_vals * 1000
            ylabel = 'RMSE (x1000)'
        else:
            ylabel = metric

        bars = ax.bar(x_vals, y_vals, alpha=0.7, edgecolor='black')
        colors = plt.cm.Set2(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        for i, y in enumerate(y_vals):
            ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x_vals)
        ax.set_xticklabels(groups['label'], fontsize=9)

        # MLP baselines - Aadam as primary reference
        if include_mlp and mlp_df is not None and len(mlp_df) > 0:
            # Aadam baseline (primary reference - red dashed line)
            aadam_data = mlp_df[mlp_df['model_type'] == 'AADAM']
            if aadam_iter is not None and 'iterations' in aadam_data.columns:
                aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
            if len(aadam_data) > 0:
                aadam_val = aadam_data[metric].mean()
                if metric == 'RMSE' and scale_rmse:
                    aadam_val = aadam_val * 1000
                iter_label = f' (iter={aadam_iter})' if aadam_iter else ''
                ax.axhline(y=aadam_val, color='red', linestyle='--', linewidth=2.5,
                          label=f'MLP Aadam{iter_label}: {aadam_val:.3f}', alpha=0.8)

            # MLP MAML (secondary reference - blue dashed line)
            maml_data = mlp_df[mlp_df['model_type'] == 'MLP_MAML']
            if len(maml_data) > 0:
                maml_val = maml_data[metric].mean()
                if metric == 'RMSE' and scale_rmse:
                    maml_val = maml_val * 1000
                ax.axhline(y=maml_val, color='blue', linestyle=':', linewidth=2,
                          label=f'MLP MAML: {maml_val:.3f}', alpha=0.7)

        ax.set_xlabel('Mode / Graph Mode', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(f'{ylabel}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    safe_arch = arch_string.replace('x', '_')
    plot_path = os.path.join(output_dir, f'gcn_selected_{safe_arch}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_per_cell_comparison(df, output_dir, include_mlp=False, mlp_df=None,
                             scale_rmse=True, top_n=10, aadam_iter=None,
                             maml_iter=None, gcn_iter=None):
    """
    Generate separate plots for each cell type.

    Args:
        df: GCN results DataFrame (must have 'cell' column for TSMC data)
        output_dir: Output directory
        include_mlp: Include MLP baselines
        mlp_df: MLP results DataFrame
        scale_rmse: Scale RMSE by 1000
        top_n: Number of top architectures to show per cell
        maml_iter: Specific iteration for MAML MLP baseline
        gcn_iter: Specific iteration for GCN results
    """
    if 'cell' not in df.columns:
        print("No 'cell' column found - skipping per-cell plots")
        return

    cells = df['cell'].dropna().unique()
    if len(cells) == 0:
        print("No cell data found - skipping per-cell plots")
        return

    print(f"\nGenerating per-cell plots for {len(cells)} cells...")

    # Create per_cell subdirectory
    per_cell_dir = os.path.join(output_dir, 'per_cell')
    os.makedirs(per_cell_dir, exist_ok=True)

    metrics = ['NRMSE', 'RMSE']

    for cell in cells:
        cell_df = df[df['cell'] == cell]
        if gcn_iter is not None and 'iterations' in cell_df.columns:
            cell_df = cell_df[cell_df['iterations'] == gcn_iter]

        if len(cell_df) == 0:
            continue

        # Get unique modes and graph_modes for this cell
        modes = cell_df['mode'].dropna().unique()
        graph_modes = cell_df['graph_mode'].dropna().unique()

        for mode in modes:
            for graph_mode in graph_modes:
                filtered_df = cell_df[(cell_df['mode'] == mode) & (cell_df['graph_mode'] == graph_mode)]

                if len(filtered_df) == 0:
                    continue

                # Group by architecture
                arch_groups = filtered_df.groupby('arch_string').agg({
                    'NRMSE': 'mean',
                    'RMSE': 'mean',
                    'num_samples': 'sum'
                }).reset_index()

                arch_groups = arch_groups.sort_values('NRMSE').head(top_n)

                if len(arch_groups) == 0:
                    continue

                # Create figure
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                fig.suptitle(f'Cell: {cell} - {mode.upper()} - {graph_mode}',
                            fontsize=16, fontweight='bold')

                # Check for MLP MAML with specific params (innerdiv=100, meta=32, layer_length=40)
                mlp_maml_vals = {}  # metric -> value
                aadam_vals = {}  # metric -> value

                if include_mlp and mlp_df is not None:
                    mlp_filtered = mlp_df.copy()

                    # Filter by cell if available in MLP data
                    if 'cell' in mlp_filtered.columns:
                        mlp_filtered = mlp_filtered[mlp_filtered['cell'] == cell]

                    # Filter by mode
                    if 'mode' in mlp_filtered.columns:
                        mlp_filtered = mlp_filtered[mlp_filtered['mode'] == mode]

                    # Filter by data_type to match GCN data (cell vs transition)
                    if 'data_type' in mlp_filtered.columns and 'data_type' in filtered_df.columns:
                        gcn_data_type = filtered_df['data_type'].iloc[0] if len(filtered_df) > 0 else 'cell'
                        mlp_filtered = mlp_filtered[mlp_filtered['data_type'] == gcn_data_type]

                    if len(mlp_filtered) > 0:
                        # Get MLP MAML with specific parameters (innerdiv=100, meta=32, layer_length=40, iterations=maml_iter)
                        maml_data = mlp_filtered[mlp_filtered['model_type'] == 'MLP_MAML']
                        required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                        maml_iter_val = maml_iter if maml_iter is not None else 300000
                        if all(col in maml_data.columns for col in required_cols):
                            maml_specific = maml_data[
                                (maml_data['innerdiv'] == 100) &
                                (maml_data['meta'] == 32) &
                                (maml_data['layer_length'] == 40) &
                                (maml_data['iterations'] == maml_iter_val)
                            ]
                            if len(maml_specific) > 0:
                                for m in metrics:
                                    val = maml_specific[m].mean()
                                    if m == 'RMSE' and scale_rmse:
                                        val = val * 1000
                                    mlp_maml_vals[m] = val

                        # Get Aadam baseline (filter by iteration if specified)
                        aadam_data = mlp_filtered[mlp_filtered['model_type'] == 'AADAM']
                        if aadam_iter is not None and 'iterations' in aadam_data.columns:
                            aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
                        if len(aadam_data) > 0:
                            for m in metrics:
                                val = aadam_data[m].mean()
                                if m == 'RMSE' and scale_rmse:
                                    val = val * 1000
                                aadam_vals[m] = val

                for idx, metric in enumerate(metrics):
                    ax = axes[idx]

                    gcn_labels = arch_groups['arch_string'].tolist()
                    gcn_y_vals = arch_groups[metric].values.copy()

                    if metric == 'RMSE' and scale_rmse:
                        gcn_y_vals = gcn_y_vals * 1000
                        ylabel = 'RMSE (x1000)'
                    else:
                        ylabel = metric

                    # Prepend MLP MAML to the left if available
                    if metric in mlp_maml_vals:
                        all_labels = ['MLP MAML\n(id100_m32_l40_i300k)'] + gcn_labels
                        all_y_vals = np.concatenate([[mlp_maml_vals[metric]], gcn_y_vals])
                        bar_colors = ['#1f77b4'] + [plt.cm.viridis(i / len(gcn_labels)) for i in range(len(gcn_labels))]
                    else:
                        all_labels = gcn_labels
                        all_y_vals = gcn_y_vals
                        bar_colors = [plt.cm.viridis(i / len(gcn_labels)) for i in range(len(gcn_labels))]

                    x_vals = range(len(all_labels))

                    # Bar plot
                    bars = ax.bar(x_vals, all_y_vals, alpha=0.7, edgecolor='black')
                    for bar, color in zip(bars, bar_colors):
                        bar.set_color(color)

                    # Value labels
                    for i, y in enumerate(all_y_vals):
                        ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

                    ax.set_xticks(x_vals)
                    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)

                    # MLP baselines - Aadam as dashed line reference
                    aadam_val = aadam_vals.get(metric)
                    if aadam_val is not None:
                        iter_label = f' (iter={aadam_iter})' if aadam_iter else ''
                        ax.axhline(y=aadam_val, color='red', linestyle='--', linewidth=2.5,
                                  label=f'MLP Aadam{iter_label}: {aadam_val:.3f}', alpha=0.8)

                    # Calculate improvement vs Aadam (using best GCN only)
                    if aadam_val is not None and len(gcn_y_vals) > 0:
                        best_gcn_val = gcn_y_vals.min()
                        if aadam_val > 0:
                            improvement = ((aadam_val - best_gcn_val) / aadam_val) * 100
                            if improvement > 0:
                                ax.text(0.02, 0.98, f'Best GCN: {improvement:.1f}% better',
                                       transform=ax.transAxes, fontsize=9, color='green',
                                       va='top', ha='left', fontweight='bold',
                                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

                    ax.set_xlabel('Architecture', fontsize=10)
                    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
                    ax.set_title(f'{ylabel}', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right', fontsize=8)

                plt.tight_layout()

                # Save - include data_type in filename if available
                safe_cell = cell.replace('/', '_').replace('\\', '_')
                data_type_suffix = ''
                if 'data_type' in filtered_df.columns and len(filtered_df) > 0:
                    dt = filtered_df['data_type'].iloc[0]
                    if dt and dt != 'cell':  # only add suffix if not default 'cell'
                        data_type_suffix = f'_{dt}'
                plot_path = os.path.join(per_cell_dir, f'{safe_cell}_{mode}_{graph_mode}{data_type_suffix}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

        print(f"  Saved plots for cell: {cell}")

    print(f"Per-cell plots saved to: {per_cell_dir}")


def plot_arch_all_cells_comparison(df, output_dir, include_mlp=False, mlp_df=None,
                                    scale_rmse=True, aadam_iter=None,
                                    maml_iter=None, gcn_iter=None,
                                    filter_mode=None, filter_graph_mode=None,
                                    filter_experiment=None):
    """
    Generate comparison plots for each architecture showing all cells.

    For each unique architecture, creates a PNG file with 2 subplots (NRMSE, RMSE)
    showing performance across all cells.

    Args:
        df: GCN results DataFrame (must have 'cell' column)
        output_dir: Output directory
        include_mlp: Include MLP baselines
        mlp_df: MLP results DataFrame
        scale_rmse: Scale RMSE by 1000
        aadam_iter: Specific AADAM iteration to use as baseline
        maml_iter: Specific iteration for MAML MLP baseline
        gcn_iter: Specific iteration for GCN results
        filter_mode: Filter by mode (interpolation/extrapolation)
        filter_graph_mode: Filter by graph_mode
        filter_experiment: Filter by experiment type
    """
    if 'cell' not in df.columns:
        print("No 'cell' column found - skipping architecture-cell comparison plots")
        return

    # Apply filters
    filtered_df = df.copy()
    if filter_mode:
        filtered_df = filtered_df[filtered_df['mode'] == filter_mode]
    if filter_graph_mode:
        filtered_df = filtered_df[filtered_df['graph_mode'] == filter_graph_mode]
    if filter_experiment and 'experiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['experiment'] == filter_experiment]
    if gcn_iter is not None and 'iterations' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['iterations'] == gcn_iter]

    if len(filtered_df) == 0:
        print(f"No data after filtering for arch-cell comparison")
        return

    architectures = filtered_df['arch_string'].dropna().unique()
    cells = sorted(filtered_df['cell'].dropna().unique())

    if len(architectures) == 0 or len(cells) == 0:
        print("No architectures or cells found - skipping arch-cell comparison")
        return

    print(f"\nGenerating architecture-cell comparison plots for {len(architectures)} architectures, {len(cells)} cells...")

    # Create arch_cell_comparison subdirectory
    arch_cell_dir = os.path.join(output_dir, 'arch_cell_comparison')
    os.makedirs(arch_cell_dir, exist_ok=True)

    metrics = ['NRMSE', 'RMSE']

    # Prepare MLP baseline values per cell
    mlp_baselines = {}  # {cell: {metric: {'maml': val, 'aadam': val}}}
    if include_mlp and mlp_df is not None:
        mlp_filtered = mlp_df.copy()
        if filter_mode and 'mode' in mlp_filtered.columns:
            mlp_filtered = mlp_filtered[mlp_filtered['mode'] == filter_mode]

        # Filter MLP by data_type to match GCN data
        if 'data_type' in filtered_df.columns and 'data_type' in mlp_filtered.columns:
            gcn_data_type = filtered_df['data_type'].iloc[0] if len(filtered_df) > 0 else 'cell'
            mlp_filtered = mlp_filtered[mlp_filtered['data_type'] == gcn_data_type]

        if 'cell' in mlp_filtered.columns:
            for cell in cells:
                cell_mlp = mlp_filtered[mlp_filtered['cell'] == cell]
                if len(cell_mlp) == 0:
                    continue

                mlp_baselines[cell] = {}

                # MLP MAML
                maml_data = cell_mlp[cell_mlp['model_type'] == 'MLP_MAML']
                required_cols = ['innerdiv', 'meta', 'layer_length', 'iterations']
                if all(col in maml_data.columns for col in required_cols):
                    maml_iter_val = maml_iter if maml_iter is not None else 300000
                    maml_specific = maml_data[
                        (maml_data['innerdiv'] == 100) &
                        (maml_data['meta'] == 32) &
                        (maml_data['layer_length'] == 40) &
                        (maml_data['iterations'] == maml_iter_val)
                    ]
                    if len(maml_specific) > 0:
                        for m in metrics:
                            val = maml_specific[m].mean()
                            if m == 'RMSE' and scale_rmse:
                                val = val * 1000
                            if m not in mlp_baselines[cell]:
                                mlp_baselines[cell][m] = {}
                            mlp_baselines[cell][m]['maml'] = val

                # AADAM
                aadam_data = cell_mlp[cell_mlp['model_type'] == 'AADAM']
                if aadam_iter is not None and 'iterations' in aadam_data.columns:
                    aadam_data = aadam_data[aadam_data['iterations'] == aadam_iter]
                if len(aadam_data) > 0:
                    for m in metrics:
                        val = aadam_data[m].mean()
                        if m == 'RMSE' and scale_rmse:
                            val = val * 1000
                        if m not in mlp_baselines[cell]:
                            mlp_baselines[cell][m] = {}
                        mlp_baselines[cell][m]['aadam'] = val

    # Generate plot for each architecture
    for arch in architectures:
        arch_df = filtered_df[filtered_df['arch_string'] == arch]

        if len(arch_df) == 0:
            continue

        # Aggregate by cell
        cell_groups = arch_df.groupby('cell').agg({
            'NRMSE': 'mean',
            'RMSE': 'mean',
            'num_samples': 'sum'
        }).reset_index()

        # Sort by NRMSE
        cell_groups = cell_groups.sort_values('NRMSE')

        if len(cell_groups) == 0:
            continue

        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Title
        title_parts = [f'Architecture: {arch}']
        if filter_experiment:
            title_parts.append(f'Exp: {filter_experiment}')
        if filter_mode:
            title_parts.append(f'Mode: {filter_mode}')
        if filter_graph_mode:
            title_parts.append(f'Graph: {filter_graph_mode}')
        fig.suptitle(' - '.join(title_parts), fontsize=14, fontweight='bold')

        cell_labels = cell_groups['cell'].tolist()
        x_vals = range(len(cell_labels))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            y_vals = cell_groups[metric].values.copy()

            if metric == 'RMSE' and scale_rmse:
                y_vals = y_vals * 1000
                ylabel = 'RMSE (x1000)'
            else:
                ylabel = metric

            # Bar plot for GCN results
            bars = ax.bar(x_vals, y_vals, alpha=0.7, edgecolor='black', label='GCN')
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Value labels on bars
            for i, y in enumerate(y_vals):
                ax.text(i, y, f'{y:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=45)

            # MLP baselines as scatter points
            if include_mlp and mlp_baselines:
                maml_x, maml_y = [], []
                aadam_x, aadam_y = [], []

                for i, cell in enumerate(cell_labels):
                    if cell in mlp_baselines and metric in mlp_baselines[cell]:
                        if 'maml' in mlp_baselines[cell][metric]:
                            maml_x.append(i)
                            maml_y.append(mlp_baselines[cell][metric]['maml'])
                        if 'aadam' in mlp_baselines[cell][metric]:
                            aadam_x.append(i)
                            aadam_y.append(mlp_baselines[cell][metric]['aadam'])

                if maml_y:
                    ax.scatter(maml_x, maml_y, color='blue', marker='s', s=80,
                              label=f'MLP MAML', zorder=5, edgecolors='black')
                if aadam_y:
                    iter_label = f' (iter={aadam_iter})' if aadam_iter else ''
                    ax.scatter(aadam_x, aadam_y, color='red', marker='^', s=100,
                              label=f'MLP Aadam{iter_label}', zorder=5, edgecolors='black')

            ax.set_xticks(x_vals)
            ax.set_xticklabels(cell_labels, rotation=45, ha='right', fontsize=9)
            ax.set_xlabel('Cell', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(f'{ylabel} by Cell', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()

        # Save - include data_type in filename if single type
        safe_arch = arch.replace('x', '_')
        filename_parts = [f'arch_{safe_arch}_all_cells']
        if 'data_type' in filtered_df.columns:
            data_types = filtered_df['data_type'].unique()
            if len(data_types) == 1 and data_types[0] != 'cell':
                filename_parts.append(data_types[0])
        if filter_experiment:
            filename_parts.append(filter_experiment)
        if filter_mode:
            filename_parts.append(filter_mode)
        if filter_graph_mode:
            filename_parts.append(filter_graph_mode)

        plot_path = os.path.join(arch_cell_dir, '_'.join(filename_parts) + '.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Architecture-cell comparison plots saved to: {arch_cell_dir}")


def export_results_csv(df, output_file):
    """Export GCN results to CSV"""
    cols_order = [
        'arch_string', 'conv_hidden_dim', 'num_conv_layers', 'fc_hidden_dim', 'num_fc_layers',
        'experiment', 'cell', 'data_type', 'mode', 'graph_mode', 'pooling', 'model_type', 'is_filtered',
        'NRMSE', 'RMSE',
        'num_samples', 'num_groups', 'source_dir', 'filename'
    ]

    cols_order = [c for c in cols_order if c in df.columns]
    df_export = df[cols_order].copy()

    # Round metrics
    metric_cols = ['NRMSE', 'RMSE']
    for col in metric_cols:
        if col in df_export.columns:
            df_export[col] = df_export[col].round(4)

    df_export = df_export.sort_values(['mode', 'graph_mode', 'NRMSE'])
    df_export.to_csv(output_file, index=False)
    print(f"Exported: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze GCN sweep results and compare with MLP baselines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis (MLP Aadam baseline included by default)
  python analyze_gcn_sweep_results.py

  # With specific architecture highlight
  python analyze_gcn_sweep_results.py --arch conv64x2_fc128x2

  # Disable MLP baseline comparison
  python analyze_gcn_sweep_results.py --no_mlp

  # Filter by mode and graph mode
  python analyze_gcn_sweep_results.py --mode interpolation --graph_mode full_graph

  # Filter by experiment type
  python analyze_gcn_sweep_results.py --experiment intra_topology
  python analyze_gcn_sweep_results.py --experiment topology_agnostic

  # Generate per-cell plots (for TSMC data)
  python analyze_gcn_sweep_results.py --per_cell --scale_rmse

  # Specify subdirectories to search
  python analyze_gcn_sweep_results.py --subdirs 10000samples 5000samples

  # Use specific AADAM iteration as baseline
  python analyze_gcn_sweep_results.py --aadam_iter 300000

  # Filter by pooling method (output-node-only)
  python analyze_gcn_sweep_results.py --pooling output

  # Combined filters
  python analyze_gcn_sweep_results.py --experiment intra_topology --mode extrapolation --pooling output

  # Generate architecture-cell comparison plots (each arch shows all cells)
  python analyze_gcn_sweep_results.py --arch_cell --mode interpolation --graph_mode stage_aware

  # Filter by extrapolation region (for extrapolation mode data)
  python analyze_gcn_sweep_results.py --ex_region left_ex --mode extrapolation    # left extrapolation only
  python analyze_gcn_sweep_results.py --ex_region right_ex --mode extrapolation   # right extrapolation only
  python analyze_gcn_sweep_results.py --ex_region ex_only --mode extrapolation    # both left+right combined
  python analyze_gcn_sweep_results.py --ex_region inter --mode extrapolation      # interpolation region only
        """
    )

    parser.add_argument('--gcn_dir', type=str,
                       default='../pretraining/model_test_code/gnn/data_result_npy_directory',
                       help='Directory containing GCN .npy result files')
    parser.add_argument('--mlp_dir', type=str,
                       default='../pretraining/model_test_code/data_result_npy_directory',
                       help='Directory containing MLP .npy result files')
    parser.add_argument('--output_dir', type=str, default='./result_summary/gcn_analysis',
                       help='Output directory for plots and CSV')
    parser.add_argument('--subdirs', type=str, nargs='+', default=None,
                       help='Subdirectories to search (e.g., 10000samples 5000samples)')
    parser.add_argument('--arch', type=str, default=None,
                       help='Specific architecture to highlight (e.g., conv64x2_fc128x2)')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['interpolation', 'extrapolation'],
                       help='Filter by mode')
    parser.add_argument('--graph_mode', type=str, default=None,
                       choices=['full_graph', 'stage_aware'],
                       help='Filter by graph mode')
    parser.add_argument('--experiment', type=str, default=None,
                       choices=['intra_topology', 'topology_agnostic'],
                       help='Filter by experiment type')
    parser.add_argument('--data_type', type=str, default=None,
                       choices=['cell', 'transition'],
                       help='Filter by data type (cell delay or transition time)')
    parser.add_argument('--pooling', type=str, default=None,
                       choices=['mean', 'max', 'add', 'output'],
                       help='Filter by pooling method')
    parser.add_argument('--include_mlp', action='store_true', default=True,
                       help='Include MLP baselines for comparison (default: True)')
    parser.add_argument('--no_mlp', action='store_true',
                       help='Disable MLP baseline comparison')
    parser.add_argument('--scale_rmse', action='store_true', default=True,
                       help='Scale RMSE by 1000 (default: True)')
    parser.add_argument('--no_scale_rmse', action='store_true',
                       help='Do not scale RMSE by 1000')
    parser.add_argument('--top_n', type=int, default=15,
                       help='Number of top architectures to show')
    parser.add_argument('--per_cell', action='store_true',
                       help='Generate separate plots for each cell type (for TSMC data)')
    parser.add_argument('--aadam_iter', type=int, default=None,
                       help='Specific iteration for AADAM baseline (default: use all iterations average)')
    parser.add_argument('--maml_iter', type=int, default=300000,
                       help='Specific iteration for MAML MLP baseline (default: 300000)')
    parser.add_argument('--gcn_iter', type=int, default=300000,
                       help='Specific iteration for GCN results (default: 300000)')
    parser.add_argument('--arch_cell', action='store_true',
                       help='Generate architecture-cell comparison plots (for each arch, show all cells)')
    parser.add_argument('--ex_region', type=str, default='all',
                       choices=['all', 'left_ex', 'right_ex', 'ex_only', 'inter'],
                       help='Filter by extrapolation region: all (default), left_ex (indices 0-4), '
                            'right_ex (indices 56-60), ex_only (left+right combined), inter (indices 5-55)')

    args = parser.parse_args()

    # When ex_region is specified, automatically force mode to extrapolation
    # (ex_region filtering only makes sense for extrapolation mode data)
    if args.ex_region != 'all' and args.mode != 'extrapolation':
        if args.mode == 'interpolation':
            print("Warning: --ex_region is specified but --mode is 'interpolation'.")
            print("         ex_region filtering only applies to extrapolation mode data.")
            print("         Automatically switching to --mode extrapolation.")
        args.mode = 'extrapolation'

    print("=" * 80)
    print("GCN ARCHITECTURE SWEEP ANALYSIS")
    print("=" * 80)
    print(f"GCN data directory: {args.gcn_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.ex_region != 'all':
        print(f"Extrapolation region filter: {args.ex_region}")
        if args.ex_region == 'left_ex':
            print("  -> Filtering to indices 0-4 (left extrapolation)")
        elif args.ex_region == 'right_ex':
            print("  -> Filtering to indices 56-60 (right extrapolation)")
        elif args.ex_region == 'ex_only':
            print("  -> Filtering to indices 0-4 + 56-60 (extrapolation only)")
        elif args.ex_region == 'inter':
            print("  -> Filtering to indices 5-55 (interpolation only)")
        print("  -> Mode automatically set to 'extrapolation'")
    print()

    # Load GCN results
    print("Loading GCN results...")
    gcn_df = load_gcn_results(args.gcn_dir, subdirs=args.subdirs, ex_region=args.ex_region)

    if gcn_df is None or len(gcn_df) == 0:
        print("No GCN results found.")
        return 1

    print(f"Loaded {len(gcn_df)} GCN result files")
    print(f"Unique architectures: {gcn_df['arch_string'].nunique()}")

    # Apply pooling filter if specified
    if args.pooling:
        gcn_df = gcn_df[gcn_df['pooling'] == args.pooling]
        print(f"Filtered to {len(gcn_df)} results with pooling='{args.pooling}'")

    # Apply data_type filter if specified
    if args.data_type:
        gcn_df = gcn_df[gcn_df['data_type'] == args.data_type]
        print(f"Filtered to {len(gcn_df)} results with data_type='{args.data_type}'")

    # Show available data_types and warn if mixed
    if 'data_type' in gcn_df.columns:
        data_types = gcn_df['data_type'].unique().tolist()
        print(f"Data types: {data_types}")
        if len(data_types) > 1 and args.data_type is None:
            print("⚠️  WARNING: Multiple data types found (cell, transition)!")
            print("   Use --data_type cell or --data_type transition to analyze separately.")

    # Show available experiments
    if 'experiment' in gcn_df.columns:
        print(f"Experiments: {gcn_df['experiment'].unique().tolist()}")
    print()

    # Load MLP results for baseline comparison (default: enabled)
    mlp_df = None
    include_mlp = args.include_mlp and not args.no_mlp
    scale_rmse = args.scale_rmse and not args.no_scale_rmse
    if include_mlp:
        print("Loading MLP results for baseline comparison...")
        mlp_df = load_mlp_results(args.mlp_dir, ex_region=args.ex_region)
        if mlp_df is not None:
            print(f"Loaded {len(mlp_df)} MLP result files")
            # Apply data_type filter to MLP if specified
            if args.data_type and 'data_type' in mlp_df.columns:
                mlp_df = mlp_df[mlp_df['data_type'] == args.data_type]
                print(f"  Filtered MLP to {len(mlp_df)} results with data_type='{args.data_type}'")
            # Show available model types
            if 'model_type' in mlp_df.columns:
                model_types = mlp_df['model_type'].unique()
                print(f"  Model types found: {', '.join(model_types)}")
            # Show available AADAM iterations
            aadam_df = mlp_df[mlp_df['model_type'] == 'AADAM']
            if len(aadam_df) > 0 and 'iterations' in aadam_df.columns:
                aadam_iters = sorted(aadam_df['iterations'].unique())
                print(f"  AADAM iterations available: {aadam_iters}")
        else:
            print("No MLP results found - baseline will not be shown")
        print()

    # Create output directory with data_type, pooling, and ex_region suffix if specified
    output_dir = args.output_dir
    if args.data_type:
        output_dir = f"{args.output_dir}_{args.data_type}"
    if args.pooling and args.pooling != 'mean':
        output_dir = f"{output_dir}_pool{args.pooling}"
    if args.ex_region != 'all':
        output_dir = f"{output_dir}_{args.ex_region}"
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("Generating plots...")
    if args.aadam_iter:
        print(f"Using AADAM iteration: {args.aadam_iter}")
    if args.maml_iter:
        print(f"Using MAML MLP iteration: {args.maml_iter}")
    if args.gcn_iter:
        print(f"Using GCN iteration: {args.gcn_iter}")

    # Multi-metric comparison (all data)
    plot_multi_metric_comparison(
        gcn_df, output_dir,
        filter_mode=args.mode, filter_graph_mode=args.graph_mode,
        filter_experiment=args.experiment,
        include_mlp=include_mlp, mlp_df=mlp_df,
        scale_rmse=scale_rmse, top_n=args.top_n, aadam_iter=args.aadam_iter,
        maml_iter=args.maml_iter, gcn_iter=args.gcn_iter
    )

    # If no specific filters, generate for each experiment/mode/graph_mode combination
    if args.mode is None and args.graph_mode is None and args.experiment is None:
        # Check if experiment column exists
        experiments = gcn_df['experiment'].dropna().unique() if 'experiment' in gcn_df.columns else [None]
        for experiment in experiments:
            for mode in gcn_df['mode'].dropna().unique():
                for graph_mode in gcn_df['graph_mode'].dropna().unique():
                    plot_multi_metric_comparison(
                        gcn_df, output_dir,
                        filter_mode=mode, filter_graph_mode=graph_mode,
                        filter_experiment=experiment,
                        include_mlp=include_mlp, mlp_df=mlp_df,
                        scale_rmse=scale_rmse, top_n=args.top_n, aadam_iter=args.aadam_iter,
                        maml_iter=args.maml_iter, gcn_iter=args.gcn_iter
                    )

    # Specific architecture analysis
    if args.arch:
        plot_selected_architecture(
            gcn_df, output_dir, args.arch,
            include_mlp=include_mlp, mlp_df=mlp_df,
            scale_rmse=scale_rmse, aadam_iter=args.aadam_iter
        )

    # Per-cell analysis (for TSMC data with cell names)
    if args.per_cell:
        plot_per_cell_comparison(
            gcn_df, output_dir,
            include_mlp=include_mlp, mlp_df=mlp_df,
            scale_rmse=scale_rmse, top_n=args.top_n, aadam_iter=args.aadam_iter,
            maml_iter=args.maml_iter, gcn_iter=args.gcn_iter
        )

    # Architecture-cell comparison (for each arch, show all cells)
    if args.arch_cell:
        # Generate for all combinations if no filters specified
        if args.mode is None and args.graph_mode is None and args.experiment is None:
            experiments = gcn_df['experiment'].dropna().unique() if 'experiment' in gcn_df.columns else [None]
            for experiment in experiments:
                for mode in gcn_df['mode'].dropna().unique():
                    for graph_mode in gcn_df['graph_mode'].dropna().unique():
                        plot_arch_all_cells_comparison(
                            gcn_df, output_dir,
                            include_mlp=include_mlp, mlp_df=mlp_df,
                            scale_rmse=scale_rmse, aadam_iter=args.aadam_iter,
                            maml_iter=args.maml_iter, gcn_iter=args.gcn_iter,
                            filter_mode=mode, filter_graph_mode=graph_mode,
                            filter_experiment=experiment
                        )
        else:
            # Use specified filters
            plot_arch_all_cells_comparison(
                gcn_df, output_dir,
                include_mlp=include_mlp, mlp_df=mlp_df,
                scale_rmse=scale_rmse, aadam_iter=args.aadam_iter,
                maml_iter=args.maml_iter, gcn_iter=args.gcn_iter,
                filter_mode=args.mode, filter_graph_mode=args.graph_mode,
                filter_experiment=args.experiment
            )

    # Export CSV
    print("\nExporting results to CSV...")
    csv_path = os.path.join(output_dir, 'gcn_results_summary.csv')
    export_results_csv(gcn_df, csv_path)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Best architectures by NRMSE
    print("\nTop 5 Architectures by NRMSE (all data):")
    best_by_nrmse = gcn_df.groupby('arch_string')['NRMSE'].mean().sort_values().head(5)
    for arch, nrmse in best_by_nrmse.items():
        print(f"  {arch}: {nrmse:.3f}%")

    # Best architectures by RMSE
    print("\nTop 5 Architectures by RMSE (all data):")
    best_by_rmse = gcn_df.groupby('arch_string')['RMSE'].mean().sort_values().head(5)
    for arch, rmse in best_by_rmse.items():
        if scale_rmse:
            print(f"  {arch}: {rmse*1000:.3f} (x1000)")
        else:
            print(f"  {arch}: {rmse:.6f}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
