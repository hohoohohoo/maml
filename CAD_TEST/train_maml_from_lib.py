#!/usr/bin/env python3
"""
MAML Training from Liberty Files - Using OptimizedMAML

Each 7x7 LUT table becomes a MAML task:
- Task: (cell, delay_type, related_pin, corner, temp, voltage)
- 49 data points (shots): (slew, load) -> delay
- Uses LOCAL normalization for V, T, slew, load (computed from lib files)

This script uses the same training approach as MAML_topology_pretraining.py
but with lib file data organized as MAML tasks.

Usage:
    python train_maml_from_lib.py --data_type cell --gpu 0 --num_iterations 100000
    python train_maml_from_lib.py --data_type transition --auto_resume
    python train_maml_from_lib.py --resume /path/to/model.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import time
import glob

# MAML import
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/model_code')
from maml_optimized import OptimizedMAML, MAMLModel_3hidden

# Parse arguments
parser = argparse.ArgumentParser(description='MAML Training from Liberty Files')
parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')
parser.add_argument('--data_type', type=str, default='cell', choices=['cell', 'transition'],
                    help='Data type to train on')
parser.add_argument('--num_iterations', type=int, default=10000, help='Number of training iterations')
parser.add_argument('--inner', type=int, default=1, help='Inner loop steps (default: 1)')
parser.add_argument('--innerdiv', type=int, default=100,
                    help='Inner learning rate divisor: inner_lr = 0.001/innerdiv (default: 100)')
parser.add_argument('--meta', type=int, default=32, help='Tasks per meta batch (default: 32)')
parser.add_argument('--K', type=int, default=3, help='Number of shots per task for inner loop (default: 3 for voltage sweep)')
parser.add_argument('--unit_convert', action='store_true',
                    help='Convert units from ns/pf to ps/ff')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to pretrained model to resume from')
parser.add_argument('--auto_resume', action='store_true',
                    help='Automatically find and resume from latest pretrained model')
parser.add_argument('--output_dir', type=str, default='./trained_models_lib',
                    help='Directory to save trained models')
parser.add_argument('--min_table_size', type=int, default=20,
                    help='Minimum number of points in LUT to use as task')
parser.add_argument('--enable_loss_logging', action='store_true',
                    help='Enable training loss logging')
parser.add_argument('--loss_log_every', type=int, default=1000,
                    help='Log training loss every N iterations')
parser.add_argument('--freeze_hidden', action='store_true',
                    help='Freeze hidden layers (l1, l2, l4) and only fine-tune output layer (l3). '
                         'This preserves TSMC pretrained knowledge while adapting output.')
parser.add_argument('--corner', type=str, default=None, choices=['SS', 'TT', 'FF'],
                    help='Filter by corner (default: all corners)')
parser.add_argument('--temperature', type=int, default=None,
                    help='Filter by temperature in Celsius, e.g., -40, 25, 125 (default: all temps)')
args = parser.parse_args()

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Data directories
DATA_DIRS = {
    '0p8v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_0p8v/base_nom_0p8v',
    '0p9v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_0p9v/base_nom_0p9v',
    '1p0v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_1p0v/base_nom_1p0v',
}

# TSMC Pretrained model paths (for initialization)
TSMC_PRETRAINED_PATHS = {
    'cell': '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/training_loss_taskdivide_all/cell_innerdiv100_meta32_combined_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth',
    'transition': '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/training_loss_taskdivide_all/transition_innerdiv100_meta32_combined_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth'
}

# TSMC Process Parameters
PARAM_A = [1.427, 1.457, 1.430, 1.470, 1.443, 1.483, 1.43, 1.47, 1.43, 1.47]
PARAM_B = [0.026, 0.045, 0, 0, -0.026, -0.05, 0.0208, -0.04, 0.036, -0.0208]
PARAM_C = [0.024, 2.000, 0.024, 2.000, 0.024, 2.000, 0.024, 2.000, 0.024, 2.000]
CORNER_TO_IDX = {'FF': 0, 'TT': 1, 'SS': 2, 'FS': 3, 'SF': 4}

# Delay types
DELAY_TYPES = {
    'cell': ['cell_rise', 'cell_fall'],
    'transition': ['rise_transition', 'fall_transition']
}

# Voltage sweep groups (same as validation)
# For each (corner, temp), we have 3 voltage conditions
VOLTAGE_SWEEP_GROUPS = {
    ('SS', -40): [0.72, 0.81, 0.9],
    ('SS', 125): [0.72, 0.81, 0.9],
    ('TT', 25): [0.8, 0.9, 1.0],
    ('FF', -40): [0.99, 1.08, 1.1],
    ('FF', 125): [0.99, 1.08, 1.1],
}

# Center point index in 7x7 LUT (0-indexed)
CENTER_SLEW_IDX = 3
CENTER_LOAD_IDX = 3

# Single-stage cell patterns
SINGLE_STAGE_PATTERNS = [
    r'^INV', r'^CKINV', r'^ND[0-9]', r'^CKND', r'^NR[0-9]', r'^CKNR',
    r'^AOI', r'^OAI', r'^MAOI', r'^MOAI',
]


def get_cell_stage(cell_name):
    for pattern in SINGLE_STAGE_PATTERNS:
        if re.match(pattern, cell_name, re.IGNORECASE):
            return 1
    return 2


def parse_filename(filename):
    match = re.search(r'(ff|tt|ss|fs|sf)(\d+p\d+)v(m?\d+)c', filename.lower())
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")
    corner = match.group(1).upper()
    voltage = float(match.group(2).replace('p', '.'))
    temp_str = match.group(3)
    temperature = -float(temp_str[1:]) if temp_str.startswith('m') else float(temp_str)
    return corner, voltage, temperature


def get_abc_params(corner):
    idx = CORNER_TO_IDX.get(corner.upper(), 1)
    nmos_idx, pmos_idx = idx * 2, idx * 2 + 1
    return {
        'a': (PARAM_A[nmos_idx] + PARAM_A[pmos_idx]) / 2,
        'b': PARAM_B[nmos_idx] + PARAM_B[pmos_idx],
        'c': PARAM_C[nmos_idx] + PARAM_C[pmos_idx]
    }


def parse_lib_file(lib_path):
    """Parse liberty file and extract timing tables."""
    with open(lib_path, 'r') as f:
        content = f.read()

    samples = []
    for cell_match in re.finditer(r'cell\s*\((\w+)\)\s*\{', content):
        cell_name = cell_match.group(1)
        cell_start = cell_match.end()
        brace_count = 1
        for i in range(cell_start, len(content)):
            if content[i] == '{': brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    cell_end = i
                    break
        cell_content = content[cell_start:cell_end]

        for timing_match in re.finditer(r'timing\s*\(\)\s*\{\s*related_pin\s*:\s*"(\w+)"', cell_content):
            related_pin = timing_match.group(1)
            timing_start = timing_match.end()
            t_brace = 1
            for i in range(timing_start, len(cell_content)):
                if cell_content[i] == '{': t_brace += 1
                elif cell_content[i] == '}':
                    t_brace -= 1
                    if t_brace == 0:
                        timing_end = i
                        break
            timing_content = cell_content[timing_start:timing_end]

            for delay_type in ['cell_rise', 'cell_fall', 'rise_transition', 'fall_transition']:
                table_match = re.search(rf'{delay_type}\s*\([^)]+\)\s*\{{', timing_content)
                if table_match:
                    table_start = table_match.end()
                    tb = 1
                    for i in range(table_start, len(timing_content)):
                        if timing_content[i] == '{': tb += 1
                        elif timing_content[i] == '}':
                            tb -= 1
                            if tb == 0:
                                table_end = i
                                break
                    table_content = timing_content[table_start:table_end]

                    idx1 = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)', table_content)
                    idx2 = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)', table_content)
                    vals = re.search(r'values\s*\(\s*(.*?)\s*\)\s*;', table_content, re.DOTALL)

                    if idx1 and idx2 and vals:
                        index_1 = [float(x.strip()) for x in idx1.group(1).split(',')]
                        index_2 = [float(x.strip()) for x in idx2.group(1).split(',')]
                        values_str = vals.group(1).replace('\\', '').replace('\n', ' ')
                        rows = re.findall(r'"([^"]+)"', values_str)
                        values = [[float(x.strip()) for x in row.split(',')] for row in rows]

                        samples.append({
                            'cell_name': cell_name,
                            'delay_type': delay_type,
                            'related_pin': related_pin,
                            'index_1': index_1,
                            'index_2': index_2,
                            'values': values,
                            'stage': get_cell_stage(cell_name)
                        })
    return samples


def extract_center_points_from_lib(parsed_samples, data_type, corner, voltage, temperature,
                                    unit_convert=True):
    """
    Extract center point (3,3) from each LUT in lib file.
    Returns dict: {(cell, delay_type, related_pin): (X, y)} for this voltage condition.
    """
    abc = get_abc_params(corner)
    valid_delay_types = DELAY_TYPES[data_type]
    center_points = {}

    for sample in parsed_samples:
        if sample['delay_type'] not in valid_delay_types:
            continue

        # Check if center point exists in this LUT
        if (CENTER_SLEW_IDX >= len(sample['index_1']) or
            CENTER_LOAD_IDX >= len(sample['index_2']) or
            CENTER_SLEW_IDX >= len(sample['values']) or
            CENTER_LOAD_IDX >= len(sample['values'][CENTER_SLEW_IDX])):
            continue

        delay_indicator = -1 if 'rise' in sample['delay_type'] else 1
        stage = sample['stage']

        slew = sample['index_1'][CENTER_SLEW_IDX]
        load = sample['index_2'][CENTER_LOAD_IDX]
        delay = sample['values'][CENTER_SLEW_IDX][CENTER_LOAD_IDX]

        if unit_convert:
            slew_val = slew * 1000
            load_val = load * 1000
        else:
            slew_val = slew
            load_val = load

        # Input: [a, b, c, temp, voltage, stage, delay_indicator, slew, load]
        inp = torch.tensor([abc['a'], abc['b'], abc['c'],
                           temperature, voltage, stage, delay_indicator,
                           slew_val, load_val], dtype=torch.float32)
        out = torch.tensor([delay], dtype=torch.float32)

        key = (sample['cell_name'], sample['delay_type'], sample['related_pin'])
        center_points[key] = (inp, out)

    return center_points


def create_voltage_based_tasks(all_center_points, corner, temp):
    """
    Create 3-shot tasks from center points across 3 voltage conditions.

    Args:
        all_center_points: dict {voltage: {(cell, delay_type, pin): (X, y)}}
        corner: corner name (e.g., 'SS')
        temp: temperature (e.g., 125)

    Returns:
        list of (X, y) tuples where X is [3, 9] and y is [3, 1]
    """
    voltage_group = VOLTAGE_SWEEP_GROUPS.get((corner.upper(), temp))
    if not voltage_group:
        return []

    # Find common cells across all 3 voltages
    voltage_keys = list(all_center_points.keys())
    if len(voltage_keys) < 3:
        return []

    # Get cell keys that exist in all 3 voltages
    common_cells = None
    for voltage in voltage_keys:
        cells = set(all_center_points[voltage].keys())
        if common_cells is None:
            common_cells = cells
        else:
            common_cells = common_cells & cells

    if not common_cells:
        return []

    # Create 3-shot tasks
    tasks = []
    for cell_key in common_cells:
        X_list = []
        y_list = []

        for voltage in sorted(voltage_keys):
            inp, out = all_center_points[voltage][cell_key]
            X_list.append(inp)
            y_list.append(out)

        X = torch.stack(X_list)  # [3, 9]
        y = torch.stack(y_list)  # [3, 1]
        tasks.append((X, y))

    return tasks


def extract_all_lut_points_for_stats(parsed_samples, data_type, corner, voltage, temperature, unit_convert=True):
    """
    Extract ALL points from each LUT (not just center) for normalization stats calculation.
    This matches the validation script's approach.

    Returns list of (temp, voltage, slew, load) tuples.
    """
    abc = get_abc_params(corner)
    valid_delay_types = DELAY_TYPES[data_type]
    all_points = []

    for sample in parsed_samples:
        if sample['delay_type'] not in valid_delay_types:
            continue

        # Extract all points from 7x7 LUT
        for slew_idx, slew in enumerate(sample['index_1']):
            for load_idx, load in enumerate(sample['index_2']):
                if slew_idx >= len(sample['values']) or load_idx >= len(sample['values'][slew_idx]):
                    continue

                if unit_convert:
                    slew_val = slew * 1000
                    load_val = load * 1000
                else:
                    slew_val = slew
                    load_val = load

                all_points.append((temperature, voltage, slew_val, load_val))

    return all_points


def compute_local_norm_stats_from_all_lut(all_lut_points):
    """
    Compute LOCAL normalization stats from ALL LUT points (not just center).
    This matches validation script's compute_local_norm_stats behavior.

    Args:
        all_lut_points: list of (temp, voltage, slew, load) tuples

    Returns:
        norm_stats dict with mean/std for indices 3, 4, 7, 8
    """
    all_values = {3: [], 4: [], 7: [], 8: []}  # temp, voltage, slew, load

    for temp, voltage, slew, load in all_lut_points:
        all_values[3].append(temp)
        all_values[4].append(voltage)
        all_values[7].append(slew)
        all_values[8].append(load)

    norm_stats = {}
    feature_names = {3: 'temperature', 4: 'voltage', 7: 'slew', 8: 'load'}

    print("\n📊 Computing LOCAL normalization stats from ALL LUT points (matching validation):")

    for idx in [3, 4, 7, 8]:
        values = torch.tensor(all_values[idx], dtype=torch.float32)
        mean_val = values.mean().item()
        std_val = values.std().item()
        norm_stats[idx] = {'mean': mean_val, 'std': std_val if std_val > 1e-8 else 1.0}
        print(f"   {feature_names[idx]} (idx {idx}): mean={mean_val:.6f}, std={std_val:.6f}")

    return norm_stats


def compute_local_norm_stats(all_tasks):
    """
    [DEPRECATED - use compute_local_norm_stats_from_all_lut instead]
    Compute LOCAL normalization stats from all task data (center points only).
    Stats for: temperature (idx 3), voltage (idx 4), slew (idx 7), load (idx 8)
    """
    all_X = torch.cat([task[0] for task in all_tasks], dim=0)

    norm_indices = [3, 4, 7, 8]  # temp, voltage, slew, load
    feature_names = ['temperature', 'voltage', 'slew', 'load']

    norm_stats = {}
    print("\n📊 Computing LOCAL normalization stats from center points only:")

    for idx, name in zip(norm_indices, feature_names):
        mean_val = all_X[:, idx].mean().item()
        std_val = all_X[:, idx].std().item()
        norm_stats[idx] = {'mean': mean_val, 'std': std_val if std_val > 1e-8 else 1.0}
        print(f"   {name} (idx {idx}): mean={mean_val:.6f}, std={std_val:.6f}")

    return norm_stats


def normalize_tasks_local(all_tasks, norm_stats):
    """
    Normalize task inputs using LOCAL normalization stats.
    Also normalize outputs per-task (as in MAML training).

    For voltage-based 3-shot tasks:
    - Input: [num_tasks, 3, 9]
    - Output: [num_tasks, 3, 1]
    """
    normalize_indices = [3, 4, 7, 8]  # temp, voltage, slew, load

    normalized_inputs = []
    normalized_outputs = []
    valid_task_count = 0
    skipped_low_std = 0
    skipped_nan = 0

    for X, y in all_tasks:
        # Normalize input features using LOCAL stats
        X_norm = X.clone()
        for idx in normalize_indices:
            mean_val = norm_stats[idx]['mean']
            std_val = norm_stats[idx]['std']
            X_norm[:, idx] = (X_norm[:, idx] - mean_val) / std_val

        # Per-task output normalization
        y_mean = y.mean()
        y_std = y.std()

        # Skip tasks with near-zero variance
        if y_std < 1e-6:
            skipped_low_std += 1
            continue

        y_norm = (y - y_mean) / y_std

        # Check for NaN/Inf
        if torch.isnan(X_norm).any() or torch.isnan(y_norm).any():
            skipped_nan += 1
            continue
        if torch.isinf(X_norm).any() or torch.isinf(y_norm).any():
            skipped_nan += 1
            continue

        normalized_inputs.append(X_norm)
        normalized_outputs.append(y_norm)
        valid_task_count += 1

    print(f"\n✅ Normalized {valid_task_count} tasks (3-shot each)")
    if skipped_low_std > 0:
        print(f"   Skipped {skipped_low_std} tasks (low output std)")
    if skipped_nan > 0:
        print(f"   Skipped {skipped_nan} tasks (NaN/Inf)")

    # Stack into tensors: [num_tasks, 3, features]
    data_input = torch.stack(normalized_inputs)
    data_output = torch.stack(normalized_outputs)

    return data_input, data_output


def extract_iteration_from_filename(filepath, layer_length, inner):
    """Extract iteration number from model filename."""
    filename = os.path.basename(filepath)
    pattern = rf'_\({layer_length}\)_(\d+)_inner{inner}'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return 0


def find_pretrained_model(model_dir, layer_length, inner, innerdiv, meta, data_type):
    """Find existing pretrained model."""
    pattern = f"{data_type}_lib_innerdiv{innerdiv}_meta{meta}_*_3hidden_({layer_length})_*_inner{inner}*.pth"
    search_pattern = os.path.join(model_dir, pattern)
    models = glob.glob(search_pattern)

    if not models:
        return None, 0

    # Find model with highest iteration
    best_model = None
    best_iter = 0
    for model_path in models:
        iteration = extract_iteration_from_filename(model_path, layer_length, inner)
        if iteration > best_iter:
            best_iter = iteration
            best_model = model_path

    return best_model, best_iter


def main():
    print("=" * 80)
    print("MAML Training from Liberty Files (OptimizedMAML)")
    print("Using LOCAL normalization for V, T, slew, load")
    print("=" * 80)

    layer_length = 40
    inner = args.inner
    innerdiv = args.innerdiv
    meta = args.meta
    data_type = args.data_type

    print(f"\n⚙️ Training configuration:")
    print(f"   Data type: {data_type}")
    print(f"   Corner filter: {args.corner if args.corner else 'all'}")
    print(f"   Temperature filter: {args.temperature}°C" if args.temperature is not None else "   Temperature filter: all")
    print(f"   Inner loop steps: {inner}")
    print(f"   Inner LR: 0.001/{innerdiv} = {0.001/innerdiv}")
    print(f"   Tasks per meta batch: {meta}")
    print(f"   Iterations: {args.num_iterations}")
    print(f"   Unit conversion: {args.unit_convert}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Parse all lib files and create voltage-based 3-shot tasks
    print("\n📂 Parsing liberty files (voltage-based 3-shot tasks)...")
    print(f"   Center point: slew[{CENTER_SLEW_IDX}], load[{CENTER_LOAD_IDX}]")

    # Step 0: Collect ALL LUT points from ALL lib files for normalization stats
    # This matches validation's approach (uses all 15 conditions, not just VOLTAGE_SWEEP_GROUPS)
    print("\n📊 Step 0: Collecting ALL LUT points for normalization stats (matching validation)...")
    all_lut_points_for_stats = []
    all_lib_files = []

    for voltage_dir, dir_path in DATA_DIRS.items():
        if not os.path.exists(dir_path):
            print(f"  ⚠️ Directory not found: {dir_path}")
            continue

        for lib_file in sorted(Path(dir_path).glob('lib1_*.tlib')):
            match = re.search(r'lib1_(\w+)_base_400\.tlib', lib_file.name)
            if not match:
                continue

            try:
                corner, voltage, temperature = parse_filename(lib_file.name)
            except ValueError as e:
                continue

            all_lib_files.append((lib_file, corner, voltage, temperature))

    # Parse all lib files and collect ALL LUT points for normalization
    for lib_file, corner, voltage, temperature in all_lib_files:
        samples = parse_lib_file(str(lib_file))
        all_points = extract_all_lut_points_for_stats(
            samples, data_type, corner, voltage, temperature, args.unit_convert
        )
        all_lut_points_for_stats.extend(all_points)
        print(f"  Parsed {lib_file.name}: {len(all_points)} LUT points")

    print(f"  Total LUT points for normalization: {len(all_lut_points_for_stats)}")

    # Step 1: Collect lib files grouped by (corner, temp) for task creation
    # Structure: {(corner, temp): {voltage: lib_file_path}}
    print("\n📂 Step 1: Grouping lib files for task creation...")
    lib_files_by_group = defaultdict(dict)

    for lib_file, corner, voltage, temperature in all_lib_files:
        # Check if this (corner, temp) is in our sweep groups
        group_key = (corner.upper(), int(temperature))
        if group_key in VOLTAGE_SWEEP_GROUPS:
            lib_files_by_group[group_key][voltage] = str(lib_file)

    # Step 2: For each (corner, temp) group, extract center points and create tasks
    filter_msg = ""
    if args.corner:
        filter_msg += f" corner={args.corner}"
    if args.temperature is not None:
        filter_msg += f" temp={args.temperature}°C"
    if filter_msg:
        print(f"\n📂 Step 2: Creating 3-shot tasks from center points... (filter:{filter_msg})")
    else:
        print("\n📂 Step 2: Creating 3-shot tasks from center points...")
    all_tasks = []

    for (corner, temp), voltage_files in lib_files_by_group.items():
        # Apply corner/temperature filter
        if args.corner and corner != args.corner.upper():
            continue
        if args.temperature is not None and temp != args.temperature:
            continue

        expected_voltages = VOLTAGE_SWEEP_GROUPS[(corner, temp)]

        # Check if we have all 3 voltages
        if len(voltage_files) < 3:
            print(f"  ({corner}, {temp}°C): Only {len(voltage_files)} voltages, skipping")
            continue

        # Extract center points from each voltage's lib file
        center_points_by_voltage = {}

        for voltage, lib_path in voltage_files.items():
            samples = parse_lib_file(lib_path)

            # Extract center points for task creation
            center_points = extract_center_points_from_lib(
                samples, data_type, corner, voltage, temp, args.unit_convert
            )
            center_points_by_voltage[voltage] = center_points

        # Create 3-shot tasks
        tasks = create_voltage_based_tasks(center_points_by_voltage, corner, temp)
        print(f"  ({corner}, {temp}°C): {len(tasks)} 3-shot tasks from {len(voltage_files)} voltages")
        all_tasks.extend(tasks)

    print(f"\n📊 Total tasks: {len(all_tasks)} (each is a 3-shot voltage sweep)")

    if len(all_tasks) == 0:
        print("❌ No tasks found. Exiting.")
        return

    # Compute LOCAL normalization stats from ALL LUT points (matching validation)
    norm_stats = compute_local_norm_stats_from_all_lut(all_lut_points_for_stats)

    # Save norm stats
    norm_stats_path = os.path.join(args.output_dir, f'local_norm_stats_{data_type}.pth')
    torch.save(norm_stats, norm_stats_path)
    print(f"💾 Saved local norm stats to: {norm_stats_path}")

    # Normalize tasks using LOCAL stats
    print("\n🔧 Normalizing tasks with LOCAL stats...")
    data_input, data_output = normalize_tasks_local(all_tasks, norm_stats)

    print(f"📊 Data shapes: Input {data_input.shape}, Output {data_output.shape}")

    # Move to GPU
    data_input = data_input.to(device)
    data_output = data_output.to(device)

    # Build loss logging configuration
    loss_logging_config = {
        'enabled': args.enable_loss_logging,
        'log_every': args.loss_log_every,
        'save_dir': args.output_dir
    }

    # Create MAML model
    print("\n🤖 Creating OptimizedMAML model...")
    input_features = data_input.shape[2]
    calculated_inner_lr = 0.001 / innerdiv

    print(f"   Input features: {input_features}")
    print(f"   Hidden layer size: {layer_length}")
    print(f"   K (shots per task): {args.K}")
    print(f"   Inner LR: {calculated_inner_lr}")
    print(f"   Meta LR: 0.0001")

    maml = OptimizedMAML(
        model=MAMLModel_3hidden(in_features=input_features, layer_length=layer_length),
        dataset_in=data_input,
        dataset_out=data_output,
        inner_lr=calculated_inner_lr,
        meta_lr=0.0001,
        K=args.K,  # 3-shot for voltage sweep
        inner_steps=inner,
        tasks_per_meta_batch=meta,
        loss_logging_config=loss_logging_config
    )

    # Load pretrained model
    # Priority: 1) --resume (specific model), 2) --auto_resume (latest local), 3) TSMC pretrained (default)
    start_iteration = 0
    loaded_model = False

    if args.resume:
        # Load specific model specified by user
        if os.path.exists(args.resume):
            print(f"📂 Loading specified model: {args.resume}")
            state_dict = torch.load(args.resume, map_location=device)
            maml.model.load_state_dict(state_dict)
            start_iteration = extract_iteration_from_filename(args.resume, layer_length, inner)
            loaded_model = True
            print(f"✅ Loaded model, resuming from iteration {start_iteration}")
        else:
            print(f"⚠️ Model file not found: {args.resume}")

    elif args.auto_resume:
        # Auto-find latest locally trained model
        model_path, iteration = find_pretrained_model(
            args.output_dir, layer_length, inner, innerdiv, meta, data_type
        )
        if model_path:
            print(f"📂 Auto-loading local model: {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            maml.model.load_state_dict(state_dict)
            start_iteration = iteration
            loaded_model = True
            print(f"✅ Loaded model, resuming from iteration {start_iteration}")

    # Default: Load TSMC pretrained model as initialization
    if not loaded_model:
        tsmc_model_path = TSMC_PRETRAINED_PATHS.get(data_type)
        if tsmc_model_path and os.path.exists(tsmc_model_path):
            print(f"\n🔄 Loading TSMC pretrained model as initialization:")
            print(f"   {os.path.basename(tsmc_model_path)}")
            state_dict = torch.load(tsmc_model_path, map_location=device)
            maml.model.load_state_dict(state_dict)
            loaded_model = True
            print(f"✅ TSMC pretrained model loaded successfully")
            print(f"   Starting fine-tuning from iteration 0")
        else:
            print(f"⚠️ TSMC pretrained model not found: {tsmc_model_path}")
            print("🆕 Starting training from scratch (random initialization)")

    # Freeze hidden layers if requested (Hybrid approach)
    if args.freeze_hidden:
        print("\n🧊 Freezing hidden layers (Hybrid approach)...")
        frozen_params = 0
        trainable_params = 0

        for name, param in maml.model.named_parameters():
            # l3 is the output layer (despite the name), keep it trainable
            # l1, l2, l4 are hidden layers, freeze them
            if 'l3' in name:
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"   ✓ Trainable: {name} ({param.numel()} params)")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
                print(f"   ✗ Frozen: {name} ({param.numel()} params)")

        print(f"\n   Total frozen: {frozen_params} params")
        print(f"   Total trainable: {trainable_params} params")
        print(f"   Trainable ratio: {trainable_params / (frozen_params + trainable_params) * 100:.1f}%")

    # Training
    print("\n" + "=" * 80)
    print("Starting MAML Training")
    print("=" * 80)

    # Determine chunk size (2000 iterations per chunk)
    chunk_size = 2000
    num_chunks = max(1, (args.num_iterations + chunk_size - 1) // chunk_size)  # Ceiling division, min 1

    start_time = time.time()
    print(f"   Chunks: {num_chunks} x {chunk_size} iterations")

    for chunk in range(1, num_chunks + 1):
        current_iteration = start_iteration + (chunk * chunk_size)
        chunk_start = start_iteration + ((chunk - 1) * chunk_size)

        print(f"\n▶️ Chunk {chunk}/{num_chunks}: iterations [{chunk_start} → {current_iteration}]")

        torch.cuda.synchronize()
        chunk_start_time = time.time()

        try:
            maml.main_loop_optimized(num_iterations=chunk_size, start_iteration=chunk_start)
        except Exception as e:
            print(f"⚠️ Optimized loop failed: {e}")
            print("   Falling back to sequential loop...")
            try:
                maml.main_loop_sequential(num_iterations=chunk_size, start_iteration=chunk_start)
            except Exception as e2:
                print(f"⚠️ Sequential loop also failed: {e2}")
                maml.inner_lr *= 0.5
                maml.meta_lr *= 0.5
                maml.main_loop_sequential(num_iterations=chunk_size // 2, start_iteration=chunk_start)

        torch.cuda.synchronize()
        chunk_time = time.time() - chunk_start_time

        print(f"⏱️ Chunk completed in {chunk_time:.2f}s ({chunk_time/chunk_size:.4f}s/iter)")

        # GPU memory
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU Memory: {mem_alloc:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        # Save checkpoint
        freeze_suffix = "_freeze" if args.freeze_hidden else ""
        condition_suffix = ""
        if args.corner:
            condition_suffix += f"_{args.corner}"
        if args.temperature is not None:
            condition_suffix += f"_{args.temperature}C"
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{data_type}_lib_innerdiv{innerdiv}_meta{meta}_local_norm{freeze_suffix}{condition_suffix}_3hidden_({layer_length})_{current_iteration}_inner{inner}.pth"
        )
        torch.save(maml.model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {os.path.basename(checkpoint_path)}")

    # Save final model
    final_iteration = start_iteration + args.num_iterations
    freeze_suffix = "_freeze" if args.freeze_hidden else ""
    condition_suffix = ""
    if args.corner:
        condition_suffix += f"_{args.corner}"
    if args.temperature is not None:
        condition_suffix += f"_{args.temperature}C"
    final_path = os.path.join(
        args.output_dir,
        f"{data_type}_lib_innerdiv{innerdiv}_meta{meta}_local_norm{freeze_suffix}{condition_suffix}_3hidden_({layer_length})_{final_iteration}_inner{inner}.pth"
    )
    torch.save(maml.model.state_dict(), final_path)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("🏁 Training Complete!")
    print("=" * 80)
    print(f"   Final model: {os.path.basename(final_path)}")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"   Final iteration: {final_iteration}")

    # Save loss log if enabled
    if args.enable_loss_logging and hasattr(maml, 'iteration_loss_log') and maml.iteration_loss_log:
        loss_log_path = os.path.join(
            args.output_dir,
            f"loss_log_{data_type}_lib_innerdiv{innerdiv}_meta{meta}{condition_suffix}_iter{final_iteration}.json"
        )
        maml.save_loss_log(loss_log_path)
        print(f"   Loss log: {loss_log_path}")


if __name__ == '__main__':
    main()
