#!/usr/bin/env python3
"""
LUT Table-based Voltage Sweep Validation - Full Cell Version
- Supports all cells from full lib files (not just AND)
- Adds stage indicator: 1 for single-stage (INV, NAND, NOR), 2 for multi-stage
- Input format: [a, b, c, temperature, voltage, stage, delay_indicator, slew, load]

Workflow:
1. For each (cell, delay_type, related_pin, corner, temp) group:
   a. Use center LUT point (3,3) with 3 voltage conditions → 3-shot Adam adaptation
   b. For each of the other 48 (slew, load) points:
      - Use 2 voltage conditions (low, high) as support
      - Apply selective Adam (if loss > threshold) or grad/move

Usage:
  python validate_lut_table_sweep_full_cells.py --unit_convert
  python validate_lut_table_sweep_full_cells.py --unit_convert --num_tables -1
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import re
from pathlib import Path
from collections import OrderedDict, defaultdict

# Parse arguments first
parser = argparse.ArgumentParser(description='LUT Table Voltage Sweep Validation - Full Cells')
parser.add_argument('--num_tables', type=int, default=-1,
                    help='Number of LUT tables to evaluate. Default: 100 (use -1 for all)')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device ID. Default: 0')
parser.add_argument('--unit_convert', action='store_true',
                    help='Convert units from ns/pf to ps/ff (multiply slew/load by 1000)')
parser.add_argument('--center_steps', type=int, default=100,
                    help='Number of Adam steps for center point adaptation. Default: 40')
parser.add_argument('--center_lr', type=float, default=3e-3,
                    help='Learning rate for center point adaptation. Default: 3e-4')
parser.add_argument('--center_loss_threshold', type=float, default=1e-4,
                    help='Loss threshold for center point Adam adaptation. Adam only runs if loss > threshold. Default: 1e-4')
parser.add_argument('--mape_threshold', type=float, default=0.05,
                    help='Center MAPE threshold (%) for applying Adam to peripheral points. Default: 1.0')
parser.add_argument('--peripheral_adam_steps', type=int, default=100,
                    help='Number of Adam steps for peripheral adaptation when loss > threshold. Default: 20')
parser.add_argument('--peripheral_adam_lr', type=float, default=3e-4,
                    help='Learning rate for peripheral Adam adaptation. Default: 1e-3')
parser.add_argument('--stage_filter', type=int, default=0, choices=[0, 1, 2],
                    help='Filter by stage: 0=all, 1=single-stage only, 2=multi-stage only. Default: 0')
parser.add_argument('--local_norm', action='store_true',
                    help='Use local normalization stats for slew/load from lib files instead of TSMC training data')
parser.add_argument('--local_temp_norm', action='store_true',
                    help='Use local normalization stats for temperature (-40, 25, 125) instead of TSMC training data')
parser.add_argument('--local_volt_norm', action='store_true',
                    help='Use local normalization stats for voltage instead of TSMC training data')
parser.add_argument('--threshold_sweep', action='store_true',
                    help='Run multi-threshold sweep in single pass (evaluates multiple thresholds efficiently)')
parser.add_argument('--ref_mode', type=str, default='corner', choices=['corner', 'middle', 'both'],
                    help='Reference voltage mode for grad/move calculation: '
                         'corner=use 0.9V equivalent per corner (TT:idx1, SS:idx2, FF:idx0), '
                         'middle=always use middle index (idx1), '
                         'both=run both modes and compare. Default: corner')
parser.add_argument('--peripheral_mode', type=str, default='grad_move',
                    choices=['grad_adam', 'adam_only', 'grad_move'],
                    help='Peripheral prediction mode: '
                         'grad_adam=selective grad+Adam based on center_mape (default), '
                         'adam_only=always use Adam without grad/move transformation, '
                         'grad_move=always use grad+move without Adam')
parser.add_argument('--cell_model_path', type=str, default=None,
                    help='Custom cell model path (overrides default). '
                         'If not specified, uses TSMC pretrained baseline.')
parser.add_argument('--transition_model_path', type=str, default=None,
                    help='Custom transition model path (overrides default). '
                         'If not specified, uses TSMC pretrained baseline.')
parser.add_argument('--norm_stats_path', type=str, default=None,
                    help='Path to pre-computed normalization stats file (.pth). '
                         'Use this to ensure training and validation use identical normalization. '
                         'Example: ./trained_models_lib/local_norm_stats_cell.pth')
parser.add_argument('--model_base_dir', type=str, default=None,
                    help='Base directory for condition-specific models. '
                         'When specified, models and norm_stats are auto-loaded from '
                         '{model_base_dir}/{data_type}_{corner}_{temp}C/ subdirectories. '
                         'Example: ./trained_models_lib_sweep')
parser.add_argument('--corner_filter', type=str, default=None,
                    help='Filter validation by corners (comma-separated). '
                         'Example: --corner_filter SS,FF to only validate SS and FF corners. '
                         'If not specified, validates all corners in VOLTAGE_SWEEP_GROUPS.')
parser.add_argument('--temp_filter', type=str, default=None,
                    help='Filter validation by temperatures (comma-separated). '
                         'Example: --temp_filter 125,-40 to only validate 125C and -40C. '
                         'If not specified, validates all temperatures in VOLTAGE_SWEEP_GROUPS.')
parser.add_argument('--iteration', type=int, default=None,
                    help='Specific iteration number for model checkpoint. '
                         'Example: --iteration 4000 to load *_4000_inner1.pth. '
                         'If not specified, uses the most recent model file.')
args = parser.parse_args()

# Parse corner and temperature filters
corner_filter = None
if args.corner_filter:
    corner_filter = [c.strip().upper() for c in args.corner_filter.split(',')]
    print(f"Corner filter: {corner_filter}")

temp_filter = None
if args.temp_filter:
    temp_filter = [int(t.strip()) for t in args.temp_filter.split(',')]
    print(f"Temperature filter: {temp_filter}")

# MAPE threshold grid for sweep mode (in percentage)
# If center_mape > threshold: apply grad-only + Adam
# If center_mape <= threshold: apply grad + move (no Adam)
THRESHOLD_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0]

# Add paths
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/model_code')
from maml_optimized import OptimizedMAML, MAMLModel_3hidden

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Unit conversion (ns/pf → ps/ff): {args.unit_convert}")
print(f"Normalization - Slew/Load: {'Local' if args.local_norm else 'TSMC'}, Temp: {'Local' if args.local_temp_norm else 'TSMC'}, Volt: {'Local' if args.local_volt_norm else 'TSMC'}")
print(f"Center adaptation: {args.center_steps} steps, lr={args.center_lr}")
print(f"Peripheral selective Adam: center_mape_threshold={args.mape_threshold}%, steps={args.peripheral_adam_steps}, lr={args.peripheral_adam_lr}")
print(f"Stage filter: {args.stage_filter} (0=all, 1=single-stage, 2=multi-stage)")
print(f"Reference voltage mode: {args.ref_mode} (corner=0.9V per corner, middle=always idx1, both=compare)")
print(f"Peripheral mode: {args.peripheral_mode} (grad_adam=selective, adam_only=direct fit, grad_move=no Adam)")
if args.model_base_dir:
    print(f"Model base dir: {args.model_base_dir}, Iteration: {args.iteration if args.iteration else 'latest'}")

# Configuration - Full lib file directories
DATA_DIRS = {
    '0p8v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_0p8v/base_nom_0p8v',
    '0p9v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_0p9v/base_nom_0p9v',
    '1p0v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_1p0v/base_nom_1p0v',
}

# Model paths for cell and transition
# Baseline model paths (TSMC pretrained)
BASELINE_MODEL_PATHS = {
    #'cell': '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/training_loss_taskdivide_all/cell_innerdiv100_meta32_combined_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth',
    'cell': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/trained_models_lib/checkpoints/cell_lib_innerdiv100_meta32_local_norm_TT_25C_3hidden_(40)_4000_inner1.pth',
    #'transition': '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/training_loss_taskdivide_all/transition_innerdiv100_meta32_combined_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth'
    'transition' : '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/trained_models_lib/checkpoints/transition_lib_innerdiv100_meta32_local_norm_TT_25C_3hidden_(40)_4000_inner1.pth'
}

# Determine actual model path (custom or baseline)
def get_model_path(data_type):
    if data_type == 'cell' and args.cell_model_path:
        return args.cell_model_path
    elif data_type == 'transition' and args.transition_model_path:
        return args.transition_model_path
    return BASELINE_MODEL_PATHS[data_type]


def get_condition_paths(data_type, corner, temp, base_dir=None, iteration=None):
    """
    Get model and norm_stats paths for a specific corner/temperature condition.

    Args:
        data_type: 'cell' or 'transition'
        corner: Process corner (e.g., 'TT', 'SS', 'FF')
        temp: Temperature (e.g., 25, -40, 125)
        base_dir: Base directory for condition-specific models (e.g., './trained_models_lib_sweep')
        iteration: Specific iteration number (e.g., 4000 for *_4000_inner1.pth). If None, uses most recent.

    Returns:
        dict with 'model_path' and 'norm_stats_path' keys, or None if not found
    """
    if base_dir is None:
        return None

    # Format temperature for directory name (e.g., -40 -> m40C, 25 -> 25C, 125 -> 125C)
    if temp < 0:
        temp_str = f"m{abs(int(temp))}C"
    else:
        temp_str = f"{int(temp)}C"

    # Construct condition directory path (e.g., cell_SS_125C)
    condition_dir = os.path.join(base_dir, f"{data_type}_{corner.upper()}_{temp_str}")

    if not os.path.exists(condition_dir):
        print(f"  Warning: Condition directory not found: {condition_dir}")
        return None

    # Find model file (pattern: {data_type}_lib_*_{corner}_{temp}*_inner1.pth)
    # For negative temps, also try -40C format (in addition to m40C)
    temp_patterns = [temp_str]
    if temp < 0:
        temp_patterns.append(f"{int(temp)}C")  # e.g., "-40C"

    # Search in both main directory and checkpoints subdirectory
    search_dirs = [condition_dir]
    checkpoints_dir = os.path.join(condition_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        search_dirs.append(checkpoints_dir)

    model_files = []
    for search_dir in search_dirs:
        for t_pattern in temp_patterns:
            model_files.extend(Path(search_dir).glob(f"{data_type}_lib_*_{corner.upper()}_{t_pattern}_*_inner1.pth"))
            if not model_files:
                # Also try without _lib_ prefix
                model_files.extend(Path(search_dir).glob(f"{data_type}_*_{corner.upper()}_{t_pattern}_*_inner1.pth"))

    if not model_files:
        print(f"  Warning: No model file found in: {condition_dir} (or checkpoints/)")
        return None

    # Filter by iteration if specified
    if iteration is not None:
        iter_pattern = f"_{iteration}_inner1.pth"
        filtered_files = [f for f in model_files if iter_pattern in str(f)]
        if not filtered_files:
            print(f"  Warning: No model file found with iteration {iteration} in: {condition_dir}")
            return None
        model_files = filtered_files

    # Use the most recent model file (by modification time)
    model_path = str(max(model_files, key=lambda x: x.stat().st_mtime))

    # Find norm_stats file
    norm_stats_path = os.path.join(condition_dir, f"local_norm_stats_{data_type}.pth")
    if not os.path.exists(norm_stats_path):
        print(f"  Warning: Norm stats file not found: {norm_stats_path}")
        return None

    return {
        'model_path': model_path,
        'norm_stats_path': norm_stats_path
    }

# TSMC training data paths for normalization stats
TSMC_TRAIN_INPUT_PATHS = {
    'cell': '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/MLP_dataset_TSMC/combined_data/tsmc_topology_agnostic_train_input_cell.pth',
    'transition': '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/MLP_dataset_TSMC/combined_data/tsmc_topology_agnostic_train_input_transition.pth'
}

# Delay types for each data type
DELAY_TYPES = {
    'cell': ['cell_rise', 'cell_fall'],
    'transition': ['rise_transition', 'fall_transition']
}

# Voltage sweep groups: (corner, temp) -> [conditions sorted by voltage]
VOLTAGE_SWEEP_GROUPS = {
    ('FF', -40): ['ff0p88vm40c', 'ff0p99vm40c', 'ff1p1vm40c'],
    ('FF', 125): ['ff0p88v125c', 'ff0p99v125c', 'ff1p1v125c'],
    ('SS', -40): ['ss0p72vm40c', 'ss0p81vm40c', 'ss0p9vm40c'],
    ('SS', 125): ['ss0p72v125c', 'ss0p81v125c', 'ss0p9v125c'],
    ('TT', 25):  ['tt0p8v25c', 'tt0p9v25c', 'tt1p0v25c'],
}

# LUT center point (for 7x7 table, center is at index 3)
CENTER_SLEW_IDX = 3
CENTER_LOAD_IDX = 3

# Reference voltage index for grad/move calculation by corner
# Each corner has 3 voltage conditions [low, mid, high], index selects which one to use as reference
# TT: 0.8V(0), 0.9V(1), 1.0V(2) → use 0.9V (index 1)
# SS: 0.72V(0), 0.81V(1), 0.9V(2) → use 0.9V (index 2)
# FF: 0.88V(0), 0.99V(1), 1.1V(2) → use 0.88V (index 0)
VOLTAGE_REF_IDX = {
    'TT': 1,  # 0.9V
    'SS': 2,  # 0.9V
    'FF': 0,  # 0.88V
}

# Peripheral point support/query index configuration by (corner, temp)
# Format: (corner, temp) -> {'support': [idx1, idx2], 'query': idx}
# FF 125, SS 125: support=[0,1], query=2
# TT (all temps): support=[1,2], query=0
# FF -40, SS -40: support=[0,2], query=1
PERIPHERAL_VOLTAGE_CONFIG = {
    ('FF', 125): {'support': [0, 1], 'query': 2},
    ('SS', 125): {'support': [0, 1], 'query': 2},
    ('TT', 25): {'support': [1, 2], 'query': 0},
    ('FF', -40): {'support': [0, 2], 'query': 1},
    ('SS', -40): {'support': [0, 2], 'query': 1},
}

def get_peripheral_voltage_indices(corner, temp):
    """Get support and query voltage indices for peripheral points based on corner and temperature."""
    key = (corner.upper(), temp)
    if key in PERIPHERAL_VOLTAGE_CONFIG:
        config = PERIPHERAL_VOLTAGE_CONFIG[key]
        return config['support'], config['query']
    # Default fallback: support=[0,1], query=2
    return [0, 1], 2

# TSMC Process Parameters
PARAM_A = [1.427, 1.457, 1.430, 1.470, 1.443, 1.483, 1.43, 1.47, 1.43, 1.47]
PARAM_B = [0.026, 0.045, 0, 0, -0.026, -0.05, 0.0208, -0.04, 0.036, -0.0208]
PARAM_C = [0.024, 2.000, 0.024, 2.000, 0.024, 2.000, 0.024, 2.000, 0.024, 2.000]
CORNER_TO_IDX = {'FF': 0, 'TT': 1, 'SS': 2, 'FS': 3, 'SF': 4}

# Single-stage cell patterns (INV, NAND, NOR, AOI, OAI, MAOI, MOAI)
SINGLE_STAGE_PATTERNS = [
    r'^INV',      # Inverter
    r'^CKINV',    # Clock inverter
    r'^ND[0-9]',  # NAND (ND2, ND3, ND4, etc.)
    r'^CKND',     # Clock NAND
    r'^NR[0-9]',  # NOR (NR2, NR3, NR4, etc.)
    r'^CKNR',     # Clock NOR
    r'^AOI',      # AND-OR-Invert (AOI21, AOI22, AOI31, etc.)
    r'^OAI',      # OR-AND-Invert (OAI21, OAI22, OAI31, etc.)
    r'^MAOI',     # MUX-AND-OR-Invert
    r'^MOAI',     # MUX-OR-AND-Invert
]


def get_cell_stage(cell_name):
    """
    Determine if cell is single-stage (1) or multi-stage (2+).
    Single-stage: INV, NAND, NOR, AOI, OAI, MAOI, MOAI (single CMOS stage)
    Multi-stage: AND, OR, XOR, MUX, BUF, etc. (cascaded stages)
    """
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
    return {'a_n': PARAM_A[nmos_idx], 'a_p': PARAM_A[pmos_idx],
            'b_n': PARAM_B[nmos_idx], 'b_p': PARAM_B[pmos_idx],
            'c_n': PARAM_C[nmos_idx], 'c_p': PARAM_C[pmos_idx]}


def parse_lib_file(lib_path):
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
                        samples.append({'cell_name': cell_name, 'delay_type': delay_type,
                                       'related_pin': related_pin, 'index_1': index_1,
                                       'index_2': index_2, 'values': values,
                                       'stage': get_cell_stage(cell_name)})
    return samples


def create_mlp_input(samples, corner, voltage, temperature, data_type='cell', stage_filter=0):
    """
    Create MLP inputs organized by (cell, delay_type, related_pin) -> (slew_idx, load_idx) -> input/output
    Input format: [a, b, c, temperature, voltage, stage, delay_indicator, slew, load]
    - stage: 1 for single-stage (INV, NAND, NOR), 2 for multi-stage
    - delay_indicator: -1 for rise, 1 for fall
    """
    abc_params = get_abc_params(corner)
    data_by_table = defaultdict(dict)
    valid_delay_types = DELAY_TYPES[data_type]

    for sample in samples:
        if sample['delay_type'] not in valid_delay_types:
            continue

        # Apply stage filter
        stage = sample['stage']
        if stage_filter == 1 and stage != 1:  # single-stage only
            continue
        if stage_filter == 2 and stage != 2:  # multi-stage only
            continue

        delay_indicator = -1 if 'rise' in sample['delay_type'] else 1
        a = (abc_params['a_n'] + abc_params['a_p']) / 2
        b = abc_params['b_n'] + abc_params['b_p']
        c = abc_params['c_n'] + abc_params['c_p']

        table_key = (sample['cell_name'], sample['delay_type'], sample['related_pin'])

        for ri, slew in enumerate(sample['index_1']):
            for ci, load in enumerate(sample['index_2']):
                if ri < len(sample['values']) and ci < len(sample['values'][ri]):
                    sl_key = (ri, ci)
                    # Input: [a, b, c, temp, voltage, stage, delay_indicator, slew, load]
                    inp = torch.tensor([a, b, c, temperature, voltage, stage, delay_indicator, slew, load],
                                       dtype=torch.float32)
                    out = torch.tensor([sample['values'][ri][ci]], dtype=torch.float32)
                    data_by_table[table_key][sl_key] = {'input': inp, 'output': out, 'stage': stage}

    return data_by_table


# Parse lib files once (contains all delay types)
print("\n" + "=" * 80)
print("Parsing lib files...")
print("=" * 80)

parsed_samples = {}  # condition -> {'samples': [...], 'corner': ..., 'voltage': ..., 'temperature': ...}

for voltage_dir, dir_path in DATA_DIRS.items():
    if not os.path.exists(dir_path):
        print(f"  Directory not found: {dir_path}")
        continue
    for lib_file in sorted(Path(dir_path).glob('lib1_*.tlib')):
        match = re.search(r'lib1_(\w+)_base_400\.tlib', lib_file.name)
        if not match:
            continue
        condition = match.group(1)
        try:
            corner, voltage, temperature = parse_filename(lib_file.name)
        except ValueError as e:
            print(f"  Skipping {lib_file.name}: {e}")
            continue
        samples = parse_lib_file(str(lib_file))
        parsed_samples[condition] = {
            'samples': samples,
            'corner': corner,
            'voltage': voltage,
            'temperature': temperature
        }

        # Count stages
        stage_counts = defaultdict(int)
        for s in samples:
            stage_counts[s['stage']] += 1
        print(f"  Parsed {condition}: {len(samples)} timing tables (stage1: {stage_counts[1]}, stage2+: {stage_counts[2]})")

print(f"Parsed {len(parsed_samples)} conditions")


# Apply normalization function
def apply_norm_to_input(inp, stats):
    normalized = inp.clone()
    for idx, (mean, std) in stats.items():
        if std > 0:
            normalized[idx] = (normalized[idx] - mean) / std
    return normalized


def load_data_for_type(parsed_samples, data_type, unit_convert, stage_filter=0):
    """Load and preprocess data for a specific data type (cell or transition)"""
    all_data = {}

    for condition, info in parsed_samples.items():
        data_by_table = create_mlp_input(
            info['samples'], info['corner'], info['voltage'], info['temperature'],
            data_type, stage_filter
        )
        if data_by_table:
            all_data[condition] = {
                'data': data_by_table,
                'corner': info['corner'],
                'voltage': info['voltage'],
                'temperature': info['temperature']
            }

    # Unit conversion
    if unit_convert:
        for cond in all_data:
            for table_key in all_data[cond]['data']:
                for sl_key in all_data[cond]['data'][table_key]:
                    inp = all_data[cond]['data'][table_key][sl_key]['input']
                    inp[7] = inp[7] * 1000  # slew: ns → ps
                    inp[8] = inp[8] * 1000  # load: pf → ff

    return all_data


def load_norm_stats(data_type):
    """Load TSMC normalization stats for a specific data type"""
    tsmc_train = torch.load(TSMC_TRAIN_INPUT_PATHS[data_type], weights_only=False)
    norm_indices = [3, 4, 7, 8]  # temp, voltage, slew, load
    norm_stats = {}
    for idx in norm_indices:
        norm_stats[idx] = (tsmc_train[:, :, idx].mean().item(), tsmc_train[:, :, idx].std().item())
    del tsmc_train
    return norm_stats


def compute_local_norm_stats(all_data):
    """
    Compute normalization stats from local lib file data.
    Computes stats for temperature (idx 3), voltage (idx 4), slew (idx 7), and load (idx 8).
    """
    # Collect all values
    all_values = {3: [], 4: [], 7: [], 8: []}  # temp, voltage, slew, load

    for cond in all_data:
        for table_key in all_data[cond]['data']:
            for sl_key in all_data[cond]['data'][table_key]:
                inp = all_data[cond]['data'][table_key][sl_key]['input']
                all_values[3].append(inp[3].item())  # temperature
                all_values[4].append(inp[4].item())  # voltage
                all_values[7].append(inp[7].item())  # slew
                all_values[8].append(inp[8].item())  # load

    # Compute mean and std
    norm_stats = {}
    for idx in [3, 4, 7, 8]:
        values = torch.tensor(all_values[idx])
        norm_stats[idx] = (values.mean().item(), values.std().item())

    return norm_stats


def load_norm_stats_hybrid(data_type, all_data, use_local_norm, use_local_temp_norm=False, use_local_volt_norm=False):
    """
    Load normalization stats with hybrid approach:
    - Temperature (idx 3): from local if use_local_temp_norm=True, else TSMC
    - Voltage (idx 4): from local if use_local_volt_norm=True, else TSMC
    - Slew (idx 7) and Load (idx 8): from local lib files if use_local_norm=True
    """
    # Load TSMC stats
    tsmc_train = torch.load(TSMC_TRAIN_INPUT_PATHS[data_type], weights_only=False)
    norm_stats = {}

    # Check if any local stats needed
    need_local = use_local_norm or use_local_temp_norm or use_local_volt_norm
    local_stats = compute_local_norm_stats(all_data) if need_local else None

    # Temperature: local or TSMC
    if use_local_temp_norm:
        norm_stats[3] = local_stats[3]
    else:
        norm_stats[3] = (tsmc_train[:, :, 3].mean().item(), tsmc_train[:, :, 3].std().item())

    # Voltage: local or TSMC
    if use_local_volt_norm:
        norm_stats[4] = local_stats[4]
    else:
        norm_stats[4] = (tsmc_train[:, :, 4].mean().item(), tsmc_train[:, :, 4].std().item())

    # Slew and load: local or TSMC
    if use_local_norm:
        norm_stats[7] = local_stats[7]
        norm_stats[8] = local_stats[8]
    else:
        for idx in [7, 8]:
            norm_stats[idx] = (tsmc_train[:, :, idx].mean().item(), tsmc_train[:, :, idx].std().item())

    del tsmc_train
    return norm_stats


def apply_normalization(all_data, norm_stats):
    """Apply normalization to all data"""
    for cond in all_data:
        for table_key in all_data[cond]['data']:
            for sl_key in all_data[cond]['data'][table_key]:
                inp = all_data[cond]['data'][table_key][sl_key]['input']
                all_data[cond]['data'][table_key][sl_key]['input'] = apply_norm_to_input(inp, norm_stats)
    return all_data


def find_valid_tables(all_data):
    """Find valid tables that exist in all voltage conditions"""
    valid_tables_by_group = {}
    for group_key, conditions in VOLTAGE_SWEEP_GROUPS.items():
        common_tables = None
        for cond in conditions:
            if cond not in all_data:
                continue
            table_keys = set(all_data[cond]['data'].keys())
            if common_tables is None:
                common_tables = table_keys
            else:
                common_tables &= table_keys

        if common_tables:
            valid = []
            for table_key in common_tables:
                center_key = (CENTER_SLEW_IDX, CENTER_LOAD_IDX)
                has_center = all(center_key in all_data[cond]['data'][table_key] for cond in conditions)
                if has_center:
                    valid.append(table_key)
            valid_tables_by_group[group_key] = valid
    return valid_tables_by_group


def load_model(data_type):
    """Load MAML model for a specific data type"""
    maml_model = OptimizedMAML(
        model=MAMLModel_3hidden(in_features=9, layer_length=40),
        dataset_in=None, dataset_out=None, inner_lr=0.001, meta_lr=0.0001
    )
    model_path = get_model_path(data_type)
    print(f"\n📦 Loading {data_type} model:")
    print(f"   {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    maml_model.model.load_state_dict(state_dict)
    maml_model.model.to(device)
    maml_model.model.eval()
    return maml_model


def create_adapted_model(initial_model):
    """Create a copy of the model for adaptation"""
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(9, 40)), ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)), ('relu3', nn.ReLU()),
        ('l4', nn.Linear(40, 40)), ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
    ])).to(device)
    model.load_state_dict(initial_model.state_dict())
    return model


def adapt_with_center_point(initial_model, X_center, y_center, corner='TT', ref_mode='corner',
                            num_steps=40, lr=3e-3, loss_threshold=1e-4):
    """
    Adapt model using center point with 3 voltage conditions (3-shot).
    Uses grad/move for training target transformation.
    Only applies Adam if initial loss > threshold (like maml_functions.py).

    Args:
        ref_mode: Reference voltage selection mode
            - 'corner': use 0.9V equivalent per corner (TT:idx1, SS:idx2, FF:idx0)
            - 'middle': always use middle index (idx1)

    Training target: y_target = (y - y_mean) / (y_std * grad) + move
    Inverse (for prediction): pred = (model_output - move) * y_std * grad + y_mean

    Returns: adapted model, y_mean, y_std, grad, move, initial_loss, final_loss, adam_used, ref_idx
    """
    model = create_adapted_model(initial_model)
    criterion = nn.MSELoss()

    # Get initial predictions
    model.eval()
    with torch.no_grad():
        pred_support = model(X_center).flatten()

    y_support_flat = y_center.flatten()
    y_mean = y_support_flat.mean()
    y_std = y_support_flat.std()
    if y_std < 1e-8:
        y_std = torch.tensor(1.0, device=device)

    # Normalize y for grad calculation
    y_norm = (y_support_flat - y_mean) / y_std

    pred_min = pred_support.min().item()
    pred_max = pred_support.max().item()
    y_min = y_norm.min().item()
    y_max = y_norm.max().item()

    # Calculate grad (scaling factor)
    if abs(pred_max - pred_min) > 1e-8:
        grad = (y_max - y_min) / (pred_max - pred_min)
    else:
        grad = 1.0

    # Get reference voltage index based on ref_mode
    if ref_mode == 'middle':
        # Always use middle index (idx 1)
        ref_idx = 1
    else:  # 'corner' mode
        # TT: 0.9V (idx 1), SS: 0.9V (idx 2), FF: 0.88V (idx 0)
        ref_idx = VOLTAGE_REF_IDX.get(corner.upper(), 1)

    # Calculate move (offset) - using corner-specific reference voltage
    ref_pred = pred_support[ref_idx].item()
    if abs(grad) > 1e-8:
        move = ref_pred - y_norm[ref_idx].item() / grad
    else:
        move = 0.0

    # Create training target using grad/move transformation
    y_target = (y_center - y_mean) / (y_std * grad) + move

    # Initial loss (before adaptation)
    with torch.no_grad():
        initial_pred = model(X_center)
        initial_loss = criterion(initial_pred, y_target).item()

    # Only apply Adam if initial loss > threshold (like maml_functions.py)
    adam_used = False
    if initial_loss > loss_threshold:
        adam_used = True
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        model.train()
        for _ in range(num_steps):
            model.zero_grad()
            pred = model(X_center)
            loss = criterion(pred, y_target)
            loss.backward()
            optimizer.step()

    # Final loss (after adaptation or no change)
    model.eval()
    with torch.no_grad():
        final_pred = model(X_center)
        final_loss = criterion(final_pred, y_target).item()

    return model, y_mean, y_std, grad, move, initial_loss, final_loss, adam_used, ref_idx


def predict_with_selective_adam(adapted_model, X_support, y_support, X_query,
                                 center_mape, mape_threshold, adam_steps, adam_lr,
                                 peripheral_mode='grad_adam'):
    """
    Predict using adapted model with selective Adam based on center_mape.

    peripheral_mode options:
    - 'grad_adam': (default) center_mape > threshold → grad + Adam, else → grad + move
    - 'adam_only': Always use Adam without grad/move transformation (direct fitting)
    - 'grad_move': Always use grad + move only (no Adam)

    Returns: prediction, center_mape, used_adam
    """
    # Copy the adapted model (don't modify the center-adapted model)
    model = create_adapted_model(adapted_model)
    y_support_flat = y_support.flatten()
    criterion = nn.MSELoss()

    # Get model predictions
    model.eval()
    with torch.no_grad():
        pred_support = model(X_support).flatten()

    used_adam = False

    if peripheral_mode == 'adam_only':
        # ======================================
        # Adam Only Mode: Direct fitting without grad/move
        # ======================================
        used_adam = True
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=1e-4)

        # Target is raw y values (no transformation)
        y_target = y_support_flat

        model.train()
        for _ in range(adam_steps):
            model.zero_grad()
            pred = model(X_support).flatten()
            loss = criterion(pred, y_target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_query = model(X_query).flatten()
            pred = pred_query  # No inverse transform needed

    else:
        # ======================================
        # Grad-based modes: compute grad/move
        # ======================================
        # Compute peripheral's OWN normalization stats
        y_mean = y_support_flat.mean()
        y_std = y_support_flat.std()
        if y_std < 1e-8:
            y_std = torch.tensor(1.0, device=device)

        # Normalize y for grad calculation
        y_norm = (y_support_flat - y_mean) / y_std

        pred_min = pred_support.min().item()
        pred_max = pred_support.max().item()
        y_min = y_norm.min().item()
        y_max = y_norm.max().item()

        # Calculate peripheral's OWN grad (scaling factor)
        if abs(pred_max - pred_min) > 1e-8:
            grad = (y_max - y_min) / (pred_max - pred_min)
        else:
            grad = 1.0

        if peripheral_mode == 'grad_move':
            # Always use grad + move (no Adam)
            use_adam_for_this = False
        else:
            # 'grad_adam' mode: decide based on center_mape
            use_adam_for_this = (center_mape > mape_threshold)

        if use_adam_for_this:
            # grad only + Adam
            y_target = (y_support_flat - y_mean) / (y_std * grad)

            used_adam = True
            optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=1e-4)

            model.train()
            for _ in range(adam_steps):
                model.zero_grad()
                pred = model(X_support).flatten()
                loss = criterion(pred, y_target)
                loss.backward()
                optimizer.step()

            model.eval()

            # Inverse with grad only (no move)
            with torch.no_grad():
                pred_query = model(X_query).flatten()
                pred = pred_query * y_std * grad + y_mean
        else:
            # grad + move only (no Adam)
            center = (pred_max + pred_min) / 2
            y_middle = (y_max + y_min) / 2
            if abs(grad) > 1e-8:
                move = center - y_middle / grad
            else:
                move = 0.0

            with torch.no_grad():
                pred_query = model(X_query).flatten()
                pred = (pred_query - move) * y_std * grad + y_mean

    return pred.cpu().numpy(), center_mape, used_adam


def compute_peripheral_gradmove(adapted_model, X_support, y_support, X_query):
    """
    Compute grad/move and support_loss WITHOUT running Adam.
    Returns all info needed to later run Adam if needed.

    Returns: pred_no_adam, support_loss, grad_move_info (for later Adam if needed)
    """
    y_support_flat = y_support.flatten()
    criterion = nn.MSELoss()

    # Get predictions from adapted model
    adapted_model.eval()
    with torch.no_grad():
        pred_support = adapted_model(X_support).flatten()

    # Compute peripheral's OWN normalization stats
    y_mean = y_support_flat.mean()
    y_std = y_support_flat.std()
    if y_std < 1e-8:
        y_std = torch.tensor(1.0, device=device)

    # Normalize y for grad calculation
    y_norm = (y_support_flat - y_mean) / y_std

    pred_min = pred_support.min().item()
    pred_max = pred_support.max().item()
    y_min = y_norm.min().item()
    y_max = y_norm.max().item()

    # Calculate grad
    if abs(pred_max - pred_min) > 1e-8:
        grad = (y_max - y_min) / (pred_max - pred_min)
    else:
        grad = 1.0

    # Calculate move
    center = (pred_max + pred_min) / 2
    y_middle = (y_max + y_min) / 2
    if abs(grad) > 1e-8:
        move = center - y_middle / grad
    else:
        move = 0.0

    # Create target
    y_target = (y_support_flat - y_mean) / (y_std * grad) + move

    # Calculate support loss
    with torch.no_grad():
        support_loss = criterion(pred_support, y_target).item()

    # Prediction WITHOUT Adam (grad/move only)
    with torch.no_grad():
        pred_query = adapted_model(X_query).flatten()
        pred_no_adam = (pred_query - move) * y_std * grad + y_mean

    # Store info needed for later Adam
    grad_move_info = {
        'y_mean': y_mean,
        'y_std': y_std,
        'grad': grad,
        'move': move,
        'y_target': y_target
    }

    return pred_no_adam.cpu().item(), support_loss, grad_move_info


def run_adam_for_point(adapted_model, X_support, X_query, grad_move_info, adam_steps, adam_lr):
    """
    Run Adam adaptation for a single peripheral point using pre-computed grad/move info.
    Returns: pred_with_adam
    """
    criterion = nn.MSELoss()

    # Create model copy
    model = create_adapted_model(adapted_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=1e-4)

    y_target = grad_move_info['y_target']
    y_mean = grad_move_info['y_mean']
    y_std = grad_move_info['y_std']
    grad = grad_move_info['grad']
    move = grad_move_info['move']

    model.train()
    for _ in range(adam_steps):
        model.zero_grad()
        pred = model(X_support).flatten()
        loss = criterion(pred, y_target)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_query = model(X_query).flatten()
        pred_with_adam = (pred_query - move) * y_std * grad + y_mean

    return pred_with_adam.cpu().item()


def run_validation_with_threshold(data_type, all_data, valid_tables_by_group, maml_model, threshold):
    """
    Run full validation flow with a specific threshold.
    Returns summary metrics for this threshold.
    """
    center_mapes = []
    peripheral_mapes = []
    adam_used_count = 0
    peripheral_count = 0

    for group_key, conditions in VOLTAGE_SWEEP_GROUPS.items():
        corner, temp = group_key
        group_name = f"{corner}_{temp}C"

        # Apply corner and temperature filters
        if corner_filter and corner.upper() not in corner_filter:
            continue
        if temp_filter and temp not in temp_filter:
            continue

        if group_key not in valid_tables_by_group:
            continue

        valid_tables = valid_tables_by_group[group_key]
        if not valid_tables:
            continue

        # Sample tables (use same sampling for all thresholds)
        if args.num_tables == -1:
            num_tables = len(valid_tables)
            sample_tables = list(valid_tables)
        else:
            num_tables = min(args.num_tables, len(valid_tables))
            # Use fixed seed based on group for reproducibility across thresholds
            rng = np.random.RandomState(hash(group_name) % (2**32))
            sample_tables = list(rng.choice(list(valid_tables), num_tables, replace=False))

        for table_key in sample_tables:
            center_key = (CENTER_SLEW_IDX, CENTER_LOAD_IDX)

            # Get center point data
            X_center_list = []
            y_center_list = []
            for cond in conditions:
                data = all_data[cond]['data'][table_key][center_key]
                X_center_list.append(data['input'])
                y_center_list.append(data['output'])

            X_center = torch.stack(X_center_list).to(device)
            y_center = torch.stack(y_center_list).to(device)

            # Adapt model using center point
            adapted_model, y_mean_center, y_std_center, grad_center, move_center, _, _, _, ref_idx = adapt_with_center_point(
                maml_model.model.model, X_center, y_center, corner=corner,
                num_steps=args.center_steps, lr=args.center_lr, loss_threshold=args.center_loss_threshold
            )

            # Center point prediction (using corner-specific reference voltage index)
            with torch.no_grad():
                center_output = adapted_model(X_center[ref_idx:ref_idx+1]).item()
                center_pred = (center_output - move_center) * y_std_center.item() * grad_center + y_mean_center.item()
            center_actual = y_center[ref_idx].item()
            center_mape = abs(center_pred - center_actual) / (abs(center_actual) + 1e-8) * 100
            center_mapes.append(center_mape)

            # Peripheral points
            for sl_key in all_data[conditions[0]]['data'][table_key]:
                if sl_key == center_key:
                    continue
                if not all(sl_key in all_data[cond]['data'][table_key] for cond in conditions):
                    continue

                # Get support/query voltage indices based on corner and temperature
                support_indices, query_idx = get_peripheral_voltage_indices(corner, temp)

                X_support_list = []
                y_support_list = []
                for cond_idx in support_indices:
                    cond = conditions[cond_idx]
                    data = all_data[cond]['data'][table_key][sl_key]
                    X_support_list.append(data['input'])
                    y_support_list.append(data['output'])

                X_support = torch.stack(X_support_list).to(device)
                y_support = torch.stack(y_support_list).to(device)

                query_cond = conditions[query_idx]
                query_data = all_data[query_cond]['data'][table_key][sl_key]
                X_query = query_data['input'].unsqueeze(0).to(device)
                y_query = query_data['output'].item()

                # Run prediction based on center_mape threshold
                pred, _, used_adam = predict_with_selective_adam(
                    adapted_model, X_support, y_support, X_query,
                    center_mape=center_mape,
                    mape_threshold=threshold,
                    adam_steps=args.peripheral_adam_steps,
                    adam_lr=args.peripheral_adam_lr,
                    peripheral_mode=args.peripheral_mode
                )
                pred_value = pred[0]

                mape = abs(pred_value - y_query) / (abs(y_query) + 1e-8) * 100
                peripheral_mapes.append(mape)
                peripheral_count += 1
                if used_adam:
                    adam_used_count += 1

    return {
        'mape_threshold': threshold,
        'center_mape': np.mean(center_mapes) if center_mapes else 0,
        'peripheral_mape': np.mean(peripheral_mapes) if peripheral_mapes else 0,
        'all_mape': np.mean(center_mapes + peripheral_mapes) if (center_mapes + peripheral_mapes) else 0,
        'adam_rate': adam_used_count / peripheral_count * 100 if peripheral_count > 0 else 0,
        'n_center': len(center_mapes),
        'n_peripheral': peripheral_count,
        'n_adam_used': adam_used_count
    }


def run_threshold_sweep(data_type, all_data, valid_tables_by_group, maml_model):
    """
    Run full validation for each threshold independently.
    Each threshold gets its own complete validation run.
    """
    print(f"\nRunning threshold sweep with {len(THRESHOLD_GRID)} thresholds...")
    print(f"Thresholds: {THRESHOLD_GRID}")

    all_results = []

    for i, threshold in enumerate(THRESHOLD_GRID):
        print(f"\n[{i+1}/{len(THRESHOLD_GRID)}] Running validation with threshold={threshold}...")
        result = run_validation_with_threshold(
            data_type, all_data, valid_tables_by_group, maml_model, threshold
        )
        all_results.append(result)
        print(f"  → All MAPE: {result['all_mape']:.2f}%, Adam Rate: {result['adam_rate']:.1f}%")

    return all_results


def run_validation(data_type, all_data, valid_tables_by_group, maml_model, ref_mode='corner'):
    """Run validation for a specific data type

    Args:
        ref_mode: 'corner' (0.9V per corner) or 'middle' (always idx1)
    """
    results = defaultdict(lambda: {
        'center_mape': [], 'peripheral_mape': [], 'all_mape': [],
        'center_mae': [], 'peripheral_mae': [], 'all_mae': [],
        'center_initial_loss': [], 'center_final_loss': [],
        'center_used_adam': [],
        'peripheral_center_mape': [],
        'peripheral_used_adam': [],
        'stage1_mape': [], 'stage2_mape': [],
        'stage1_mae': [], 'stage2_mae': []
    })

    for group_key, conditions in VOLTAGE_SWEEP_GROUPS.items():
        corner, temp = group_key
        group_name = f"{corner}_{temp}C"

        # Apply corner and temperature filters
        if corner_filter and corner.upper() not in corner_filter:
            continue
        if temp_filter and temp not in temp_filter:
            continue

        if group_key not in valid_tables_by_group:
            continue

        valid_tables = valid_tables_by_group[group_key]

        # Sample tables
        if args.num_tables == -1:
            num_tables = len(valid_tables)
            sample_tables = valid_tables
        else:
            num_tables = min(args.num_tables, len(valid_tables))
            sample_indices = np.random.choice(len(valid_tables), num_tables, replace=False)
            sample_tables = [valid_tables[i] for i in sample_indices]

        print(f"\n{group_name}: Processing {num_tables} tables...")

        for table_idx, table_key in enumerate(sample_tables):
            if table_idx % 50 == 0:
                print(f"  Table {table_idx + 1}/{num_tables}", flush=True)

            # Get stage from first condition
            center_key = (CENTER_SLEW_IDX, CENTER_LOAD_IDX)
            stage = all_data[conditions[0]]['data'][table_key][center_key].get('stage', 2)

            # Step 1: Get center point data (3 voltage conditions)
            X_center_list = []
            y_center_list = []

            for cond in conditions:
                data = all_data[cond]['data'][table_key][center_key]
                X_center_list.append(data['input'])
                y_center_list.append(data['output'])

            X_center = torch.stack(X_center_list).to(device)  # [3, 9]
            y_center = torch.stack(y_center_list).to(device)  # [3, 1]

            # Step 2: Adapt model using center point (3-shot) with grad/move
            # Only applies Adam if initial loss > threshold
            # ref_mode: 'corner' = 0.9V per corner, 'middle' = always idx1
            adapted_model, y_mean_center, y_std_center, grad_center, move_center, initial_loss, final_loss, center_adam_used, ref_idx = adapt_with_center_point(
                maml_model.model.model, X_center, y_center, corner=corner, ref_mode=ref_mode,
                num_steps=args.center_steps, lr=args.center_lr, loss_threshold=args.center_loss_threshold
            )

            # Evaluate center point prediction (using corner-specific reference voltage)
            # Inverse transformation: pred = (model_output - move) * y_std * grad + y_mean
            with torch.no_grad():
                center_output = adapted_model(X_center[ref_idx:ref_idx+1]).item()
                center_pred = (center_output - move_center) * y_std_center.item() * grad_center + y_mean_center.item()
            center_actual = y_center[ref_idx].item()
            center_mape = abs(center_pred - center_actual) / (abs(center_actual) + 1e-8) * 100
            center_mae = abs(center_pred - center_actual)
            results[group_name]['center_mape'].append(center_mape)
            results[group_name]['all_mape'].append(center_mape)
            results[group_name]['center_mae'].append(center_mae)
            results[group_name]['all_mae'].append(center_mae)
            results[group_name]['center_initial_loss'].append(initial_loss)
            results[group_name]['center_final_loss'].append(final_loss)
            results[group_name]['center_used_adam'].append(1 if center_adam_used else 0)

            # Track by stage
            if stage == 1:
                results[group_name]['stage1_mape'].append(center_mape)
                results[group_name]['stage1_mae'].append(center_mae)
            else:
                results[group_name]['stage2_mape'].append(center_mape)
                results[group_name]['stage2_mae'].append(center_mae)

            # Step 3: For each peripheral point, use selective Adam
            for sl_key in all_data[conditions[0]]['data'][table_key]:
                if sl_key == center_key:
                    continue

                # Check if this point exists in all conditions
                if not all(sl_key in all_data[cond]['data'][table_key] for cond in conditions):
                    continue

                # Get support/query voltage indices based on corner and temperature
                # FF 125, SS 125: support=[0,1], query=2
                # TT (all temps): support=[1,2], query=0
                # FF -40, SS -40: support=[0,2], query=1
                support_indices, query_idx = get_peripheral_voltage_indices(corner, temp)

                X_support_list = []
                y_support_list = []
                for cond_idx in support_indices:
                    cond = conditions[cond_idx]
                    data = all_data[cond]['data'][table_key][sl_key]
                    X_support_list.append(data['input'])
                    y_support_list.append(data['output'])

                X_support = torch.stack(X_support_list).to(device)  # [2, 9]
                y_support = torch.stack(y_support_list).to(device)  # [2, 1]

                # Query: based on corner/temp configuration
                query_cond = conditions[query_idx]
                query_data = all_data[query_cond]['data'][table_key][sl_key]
                X_query = query_data['input'].unsqueeze(0).to(device)  # [1, 9]
                y_query = query_data['output'].item()

                # Predict using selective Adam based on center_mape threshold
                pred, _, used_adam = predict_with_selective_adam(
                    adapted_model, X_support, y_support, X_query,
                    center_mape=center_mape,
                    mape_threshold=args.mape_threshold,
                    adam_steps=args.peripheral_adam_steps,
                    adam_lr=args.peripheral_adam_lr,
                    peripheral_mode=args.peripheral_mode
                )
                pred_value = pred[0]

                mape = abs(pred_value - y_query) / (abs(y_query) + 1e-8) * 100
                mae = abs(pred_value - y_query)
                results[group_name]['peripheral_mape'].append(mape)
                results[group_name]['all_mape'].append(mape)
                results[group_name]['peripheral_mae'].append(mae)
                results[group_name]['all_mae'].append(mae)
                results[group_name]['peripheral_center_mape'].append(center_mape)
                results[group_name]['peripheral_used_adam'].append(1 if used_adam else 0)

                # Track by stage
                if stage == 1:
                    results[group_name]['stage1_mape'].append(mape)
                    results[group_name]['stage1_mae'].append(mae)
                else:
                    results[group_name]['stage2_mape'].append(mape)
                    results[group_name]['stage2_mae'].append(mae)

    return results


def print_results(results, data_type):
    """Print results for a specific data type"""

    # First, print LOSS information
    print("\n" + "=" * 120)
    print(f"LOSS ANALYSIS - {data_type.upper()}")
    print("=" * 120)

    print(f"\n{'Group':<15} {'Center Init Loss':<18} {'Center Final Loss':<18} {'Loss Reduction':<15}")
    print("-" * 80)

    overall_center_initial_loss = []
    overall_center_final_loss = []

    for group_name in sorted(results.keys()):
        r = results[group_name]
        center_init = np.mean(r['center_initial_loss']) if r['center_initial_loss'] else 0
        center_final = np.mean(r['center_final_loss']) if r['center_final_loss'] else 0
        loss_reduction = (center_init - center_final) / (center_init + 1e-8) * 100

        overall_center_initial_loss.extend(r['center_initial_loss'])
        overall_center_final_loss.extend(r['center_final_loss'])

        print(f"{group_name:<15} {center_init:<18.6f} {center_final:<18.6f} {loss_reduction:<15.1f}%")

    print("-" * 80)
    overall_init = np.mean(overall_center_initial_loss) if overall_center_initial_loss else 0
    overall_final = np.mean(overall_center_final_loss) if overall_center_final_loss else 0
    overall_reduction = (overall_init - overall_final) / (overall_init + 1e-8) * 100
    print(f"{'OVERALL':<15} {overall_init:<18.6f} {overall_final:<18.6f} {overall_reduction:<15.1f}%")

    print(f"\n** Note: All losses are normalized MSE (after y normalization)")
    print(f"** Loss ~0: perfect prediction, Loss ~1: prediction error ≈ 1 std, Loss >1: large error")
    print(f"** Center Adam threshold: 1e-4 (only apply Adam if initial loss > threshold)")

    # Print CENTER Adam usage statistics
    print("\n" + "=" * 120)
    print(f"CENTER ADAM USAGE - {data_type.upper()} (threshold=1e-4)")
    print("=" * 120)

    print(f"\n{'Group':<15} {'Total Center':<18} {'Used Adam':<15} {'Adam Rate (%)':<15}")
    print("-" * 70)

    overall_center_used_adam = []
    for group_name in sorted(results.keys()):
        r = results[group_name]
        total = len(r['center_used_adam'])
        used_adam = sum(r['center_used_adam'])
        adam_rate = used_adam / total * 100 if total > 0 else 0
        overall_center_used_adam.extend(r['center_used_adam'])

        print(f"{group_name:<15} {total:<18} {used_adam:<15} {adam_rate:<15.1f}")

    print("-" * 70)
    total_all = len(overall_center_used_adam)
    used_all = sum(overall_center_used_adam)
    rate_all = used_all / total_all * 100 if total_all > 0 else 0
    print(f"{'OVERALL':<15} {total_all:<18} {used_all:<15} {rate_all:<15.1f}")

    # Print PERIPHERAL Adam usage statistics
    print("\n" + "=" * 120)
    print(f"PERIPHERAL ADAM USAGE - {data_type.upper()} (center_mape_threshold={args.mape_threshold}%)")
    print("=" * 120)

    print(f"\n{'Group':<15} {'Total Peripheral':<18} {'Used Adam':<15} {'Adam Rate (%)':<15}")
    print("-" * 70)

    overall_used_adam = []
    for group_name in sorted(results.keys()):
        r = results[group_name]
        total = len(r['peripheral_used_adam'])
        used_adam = sum(r['peripheral_used_adam'])
        adam_rate = used_adam / total * 100 if total > 0 else 0
        overall_used_adam.extend(r['peripheral_used_adam'])

        print(f"{group_name:<15} {total:<18} {used_adam:<15} {adam_rate:<15.1f}")

    print("-" * 70)
    total_all = len(overall_used_adam)
    used_all = sum(overall_used_adam)
    rate_all = used_all / total_all * 100 if total_all > 0 else 0
    print(f"{'OVERALL':<15} {total_all:<18} {used_all:<15} {rate_all:<15.1f}")

    # Print PERIPHERAL center MAPE distribution (used for Adam decision)
    print("\n" + "=" * 120)
    print(f"PERIPHERAL POINT CENTER MAPE DISTRIBUTION - {data_type.upper()}")
    print(f"(This center MAPE determines whether Adam is applied: if center_mape > {args.mape_threshold}% → Adam)")
    print("=" * 120)

    print(f"\n{'Group':<15} {'Mean MAPE%':<15} {'Median MAPE%':<15} {'Min MAPE%':<15} {'Max MAPE%':<15} {'P90 MAPE%':<15}")
    print("-" * 100)

    overall_center_mape_for_peripheral = []
    for group_name in sorted(results.keys()):
        r = results[group_name]
        mapes = r['peripheral_center_mape']
        if mapes:
            mean_mape = np.mean(mapes)
            median_mape = np.median(mapes)
            min_mape = np.min(mapes)
            max_mape = np.max(mapes)
            p90_mape = np.percentile(mapes, 90)
            overall_center_mape_for_peripheral.extend(mapes)
            print(f"{group_name:<15} {mean_mape:<15.2f} {median_mape:<15.2f} {min_mape:<15.2f} {max_mape:<15.2f} {p90_mape:<15.2f}")
        else:
            print(f"{group_name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

    print("-" * 100)
    if overall_center_mape_for_peripheral:
        print(f"{'OVERALL':<15} {np.mean(overall_center_mape_for_peripheral):<15.2f} {np.median(overall_center_mape_for_peripheral):<15.2f} "
              f"{np.min(overall_center_mape_for_peripheral):<15.2f} {np.max(overall_center_mape_for_peripheral):<15.2f} {np.percentile(overall_center_mape_for_peripheral, 90):<15.2f}")
        print(f"\nPercentiles: P10={np.percentile(overall_center_mape_for_peripheral, 10):.2f}%, P25={np.percentile(overall_center_mape_for_peripheral, 25):.2f}%, "
              f"P50={np.percentile(overall_center_mape_for_peripheral, 50):.2f}%, P75={np.percentile(overall_center_mape_for_peripheral, 75):.2f}%, P95={np.percentile(overall_center_mape_for_peripheral, 95):.2f}%")

    # Print MAPE results
    print("\n" + "=" * 120)
    print(f"RESULTS - {data_type.upper()} - MAPE (%)")
    print("=" * 120)

    print(f"\n{'Group':<15} {'Center MAPE%':<15} {'Peripheral MAPE%':<18} {'All MAPE%':<15} {'N_center':<10} {'N_peripheral':<12}")
    print("-" * 90)

    overall_center_mape = []
    overall_peripheral_mape = []
    overall_all_mape = []
    overall_center_mae = []
    overall_peripheral_mae = []
    overall_all_mae = []

    for group_name in sorted(results.keys()):
        r = results[group_name]
        center_mape = np.mean(r['center_mape']) if r['center_mape'] else 0
        peripheral_mape = np.mean(r['peripheral_mape']) if r['peripheral_mape'] else 0
        all_mape = np.mean(r['all_mape']) if r['all_mape'] else 0

        overall_center_mape.extend(r['center_mape'])
        overall_peripheral_mape.extend(r['peripheral_mape'])
        overall_all_mape.extend(r['all_mape'])
        overall_center_mae.extend(r['center_mae'])
        overall_peripheral_mae.extend(r['peripheral_mae'])
        overall_all_mae.extend(r['all_mae'])

        print(f"{group_name:<15} {center_mape:<15.2f} {peripheral_mape:<18.2f} {all_mape:<15.2f} "
              f"{len(r['center_mape']):<10} {len(r['peripheral_mape']):<12}")

    print("-" * 90)
    print(f"{'OVERALL':<15} {np.mean(overall_center_mape) if overall_center_mape else 0:<15.2f} {np.mean(overall_peripheral_mape) if overall_peripheral_mape else 0:<18.2f} "
          f"{np.mean(overall_all_mape) if overall_all_mape else 0:<15.2f} {len(overall_center_mape):<10} {len(overall_peripheral_mape):<12}")

    # Print MAE results
    print("\n" + "=" * 120)
    print(f"RESULTS - {data_type.upper()} - MAE (ns)")
    print("=" * 120)

    print(f"\n{'Group':<15} {'Center MAE':<15} {'Peripheral MAE':<18} {'All MAE':<15}")
    print("-" * 70)

    for group_name in sorted(results.keys()):
        r = results[group_name]
        center_mae = np.mean(r['center_mae']) if r['center_mae'] else 0
        peripheral_mae = np.mean(r['peripheral_mae']) if r['peripheral_mae'] else 0
        all_mae = np.mean(r['all_mae']) if r['all_mae'] else 0

        print(f"{group_name:<15} {center_mae:<15.6f} {peripheral_mae:<18.6f} {all_mae:<15.6f}")

    print("-" * 70)
    print(f"{'OVERALL':<15} {np.mean(overall_center_mae) if overall_center_mae else 0:<15.6f} {np.mean(overall_peripheral_mae) if overall_peripheral_mae else 0:<18.6f} "
          f"{np.mean(overall_all_mae) if overall_all_mae else 0:<15.6f}")

    # Print by Stage
    print("\n" + "=" * 120)
    print(f"RESULTS BY STAGE - {data_type.upper()}")
    print("=" * 120)

    print(f"\n{'Group':<15} {'Stage1 MAPE%':<15} {'Stage2 MAPE%':<15} {'Stage1 MAE':<15} {'Stage2 MAE':<15} {'N_stage1':<10} {'N_stage2':<10}")
    print("-" * 100)

    overall_stage1_mape = []
    overall_stage2_mape = []
    overall_stage1_mae = []
    overall_stage2_mae = []

    for group_name in sorted(results.keys()):
        r = results[group_name]
        s1_mape = np.mean(r['stage1_mape']) if r['stage1_mape'] else 0
        s2_mape = np.mean(r['stage2_mape']) if r['stage2_mape'] else 0
        s1_mae = np.mean(r['stage1_mae']) if r['stage1_mae'] else 0
        s2_mae = np.mean(r['stage2_mae']) if r['stage2_mae'] else 0

        overall_stage1_mape.extend(r['stage1_mape'])
        overall_stage2_mape.extend(r['stage2_mape'])
        overall_stage1_mae.extend(r['stage1_mae'])
        overall_stage2_mae.extend(r['stage2_mae'])

        print(f"{group_name:<15} {s1_mape:<15.2f} {s2_mape:<15.2f} {s1_mae:<15.6f} {s2_mae:<15.6f} "
              f"{len(r['stage1_mape']):<10} {len(r['stage2_mape']):<10}")

    print("-" * 100)
    print(f"{'OVERALL':<15} {np.mean(overall_stage1_mape) if overall_stage1_mape else 0:<15.2f} "
          f"{np.mean(overall_stage2_mape) if overall_stage2_mape else 0:<15.2f} "
          f"{np.mean(overall_stage1_mae) if overall_stage1_mae else 0:<15.6f} "
          f"{np.mean(overall_stage2_mae) if overall_stage2_mae else 0:<15.6f} "
          f"{len(overall_stage1_mape):<10} {len(overall_stage2_mape):<10}")

    print("\n" + "=" * 100)
    print(f"SUMMARY - {data_type.upper()}")
    print("=" * 100)
    print(f"\nCenter Point (3-shot adaptation):")
    print(f"  Mean MAPE: {np.mean(overall_center_mape) if overall_center_mape else 0:.2f}%")
    print(f"  Median MAPE: {np.median(overall_center_mape) if overall_center_mape else 0:.2f}%")
    print(f"  Mean MAE: {np.mean(overall_center_mae) if overall_center_mae else 0:.6f} ns")
    print(f"  Median MAE: {np.median(overall_center_mae) if overall_center_mae else 0:.6f} ns")

    print(f"\nPeripheral Points (selective Adam):")
    print(f"  Mean MAPE: {np.mean(overall_peripheral_mape) if overall_peripheral_mape else 0:.2f}%")
    print(f"  Median MAPE: {np.median(overall_peripheral_mape) if overall_peripheral_mape else 0:.2f}%")
    print(f"  Mean MAE: {np.mean(overall_peripheral_mae) if overall_peripheral_mae else 0:.6f} ns")
    print(f"  Median MAE: {np.median(overall_peripheral_mae) if overall_peripheral_mae else 0:.6f} ns")

    print(f"\nAll Points:")
    print(f"  Mean MAPE: {np.mean(overall_all_mape) if overall_all_mape else 0:.2f}%")
    print(f"  Median MAPE: {np.median(overall_all_mape) if overall_all_mape else 0:.2f}%")
    print(f"  Mean MAE: {np.mean(overall_all_mae) if overall_all_mae else 0:.6f} ns")
    print(f"  Median MAE: {np.median(overall_all_mae) if overall_all_mae else 0:.6f} ns")

    print(f"\nBy Stage:")
    print(f"  Stage 1 (INV/NAND/NOR/AOI/OAI/MAOI/MOAI): Mean MAPE={np.mean(overall_stage1_mape) if overall_stage1_mape else 0:.2f}%, MAE={np.mean(overall_stage1_mae) if overall_stage1_mae else 0:.6f} ns ({len(overall_stage1_mape)} samples)")
    print(f"  Stage 2+ (Multi-stage): Mean MAPE={np.mean(overall_stage2_mape) if overall_stage2_mape else 0:.2f}%, MAE={np.mean(overall_stage2_mae) if overall_stage2_mae else 0:.6f} ns ({len(overall_stage2_mape)} samples)")


# ============================================================
# Main: Run validation for both CELL and TRANSITION
# ============================================================

def load_norm_stats_from_file(norm_stats_path):
    """Load normalization stats from a file and convert to validation format."""
    loaded_stats = torch.load(norm_stats_path, weights_only=False)
    norm_stats = {}
    for idx in [3, 4, 7, 8]:
        if isinstance(loaded_stats[idx], dict):
            norm_stats[idx] = (loaded_stats[idx]['mean'], loaded_stats[idx]['std'])
        else:
            norm_stats[idx] = loaded_stats[idx]
    return norm_stats


def run_validation_per_condition(data_type, parsed_samples, model_base_dir):
    """
    Run validation with per-condition model and norm_stats loading.
    Each (corner, temp) group uses its own trained model and normalization stats.
    """
    import copy

    all_results = defaultdict(lambda: {
        'center_mape': [], 'peripheral_mape': [], 'all_mape': [],
        'center_mae': [], 'peripheral_mae': [], 'all_mae': [],
        'center_initial_loss': [], 'center_final_loss': [],
        'center_used_adam': [],
        'peripheral_center_mape': [],
        'peripheral_used_adam': [],
        'stage1_mape': [], 'stage2_mape': [],
        'stage1_mae': [], 'stage2_mae': []
    })

    for group_key, conditions in VOLTAGE_SWEEP_GROUPS.items():
        corner, temp = group_key
        group_name = f"{corner}_{temp}C"

        # Apply corner and temperature filters
        if corner_filter and corner.upper() not in corner_filter:
            print(f"\nSkipping {group_name}: corner not in filter {corner_filter}")
            continue
        if temp_filter and temp not in temp_filter:
            print(f"\nSkipping {group_name}: temperature not in filter {temp_filter}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {group_name} with condition-specific model...")
        print(f"{'='*80}")

        # Get condition-specific paths
        condition_paths = get_condition_paths(data_type, corner, temp, model_base_dir, args.iteration)
        if condition_paths is None:
            print(f"  Skipping {group_name}: No condition-specific model/norm_stats found")
            continue

        print(f"  Model: {condition_paths['model_path']}")
        print(f"  Norm stats: {condition_paths['norm_stats_path']}")

        # Load raw data (without normalization)
        raw_data = load_data_for_type(parsed_samples, data_type, args.unit_convert, args.stage_filter)
        if not raw_data:
            print(f"  No {data_type} data found. Skipping...")
            continue

        # Load condition-specific norm stats
        norm_stats = load_norm_stats_from_file(condition_paths['norm_stats_path'])
        for idx, (mean, std) in norm_stats.items():
            idx_name = {3: 'temp', 4: 'voltage', 7: 'slew', 8: 'load'}[idx]
            print(f"    Index {idx} ({idx_name}): mean={mean:.6f}, std={std:.6f}")

        # Apply normalization
        all_data = apply_normalization(raw_data, norm_stats)

        # Find valid tables for this group
        valid_tables_by_group = find_valid_tables(all_data)
        if group_key not in valid_tables_by_group:
            print(f"  No valid tables found for {group_name}")
            continue

        valid_tables = valid_tables_by_group[group_key]
        print(f"  Found {len(valid_tables)} valid tables")

        # Load condition-specific model
        maml_model = OptimizedMAML(
            model=MAMLModel_3hidden(in_features=9, layer_length=40),
            dataset_in=None, dataset_out=None, inner_lr=0.001, meta_lr=0.0001
        )
        state_dict = torch.load(condition_paths['model_path'], map_location=device, weights_only=False)
        maml_model.model.load_state_dict(state_dict)
        maml_model.model.to(device)
        maml_model.model.eval()

        # Sample tables
        if args.num_tables == -1:
            num_tables = len(valid_tables)
            sample_tables = valid_tables
        else:
            num_tables = min(args.num_tables, len(valid_tables))
            sample_indices = np.random.choice(len(valid_tables), num_tables, replace=False)
            sample_tables = [valid_tables[i] for i in sample_indices]

        print(f"  Processing {num_tables} tables...")

        # Run validation for this group
        for table_idx, table_key in enumerate(sample_tables):
            if table_idx % 50 == 0:
                print(f"    Table {table_idx + 1}/{num_tables}", flush=True)

            center_key = (CENTER_SLEW_IDX, CENTER_LOAD_IDX)
            stage = all_data[conditions[0]]['data'][table_key][center_key].get('stage', 2)

            # Get center point data (3 voltage conditions)
            X_center_list = []
            y_center_list = []
            for cond in conditions:
                data = all_data[cond]['data'][table_key][center_key]
                X_center_list.append(data['input'])
                y_center_list.append(data['output'])

            X_center = torch.stack(X_center_list).to(device)
            y_center = torch.stack(y_center_list).to(device)

            # Adapt model using center point
            adapted_model, y_mean_center, y_std_center, grad_center, move_center, initial_loss, final_loss, center_adam_used, ref_idx = adapt_with_center_point(
                maml_model.model.model, X_center, y_center, corner=corner, ref_mode=args.ref_mode,
                num_steps=args.center_steps, lr=args.center_lr, loss_threshold=args.center_loss_threshold
            )

            # Evaluate center point prediction
            with torch.no_grad():
                center_output = adapted_model(X_center[ref_idx:ref_idx+1]).item()
                center_pred = (center_output - move_center) * y_std_center.item() * grad_center + y_mean_center.item()
            center_actual = y_center[ref_idx].item()
            center_mape = abs(center_pred - center_actual) / (abs(center_actual) + 1e-8) * 100
            center_mae = abs(center_pred - center_actual)

            all_results[group_name]['center_mape'].append(center_mape)
            all_results[group_name]['all_mape'].append(center_mape)
            all_results[group_name]['center_mae'].append(center_mae)
            all_results[group_name]['all_mae'].append(center_mae)
            all_results[group_name]['center_initial_loss'].append(initial_loss)
            all_results[group_name]['center_final_loss'].append(final_loss)
            all_results[group_name]['center_used_adam'].append(1 if center_adam_used else 0)

            if stage == 1:
                all_results[group_name]['stage1_mape'].append(center_mape)
                all_results[group_name]['stage1_mae'].append(center_mae)
            else:
                all_results[group_name]['stage2_mape'].append(center_mape)
                all_results[group_name]['stage2_mae'].append(center_mae)

            # Peripheral points
            for sl_key in all_data[conditions[0]]['data'][table_key]:
                if sl_key == center_key:
                    continue
                if not all(sl_key in all_data[cond]['data'][table_key] for cond in conditions):
                    continue

                support_indices, query_idx = get_peripheral_voltage_indices(corner, temp)

                X_support_list = []
                y_support_list = []
                for cond_idx in support_indices:
                    cond = conditions[cond_idx]
                    data = all_data[cond]['data'][table_key][sl_key]
                    X_support_list.append(data['input'])
                    y_support_list.append(data['output'])

                X_support = torch.stack(X_support_list).to(device)
                y_support = torch.stack(y_support_list).to(device)

                query_cond = conditions[query_idx]
                query_data = all_data[query_cond]['data'][table_key][sl_key]
                X_query = query_data['input'].unsqueeze(0).to(device)
                y_query = query_data['output'].item()

                pred, _, used_adam = predict_with_selective_adam(
                    adapted_model, X_support, y_support, X_query,
                    center_mape=center_mape,
                    mape_threshold=args.mape_threshold,
                    adam_steps=args.peripheral_adam_steps,
                    adam_lr=args.peripheral_adam_lr,
                    peripheral_mode=args.peripheral_mode
                )
                pred_value = pred[0]

                mape = abs(pred_value - y_query) / (abs(y_query) + 1e-8) * 100
                mae = abs(pred_value - y_query)

                all_results[group_name]['peripheral_mape'].append(mape)
                all_results[group_name]['all_mape'].append(mape)
                all_results[group_name]['peripheral_mae'].append(mae)
                all_results[group_name]['all_mae'].append(mae)
                all_results[group_name]['peripheral_center_mape'].append(center_mape)
                all_results[group_name]['peripheral_used_adam'].append(1 if used_adam else 0)

                if stage == 1:
                    all_results[group_name]['stage1_mape'].append(mape)
                    all_results[group_name]['stage1_mae'].append(mae)
                else:
                    all_results[group_name]['stage2_mape'].append(mape)
                    all_results[group_name]['stage2_mae'].append(mae)

        # Print intermediate results for this group
        group_mapes = all_results[group_name]['all_mape']
        if group_mapes:
            print(f"  {group_name} Mean MAPE: {np.mean(group_mapes):.2f}%")

    return dict(all_results)


for data_type in ['cell', 'transition']:
    print("\n" + "#" * 120)
    print(f"# {data_type.upper()} DELAY VALIDATION")
    print("#" * 120)

    # Check if using per-condition model loading
    if args.model_base_dir:
        print(f"\n*** PER-CONDITION MODE: Loading models from {args.model_base_dir} ***")

        # Run validation with per-condition loading
        results = run_validation_per_condition(data_type, parsed_samples, args.model_base_dir)

        if results:
            print_results(results, data_type)
        else:
            print(f"  No results generated for {data_type}")

        continue

    # Original flow: global model and norm_stats
    # Load data for this type
    print(f"\nLoading {data_type} data...")
    all_data = load_data_for_type(parsed_samples, data_type, args.unit_convert, args.stage_filter)
    print(f"  Loaded {len(all_data)} conditions with {data_type} tables")

    if not all_data:
        print(f"  No {data_type} data found. Skipping...")
        continue

    # Count tables and stages
    sample_cond = list(all_data.keys())[0]
    stage_counts = defaultdict(int)
    for table_key in all_data[sample_cond]['data']:
        for sl_key in all_data[sample_cond]['data'][table_key]:
            stage = all_data[sample_cond]['data'][table_key][sl_key].get('stage', 2)
            stage_counts[stage] += 1
            break  # Just count once per table
    print(f"  Sample condition {sample_cond}: {len(all_data[sample_cond]['data'])} tables (stage1: {stage_counts[1]}, stage2+: {stage_counts[2]})")

    # Load normalization stats
    if args.norm_stats_path:
        # Load pre-computed norm stats from file (ensures consistency with training)
        print(f"\nLoading normalization stats from file: {args.norm_stats_path}")
        loaded_stats = torch.load(args.norm_stats_path, weights_only=False)
        # Convert from training format {'mean': x, 'std': y} to validation format (mean, std)
        norm_stats = {}
        for idx in [3, 4, 7, 8]:
            if isinstance(loaded_stats[idx], dict):
                norm_stats[idx] = (loaded_stats[idx]['mean'], loaded_stats[idx]['std'])
            else:
                norm_stats[idx] = loaded_stats[idx]
        for idx, (mean, std) in norm_stats.items():
            idx_name = {3: 'temp', 4: 'voltage', 7: 'slew', 8: 'load'}[idx]
            print(f"  Index {idx} ({idx_name}): mean={mean:.6f}, std={std:.6f} [from file]")
    elif args.local_norm or args.local_temp_norm or args.local_volt_norm:
        print(f"\nComputing normalization stats for {data_type}...")
        temp_src = 'local' if args.local_temp_norm else 'TSMC'
        volt_src = 'local' if args.local_volt_norm else 'TSMC'
        slew_load_src = 'local' if args.local_norm else 'TSMC'
        print(f"  Temperature: {temp_src}, Voltage: {volt_src}, Slew/Load: {slew_load_src}")
        norm_stats = load_norm_stats_hybrid(data_type, all_data, args.local_norm, args.local_temp_norm, args.local_volt_norm)
        for idx, (mean, std) in norm_stats.items():
            idx_name = {3: 'temp', 4: 'voltage', 7: 'slew', 8: 'load'}[idx]
            if idx == 3:
                source = 'local' if args.local_temp_norm else 'TSMC'
            elif idx == 4:
                source = 'local' if args.local_volt_norm else 'TSMC'
            elif idx in [7, 8]:
                source = 'local' if args.local_norm else 'TSMC'
            else:
                source = 'TSMC'
            print(f"  Index {idx} ({idx_name}): mean={mean:.6f}, std={std:.6f} [{source}]")
    else:
        print(f"\nLoading TSMC normalization stats for {data_type}...")
        norm_stats = load_norm_stats_hybrid(data_type, all_data, args.local_norm, args.local_temp_norm, args.local_volt_norm)
        for idx, (mean, std) in norm_stats.items():
            idx_name = {3: 'temp', 4: 'voltage', 7: 'slew', 8: 'load'}[idx]
            print(f"  Index {idx} ({idx_name}): mean={mean:.6f}, std={std:.6f} [TSMC]")

    # Apply normalization
    all_data = apply_normalization(all_data, norm_stats)

    # Find valid tables
    print(f"\nFinding valid LUT tables for {data_type}...")
    valid_tables_by_group = find_valid_tables(all_data)
    for group_key, tables in valid_tables_by_group.items():
        print(f"  {group_key}: {len(tables)} valid tables")

    # Load model
    print(f"\nLoading {data_type} model...")
    maml_model = load_model(data_type)
    print(f"  Model loaded.")

    # Run validation
    print("\n" + "=" * 100)
    print(f"LUT TABLE VALIDATION - {data_type.upper()}")
    print("Center point (3,3): 3-shot adaptation → Other 48 points: selective Adam/grad-move")
    print("=" * 100)

    if args.threshold_sweep:
        # Threshold sweep mode: run full validation for each threshold
        print("\n*** THRESHOLD SWEEP MODE ***")
        print(f"Running full validation for each of {len(THRESHOLD_GRID)} thresholds")
        print("=" * 100)

        sweep_results = run_threshold_sweep(data_type, all_data, valid_tables_by_group, maml_model)

        # Print comparison table
        print("\n" + "=" * 100)
        print(f"THRESHOLD COMPARISON RESULTS - {data_type.upper()}")
        print("=" * 100)

        print(f"\n{'MAPE Thresh%':<15} {'Center MAPE%':<15} {'Periph MAPE%':<15} {'All MAPE%':<12} {'Adam Rate%':<12} {'N_Adam':<10}")
        print("-" * 80)

        for r in sweep_results:
            print(f"{r['mape_threshold']:<15.2f} {r['center_mape']:<15.2f} {r['peripheral_mape']:<15.2f} "
                  f"{r['all_mape']:<12.2f} {r['adam_rate']:<12.1f} {r['n_adam_used']:<10}")

        # Find best thresholds
        print("\n" + "=" * 100)
        print("BEST THRESHOLDS")
        print("=" * 100)

        best_all = min(sweep_results, key=lambda x: x['all_mape'])
        print(f"Best All MAPE:        mape_threshold={best_all['mape_threshold']:.2f}% → {best_all['all_mape']:.2f}% (Adam: {best_all['adam_rate']:.1f}%)")

        best_peripheral = min(sweep_results, key=lambda x: x['peripheral_mape'])
        print(f"Best Peripheral MAPE: mape_threshold={best_peripheral['mape_threshold']:.2f}% → {best_peripheral['peripheral_mape']:.2f}% (Adam: {best_peripheral['adam_rate']:.1f}%)")

        # Find best trade-off (low MAPE with reasonable Adam usage)
        # Score: MAPE + 0.05 * adam_rate (penalize high Adam usage slightly)
        best_tradeoff = min(sweep_results, key=lambda x: x['all_mape'] + 0.02 * x['adam_rate'])
        print(f"Best Trade-off:       mape_threshold={best_tradeoff['mape_threshold']:.2f}% → MAPE={best_tradeoff['all_mape']:.2f}%, Adam={best_tradeoff['adam_rate']:.1f}%")

        print("\n" + "=" * 100)
        print("THRESHOLD SWEEP COMPLETE")
        print("=" * 100)

    else:
        # Normal validation mode
        results = run_validation(data_type, all_data, valid_tables_by_group, maml_model)

        # Print results
        print_results(results, data_type)
