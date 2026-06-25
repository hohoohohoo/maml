#!/usr/bin/env python3
"""
Analyze SS_-40C performance issue in detail.
Compare with other conditions to understand why it performs poorly.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import re
from pathlib import Path
from collections import OrderedDict, defaultdict

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add paths
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/model_code')
from maml_optimized import OptimizedMAML, MAMLModel_3hidden

# Configuration
DATA_DIRS = {
    '0p8v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_0p8v/base_nom_0p8v',
    '0p9v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_0p9v/base_nom_0p9v',
    '1p0v': '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/20200414_base_nom_1p0v/base_nom_1p0v',
}

MODEL_PATH = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/training_loss_taskdivide_all/transition_innerdiv100_meta32_combined_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth'

TSMC_TRAIN_PATH = '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/MLP_dataset_TSMC/combined_data/tsmc_topology_agnostic_train_input_transition.pth'

# Voltage sweep groups
VOLTAGE_SWEEP_GROUPS = {
    ('FF', -40): ['ff0p88vm40c', 'ff0p99vm40c', 'ff1p1vm40c'],
    ('FF', 125): ['ff0p88v125c', 'ff0p99v125c', 'ff1p1v125c'],
    ('SS', -40): ['ss0p72vm40c', 'ss0p81vm40c', 'ss0p9vm40c'],
    ('SS', 125): ['ss0p72v125c', 'ss0p81v125c', 'ss0p9v125c'],
    ('TT', 25):  ['tt0p8v25c', 'tt0p9v25c', 'tt1p0v25c'],
}

CENTER_SLEW_IDX = 3
CENTER_LOAD_IDX = 3

PARAM_A = [1.427, 1.457, 1.430, 1.470, 1.443, 1.483, 1.43, 1.47, 1.43, 1.47]
PARAM_B = [0.026, 0.045, 0, 0, -0.026, -0.05, 0.0208, -0.04, 0.036, -0.0208]
PARAM_C = [0.024, 2.000, 0.024, 2.000, 0.024, 2.000, 0.024, 2.000, 0.024, 2.000]
CORNER_TO_IDX = {'FF': 0, 'TT': 1, 'SS': 2, 'FS': 3, 'SF': 4}


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
            for delay_type in ['rise_transition', 'fall_transition']:
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
                                       'index_2': index_2, 'values': values})
    return samples


def create_mlp_input(samples, corner, voltage, temperature):
    abc_params = get_abc_params(corner)
    data_by_table = defaultdict(dict)

    for sample in samples:
        if sample['delay_type'] not in ['rise_transition', 'fall_transition']:
            continue

        delay_indicator = -1 if 'rise' in sample['delay_type'] else 1
        a = (abc_params['a_n'] + abc_params['a_p']) / 2
        b = abc_params['b_n'] + abc_params['b_p']
        c = abc_params['c_n'] + abc_params['c_p']
        stage = 2  # default

        table_key = (sample['cell_name'], sample['delay_type'], sample['related_pin'])

        for ri, slew in enumerate(sample['index_1']):
            for ci, load in enumerate(sample['index_2']):
                if ri < len(sample['values']) and ci < len(sample['values'][ri]):
                    sl_key = (ri, ci)
                    inp = torch.tensor([a, b, c, temperature, voltage, stage, delay_indicator, slew * 1000, load * 1000],
                                       dtype=torch.float32)
                    out = torch.tensor([sample['values'][ri][ci]], dtype=torch.float32)
                    data_by_table[table_key][sl_key] = {'input': inp, 'output': out}

    return data_by_table


# Parse lib files
print("Parsing lib files...")
parsed_samples = {}

for voltage_dir, dir_path in DATA_DIRS.items():
    if not os.path.exists(dir_path):
        continue
    for lib_file in sorted(Path(dir_path).glob('lib1_*.tlib')):
        match = re.search(r'lib1_(\w+)_base_400\.tlib', lib_file.name)
        if not match:
            continue
        condition = match.group(1)
        try:
            corner, voltage, temperature = parse_filename(lib_file.name)
        except ValueError:
            continue
        samples = parse_lib_file(str(lib_file))
        parsed_samples[condition] = {
            'samples': samples,
            'corner': corner,
            'voltage': voltage,
            'temperature': temperature
        }

print(f"Parsed {len(parsed_samples)} conditions")

# Load data
all_data = {}
for condition, info in parsed_samples.items():
    data_by_table = create_mlp_input(
        info['samples'], info['corner'], info['voltage'], info['temperature']
    )
    if data_by_table:
        all_data[condition] = {
            'data': data_by_table,
            'corner': info['corner'],
            'voltage': info['voltage'],
            'temperature': info['temperature']
        }

# Load normalization stats
tsmc_train = torch.load(TSMC_TRAIN_PATH, weights_only=False)
norm_stats = {}
for idx in [3, 4, 7, 8]:
    norm_stats[idx] = (tsmc_train[:, :, idx].mean().item(), tsmc_train[:, :, idx].std().item())
del tsmc_train

# Apply normalization
for cond in all_data:
    for table_key in all_data[cond]['data']:
        for sl_key in all_data[cond]['data'][table_key]:
            inp = all_data[cond]['data'][table_key][sl_key]['input']
            for idx, (mean, std) in norm_stats.items():
                if std > 0:
                    inp[idx] = (inp[idx] - mean) / std

# Load model
maml_model = OptimizedMAML(
    model=MAMLModel_3hidden(in_features=9, layer_length=40),
    dataset_in=None, dataset_out=None, inner_lr=0.001, meta_lr=0.0001
)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
maml_model.model.load_state_dict(state_dict)
maml_model.model.to(device)
maml_model.model.eval()


def create_adapted_model(initial_model):
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(9, 40)), ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)), ('relu3', nn.ReLU()),
        ('l4', nn.Linear(40, 40)), ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
    ])).to(device)
    model.load_state_dict(initial_model.state_dict())
    return model


# ============================================================
# ANALYSIS: Compare SS_-40C vs other groups
# ============================================================

print("\n" + "=" * 100)
print("DETAILED ANALYSIS: Comparing SS_-40C with other groups")
print("=" * 100)

# For each group, analyze:
# 1. Delay value statistics (range, mean, std)
# 2. Voltage sensitivity (how much delay changes with voltage)
# 3. Model initial predictions
# 4. Grad/move values

analysis_results = {}

for group_key, conditions in VOLTAGE_SWEEP_GROUPS.items():
    corner, temp = group_key
    group_name = f"{corner}_{temp}C"

    if not all(c in all_data for c in conditions):
        print(f"\n{group_name}: Missing conditions, skipping...")
        continue

    print(f"\n{'='*80}")
    print(f"GROUP: {group_name}")
    print(f"Conditions: {conditions}")
    print(f"{'='*80}")

    # Find common tables
    common_tables = None
    for cond in conditions:
        table_keys = set(all_data[cond]['data'].keys())
        if common_tables is None:
            common_tables = table_keys
        else:
            common_tables &= table_keys

    center_key = (CENTER_SLEW_IDX, CENTER_LOAD_IDX)
    valid_tables = [t for t in common_tables if all(center_key in all_data[c]['data'][t] for c in conditions)]

    print(f"Valid tables: {len(valid_tables)}")

    # Sample 20 tables for analysis
    np.random.seed(42)
    valid_tables_list = list(valid_tables)
    sample_indices = np.random.choice(len(valid_tables_list), min(20, len(valid_tables_list)), replace=False)
    sample_tables = [valid_tables_list[i] for i in sample_indices]

    # Collect statistics
    delay_values_by_voltage = {cond: [] for cond in conditions}
    voltage_sensitivities = []
    initial_predictions = []
    grad_values = []
    move_values = []
    center_mapes = []

    for table_key in sample_tables:
        # Get center point data for all 3 voltages
        y_values = []
        for cond in conditions:
            data = all_data[cond]['data'][table_key][center_key]
            y_val = data['output'].item()
            y_values.append(y_val)
            delay_values_by_voltage[cond].append(y_val)

        # Calculate voltage sensitivity (delay range / voltage range)
        y_range = max(y_values) - min(y_values)
        voltage_sensitivities.append(y_range)

        # Get model predictions
        X_center_list = []
        y_center_list = []
        for cond in conditions:
            data = all_data[cond]['data'][table_key][center_key]
            X_center_list.append(data['input'])
            y_center_list.append(data['output'])

        X_center = torch.stack(X_center_list).to(device)
        y_center = torch.stack(y_center_list).to(device)

        # Initial model prediction (before any adaptation)
        with torch.no_grad():
            pred_init = maml_model.model.model(X_center).flatten().cpu().numpy()

        initial_predictions.append(pred_init)

        # Calculate grad/move
        y_support_flat = y_center.flatten()
        y_mean = y_support_flat.mean()
        y_std = y_support_flat.std()
        if y_std < 1e-8:
            y_std = torch.tensor(1.0)

        y_norm = (y_support_flat - y_mean) / y_std

        pred_tensor = torch.tensor(pred_init, device=device)
        pred_min = pred_tensor.min().item()
        pred_max = pred_tensor.max().item()
        y_min = y_norm.min().item()
        y_max = y_norm.max().item()

        if abs(pred_max - pred_min) > 1e-8:
            grad = (y_max - y_min) / (pred_max - pred_min)
        else:
            grad = 1.0

        center_pred = pred_tensor[1].item()
        if abs(grad) > 1e-8:
            move = center_pred - y_norm[1].item() / grad
        else:
            move = 0.0

        grad_values.append(grad)
        move_values.append(move)

        # Calculate center prediction after grad/move (middle voltage)
        pred_center_adapted = (pred_tensor[1].item() - move) * y_std.item() * grad + y_mean.item()
        actual_center = y_center[1].item()
        center_mape = abs(pred_center_adapted - actual_center) / (abs(actual_center) + 1e-8) * 100
        center_mapes.append(center_mape)

    # Print statistics
    print(f"\n--- Delay Value Statistics ---")
    for cond in conditions:
        vals = delay_values_by_voltage[cond]
        print(f"  {cond}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    print(f"\n--- Voltage Sensitivity (delay range across voltages) ---")
    print(f"  Mean: {np.mean(voltage_sensitivities):.6f}")
    print(f"  Std:  {np.std(voltage_sensitivities):.6f}")
    print(f"  Range: [{np.min(voltage_sensitivities):.6f}, {np.max(voltage_sensitivities):.6f}]")

    print(f"\n--- Initial Model Prediction Statistics ---")
    pred_arr = np.array(initial_predictions)
    print(f"  Mean prediction: {pred_arr.mean():.4f}")
    print(f"  Std prediction:  {pred_arr.std():.4f}")
    print(f"  Range: [{pred_arr.min():.4f}, {pred_arr.max():.4f}]")

    print(f"\n--- Grad/Move Statistics ---")
    print(f"  Grad - Mean: {np.mean(grad_values):.4f}, Std: {np.std(grad_values):.4f}")
    print(f"  Move - Mean: {np.mean(move_values):.4f}, Std: {np.std(move_values):.4f}")

    print(f"\n--- Center MAPE (after grad/move, before Adam) ---")
    print(f"  Mean: {np.mean(center_mapes):.2f}%")
    print(f"  Std:  {np.std(center_mapes):.2f}%")
    print(f"  Range: [{np.min(center_mapes):.2f}%, {np.max(center_mapes):.2f}%]")

    analysis_results[group_name] = {
        'voltage_sensitivity': np.mean(voltage_sensitivities),
        'grad_mean': np.mean(grad_values),
        'grad_std': np.std(grad_values),
        'move_mean': np.mean(move_values),
        'move_std': np.std(move_values),
        'center_mape_mean': np.mean(center_mapes),
        'center_mape_std': np.std(center_mapes),
    }

# Summary comparison
print("\n" + "=" * 100)
print("SUMMARY COMPARISON")
print("=" * 100)

print(f"\n{'Group':<15} {'V_Sensitivity':<15} {'Grad Mean':<12} {'Grad Std':<12} {'Move Mean':<12} {'Center MAPE%':<15}")
print("-" * 90)

for group_name in sorted(analysis_results.keys()):
    r = analysis_results[group_name]
    print(f"{group_name:<15} {r['voltage_sensitivity']:<15.6f} {r['grad_mean']:<12.4f} {r['grad_std']:<12.4f} "
          f"{r['move_mean']:<12.4f} {r['center_mape_mean']:<15.2f}")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
