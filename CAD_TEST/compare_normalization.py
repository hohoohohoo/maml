#!/usr/bin/env python3
"""
Compare CAD_TEST normalization vs TSMC training normalization
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import re
from pathlib import Path
from collections import OrderedDict, defaultdict

# Add paths
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/model_code')
sys.path.insert(0, '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/utils')

from maml_optimized import OptimizedMAML, MAMLModel_3hidden

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/AND_cells_extracted'
DATA_TYPE = 'cell'
MODEL_PATH = '/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/training_loss_taskdivide_all/cell_innerdiv100_meta32_combined_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth'

# TSMC training data path
TSMC_TRAIN_INPUT_PATH = '/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/MLP_dataset_TSMC/combined_data/tsmc_topology_agnostic_train_input_cell.pth'

TEST_CONDITIONS = ['ff0p88v125c', 'ss0p72v125c', 'ff0p99vm40c', 'ss0p81vm40c', 'tt1p0v25c']

# TSMC Process Parameters
PARAM_A = [1.427, 1.457, 1.430, 1.470, 1.443, 1.483, 1.43, 1.47, 1.43, 1.47]
PARAM_B = [0.026, 0.045, 0, 0, -0.026, -0.05, 0.0208, -0.04, 0.036, -0.0208]
PARAM_C = [0.024, 2.000, 0.024, 2.000, 0.024, 2.000, 0.024, 2.000, 0.024, 2.000]
CORNER_TO_IDX = {'FF': 0, 'TT': 1, 'SS': 2, 'FS': 3, 'SF': 4}

# ============================================================
# HELPER FUNCTIONS (same as before)
# ============================================================

def parse_filename(filename):
    match = re.search(r'(ff|tt|ss|fs|sf)(\d+p\d+)v(m?\d+)c', filename.lower())
    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")
    corner = match.group(1).upper()
    voltage_str = match.group(2).replace('p', '.')
    voltage = float(voltage_str)
    temp_str = match.group(3)
    if temp_str.startswith('m'):
        temperature = -float(temp_str[1:])
    else:
        temperature = float(temp_str)
    return corner, voltage, temperature

def get_abc_params(corner):
    idx = CORNER_TO_IDX.get(corner.upper(), 1)
    nmos_idx = idx * 2
    pmos_idx = idx * 2 + 1
    return {
        'a_n': PARAM_A[nmos_idx], 'a_p': PARAM_A[pmos_idx],
        'b_n': PARAM_B[nmos_idx], 'b_p': PARAM_B[pmos_idx],
        'c_n': PARAM_C[nmos_idx], 'c_p': PARAM_C[pmos_idx],
    }

def compute_abc_for_and_cell(abc_params, delay_indicator):
    a_param = (abc_params['a_n'] + abc_params['a_p']) / 2
    b_param = abc_params['b_n'] + abc_params['b_p']
    c_param = abc_params['c_n'] + abc_params['c_p']
    return a_param, b_param, c_param

def parse_lib_file(lib_path):
    with open(lib_path, 'r') as f:
        content = f.read()

    samples = []
    cell_pattern = r'cell\s*\((\w+)\)\s*\{'

    for cell_match in re.finditer(cell_pattern, content):
        cell_name = cell_match.group(1)
        cell_start = cell_match.end()

        brace_count = 1
        cell_end = cell_start
        for i in range(cell_start, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    cell_end = i
                    break

        cell_content = content[cell_start:cell_end]
        timing_pattern = r'timing\s*\(\)\s*\{\s*related_pin\s*:\s*"(\w+)"'

        for timing_match in re.finditer(timing_pattern, cell_content):
            related_pin = timing_match.group(1)
            timing_start = timing_match.end()

            t_brace_count = 1
            timing_end = timing_start
            for i in range(timing_start, len(cell_content)):
                if cell_content[i] == '{':
                    t_brace_count += 1
                elif cell_content[i] == '}':
                    t_brace_count -= 1
                    if t_brace_count == 0:
                        timing_end = i
                        break

            timing_content = cell_content[timing_start:timing_end]

            for delay_type in ['cell_rise', 'cell_fall']:
                table_pattern = rf'{delay_type}\s*\([^)]+\)\s*\{{'
                table_match = re.search(table_pattern, timing_content)

                if table_match:
                    table_start = table_match.end()
                    tb_count = 1
                    table_end = table_start
                    for i in range(table_start, len(timing_content)):
                        if timing_content[i] == '{':
                            tb_count += 1
                        elif timing_content[i] == '}':
                            tb_count -= 1
                            if tb_count == 0:
                                table_end = i
                                break

                    table_content = timing_content[table_start:table_end]

                    idx1_match = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)', table_content)
                    idx2_match = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)', table_content)
                    values_match = re.search(r'values\s*\(\s*(.*?)\s*\)\s*;', table_content, re.DOTALL)

                    if idx1_match and idx2_match and values_match:
                        index_1 = [float(x.strip()) for x in idx1_match.group(1).split(',')]
                        index_2 = [float(x.strip()) for x in idx2_match.group(1).split(',')]

                        values_str = values_match.group(1).replace('\\', '').replace('\n', ' ')
                        rows = re.findall(r'"([^"]+)"', values_str)
                        values = [[float(x.strip()) for x in row.split(',')] for row in rows]

                        samples.append({
                            'cell_name': cell_name,
                            'delay_type': delay_type,
                            'related_pin': related_pin,
                            'index_1': index_1,
                            'index_2': index_2,
                            'values': values,
                        })
    return samples

def create_mlp_input(samples, corner, voltage, temperature, data_type='cell'):
    abc_params = get_abc_params(corner)
    inputs, outputs, metadata = [], [], []

    target_types = ['cell_rise', 'cell_fall'] if data_type == 'cell' else ['rise_transition', 'fall_transition']

    for sample in samples:
        if sample['delay_type'] not in target_types:
            continue

        delay_indicator = -1 if 'rise' in sample['delay_type'] else 1
        a_param, b_param, c_param = compute_abc_for_and_cell(abc_params, delay_indicator)
        additional_dim = 2

        for row_idx, slew in enumerate(sample['index_1']):
            for col_idx, load in enumerate(sample['index_2']):
                if row_idx < len(sample['values']) and col_idx < len(sample['values'][row_idx]):
                    value = sample['values'][row_idx][col_idx]
                    input_vec = [a_param, b_param, c_param, temperature, voltage,
                                 additional_dim, delay_indicator, slew, load]
                    inputs.append(input_vec)
                    outputs.append([value])
                    metadata.append({
                        'cell_name': sample['cell_name'],
                        'delay_type': sample['delay_type'],
                        'related_pin': sample['related_pin'],
                        'slew': slew, 'load': load,
                        'slew_idx': row_idx, 'load_idx': col_idx,
                    })

    if len(inputs) == 0:
        return torch.tensor([]), torch.tensor([]), []
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(outputs, dtype=torch.float32), metadata

# ============================================================
# LOAD CAD_TEST DATA
# ============================================================
print("=" * 80)
print("LOADING CAD_TEST DATA")
print("=" * 80)

base_path = Path(DATA_DIR)
test_data = {}
support_data = {}

for voltage_dir in ['0p8v', '0p9v', '1p0v']:
    dir_path = base_path / voltage_dir
    if not dir_path.exists():
        continue

    for lib_file in sorted(dir_path.glob('AND_*.tlib')):
        match = re.search(r'AND_lib1_(\w+)_base_400\.tlib', lib_file.name)
        if not match:
            continue

        condition = match.group(1)
        corner, voltage, temperature = parse_filename(lib_file.name)

        samples = parse_lib_file(str(lib_file))
        inputs, outputs, metadata = create_mlp_input(samples, corner, voltage, temperature, DATA_TYPE)

        if len(inputs) == 0:
            continue

        if condition in TEST_CONDITIONS:
            test_data[condition] = (inputs.clone(), outputs.clone(), metadata)
        else:
            support_data[condition] = (inputs.clone(), outputs.clone(), metadata)

print(f"Support conditions: {len(support_data)}")
print(f"Test conditions: {len(test_data)}")

# ============================================================
# CALCULATE CAD_TEST NORMALIZATION STATS
# ============================================================
print("\n" + "=" * 80)
print("CAD_TEST NORMALIZATION STATISTICS")
print("=" * 80)

all_support_inputs = torch.cat([inputs for inputs, _, _ in support_data.values()], dim=0)

norm_indices = [3, 4, 7, 8]  # temperature, voltage, slew, load
feature_names = {3: 'temperature', 4: 'voltage', 7: 'slew', 8: 'load'}
cad_norm_stats = {}

print("\nCAD_TEST Statistics (from support data):")
for idx in norm_indices:
    mean = all_support_inputs[:, idx].mean().item()
    std = all_support_inputs[:, idx].std().item()
    cad_norm_stats[idx] = (mean, std)
    print(f"  {feature_names[idx]} (idx {idx}): mean={mean:.6f}, std={std:.6f}")

# ============================================================
# LOAD TSMC TRAINING DATA AND CALCULATE STATS
# ============================================================
print("\n" + "=" * 80)
print("TSMC TRAINING NORMALIZATION STATISTICS")
print("=" * 80)

print(f"\nLoading: {TSMC_TRAIN_INPUT_PATH}")
tsmc_train_input = torch.load(TSMC_TRAIN_INPUT_PATH)
print(f"TSMC train input shape: {tsmc_train_input.shape}")

tsmc_norm_stats = {}
print("\nTSMC Training Statistics:")
for idx in norm_indices:
    mean = tsmc_train_input[:, :, idx].mean().item()
    std = tsmc_train_input[:, :, idx].std().item()
    tsmc_norm_stats[idx] = (mean, std)
    print(f"  {feature_names[idx]} (idx {idx}): mean={mean:.6f}, std={std:.6f}")

del tsmc_train_input  # Free memory

# ============================================================
# COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("COMPARISON: CAD_TEST vs TSMC NORMALIZATION")
print("=" * 80)

print(f"\n{'Feature':<15} {'CAD_TEST Mean':<15} {'TSMC Mean':<15} {'Mean Diff':<15} {'CAD_TEST Std':<15} {'TSMC Std':<15} {'Std Ratio':<10}")
print("-" * 100)

for idx in norm_indices:
    cad_mean, cad_std = cad_norm_stats[idx]
    tsmc_mean, tsmc_std = tsmc_norm_stats[idx]
    mean_diff = cad_mean - tsmc_mean
    std_ratio = cad_std / tsmc_std if tsmc_std > 0 else 0
    print(f"{feature_names[idx]:<15} {cad_mean:<15.6f} {tsmc_mean:<15.6f} {mean_diff:<15.6f} {cad_std:<15.6f} {tsmc_std:<15.6f} {std_ratio:<10.4f}")

# ============================================================
# RUN VALIDATION WITH BOTH NORMALIZATIONS
# ============================================================
print("\n" + "=" * 80)
print("VALIDATION COMPARISON")
print("=" * 80)

# Load model
maml_model = OptimizedMAML(
    model=MAMLModel_3hidden(in_features=9, layer_length=40),
    dataset_in=None,
    dataset_out=None,
    inner_lr=0.001,
    meta_lr=0.0001
)
state_dict = torch.load(MODEL_PATH, map_location=device)
maml_model.model.load_state_dict(state_dict)
maml_model.model.to(device)
maml_model.model.eval()
print("Model loaded.")

def apply_normalization(data, norm_stats):
    """Apply normalization to data (in-place)"""
    for idx, (mean, std) in norm_stats.items():
        if std > 0:
            data[:, idx] = (data[:, idx] - mean) / std

def build_index(data_dict):
    index = defaultdict(lambda: defaultdict(dict))
    for condition, (inputs, outputs, metadata) in data_dict.items():
        for idx, meta in enumerate(metadata):
            cell_key = (meta['cell_name'], meta['delay_type'], meta['related_pin'])
            sl_key = (meta['slew_idx'], meta['load_idx'])
            index[cell_key][condition][sl_key] = idx
    return index

def run_adaptation(initial_model, X_support, y_support, X_query, y_query, num_steps=40, lr=3e-4):
    """Run adaptation and return predictions"""
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(9, 40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)),
        ('relu3', nn.ReLU()),
        ('l4', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
    ])).to(device)
    model.load_state_dict(initial_model.state_dict())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    y_mean = y_support.mean()
    y_std = y_support.std()
    y_target = (y_support - y_mean) / y_std

    for step in range(num_steps):
        model.zero_grad()
        loss = criterion(model(X_support), y_target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred_query = model(X_query)
        pred_query_orig = pred_query * y_std + y_mean

    return pred_query_orig.cpu().numpy().flatten()

def evaluate_with_normalization(norm_stats, norm_name):
    """Run full evaluation with given normalization stats"""
    # Reload fresh data
    test_data_eval = {}
    support_data_eval = {}

    for voltage_dir in ['0p8v', '0p9v', '1p0v']:
        dir_path = base_path / voltage_dir
        if not dir_path.exists():
            continue

        for lib_file in sorted(dir_path.glob('AND_*.tlib')):
            match = re.search(r'AND_lib1_(\w+)_base_400\.tlib', lib_file.name)
            if not match:
                continue

            condition = match.group(1)
            corner, voltage, temperature = parse_filename(lib_file.name)

            samples = parse_lib_file(str(lib_file))
            inputs, outputs, metadata = create_mlp_input(samples, corner, voltage, temperature, DATA_TYPE)

            if len(inputs) == 0:
                continue

            if condition in TEST_CONDITIONS:
                test_data_eval[condition] = (inputs, outputs, metadata)
            else:
                support_data_eval[condition] = (inputs, outputs, metadata)

    # Apply normalization
    for condition in support_data_eval:
        inputs, outputs, metadata = support_data_eval[condition]
        apply_normalization(inputs, norm_stats)

    for condition in test_data_eval:
        inputs, outputs, metadata = test_data_eval[condition]
        apply_normalization(inputs, norm_stats)

    # Build indices
    support_index = build_index(support_data_eval)
    test_index = build_index(test_data_eval)

    support_conditions = list(support_data_eval.keys())
    test_conditions = list(test_data_eval.keys())

    # Collect all valid task keys
    all_task_keys = []
    for cell_key in support_index:
        available_sl_keys = None
        for condition in support_conditions:
            if condition in support_index[cell_key]:
                condition_sl_keys = set(support_index[cell_key][condition].keys())
                if available_sl_keys is None:
                    available_sl_keys = condition_sl_keys
                else:
                    available_sl_keys = available_sl_keys & condition_sl_keys

        if available_sl_keys:
            for sl_key in available_sl_keys:
                exists_in_test = all(
                    cell_key in test_index and
                    condition in test_index[cell_key] and
                    sl_key in test_index[cell_key][condition]
                    for condition in test_conditions
                )
                if exists_in_test:
                    all_task_keys.append((cell_key, sl_key))

    # Run validation on subset of tasks
    num_tasks = min(500, len(all_task_keys))  # Sample 500 tasks
    sample_indices = np.random.choice(len(all_task_keys), num_tasks, replace=False)

    all_nrmses = []
    all_mapes = []

    for i, task_idx in enumerate(sample_indices):
        cell_key, sl_key = all_task_keys[task_idx]

        # Build support set
        X_support_list, y_support_list = [], []
        for condition in support_conditions:
            sample_idx = support_index[cell_key][condition][sl_key]
            inputs, outputs, metadata = support_data_eval[condition]
            X_support_list.append(inputs[sample_idx:sample_idx+1])
            y_support_list.append(outputs[sample_idx:sample_idx+1])

        # Build query set
        X_query_list, y_query_list = [], []
        for condition in test_conditions:
            sample_idx = test_index[cell_key][condition][sl_key]
            inputs, outputs, metadata = test_data_eval[condition]
            X_query_list.append(inputs[sample_idx:sample_idx+1])
            y_query_list.append(outputs[sample_idx:sample_idx+1])

        X_support = torch.cat(X_support_list, dim=0).to(device)
        y_support = torch.cat(y_support_list, dim=0).to(device)
        X_query = torch.cat(X_query_list, dim=0).to(device)
        y_query = torch.cat(y_query_list, dim=0).to(device)

        # Run adaptation
        predictions = run_adaptation(maml_model.model.model, X_support, y_support, X_query, y_query)
        actuals = y_query.cpu().numpy().flatten()

        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        range_val = actuals.max() - actuals.min()
        nrmse = (rmse / range_val) * 100 if range_val > 0 else 0
        mape = np.mean(np.abs((predictions - actuals) / (actuals + 1e-8))) * 100

        all_nrmses.append(nrmse)
        all_mapes.append(mape)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_tasks}")

    avg_nrmse = np.mean(all_nrmses)
    avg_mape = np.mean(all_mapes)

    return avg_nrmse, avg_mape

# Run with CAD_TEST normalization
print("\n[1] Running with CAD_TEST normalization...")
cad_nrmse, cad_mape = evaluate_with_normalization(cad_norm_stats, "CAD_TEST")
print(f"  Result: NRMSE = {cad_nrmse:.2f}%, MAPE = {cad_mape:.2f}%")

# Run with TSMC normalization
print("\n[2] Running with TSMC normalization...")
tsmc_nrmse, tsmc_mape = evaluate_with_normalization(tsmc_norm_stats, "TSMC")
print(f"  Result: NRMSE = {tsmc_nrmse:.2f}%, MAPE = {tsmc_mape:.2f}%")

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print(f"\n{'Normalization':<20} {'NRMSE (%)':<15} {'MAPE (%)':<15}")
print("-" * 50)
print(f"{'CAD_TEST':<20} {cad_nrmse:<15.2f} {cad_mape:<15.2f}")
print(f"{'TSMC':<20} {tsmc_nrmse:<15.2f} {tsmc_mape:<15.2f}")
print("-" * 50)
print(f"{'Difference':<20} {tsmc_nrmse - cad_nrmse:<15.2f} {tsmc_mape - cad_mape:<15.2f}")

if tsmc_nrmse < cad_nrmse:
    print("\n=> TSMC normalization gives BETTER results!")
else:
    print("\n=> CAD_TEST normalization gives BETTER results!")
