#!/usr/bin/env python3
"""
Build Voltage Sweep Dataset from lib files
- Parses 15 lib files and creates support/query datasets
- Saves as .pth files for efficient loading

Usage:
  python build_voltage_sweep_dataset.py
  python build_voltage_sweep_dataset.py --output_dir ./voltage_sweep_data
"""

import os
import sys
import argparse
import torch
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

# Configuration
DATA_DIR = '/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/AND_cells_extracted'

# Voltage sweep groups: (corner, temp) -> [conditions sorted by voltage]
VOLTAGE_SWEEP_GROUPS = {
    ('FF', -40): ['ff0p88vm40c', 'ff0p99vm40c', 'ff1p1vm40c'],      # 0.88V, 0.99V, 1.1V
    ('FF', 125): ['ff0p88v125c', 'ff0p99v125c', 'ff1p1v125c'],      # 0.88V, 0.99V, 1.1V
    ('SS', -40): ['ss0p72vm40c', 'ss0p81vm40c', 'ss0p9vm40c'],      # 0.72V, 0.81V, 0.9V
    ('SS', 125): ['ss0p72v125c', 'ss0p81v125c', 'ss0p9v125c'],      # 0.72V, 0.81V, 0.9V
    ('TT', 25):  ['tt0p8v25c', 'tt0p9v25c', 'tt1p0v25c'],           # 0.8V, 0.9V, 1.0V
}

# Test scenarios: support indices -> query index
TEST_SCENARIOS = {
    'interpolate': {'support': [0, 2], 'query': 1},      # low & high -> middle
    'extrapolate_high': {'support': [0, 1], 'query': 2}, # low & mid -> high
    'extrapolate_low': {'support': [1, 2], 'query': 0},  # mid & high -> low
}

# TSMC Process Parameters
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
            for delay_type in ['cell_rise', 'cell_fall']:
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
    inputs, outputs, metadata = [], [], []
    for sample in samples:
        if sample['delay_type'] not in ['cell_rise', 'cell_fall']:
            continue
        delay_indicator = -1 if 'rise' in sample['delay_type'] else 1
        a = (abc_params['a_n'] + abc_params['a_p']) / 2
        b = abc_params['b_n'] + abc_params['b_p']
        c = abc_params['c_n'] + abc_params['c_p']
        for ri, slew in enumerate(sample['index_1']):
            for ci, load in enumerate(sample['index_2']):
                if ri < len(sample['values']) and ci < len(sample['values'][ri]):
                    inputs.append([a, b, c, temperature, voltage, 2, delay_indicator, slew, load])
                    outputs.append([sample['values'][ri][ci]])
                    metadata.append({'cell_name': sample['cell_name'], 'delay_type': sample['delay_type'],
                                    'related_pin': sample['related_pin'], 'slew_idx': ri, 'load_idx': ci})
    return (torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(outputs, dtype=torch.float32), metadata) if inputs else (torch.tensor([]), torch.tensor([]), [])


def main():
    parser = argparse.ArgumentParser(description='Build Voltage Sweep Dataset')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Directory containing lib files')
    parser.add_argument('--output_dir', type=str, default='/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/voltage_sweep_data',
                        help='Output directory for datasets')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load all lib files
    print("=" * 80)
    print("Step 1: Loading lib files...")
    print("=" * 80)

    base_path = Path(args.data_dir)
    all_data = {}

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
            inputs, outputs, metadata = create_mlp_input(samples, corner, voltage, temperature)
            if len(inputs) > 0:
                all_data[condition] = {
                    'inputs': inputs,
                    'outputs': outputs,
                    'metadata': metadata,
                    'corner': corner,
                    'voltage': voltage,
                    'temperature': temperature
                }
                print(f"  Loaded {condition}: {len(inputs)} samples, V={voltage}V, T={temperature}C")

    print(f"\nLoaded {len(all_data)} conditions")

    # Step 2: Compute normalization stats from all data
    print("\n" + "=" * 80)
    print("Step 2: Computing normalization stats...")
    print("=" * 80)

    all_inputs = torch.cat([all_data[cond]['inputs'] for cond in all_data], dim=0)
    norm_indices = [3, 4, 7, 8]  # temperature, voltage, slew, load
    norm_stats = {}
    for idx in norm_indices:
        col_data = all_inputs[:, idx]
        norm_stats[idx] = {'mean': col_data.mean().item(), 'std': col_data.std().item()}
        print(f"  Index {idx}: mean={norm_stats[idx]['mean']:.6f}, std={norm_stats[idx]['std']:.6f}")

    # Step 3: Build task index
    print("\n" + "=" * 80)
    print("Step 3: Building task index...")
    print("=" * 80)

    def build_index(data_dict):
        index = defaultdict(lambda: defaultdict(dict))
        for cond, data in data_dict.items():
            for idx, meta in enumerate(data['metadata']):
                cell_key = (meta['cell_name'], meta['delay_type'], meta['related_pin'])
                sl_key = (meta['slew_idx'], meta['load_idx'])
                index[cell_key][cond][sl_key] = idx
        return index

    data_index = build_index(all_data)

    # Collect valid tasks (exist in all conditions)
    all_conditions = set()
    for group_conds in VOLTAGE_SWEEP_GROUPS.values():
        all_conditions.update(group_conds)

    all_task_keys = []
    for cell_key in data_index:
        sl_keys = None
        for cond in all_conditions:
            if cond in data_index[cell_key]:
                cond_sl = set(data_index[cell_key][cond].keys())
                sl_keys = cond_sl if sl_keys is None else sl_keys & cond_sl
        if sl_keys:
            for sl_key in sl_keys:
                if all(cond in data_index[cell_key] and sl_key in data_index[cell_key][cond]
                       for cond in all_conditions):
                    all_task_keys.append((cell_key, sl_key))

    print(f"  Total valid tasks: {len(all_task_keys)}")

    # Step 4: Build support/query datasets for each scenario
    print("\n" + "=" * 80)
    print("Step 4: Building support/query datasets...")
    print("=" * 80)

    datasets = {}

    for group_key, conditions in VOLTAGE_SWEEP_GROUPS.items():
        corner, temp = group_key
        group_name = f"{corner}_{temp}C"

        for scenario_name, scenario in TEST_SCENARIOS.items():
            dataset_key = f"{group_name}_{scenario_name}"
            print(f"\n  Building {dataset_key}...")

            support_X_list = []
            support_y_list = []
            query_X_list = []
            query_y_list = []
            task_metadata = []

            support_conds = [conditions[i] for i in scenario['support']]
            query_cond = conditions[scenario['query']]

            for task_idx, (cell_key, sl_key) in enumerate(all_task_keys):
                # Support data (2 voltage points)
                support_X = []
                support_y = []
                for cond in support_conds:
                    idx = data_index[cell_key][cond][sl_key]
                    support_X.append(all_data[cond]['inputs'][idx])
                    support_y.append(all_data[cond]['outputs'][idx])

                support_X = torch.stack(support_X)  # [2, 9]
                support_y = torch.stack(support_y)  # [2, 1]

                # Query data (1 voltage point)
                query_idx = data_index[cell_key][query_cond][sl_key]
                query_X = all_data[query_cond]['inputs'][query_idx].unsqueeze(0)  # [1, 9]
                query_y = all_data[query_cond]['outputs'][query_idx].unsqueeze(0)  # [1, 1]

                support_X_list.append(support_X)
                support_y_list.append(support_y)
                query_X_list.append(query_X)
                query_y_list.append(query_y)

                task_metadata.append({
                    'cell_key': cell_key,
                    'sl_key': sl_key,
                    'support_conditions': support_conds,
                    'query_condition': query_cond,
                })

            datasets[dataset_key] = {
                'support_X': torch.stack(support_X_list),  # [num_tasks, 2, 9]
                'support_y': torch.stack(support_y_list),  # [num_tasks, 2, 1]
                'query_X': torch.stack(query_X_list),      # [num_tasks, 1, 9]
                'query_y': torch.stack(query_y_list),      # [num_tasks, 1, 1]
                'metadata': task_metadata,
                'group': group_name,
                'scenario': scenario_name,
                'support_conditions': support_conds,
                'query_condition': query_cond,
            }

            print(f"    support_X: {datasets[dataset_key]['support_X'].shape}")
            print(f"    support_y: {datasets[dataset_key]['support_y'].shape}")
            print(f"    query_X: {datasets[dataset_key]['query_X'].shape}")
            print(f"    query_y: {datasets[dataset_key]['query_y'].shape}")

    # Step 5: Save datasets
    print("\n" + "=" * 80)
    print("Step 5: Saving datasets...")
    print("=" * 80)

    # Save combined dataset
    output_path = os.path.join(args.output_dir, 'voltage_sweep_dataset.pth')
    torch.save({
        'datasets': datasets,
        'norm_stats': norm_stats,
        'all_task_keys': all_task_keys,
        'voltage_sweep_groups': VOLTAGE_SWEEP_GROUPS,
        'test_scenarios': TEST_SCENARIOS,
        'num_tasks': len(all_task_keys),
    }, output_path)
    print(f"  Saved combined dataset: {output_path}")

    # Also save individual datasets for each scenario
    for dataset_key, dataset in datasets.items():
        individual_path = os.path.join(args.output_dir, f'{dataset_key}.pth')
        torch.save({
            **dataset,
            'norm_stats': norm_stats,
        }, individual_path)
        print(f"  Saved: {individual_path}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"\nTotal tasks: {len(all_task_keys)}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
