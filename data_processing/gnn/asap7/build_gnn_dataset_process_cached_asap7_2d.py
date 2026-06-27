#!/usr/bin/env python
"""
Build ASAP7 GNN dataset with 2-D V×T tensor format for MAML adaptation.

Mirrors `build_gnn_dataset_process_cached_tsmc_2d.py` but for ASAP7:
- Lib folder pattern: {prefix}_{a_idx}_{b_idx}_{c_idx}_{temp_str}
- Train: 4×4×4 = 64 (a,b,c) corners × 6 temps  (= 384 folders per prefix)
- Test:  3×3×3 = 27 (a,b,c) corners × 5 temps  (= 135 folders per prefix)
- Voltage axis: 0.40 → 1.00 V, 61 lib files per folder (nominal 0.70 V)
- Process params parsed via existing parse_process_conditions_from_filename
- Topology applied via existing apply_topology_with_process

Train tensor: [61_V, 6_T, total_nodes, num_features]
Test  tensor: per-cell [61_V, 5_T, total_nodes_per_cell, num_features]

Sampling controls (new vs TSMC 2-D):
- --sampling_ratio R     : after V×T-alignment, randomly keep R fraction of train
                            tasks (e.g. 0.10 reproduces the "10pct" 1-D file size).
- --max_test_tasks_per_cell N : cap per-(corner) test tasks at N (random subsample).

Author: claude (mirroring TSMC 2-D builder, ASAP7 conventions from
        build_gnn_dataset_process_cached_asap7.py).
"""

import argparse
import gc
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add paths so we can import helpers from the generic 1-D ASAP7 builder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MLP', 'utils'))

# Reuse from the 1-D ASAP7 generic builder
from build_gnn_dataset_process_cached_asap7 import (
    INTRA_TOPOLOGY_CELLS,
    parse_process_conditions_from_filename,
    apply_topology_with_process,
)


# ---------------------------------------------------------------------------
# ASAP7-specific constants
# ---------------------------------------------------------------------------
TRAIN_TEMPERATURES = [-25, 12.5, 37.5, 62.5, 87.5, 125]   # 6 temps
TEST_TEMPERATURES  = [0, 25, 50, 75, 100]                  # 5 temps

# ASAP7 (a, b, c) index grids — see parse_process_conditions_from_filename
TRAIN_A_IDX = [0, 1, 2, 3]   # 4 values: 0.625, 0.875, 1.125, 1.375
TRAIN_B_IDX = [0, 1, 2, 3]
TRAIN_C_IDX = [0, 1, 2, 3]
TEST_A_IDX  = [0, 1, 2]      # 3 values: 0.75, 1.00, 1.25
TEST_B_IDX  = [0, 1, 2]
TEST_C_IDX  = [0, 1, 2]


# ---------------------------------------------------------------------------
# Enumeration / naming helpers (small, kept as top-level public API)
# ---------------------------------------------------------------------------
def temp_to_folder_str(temp: float) -> str:
    """`12.5` -> `'12p5'`, `-25` -> `'-25'`."""
    if temp == int(temp):
        return str(int(temp))
    return str(temp).replace('.', 'p')


def corner_str(a: int, b: int, c: int) -> str:
    return f"{a}_{b}_{c}"


def enumerate_train_corners() -> List[Tuple[int, int, int]]:
    return [(a, b, c) for a in TRAIN_A_IDX for b in TRAIN_B_IDX for c in TRAIN_C_IDX]


def enumerate_test_corners() -> List[Tuple[int, int, int]]:
    return [(a, b, c) for a in TEST_A_IDX for b in TEST_B_IDX for c in TEST_C_IDX]


def enumerate_train_folders(lib_base_paths: List[Path], prefixes: List[str]) -> List[Tuple[Tuple[int, int, int], float, Path, str, str]]:
    """
    Return list of (corner, temp, folder_path, prefix, source_dir) for all
    EXISTING train folders across all given lib_base_paths × prefixes.

    Train folder = `{prefix}_{a}_{b}_{c}_{temp_str}`.
    """
    out = []
    for base in lib_base_paths:
        base = Path(base)
        for prefix in prefixes:
            for (a, b, c) in enumerate_train_corners():
                for t in TRAIN_TEMPERATURES:
                    folder = base / f"{prefix}_{a}_{b}_{c}_{temp_to_folder_str(t)}"
                    if folder.is_dir():
                        out.append(((a, b, c), float(t), folder, prefix, str(base)))
    return out


def enumerate_test_folders(lib_base_paths: List[Path], prefixes: List[str]) -> List[Tuple[Tuple[int, int, int], float, Path, str, str]]:
    out = []
    for base in lib_base_paths:
        base = Path(base)
        for prefix in prefixes:
            for (a, b, c) in enumerate_test_corners():
                for t in TEST_TEMPERATURES:
                    folder = base / f"{prefix}_{a}_{b}_{c}_{temp_to_folder_str(t)}"
                    if folder.is_dir():
                        out.append(((a, b, c), float(t), folder, prefix, str(base)))
    return out


# ---------------------------------------------------------------------------
# Lib file → list of minimal samples (ASAP7 version)
# ---------------------------------------------------------------------------
def _process_lib_file_asap7(lib_file_path: str, topology_cache: dict, cache_type: str,
                            process_params: Dict[str, float], data_type: str = 'cell',
                            voltage_mode: str = 'all_nodes', slew_mode: str = 'all') -> List[dict]:
    """
    Process a single .lib file → list of minimal samples (ASAP7 path).
    Mirrors TSMC's process_lib_file_for_unified but uses
    apply_topology_with_process (the ASAP7 generic helper).
    """
    if data_type == 'cell':
        from libdata_extract_MAML_cell import parse_liberty_pin_blocks, flatten_pin_data
    else:
        from libdata_extract_MAML_transition import parse_liberty_pin_blocks, flatten_pin_data

    with open(lib_file_path, 'r') as f:
        lines = f.readlines()

    pin_data = parse_liberty_pin_blocks(lines)
    flattened, _cap = flatten_pin_data(pin_data)

    minimal_samples: List[dict] = []
    for sample in flattened:
        cell_name = sample['cell']
        voltage = sample['Voltage']
        delay_type = sample['delay_type']
        input_port_names = sample['input_port_name']
        output_port_name = sample.get('pin_name', 'Y')   # ASAP7 typically uses 'Y'
        related_pin = sample.get('related_pin', None)

        if cell_name not in topology_cache:
            continue

        if cache_type == 'stage_aware':
            stage_delay_type = 'rise_transition' if 'rise' in delay_type else 'fall_transition'
        else:
            stage_delay_type = delay_type

        input_slews = sample.get('index_1', [40.0])
        output_loads = sample.get('index_2', [5.76])
        timing_values = sample.get('values', [[0.0]])

        actual_rows = len(timing_values) if isinstance(timing_values, list) else 0
        actual_cols = (len(timing_values[0]) if actual_rows > 0 and isinstance(timing_values[0], list)
                       else 0)
        effective_rows = (min(len(input_slews), actual_rows)
                          if actual_rows > 0 else len(input_slews))
        effective_cols = (min(len(output_loads), actual_cols)
                          if actual_cols > 0 else len(output_loads))

        for row_idx in range(effective_rows):
            for col_idx in range(effective_cols):
                input_slew = input_slews[row_idx]
                output_load = output_loads[col_idx]
                output_value = timing_values[row_idx][col_idx]

                try:
                    if cache_type == 'stage_aware':
                        cell_cache = topology_cache[cell_name]
                        if 'output_topologies' not in cell_cache:
                            continue
                        if output_port_name not in cell_cache['output_topologies']:
                            continue

                    graph_sample = apply_topology_with_process(
                        topology_cache, cache_type,
                        cell_name, output_port_name, stage_delay_type,
                        voltage, input_slew, output_load, input_port_names,
                        process_params,
                        slew_mode=slew_mode, related_pin=related_pin,
                        voltage_mode=voltage_mode,
                    )

                    if graph_sample['node_features'].shape[1] != 11:
                        continue

                    minimal_samples.append({
                        'node_features': graph_sample['node_features'],
                        'output': output_value,
                        'cell_name': cell_name,
                        'delay_type': stage_delay_type,
                        'output_name': output_port_name,
                        'num_nodes': graph_sample['node_features'].shape[0],
                    })
                except Exception:
                    # Try alternative output name (Y vs YN)
                    alt_output = 'YN' if output_port_name == 'Y' else 'Y'
                    try:
                        graph_sample = apply_topology_with_process(
                            topology_cache, cache_type,
                            cell_name, alt_output, stage_delay_type,
                            voltage, input_slew, output_load, input_port_names,
                            process_params,
                            slew_mode=slew_mode, related_pin=related_pin,
                            voltage_mode=voltage_mode,
                        )
                        if graph_sample['node_features'].shape[1] == 11:
                            minimal_samples.append({
                                'node_features': graph_sample['node_features'],
                                'output': output_value,
                                'cell_name': cell_name,
                                'delay_type': stage_delay_type,
                                'output_name': alt_output,
                                'num_nodes': graph_sample['node_features'].shape[0],
                            })
                    except Exception:
                        pass

    return minimal_samples


def _process_directory_asap7(folder_path: Path, topology_cache: dict, cache_type: str,
                              process_params: Dict[str, float], data_type: str = 'cell',
                              voltage_mode: str = 'all_nodes', slew_mode: str = 'all') -> Tuple[List[List[dict]], int]:
    """
    Process all .lib files in folder → list[per_lib_idx][task_idx].
    Returns (all_samples_per_lib, num_tasks).
    """
    lib_files = sorted(folder_path.glob('*.lib'))
    if not lib_files:
        return [], 0

    all_samples_per_lib: List[List[dict]] = []
    for lib_file in lib_files:
        lib_samples = _process_lib_file_asap7(
            str(lib_file), topology_cache, cache_type, process_params, data_type,
            voltage_mode=voltage_mode, slew_mode=slew_mode,
        )
        all_samples_per_lib.append(lib_samples)

    if not all_samples_per_lib:
        return [], 0
    num_tasks = min(len(s) for s in all_samples_per_lib)
    return all_samples_per_lib, num_tasks


# ---------------------------------------------------------------------------
# Setup / config helpers
# ---------------------------------------------------------------------------

def _print_config_banner(
    cache_path, cache_type, lib_base_paths, output_dir, prefixes, data_type,
    voltage_mode, slew_mode, include_zeros_in_norm, topology_suffix,
    sampling_ratio, max_test_tasks_per_cell, sampling_seed,
) -> None:
    print('=' * 80)
    print('BUILDING ASAP7 GNN DATASET - 2-D V×T FORMAT')
    print('=' * 80)
    print(f'Cache path        : {cache_path}')
    print(f'Cache type        : {cache_type}')
    print(f'Lib base paths    : {lib_base_paths}')
    print(f'Output dir        : {output_dir}')
    print(f'Prefixes          : {prefixes}')
    print(f'Data type         : {data_type}')
    print(f'Voltage mode      : {voltage_mode}')
    print(f'Slew mode         : {slew_mode}')
    print(f'Include zeros norm: {include_zeros_in_norm}')
    print(f'Topology suffix   : "{topology_suffix}"')
    print(f'Sampling ratio    : {sampling_ratio} (train V×T-aligned tasks)')
    print(f'Max test tasks/cell: {max_test_tasks_per_cell}')
    print(f'Sampling seed     : {sampling_seed}')
    print(f'\nTrain temperatures: {TRAIN_TEMPERATURES}')
    print(f'Test  temperatures: {TEST_TEMPERATURES}')
    print(f'Train corners (a,b,c): {len(enumerate_train_corners())} combos (4×4×4)')
    print(f'Test  corners (a,b,c): {len(enumerate_test_corners())} combos (3×3×3)')
    print(f'\n⚠️  Excluding INTRA_TOPOLOGY_CELLS from train data:')
    for cell in INTRA_TOPOLOGY_CELLS:
        print(f'   - {cell}')
    print('=' * 80)


def _setup_paths_and_suffix(output_dir, voltage_mode, slew_mode, topology_suffix) -> Tuple[Path, str]:
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    voltage_suffix = f'_{voltage_mode}' if voltage_mode != 'all_nodes' else ''
    slew_suffix = '_relpin' if slew_mode == 'related_pin_only' else ''
    mode_suffix = f'{topology_suffix}{voltage_suffix}{slew_suffix}'
    return output_dir_p, mode_suffix


# ---------------------------------------------------------------------------
# Train data: collection + tensor build + save
# ---------------------------------------------------------------------------

def _collect_all_train_tasks(
    train_folders, topology_cache, cache_type, data_type,
    voltage_mode, slew_mode, sampling_ratio, rng: random.Random,
) -> Tuple[List[dict], Optional[int]]:
    """For each (corner, temp) train folder, run _process_directory_asap7 and
    fold per-lib tasks aligned by (volt_idx, temp_idx) into the V×T plane.
    Applies per-corner `sampling_ratio` and excludes INTRA_TOPOLOGY_CELLS.
    Returns (all_train_tasks, num_voltages_observed).
    """
    num_train_temps = len(TRAIN_TEMPERATURES)

    # Group train folders by `{prefix}_{a}_{b}_{c}` corner label
    corner_to_temp_folders: Dict[str, List[Tuple[float, Path, str, str]]] = defaultdict(list)
    for (corner, temp, folder, prefix, base) in train_folders:
        corner_key = f"{prefix}_{corner[0]}_{corner[1]}_{corner[2]}"
        corner_to_temp_folders[corner_key].append((temp, folder, prefix, base))
    temp_order = {t: i for i, t in enumerate(TRAIN_TEMPERATURES)}
    for k in corner_to_temp_folders:
        corner_to_temp_folders[k].sort(key=lambda x: temp_order.get(x[0], 999))

    all_train_tasks: List[dict] = []
    num_voltages_observed: Optional[int] = None

    sorted_corners = sorted(corner_to_temp_folders.keys())
    for corner_idx, corner in enumerate(sorted_corners):
        tf_list = corner_to_temp_folders[corner]
        if len(tf_list) != num_train_temps:
            print(f'\n[Corner {corner_idx+1}/{len(corner_to_temp_folders)}] {corner} '
                  f'— incomplete: {len(tf_list)}/{num_train_temps} temps, skipping')
            continue

        print(f'\n[Corner {corner_idx+1}/{len(corner_to_temp_folders)}] {corner}')

        samples_2d_per_temp: List[List[List[dict]]] = []
        num_tasks_per_temp: List[int] = []
        for (temp, folder, prefix, _base) in tf_list:
            lib_files_check = sorted(folder.glob('*.lib'))
            if not lib_files_check:
                print(f'   [temp {temp}] {folder.name} — no .lib files, ABORT corner')
                samples_2d_per_temp = []
                break
            lib_prefix = lib_files_check[0].stem.rsplit('_', 1)[0] + '_'
            process_params = parse_process_conditions_from_filename(lib_prefix, is_test=False)

            print(f'   [temp {temp:>6.1f}°C] {folder.name}', flush=True)

            if num_voltages_observed is None:
                num_voltages_observed = len(lib_files_check)
                print(f'      Number of voltage lib files: {num_voltages_observed}')
            elif len(lib_files_check) != num_voltages_observed:
                print(f'      ⚠️ voltage axis mismatch ({len(lib_files_check)} vs '
                      f'expected {num_voltages_observed}), skipping corner')
                samples_2d_per_temp = []
                break

            all_samples_per_lib, num_tasks_dir = _process_directory_asap7(
                folder, topology_cache, cache_type, process_params, data_type,
                voltage_mode=voltage_mode, slew_mode=slew_mode,
            )
            print(f'      ✓ {num_tasks_dir} tasks from {len(lib_files_check)} lib files')
            samples_2d_per_temp.append(all_samples_per_lib)
            num_tasks_per_temp.append(num_tasks_dir)

        if not samples_2d_per_temp:
            continue

        num_tasks_per_corner = min(num_tasks_per_temp) if num_tasks_per_temp else 0
        if len(set(num_tasks_per_temp)) > 1:
            print(f'   ⚠️ num_tasks varies across temps: {num_tasks_per_temp}, '
                  f'taking min={num_tasks_per_corner}')

        # First pass: collect candidate aligned task indices (cell-name & INTRA filter)
        candidate_indices: List[int] = []
        excluded_count = 0
        skipped_mismatch = 0
        for task_idx in range(num_tasks_per_corner):
            cell_names_at_idx = set()
            for temp_idx in range(num_train_temps):
                per_lib_lists = samples_2d_per_temp[temp_idx]
                if 0 < len(per_lib_lists) and task_idx < len(per_lib_lists[0]):
                    cell_names_at_idx.add(per_lib_lists[0][task_idx]['cell_name'])
            if len(cell_names_at_idx) != 1:
                skipped_mismatch += 1
                continue
            cell_name = cell_names_at_idx.pop()
            if cell_name in INTRA_TOPOLOGY_CELLS:
                excluded_count += 1
                continue
            candidate_indices.append(task_idx)

        # Apply sampling_ratio at corner level
        if sampling_ratio < 1.0 and candidate_indices:
            n_keep = max(1, int(round(len(candidate_indices) * sampling_ratio)))
            rng.shuffle(candidate_indices)
            candidate_indices = sorted(candidate_indices[:n_keep])

        # Build the (V, T) plane for each selected task
        corner_tasks_added = 0
        for task_idx in candidate_indices:
            samples_by_TV = {}
            valid = True
            for temp_idx in range(num_train_temps):
                per_lib_lists = samples_2d_per_temp[temp_idx]
                for volt_idx in range(num_voltages_observed):
                    if volt_idx >= len(per_lib_lists) or task_idx >= len(per_lib_lists[volt_idx]):
                        valid = False
                        break
                    samples_by_TV[(volt_idx, temp_idx)] = per_lib_lists[volt_idx][task_idx]
                if not valid:
                    break
            if not valid:
                skipped_mismatch += 1
                continue
            all_train_tasks.append({
                'corner': corner,
                'samples_by_TV': samples_by_TV,
            })
            corner_tasks_added += 1

        print(f'   ✓ {corner_tasks_added} tasks added (after {sampling_ratio:.2f} sampling); '
              f'{excluded_count} INTRA_TOPOLOGY excluded; '
              f'{skipped_mismatch} skipped (mismatch)')

        del samples_2d_per_temp
        gc.collect()

    print(f'\n📊 Total train tasks collected: {len(all_train_tasks)}')
    print(f'   num_voltages: {num_voltages_observed}, num_temps: {num_train_temps}')
    return all_train_tasks, num_voltages_observed


def _allocate_and_fill_train_tensors(
    all_train_tasks: List[dict], num_voltages: int, num_train_temps: int, num_features: int,
) -> dict:
    task_node_counts, cell_names, delay_types, output_names, task_corners = [], [], [], [], []
    for t in all_train_tasks:
        sample0 = t['samples_by_TV'][(0, 0)]
        task_node_counts.append(sample0['num_nodes'])
        cell_names.append(sample0['cell_name'])
        delay_types.append(sample0['delay_type'])
        output_names.append(sample0['output_name'])
        task_corners.append(t['corner'])

    num_tasks = len(all_train_tasks)
    total_nodes = sum(task_node_counts)
    print(f'\n🔧 Train task metadata: {num_tasks} tasks, {total_nodes} total nodes')
    print(f'💾 Allocating 4-D tensors:')
    print(f'   node_features: [{num_voltages}, {num_train_temps}, {total_nodes}, {num_features}]')
    print(f'   outputs:       [{num_voltages}, {num_train_temps}, {num_tasks}]')

    node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
    node_slices[1:] = np.cumsum(task_node_counts)

    all_node_features = np.zeros(
        (num_voltages, num_train_temps, total_nodes, num_features), dtype=np.float32,
    )
    all_outputs = np.zeros((num_voltages, num_train_temps, num_tasks), dtype=np.float32)

    print(f'\n📝 Filling tensors...')
    for task_idx, t in enumerate(all_train_tasks):
        if task_idx % 10000 == 0:
            print(f'   Processing task {task_idx}/{num_tasks}...')
        node_start = node_slices[task_idx]
        node_end = node_slices[task_idx + 1]
        for (v, temp_idx), sample in t['samples_by_TV'].items():
            nf = sample['node_features']
            if isinstance(nf, torch.Tensor):
                nf = nf.cpu().numpy()
            all_node_features[v, temp_idx, node_start:node_end, :] = nf
            out = sample['output']
            if isinstance(out, torch.Tensor):
                out = out.item()
            all_outputs[v, temp_idx, task_idx] = out

    return {
        'node_features': all_node_features,
        'outputs': all_outputs,
        'node_slices': node_slices,
        'task_node_counts': task_node_counts,
        'cell_names': cell_names,
        'delay_types': delay_types,
        'output_names': output_names,
        'task_corners': task_corners,
        'num_tasks': num_tasks,
        'total_nodes': total_nodes,
    }


def _extract_voltage_axis(node_features: np.ndarray, num_voltages: int) -> np.ndarray:
    """Read voltage value at (v, t=0) for each voltage slice (feature index 4 = voltage)."""
    voltages = np.zeros(num_voltages, dtype=np.float32)
    for v in range(num_voltages):
        v_slice = node_features[v, 0, :, 4]
        nz = v_slice[v_slice != 0]
        voltages[v] = float(nz[0]) if len(nz) > 0 else 0.0
    return voltages


def _compute_normalization_stats(
    node_features: np.ndarray, include_zeros_in_norm: bool,
) -> Tuple[Dict[int, Dict[str, float]], List[int], List[str]]:
    normalize_indices = [4, 5, 6, 10]
    normalize_names = ['voltage', 'input_slew', 'output_load', 'temperature']
    print(f'\n📊 Calculating normalization statistics...')

    norm_stats: Dict[int, Dict[str, float]] = {}
    for idx, name in zip(normalize_indices, normalize_names):
        feature_data = node_features[:, :, :, idx].flatten()
        if include_zeros_in_norm:
            mean = float(np.mean(feature_data))
            std = float(np.std(feature_data))
            if std == 0:
                std = 1.0
        else:
            nz = feature_data[feature_data != 0]
            if len(nz) > 0:
                mean = float(np.mean(nz))
                std = float(np.std(nz))
            else:
                mean = 0.0
                std = 1.0
        norm_stats[idx] = {'name': name, 'mean': mean, 'std': std}
        print(f'   {name}: mean={mean:.6f}, std={std:.6f}')
    return norm_stats, normalize_indices, normalize_names


def _save_train_dataset(
    train_tensors: dict, norm_stats, normalize_indices, normalize_names,
    output_dir: Path, data_type, cache_type, mode_suffix,
    cache_path, num_voltages, num_train_temps, num_features,
    voltage_mode, slew_mode, topology_suffix, sampling_ratio, sampling_seed,
    include_zeros_in_norm,
) -> Path:
    voltages = _extract_voltage_axis(train_tensors['node_features'], num_voltages)
    temperatures = np.array(TRAIN_TEMPERATURES, dtype=np.float32)

    sampling_tag = '' if sampling_ratio >= 1.0 else f'_samp{int(round(sampling_ratio*100)):02d}pct'
    train_path = output_dir / f'train_{data_type}_{cache_type}{mode_suffix}{sampling_tag}_2d.pth'
    print(f'\n💾 Saving train dataset: {train_path}')

    train_data = {
        'node_features': torch.from_numpy(train_tensors['node_features']),
        'outputs': torch.from_numpy(train_tensors['outputs']),
        'node_slices': torch.from_numpy(train_tensors['node_slices']),
        'voltages': torch.from_numpy(voltages),
        'temperatures': torch.from_numpy(temperatures),
        'cell_names': train_tensors['cell_names'],
        'delay_types': train_tensors['delay_types'],
        'output_names': train_tensors['output_names'],
        'task_corners': train_tensors['task_corners'],
        'node_counts': train_tensors['task_node_counts'],
        'num_tasks': train_tensors['num_tasks'],
        'num_voltages': num_voltages,
        'num_temps': num_train_temps,
        'num_features': num_features,
        'total_nodes': train_tensors['total_nodes'],
        'format': 'unified_4d_VT',
        'process_node': 'ASAP7',
        'data_type': data_type,
        'graph_mode': cache_type,
        'cache_path': cache_path,
        'train_temperatures': TRAIN_TEMPERATURES,
        'voltage_mode': voltage_mode,
        'slew_mode': slew_mode,
        'topology_suffix': topology_suffix,
        'sampling_ratio': sampling_ratio,
        'sampling_seed': sampling_seed,
        'excluded_cells': INTRA_TOPOLOGY_CELLS,
        'norm_stats': {
            'node_features': {
                s['name']: {'mean': s['mean'], 'std': s['std']}
                for s in norm_stats.values()
            }
        },
        'normalize_indices': normalize_indices,
        'normalize_names': normalize_names,
        'normalize_nonzero_only': not include_zeros_in_norm,
        'include_zeros_in_norm': include_zeros_in_norm,
    }
    torch.save(train_data, train_path)
    print(f'   ✅ Saved: {train_path}')
    print(f'   node_features: {train_data["node_features"].shape}')
    print(f'   outputs:       {train_data["outputs"].shape}')
    return train_path


# ---------------------------------------------------------------------------
# Test data: extract partials → merge per cell
# ---------------------------------------------------------------------------

def _extract_test_partials(
    test_folders, topology_cache, cache_type, data_type, temp_dir: Path,
    voltage_mode, slew_mode,
) -> set:
    """Step 1: per test folder, compute samples and dump per-cell partial .pth files
    into temp_dir. Returns the set of all cell_names seen.
    """
    all_cell_names: set = set()
    print(f'\n📂 Step 1: Processing test folders and saving partials...')
    for dir_idx, (corner, temperature, folder, prefix, _base) in enumerate(test_folders):
        lib_files_check = sorted(folder.glob('*.lib'))
        if not lib_files_check:
            continue
        lib_prefix = lib_files_check[0].stem.rsplit('_', 1)[0] + '_'
        process_params = parse_process_conditions_from_filename(lib_prefix, is_test=True)
        corner_label = f"{prefix}_{corner[0]}_{corner[1]}_{corner[2]}"

        print(f'   [{dir_idx+1}/{len(test_folders)}] {folder.name} '
              f'(corner={corner_label}, temp={temperature})', flush=True)

        all_samples_per_lib, num_tasks_dir = _process_directory_asap7(
            folder, topology_cache, cache_type, process_params, data_type,
            voltage_mode=voltage_mode, slew_mode=slew_mode,
        )
        print(f'      ✓ {num_tasks_dir} tasks from {len(lib_files_check)} lib files')

        folder_samples_by_cell = defaultdict(list)
        for task_idx in range(num_tasks_dir):
            samples_by_lib = {}
            cell_name = None
            valid = True
            for lib_idx in range(len(lib_files_check)):
                if (lib_idx < len(all_samples_per_lib) and
                        task_idx < len(all_samples_per_lib[lib_idx])):
                    s = all_samples_per_lib[lib_idx][task_idx]
                    samples_by_lib[lib_idx] = s
                    if cell_name is None:
                        cell_name = s['cell_name']
                else:
                    valid = False
                    break
            if valid and cell_name and len(samples_by_lib) == len(lib_files_check):
                folder_samples_by_cell[cell_name].append({
                    'corner': corner_label,
                    'temperature': temperature,
                    'samples_by_lib': samples_by_lib,
                })
                all_cell_names.add(cell_name)

        for cell_name, tasks in folder_samples_by_cell.items():
            partial_path = temp_dir / f'{cell_name}_partial_{dir_idx:04d}.pth'
            torch.save(tasks, partial_path)

        del all_samples_per_lib, folder_samples_by_cell
        gc.collect()

    print(f'\n   ✅ Processed {len(test_folders)} folders, found {len(all_cell_names)} unique cells')
    return all_cell_names


def _merge_partials_for_one_cell(
    cell_name, cell_idx, total_cells, temp_dir: Path, test_output_dir: Path,
    num_features, voltage_mode, slew_mode, topology_suffix,
    include_zeros_in_norm, max_test_tasks_per_cell, sampling_seed,
) -> bool:
    """Load all partials for `cell_name`, build the per-cell 2-D tensor and save it.
    Honors `max_test_tasks_per_cell` by capping per-corner task counts (with a
    deterministic per-cell rng for reproducibility). Returns True if saved.
    """
    partial_files = sorted(temp_dir.glob(f'{cell_name}_partial_*.pth'))
    if not partial_files:
        return False

    flat_records = []
    for pf in partial_files:
        recs = torch.load(pf, weights_only=False)
        flat_records.extend(recs)
        pf.unlink()
    if not flat_records:
        return False

    corners_seen = sorted({r['corner'] for r in flat_records})
    temps_seen = sorted({r['temperature'] for r in flat_records})

    ct_groups: Dict[Tuple[str, float], List[dict]] = defaultdict(list)
    for r in flat_records:
        ct_groups[(r['corner'], r['temperature'])].append(r)

    counts, missing_ct = [], []
    for c in corners_seen:
        for t in temps_seen:
            cnt = len(ct_groups.get((c, t), []))
            counts.append(cnt)
            if cnt == 0:
                missing_ct.append((c, t))
    if missing_ct:
        print(f'   [{cell_idx+1}/{total_cells}] {cell_name}: '
              f'missing (corner, temp) groups {missing_ct} — skipping cell')
        return False

    tasks_per_corner = min(counts)
    if len(set(counts)) > 1:
        print(f'   [{cell_idx+1}/{total_cells}] {cell_name}: '
              f'task-count mismatch {sorted(set(counts))}, using min={tasks_per_corner}')

    if max_test_tasks_per_cell is not None and tasks_per_corner > 0:
        cap_per_corner = max(1, int(round(max_test_tasks_per_cell / max(len(corners_seen), 1))))
        tasks_per_corner = min(tasks_per_corner, cap_per_corner)

    num_corners = len(corners_seen)
    num_temps = len(temps_seen)
    num_libs_cell = len(flat_records[0]['samples_by_lib'])
    num_tasks_cell = num_corners * tasks_per_corner

    # Random per-corner subselection (deterministic per cell)
    if max_test_tasks_per_cell is not None:
        local_rng = random.Random(sampling_seed + cell_idx)
        chosen_per_corner = {}
        for c in corners_seen:
            full_n = min(len(ct_groups[(c, t)]) for t in temps_seen)
            idx_list = list(range(full_n))
            local_rng.shuffle(idx_list)
            chosen_per_corner[c] = sorted(idx_list[:tasks_per_corner])
    else:
        chosen_per_corner = {c: list(range(tasks_per_corner)) for c in corners_seen}

    task_index_list = [(corner, ti) for corner in corners_seen for ti in chosen_per_corner[corner]]

    task_node_counts, delay_types, output_names, task_corners_out = [], [], [], []
    for (corner, ti) in task_index_list:
        sample0 = ct_groups[(corner, temps_seen[0])][ti]['samples_by_lib'][0]
        task_node_counts.append(sample0['num_nodes'])
        delay_types.append(sample0['delay_type'])
        output_names.append(sample0['output_name'])
        task_corners_out.append(corner)

    total_nodes_cell = sum(task_node_counts)
    node_slices = np.zeros(num_tasks_cell + 1, dtype=np.int64)
    node_slices[1:] = np.cumsum(task_node_counts)

    node_features = np.zeros((num_libs_cell, num_temps, total_nodes_cell, num_features),
                              dtype=np.float32)
    outputs = np.zeros((num_libs_cell, num_temps, num_tasks_cell), dtype=np.float32)

    for task_out_idx, (corner, ti) in enumerate(task_index_list):
        node_start = node_slices[task_out_idx]
        node_end = node_slices[task_out_idx + 1]
        for temp_idx, temp in enumerate(temps_seen):
            record = ct_groups[(corner, temp)][ti]
            samples_by_lib = record['samples_by_lib']
            for lib_idx, sample in samples_by_lib.items():
                nf = sample['node_features']
                if isinstance(nf, torch.Tensor):
                    nf = nf.cpu().numpy()
                node_features[lib_idx, temp_idx, node_start:node_end, :] = nf
                out = sample['output']
                if isinstance(out, torch.Tensor):
                    out = out.item()
                outputs[lib_idx, temp_idx, task_out_idx] = out

    voltages = _extract_voltage_axis(node_features, num_libs_cell)
    temperatures_arr = np.array(temps_seen, dtype=np.float32)

    cell_path = test_output_dir / f'{cell_name}.pth'
    cell_data = {
        'node_features': torch.from_numpy(node_features),
        'outputs': torch.from_numpy(outputs),
        'node_slices': torch.from_numpy(node_slices),
        'voltages': torch.from_numpy(voltages),
        'temperatures': torch.from_numpy(temperatures_arr),
        'delay_types': delay_types,
        'output_names': output_names,
        'task_corners': task_corners_out,
        'cell_name': cell_name,
        'num_tasks': num_tasks_cell,
        'num_voltages': num_libs_cell,
        'num_temps': num_temps,
        'num_features': num_features,
        'total_nodes': total_nodes_cell,
        'format': 'unified_4d_VT',
        'process_node': 'ASAP7',
        'voltage_mode': voltage_mode,
        'slew_mode': slew_mode,
        'topology_suffix': topology_suffix,
        'normalize_nonzero_only': not include_zeros_in_norm,
        'include_zeros_in_norm': include_zeros_in_norm,
        'max_test_tasks_per_cell': max_test_tasks_per_cell,
    }
    torch.save(cell_data, cell_path)

    del node_features, outputs, flat_records, ct_groups
    gc.collect()
    return True


def _merge_all_test_partials(
    all_cell_names: set, temp_dir: Path, test_output_dir: Path,
    num_features, voltage_mode, slew_mode, topology_suffix,
    include_zeros_in_norm, max_test_tasks_per_cell, sampling_seed,
) -> int:
    print(f'\n📦 Step 2: Merging partials into per-cell 2-D files...')
    print(f'   Output directory: {test_output_dir}')

    saved_count = 0
    sorted_cells = sorted(all_cell_names)
    for cell_idx, cell_name in enumerate(sorted_cells):
        ok = _merge_partials_for_one_cell(
            cell_name, cell_idx, len(sorted_cells), temp_dir, test_output_dir,
            num_features, voltage_mode, slew_mode, topology_suffix,
            include_zeros_in_norm, max_test_tasks_per_cell, sampling_seed,
        )
        if ok:
            saved_count += 1
        if (cell_idx + 1) % 10 == 0:
            print(f'   Processed {cell_idx+1}/{len(sorted_cells)} cells')

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    print(f'   ✅ Saved {saved_count} cell files')
    return saved_count


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def build_unified_datasets_2d_asap7(
    cache_path: str,
    cache_type: str,
    lib_base_paths: List[str],
    output_dir: str,
    prefixes: List[str],
    data_type: str = 'cell',
    skip_train: bool = False,
    voltage_mode: str = 'all_nodes',
    include_zeros_in_norm: bool = False,
    topology_suffix: str = '',
    slew_mode: str = 'all',
    sampling_ratio: float = 1.0,
    max_test_tasks_per_cell: Optional[int] = None,
    sampling_seed: int = 0,
):
    """
    Build the 2-D V×T train + per-cell test datasets for ASAP7.
    """
    num_features = 11
    num_train_temps = len(TRAIN_TEMPERATURES)

    _print_config_banner(
        cache_path, cache_type, lib_base_paths, output_dir, prefixes, data_type,
        voltage_mode, slew_mode, include_zeros_in_norm, topology_suffix,
        sampling_ratio, max_test_tasks_per_cell, sampling_seed,
    )

    print(f'\n📦 Loading topology cache...')
    topology_cache = torch.load(cache_path, weights_only=False)
    print(f'   ✓ Loaded {len(topology_cache)} cells')

    lib_base_paths_p = [Path(p) for p in lib_base_paths]
    output_dir_p, mode_suffix = _setup_paths_and_suffix(
        output_dir, voltage_mode, slew_mode, topology_suffix,
    )

    train_folders = enumerate_train_folders(lib_base_paths_p, prefixes)
    test_folders = enumerate_test_folders(lib_base_paths_p, prefixes)
    print(f'\n🔍 Discovered train folders: {len(train_folders)}')
    print(f'🔍 Discovered test  folders: {len(test_folders)}')
    if not train_folders and not skip_train:
        raise RuntimeError(f'No train folders found under {lib_base_paths} for prefixes {prefixes}')

    rng = random.Random(sampling_seed)

    # ----- Train -----
    train_path: Optional[Path] = None
    if not skip_train:
        print(f'\n{"=" * 80}')
        print(f'PROCESSING TRAIN DATA (2-D V×T)')
        print(f'{"=" * 80}')

        all_train_tasks, num_voltages_observed = _collect_all_train_tasks(
            train_folders, topology_cache, cache_type, data_type,
            voltage_mode, slew_mode, sampling_ratio, rng,
        )
        if not all_train_tasks:
            print('❌ No valid train tasks!')
            return

        train_tensors = _allocate_and_fill_train_tensors(
            all_train_tasks, num_voltages_observed, num_train_temps, num_features,
        )
        norm_stats, normalize_indices, normalize_names = _compute_normalization_stats(
            train_tensors['node_features'], include_zeros_in_norm,
        )
        train_path = _save_train_dataset(
            train_tensors, norm_stats, normalize_indices, normalize_names,
            output_dir_p, data_type, cache_type, mode_suffix,
            cache_path, num_voltages_observed, num_train_temps, num_features,
            voltage_mode, slew_mode, topology_suffix, sampling_ratio, sampling_seed,
            include_zeros_in_norm,
        )
        del train_tensors, all_train_tasks
        gc.collect()
    else:
        print('\nSKIPPING TRAIN (--skip_train)')

    # ----- Test -----
    print(f'\n{"=" * 80}')
    print('PROCESSING TEST DATA (2-D V×T)')
    print(f'{"=" * 80}')

    test_output_dir = output_dir_p / f'test_by_{data_type}_{cache_type}{mode_suffix}_2d'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = test_output_dir / '.temp_partials'
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_cell_names = _extract_test_partials(
        test_folders, topology_cache, cache_type, data_type, temp_dir,
        voltage_mode, slew_mode,
    )
    _merge_all_test_partials(
        all_cell_names, temp_dir, test_output_dir,
        num_features, voltage_mode, slew_mode, topology_suffix,
        include_zeros_in_norm, max_test_tasks_per_cell, sampling_seed,
    )

    # ----- Summary -----
    print(f'\n{"=" * 80}')
    print('SUMMARY')
    print(f'{"=" * 80}')
    if not skip_train and train_path is not None:
        print(f'Train: {train_path}')
    print(f'Test:  {test_output_dir}')
    print(f'\n✅ Done!')


def main():
    parser = argparse.ArgumentParser(
        description='Build ASAP7 GNN dataset (2-D V×T format)',
    )
    parser.add_argument('--cache_path', type=str, required=True,
                        help='Path to topology cache')
    parser.add_argument('--cache_type', type=str, required=True,
                        choices=['stage_aware', 'full_graph'])
    parser.add_argument('--lib_base_paths', type=str, nargs='+', required=True,
                        help='One or more lib root dirs '
                             '(e.g. .../processed .../processed_simple).')
    parser.add_argument('--test_lib_base_paths', type=str, nargs='+', default=None,
                        help='Optional separate lib root dirs for test (e.g. test_processed). '
                             'If omitted, the test folders are looked up under --lib_base_paths.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prefixes', type=str, nargs='+', default=['invbuf'],
                        help='Cell-type prefixes used in folder names. '
                             'Default: invbuf. Use e.g. `invbuf simple` for both.')
    parser.add_argument('--data_type', type=str, default='cell',
                        choices=['cell', 'transition'])
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--voltage_mode', type=str, default='all_nodes',
                        choices=['all_nodes', 'vdd_only', 'vdd_mos'])
    parser.add_argument('--slew_mode', type=str, default='all',
                        choices=['all', 'related_pin_only'])
    parser.add_argument('--include_zeros_in_norm', action='store_true')
    parser.add_argument('--topology_suffix', type=str, default='')
    parser.add_argument('--sampling_ratio', type=float, default=1.0,
                        help='Fraction of V×T-aligned train tasks to keep per corner '
                             '(0.10 ≈ "10pct" 1-D parity). Default 1.0 = keep all.')
    parser.add_argument('--max_test_tasks_per_cell', type=int, default=None,
                        help='Per-cell test cap. None = no cap. e.g. 2000.')
    parser.add_argument('--sampling_seed', type=int, default=0)

    args = parser.parse_args()

    # Train uses lib_base_paths. Test optionally swaps to test_lib_base_paths.
    # We implement test path swap by calling enumerate_test_folders on those dirs:
    # but to keep main flow simple, we delegate by running build with the test paths
    # appended when --test_lib_base_paths is given. The orchestration already only
    # picks folders whose names match the expected test corner grid (3×3×3 indices),
    # so train folders in the same base won't be picked as test or vice versa.
    lib_base_paths_all = list(args.lib_base_paths)
    if args.test_lib_base_paths:
        for p in args.test_lib_base_paths:
            if p not in lib_base_paths_all:
                lib_base_paths_all.append(p)

    build_unified_datasets_2d_asap7(
        cache_path=args.cache_path,
        cache_type=args.cache_type,
        lib_base_paths=lib_base_paths_all,
        output_dir=args.output_dir,
        prefixes=args.prefixes,
        data_type=args.data_type,
        skip_train=args.skip_train,
        voltage_mode=args.voltage_mode,
        include_zeros_in_norm=args.include_zeros_in_norm,
        topology_suffix=args.topology_suffix,
        slew_mode=args.slew_mode,
        sampling_ratio=args.sampling_ratio,
        max_test_tasks_per_cell=args.max_test_tasks_per_cell,
        sampling_seed=args.sampling_seed,
    )


if __name__ == '__main__':
    main()
