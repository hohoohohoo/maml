#!/usr/bin/env python
"""
Build TSMC GNN dataset with unified 3D tensor format.

Train: 5 corners (FF, FS, TT, SF, SS) × 6 temps (-25, 12.5, 37.5, 62.5, 87.5, 125) = 30 conditions
       All tasks combined into [61, total_nodes, num_features] format
Test: temps (0, 25, 50, 75, 100) + FF2, TTseq, etc. variants → grouped by cell_name as separate npy files

Node features:
- Standard: 11D (7 base + 4 process parameters)
- With parasitic cap: 12D (7 base + 4 process + 1 parasitic cap)
  - Parasitic cap is already normalized (mean/std per cell) in topology cache

Output format:
- Train: train_cell_stage_aware.pth with [61, total_nodes, num_features] node_features
- Test: test_{cell_name}_stage_aware.npy files

Usage:
    # Standard (11D features)
    python build_gnn_dataset_tsmc_unified.py \
        --cache_path topology_cache/stage_aware_topology_cache_tsmc.pth \
        --cache_type stage_aware \
        --lib_base_path /path/to/TSMC_lib_files \
        --output_dir /path/to/output \
        --data_type cell

    # With parasitic capacitance (12D features)
    # Requires topology cache generated with --weighted option
    python build_gnn_dataset_tsmc_unified.py \
        --cache_path topology_cache/stage_aware_topology_cache_tsmc_weighted.pth \
        --cache_type stage_aware \
        --lib_base_path /path/to/TSMC_lib_files \
        --output_dir /path/to/output \
        --data_type cell \
        --include_parasitic_cap
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import argparse
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'precompute'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from precompute_full_graph_topology import apply_topology_to_sample
from precompute_stage_aware_topology import apply_stage_aware_topology


# Fixed process parameters (FF_n, FF_p, TT_n, TT_p, SS_n, SS_p, FS_n, FS_p, SF_n, SF_p)
FIXED_PARAM_A = "1.427,1.457,1.430,1.470,1.443,1.483,1.43,1.47,1.43,1.47"
FIXED_PARAM_B = "0.026,0.045,0,0,-0.026,-0.05,0.0208,-0.04,0.036,-0.0208"
FIXED_PARAM_C = "0.024,2.000,0.024,2.000,0.024,2.000,0.024,2.000,0.024,2.000"

# Train/Test temperature split
TRAIN_TEMPERATURES = [-25, 12.5, 37.5, 62.5, 87.5, 125]  # 6 temps for train
TEST_TEMPERATURES = [0, 25, 50, 75, 100]  # 5 temps for test

# Base corners for train (standard corners only)
TRAIN_CORNERS = ['FF', 'FS', 'TT', 'SF', 'SS']

# Intra topology cells - these cells are used for intra-topology evaluation
# and should be excluded from training when --exclude_intra_topology_cells is set
INTRA_TOPOLOGY_CELLS = [
    'AN4D0BWP30P140',
    'ND3D0BWP30P140',
    'NR3D1BWP30P140',
    'OR4D0BWP30P140',
    'XNR3D1BWP30P140',
    'XOR3D1BWP30P140',
]


def parse_tsmc_folder_name(folder_name: str) -> Tuple[Optional[str], Optional[float], bool]:
    """
    Parse TSMC folder name to extract corner, temperature, and variant info.
    """
    # Check for variant patterns (FF2, TT2, TTseq, etc.) - these go to test
    pattern_2 = r'TSMC_([A-Z]+)2_(-?\d+(?:p\d+)?)'
    match = re.match(pattern_2, folder_name)
    if match:
        corner = match.group(1)
        temp_str = match.group(2).replace('p', '.')
        temperature = float(temp_str)
        return corner, temperature, True

    pattern_seq = r'TSMC_([A-Z]+)seq_(-?\d+(?:p\d+)?)'
    match = re.match(pattern_seq, folder_name)
    if match:
        corner = match.group(1)
        temp_str = match.group(2).replace('p', '.')
        temperature = float(temp_str)
        return corner, temperature, True

    pattern_seq2 = r'TSMC_Seq_([A-Z]+)_(-?\d+(?:p\d+)?)'
    match = re.match(pattern_seq2, folder_name)
    if match:
        corner = match.group(1)
        temp_str = match.group(2).replace('p', '.')
        temperature = float(temp_str)
        return corner, temperature, True

    # # Standard TSMC_{CORNER}_{TEMP} pattern
    pattern = r'TSMC_([A-Z]+)_(-?\d+(?:p\d+)?)'
    match = re.match(pattern, folder_name)
    if match:
        corner = match.group(1)
        temp_str = match.group(2).replace('p', '.')
        temperature = float(temp_str)
        return corner, temperature, False

    return None, None, False


def get_abc_parameters(corner: str, temperature: float) -> Dict[str, float]:
    """Map corner and temperature to a,b,c parameters."""
    param_a_list = [float(x.strip()) for x in FIXED_PARAM_A.split(',')]
    param_b_list = [float(x.strip()) for x in FIXED_PARAM_B.split(',')]
    param_c_list = [float(x.strip()) for x in FIXED_PARAM_C.split(',')]

    corner_to_idx = {
        'FF': 0, 'TT': 1, 'SS': 2, 'FS': 3, 'SF': 4,
    }

    corner_idx = corner_to_idx.get(corner.upper(), 1)
    nmos_idx = corner_idx * 2
    pmos_idx = corner_idx * 2 + 1

    a_nmos = param_a_list[nmos_idx] if nmos_idx < len(param_a_list) else param_a_list[0]
    a_pmos = param_a_list[pmos_idx] if pmos_idx < len(param_a_list) else param_a_list[1]
    b_nmos = param_b_list[nmos_idx] if nmos_idx < len(param_b_list) else param_b_list[0]
    b_pmos = param_b_list[pmos_idx] if pmos_idx < len(param_b_list) else param_b_list[1]
    c_nmos = param_c_list[nmos_idx] if nmos_idx < len(param_c_list) else param_c_list[0]
    c_pmos = param_c_list[pmos_idx] if pmos_idx < len(param_c_list) else param_c_list[1]

    return {
        'param_a_nmos': a_nmos, 'param_a_pmos': a_pmos,
        'param_b_nmos': b_nmos, 'param_b_pmos': b_pmos,
        'param_c_nmos': c_nmos, 'param_c_pmos': c_pmos,
        'temperature': float(temperature)
    }


def apply_topology_with_process_tsmc(topology_cache, cache_type, cell_name, output_name, delay_type,
                                      voltage, input_slew, output_load, input_port_names, process_params,
                                      include_parasitic_cap=False, voltage_mode='all_nodes',
                                      temperature_mode='mos_only', slew_mode='all',
                                      related_pin=None):
    """
    Apply topology from cache and add TSMC process parameters to node features.

    Args:
        topology_cache: Pre-computed topology cache
        cache_type: 'stage_aware' or 'full_graph'
        cell_name: Cell name
        output_name: Output port name
        delay_type: Delay type ('rise_transition' or 'fall_transition')
        voltage: Voltage value
        input_slew: Input slew value
        output_load: Output load value
        input_port_names: List of input port names
        process_params: Process parameter dictionary
        include_parasitic_cap: If True, add parasitic capacitance sum as additional feature (12D instead of 11D)
        voltage_mode: 'all_nodes' (default) applies voltage to all nodes,
                      'vdd_only' applies voltage only to VDD node (0 for others)
        temperature_mode: 'mos_only' (default) applies temperature to MOS nodes only,
                          'temp_all' applies temperature to all nodes
        slew_mode: 'all' (default) applies input_slew to all input ports/connected MOS,
                   'related_pin_only' applies input_slew only to related_pin node/connected MOS
        related_pin: The specific input pin that triggered the timing arc (used when slew_mode='related_pin_only')

    Returns:
        graph_sample with enhanced node features
    """
    if cache_type == 'stage_aware':
        graph_sample = apply_stage_aware_topology(
            topology_cache, cell_name, output_name, delay_type,
            voltage, input_slew, output_load, input_port_names,
            voltage_mode=voltage_mode, slew_mode=slew_mode,
            related_pin=related_pin
        )
    else:
        graph_sample = apply_topology_to_sample(
            topology_cache, cell_name, voltage, input_slew, output_load,
            output_value=0.0, input_port_names=input_port_names,
            voltage_mode=voltage_mode, slew_mode=slew_mode,
            related_pin=related_pin
        )

    base_node_features = graph_sample['node_features']
    num_nodes = base_node_features.shape[0]

    # Process features: 4D (param_a, param_b, param_c, temperature)
    process_features = torch.zeros(num_nodes, 4, dtype=torch.float32)

    cell_cache = topology_cache[cell_name]
    transistor_info = cell_cache.get('transistor_info', {})

    # Process parameters (param_a, param_b, param_c) are only added to MOS (transistor) nodes
    # Non-transistor nodes (VDD, GND, internal nets) remain with zeros for param_a/b/c
    # Temperature can be applied to all nodes (temp_all) or only MOS nodes (mos_only)
    for node_idx, node_name in enumerate(graph_sample.get('all_nodes', [])):
        if node_name in transistor_info:
            trans_info = transistor_info[node_name]
            trans_type = trans_info['type']

            if trans_type > 0:  # NMOS
                process_features[node_idx, 0] = process_params['param_a_nmos']
                process_features[node_idx, 1] = process_params['param_b_nmos']
                process_features[node_idx, 2] = process_params['param_c_nmos']
            else:  # PMOS
                process_features[node_idx, 0] = process_params['param_a_pmos']
                process_features[node_idx, 1] = process_params['param_b_pmos']
                process_features[node_idx, 2] = process_params['param_c_pmos']
            # MOS nodes always get temperature
            process_features[node_idx, 3] = process_params['temperature']
        else:
            # Non-MOS nodes: param_a/b/c remain zeros
            # Temperature depends on temperature_mode
            if temperature_mode == 'temp_all':
                process_features[node_idx, 3] = process_params['temperature']
            # For 'mos_only' mode, temperature remains 0 for non-MOS nodes

    # Optionally add parasitic capacitance feature
    if include_parasitic_cap:
        node_capacitance = cell_cache.get('node_capacitance', {})
        cap_features = torch.zeros(num_nodes, 1, dtype=torch.float32)

        # NOTE: node_capacitance values are already normalized (mean/std) in topology cache
        # No additional scaling needed

        for node_idx, node_name in enumerate(graph_sample.get('all_nodes', [])):
            # Try exact match first
            if node_name in node_capacitance:
                cap_features[node_idx, 0] = node_capacitance[node_name]
            else:
                # For transistor nodes (XM1, XM2), try terminal formats (M1:DRN, M1:GATE, etc.)
                if node_name.startswith('XM'):
                    mos_name = node_name.replace('X', '')  # XM1 -> M1
                    total_cap = 0.0
                    for suffix in ['DRN', 'SRC', 'GATE', 'BULK']:
                        terminal = f"{mos_name}:{suffix}"
                        if terminal in node_capacitance:
                            total_cap += node_capacitance[terminal]
                    cap_features[node_idx, 0] = total_cap

        # Enhanced: 7 base + 4 process + 1 parasitic cap = 12D
        enhanced_node_features = torch.cat([base_node_features, process_features, cap_features], dim=1)
    else:
        # Standard: 7 base + 4 process = 11D
        enhanced_node_features = torch.cat([base_node_features, process_features], dim=1)

    graph_sample['node_features'] = enhanced_node_features
    graph_sample['process_params'] = process_params
    return graph_sample


def temp_to_folder_str(temp: float) -> str:
    """Convert temperature value to folder name string format."""
    if temp == int(temp):
        return str(int(temp))
    else:
        return str(temp).replace('.', 'p')


def get_expected_train_folders(lib_base_path: Path) -> Tuple[List[Path], List[str]]:
    """Get expected train folders (5 corners × 6 temps = 30)."""
    existing_folders = []
    missing_folders = []

    for corner in TRAIN_CORNERS:
        for temp in TRAIN_TEMPERATURES:
            temp_str = temp_to_folder_str(temp)
            folder_name = f"TSMC_{corner}_{temp_str}"
            folder_path = lib_base_path / folder_name

            if folder_path.exists() and folder_path.is_dir():
                existing_folders.append(folder_path)
            else:
                missing_folders.append(folder_name)

    return existing_folders, missing_folders


def get_test_folders(lib_base_path: Path) -> List[Path]:
    """Get test folders."""
    test_folders = []

    # Temporarily commented out - only processing seq folders
    for corner in TRAIN_CORNERS:
        for temp in TEST_TEMPERATURES:
            temp_str = temp_to_folder_str(temp)
            folder_name = f"TSMC_{corner}_{temp_str}"
            folder_path = lib_base_path / folder_name
    
            if folder_path.exists() and folder_path.is_dir():
                test_folders.append(folder_path)

    all_folders = sorted([f for f in lib_base_path.iterdir() if f.is_dir() and f.name.startswith('TSMC_')])

    for folder in all_folders:
        corner, temperature, is_variant = parse_tsmc_folder_name(folder.name)
        if is_variant and folder not in test_folders:
            test_folders.append(folder)

    return sorted(test_folders, key=lambda x: x.name)


def process_lib_file_for_unified(lib_file_path, topology_cache, cache_type, process_params,
                                  data_type="cell", include_parasitic_cap=False, voltage_mode='all_nodes',
                                  temperature_mode='mos_only', slew_mode='all'):
    """
    Process a single .lib file and return samples as a list (preserves all data).

    Uses the same approach as build_gnn_dataset_stage_aware_cached_tsmc.py:
    - Appends all samples to a list (no deduplication)
    - Same processing order ensures alignment across lib files

    Args:
        lib_file_path: Path to .lib file
        topology_cache: Pre-computed topology cache
        cache_type: 'stage_aware' or 'full_graph'
        process_params: Process parameter dictionary
        data_type: 'cell' or 'transition'
        include_parasitic_cap: If True, add parasitic cap feature (12D instead of 11D)
        voltage_mode: 'all_nodes', 'vdd_only', or 'vdd_mos'
        temperature_mode: 'mos_only' or 'temp_all'
        slew_mode: 'all' (default) applies input_slew to all input ports/connected MOS,
                   'related_pin_only' applies input_slew only to related_pin node/connected MOS

    Returns:
        List[sample]: List of samples in deterministic order
    """
    if data_type == 'cell':
        from libdata_extract_MAML_cell import parse_liberty_pin_blocks, flatten_pin_data
    else:
        from libdata_extract_MAML_transition import parse_liberty_pin_blocks, flatten_pin_data

    with open(lib_file_path, "r") as f:
        lines = f.readlines()

    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)

    # Use list to preserve all samples (same as build_gnn_dataset_stage_aware_cached_tsmc.py)
    minimal_samples = []

    for sample in flattened:
        cell_name = sample['cell']
        voltage = sample['Voltage']
        delay_type = sample['delay_type']
        input_port_names = sample['input_port_name']
        output_port_name = sample.get('pin_name', 'Z')  # TSMC typically uses 'Z' or 'ZN'
        # Get related_pin for slew_mode='related_pin_only'
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
        actual_cols = len(timing_values[0]) if actual_rows > 0 and isinstance(timing_values[0], list) else 0
        effective_rows = min(len(input_slews), actual_rows) if actual_rows > 0 else len(input_slews)
        effective_cols = min(len(output_loads), actual_cols) if actual_cols > 0 else len(output_loads)

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

                    graph_sample = apply_topology_with_process_tsmc(
                        topology_cache, cache_type, cell_name, output_port_name, stage_delay_type,
                        voltage, input_slew, output_load, input_port_names, process_params,
                        include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
                        temperature_mode=temperature_mode, slew_mode=slew_mode,
                        related_pin=related_pin
                    )

                    expected_features = 12 if include_parasitic_cap else 11
                    if graph_sample['node_features'].shape[1] != expected_features:
                        continue

                    # Append to list (preserves all samples including different states)
                    minimal_samples.append({
                        'node_features': graph_sample['node_features'],
                        'output': output_value,
                        'cell_name': cell_name,
                        'delay_type': stage_delay_type,
                        'output_name': output_port_name,
                        'num_nodes': graph_sample['node_features'].shape[0]
                    })

                except Exception as e:
                    # Try alternative output name (Z vs ZN)
                    alt_output = 'ZN' if output_port_name == 'Z' else 'Z'
                    try:
                        graph_sample = apply_topology_with_process_tsmc(
                            topology_cache, cache_type, cell_name, alt_output, stage_delay_type,
                            voltage, input_slew, output_load, input_port_names, process_params,
                            include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
                            temperature_mode=temperature_mode, slew_mode=slew_mode,
                            related_pin=related_pin
                        )
                        expected_features = 12 if include_parasitic_cap else 11
                        if graph_sample['node_features'].shape[1] == expected_features:
                            minimal_samples.append({
                                'node_features': graph_sample['node_features'],
                                'output': output_value,
                                'cell_name': cell_name,
                                'delay_type': stage_delay_type,
                                'output_name': alt_output,
                                'num_nodes': graph_sample['node_features'].shape[0]
                            })
                    except Exception as e2:
                        pass  # Skip this sample

    return minimal_samples


def process_directory_for_unified(folder_path, topology_cache, cache_type, process_params,
                                   data_type="cell", include_parasitic_cap=False, voltage_mode='all_nodes',
                                   temperature_mode='mos_only', slew_mode='all'):
    """
    Process all lib files in a directory and return unified samples.

    Each directory has 61 lib files with different voltages.
    Uses list-based approach (same as build_gnn_dataset_stage_aware_cached_tsmc.py).
    Same processing order ensures alignment across lib files.

    Args:
        folder_path: Path to directory containing .lib files
        topology_cache: Pre-computed topology cache
        cache_type: 'stage_aware' or 'full_graph'
        process_params: Process parameter dictionary
        data_type: 'cell' or 'transition'
        include_parasitic_cap: If True, add parasitic cap feature (12D instead of 11D)
        voltage_mode: 'all_nodes', 'vdd_only', or 'vdd_mos'
        temperature_mode: 'mos_only' or 'temp_all'
        slew_mode: 'all' (default) or 'related_pin_only'

    Returns:
        all_samples_per_lib: List[List[sample]] - samples per lib file
        num_tasks: int - number of tasks (samples per lib file)
    """
    lib_files = sorted(folder_path.glob("*.lib"))
    if not lib_files:
        return [], 0

    num_libs = len(lib_files)
    all_samples_per_lib = []

    # Process all lib files
    for lib_idx, lib_file in enumerate(lib_files):
        lib_samples = process_lib_file_for_unified(
            str(lib_file), topology_cache, cache_type, process_params, data_type,
            include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
            temperature_mode=temperature_mode, slew_mode=slew_mode
        )
        all_samples_per_lib.append(lib_samples)

    # Verify all lib files have the same number of samples
    num_tasks = len(all_samples_per_lib[0]) if all_samples_per_lib else 0
    sample_counts = [len(samples) for samples in all_samples_per_lib]

    if len(set(sample_counts)) > 1:
        print(f"   ⚠️ WARNING: Sample count mismatch across lib files!")
        print(f"      Expected: {num_tasks}, Unique counts: {sorted(set(sample_counts))}")
        # Show which libs have different counts
        for lib_idx, count in enumerate(sample_counts):
            if count != num_tasks:
                lib_name = lib_files[lib_idx].name
                print(f"      Lib {lib_idx} ({lib_name}): {count} samples")
                # Check cell name difference
                if count > 0 and num_tasks > 0:
                    lib0_cells = [s['cell_name'] for s in all_samples_per_lib[0][:5]]
                    this_lib_cells = [s['cell_name'] for s in all_samples_per_lib[lib_idx][:5]]
                    print(f"        First 5 cells in lib0: {lib0_cells}")
                    print(f"        First 5 cells in lib{lib_idx}: {this_lib_cells}")

    return all_samples_per_lib, num_tasks


def build_unified_datasets(
    cache_path: str,
    cache_type: str,
    lib_base_path: str,
    output_dir: str,
    data_type: str = "cell",
    skip_train: bool = False,
    include_parasitic_cap: bool = False,
    voltage_mode: str = 'all_nodes',
    temperature_mode: str = 'mos_only',
    include_zeros_in_norm: bool = False,
    topology_suffix: str = "",
    slew_mode: str = 'all'
):
    """
    Build train and test datasets with unified 3D tensor format.

    Train: [61, total_nodes, num_features] format, all directories combined
    Test: Separate npy files per cell_name

    Args:
        cache_path: Path to topology cache
        cache_type: 'stage_aware' or 'full_graph'
        lib_base_path: Base path to TSMC library files
        output_dir: Output directory
        data_type: 'cell' or 'transition'
        skip_train: Skip train data processing
        include_parasitic_cap: Include parasitic capacitance as additional feature (12D vs 11D)
        voltage_mode: 'all_nodes' applies voltage to all nodes (default),
                      'vdd_only' applies voltage only to VDD node (0 for others)
        temperature_mode: 'mos_only' applies temperature to MOS nodes only (default),
                          'temp_all' applies temperature to all nodes
        include_zeros_in_norm: If True, include zeros when calculating normalization statistics (original method).
                               If False (default), exclude zeros from stats (norm2 method).
        slew_mode: 'all' (default) applies input_slew to all input ports/connected MOS,
                   'related_pin_only' applies input_slew only to related_pin node/connected MOS
    """
    num_features = 12 if include_parasitic_cap else 11
    print("=" * 80)
    print("BUILDING TSMC GNN DATASET - UNIFIED 3D FORMAT")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Cache type: {cache_type}")
    print(f"Lib base path: {lib_base_path}")
    print(f"Output dir: {output_dir}")
    print(f"Data type: {data_type}")
    print(f"Include parasitic cap: {include_parasitic_cap}")
    print(f"Voltage mode: {voltage_mode}")
    print(f"Temperature mode: {temperature_mode}")
    print(f"Slew mode: {slew_mode}")
    print(f"Include zeros in norm: {include_zeros_in_norm} ({'original' if include_zeros_in_norm else 'norm2'})")
    print(f"Topology suffix: '{topology_suffix}'")
    print(f"Node features: {num_features}D ({'7 base + 4 process + 1 cap' if include_parasitic_cap else '7 base + 4 process'})")
    print(f"\nTrain temperatures: {TRAIN_TEMPERATURES}")
    print(f"Test temperatures: {TEST_TEMPERATURES}")
    print(f"Train corners: {TRAIN_CORNERS}")
    print(f"\n⚠️  Excluding INTRA_TOPOLOGY_CELLS from train data:")
    for cell in INTRA_TOPOLOGY_CELLS:
        print(f"   - {cell}")
    print("=" * 80)

    # Load topology cache
    print(f"\n📦 Loading topology cache...")
    topology_cache = torch.load(cache_path, weights_only=False)
    print(f"   ✓ Loaded {len(topology_cache)} cells")

    lib_base_path = Path(lib_base_path)
    output_dir = Path(output_dir)

    # Use subdirectory for parasitic cap version to avoid overwriting standard datasets
    if include_parasitic_cap:
        output_dir = output_dir / "with_parasitic_cap"
        print(f"   Using subdirectory for 12D features: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Voltage mode suffix for output filenames (only add suffix for non-default mode)
    voltage_suffix = f"_{voltage_mode}" if voltage_mode != 'all_nodes' else ""
    # Temperature mode suffix for output filenames (only add suffix for non-default mode)
    temperature_suffix = f"_{temperature_mode}" if temperature_mode != 'mos_only' else ""
    # Slew mode suffix for output filenames (only add suffix for non-default mode)
    slew_suffix = "_relpin" if slew_mode == 'related_pin_only' else ""
    # Combined suffix for output filenames (includes topology options like _weighted, _inputport, _gatectrl)
    mode_suffix = f"{topology_suffix}{voltage_suffix}{temperature_suffix}{slew_suffix}"

    # Check expected train folders — only enforced when we actually use them.
    # With --skip_train the train-folder discovery is informational; missing
    # folders should not abort the build (e.g., test-only PT corpora like the
    # _x10 seq libs do not ship comb train folders).
    if not skip_train:
        print(f"\n🔍 Checking expected train folders (5 corners × 6 temps = 30)...")
        train_folders, missing_train = get_expected_train_folders(lib_base_path)

        if missing_train:
            print(f"\n❌ ERROR: Missing {len(missing_train)} train folders!")
            for folder_name in missing_train:
                print(f"      - {folder_name}")
            raise FileNotFoundError(f"Missing {len(missing_train)} required train folders")

        print(f"   ✅ All 30 train folders found!")
    else:
        print(f"\n(skip_train=True — train-folder discovery skipped)")
        train_folders = []

    # Get test folders
    test_folders = get_test_folders(lib_base_path)
    print(f"   Found {len(test_folders)} test folders")

    # ==================== PROCESS TRAIN DATA ====================
    if not skip_train:
        print(f"\n{'='*80}")
        print("PROCESSING TRAIN DATA - UNIFIED 3D FORMAT")
        print(f"{'='*80}")

        # Collect all tasks from all directories using list-based approach
        # Same as build_gnn_dataset_stage_aware_cached_tsmc.py - preserves all samples
        all_train_tasks = []  # List of {'samples_per_lib': [...], 'metadata': {...}}

        num_libs = None  # Will be determined from first directory

        for dir_idx, folder in enumerate(train_folders):
            corner, temperature, _ = parse_tsmc_folder_name(folder.name)
            process_params = get_abc_parameters(corner, temperature)

            print(f"\n[{dir_idx+1}/{len(train_folders)}] Processing {folder.name} (corner={corner}, temp={temperature})")

            lib_files = sorted(folder.glob("*.lib"))
            if not lib_files:
                print(f"   ⚠️  No .lib files found, skipping...")
                continue

            if num_libs is None:
                num_libs = len(lib_files)
                print(f"   Number of lib files (voltages): {num_libs}")
            elif len(lib_files) != num_libs:
                print(f"   ⚠️  Warning: Expected {num_libs} lib files, found {len(lib_files)}")

            # Process all lib files and collect samples (list-based)
            all_samples_per_lib, num_tasks_dir = process_directory_for_unified(
                folder, topology_cache, cache_type, process_params, data_type,
                include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
                temperature_mode=temperature_mode, slew_mode=slew_mode
            )

            print(f"   ✓ {num_tasks_dir} tasks from {len(lib_files)} lib files")

            # Add tasks from this directory
            # Each task is indexed by position (same index = same timing entry across lib files)
            # Exclude INTRA_TOPOLOGY_CELLS from training data
            excluded_count = 0
            for task_idx in range(num_tasks_dir):
                samples_by_lib = {}
                valid = True
                for lib_idx in range(num_libs):
                    if lib_idx < len(all_samples_per_lib) and task_idx < len(all_samples_per_lib[lib_idx]):
                        samples_by_lib[lib_idx] = all_samples_per_lib[lib_idx][task_idx]
                    else:
                        valid = False
                        break

                if valid and len(samples_by_lib) == num_libs:
                    # Check if cell is in INTRA_TOPOLOGY_CELLS - exclude from train
                    cell_name = samples_by_lib[0]['cell_name']
                    if cell_name in INTRA_TOPOLOGY_CELLS:
                        excluded_count += 1
                        continue  # Skip this cell for training

                    all_train_tasks.append({
                        'dir_name': folder.name,
                        'corner': corner,
                        'temperature': temperature,
                        'samples_by_lib': samples_by_lib
                    })

            if excluded_count > 0:
                print(f"   ⚠️  Excluded {excluded_count} tasks from INTRA_TOPOLOGY_CELLS")

            gc.collect()

        print(f"\n📊 Total tasks collected: {len(all_train_tasks)}")
        print(f"   Number of libs (voltages): {num_libs}")

        if len(all_train_tasks) == 0:
            print("❌ No valid tasks found!")
            return

        # Calculate total nodes and verify consistency
        print(f"\n🔧 Validating task consistency...")

        # Get node counts for each task (should be same across all libs)
        task_node_counts = []
        cell_names = []
        delay_types = []
        output_names = []

        for task_info in all_train_tasks:
            samples_by_lib = task_info['samples_by_lib']
            sample = samples_by_lib[0]  # Use first lib's sample for metadata
            task_node_counts.append(sample['num_nodes'])
            cell_names.append(sample['cell_name'])
            delay_types.append(sample['delay_type'])
            output_names.append(sample['output_name'])

        total_nodes = sum(task_node_counts)
        num_tasks = len(all_train_tasks)
        # num_features already defined at function start

        print(f"   Total tasks: {num_tasks}")
        print(f"   Total nodes: {total_nodes}")
        print(f"   Num libs: {num_libs}")
        print(f"   Features: {num_features}")

        # Create node slices (shared across all libs)
        node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
        node_slices[1:] = np.cumsum(task_node_counts)

        # Allocate 3D tensors
        print(f"\n💾 Allocating 3D tensors...")
        print(f"   node_features: [{num_libs}, {total_nodes}, {num_features}]")
        print(f"   outputs: [{num_libs}, {num_tasks}]")

        all_node_features = np.zeros((num_libs, total_nodes, num_features), dtype=np.float32)
        all_outputs = np.zeros((num_libs, num_tasks), dtype=np.float32)

        # Fill tensors
        print(f"\n📝 Filling tensors...")
        for task_idx, task_info in enumerate(all_train_tasks):
            if task_idx % 10000 == 0:
                print(f"   Processing task {task_idx}/{num_tasks}...")

            samples_by_lib = task_info['samples_by_lib']
            node_start = node_slices[task_idx]
            node_end = node_slices[task_idx + 1]

            for lib_idx, sample in samples_by_lib.items():
                node_features = sample['node_features']
                if isinstance(node_features, torch.Tensor):
                    node_features = node_features.cpu().numpy()

                all_node_features[lib_idx, node_start:node_end, :] = node_features

                output = sample['output']
                if isinstance(output, torch.Tensor):
                    output = output.item()
                all_outputs[lib_idx, task_idx] = output

        # Calculate normalization stats from train data
        # include_zeros_in_norm=True: original method (include zeros in stats)
        # include_zeros_in_norm=False: norm2 method (exclude zeros from stats)
        norm_method = "all values" if include_zeros_in_norm else "non-zero values only"
        print(f"\n📊 Calculating normalization statistics ({norm_method})...")
        if include_parasitic_cap:
            NORMALIZE_INDICES = [4, 5, 6, 10, 11]  # voltage, input_slew, output_load, temperature, parasitic_cap
            NORMALIZE_NAMES = ['voltage', 'input_slew', 'output_load', 'temperature', 'parasitic_cap']
        else:
            NORMALIZE_INDICES = [4, 5, 6, 10]  # voltage, input_slew, output_load, temperature
            NORMALIZE_NAMES = ['voltage', 'input_slew', 'output_load', 'temperature']

        norm_stats = {}
        for idx, name in zip(NORMALIZE_INDICES, NORMALIZE_NAMES):
            feature_data = all_node_features[:, :, idx].flatten()

            if include_zeros_in_norm:
                # Original method: include zeros in stats calculation
                mean = float(np.mean(feature_data))
                std = float(np.std(feature_data))
                if std == 0:
                    std = 1.0  # Avoid division by zero
                nonzero_data = feature_data[feature_data != 0]
                nonzero_ratio = len(nonzero_data) / len(feature_data) * 100
            else:
                # Norm2 method: calculate stats from non-zero values only
                nonzero_data = feature_data[feature_data != 0]
                if len(nonzero_data) > 0:
                    mean = float(np.mean(nonzero_data))
                    std = float(np.std(nonzero_data))
                else:
                    mean = 0.0
                    std = 1.0  # Avoid division by zero
                nonzero_ratio = len(nonzero_data) / len(feature_data) * 100

            norm_stats[idx] = {'name': name, 'mean': mean, 'std': std}
            print(f"   {name}: mean={mean:.6f}, std={std:.6f} (non-zero: {nonzero_ratio:.1f}%)")

        # NOTE: Normalization is NOT applied here - only stats are saved
        # During training: normalized = (x - mean) / std for non-zero values, keep 0 as 0

        # Save train dataset as single .pth file (same format as build_gnn_dataset_stage_aware_cached_tsmc.py)
        # mode_suffix already defined above (combines voltage_suffix and temperature_suffix)
        train_path = output_dir / f"train_{data_type}_{cache_type}{mode_suffix}.pth"
        print(f"\n💾 Saving train dataset: {train_path}")

        # Convert to torch tensors
        node_features_tensor = torch.from_numpy(all_node_features)
        outputs_tensor = torch.from_numpy(all_outputs)
        node_slices_tensor = torch.from_numpy(node_slices)

        train_data = {
            'node_features': node_features_tensor,
            'outputs': outputs_tensor,
            'node_slices': node_slices_tensor,
            'cell_names': cell_names,
            'delay_types': delay_types,
            'output_names': output_names,
            'node_counts': task_node_counts,
            'num_tasks': num_tasks,
            'num_libs': num_libs,
            'num_features': num_features,
            'total_nodes': total_nodes,
            'format': 'unified_3d',
            'process_node': 'TSMC',
            'data_type': data_type,
            'graph_mode': cache_type,
            'cache_path': cache_path,
            'train_corners': TRAIN_CORNERS,
            'train_temperatures': TRAIN_TEMPERATURES,
            'num_conditions': len(train_folders),
            'include_parasitic_cap': include_parasitic_cap,
            'voltage_mode': voltage_mode,
            'slew_mode': slew_mode,  # 'all' or 'related_pin_only'
            'topology_suffix': topology_suffix,  # e.g., '_weighted_inputport'
            'excluded_cells': INTRA_TOPOLOGY_CELLS,  # Cells excluded from training (for intra-topology test)
            'norm_stats': {
                'node_features': {stats['name']: {'mean': stats['mean'], 'std': stats['std']}
                                  for idx, stats in norm_stats.items()}
            },
            'normalize_indices': NORMALIZE_INDICES,
            'normalize_names': NORMALIZE_NAMES,
            'normalize_nonzero_only': not include_zeros_in_norm,  # If include_zeros=True, normalize_nonzero_only=False (original); else True (norm2)
            'include_zeros_in_norm': include_zeros_in_norm,  # Explicit flag for clarity
        }

        torch.save(train_data, train_path)
        print(f"   ✅ Saved: {train_path}")
        print(f"   node_features: {node_features_tensor.shape}")
        print(f"   outputs: {outputs_tensor.shape}")

        # Clean up train data
        del all_node_features, all_outputs, node_features_tensor, outputs_tensor
        del all_train_tasks
        gc.collect()

    else:
        print(f"\n{'='*80}")
        print("SKIPPING TRAIN DATA PROCESSING (--skip_train)")
        print(f"{'='*80}")

        # Load existing norm_stats from .pth file (mode_suffix already defined above)
        train_path = output_dir / f"train_{data_type}_{cache_type}{mode_suffix}.pth"
        if train_path.exists():
            train_data = torch.load(train_path, weights_only=False)
            norm_stats = {}
            for idx, name in enumerate(['voltage', 'input_slew', 'output_load', 'temperature']):
                if name in train_data.get('norm_stats', {}).get('node_features', {}):
                    stats = train_data['norm_stats']['node_features'][name]
                    norm_stats[[4, 5, 6, 10][idx]] = {'name': name, 'mean': stats['mean'], 'std': stats['std']}
            num_libs = train_data.get('num_libs', 61)
        else:
            print(f"❌ Train data not found: {train_path}")
            return

    # ==================== PROCESS TEST DATA ====================
    print(f"\n{'='*80}")
    print("PROCESSING TEST DATA - MEMORY-EFFICIENT MODE (SAVE PER FOLDER)")
    print(f"{'='*80}")

    # Create output and temp directories (mode_suffix already defined above)
    test_output_dir = output_dir / f"test_by_{data_type}_{cache_type}{mode_suffix}"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = test_output_dir / ".temp_partials"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Track which cells we've seen
    all_cell_names = set()

    # Step 1: Process each folder and save partial files immediately
    print(f"\n📂 Step 1: Processing folders and saving partial files...")

    for dir_idx, folder in enumerate(test_folders):
        corner, temperature, is_variant = parse_tsmc_folder_name(folder.name)
        process_params = get_abc_parameters(corner, temperature)

        variant_tag = " [variant]" if is_variant else ""
        print(f"\n[{dir_idx+1}/{len(test_folders)}] Processing {folder.name}{variant_tag}")

        lib_files = sorted(folder.glob("*.lib"))
        if not lib_files:
            print(f"   ⚠️  No .lib files found, skipping...")
            continue

        # Process all lib files (list-based)
        all_samples_per_lib, num_tasks_dir = process_directory_for_unified(
            folder, topology_cache, cache_type, process_params, data_type,
            include_parasitic_cap=include_parasitic_cap, voltage_mode=voltage_mode,
            temperature_mode=temperature_mode, slew_mode=slew_mode
        )

        print(f"   ✓ {num_tasks_dir} tasks from {len(lib_files)} lib files")

        # Group by cell name for this folder only
        folder_samples_by_cell = defaultdict(list)

        for task_idx in range(num_tasks_dir):
            samples_by_lib = {}
            valid = True
            cell_name = None

            for lib_idx in range(len(lib_files)):
                if lib_idx < len(all_samples_per_lib) and task_idx < len(all_samples_per_lib[lib_idx]):
                    sample = all_samples_per_lib[lib_idx][task_idx]
                    samples_by_lib[lib_idx] = sample
                    if cell_name is None:
                        cell_name = sample['cell_name']
                else:
                    valid = False
                    break

            if valid and len(samples_by_lib) == len(lib_files) and cell_name:
                folder_samples_by_cell[cell_name].append({
                    'dir_name': folder.name,
                    'samples_by_lib': samples_by_lib
                })
                all_cell_names.add(cell_name)

        # Save partial files for each cell in this folder
        for cell_name, tasks in folder_samples_by_cell.items():
            partial_path = temp_dir / f"{cell_name}_partial_{dir_idx:04d}.pth"
            torch.save(tasks, partial_path)

        # Clear memory for this folder
        del all_samples_per_lib, folder_samples_by_cell
        gc.collect()

    print(f"\n   ✅ Processed {len(test_folders)} folders")
    print(f"   ✅ Found {len(all_cell_names)} unique cells")

    # Step 2: Merge partial files for each cell
    print(f"\n📦 Step 2: Merging partial files into final .pth files...")
    print(f"   Output directory: {test_output_dir}")

    saved_count = 0
    for cell_idx, cell_name in enumerate(sorted(all_cell_names)):
        # Find all partial files for this cell
        partial_files = sorted(temp_dir.glob(f"{cell_name}_partial_*.pth"))

        if not partial_files:
            continue

        # Load and merge all partial data
        all_tasks = []
        for pf in partial_files:
            tasks = torch.load(pf, weights_only=False)
            all_tasks.extend(tasks)
            # Delete partial file after loading
            pf.unlink()

        if len(all_tasks) == 0:
            continue

        # Determine num_libs from first task
        first_task = all_tasks[0]
        num_libs_cell = len(first_task['samples_by_lib'])
        num_tasks = len(all_tasks)

        # Calculate total nodes and collect metadata
        task_node_counts = []
        delay_types = []
        output_names = []
        for task in all_tasks:
            sample = task['samples_by_lib'][0]
            task_node_counts.append(sample['num_nodes'])
            delay_types.append(sample.get('delay_type', 'rise'))
            output_names.append(sample.get('output_name', ''))

        total_nodes = sum(task_node_counts)
        node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
        node_slices[1:] = np.cumsum(task_node_counts)

        # Allocate arrays
        node_features = np.zeros((num_libs_cell, total_nodes, num_features), dtype=np.float32)
        outputs = np.zeros((num_libs_cell, num_tasks), dtype=np.float32)

        # Fill arrays
        for task_idx, task in enumerate(all_tasks):
            samples_by_lib = task['samples_by_lib']
            node_start = node_slices[task_idx]
            node_end = node_slices[task_idx + 1]

            for lib_idx, sample in samples_by_lib.items():
                nf = sample['node_features']
                if isinstance(nf, torch.Tensor):
                    nf = nf.cpu().numpy()
                node_features[lib_idx, node_start:node_end, :] = nf

                out = sample['output']
                if isinstance(out, torch.Tensor):
                    out = out.item()
                outputs[lib_idx, task_idx] = out

        # Save as single .pth file per cell
        cell_path = test_output_dir / f"{cell_name}.pth"
        cell_data = {
            'node_features': torch.from_numpy(node_features),
            'outputs': torch.from_numpy(outputs),
            'node_slices': torch.from_numpy(node_slices),
            'delay_types': delay_types,
            'output_names': output_names,
            'num_tasks': num_tasks,
            'num_libs': num_libs_cell,
            'num_features': num_features,
            'total_nodes': total_nodes,
            'cell_name': cell_name,
            'format': 'unified_3d',
            'include_parasitic_cap': include_parasitic_cap,
            'voltage_mode': voltage_mode,
            'slew_mode': slew_mode,  # 'all' or 'related_pin_only'
            'topology_suffix': topology_suffix,  # e.g., '_weighted_inputport'
            'normalize_nonzero_only': not include_zeros_in_norm,  # If include_zeros=True, normalize_nonzero_only=False (original)
            'include_zeros_in_norm': include_zeros_in_norm,  # Explicit flag for clarity
        }
        torch.save(cell_data, cell_path)
        saved_count += 1

        # Clear memory after each cell
        del all_tasks, node_features, outputs, cell_data
        gc.collect()

        if (cell_idx + 1) % 10 == 0:
            print(f"   Processed {cell_idx + 1}/{len(all_cell_names)} cells...")

    # Clean up temp directory
    try:
        temp_dir.rmdir()
    except:
        pass  # Directory might not be empty if there were errors

    print(f"   ✅ Saved {saved_count} cell .pth files")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    if not skip_train:
        print(f"Train: {train_path}")
    print(f"Test: {test_output_dir}")
    print(f"\n✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build TSMC GNN dataset with unified 3D tensor format"
    )
    parser.add_argument("--cache_path", type=str, required=True,
                       help="Path to topology cache")
    parser.add_argument("--cache_type", type=str, required=True,
                       choices=['full_graph', 'stage_aware'],
                       help="Cache type: full_graph or stage_aware")
    parser.add_argument("--lib_base_path", type=str, required=True,
                       help="Base path to TSMC library files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for datasets")
    parser.add_argument("--data_type", type=str, default="cell",
                       choices=['cell', 'transition'],
                       help="Data type: cell (delay) or transition (slew)")
    parser.add_argument("--skip_train", action="store_true",
                       help="Skip train data processing (only process test data)")
    parser.add_argument("--include_parasitic_cap", action="store_true",
                       help="Include parasitic capacitance as additional node feature (12D instead of 11D). "
                            "Requires topology cache generated with --weighted option.")
    parser.add_argument("--voltage_mode", type=str, default="all_nodes",
                       choices=['all_nodes', 'vdd_only', 'vdd_mos'],
                       help="Voltage feature mode: 'all_nodes' (default) applies voltage to all nodes, "
                            "'vdd_only' applies voltage only to VDD node (0 for others), "
                            "'vdd_mos' applies voltage to VDD and MOS transistor nodes only")
    parser.add_argument("--temperature_mode", type=str, default="mos_only",
                       choices=["mos_only", "temp_all"],
                       help="Temperature feature mode: "
                            "'mos_only' (default) applies temperature to MOS transistor nodes only, "
                            "'temp_all' applies temperature to all nodes")
    parser.add_argument("--include_zeros_in_norm", action="store_true",
                       help="Include zeros when calculating normalization statistics (original method). "
                            "Default is False (norm2 method - exclude zeros from stats calculation).")
    parser.add_argument("--topology_suffix", type=str, default="",
                       help="Suffix for topology options (e.g., '_weighted_inputport'). "
                            "This is appended to output filenames to distinguish datasets "
                            "generated with different topology configurations.")
    parser.add_argument("--slew_mode", type=str, default="all",
                       choices=['all', 'related_pin_only'],
                       help="Input slew assignment mode: "
                            "'all' (default) applies input_slew to all input ports or connected MOS, "
                            "'related_pin_only' applies input_slew only to the related_pin node or its connected MOS")

    args = parser.parse_args()

    build_unified_datasets(
        cache_path=args.cache_path,
        cache_type=args.cache_type,
        lib_base_path=args.lib_base_path,
        output_dir=args.output_dir,
        data_type=args.data_type,
        skip_train=args.skip_train,
        include_parasitic_cap=args.include_parasitic_cap,
        voltage_mode=args.voltage_mode,
        temperature_mode=args.temperature_mode,
        include_zeros_in_norm=args.include_zeros_in_norm,
        topology_suffix=args.topology_suffix,
        slew_mode=args.slew_mode
    )
