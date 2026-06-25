#!/usr/bin/env python
"""
Generate lib files with GCN/MLP predicted timing values (canonical).

Handles both delay LUTs (cell_rise / cell_fall / rise_transition /
fall_transition, delay_template_7x7) and constraint LUTs (rise_constraint /
fall_constraint, constraint_template_3x3) plus all shared helpers
(LibFileParser, CellTestDataset, merge_libs, run_predictions_with_*).

Constraint handling:
  * Reads timing_type to route each LUT into one of the 6 constraint
    categories: setup, hold, recovery, removal, non_seq_setup, non_seq_hold.
  * Loads test PTHs from `test_by_<category>_stage_aware/` (not _cell_/_transition_).
  * Reuses the cell-delay pretrained model unchanged, with two inference-time
    tweaks documented in the rebuttal write-up:
      (a) output_load slot normalization aliases input_slew's norm_stats
          (both axes of constraint_template_3x3 are slews, in ns).
      (b) Predictions are made in abs() space, then signed back per category:
          hold / recovery / non_seq_hold negative, setup / removal /
          non_seq_setup positive.  Predictions inherit the sign of the
          original lib value when present (preserves cross-zero tasks).
  * Slot 6 (output_load) is fed the constrained-slew value directly (raw ns) —
    aliasing is what makes that semantically OK.

This script:
1. Loads GCN/MLP model and test dataset
2. Runs prediction for all tasks (all timing arcs)
3. Replaces timing table values in lib files with predictions
4. Saves new lib files to output directory

Usage:
    # Process specific cell with test folder
    python predict_comb_lib.py \
        --cell HA1D0BWP30P140 \
        --test_folder TSMC_TT_75 \
        --model_arch conv64x2_fc256x2 \
        --experiment topology_agnostic \
        --data_type cell \
        --graph_mode stage_aware

    # Process all cells from experiment
    python predict_comb_lib.py \
        --all_cells \
        --test_folder TSMC_FF_0 \
        --model_arch conv64x2_fc256x2 \
        --experiment topology_agnostic

    # NEW: Process single lib file (auto-selects cells from lib file that have test data)
    python predict_comb_lib.py \
        --lib_file /path/to/TSMC_TT_75_100.lib \
        --model_arch conv64x2_fc256x2 \
        --data_type cell \
        --graph_mode stage_aware

    The --lib_file mode:
    - Parses PVT (corner, temperature, voltage) from lib filename
    - Extracts cells from the lib file
    - Finds intersection with cells that have test data (.pth files)
    - Predicts only for those matching cells
    - Generates single output lib file with predictions

    # NEW: Use precomputed predictions from .npy files (fast, no model loading)
    python predict_comb_lib.py \
        --lib_file /path/to/TSMC_TT_75_100.lib \
        --use_precomputed \
        --pred_dir /path/to/data_result_npy_directory_final \
        --model_arch conv64x2_fc256x2 \
        --data_type all \
        --graph_mode stage_aware

    The --use_precomputed mode:
    - Loads predictions from pre-generated .npy files instead of running the model
    - Much faster since no model loading or inference is required
    - Requires that validation has been run first to generate the .npy files
    - Falls back to skipping cells if precomputed file is not found

    # NEW: Use lib files for few-shot support values (actual model inference)
    python predict_comb_lib.py \
        --lib_file /path/to/TSMC_TT_75_100.lib \
        --lib_few_shot \
        --lib_dir /path/to/TSMC_lib_files \
        --model_arch conv64x2_fc256x2 \
        --data_type all \
        --graph_mode stage_aware

    The --lib_few_shot mode:
    - Uses timing values from actual lib files at multiple voltages as few-shot support
    - Requires lib files at different voltages (e.g., TSMC_TT_75/TSMC_TT_75_060.lib to _120.lib)
    - Node features come from test data (.pth), support values from lib files
    - Runs actual model inference with MAML adaptation
    - Useful when you want to use measured lib values instead of simulation data
"""

import os
import sys
import re
import argparse
import shutil
from typing import Dict, List, Optional
import numpy as np

# Set CUDA device before importing torch
def get_gpu_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return '0'

os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_from_args()

import torch
import gc
from torch_geometric.data import Data, Batch

# Add paths
sys.path.append('/home/tkdgn2907/Deepsets_test/MAML/Projects/model_code/')
sys.path.append('/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/')
sys.path.append('/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/gnn/utils/')

from gnn_maml import create_maml_gcn_model


# Cell lists
INTRA_TOPOLOGY_CELLS = [
    'AN4D0BWP30P140', 'ND3D0BWP30P140', 'NR3D1BWP30P140',
    'OR4D0BWP30P140', 'XNR3D1BWP30P140', 'XOR3D1BWP30P140',
]

TOPOLOGY_AGNOSTIC_CELLS = [
    # 'HA1D0BWP30P140', 'FA1D0BWP30P140', 'IOA21D0BWP30P140', 'IOA21D1BWP30P140',
    # 'OA21D0BWP30P140', 'OA21D1BWP30P140', 'OA211D0BWP30P140', 'OA211D1BWP30P140',
    # 'IAO21D0BWP30P140', 'IAO21D1BWP30P140', 'AO21D0BWP30P140', 'AO21D1BWP30P140',
    # 'AO211D0BWP30P140', 'AO211D1BWP30P140',
    'SDFSNQD0BWP30P140', 'DFCNQD1BWP30P140', 'SDFCSNQD1BWP30P140',
]


def get_experiment_type_for_cell(cell_name: str) -> str:
    """Determine experiment type (intra_topology or topology_agnostic) for a cell."""
    if cell_name in INTRA_TOPOLOGY_CELLS:
        return 'intra_topology'
    else:
        return 'topology_agnostic'


# Process parameter mapping (nmos, pmos pairs for each corner)
CORNER_PARAMS = {
    'FF': {'param_a': (1.427, 1.457), 'param_b': (0.026, 0.045), 'param_c': (0.024, 2.0)},
    'TT': {'param_a': (1.430, 1.470), 'param_b': (0.0, 0.0), 'param_c': (0.024, 2.0)},
    'SS': {'param_a': (1.443, 1.483), 'param_b': (-0.026, -0.05), 'param_c': (0.024, 2.0)},
    'FS': {'param_a': (1.43, 1.47), 'param_b': (0.0208, -0.04), 'param_c': (0.024, 2.0)},
    'SF': {'param_a': (1.43, 1.47), 'param_b': (0.036, -0.0208), 'param_c': (0.024, 2.0)},
}


def extract_cells_from_lib(lib_path: str) -> List[str]:
    """Extract all cell names from a Liberty (.lib) file.

    Args:
        lib_path: Path to the lib file

    Returns:
        List of cell names found in the lib file
    """
    cells = []
    with open(lib_path, 'r') as f:
        for line in f:
            match = re.match(r'\s*cell\s*\(\s*(\w+)\s*\)', line.strip())
            if match:
                cells.append(match.group(1))
    return cells


def get_available_test_cells(dataset_dir: str, data_type: str, graph_mode: str) -> List[str]:
    """Get list of cells that have test data available.

    Args:
        dataset_dir: Base dataset directory
        data_type: 'cell' or 'transition'
        graph_mode: 'stage_aware' or 'full_graph'

    Returns:
        List of cell names that have .pth test files
    """
    test_dir = os.path.join(dataset_dir, f"test_by_{data_type}_{graph_mode}")
    if not os.path.exists(test_dir):
        return []

    cells = []
    for f in os.listdir(test_dir):
        if f.endswith('.pth'):
            cell_name = f[:-4]  # Remove .pth extension
            cells.append(cell_name)
    return cells


def parse_lib_file_name(lib_filename: str) -> tuple:
    """Parse lib file name to extract corner, temperature, and voltage.

    Examples:
        TSMC_TT_75_100.lib -> ('TT', 75.0, 1.00)
        TSMC_FF_0_088.lib -> ('FF', 0.0, 0.88)
        TSMC_SS_-25_110.lib -> ('SS', -25.0, 1.10)
        TSMC_FF_12.5_100.lib -> ('FF', 12.5, 1.00)  # decimal with dot
        TSMC_FF_12p5_100.lib -> ('FF', 12.5, 1.00)  # decimal with 'p'
    """
    import re

    # Remove .lib extension and path
    basename = os.path.basename(lib_filename)
    if basename.endswith('.lib'):
        basename = basename[:-4]

    # Pattern: TSMC_{corner}_{temperature}_{voltage}
    # Temperature can be: integer, negative, decimal with 'p' or '.'
    # Also support seq variants: TSMC_{corner}_Seq_{temperature}_{voltage}
    pattern_seq = r'TSMC_([A-Z]+)_Seq_(-?\d+(?:[p.]\d+)?)_(\d+)'
    match = re.match(pattern_seq, basename)
    if match:
        corner = match.group(1)
        temp_str = match.group(2).replace('p', '.')
        voltage_int = int(match.group(3))
        voltage = voltage_int / 100.0  # 100 -> 1.00V, 088 -> 0.88V
        return corner, float(temp_str), voltage

    # Standard pattern: TSMC_{corner}_{temperature}_{voltage}
    pattern = r'TSMC_([A-Z]+)_(-?\d+(?:[p.]\d+)?)_(\d+)'
    match = re.match(pattern, basename)
    if match:
        corner = match.group(1)
        temp_str = match.group(2).replace('p', '.')
        voltage_int = int(match.group(3))
        voltage = voltage_int / 100.0  # 100 -> 1.00V, 088 -> 0.88V
        return corner, float(temp_str), voltage

    return None, None, None


def parse_test_folder_name(folder_name: str) -> tuple:
    """Parse test folder name to extract corner and temperature.

    Examples:
        TSMC_FF_0 -> ('FF', 0.0)
        TSMC_SS_100 -> ('SS', 100.0)
        TSMC_TT_-25 -> ('TT', -25.0)
        TSMC_FF2_25 -> ('FF', 25.0)  # variant
        TSMC_TTseq_50 -> ('TT', 50.0)  # sequential variant
    """
    import re

    # Check for variant patterns (FF2, TT2, TTseq, etc.)
    pattern_variant = r'TSMC_([A-Z]+)(?:2|seq)_(-?\d+(?:p\d+)?)'
    match = re.match(pattern_variant, folder_name)
    if match:
        corner = match.group(1)
        temp_str = match.group(2).replace('p', '.')
        return corner, float(temp_str)

    # Standard pattern
    pattern_standard = r'TSMC_([A-Z]+)_(-?\d+(?:p\d+)?)'
    match = re.match(pattern_standard, folder_name)
    if match:
        corner = match.group(1)
        temp_str = match.group(2).replace('p', '.')
        return corner, float(temp_str)

    return None, None


def get_task_indices_for_folder(cell_dataset, test_folder: str) -> List[int]:
    """Get task indices that match the specified test folder.

    Uses node features to filter tasks based on process parameters and temperature.
    Matches both param_a AND param_b to uniquely identify each corner.
    """
    corner, temperature = parse_test_folder_name(test_folder)
    if corner is None:
        print(f"Warning: Could not parse test folder name: {test_folder}")
        return list(range(cell_dataset.num_tasks))

    if corner not in CORNER_PARAMS:
        print(f"Warning: Unknown corner: {corner}")
        return list(range(cell_dataset.num_tasks))

    corner_params = CORNER_PARAMS[corner]
    param_a_nmos, param_a_pmos = corner_params['param_a']
    param_b_nmos, param_b_pmos = corner_params['param_b']

    matching_indices = []

    for task_idx in range(cell_dataset.num_tasks):
        # Get node features for lib 0 (any lib works, process params same across libs)
        sample = cell_dataset.get_task_data(task_idx, 0)
        node_features = sample['node_features']

        # Check last node (output node typically has process params)
        last_node = node_features[-1]
        task_temp = last_node[10].item()
        task_param_a = last_node[7].item()
        task_param_b = last_node[8].item()

        # Match temperature
        if abs(task_temp - temperature) > 0.1:
            continue

        # Match corner: check if (param_a, param_b) matches nmos or pmos values
        # Use tolerance 0.005 for param_a and 0.005 for param_b
        nmos_match = (abs(task_param_a - param_a_nmos) < 0.005 and
                      abs(task_param_b - param_b_nmos) < 0.005)
        pmos_match = (abs(task_param_a - param_a_pmos) < 0.005 and
                      abs(task_param_b - param_b_pmos) < 0.005)

        if nmos_match or pmos_match:
            matching_indices.append(task_idx)

    return matching_indices


def voltage_to_lib_idx(voltage: float, min_voltage: float = 0.60, max_voltage: float = 1.20, num_libs: int = 61) -> int:
    """Convert voltage to lib index.

    Assumes linear spacing from min_voltage to max_voltage.

    Args:
        voltage: Target voltage in Volts
        min_voltage: Minimum voltage in dataset (default: 0.60V)
        max_voltage: Maximum voltage in dataset (default: 1.20V)
        num_libs: Number of lib points (default: 61)

    Returns:
        lib_idx corresponding to the voltage
    """
    step = (max_voltage - min_voltage) / (num_libs - 1)
    lib_idx = round((voltage - min_voltage) / step)
    return max(0, min(num_libs - 1, lib_idx))


def get_task_indices_for_pvt(cell_dataset, corner: str, temperature: float) -> List[int]:
    """Get task indices that match the specified Process and Temperature.

    This is used when --lib_file is provided to filter tasks by corner and temperature.
    Note: Voltage filtering is not done here because each task contains all 61 voltage points.
    The voltage is used later to select the correct lib_idx for extracting predictions.

    Args:
        cell_dataset: Cell test dataset
        corner: Process corner (FF, TT, SS, etc.)
        temperature: Temperature in Celsius

    Returns:
        List of task indices matching the corner and temperature conditions
    """
    if corner not in CORNER_PARAMS:
        print(f"Warning: Unknown corner: {corner}")
        return list(range(cell_dataset.num_tasks))

    corner_params = CORNER_PARAMS[corner]
    param_a_nmos, param_a_pmos = corner_params['param_a']
    param_b_nmos, param_b_pmos = corner_params['param_b']

    matching_indices = []

    for task_idx in range(cell_dataset.num_tasks):
        # Get node features for lib 0 (any lib works, process params same across libs)
        sample = cell_dataset.get_task_data(task_idx, 0)
        node_features = sample['node_features']

        # Check last node (output node typically has process params)
        last_node = node_features[-1]
        task_temp = last_node[10].item()
        task_param_a = last_node[7].item()
        task_param_b = last_node[8].item()

        # Match temperature (tolerance 0.1)
        if abs(task_temp - temperature) > 0.1:
            continue

        # Match corner: check if (param_a, param_b) matches nmos or pmos values
        nmos_match = (abs(task_param_a - param_a_nmos) < 0.005 and
                      abs(task_param_b - param_b_nmos) < 0.005)
        pmos_match = (abs(task_param_a - param_a_pmos) < 0.005 and
                      abs(task_param_b - param_b_pmos) < 0.005)

        if nmos_match or pmos_match:
            matching_indices.append(task_idx)

    return matching_indices


def find_precomputed_pred_file(
    pred_dir: str,
    cell_name: str,
    data_type: str,
    graph_mode: str,
    mode: str,
    model_type: str,
    experiment: str,
    args,
) -> Optional[str]:
    """
    Find the precomputed pred.npy file matching the given parameters.

    File naming pattern examples:
    - TSMC_GCN_intra_topology_AN4D0BWP30P140_cell_stage_aware_extrapolation_maml_innerdiv10_meta16_iter300000_inner1_conv64x2_fc256x2_filtered_pred.npy
    - TSMC_GCN_topology_agnostic_HA1D0BWP30P140_transition_stage_aware_interpolation_baseline_iter300000_conv64x2_fc256x2_pooloutput_filtered_pred.npy
    """
    import glob

    # Build search pattern
    pool_suffix = "_pooloutput" if data_type == 'transition' else ""
    arch_suffix = f"_conv{args.conv_hidden_dim}x{args.num_conv_layers}_fc{args.fc_hidden_dim}x{args.num_fc_layers}{pool_suffix}"

    if model_type == 'baseline':
        model_pattern = f"baseline_iter{args.num_iterations}{arch_suffix}"
    else:
        model_pattern = f"maml_innerdiv{args.innerdiv}_meta{args.meta}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}"

    # Build filename pattern
    pattern = f"TSMC_GCN_{experiment}_{cell_name}_{data_type}_{graph_mode}_{mode}_{model_pattern}_filtered*_pred.npy"

    # Search for matching files
    search_path = os.path.join(pred_dir, pattern)
    matches = glob.glob(search_path)

    if matches:
        # Return the first match (or could implement more specific selection)
        return matches[0]

    # Try without some optional suffixes (adam, vddonly, relpin)
    pattern_simple = f"TSMC_GCN_{experiment}_{cell_name}_{data_type}_{graph_mode}_{mode}_{model_pattern}_filtered_pred.npy"
    search_path_simple = os.path.join(pred_dir, pattern_simple)
    matches_simple = glob.glob(search_path_simple)

    if matches_simple:
        return matches_simple[0]

    return None


def load_precomputed_predictions(
    pred_file: str,
    cell_dataset,
    num_libs: int = 61,
    task_indices: Optional[List[int]] = None,
) -> Dict:
    """
    Load precomputed predictions from .npy file and map to task/lib structure.

    IMPORTANT: The pred.npy is saved in RANDOM order (from validation's random sampling).
    We must use act.npy values to match predictions to the correct tasks via value-based matching.

    The pred.npy file is a flat 1D array of shape (num_sampled_tasks * num_libs,)
    where predictions are stored in the order they were sampled during validation.

    Args:
        pred_file: Path to the _pred.npy file
        cell_dataset: CellTestDataset to get task count
        num_libs: Number of lib points per task (default: 61)
        task_indices: Optional list of task indices to filter (if None, use all)

    Returns:
        Dict {task_idx: [lib_predictions]} matching the format expected by lib generation
    """
    # Load pred and act arrays
    pred = np.load(pred_file)
    act_file = pred_file.replace('_pred.npy', '_act.npy')

    if not os.path.exists(act_file):
        print(f"  WARNING: act.npy not found, cannot do value-based matching!")
        return {}

    act = np.load(act_file)

    # Calculate number of tasks from the flat array
    total_elements = len(pred)
    num_tasks_in_file = total_elements // num_libs

    print(f"  Loaded: {num_tasks_in_file} predictions from file")

    # Reshape to (num_tasks_in_file, num_libs)
    pred_reshaped = pred.reshape(num_tasks_in_file, num_libs)
    act_reshaped = act.reshape(num_tasks_in_file, num_libs)

    # Build ground truth lookup from cell_dataset for target task_indices
    # Use multiple lib indices for robust matching
    ref_lib_indices = [0, 30, 60]  # Use multiple reference points

    if task_indices is None:
        task_indices = list(range(cell_dataset.num_tasks))

    # Build lookup: tuple of ground truth values at ref_libs -> task_idx
    gt_to_task = {}
    for task_idx in task_indices:
        gt_key = tuple(
            round(cell_dataset.get_task_data(task_idx, lib_idx)['output'], 6)
            for lib_idx in ref_lib_indices
        )
        gt_to_task[gt_key] = task_idx

    # Match predictions to tasks using act values
    predictions = {}
    matched_count = 0
    unmatched_count = 0

    for pred_row_idx in range(num_tasks_in_file):
        # Get act values at reference lib indices
        act_key = tuple(
            round(act_reshaped[pred_row_idx, lib_idx], 6)
            for lib_idx in ref_lib_indices
        )

        if act_key in gt_to_task:
            task_idx = gt_to_task[act_key]
            predictions[task_idx] = pred_reshaped[pred_row_idx].tolist()
            matched_count += 1
        else:
            # Try with tolerance
            found = False
            for gt_key, task_idx in gt_to_task.items():
                if all(abs(a - g) < 1e-5 for a, g in zip(act_key, gt_key)):
                    predictions[task_idx] = pred_reshaped[pred_row_idx].tolist()
                    matched_count += 1
                    found = True
                    break
            if not found:
                unmatched_count += 1

    print(f"  Value-based matching: {matched_count} matched, {unmatched_count} unmatched")
    print(f"  Coverage: {len(predictions)}/{len(task_indices)} target tasks")

    return predictions


def get_lib_folder_name(corner: str, temperature: float) -> str:
    """Get lib folder name from corner and temperature.

    Examples:
        ('TT', 75.0) -> 'TSMC_TT_75'
        ('FF', 0.0) -> 'TSMC_FF_0'
        ('SS', -25.0) -> 'TSMC_SS_-25'
    """
    temp_str = str(int(temperature)) if temperature == int(temperature) else str(temperature).replace('.', 'p')
    return f"TSMC_{corner}_{temp_str}"


def get_lib_file_for_voltage(lib_folder_path: str, voltage: float) -> Optional[str]:
    """Find lib file matching the given voltage.

    Args:
        lib_folder_path: Path to folder containing lib files
        voltage: Target voltage (e.g., 0.90 for 0.90V)

    Returns:
        Path to matching lib file, or None if not found
    """
    voltage_int = int(round(voltage * 100))  # 0.90 -> 90, 1.00 -> 100
    voltage_str = f"{voltage_int:03d}"  # 90 -> "090", 100 -> "100"

    # List files and find match
    if not os.path.exists(lib_folder_path):
        return None

    for f in os.listdir(lib_folder_path):
        if f.endswith('.lib') and voltage_str in f:
            # Verify it's the right voltage (not just substring)
            # Pattern: TSMC_XX_YY_ZZZ.lib where ZZZ is voltage
            parts = f.replace('.lib', '').split('_')
            if len(parts) >= 4 and parts[-1] == voltage_str:
                return os.path.join(lib_folder_path, f)

    return None


def extract_lib_support_data(
    lib_dir: str,
    corner: str,
    temperature: float,
    cell_name: str,
    support_indices: List[int],
    num_libs: int = 61,
    min_voltage: float = 0.60,
    max_voltage: float = 1.20,
    seq_lib_dir: Optional[str] = None,
    seq_folder_suffix: str = '',
) -> Optional[Dict]:
    """
    Extract timing values and index arrays from lib files for support set.

    All lib files at different voltages have the same structure (same order),
    so we can directly match by position.

    Args:
        lib_dir: Base directory containing lib folders (used for comb cells)
        corner: Process corner (FF, TT, SS, etc.)
        temperature: Temperature in Celsius
        cell_name: Name of the cell
        support_indices: List of lib indices to use as support (e.g., [0, 13, 30, 45, 60])
        num_libs: Total number of lib points (default: 61)
        min_voltage: Minimum voltage (default: 0.60V)
        max_voltage: Maximum voltage (default: 1.20V)
        seq_lib_dir: Optional base dir override for sequential cells. If None, uses lib_dir.
            Use this to point seq lookup at fine-resolution (1ps) libs while keeping
            comb on the original 10ps libs.
        seq_folder_suffix: Optional suffix appended to seq folder name (e.g., '_x10').
            Final seq path: {seq_lib_dir or lib_dir}/TSMC_{corner}seq_{temp}{suffix}

    Returns:
        Dict with:
            'tables': List of table info dicts, each with:
                - output_pin, table_type, index_1, index_2
                - support_values: 2D list [n_slews * n_loads][n_support_points]
            'support_indices': The support indices used
    """
    # Try combinational folder first, then sequential folder
    temp_str = str(int(temperature)) if temperature == int(temperature) else str(temperature).replace('.', 'p')
    comb_folder_name = f"TSMC_{corner}_{temp_str}"
    seq_folder_name = f"TSMC_{corner}seq_{temp_str}{seq_folder_suffix}"

    seq_base_dir = seq_lib_dir if seq_lib_dir else lib_dir
    comb_folder_path = os.path.join(lib_dir, comb_folder_name)
    seq_folder_path = os.path.join(seq_base_dir, seq_folder_name)

    # Calculate voltages for support indices
    step = (max_voltage - min_voltage) / (num_libs - 1)
    support_voltages = [min_voltage + idx * step for idx in support_indices]

    print(f"  Support indices: {support_indices}")
    print(f"  Support voltages: {[f'{v:.2f}V' for v in support_voltages]}")

    # Try combinational folder first
    lib_folder_path = comb_folder_path
    is_sequential = False

    if not os.path.exists(lib_folder_path):
        # Try sequential folder
        if os.path.exists(seq_folder_path):
            lib_folder_path = seq_folder_path
            is_sequential = True
        else:
            print(f"  Lib folder not found: {lib_folder_path}")
            return None

    # Find lib files for each support voltage
    support_lib_files = []
    for voltage in support_voltages:
        lib_file = get_lib_file_for_voltage(lib_folder_path, voltage)
        if lib_file:
            support_lib_files.append(lib_file)
        else:
            print(f"  WARNING: Lib file not found for voltage {voltage:.2f}V in {lib_folder_path}")
            return None

    # Parse first lib file to get table structure
    first_parser = LibFileParser(support_lib_files[0])
    first_tables = first_parser.find_timing_tables()
    cell_tables = [t for t in first_tables if t['cell_name'] == cell_name]

    # If not found in combinational, try sequential folder
    if len(cell_tables) == 0 and not is_sequential and os.path.exists(seq_folder_path):
        print(f"  Cell not found in combinational lib, trying sequential lib...")
        lib_folder_path = seq_folder_path
        is_sequential = True

        # Re-find lib files for sequential folder
        support_lib_files = []
        for voltage in support_voltages:
            lib_file = get_lib_file_for_voltage(lib_folder_path, voltage)
            if lib_file:
                support_lib_files.append(lib_file)
            else:
                print(f"  WARNING: Lib file not found for voltage {voltage:.2f}V in {lib_folder_path}")
                return None

        # Re-parse first lib file
        first_parser = LibFileParser(support_lib_files[0])
        first_tables = first_parser.find_timing_tables()
        cell_tables = [t for t in first_tables if t['cell_name'] == cell_name]

    print(f"  Found {len(cell_tables)} timing tables for {cell_name}" + (" (sequential)" if is_sequential else ""))

    # Drop 1-D LUTs (no index_2) — only mpw_constraint_template in current libs.
    # We don't model min_pulse_width and the rest of the pipeline assumes 2-D
    # (n_slews × n_loads) LUTs.  This is the *only* skip rule; everything else
    # is treated as required-uniform and any mismatch raises below.
    n_before_mpw = len(cell_tables)
    cell_tables = [t for t in cell_tables if t.get('index_2')]
    if n_before_mpw != len(cell_tables):
        print(f"  Dropped {n_before_mpw - len(cell_tables)} 1-D LUT(s) (mpw_constraint-like)")

    # Strict mode: assume all voltage files have the *same* LUT structure (same
    # number of tables in the same order, same index_1/index_2 dimensions, full
    # original_values arrays).  Any mismatch raises — silently skipping was the
    # source of cross-voltage stitching bugs that mixed cell_rise values with
    # constraint LUT values.
    result_tables = []

    for orig_idx, table in enumerate(cell_tables):
        index_1 = table['index_1'] or []
        index_2 = table['index_2'] or []
        n_slews = len(index_1) if index_1 else 7
        n_loads = len(index_2) if index_2 else 7
        n_values = n_slews * n_loads

        first_values = table.get('original_values', [])
        if len(first_values) != n_values:
            raise ValueError(
                f"[extract_lib_support_data] cell={cell_name} "
                f"table[{orig_idx}] ({table.get('type')} "
                f"{table.get('related_pin','-')}->{table['output_pin']} "
                f"when='{table.get('when','-')}') at V=0.60: "
                f"original_values={len(first_values)} but n_slews*n_loads={n_values} "
                f"(n_idx1={n_slews}, n_idx2={n_loads}). "
                f"Fix LibFileParser or filter the cell's LUTs upstream."
            )

        support_values = [[first_values[i]] for i in range(n_values)]
        result_tables.append({
            'output_pin': table['output_pin'],
            'related_pin': table.get('related_pin', ''),
            'when': table.get('when', ''),
            'table_type': table['type'],
            'index_1': index_1,
            'index_2': index_2,
            'n_slews': n_slews,
            'n_loads': n_loads,
            'support_values': support_values,
        })

    for lib_idx, lib_file in enumerate(support_lib_files[1:], start=1):
        parser = LibFileParser(lib_file)
        tables = parser.find_timing_tables()
        cell_tables_this = [t for t in tables if t['cell_name'] == cell_name]
        cell_tables_this = [t for t in cell_tables_this if t.get('index_2')]

        if len(cell_tables_this) != len(result_tables):
            raise ValueError(
                f"[extract_lib_support_data] cell={cell_name} table count mismatch: "
                f"V=0.60 has {len(result_tables)} tables but voltage file "
                f"#{lib_idx} ({os.path.basename(lib_file)}) has {len(cell_tables_this)}."
            )

        for orig_idx, result_table in enumerate(result_tables):
            table = cell_tables_this[orig_idx]
            values = table.get('original_values', [])
            n_values = result_table['n_slews'] * result_table['n_loads']
            if len(values) != n_values:
                raise ValueError(
                    f"[extract_lib_support_data] cell={cell_name} "
                    f"table[{orig_idx}] ({table.get('type')} "
                    f"{table.get('related_pin','-')}->{table['output_pin']}) "
                    f"voltage file #{lib_idx} ({os.path.basename(lib_file)}): "
                    f"original_values={len(values)} but expected {n_values}."
                )
            for i in range(n_values):
                result_table['support_values'][i].append(values[i])

    print(f"  Tables loaded: {len(result_tables)}")

    return {
        'tables': result_tables,
        'support_indices': support_indices,
    }


class LibFileParser:
    """Parse and modify Liberty (.lib) files."""

    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        with open(lib_path, 'r') as f:
            self.content = f.read()
        self.lines = self.content.split('\n')

    def find_timing_tables(self) -> List[Dict]:
        """
        Find all timing tables in the lib file.
        Returns list of dicts with table info.
        """
        tables = []
        current_cell = None
        current_pin = None
        current_related_pin = None
        current_when = None
        current_timing_type = None   # constraint variant: captures the timing_type token

        i = 0
        while i < len(self.lines):
            line = self.lines[i].strip()

            # Track cell context
            cell_match = re.match(r'cell\s*\(\s*(\w+)\s*\)', line)
            if cell_match:
                current_cell = cell_match.group(1)

            # Track pin context
            pin_match = re.match(r'pin\s*\(\s*(\w+)\s*\)', line)
            if pin_match:
                current_pin = pin_match.group(1)

            # Track timing() group start - reset when / related_pin / timing_type
            if re.match(r'timing\s*\(\s*\)\s*\{', line):
                current_related_pin = None
                current_when = None
                current_timing_type = None

            # Track related_pin in timing group
            related_match = re.match(r'related_pin\s*:\s*"?(\w+)"?\s*;', line)
            if related_match:
                current_related_pin = related_match.group(1)

            # Track when condition in timing group
            when_match = re.match(r'when\s*:\s*"([^"]+)"\s*;', line)
            if when_match:
                current_when = when_match.group(1)

            # Track timing_type — needed to route constraint LUTs into setup / hold / etc.
            tt_match = re.match(r'timing_type\s*:\s*(\w+)\s*;', line)
            if tt_match:
                current_timing_type = tt_match.group(1)

            # Find timing tables.  CONSTRAINT variant: detect 6 table types — 4 delay + 2 constraint.
            # Constraint blocks are gated on the template name (constraint_template_3x3) to skip
            # the mpw_constraint_template_3x3 (minimum-pulse-width) blocks.
            DELAY_TABLE_TYPES      = ['cell_rise', 'rise_transition', 'cell_fall', 'fall_transition']
            CONSTRAINT_TABLE_TYPES = ['rise_constraint', 'fall_constraint']
            for table_type in DELAY_TABLE_TYPES + CONSTRAINT_TABLE_TYPES:
                if not line.startswith(f'{table_type}('):
                    continue
                # constraint variant: only constraint_template_3x3 (skip mpw)
                if table_type in CONSTRAINT_TABLE_TYPES and 'constraint_template_3x3' not in line:
                    continue
                table_info = self._parse_timing_table(i, table_type)
                if table_info:
                    table_info['cell_name']   = current_cell
                    table_info['output_pin']  = current_pin
                    table_info['related_pin'] = current_related_pin
                    table_info['when']        = current_when
                    table_info['timing_type'] = current_timing_type
                    tables.append(table_info)

            i += 1

        return tables

    def _parse_timing_table(self, start_line: int, table_type: str) -> Optional[Dict]:
        """Parse a single timing table starting at given line."""
        info = {
            'type': table_type,
            'start_line': start_line,
            'index_1': None,
            'index_2': None,
            'values_start': None,
            'values_end': None,
            'original_values': [],
        }

        i = start_line
        brace_count = 0
        in_values = False
        values_lines = []

        while i < len(self.lines):
            line = self.lines[i]

            # Count braces
            brace_count += line.count('{') - line.count('}')

            # Parse index_1
            if 'index_1' in line:
                match = re.search(r'index_1\s*\(\s*"([^"]+)"\s*\)', line)
                if match:
                    info['index_1'] = [float(x) for x in match.group(1).split(',')]

            # Parse index_2
            if 'index_2' in line:
                match = re.search(r'index_2\s*\(\s*"([^"]+)"\s*\)', line)
                if match:
                    info['index_2'] = [float(x) for x in match.group(1).split(',')]

            # Parse values
            if 'values(' in line:
                info['values_start'] = i
                in_values = True

            if in_values:
                values_lines.append(i)
                if ');' in line:
                    info['values_end'] = i
                    in_values = False

            # End of table
            if brace_count == 0 and i > start_line:
                info['end_line'] = i
                break

            i += 1

        # Extract original values
        if info['values_start'] is not None and info['values_end'] is not None:
            values_text = ''
            for line_idx in range(info['values_start'], info['values_end'] + 1):
                values_text += self.lines[line_idx]

            # Parse values from text
            match = re.search(r'values\s*\((.*?)\);', values_text, re.DOTALL)
            if match:
                values_str = match.group(1)
                # Remove quotes and backslashes
                values_str = values_str.replace('"', '').replace('\\', '').replace('\n', ' ')
                values = []
                for v in values_str.split(','):
                    v = v.strip()
                    if v:
                        try:
                            values.append(float(v))
                        except ValueError:
                            pass
                info['original_values'] = values

        return info

    def replace_timing_values(self, table_info: Dict, new_values: List[float]) -> None:
        """Replace timing table values with new values."""
        if table_info['values_start'] is None or table_info['values_end'] is None:
            return

        # Get index dimensions
        n_rows = len(table_info['index_1']) if table_info['index_1'] else 7
        n_cols = len(table_info['index_2']) if table_info['index_2'] else 7

        if len(new_values) != n_rows * n_cols:
            print(f"Warning: Expected {n_rows * n_cols} values, got {len(new_values)}")
            return

        # Format new values in Liberty format (matching original format)
        values_lines = []

        for row in range(n_rows):
            row_values = new_values[row * n_cols : (row + 1) * n_cols]
            row_str = ', '.join([f'{v:.4g}' for v in row_values])

            if row == 0:
                # First row: values(" on same line
                if n_rows == 1:
                    values_lines.append(f'          values("{row_str}");')
                else:
                    values_lines.append(f'          values("{row_str}",\\')
            elif row < n_rows - 1:
                # Middle rows
                values_lines.append(f'                 "{row_str}",\\')
            else:
                # Last row
                values_lines.append(f'                 "{row_str}");')

        # Replace lines
        new_lines = (
            self.lines[:table_info['values_start']] +
            values_lines +
            self.lines[table_info['values_end'] + 1:]
        )
        self.lines = new_lines
        self.content = '\n'.join(self.lines)

    def save(self, output_path: str) -> None:
        """Save modified lib file."""
        with open(output_path, 'w') as f:
            f.write(self.content)


class CellTestDataset:
    """Load per-cell test data."""

    def __init__(self, cell_path: str):
        self.cell_path = cell_path
        data = torch.load(cell_path, weights_only=False, map_location='cpu')

        self._node_features = data['node_features']
        self._outputs = data['outputs']
        self._node_slices = data['node_slices']
        self._delay_types = data.get('delay_types', None)
        self._output_names = data.get('output_names', None)

        self.num_libs = data['num_libs']
        self.num_tasks = data['num_tasks']
        self.cell_name = data['cell_name']

    def get_task_data(self, task_idx: int, lib_idx: int):
        """Get data for specific task and lib."""
        node_start = self._node_slices[task_idx].item()
        node_end = self._node_slices[task_idx + 1].item()

        node_features = self._node_features[lib_idx, node_start:node_end, :].clone()
        output = self._outputs[lib_idx, task_idx].item()

        delay_type = self._delay_types[task_idx] if self._delay_types else 'rise'
        output_name = self._output_names[task_idx] if self._output_names else ''

        return {
            'node_features': node_features,
            'output': output,
            'delay_type': delay_type,
            'output_name': output_name,
        }

    def get_task_info(self, task_idx: int) -> Dict:
        """Get task metadata."""
        return {
            'delay_type': self._delay_types[task_idx] if self._delay_types else 'rise',
            'output_name': self._output_names[task_idx] if self._output_names else '',
        }


class LightweightCellDataset:
    """Drop-in replacement for CellTestDataset used at lib write phase.

    Keeps only the small per-task metadata that update_lib_file_for_cell needs
    (output_name, delay_type, lib_idx=0 node_features for slew/load lookup) so
    the heavy `_node_features` + `_outputs` PTH tensors can be discarded once
    predict phase completes — cuts per-cell RSS growth from ~250 MB to <1 MB.
    """

    def __init__(self, cell_name: str, num_libs: int, num_tasks: int,
                 task_info_map: Dict, task_features_map: Dict):
        self.cell_name = cell_name
        self.num_libs = num_libs
        self.num_tasks = num_tasks
        self._info = task_info_map
        self._feats = task_features_map

    def get_task_info(self, task_idx: int) -> Dict:
        return self._info[task_idx]

    def get_task_data(self, task_idx: int, lib_idx: int = 0):
        # Cached snapshot is from lib_idx=0; the slew/load values that the
        # write phase reads (columns 5, 6) are voltage-independent, so this is
        # fine for lookup purposes.  We do NOT support lookups at arbitrary
        # lib_idx — the predict phase is over.
        return {'node_features': self._feats[task_idx]}


def cache_minimal_cell_meta(cell_dataset: 'CellTestDataset',
                            predictions: Dict) -> 'LightweightCellDataset':
    """Build a LightweightCellDataset from the predictions' task_idx set.

    `predictions` keys are either int task_idx or (task_idx, related_pin,
    when) tuples — we pull task_idx from both.
    """
    task_indices = set()
    for k in predictions.keys():
        ti = k[0] if isinstance(k, tuple) else k
        task_indices.add(ti)

    task_info_map = {ti: cell_dataset.get_task_info(ti) for ti in task_indices}
    # Snapshot node_features at lib_idx=0 — small (~few KB per task).
    task_features_map = {
        ti: cell_dataset.get_task_data(ti, 0)['node_features']
        for ti in task_indices
    }
    return LightweightCellDataset(
        cell_name=cell_dataset.cell_name,
        num_libs=cell_dataset.num_libs,
        num_tasks=cell_dataset.num_tasks,
        task_info_map=task_info_map,
        task_features_map=task_features_map,
    )


def normalize_node_features(node_features, norm_stats):
    """Normalize node features using saved statistics."""
    if norm_stats is None:
        return node_features

    normalized = node_features.clone()

    # Normalize voltage (column 4)
    voltage_mask = normalized[:, 4] != 0
    if voltage_mask.any():
        normalized[voltage_mask, 4] = (
            normalized[voltage_mask, 4] - norm_stats['node_features']['voltage']['mean']
        ) / norm_stats['node_features']['voltage']['std']

    # Normalize input_slew (column 5)
    slew_mask = normalized[:, 5] != 0
    if slew_mask.any():
        normalized[slew_mask, 5] = (
            normalized[slew_mask, 5] - norm_stats['node_features']['input_slew']['mean']
        ) / norm_stats['node_features']['input_slew']['std']

    # Normalize output_load (column 6)
    load_mask = normalized[:, 6] != 0
    if load_mask.any():
        normalized[load_mask, 6] = (
            normalized[load_mask, 6] - norm_stats['node_features']['output_load']['mean']
        ) / norm_stats['node_features']['output_load']['std']

    # Normalize temperature (column 10) if available
    # Match train (normalize_node_features_safe) and validation behavior: use
    # MOSFET-type mask (column 2 != 0) so MOS nodes at temp=0 also get
    # normalized to (0 - mean)/std instead of being left at raw 0 — otherwise
    # the model sees OOD features for temp=0 evaluations (build script uses
    # temperature_mode='mos_only' which assigns temperature only to MOS nodes).
    if 'temperature' in norm_stats['node_features'] and normalized.shape[1] > 10:
        mosfet_mask = normalized[:, 2] != 0  # PMOS=+1, NMOS=-1
        if mosfet_mask.any():
            normalized[mosfet_mask, 10] = (
                normalized[mosfet_mask, 10] - norm_stats['node_features']['temperature']['mean']
            ) / norm_stats['node_features']['temperature']['std']

    return normalized


def run_predictions_with_adaptation(model, cell_dataset, topology_cache, norm_stats, device, args, task_indices=None, data_type=None, task_outputs_override=None):
    """
    Run predictions for specified tasks with MAML adaptation.
    Same logic as TSMC_GCN_topology_validation.py.

    Args:
        task_indices: List of task indices to process. If None, processes all tasks.
        data_type: Optional. When set to a constraint category
            (setup/hold/recovery/removal/non_seq_setup/non_seq_hold), enable
            abs-space adaptation: support y is abs()'d so the cell-delay model
            sees a positive monotone-decreasing curve regardless of the
            constraint's native sign.  Sign is re-applied at lib write time
            (_update_lib_constraint_orderbased).

    Returns: predictions dict {task_idx: [lib_predictions]} for processed tasks
    """
    cell_name = cell_dataset.cell_name
    num_libs = cell_dataset.num_libs

    # If no task_indices specified, use all
    if task_indices is None:
        task_indices = list(range(cell_dataset.num_tasks))

    num_tasks_to_process = len(task_indices)
    predictions = {}

    CONSTRAINT_CATS = {'setup','hold','recovery','removal','non_seq_setup','non_seq_hold'}
    # v6: per-task slope-direction sign_flip replaces categorical NEG_SIGN_CATS
    # abs+force-sign.  For each task we look at y_low vs y_high.  If the curve
    # is already decreasing with V (matches the cell-delay model's prior) we
    # leave it alone.  If it is inverted (e.g. y goes -0.4 → -0.1, "increasing"
    # in signed space), we multiply y by -1 before adapting so the model sees
    # a decreasing curve in its natural prior space; predictions are then
    # multiplied back by the same sign at the end.  This preserves curve
    # SHAPE (no abs() kink at zero-crossings) and removes the rigid
    # categorical-sign assumption that mishandled cells whose hold/recovery
    # had positive outliers (~22%) or whose setup/removal had negative outliers.
    is_constraint = data_type in CONSTRAINT_CATS
    if is_constraint:
        # Constraint LUTs cram the constrained-pin slew into the output_load
        # (col 6) slot, but norm_stats['output_load'] was learned on capacitance
        # (mean≈0.001 pF, std≈0.01 pF).  Slew values 0.0017–0.6113 ns then get
        # normalized to +0.07…+61σ — wildly OOD.  Alias output_load's stats to
        # input_slew's locally so the model sees in-distribution values.
        # (Matches the alias in TSMC_GCN_topology_validation.py and in
        # run_predictions_with_lib_support.)
        if norm_stats is not None:
            import copy as _copy
            norm_stats = _copy.deepcopy(norm_stats)
            nf = norm_stats.get('node_features', {})
            if 'input_slew' in nf:
                nf['output_load'] = dict(nf['input_slew'])
                print(f"  [constraint mode] aliased output_load norm_stats → input_slew "
                      f"(both axes are slews in ns).")
        print(f"  [constraint mode v6] data_type={data_type}: per-task sign_flip "
              f"based on y_low vs y_high; predictions returned in original sign space.")

    print(f"Running predictions for {cell_name}: {num_tasks_to_process} tasks x {num_libs} libs")
    print(f"  Mode: {args.mode}, Adaptation: {args.adaptation_method}")

    cell_cache = topology_cache[cell_name]

    # Set indices based on mode
    if args.mode == 'extrapolation':
        indices = [5, 30, 55]
    else:  # interpolation
        indices = [0, 13, 30, 45, 60]

    left_bound = min(indices)
    right_bound = max(indices) + 1
    middle_idx = len(indices) // 2

    for i, task_idx in enumerate(task_indices):
        if i % 500 == 0:
            print(f"  Task {i}/{num_tasks_to_process} (idx={task_idx})", flush=True)

        task_info = cell_dataset.get_task_info(task_idx)
        delay_type = task_info['delay_type']
        output_name = task_info['output_name']

        # Get adjacency matrix
        if args.graph_mode == 'stage_aware':
            if 'output_topologies' in cell_cache and output_name in cell_cache['output_topologies']:
                output_topo = cell_cache['output_topologies'][output_name]
                if 'rise' in delay_type:
                    adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
                else:
                    adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
            else:
                adjacency_matrix = cell_cache.get('adjacency_matrix', torch.zeros((1, 1)))
        else:
            adjacency_matrix = cell_cache['adjacency_matrix']

        edge_index = adjacency_matrix.nonzero().t()

        # Get all lib outputs for this task
        task_outputs = []
        task_samples = []
        for lib_idx in range(num_libs):
            sample = cell_dataset.get_task_data(task_idx, lib_idx)
            sample['node_features'] = normalize_node_features(sample['node_features'], norm_stats)
            task_samples.append(sample)
            task_outputs.append(sample['output'])

        # Optional override: replace PTH-based task_outputs with values from a
        # different lib resolution (e.g. _x10 0.1ps libs) so the model adapts
        # to those targets instead of the PTH's original (1ps) values.
        if task_outputs_override is not None and task_idx in task_outputs_override:
            task_outputs = list(task_outputs_override[task_idx])

        task_outputs_tensor = torch.tensor(task_outputs, dtype=torch.float32)
        # v6 sign_flip: detect per-task natural curve direction.  Sum support y
        # values at the two extreme indices and compare; if y_high > y_low the
        # curve is "inverted" (increases with V in raw signed space) and we
        # flip it so the model adapts on a decreasing curve matching its prior.
        # Predictions are unflipped at output.  This is a pure vertical mirror
        # — no shape distortion, no zero-crossing kinks.
        task_sign_flip = 1.0
        if is_constraint:
            y_low  = task_outputs_tensor[indices[0]].item()
            y_high = task_outputs_tensor[indices[-1]].item()
            if y_high > y_low:
                task_sign_flip = -1.0
                task_outputs_tensor = -task_outputs_tensor

        # Get support set
        y = task_outputs_tensor[indices]
        y_mean = y.mean()
        y_std = y.std()

        if y_std <= 0:
            # No variation, use raw model predictions
            model.eval()
            task_preds = []
            for lib_idx in range(num_libs):
                node_features = task_samples[lib_idx]['node_features']
                data = Data(x=node_features, edge_index=edge_index)
                batch = Batch.from_data_list([data]).to(device)
                with torch.no_grad():
                    task_preds.append(model(batch).item())
            predictions[task_idx] = task_preds
            continue

        y_norm = (y - y_mean) / y_std

        # Calculate center prediction for grad/move.
        # Match validation: project nominal 0.9V through train-time voltage
        # normalization and apply only to voltage-bearing nodes (mask != 0).
        center_sample = task_samples[indices[0]]
        center_node_features = center_sample['node_features'].clone()
        _vs = norm_stats['node_features']['voltage']
        if 'method' in _vs and _vs['method'] == 'minmax_positive':
            _eps = _vs.get('epsilon', 0.01)
            _norm_nom = _eps + (0.9 - _vs['min']) / (_vs['max'] - _vs['min']) * (1 - _eps)
        else:
            _norm_nom = (0.9 - _vs['mean']) / _vs['std']
        _vmask = center_node_features[:, 4] != 0
        center_node_features[_vmask, 4] = _norm_nom

        center_data = Data(x=center_node_features, edge_index=edge_index)
        center_batch = Batch.from_data_list([center_data]).to(device)

        model.eval()
        with torch.no_grad():
            center = model(center_batch).item()

        y_max = y_norm.max().item()
        y_min = y_norm.min().item()

        # Get model predictions at SUPPORT indices for scaling (mirrors
        # TSMC_GCN_topology_validation.py).  Earlier this loop sampled every
        # voltage point in [left_bound, right_bound), which gave a wider
        # min/max range for non-monotone constraint curves and consequently a
        # smaller `grad`, shrinking the inner-loop training signal so much
        # that the selective_adam threshold was rarely crossed.
        support_predictions = []
        for idx in indices:
            node_features = task_samples[idx]['node_features']
            data = Data(x=node_features, edge_index=edge_index)
            batch = Batch.from_data_list([data]).to(device)
            with torch.no_grad():
                pred = model(batch).item()
            support_predictions.append(pred)

        support_predictions = torch.tensor(support_predictions)
        min_val = support_predictions.min().item()
        max_val = support_predictions.max().item()

        if abs(max_val - min_val) <= 1e-8:
            # No variation in predictions, use raw values (already in original
            # sign space — task_sign_flip would just re-invert the inversion).
            predictions[task_idx] = task_outputs
            continue

        grad = (y_max - y_min) / (max_val - min_val)
        y_norm_middle = y_norm[middle_idx].item()
        move = center - y_norm_middle / grad

        # Create adapted model copy
        node_features_dim = model.convs[0].lin.weight.shape[1]
        adapted_model = create_maml_gcn_model(
            node_features=node_features_dim,
            pooling=model.pooling_type,
            output_dim=1,
            dropout=0.0,
            conv_hidden_dim=model.conv_hidden_dim,
            num_conv_layers=model.num_conv_layers,
            fc_hidden_dim=model.fc_hidden_dim,
            num_fc_layers=model.num_fc_layers
        ).to(device)
        adapted_model.load_state_dict(model.state_dict())

        # Prepare support batch
        support_data_list = []
        for idx in indices:
            node_features = task_samples[idx]['node_features']
            data = Data(x=node_features, edge_index=edge_index)
            support_data_list.append(data)

        X_batch = Batch.from_data_list(support_data_list).to(device)

        # Calculate scaled y for training
        y_scaled = y_std * grad
        y_train = (y - y_mean) / y_scaled + move
        y_train = y_train.to(device).view(-1, 1)

        # Adam adaptation
        criterion = torch.nn.MSELoss()
        K = len(indices)
        loss = criterion(adapted_model(X_batch), y_train) / K

        if args.adaptation_method == 'adam' or (args.adaptation_method == 'selective_adam' and loss > 1e-4):
            inner_lr = getattr(args, 'inner_lr', 3e-4)
            inner_steps = getattr(args, 'inner_steps_adapt', 40)
            optimizer = torch.optim.Adam(adapted_model.parameters(), lr=inner_lr, weight_decay=1e-4)
            for step in range(inner_steps):
                loss = criterion(adapted_model(X_batch), y_train) / K
                adapted_model.zero_grad()
                loss.backward()
                optimizer.step()

        # Predict all 61 points with adapted model
        adapted_model.eval()
        task_preds = []
        with torch.no_grad():
            for lib_idx in range(num_libs):
                node_features = task_samples[lib_idx]['node_features']
                data = Data(x=node_features, edge_index=edge_index)
                batch = Batch.from_data_list([data]).to(device)

                raw_pred = adapted_model(batch).item()
                # Denormalize
                pred_value = (raw_pred - move) * y_scaled + y_mean
                task_preds.append(pred_value.item() if isinstance(pred_value, torch.Tensor) else pred_value)

        # v6: undo per-task sign_flip so predictions return to original sign space
        if task_sign_flip < 0:
            task_preds = [-p for p in task_preds]
        predictions[task_idx] = task_preds

        # Per-task cleanup — keep RSS flat across all tasks in the cell.
        try:
            del adapted_model, optimizer, task_samples
        except (NameError, UnboundLocalError):
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions


def build_task_lookup(cell_dataset, task_indices: List[int]) -> Dict:
    """
    Build a lookup table from (output_name, delay_type, slew, load) -> task_idx.

    The slew and load values are extracted from node_features columns 5 and 6.
    """
    lookup = {}

    for task_idx in task_indices:
        task_info = cell_dataset.get_task_info(task_idx)
        output_name = task_info['output_name']
        delay_type = task_info['delay_type']

        # Get slew and load from node_features (lib 0 is fine, same across libs)
        sample = cell_dataset.get_task_data(task_idx, 0)
        node_features = sample['node_features']

        # Column 5 = input_slew, Column 6 = output_load
        slew_vals = node_features[:, 5][node_features[:, 5] != 0]
        load_vals = node_features[:, 6][node_features[:, 6] != 0]

        slew = slew_vals[0].item() if len(slew_vals) > 0 else 0
        load = load_vals[0].item() if len(load_vals) > 0 else 0

        # Round for matching (6 decimal places)
        key = (output_name, delay_type, round(slew, 6), round(load, 6))
        lookup[key] = task_idx

    return lookup


def run_predictions_with_lib_support(
    model,
    cell_dataset,
    topology_cache,
    norm_stats,
    device,
    args,
    lib_support_data: Dict,
    data_type: str,
    task_indices: List[int],
):
    """
    Run predictions using lib file values as support set.

    This is the --lib_few_shot mode: uses actual lib file timing values
    instead of test data outputs for the few-shot support set.

    The approach:
    1. Build task lookup by (output_name, delay_type, slew, load) -> task_idx
    2. For each timing table entry in lib files:
       - Get slew/load from index_1/index_2
       - Find matching task using lookup
       - Run prediction with lib support values and task's node_features

    Args:
        model: GNN model
        cell_dataset: Cell test dataset (for node_features)
        topology_cache: Topology cache for adjacency matrices
        norm_stats: Normalization statistics
        device: PyTorch device
        args: Command line arguments
        lib_support_data: Dict from extract_lib_support_data()
        data_type: 'cell' or 'transition'
        task_indices: List of task indices (filtered by PT)

    Returns:
        Dict {task_idx: [lib_predictions]} for processed tasks
    """
    cell_name = cell_dataset.cell_name
    num_libs = cell_dataset.num_libs

    print(f"Running predictions with lib file support for {cell_name}")
    print(f"  Mode: {args.mode}, Adaptation: {args.adaptation_method}")

    # === Constraint variant — inference-time tweaks ============================
    # Both axes of constraint_template_3x3 are slews (ns), but the builder writes
    # the constrained-pin slew into the output_load (pF) slot.  Alias
    # output_load's normalization stats to input_slew's so the model isn't fed
    # a +20-sigma OOD on that slot.  Applied locally (deep copy) so we never
    # mutate the caller's norm_stats.
    is_constraint = data_type in {'setup', 'hold', 'recovery', 'removal',
                                  'non_seq_setup', 'non_seq_hold'}
    abs_target = is_constraint  # match validation script's --abs_target
    sign = -1.0 if data_type in {'hold', 'recovery', 'non_seq_hold'} else 1.0
    if is_constraint and norm_stats is not None:
        import copy as _copy
        norm_stats = _copy.deepcopy(norm_stats)
        nf = norm_stats.get('node_features', {})
        if 'input_slew' in nf:
            nf['output_load'] = dict(nf['input_slew'])
            print(f"  [constraint mode] aliased output_load norm_stats → input_slew, "
                  f"abs_target=True, sign={sign:+.0f}")
    # =========================================================================

    tables = lib_support_data['tables']
    support_indices = lib_support_data['support_indices']

    print(f"  Lib tables: {len(tables)}, Support indices: {support_indices}")

    # Build task lookup
    task_lookup = build_task_lookup(cell_dataset, task_indices)
    print(f"  Task lookup built: {len(task_lookup)} entries")

    cell_cache = topology_cache[cell_name]

    # MAML settings
    left_bound = min(support_indices)
    right_bound = max(support_indices) + 1
    middle_idx = len(support_indices) // 2
    k = len(support_indices)

    predictions = {}
    matched_count = 0
    unmatched_count = 0

    for table_idx, table in enumerate(tables):
        output_pin = table['output_pin']
        related_pin = table.get('related_pin', '')
        when_cond = table.get('when', '')
        table_type = table['table_type']  # cell_rise, cell_fall, rise_transition, fall_transition
        index_1 = table['index_1']  # slew values
        index_2 = table['index_2']  # load values
        n_slews = table['n_slews']
        n_loads = table['n_loads']
        support_values_list = table['support_values']  # [n_slews * n_loads][n_support]

        # Map table_type to delay_type.
        # Constraint variant: also handle rise_constraint / fall_constraint with
        # timing_type-based routing.
        timing_type = table.get('timing_type') or ''
        if table_type == 'cell_rise':
            delay_type_match = 'rise' if data_type == 'cell' else None
        elif table_type == 'cell_fall':
            delay_type_match = 'fall' if data_type == 'cell' else None
        elif table_type == 'rise_transition':
            delay_type_match = 'rise' if data_type == 'transition' else None
        elif table_type == 'fall_transition':
            delay_type_match = 'fall' if data_type == 'transition' else None
        elif table_type in ('rise_constraint', 'fall_constraint'):
            # only emit predictions for the matching constraint category
            # (data_type ∈ {setup, hold, recovery, removal, non_seq_setup, non_seq_hold})
            cat = None
            for c in ('setup', 'hold', 'recovery', 'removal',
                      'non_seq_setup', 'non_seq_hold'):
                if timing_type.startswith(c + '_'):
                    cat = c; break
            if cat is None or cat != data_type:
                continue
            # 'rise'/'fall' here refers to the constrained-pin edge, used purely
            # to look up the matching task in the constraint test PTH.
            delay_type_match = 'rise' if table_type == 'rise_constraint' else 'fall'
        else:
            continue

        if delay_type_match is None:
            continue

        if table_idx % 10 == 0:
            print(f"  Processing table {table_idx}/{len(tables)}: {related_pin}->{output_pin} {table_type}", flush=True)

        # Process each (slew, load) entry
        for slew_idx in range(n_slews):
            for load_idx in range(n_loads):
                val_idx = slew_idx * n_loads + load_idx
                slew = index_1[slew_idx] if slew_idx < len(index_1) else 0
                load = index_2[load_idx] if load_idx < len(index_2) else 0

                # Find matching task by trying different delay_type patterns
                task_idx = None
                for dt_pattern in [f'{delay_type_match}_transition', f'cell_{delay_type_match}', delay_type_match]:
                    key = (output_pin, dt_pattern, round(slew, 6), round(load, 6))
                    if key in task_lookup:
                        task_idx = task_lookup[key]
                        break

                if task_idx is None:
                    unmatched_count += 1
                    continue

                matched_count += 1

                # Use (task_idx, related_pin, when) as key to distinguish different timing arcs
                pred_key = (task_idx, related_pin, when_cond)

                # Skip if already processed for this specific timing arc
                if pred_key in predictions:
                    continue

                # Get support values for this entry
                y = torch.tensor(support_values_list[val_idx], dtype=torch.float32)
                # Constraint variant: predict in abs() space so the curve shape
                # (V↑ → |y|↓) lines up with the cell-delay model's monotone
                # decreasing prior.  Sign is re-applied after inference.
                if abs_target:
                    y = y.abs()
                y_mean = y.mean()
                y_std = y.std()

                if y_std <= 0:
                    continue

                y_norm = (y - y_mean) / y_std

                # Get task info and adjacency matrix
                task_info = cell_dataset.get_task_info(task_idx)
                delay_type = task_info['delay_type']

                if args.graph_mode == 'stage_aware':
                    if 'output_topologies' in cell_cache and output_pin in cell_cache['output_topologies']:
                        output_topo = cell_cache['output_topologies'][output_pin]
                        if 'rise' in delay_type:
                            adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
                        else:
                            adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
                    else:
                        adjacency_matrix = cell_cache.get('adjacency_matrix', torch.zeros((1, 1)))
                else:
                    adjacency_matrix = cell_cache['adjacency_matrix']

                edge_index = adjacency_matrix.nonzero().t()

                # Get node_features for all 61 lib points
                task_samples = []
                for lib_idx in range(num_libs):
                    sample = cell_dataset.get_task_data(task_idx, lib_idx)
                    sample['node_features'] = normalize_node_features(sample['node_features'], norm_stats)
                    task_samples.append(sample)

                # Calculate center prediction.  Match validation: project
                # nominal 0.9V through train-time voltage normalization and
                # apply only to voltage-bearing nodes (mask != 0).
                center_sample = task_samples[support_indices[0]]
                center_node_features = center_sample['node_features'].clone()
                _vs = norm_stats['node_features']['voltage']
                if 'method' in _vs and _vs['method'] == 'minmax_positive':
                    _eps = _vs.get('epsilon', 0.01)
                    _norm_nom = _eps + (0.9 - _vs['min']) / (_vs['max'] - _vs['min']) * (1 - _eps)
                else:
                    _norm_nom = (0.9 - _vs['mean']) / _vs['std']
                _vmask = center_node_features[:, 4] != 0
                center_node_features[_vmask, 4] = _norm_nom

                center_data = Data(x=center_node_features, edge_index=edge_index)
                center_batch = Batch.from_data_list([center_data]).to(device)

                model.eval()
                with torch.no_grad():
                    center = model(center_batch).item()

                y_max = y_norm.max().item()
                y_min = y_norm.min().item()

                # 5-pt support model predictions for grad scaling (mirrors
                # validation script).  Using inter_predictions over all 61
                # voltages widens the model min/max range whenever the model
                # output is non-monotone in V, shrinks `grad`, and breaks the
                # post-adapt scaling — predictions then collapse toward y_mean
                # at non-support voltages (the bug that produced ~20 ps
                # interpolation errors on seq cell delay).
                support_predictions = []
                for idx in support_indices:
                    node_features = task_samples[idx]['node_features']
                    data = Data(x=node_features, edge_index=edge_index)
                    batch = Batch.from_data_list([data]).to(device)
                    with torch.no_grad():
                        pred = model(batch).item()
                    support_predictions.append(pred)

                support_preds_tensor = torch.tensor(support_predictions)
                min_val = support_preds_tensor.min().item()
                max_val = support_preds_tensor.max().item()

                if abs(max_val - min_val) <= 1e-8:
                    continue

                grad = (y_max - y_min) / (max_val - min_val)
                y_norm_middle = y_norm[middle_idx].item()
                move = center - y_norm_middle / grad

                # Create adapted model copy
                node_features_dim = model.convs[0].lin.weight.shape[1]
                adapted_model = create_maml_gcn_model(
                    node_features=node_features_dim,
                    pooling=model.pooling_type,
                    output_dim=1,
                    dropout=0.0,
                    conv_hidden_dim=model.conv_hidden_dim,
                    num_conv_layers=model.num_conv_layers,
                    fc_hidden_dim=model.fc_hidden_dim,
                    num_fc_layers=model.num_fc_layers
                ).to(device)
                adapted_model.load_state_dict(model.state_dict())

                # Prepare support batch
                support_data_list = []
                for idx in support_indices:
                    node_features = task_samples[idx]['node_features']
                    data = Data(x=node_features, edge_index=edge_index)
                    support_data_list.append(data)

                X_batch = Batch.from_data_list(support_data_list).to(device)

                # Calculate scaled y for training
                y_scaled = y_std * grad
                y_train = (y - y_mean) / y_scaled + move
                y_train = y_train.to(device).view(-1, 1)

                # Adam adaptation
                criterion = torch.nn.MSELoss()
                loss = criterion(adapted_model(X_batch), y_train) / k

                if args.adaptation_method == 'adam' or (args.adaptation_method == 'selective_adam' and loss > 1e-4):
                    inner_lr = getattr(args, 'inner_lr', 3e-4)
                    inner_steps = getattr(args, 'inner_steps_adapt', 40)
                    optimizer = torch.optim.Adam(adapted_model.parameters(), lr=inner_lr, weight_decay=1e-4)
                    for step in range(inner_steps):
                        loss = criterion(adapted_model(X_batch), y_train) / k
                        adapted_model.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Predict all 61 points with adapted model
                adapted_model.eval()
                task_preds = []
                with torch.no_grad():
                    for lib_idx in range(num_libs):
                        node_features = task_samples[lib_idx]['node_features']
                        data = Data(x=node_features, edge_index=edge_index)
                        batch = Batch.from_data_list([data]).to(device)

                        raw_pred = adapted_model(batch).item()
                        # Denormalize
                        pred_value = (raw_pred - move) * y_scaled + y_mean
                        pv = pred_value.item() if isinstance(pred_value, torch.Tensor) else pred_value
                        # Constraint variant: predictions live in abs() space; re-apply
                        # category sign here so the stored task_preds match lib conventions
                        # (hold / recovery / non_seq_hold negative; setup / removal /
                        # non_seq_setup positive).  Take abs() of the raw prediction first
                        # to suppress any small per-task standardization undershoot
                        # (rare, but the model can output slightly negative magnitudes).
                        if is_constraint:
                            pv = sign * abs(float(pv))
                        task_preds.append(pv)

                predictions[pred_key] = task_preds
                # Per-task cleanup — drop adapted_model + optimizer + task_samples refs
                # so PyTorch can reclaim graph tensors between tasks.
                try:
                    del adapted_model, optimizer, task_samples
                except (NameError, UnboundLocalError):
                    pass

        # End of one table — flush any straggling refs.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  Matched entries: {matched_count}, Unmatched: {unmatched_count}")
    print(f"  Predictions (task_idx, related_pin, when): {len(predictions)}")
    return predictions


def run_predictions_with_lib_support_constraint(
    model,
    cell_dataset,
    topology_cache,
    norm_stats,
    device,
    args,
    lib_support_data: Dict,
    data_type: str,
    task_indices: List[int],
):
    """Constraint-aware lib_few_shot prediction (order-based task → table mapping).

    The base `run_predictions_with_lib_support` looks up tasks by
    (output_name, delay_type, slew, load).  Constraint test PTHs do not carry
    that metadata — output_name is the cell's primary output (Q) rather than
    the constrained pin (D), and node_features columns 5/6 store voltages
    instead of (slew, load).  This sibling function instead walks the
    constraint lib tables in lib order and pulls 9 entries per table from
    rise/fall task queues sorted by task_idx — the same order convention used
    by `_update_lib_constraint_orderbased` at write time.

    All inference-time tweaks from the original lib_support path are kept:
      - norm_stats output_load → input_slew alias (constraint LUT axes are
        both slews in ns; output_load stats were learned on capacitance).
      - v6 per-task slope-direction sign_flip: detect whether y_low < y_high
        (curve "inverted") and flip y for adapt so the cell-delay model sees a
        monotone-decreasing prior; unflip predictions at the end.  No abs()
        kink at zero crossings, no rigid categorical sign assumption.

    Returns: predictions dict keyed by task_idx (compatible with the
    constraint write path).
    """
    CONSTRAINT_CATS = {'setup','hold','recovery','removal','non_seq_setup','non_seq_hold'}
    is_constraint = data_type in CONSTRAINT_CATS
    if not is_constraint:
        raise ValueError(f"run_predictions_with_lib_support_constraint called with "
                         f"non-constraint data_type={data_type}")

    cell_name = cell_dataset.cell_name
    num_libs  = cell_dataset.num_libs
    print(f"Running lib_few_shot constraint predictions for {cell_name} ({data_type})")

    # Local norm_stats with constraint-axis alias.
    import copy as _copy
    norm_stats_local = _copy.deepcopy(norm_stats) if norm_stats is not None else None
    if norm_stats_local is not None:
        nf = norm_stats_local.get('node_features', {})
        if 'input_slew' in nf:
            nf['output_load'] = dict(nf['input_slew'])
            print(f"  [constraint mode] aliased output_load → input_slew")

    tables = lib_support_data['tables']
    support_indices = lib_support_data['support_indices']
    print(f"  Lib tables: {len(tables)}, Support indices: {support_indices}")

    # Filter lib tables to constraint blocks matching data_type.  Keep order.
    relevant_tables = []
    for tab in tables:
        if tab['table_type'] not in ('rise_constraint', 'fall_constraint'):
            continue
        if not (tab.get('timing_type') or '').startswith(data_type + '_'):
            continue
        relevant_tables.append(tab)
    print(f"  Relevant constraint tables for {data_type}: {len(relevant_tables)}")

    # Split filtered task_indices by delay_type into rise/fall queues, sorted.
    dt = {ti: cell_dataset.get_task_info(ti)['delay_type'] for ti in task_indices}
    rise_q = sorted(t for t in task_indices if 'rise' in dt[t])
    fall_q = sorted(t for t in task_indices if 'fall' in dt[t])
    print(f"  Task queues: rise={len(rise_q)}, fall={len(fall_q)}")

    cell_cache = topology_cache[cell_name]
    middle_idx = len(support_indices) // 2

    predictions = {}
    matched_count = unmatched_count = 0
    ri = iter(rise_q); fi = iter(fall_q)

    for table_idx, tab in enumerate(relevant_tables):
        n_slews = tab['n_slews']; n_loads = tab['n_loads']
        n_entries = n_slews * n_loads
        support_values_list = tab['support_values']  # [n_entries][n_support]
        q_iter = ri if tab['table_type'] == 'rise_constraint' else fi
        if table_idx % 10 == 0:
            print(f"  Table {table_idx}/{len(relevant_tables)}: "
                  f"{tab.get('related_pin','')}->{tab['output_pin']} {tab['table_type']} "
                  f"when='{tab.get('when','')}' n_entries={n_entries}", flush=True)

        for k in range(n_entries):
            try:
                task_idx = next(q_iter)
            except StopIteration:
                unmatched_count += 1
                continue
            matched_count += 1
            # 5-shot support from lib values at this entry.
            y = torch.tensor(support_values_list[k], dtype=torch.float32)

            # v6 sign_flip: detect direction from 5 support values.
            sign_flip = 1.0
            if y[0].item() < y[-1].item():
                sign_flip = -1.0
                y = -y

            y_mean = y.mean(); y_std = y.std()
            if y_std <= 0:
                continue
            y_norm = (y - y_mean) / y_std

            # Task-specific graph + node_features.
            task_info = cell_dataset.get_task_info(task_idx)
            delay_type_pth = task_info['delay_type']
            output_name = task_info['output_name']
            if 'output_topologies' in cell_cache and output_name in cell_cache['output_topologies']:
                otopo = cell_cache['output_topologies'][output_name]
                adj = otopo['pull_up' if 'rise' in delay_type_pth else 'pull_down']['adjacency_matrix']
            else:
                adj = cell_cache.get('adjacency_matrix', torch.zeros((1, 1)))
            edge_index = adj.nonzero().t()

            task_samples = []
            for li in range(num_libs):
                s = cell_dataset.get_task_data(task_idx, li)
                s['node_features'] = normalize_node_features(s['node_features'], norm_stats_local)
                task_samples.append(s)

            # Center prediction.  Match validation: project nominal 0.9V
            # through train-time voltage normalization (mask voltage-bearing
            # nodes only).
            cnf = task_samples[support_indices[0]]['node_features'].clone()
            _vs = norm_stats_local['node_features']['voltage']
            if 'method' in _vs and _vs['method'] == 'minmax_positive':
                _eps = _vs.get('epsilon', 0.01)
                _norm_nom = _eps + (0.9 - _vs['min']) / (_vs['max'] - _vs['min']) * (1 - _eps)
            else:
                _norm_nom = (0.9 - _vs['mean']) / _vs['std']
            _vmask = cnf[:, 4] != 0
            cnf[_vmask, 4] = _norm_nom
            cd = Data(x=cnf, edge_index=edge_index)
            cb = Batch.from_data_list([cd]).to(device)
            model.eval()
            with torch.no_grad():
                center = model(cb).item()

            # 5-pt support model predictions (consistent with validation grad scaling).
            sp = []
            for sidx in support_indices:
                d = Data(x=task_samples[sidx]['node_features'], edge_index=edge_index)
                b = Batch.from_data_list([d]).to(device)
                with torch.no_grad():
                    sp.append(model(b).item())
            sp = torch.tensor(sp)
            min_val = sp.min().item(); max_val = sp.max().item()
            if abs(max_val - min_val) <= 1e-8:
                # Degenerate model output — just store raw lib values (sign restored).
                preds = [float(v) * sign_flip for v in support_values_list[k]]
                # Expand to 61 by repeating? — better: fall back to raw signed support
                # across all 61 (not great, but matches the run_predictions_with_adaptation
                # behavior on the same degenerate path).
                predictions[task_idx] = preds + [preds[-1]] * (num_libs - len(preds))
                continue

            grad = (y_norm.max().item() - y_norm.min().item()) / (max_val - min_val)
            move = center - y_norm[middle_idx].item() / grad
            y_scaled = y_std * grad
            y_train = ((y - y_mean) / y_scaled + move).to(device).view(-1, 1)

            am = create_maml_gcn_model(
                node_features=model.convs[0].lin.weight.shape[1],
                pooling=model.pooling_type, output_dim=1, dropout=0.0,
                conv_hidden_dim=model.conv_hidden_dim, num_conv_layers=model.num_conv_layers,
                fc_hidden_dim=model.fc_hidden_dim, num_fc_layers=model.num_fc_layers,
            ).to(device)
            am.load_state_dict(model.state_dict())

            sdl = [Data(x=task_samples[sidx]['node_features'], edge_index=edge_index)
                   for sidx in support_indices]
            X_batch = Batch.from_data_list(sdl).to(device)
            criterion = torch.nn.MSELoss(); K = len(support_indices)
            loss = criterion(am(X_batch), y_train) / K
            if args.adaptation_method == 'adam' or \
               (args.adaptation_method == 'selective_adam' and loss > 1e-4):
                inner_lr = getattr(args, 'inner_lr', 3e-4)
                inner_steps = getattr(args, 'inner_steps_adapt', 40)
                opt = torch.optim.Adam(am.parameters(), lr=inner_lr, weight_decay=1e-4)
                for _ in range(inner_steps):
                    loss = criterion(am(X_batch), y_train) / K
                    am.zero_grad(); loss.backward(); opt.step()

            am.eval()
            task_preds = []
            with torch.no_grad():
                for li in range(num_libs):
                    d = Data(x=task_samples[li]['node_features'], edge_index=edge_index)
                    b = Batch.from_data_list([d]).to(device)
                    raw = am(b).item()
                    pv = (raw - move) * y_scaled.item() + y_mean.item()
                    task_preds.append(pv)
            # v6: unflip
            if sign_flip < 0:
                task_preds = [-p for p in task_preds]
            predictions[task_idx] = task_preds

    print(f"  Matched: {matched_count}, Unmatched: {unmatched_count}, "
          f"predictions={len(predictions)}")
    return predictions


def run_predictions_with_lib_support_seq_delay(
    model,
    cell_dataset,
    topology_cache,
    norm_stats,
    device,
    args,
    lib_support_data: Dict,
    data_type: str,
    task_indices: List[int],
):
    """Order-based seq cell delay/transition prediction (parallels constraint variant).

    Background: the base `run_predictions_with_lib_support` builds a
    (output_name, delay_type, slew, load) lookup that collapses N seq-cell arcs
    into a single task_idx per (slew, load).  All arcs share that ONE task's
    GRAPH features.  When the adapter fits the support y the support points
    match, but the V-direction interpolation uses the wrong arc's graph — at
    non-support voltages predictions diverge by tens of ps.

    Fix here: walk the lib tables in builder/lib order and pull one PTH task
    per (slew_idx, load_idx) raster position per table.  Each task carries the
    physical graph features for its own arc, so the adapter sees the correct
    structure.  PTHs usually keep only the "main" timing arc per (cell,
    table_type); we just predict for those (queue runs out → remaining lib
    tables left untouched).

    Returns a {task_idx: [61 lib predictions]} dict — directly compatible with
    `_update_lib_delay_orderbased` and consistent with constraint variant.
    """
    cell_name = cell_dataset.cell_name
    num_libs  = cell_dataset.num_libs
    print(f"Order-based lib_few_shot delay predictions for {cell_name} ({data_type})")

    # Map data_type → lib table types we care about.
    if data_type == 'cell':
        rise_tab = 'cell_rise'; fall_tab = 'cell_fall'
    elif data_type == 'transition':
        rise_tab = 'rise_transition'; fall_tab = 'fall_transition'
    else:
        raise ValueError(f"unsupported data_type={data_type}")

    tables = lib_support_data['tables']
    support_indices = lib_support_data['support_indices']
    relevant = [t for t in tables if t['table_type'] in (rise_tab, fall_tab)]
    print(f"  Lib tables (filtered): {len(relevant)}, support_indices: {support_indices}")

    # rise / fall task queues sorted by task_idx.
    dt_map = {ti: cell_dataset.get_task_info(ti)['delay_type'] for ti in task_indices}
    rise_q = sorted(t for t in task_indices if 'rise' in dt_map[t])
    fall_q = sorted(t for t in task_indices if 'fall' in dt_map[t])
    print(f"  Task queues: rise={len(rise_q)} fall={len(fall_q)}")

    cell_cache = topology_cache[cell_name]
    middle_idx = len(support_indices) // 2
    predictions = {}
    matched = unmatched = 0
    ri = iter(rise_q); fi = iter(fall_q)

    for table_idx, tab in enumerate(relevant):
        n_slews = tab['n_slews']; n_loads = tab['n_loads']
        n_entries = n_slews * n_loads
        support_values_list = tab['support_values']  # [n_entries][n_support]
        q_iter = ri if tab['table_type'] == rise_tab else fi
        index_1 = tab['index_1']; index_2 = tab['index_2']
        out_pin = tab['output_pin']
        dt_match = 'rise' if tab['table_type'] == rise_tab else 'fall'
        if table_idx % 5 == 0:
            print(f"  Table {table_idx}/{len(relevant)}: "
                  f"{tab.get('related_pin','')}->{out_pin} "
                  f"{tab['table_type']} when='{tab.get('when','')}' "
                  f"n_entries={n_entries}", flush=True)

        for k in range(n_entries):
            try:
                task_idx = next(q_iter)
            except StopIteration:
                unmatched += 1
                continue
            matched += 1

            # (Cross-check via task_lookup was removed: for seq cells with
            # multi-arc cell_fall, build_task_lookup collapses N arcs into a
            # single key, so the lookup returns the LAST-overwritten task_idx
            # while the order pop returns the FIRST sorted one — guaranteed
            # mismatch even though the order-based assignment is correct.  The
            # FF/0 smoke test already verified order-based assignment matches
            # validation; rely on that instead of this false-positive check.)
            y = torch.tensor(support_values_list[k], dtype=torch.float32)
            y_mean = y.mean(); y_std = y.std()
            if y_std <= 0:
                # degenerate constant curve — store flat at support value
                predictions[task_idx] = [float(support_values_list[k][0])] * num_libs
                continue
            y_norm = (y - y_mean) / y_std

            task_info = cell_dataset.get_task_info(task_idx)
            delay_type_pth = task_info['delay_type']
            output_name = task_info['output_name']
            if 'output_topologies' in cell_cache and output_name in cell_cache['output_topologies']:
                otopo = cell_cache['output_topologies'][output_name]
                adj = otopo['pull_up' if 'rise' in delay_type_pth else 'pull_down']['adjacency_matrix']
            else:
                adj = cell_cache.get('adjacency_matrix', torch.zeros((1, 1)))
            edge_index = adj.nonzero().t()

            task_samples = []
            for li in range(num_libs):
                s = cell_dataset.get_task_data(task_idx, li)
                s['node_features'] = normalize_node_features(s['node_features'], norm_stats)
                task_samples.append(s)

            # Center prediction.  Match validation: project nominal 0.9V
            # through train-time voltage normalization (mask voltage-bearing
            # nodes only).
            cnf = task_samples[support_indices[0]]['node_features'].clone()
            _vs = norm_stats['node_features']['voltage']
            if 'method' in _vs and _vs['method'] == 'minmax_positive':
                _eps = _vs.get('epsilon', 0.01)
                _norm_nom = _eps + (0.9 - _vs['min']) / (_vs['max'] - _vs['min']) * (1 - _eps)
            else:
                _norm_nom = (0.9 - _vs['mean']) / _vs['std']
            _vmask = cnf[:, 4] != 0
            cnf[_vmask, 4] = _norm_nom
            cd = Data(x=cnf, edge_index=edge_index)
            cb = Batch.from_data_list([cd]).to(device)
            model.eval()
            with torch.no_grad():
                center = model(cb).item()

            sp = []
            for sidx in support_indices:
                d = Data(x=task_samples[sidx]['node_features'], edge_index=edge_index)
                b = Batch.from_data_list([d]).to(device)
                with torch.no_grad():
                    sp.append(model(b).item())
            sp = torch.tensor(sp)
            min_val = sp.min().item(); max_val = sp.max().item()
            if abs(max_val - min_val) <= 1e-8:
                predictions[task_idx] = [float(support_values_list[k][0])] * num_libs
                continue

            grad = (y_norm.max().item() - y_norm.min().item()) / (max_val - min_val)
            move = center - y_norm[middle_idx].item() / grad
            y_scaled = y_std * grad
            y_train = ((y - y_mean) / y_scaled + move).to(device).view(-1, 1)

            am = create_maml_gcn_model(
                node_features=model.convs[0].lin.weight.shape[1],
                pooling=model.pooling_type, output_dim=1, dropout=0.0,
                conv_hidden_dim=model.conv_hidden_dim, num_conv_layers=model.num_conv_layers,
                fc_hidden_dim=model.fc_hidden_dim, num_fc_layers=model.num_fc_layers,
            ).to(device)
            am.load_state_dict(model.state_dict())

            sdl = [Data(x=task_samples[sidx]['node_features'], edge_index=edge_index)
                   for sidx in support_indices]
            X_batch = Batch.from_data_list(sdl).to(device)
            criterion = torch.nn.MSELoss(); K = len(support_indices)
            loss = criterion(am(X_batch), y_train) / K
            if args.adaptation_method == 'adam' or \
               (args.adaptation_method == 'selective_adam' and loss > 1e-4):
                inner_lr = getattr(args, 'inner_lr', 3e-4)
                inner_steps = getattr(args, 'inner_steps_adapt', 40)
                opt = torch.optim.Adam(am.parameters(), lr=inner_lr, weight_decay=1e-4)
                for _ in range(inner_steps):
                    loss = criterion(am(X_batch), y_train) / K
                    am.zero_grad(); loss.backward(); opt.step()

            am.eval()
            task_preds = []
            with torch.no_grad():
                for li in range(num_libs):
                    d = Data(x=task_samples[li]['node_features'], edge_index=edge_index)
                    b = Batch.from_data_list([d]).to(device)
                    raw = am(b).item()
                    pv = (raw - move) * y_scaled.item() + y_mean.item()
                    task_preds.append(pv)
            predictions[task_idx] = task_preds

    print(f"  matched={matched} unmatched={unmatched} predictions={len(predictions)}")
    return predictions


def _update_lib_delay_orderbased(
    parser: 'LibFileParser',
    cell_name: str,
    predictions: Dict,
    cell_dataset: 'CellTestDataset',
    lib_idx: int,
    data_type: str,
    collect_comparison: bool = False,
):
    """Order-based write for seq cell delay / transition LUTs.

    Mirrors `_update_lib_constraint_orderbased`.  Walks the lib's
    cell_rise / cell_fall (or rise_transition / fall_transition) tables in
    lib order; for each table pops the next 49 tasks from the matching
    rise/fall queue (sorted by task_idx) and writes the prediction at the
    (slew_idx, load_idx) raster position.  Tables left without an assigned
    queue task remain untouched.
    """
    if data_type == 'cell':
        rise_tab = 'cell_rise'; fall_tab = 'cell_fall'
    elif data_type == 'transition':
        rise_tab = 'rise_transition'; fall_tab = 'fall_transition'
    else:
        raise ValueError(f"unsupported data_type={data_type}")

    # Build rise / fall task queues from the (already filtered) predictions dict.
    rise_q, fall_q = [], []
    for pred_key, lib_preds in predictions.items():
        task_idx = pred_key[0] if isinstance(pred_key, tuple) else pred_key
        info = cell_dataset.get_task_info(task_idx)
        dt = info.get('delay_type', '')
        if 'rise' in dt: rise_q.append((task_idx, lib_preds))
        elif 'fall' in dt: fall_q.append((task_idx, lib_preds))
    rise_q.sort(key=lambda x: x[0]); fall_q.sort(key=lambda x: x[0])
    rise_iter = iter(rise_q); fall_iter = iter(fall_q)

    tables = parser.find_timing_tables()
    cell_tables = [t for t in tables if t['cell_name'] == cell_name]
    replacements = []
    matched_count = 0; unmatched_count = 0
    comparison_data = []

    for table in cell_tables:
        table_type = table['type']
        if table_type not in (rise_tab, fall_tab):
            continue
        original_values = table.get('original_values', [])
        index_1 = table.get('index_1') or []
        index_2 = table.get('index_2') or []
        n_rows = len(index_1) if index_1 else 7
        n_cols = len(index_2) if index_2 else 7
        n_entries = n_rows * n_cols
        if len(original_values) != n_entries:
            continue

        q_iter = rise_iter if table_type == rise_tab else fall_iter
        new_values = []
        for k in range(n_entries):
            try:
                task_idx, lib_preds = next(q_iter)
            except StopIteration:
                new_values.append(original_values[k])
                unmatched_count += 1
                continue
            new_values.append(float(lib_preds[lib_idx]))
            matched_count += 1
        if len(new_values) == n_entries:
            replacements.append((table, new_values, original_values))

    replacements.sort(key=lambda x: x[0]['values_start'], reverse=True)
    for table, new_values, orig_values in replacements:
        parser.replace_timing_values(table, new_values)
        if collect_comparison:
            n_cols = len(table.get('index_2') or [7])
            for idx, (orig, pred) in enumerate(zip(orig_values, new_values)):
                comparison_data.append({
                    'cell_name':   cell_name,
                    'output_pin':  table['output_pin'],
                    'input_pin':   table.get('related_pin', ''),
                    'when':        table.get('when', ''),
                    'table_type':  table['type'],
                    'index':       idx,
                    'row':         idx // n_cols,
                    'col':         idx % n_cols,
                    'original':    orig,
                    'predicted':   pred,
                    'error':       pred - orig,
                    'percent_error': ((pred - orig) / orig * 100) if orig != 0 else 0,
                })

    return len(replacements), comparison_data, matched_count, unmatched_count


def get_timing_table_type(delay_type: str, data_type: str) -> str:
    """
    Map delay_type and data_type to lib file timing table type.

    In stage_aware mode:
    - delay_type="rise_transition" means pull-up path (output rising)
    - delay_type="fall_transition" means pull-down path (output falling)

    data_type determines which table to use:
    - data_type="cell": cell_rise or cell_fall
    - data_type="transition": rise_transition or fall_transition
    """
    if data_type == 'cell':
        if 'rise' in delay_type:
            return 'cell_rise'
        else:
            return 'cell_fall'
    else:  # transition
        if 'rise' in delay_type:
            return 'rise_transition'
        else:
            return 'fall_transition'


def generate_lib_files_for_cell(
    cell_name: str,
    predictions: Dict,
    cell_dataset: CellTestDataset,
    lib_base_path: str,
    output_base_path: str,
    test_folder: str,
    data_type: str,
):
    """
    Generate new lib files with predicted values for one cell.
    Uses value-based matching to correctly map predictions to timing tables.

    Args:
        cell_name: Cell name
        predictions: Dict {task_idx: [lib_predictions]} for processed tasks
        cell_dataset: Cell test dataset
        lib_base_path: Base path to lib files
        output_base_path: Output directory
        test_folder: Test folder name (e.g., TSMC_FF_0)
        data_type: 'cell' or 'transition'
    """
    # Get num_libs from first prediction
    first_task_idx = next(iter(predictions.keys()))
    num_libs = len(predictions[first_task_idx])

    print(f"\nGenerating lib files for {cell_name}")
    print(f"  Predictions: {len(predictions)} tasks x {num_libs} libs")
    print(f"  Test folder: {test_folder}")
    print(f"  Using value-based matching for correct table mapping")

    # Get lib folder path
    lib_folder_path = os.path.join(lib_base_path, test_folder)
    output_folder_path = os.path.join(output_base_path, test_folder)

    if not os.path.exists(lib_folder_path):
        print(f"  Lib folder not found: {lib_folder_path}")
        return

    # Create output folder
    os.makedirs(output_folder_path, exist_ok=True)

    # Get lib files (sorted)
    lib_files = sorted([f for f in os.listdir(lib_folder_path) if f.endswith('.lib')])
    print(f"  Found {len(lib_files)} lib files, processing {min(num_libs, len(lib_files))}")

    total_tables = 0
    total_matched = 0
    total_unmatched = 0

    # Process each lib file using value-based matching
    for lib_idx in range(min(num_libs, len(lib_files))):
        lib_file = lib_files[lib_idx]
        src_path = os.path.join(lib_folder_path, lib_file)
        dst_path = os.path.join(output_folder_path, lib_file)

        # Copy lib file
        shutil.copy2(src_path, dst_path)

        # Parse and update using value-based matching
        parser = LibFileParser(dst_path)
        tables_updated, _, matched, unmatched = update_lib_file_for_cell(
            parser, cell_name, predictions, cell_dataset, lib_idx, data_type
        )
        parser.save(dst_path)
        total_tables += tables_updated
        total_matched += matched
        total_unmatched += unmatched

        if lib_idx == 0:
            print(f"  Lib 0: {tables_updated} timing tables updated")

    print(f"  Generated {min(num_libs, len(lib_files))} lib files in {output_folder_path}")
    print(f"  Total tables: {total_tables}")
    print(f"  Match statistics: {total_matched} matched, {total_unmatched} unmatched")
    if total_matched + total_unmatched > 0:
        match_rate = total_matched / (total_matched + total_unmatched) * 100
        print(f"  Match rate: {match_rate:.2f}%")


def update_lib_file_for_cell(
    parser: LibFileParser,
    cell_name: str,
    predictions: Dict,
    cell_dataset: CellTestDataset,
    lib_idx: int,
    data_type: str,
    collect_comparison: bool = False,
) -> tuple:
    """
    Update a single lib file with predictions for one cell.
    Uses slew/load-based matching to find correct table positions.
    Returns tuple of (number of tables updated, comparison_data list).
    comparison_data contains dicts with original/predicted values if collect_comparison=True.

    Note: predictions dict uses (task_idx, related_pin, when) as key to distinguish
    different timing arcs that share the same task_idx.
    """
    CONSTRAINT_CATS = ('setup','hold','recovery','removal','non_seq_setup','non_seq_hold')
    is_constraint_mode = data_type in CONSTRAINT_CATS
    if is_constraint_mode:
        return _update_lib_constraint_orderbased(
            parser=parser, cell_name=cell_name, predictions=predictions,
            cell_dataset=cell_dataset, lib_idx=lib_idx, data_type=data_type,
            collect_comparison=collect_comparison,
        )
    # Seq cell delay / transition: predictions keyed by plain task_idx (from
    # run_predictions_with_lib_support_seq_delay) → route to order-based write
    # so each task lands at its own (table, entry) without the multi-arc graph
    # collision the slew/load lookup suffers from.
    sample_key = next(iter(predictions.keys())) if predictions else None
    if data_type in ('cell', 'transition') and sample_key is not None and not isinstance(sample_key, tuple):
        return _update_lib_delay_orderbased(
            parser=parser, cell_name=cell_name, predictions=predictions,
            cell_dataset=cell_dataset, lib_idx=lib_idx, data_type=data_type,
            collect_comparison=collect_comparison,
        )

    comparison_data = []

    # Build a lookup: (output_name, related_pin, when, delay_type, slew, load) -> prediction
    slew_load_to_pred = {}
    for pred_key in predictions.keys():
        # pred_key is (task_idx, related_pin, when) tuple
        if isinstance(pred_key, tuple) and len(pred_key) == 3:
            task_idx, related_pin, when_cond = pred_key
        elif isinstance(pred_key, tuple) and len(pred_key) == 2:
            # Fallback for old format (task_idx, related_pin)
            task_idx, related_pin = pred_key
            when_cond = ''
        else:
            # Fallback for oldest format (just task_idx)
            task_idx = pred_key
            related_pin = ''
            when_cond = ''

        pred_value = predictions[pred_key][lib_idx]
        info = cell_dataset.get_task_info(task_idx)

        # Get slew and load from node_features
        sample = cell_dataset.get_task_data(task_idx, 0)
        node_features = sample['node_features']
        slew_vals = node_features[:, 5][node_features[:, 5] != 0]
        load_vals = node_features[:, 6][node_features[:, 6] != 0]
        slew = round(slew_vals[0].item(), 6) if len(slew_vals) > 0 else 0
        load = round(load_vals[0].item(), 6) if len(load_vals) > 0 else 0

        # Include related_pin and when in key for proper arc matching
        key = (info['output_name'], related_pin, when_cond, info['delay_type'], slew, load)
        slew_load_to_pred[key] = pred_value

    # Find timing tables for this cell
    tables = parser.find_timing_tables()
    cell_tables = [t for t in tables if t['cell_name'] == cell_name]

    # Collect all replacements first
    replacements = []  # List of (table, new_values)
    matched_count = 0
    unmatched_count = 0

    for table in cell_tables:
        output_pin = table['output_pin']
        related_pin = table.get('related_pin', '')
        when_cond = table.get('when', '')
        table_type = table['type']

        # Map table_type to delay_type for matching.
        # Constraint variant: rise_constraint / fall_constraint additionally
        # require the timing_type prefix to match the active category.
        timing_type = table.get('timing_type') or ''
        CONSTRAINT_CATS = ('setup','hold','recovery','removal','non_seq_setup','non_seq_hold')
        if table_type == 'cell_rise':
            delay_type_match = 'rise' if data_type in ['cell', 'all'] else None
        elif table_type == 'cell_fall':
            delay_type_match = 'fall' if data_type in ['cell', 'all'] else None
        elif table_type == 'rise_transition':
            delay_type_match = 'rise' if data_type in ['transition', 'all'] else None
        elif table_type == 'fall_transition':
            delay_type_match = 'fall' if data_type in ['transition', 'all'] else None
        elif table_type in ('rise_constraint', 'fall_constraint'):
            if data_type not in CONSTRAINT_CATS:
                continue
            if not timing_type.startswith(data_type + '_'):
                continue
            delay_type_match = 'rise' if table_type == 'rise_constraint' else 'fall'
        else:
            continue

        if delay_type_match is None:
            continue

        # Get original values and index values from this table
        original_values = table.get('original_values', [])
        index_1 = table.get('index_1') or []  # slew values
        index_2 = table.get('index_2') or []  # load values

        if len(original_values) != 49:
            continue

        # Use default 7x7 indices if not available
        n_slews = len(index_1) if index_1 else 7
        n_loads = len(index_2) if index_2 else 7

        # Match each (slew, load) position to a task and get prediction
        new_values = []
        all_matched = True

        for val_idx, orig_val in enumerate(original_values):
            slew_idx = val_idx // n_loads
            load_idx = val_idx % n_loads

            slew = round(index_1[slew_idx], 6) if slew_idx < len(index_1) else 0
            load = round(index_2[load_idx], 6) if load_idx < len(index_2) else 0

            # Try to find matching prediction by (output, related_pin, when, delay_type, slew, load)
            matched = False
            for d_type in [f'{delay_type_match}_transition', f'cell_{delay_type_match}',
                          delay_type_match]:
                # Try with full key (including when condition)
                key = (output_pin, related_pin, when_cond, d_type, slew, load)
                if key in slew_load_to_pred:
                    pred_value = slew_load_to_pred[key]
                    new_values.append(pred_value)
                    matched = True
                    matched_count += 1
                    break

            if not matched:
                # Fallback: search with tolerance, matching when condition if present
                found = False
                for (out_name, rel_pin, when_key, d_type, s, l), pred_val in slew_load_to_pred.items():
                    if out_name == output_pin and rel_pin == related_pin and delay_type_match in d_type:
                        # Match when condition: both empty, or both equal
                        when_matches = (not when_cond and not when_key) or (when_cond == when_key)
                        if when_matches and abs(s - slew) < 1e-5 and abs(l - load) < 1e-5:
                            new_values.append(pred_val)
                            found = True
                            matched_count += 1
                            break

                if not found:
                    # Keep original value if no match
                    new_values.append(orig_val)
                    all_matched = False
                    unmatched_count += 1

        if len(new_values) == 49:
            replacements.append((table, new_values, original_values))

    # Sort by line number descending (process from end of file backwards)
    replacements.sort(key=lambda x: x[0]['values_start'], reverse=True)

    # Apply replacements and collect comparison data
    for table, new_values, orig_values in replacements:
        parser.replace_timing_values(table, new_values)

        # Collect comparison data if requested
        if collect_comparison:
            for idx, (orig, pred) in enumerate(zip(orig_values, new_values)):
                comparison_data.append({
                    'cell_name': cell_name,
                    'output_pin': table['output_pin'],
                    'input_pin': table.get('related_pin', ''),
                    'table_type': table['type'],
                    'index': idx,
                    'row': idx // 7,
                    'col': idx % 7,
                    'original': orig,
                    'predicted': pred,
                    'error': pred - orig,
                    'percent_error': ((pred - orig) / orig * 100) if orig != 0 else 0,
                })

    return len(replacements), comparison_data, matched_count, unmatched_count


def _update_lib_constraint_orderbased(
    parser: LibFileParser,
    cell_name: str,
    predictions: Dict,
    cell_dataset: CellTestDataset,
    lib_idx: int,
    data_type: str,
    collect_comparison: bool = False,
) -> tuple:
    """
    Constraint variant of update_lib_file_for_cell.

    The constraint test PTH drops per-task metadata (related_pin, when,
    related_slew, constrained_slew) at save time and stores only delay_type
    ('rise_transition' / 'fall_transition') and output_name ('Q').  That kills
    the (output_pin, related_pin, when, slew, load) keying the cell-delay path
    relies on.  We instead use **order-based matching** that mirrors the
    builder's iteration order:

      For one filtered (corner, temp) bucket the PTH stores tasks in lib
      iteration order; within each (cell, timing_arc) the inner loop is
      `for ri (related_slew_idx): for ci (constrained_slew_idx)`.  So tasks
      group as [rise_arc1(9) → fall_arc1(9) → rise_arc2(9) → fall_arc2(9) → …]
      when iterating the lib text, where the rise/fall pair belongs to the
      same `when` clause.

    Algorithm:
      1. Walk the lib's timing tables in text order, picking only the ones
         whose `timing_type` starts with `data_type + '_'` (e.g. 'setup_').
      2. For each such table, consume the next prediction from the matching
         delay_type queue, with 9 (= 3×3) entries written per table.
      3. Predictions are written into the lib's `values(...)` block in raster
         order (ri × n_cols + ci).
    """
    comparison_data = []

    # Build delay_type → ordered queue of (pred_key, lib_predictions)
    rise_queue, fall_queue = [], []
    for pred_key, lib_preds in predictions.items():
        # pred_key may be int task_idx (run_predictions_with_adaptation) or a
        # tuple (run_predictions_with_lib_support) — we only need task_idx.
        if isinstance(pred_key, tuple):
            task_idx = pred_key[0]
        else:
            task_idx = pred_key
        info = cell_dataset.get_task_info(task_idx)
        dt = info.get('delay_type', '')
        entry = (task_idx, lib_preds)
        if 'rise' in dt:
            rise_queue.append(entry)
        elif 'fall' in dt:
            fall_queue.append(entry)
    rise_queue.sort(key=lambda e: e[0])
    fall_queue.sort(key=lambda e: e[0])
    rise_iter = iter(rise_queue)
    fall_iter = iter(fall_queue)

    tables = parser.find_timing_tables()
    cell_tables = [t for t in tables if t['cell_name'] == cell_name]

    replacements = []
    matched_count = 0
    unmatched_count = 0

    for table in cell_tables:
        table_type = table['type']
        if table_type not in ('rise_constraint', 'fall_constraint'):
            continue
        timing_type = table.get('timing_type') or ''
        if not timing_type.startswith(data_type + '_'):
            continue

        original_values = table.get('original_values', [])
        index_1 = table.get('index_1') or []
        index_2 = table.get('index_2') or []
        n_rows = len(index_1) if index_1 else 3
        n_cols = len(index_2) if index_2 else 3
        n_entries = n_rows * n_cols
        if len(original_values) != n_entries:
            continue

        queue_iter = rise_iter if table_type == 'rise_constraint' else fall_iter
        new_values = []
        all_matched = True
        # v6: predictions are already in original sign space (per-task sign_flip
        # undone inside run_predictions_with_adaptation).  Just write through.
        for k in range(n_entries):
            try:
                task_idx, lib_preds = next(queue_iter)
            except StopIteration:
                # No more predictions in this queue — keep original
                new_values.append(original_values[k])
                all_matched = False
                unmatched_count += 1
                continue
            pv = float(lib_preds[lib_idx])
            new_values.append(pv)
            matched_count += 1

        if len(new_values) == n_entries:
            replacements.append((table, new_values, original_values))

    # Process from end of file backwards so line offsets stay valid.
    replacements.sort(key=lambda x: x[0]['values_start'], reverse=True)

    for table, new_values, orig_values in replacements:
        parser.replace_timing_values(table, new_values)
        if collect_comparison:
            n_cols = len(table.get('index_2') or [3])
            for idx, (orig, pred) in enumerate(zip(orig_values, new_values)):
                comparison_data.append({
                    'cell_name':   cell_name,
                    'output_pin':  table['output_pin'],
                    'input_pin':   table.get('related_pin', ''),
                    'when':        table.get('when', ''),
                    'timing_type': table.get('timing_type', ''),
                    'table_type':  table['type'],
                    'index':       idx,
                    'row':         idx // n_cols,
                    'col':         idx % n_cols,
                    'original':    orig,
                    'predicted':   pred,
                    'error':       pred - orig,
                    'percent_error': ((pred - orig) / orig * 100) if orig != 0 else 0,
                })

    return len(replacements), comparison_data, matched_count, unmatched_count


def generate_unified_lib_files(
    all_predictions: Dict[str, Dict],
    all_cell_datasets: Dict[str, CellTestDataset],
    lib_base_path: str,
    output_path: str,
    test_folder: str,
    data_type: str,
):
    """
    Generate unified lib files with predictions for all cells combined.

    Args:
        all_predictions: Dict {cell_name: {task_idx: [lib_predictions]}}
        all_cell_datasets: Dict {cell_name: CellTestDataset}
        lib_base_path: Base path to lib files
        output_path: Output directory
        test_folder: Test folder name (e.g., TSMC_FF_0)
        data_type: 'cell' or 'transition'
    """
    # Get num_libs from first cell's predictions
    first_cell = next(iter(all_predictions.keys()))
    first_task_idx = next(iter(all_predictions[first_cell].keys()))
    num_libs = len(all_predictions[first_cell][first_task_idx])

    print(f"\n{'='*60}")
    print(f"Generating unified lib files")
    print(f"{'='*60}")
    print(f"  Cells: {len(all_predictions)}")
    print(f"  Test folder: {test_folder}")

    # Get lib folder path
    lib_folder_path = os.path.join(lib_base_path, test_folder)
    output_folder_path = os.path.join(output_path, test_folder)

    if not os.path.exists(lib_folder_path):
        print(f"  Lib folder not found: {lib_folder_path}")
        return

    os.makedirs(output_folder_path, exist_ok=True)

    lib_files = sorted([f for f in os.listdir(lib_folder_path) if f.endswith('.lib')])
    print(f"  Found {len(lib_files)} lib files, processing {min(num_libs, len(lib_files))}")

    grand_total_tables = 0
    grand_total_matched = 0
    grand_total_unmatched = 0

    # Process each lib file
    for lib_idx in range(min(num_libs, len(lib_files))):
        lib_file = lib_files[lib_idx]
        src_path = os.path.join(lib_folder_path, lib_file)
        dst_path = os.path.join(output_folder_path, lib_file)

        # Copy original lib file
        shutil.copy2(src_path, dst_path)

        # Parse lib file once
        parser = LibFileParser(dst_path)

        lib_tables = 0
        lib_matched = 0
        lib_unmatched = 0
        # Update for each cell
        for cell_name in all_predictions:
            predictions = all_predictions[cell_name]
            cell_dataset = all_cell_datasets[cell_name]
            tables_updated, _, matched, unmatched = update_lib_file_for_cell(
                parser, cell_name, predictions, cell_dataset, lib_idx, data_type
            )
            lib_tables += tables_updated
            lib_matched += matched
            lib_unmatched += unmatched

        # Save updated lib file
        parser.save(dst_path)
        grand_total_tables += lib_tables
        grand_total_matched += lib_matched
        grand_total_unmatched += lib_unmatched

        if lib_idx == 0:
            print(f"  Lib {lib_idx}: {lib_tables} timing tables updated across {len(all_predictions)} cells")

    print(f"  Generated {min(num_libs, len(lib_files))} unified lib files in {output_folder_path}")
    print(f"  Total tables: {grand_total_tables}")
    print(f"  Match statistics: {grand_total_matched} matched, {grand_total_unmatched} unmatched")
    if grand_total_matched + grand_total_unmatched > 0:
        match_rate = grand_total_matched / (grand_total_matched + grand_total_unmatched) * 100
        print(f"  Match rate: {match_rate:.2f}%")


def generate_single_lib_file(
    all_predictions: Dict[str, Dict],
    all_cell_datasets: Dict[str, CellTestDataset],
    lib_file_path: str,
    output_path: str,
    data_type: str,
    target_lib_idx: int,
):
    """
    Generate a single lib file with predictions for all cells.

    This is used when --lib_file is provided directly instead of a test_folder.

    Args:
        all_predictions: Dict {cell_name: {task_idx: [lib_predictions]}}
        all_cell_datasets: Dict {cell_name: CellTestDataset}
        lib_file_path: Path to the input lib file
        output_path: Output file path
        data_type: 'cell' or 'transition'
        target_lib_idx: The lib index corresponding to the target voltage
    """
    print(f"\n{'='*60}")
    print(f"Generating predicted lib file")
    print(f"{'='*60}")
    print(f"  Input: {lib_file_path}")
    print(f"  Output: {output_path}")
    print(f"  Cells: {len(all_predictions)}")
    print(f"  Target lib_idx: {target_lib_idx}")

    # Copy original lib file (only if different from output)
    if os.path.abspath(lib_file_path) != os.path.abspath(output_path):
        shutil.copy2(lib_file_path, output_path)

    # Parse lib file
    parser = LibFileParser(output_path)

    total_tables = 0
    total_matched = 0
    total_unmatched = 0
    all_comparison_data = []

    # Update for each cell
    for cell_name in all_predictions:
        predictions = all_predictions[cell_name]
        cell_dataset = all_cell_datasets[cell_name]

        # Use the target_lib_idx corresponding to the voltage from lib filename
        tables_updated, comparison_data, matched, unmatched = update_lib_file_for_cell(
            parser, cell_name, predictions, cell_dataset,
            lib_idx=target_lib_idx, data_type=data_type, collect_comparison=True
        )
        total_tables += tables_updated
        total_matched += matched
        total_unmatched += unmatched
        all_comparison_data.extend(comparison_data)
        if tables_updated > 0:
            print(f"  {cell_name}: {tables_updated} timing tables updated (matched: {matched}, unmatched: {unmatched})")

    # Save updated lib file
    parser.save(output_path)

    # Save comparison data to CSV
    if all_comparison_data:
        import csv
        csv_path = output_path.replace('.lib', '_predictions.csv')
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['cell_name', 'output_pin', 'input_pin', 'table_type',
                         'index', 'row', 'col', 'original', 'predicted', 'error', 'percent_error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_comparison_data)
        print(f"  Predictions saved to: {csv_path}")

        # Print summary statistics
        import numpy as np
        errors = [d['error'] for d in all_comparison_data]
        percent_errors = [abs(d['percent_error']) for d in all_comparison_data]
        print(f"\n  Prediction Statistics:")
        print(f"    Total values: {len(errors)}")
        print(f"    Mean Absolute Error: {np.mean(np.abs(errors)):.6f}")
        print(f"    Max Absolute Error: {np.max(np.abs(errors)):.6f}")
        print(f"    Mean Percent Error: {np.mean(percent_errors):.4f}%")
        print(f"    Max Percent Error: {np.max(percent_errors):.4f}%")

    print(f"\n  Total: {total_tables} timing tables updated across {len(all_predictions)} cells")
    print(f"  Match statistics: {total_matched} matched, {total_unmatched} unmatched")
    if total_matched + total_unmatched > 0:
        match_rate = total_matched / (total_matched + total_unmatched) * 100
        print(f"  Match rate: {match_rate:.2f}%")
    print(f"  Output saved to: {output_path}")

    return total_tables


def generate_all_voltage_lib_files(
    all_predictions: Dict[str, Dict],
    all_cell_datasets: Dict[str, CellTestDataset],
    lib_dir: str,
    corner: str,
    temperature: int,
    output_dir: str,
    data_type: str,
    num_libs: int = 61,
    seq_lib_dir: Optional[str] = None,
    seq_folder_suffix: str = '',
):
    """
    Generate lib files for all voltage points (0.6V to 1.2V, 61 files).

    Args:
        all_predictions: Dict {cell_name: {task_idx: [lib_predictions]}} - predictions for all 61 voltages
        all_cell_datasets: Dict {cell_name: CellTestDataset}
        lib_dir: Directory containing original lib files (e.g., TSMC_lib_files)
        corner: Process corner (FF, TT, etc.)
        temperature: Temperature (0, 25, 50, 75, 100)
        output_dir: Output directory for generated lib files
        data_type: 'cell' or 'transition'
        num_libs: Number of voltage points (default 61)
        seq_lib_dir: Optional override base dir for sequential cells.
        seq_folder_suffix: Optional suffix appended to seq folder name (e.g., '_x10').
    """
    # Convert temperature to proper string format for directory names
    temp_str = str(int(temperature)) if temperature == int(temperature) else str(temperature).replace('.', 'p')

    print(f"\n{'='*60}")
    print(f"Generating {num_libs} lib files for all voltages")
    print(f"{'='*60}")
    print(f"  Corner: {corner}, Temperature: {temperature}C")
    print(f"  Voltage range: 0.60V to 1.20V (step 0.01V)")
    print(f"  Output directory: {output_dir}")

    # Find source lib directory
    # Try combinational first, then check for sequential
    CONSTRAINT_CATS = ('setup','hold','recovery','removal','non_seq_setup','non_seq_hold')
    is_constraint_mode = data_type in CONSTRAINT_CATS
    seq_base_dir = seq_lib_dir if seq_lib_dir else lib_dir
    comb_dir = os.path.join(lib_dir, f"TSMC_{corner}_{temp_str}")
    seq_dir = os.path.join(seq_base_dir, f"TSMC_{corner}seq_{temp_str}{seq_folder_suffix}")
    if seq_lib_dir or seq_folder_suffix:
        print(f"  Seq lib dir: {seq_dir}")

    # In constraint mode we operate only on the seq lib (no comb merge).  In
    # cell/transition mode we still need the comb lib as the canonical base.
    if is_constraint_mode:
        if not os.path.exists(seq_dir):
            print(f"  ERROR: Sequential lib directory not found: {seq_dir}")
            return 0
    else:
        if not os.path.exists(comb_dir):
            print(f"  ERROR: Combinational lib directory not found: {comb_dir}")
            return 0

    os.makedirs(output_dir, exist_ok=True)

    total_files = 0
    total_tables = 0
    total_matched = 0
    total_unmatched = 0

    for lib_idx in range(num_libs):
        voltage = 60 + lib_idx  # 060 to 120
        voltage_str = f"{voltage:03d}"

        # Find source lib files
        comb_lib_name = f"TSMC_{corner}_{temp_str}_{voltage_str}.lib"
        comb_lib_path = os.path.join(comb_dir, comb_lib_name)
        seq_lib_name = f"TSMC_{corner}_Seq_{temp_str}_{voltage_str}.lib"
        seq_lib_path = os.path.join(seq_dir, seq_lib_name) if os.path.exists(seq_dir) else None

        # Output filename
        output_lib_name = f"predicted_TSMC_{corner}_{temp_str}_{voltage_str}.lib"
        output_lib_path = os.path.join(output_dir, output_lib_name)

        if is_constraint_mode:
            # Constraint runs touch only constraint LUTs on seq cells — copy
            # the seq lib as-is; no comb merge.
            if not seq_lib_path or not os.path.exists(seq_lib_path):
                print(f"  WARNING: Seq lib not found: {seq_lib_path}")
                continue
            shutil.copy2(seq_lib_path, output_lib_path)
        else:
            if not os.path.exists(comb_lib_path):
                print(f"  WARNING: Source lib not found: {comb_lib_path}")
                continue
            if seq_lib_path and os.path.exists(seq_lib_path):
                merge_libs(comb_lib_path, seq_lib_path, output_lib_path)
            else:
                shutil.copy2(comb_lib_path, output_lib_path)

        # Parse and update with predictions
        parser = LibFileParser(output_lib_path)

        lib_tables = 0
        lib_matched = 0
        lib_unmatched = 0
        for cell_name in all_predictions:
            predictions = all_predictions[cell_name]
            cell_dataset = all_cell_datasets[cell_name]

            tables_updated, _, matched, unmatched = update_lib_file_for_cell(
                parser, cell_name, predictions, cell_dataset,
                lib_idx=lib_idx, data_type=data_type, collect_comparison=False
            )
            lib_tables += tables_updated
            lib_matched += matched
            lib_unmatched += unmatched

        parser.save(output_lib_path)
        total_tables += lib_tables
        total_matched += lib_matched
        total_unmatched += lib_unmatched
        total_files += 1

        if lib_idx % 10 == 0 or lib_idx == num_libs - 1:
            print(f"  [{lib_idx+1:2d}/{num_libs}] {voltage/100:.2f}V: {lib_tables} tables updated -> {output_lib_name}")

    print(f"\n  Total: {total_files} lib files generated, {total_tables} timing tables updated")
    print(f"  Match statistics: {total_matched} matched, {total_unmatched} unmatched")
    if total_matched + total_unmatched > 0:
        match_rate = total_matched / (total_matched + total_unmatched) * 100
        print(f"  Match rate: {match_rate:.2f}%")
    return total_tables


def generate_all_voltage_lib_files_combined(
    combined_predictions: Dict[str, Dict[str, Dict]],
    combined_cell_datasets: Dict[str, Dict[str, CellTestDataset]],
    lib_dir: str,
    corner: str,
    temperature: int,
    output_dir: str,
    num_libs: int = 61,
    seq_lib_dir: Optional[str] = None,
    seq_folder_suffix: str = '',
):
    """
    Generate lib files for all voltage points with BOTH cell and transition predictions.

    This function is called once after ALL predictions (both cell delay and transition)
    are collected, ensuring that each lib file contains all predicted values.

    Args:
        combined_predictions: Dict {data_type: {cell_name: {task_idx: [lib_predictions]}}}
            - data_type is 'cell' or 'transition'
        combined_cell_datasets: Dict {data_type: {cell_name: CellTestDataset}}
        lib_dir: Directory containing original lib files (e.g., TSMC_lib_files)
        corner: Process corner (FF, TT, etc.)
        temperature: Temperature (0, 25, 50, 75, 100)
        output_dir: Output directory for generated lib files
        num_libs: Number of voltage points (default 61)
        seq_lib_dir: Optional override base dir for sequential cells.
        seq_folder_suffix: Optional suffix appended to seq folder name (e.g., '_x10').
    """
    # Convert temperature to proper string format for directory names
    temp_str = str(int(temperature)) if temperature == int(temperature) else str(temperature).replace('.', 'p')

    print(f"\n{'='*60}")
    print(f"Generating {num_libs} lib files with ALL predictions (cell + transition)")
    print(f"{'='*60}")
    print(f"  Corner: {corner}, Temperature: {temperature}C")
    print(f"  Voltage range: 0.60V to 1.20V (step 0.01V)")
    print(f"  Output directory: {output_dir}")

    # Print summary of predictions
    for data_type, preds in combined_predictions.items():
        print(f"  {data_type} predictions: {len(preds)} cells")

    # Detect constraint-only mode: every combined data_type is a constraint
    # category → write into seq lib (no comb merge).  Mixed cell+constraint is
    # not supported here; route through separate runs in that case.
    CONSTRAINT_CATS = ('setup','hold','recovery','removal','non_seq_setup','non_seq_hold')
    is_constraint_mode = all(dt in CONSTRAINT_CATS for dt in combined_predictions.keys())
    seq_base_dir = seq_lib_dir if seq_lib_dir else lib_dir
    comb_dir = os.path.join(lib_dir, f"TSMC_{corner}_{temp_str}")
    seq_dir = os.path.join(seq_base_dir, f"TSMC_{corner}seq_{temp_str}{seq_folder_suffix}")
    if seq_lib_dir or seq_folder_suffix:
        print(f"  Seq lib dir: {seq_dir}")
    if is_constraint_mode:
        print(f"  [constraint-only combined mode] writing into seq lib copies (no comb merge)")
        if not os.path.exists(seq_dir):
            print(f"  ERROR: Sequential lib directory not found: {seq_dir}")
            return 0
    else:
        if not os.path.exists(comb_dir):
            print(f"  ERROR: Combinational lib directory not found: {comb_dir}")
            return 0

    os.makedirs(output_dir, exist_ok=True)

    total_files = 0
    total_tables = 0
    total_matched = 0
    total_unmatched = 0

    for lib_idx in range(num_libs):
        voltage = 60 + lib_idx  # 060 to 120
        voltage_str = f"{voltage:03d}"

        # Find source lib files
        comb_lib_name = f"TSMC_{corner}_{temp_str}_{voltage_str}.lib"
        comb_lib_path = os.path.join(comb_dir, comb_lib_name)
        seq_lib_name = f"TSMC_{corner}_Seq_{temp_str}_{voltage_str}.lib"
        seq_lib_path = os.path.join(seq_dir, seq_lib_name) if os.path.exists(seq_dir) else None

        # Output filename
        output_lib_name = f"predicted_TSMC_{corner}_{temp_str}_{voltage_str}.lib"
        output_lib_path = os.path.join(output_dir, output_lib_name)

        if is_constraint_mode:
            # Constraint runs touch only constraint LUTs on seq cells — copy
            # the seq lib as-is; no comb merge.
            if not seq_lib_path or not os.path.exists(seq_lib_path):
                print(f"  WARNING: Seq lib not found: {seq_lib_path}")
                continue
            shutil.copy2(seq_lib_path, output_lib_path)
        else:
            if not os.path.exists(comb_lib_path):
                print(f"  WARNING: Source lib not found: {comb_lib_path}")
                continue
            if seq_lib_path and os.path.exists(seq_lib_path):
                merge_libs(comb_lib_path, seq_lib_path, output_lib_path)
            else:
                shutil.copy2(comb_lib_path, output_lib_path)

        # Parse and update with predictions for ALL data types
        parser = LibFileParser(output_lib_path)

        lib_tables = 0
        lib_matched = 0
        lib_unmatched = 0

        # Update for each data_type (cell and transition)
        for data_type, all_predictions in combined_predictions.items():
            all_cell_datasets = combined_cell_datasets[data_type]

            for cell_name in all_predictions:
                predictions = all_predictions[cell_name]
                cell_dataset = all_cell_datasets[cell_name]

                tables_updated, _, matched, unmatched = update_lib_file_for_cell(
                    parser, cell_name, predictions, cell_dataset,
                    lib_idx=lib_idx, data_type=data_type, collect_comparison=False
                )
                lib_tables += tables_updated
                lib_matched += matched
                lib_unmatched += unmatched

        parser.save(output_lib_path)
        total_tables += lib_tables
        total_matched += lib_matched
        total_unmatched += lib_unmatched
        total_files += 1

        if lib_idx % 10 == 0 or lib_idx == num_libs - 1:
            print(f"  [{lib_idx+1:2d}/{num_libs}] {voltage/100:.2f}V: {lib_tables} tables updated -> {output_lib_name}")

    print(f"\n  Total: {total_files} lib files generated, {total_tables} timing tables updated")
    print(f"  Match statistics: {total_matched} matched, {total_unmatched} unmatched")
    if total_matched + total_unmatched > 0:
        match_rate = total_matched / (total_matched + total_unmatched) * 100
        print(f"  Match rate: {match_rate:.2f}%")
    return total_tables


def main():
    parser = argparse.ArgumentParser(description='Generate lib files with predicted timing values')

    # Cell selection
    parser.add_argument('--cell', type=str, default=None, help='Single cell to process')
    parser.add_argument('--all_cells', action='store_true', help='Process all cells')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['topology_agnostic', 'intra_topology', 'all'],
                        help='Experiment type: topology_agnostic, intra_topology, or all (default: topology_agnostic)')

    # Model configuration
    parser.add_argument('--model_arch', type=str, default='conv64x2_fc256x2',
                        help='Model architecture (default: conv64x2_fc256x2)')
    parser.add_argument('--model_type', type=str, default='maml',
                        choices=['baseline', 'maml'],
                        help='Model type (default: maml)')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='Model iterations (default: 300000)')
    parser.add_argument('--innerdiv', type=int, default=10, help='MAML innerdiv')
    parser.add_argument('--meta', type=int, default=16, help='MAML meta batch')
    parser.add_argument('--inner_steps', type=int, default=1, help='MAML inner steps')

    # Prediction configuration (matching validation)
    parser.add_argument('--mode', type=str, default='interpolation',
                        choices=['interpolation', 'extrapolation'],
                        help='Prediction mode (default: interpolation)')
    parser.add_argument('--adaptation_method', type=str, default='selective_adam',
                        choices=['selective_adam', 'adam'],
                        help='Adaptation method (default: selective_adam)')
    parser.add_argument('--total_points', type=int, default=61,
                        help='Total number of lib points (default: 61)')

    # Data configuration
    parser.add_argument('--data_type', type=str, default='all',
                        choices=['cell', 'transition',
                                 'setup', 'hold', 'recovery', 'removal',
                                 'non_seq_setup', 'non_seq_hold',
                                 'all', 'all_delay', 'all_constraint'],
                        help=('Data type: '
                              'cell / transition (existing delay LUTs) | '
                              'setup / hold / recovery / removal / non_seq_setup / non_seq_hold '
                              '(constraint LUTs — this script\'s reason for being) | '
                              'all = cell + transition (paper default) | '
                              'all_delay = same as all | '
                              'all_constraint = setup + hold + recovery + removal.'))
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                        choices=['stage_aware', 'full_graph'],
                        help='Graph mode (default: stage_aware)')

    # Paths
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN_unified',
                        help='Dataset directory')
    parser.add_argument('--lib_base_path', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/Lib_file_generation',
                        help='Original lib files directory')
    parser.add_argument('--output_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/Lib_file_generation',
                        help='Output directory')
    parser.add_argument('--test_folder', type=str, default=None,
                        help='Test folder name (e.g., TSMC_FF_0)')
    parser.add_argument('--lib_file', type=str, default=None,
                        help='Direct lib file path (extracts PVT from filename, e.g., TSMC_TT_75_100.lib)')

    # GPU
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID')

    # Output mode
    parser.add_argument('--unified', action='store_true',
                        help='Generate unified lib files with all cells in one file (like original)')

    # Precomputed predictions mode
    parser.add_argument('--use_precomputed', action='store_true',
                        help='Use precomputed predictions from .npy files instead of running model')
    parser.add_argument('--pred_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_test_code/gnn/data_result_npy_directory_final',
                        help='Directory containing precomputed pred.npy and act.npy files')

    # Lib file few-shot mode (NEW)
    parser.add_argument('--lib_few_shot', action='store_true',
                        help='Use actual lib files for few-shot support values instead of test data')
    parser.add_argument('--lib_dir', type=str,
                        default='/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files',
                        help='Directory containing lib files at multiple voltages (e.g., TSMC_TT_75/)')
    parser.add_argument('--seq_lib_dir', type=str, default=None,
                        help='Optional override base dir for SEQUENTIAL cells '
                             '(e.g., /.../TSMC_lib_files/TSMC_seq_cell). '
                             'Comb cells still use --lib_dir. If unset, seq also uses --lib_dir.')
    parser.add_argument('--seq_folder_suffix', type=str, default='',
                        help='Optional suffix appended to seq folder name. Default: empty.')
    parser.add_argument('--model_data_type', type=str, default=None,
                        help='Override data_type used for model PTH / train.pth lookup. '
                             'Useful for constraint runs (setup/hold/...) when only the cell-delay '
                             'model is available — pass "cell" to reuse the cell delay model for '
                             'constraint LUT inference (LUT structure is identical 3x3 / 7x7).')
    parser.add_argument('--all_voltages', action='store_true',
                        help='Generate lib files for all 61 voltage points (0.6V to 1.2V)')
    parser.add_argument('--all_corners_temps', action='store_true',
                        help='Run for all corner/temperature combinations (SS,TT,FF,FS,SF x 0,25,50,75,100)')
    parser.add_argument('--corners', type=str, default='SS,TT,FF,FS,SF',
                        help='Comma-separated list of corners (default: SS,TT,FF,FS,SF)')
    parser.add_argument('--temps', type=str, default='0,25,50,75,100',
                        help='Comma-separated list of temperatures (default: 0,25,50,75,100)')
    parser.add_argument('--reference_voltage', type=int, default=100,
                        help='Reference voltage for lib file selection (default: 100 = 1.0V)')

    args = parser.parse_args()

    # Parse model architecture
    arch_match = re.match(r'conv(\d+)x(\d+)_fc(\d+)x(\d+)', args.model_arch)
    if arch_match:
        args.conv_hidden_dim = int(arch_match.group(1))
        args.num_conv_layers = int(arch_match.group(2))
        args.fc_hidden_dim = int(arch_match.group(3))
        args.num_fc_layers = int(arch_match.group(4))
    else:
        print(f"Invalid model architecture format: {args.model_arch}")
        return 1

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Handle --all_corners_temps mode: loop through all combinations
    if args.all_corners_temps:
        corners = [c.strip() for c in args.corners.split(',')]
        temps = [int(t.strip()) for t in args.temps.split(',')]
        ref_voltage = args.reference_voltage

        print(f"\n{'='*80}")
        print(f"ALL CORNERS/TEMPS MODE")
        print(f"{'='*80}")
        print(f"Corners: {corners}")
        print(f"Temperatures: {temps}")
        print(f"Reference voltage: {ref_voltage}")
        print(f"Total combinations: {len(corners) * len(temps)}")
        print(f"{'='*80}\n")

        total_success = 0
        total_failed = 0
        failed_combos = []

        for corner in corners:
            for temp in temps:
                # Construct lib file path
                lib_filename = f"TSMC_{corner}_{temp}_{ref_voltage}.lib"
                lib_file_path = os.path.join(args.lib_base_path, lib_filename)

                # Check if lib file exists, try lib_dir as fallback
                if not os.path.exists(lib_file_path):
                    lib_dir_path = os.path.join(args.lib_dir, f"TSMC_{corner}_{temp}", lib_filename)
                    if os.path.exists(lib_dir_path):
                        lib_file_path = lib_dir_path
                    else:
                        print(f"\n⚠️  Skipping {corner}/{temp}C: lib file not found")
                        print(f"   Tried: {lib_file_path}")
                        print(f"   Tried: {lib_dir_path}")
                        failed_combos.append((corner, temp, "lib file not found"))
                        total_failed += 1
                        continue

                print(f"\n{'='*80}")
                print(f"Processing: {corner} / {temp}C")
                print(f"Lib file: {lib_file_path}")
                print(f"{'='*80}")

                # Build command for subprocess
                cmd = [
                    sys.executable,
                    __file__,
                    '--data_type', args.data_type,
                    '--graph_mode', args.graph_mode,
                    '--gpu', args.gpu,
                    '--lib_file', lib_file_path,
                    '--mode', args.mode,
                    '--lib_dir', args.lib_dir,
                    '--dataset_dir', args.dataset_dir,
                    '--output_dir', args.output_dir,
                    '--model_arch', args.model_arch,
                ]

                if args.lib_few_shot:
                    cmd.append('--lib_few_shot')
                if args.all_voltages:
                    cmd.append('--all_voltages')
                if args.cell:
                    cmd.extend(['--cell', args.cell])

                import subprocess
                result = subprocess.run(cmd, capture_output=False)

                if result.returncode == 0:
                    total_success += 1
                    print(f"✓ {corner}/{temp}C completed successfully")
                else:
                    total_failed += 1
                    failed_combos.append((corner, temp, f"exit code {result.returncode}"))
                    print(f"✗ {corner}/{temp}C failed with exit code {result.returncode}")

        # Summary
        print(f"\n{'='*80}")
        print(f"ALL CORNERS/TEMPS SUMMARY")
        print(f"{'='*80}")
        print(f"Total combinations: {len(corners) * len(temps)}")
        print(f"Successful: {total_success}")
        print(f"Failed: {total_failed}")

        if failed_combos:
            print(f"\nFailed combinations:")
            for corner, temp, reason in failed_combos:
                print(f"  - {corner}/{temp}C: {reason}")

        print(f"{'='*80}")
        return 0 if total_failed == 0 else 1

    # Handle --lib_file mode: auto-select cells from lib file that have test data
    lib_file_mode = args.lib_file is not None
    lib_corner, lib_temp, lib_voltage = None, None, None

    if lib_file_mode:
        if not os.path.exists(args.lib_file):
            print(f"Lib file not found: {args.lib_file}")
            return 1

        # Parse PVT from lib filename
        lib_corner, lib_temp, lib_voltage = parse_lib_file_name(args.lib_file)
        if lib_corner is None:
            print(f"Could not parse PVT from lib filename: {args.lib_file}")
            return 1

        print(f"\n{'='*80}")
        print(f"Lib File Mode")
        print(f"{'='*80}")
        print(f"Input lib file: {args.lib_file}")
        print(f"Parsed PVT: Corner={lib_corner}, Temp={lib_temp}C, Voltage={lib_voltage}V")

        # Extract cells from lib file
        lib_cells = extract_cells_from_lib(args.lib_file)
        print(f"Cells in lib file: {len(lib_cells)}")

        # Get cells that have test data available
        # For data_type='all', check both cell and transition, use intersection
        if args.data_type == 'all':
            cell_available = set(get_available_test_cells(args.dataset_dir, 'cell', args.graph_mode))
            transition_available = set(get_available_test_cells(args.dataset_dir, 'transition', args.graph_mode))
            available_cells = cell_available & transition_available  # Intersection
            print(f"Cells with test data (cell): {len(cell_available)}")
            print(f"Cells with test data (transition): {len(transition_available)}")
            print(f"Cells with both: {len(available_cells)}")
        elif args.data_type == 'all_constraint':
            # Union across the 4 base constraint categories (setup/hold/recovery/removal)
            available_cells = set()
            for c in ('setup', 'hold', 'recovery', 'removal'):
                available_cells |= set(get_available_test_cells(args.dataset_dir, c, args.graph_mode))
            print(f"Cells with test data (any of setup/hold/recovery/removal): {len(available_cells)}")
        elif args.data_type == 'all_delay':
            cell_available = set(get_available_test_cells(args.dataset_dir, 'cell', args.graph_mode))
            transition_available = set(get_available_test_cells(args.dataset_dir, 'transition', args.graph_mode))
            available_cells = cell_available & transition_available
            print(f"Cells with test data (cell ∩ transition): {len(available_cells)}")
        else:
            available_cells = set(get_available_test_cells(args.dataset_dir, args.data_type, args.graph_mode))
            print(f"Cells with test data: {len(available_cells)}")

        # Find intersection
        cell_list = [c for c in lib_cells if c in available_cells]

        # If --cell is specified, filter to only that cell
        if args.cell:
            if args.cell in cell_list:
                cell_list = [args.cell]
                print(f"Filtering to single cell: {args.cell}")
            else:
                print(f"Specified cell '{args.cell}' not found in lib file or test data!")
                return 1

        print(f"Cells to process (intersection): {len(cell_list)}")

        if len(cell_list) == 0:
            print("No cells found in both lib file and test data!")
            return 1

        for cell in cell_list:
            print(f"  - {cell}")
        print(f"{'='*80}")

    else:
        # Original cell list logic
        if args.cell:
            cell_list = [args.cell]
        elif args.all_cells or args.experiment:
            if args.experiment == 'intra_topology':
                cell_list = INTRA_TOPOLOGY_CELLS
            elif args.experiment == 'all':
                cell_list = INTRA_TOPOLOGY_CELLS + TOPOLOGY_AGNOSTIC_CELLS
            else:
                cell_list = TOPOLOGY_AGNOSTIC_CELLS
        else:
            print("Please specify --cell, --all_cells, or --lib_file")
            return 1

        if not args.test_folder:
            print("Please specify --test_folder when not using --lib_file mode")
            return 1

    # Determine data_types to process
    if args.data_type == 'all' or args.data_type == 'all_delay':
        data_types = ['cell', 'transition']
    elif args.data_type == 'all_constraint':
        data_types = ['setup', 'hold', 'recovery', 'removal']
    else:
        data_types = [args.data_type]

    # ─── Constraint-LUT helper maps (constraint variant) ─────────────────────────
    # Categories are exactly the names of the test_by_<category>_stage_aware/ dirs
    # built by tsmc/build_gnn_dataset_process_cached_tsmc_constraint.py.
    CONSTRAINT_CATEGORIES   = {'setup', 'hold', 'recovery', 'removal',
                               'non_seq_setup', 'non_seq_hold'}
    NEGATIVE_SIGN_CATEGORIES = {'hold', 'recovery', 'non_seq_hold'}    # predicted as -|abs|

    def category_from_timing_type(tt):
        """Map a lib `timing_type` token to one of CONSTRAINT_CATEGORIES, or None."""
        if not tt: return None
        for cat in CONSTRAINT_CATEGORIES:
            if tt.startswith(cat + '_'):
                return cat
        return None

    # Validate lib_few_shot requires lib_file mode
    if args.lib_few_shot and not lib_file_mode:
        print("ERROR: --lib_few_shot requires --lib_file to be specified")
        print("  (We need corner and temperature from the lib filename)")
        return 1

    # Determine prediction mode
    if args.use_precomputed:
        pred_mode = "precomputed (from .npy files)"
    elif args.lib_few_shot:
        pred_mode = "lib_few_shot (support from lib files)"
    else:
        pred_mode = "model inference (from test data)"

    print(f"\n{'='*80}")
    print(f"Lib File Generation with GCN Predictions")
    print(f"{'='*80}")
    print(f"Cells: {len(cell_list)}")
    print(f"Model: {args.model_type} {args.model_arch}")
    print(f"Data type(s): {', '.join(data_types)}, Graph mode: {args.graph_mode}")
    print(f"Prediction mode: {pred_mode}")
    print(f"Adaptation: {args.mode}, Method: {args.adaptation_method}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}")

    # Create output directory with model info
    output_subdir = f"{args.model_type}_{args.model_arch}_{args.data_type}_{args.graph_mode}"
    if args.unified:
        output_subdir += "_unified"
    # Add shot count based on mode (for lib_few_shot)
    if args.lib_few_shot:
        if args.mode == 'extrapolation':
            output_subdir += "_3shot"
        else:  # interpolation
            output_subdir += "_5shot"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    # For lib_file mode, prepare output path
    output_lib_path = None
    target_lib_idx = None
    if lib_file_mode:
        lib_basename = os.path.basename(args.lib_file)
        output_lib_path = os.path.join(output_path, f"predicted_{lib_basename}")
        target_lib_idx = voltage_to_lib_idx(lib_voltage)
        print(f"\nVoltage {lib_voltage}V maps to lib_idx {target_lib_idx}")

    # Track grand total across all data_types
    grand_total_tables = 0

    # For all_voltages mode with data_type='all':
    # Collect ALL predictions (both cell and transition) first, then generate lib files once
    combined_predictions = {}  # {data_type: {cell_name: predictions}}
    combined_cell_datasets = {}  # {data_type: {cell_name: CellTestDataset}}

    # Process each data_type
    for data_type_idx, current_data_type in enumerate(data_types):
        print(f"\n{'#'*80}")
        print(f"# Processing data_type: {current_data_type} ({data_type_idx + 1}/{len(data_types)})")
        print(f"{'#'*80}")

        # Initialize model-related variables
        model = None
        topology_cache = None
        norm_stats = None

        # Load model and related data if not using precomputed predictions OR using lib_few_shot
        if not args.use_precomputed or args.lib_few_shot:
            # Allow constraint runs to reuse the cell-delay model (no constraint-specific PTH exists)
            model_dt = args.model_data_type if args.model_data_type else current_data_type
            # Load train data for norm_stats
            train_path = os.path.join(args.dataset_dir, f"train_{model_dt}_{args.graph_mode}.pth")
            print(f"\nLoading train data: {train_path}")
            train_data = torch.load(train_path, weights_only=False, map_location='cpu', mmap=True)
            norm_stats = train_data.get('norm_stats', None)
            cache_path = train_data.get('cache_path', None)

            # Load topology cache
            if cache_path:
                if cache_path.startswith('/mnt/home/'):
                    cache_path = cache_path.replace('/mnt/home/', '/home/')
                if not os.path.exists(cache_path):
                    cache_filename = os.path.basename(cache_path)
                    cache_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"

            print(f"Loading topology cache: {cache_path}")
            topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu', mmap=True)

            # Load model for this data_type
            # Use _pooloutput suffix for transition models
            pool_suffix = "_pooloutput" if model_dt == 'transition' else ""
            arch_suffix = f"_conv{args.conv_hidden_dim}x{args.num_conv_layers}_fc{args.fc_hidden_dim}x{args.num_fc_layers}{pool_suffix}"

            # NOTE: gnn_maml_tsmc_process_final/ and gnn_maml_tsmc_process_checkpoints/
            # contain *different* trained weights despite identical filenames/sizes.
            # TSMC_GCN_topology_validation.py loads from *_checkpoints/ — use the
            # same source so lib-generation predictions reproduce validation RMSE
            # (verified ≈4–6× lower RMSE on the constraint sweep).
            if args.model_type == 'baseline':
                model_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/gnn_baseline_tsmc_process_checkpoints"
                model_filename = f"gnn_baseline_tsmc_process_{model_dt}_{args.graph_mode}_iter{args.num_iterations}{arch_suffix}.pth"
            else:
                model_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/pretrained_models/gnn_maml_tsmc_process_checkpoints"
                model_filename = f"gnn_maml_tsmc_process_{model_dt}_{args.graph_mode}_innerdiv{args.innerdiv}_meta{args.meta}_iter{args.num_iterations}_inner{args.inner_steps}{arch_suffix}.pth"

            model_path = os.path.join(model_dir, model_filename)
            print(f"Loading model: {model_path}")

            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}, skipping {current_data_type}...")
                continue

            checkpoint = torch.load(model_path, weights_only=False, map_location=device)

            # Get node_features from checkpoint
            node_features_dim = checkpoint['model_state_dict']['convs.0.lin.weight'].shape[1]
            config = checkpoint.get('config', {})
            pooling_mode = config.get('pooling', 'mean')

            model = create_maml_gcn_model(
                node_features=node_features_dim,
                pooling=pooling_mode,
                output_dim=1,
                dropout=0.0,
                conv_hidden_dim=args.conv_hidden_dim,
                num_conv_layers=args.num_conv_layers,
                fc_hidden_dim=args.fc_hidden_dim,
                num_fc_layers=args.num_fc_layers
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model = model.to(device)

            print(f"Model loaded: {args.conv_hidden_dim}x{args.num_conv_layers}, fc={args.fc_hidden_dim}x{args.num_fc_layers}{pool_suffix}")
        else:
            print(f"\nUsing precomputed predictions mode - skipping model loading")

        # Collect predictions for this data_type
        all_predictions = {}
        all_cell_datasets = {}

        # Process each cell
        for cell_name in cell_list:
            print(f"\n{'='*60}")
            print(f"Processing cell: {cell_name} ({current_data_type})")
            print(f"{'='*60}")

            # Load cell test data
            cell_path = os.path.join(
                args.dataset_dir,
                f"test_by_{current_data_type}_{args.graph_mode}",
                f"{cell_name}.pth"
            )

            if not os.path.exists(cell_path):
                print(f"Cell data not found: {cell_path}")
                continue

            cell_dataset = CellTestDataset(cell_path)
            print(f"Loaded: {cell_dataset.num_tasks} tasks, {cell_dataset.num_libs} libs")

            # Filter tasks based on mode
            if lib_file_mode:
                filtered_task_indices = get_task_indices_for_pvt(
                    cell_dataset, lib_corner, lib_temp
                )
                print(f"Filtered for PT ({lib_corner}, {lib_temp}C): {len(filtered_task_indices)} tasks")
            else:
                filtered_task_indices = get_task_indices_for_folder(cell_dataset, args.test_folder)
                print(f"Filtered for {args.test_folder}: {len(filtered_task_indices)} tasks")

            if len(filtered_task_indices) == 0:
                print(f"No tasks found, skipping...")
                continue

            # Get predictions: precomputed, lib_few_shot, or by running model
            if args.use_precomputed:
                # Use precomputed predictions from .npy files
                cell_experiment = get_experiment_type_for_cell(cell_name)
                pred_file = find_precomputed_pred_file(
                    pred_dir=args.pred_dir,
                    cell_name=cell_name,
                    data_type=current_data_type,
                    graph_mode=args.graph_mode,
                    mode=args.mode,
                    model_type=args.model_type,
                    experiment=cell_experiment,
                    args=args,
                )

                if pred_file:
                    print(f"  Loading precomputed: {os.path.basename(pred_file)}")
                    predictions = load_precomputed_predictions(
                        pred_file=pred_file,
                        cell_dataset=cell_dataset,
                        num_libs=args.total_points,
                        task_indices=filtered_task_indices,
                    )
                else:
                    print(f"  WARNING: Precomputed file not found for {cell_name}, skipping...")
                    print(f"    Expected pattern: TSMC_GCN_{cell_experiment}_{cell_name}_{current_data_type}_{args.graph_mode}_{args.mode}_*_pred.npy")
                    continue
            elif args.lib_few_shot:
                # Use lib files for few-shot support values
                # Determine support indices based on mode
                if args.mode == 'extrapolation':
                    support_indices = [5, 30, 55]
                else:  # interpolation
                    support_indices = [0, 13, 30, 45, 60]

                # Extract support data from lib files (values + index arrays)
                lib_support_data = extract_lib_support_data(
                    lib_dir=args.lib_dir,
                    corner=lib_corner,
                    temperature=lib_temp,
                    cell_name=cell_name,
                    support_indices=support_indices,
                    num_libs=args.total_points,
                    seq_lib_dir=args.seq_lib_dir,
                    seq_folder_suffix=args.seq_folder_suffix,
                )

                if not lib_support_data:
                    print(f"  WARNING: Could not extract lib support data for {cell_name}, skipping...")
                    continue

                # Run predictions with lib file support values
                predictions = run_predictions_with_lib_support(
                    model=model,
                    cell_dataset=cell_dataset,
                    topology_cache=topology_cache,
                    norm_stats=norm_stats,
                    device=device,
                    args=args,
                    lib_support_data=lib_support_data,
                    data_type=current_data_type,
                    task_indices=filtered_task_indices,
                )
            else:
                # Run predictions with MAML adaptation (original behavior)
                predictions = run_predictions_with_adaptation(
                    model, cell_dataset, topology_cache, norm_stats, device, args,
                    task_indices=filtered_task_indices,
                    data_type=current_data_type,
                )
                # Debug: Print prediction vs ground truth comparison
                print(f"\n  === Debug: {cell_name} Predictions (lib_idx=40, 1.00V) ===")
                print(f"  Number of tasks: {len(predictions)}")
                for task_idx in list(predictions.keys())[:3]:  # 처음 3개만
                    sample = cell_dataset.get_task_data(task_idx, lib_idx=40)  # 1.00V
                    gt = sample['output']
                    pred = predictions[task_idx][40]
                    info = cell_dataset.get_task_info(task_idx)
                    print(f"  Task {task_idx} ({info.get('delay_type', 'N/A')}):")
                    print(f"    Ground truth: {gt:.6f}")
                    print(f"    Prediction:   {pred:.6f}")
                    print(f"    Error:        {pred - gt:.6f} ({(pred - gt) / gt * 100:.2f}%)")
            if lib_file_mode or args.unified:
                # Replace the heavy CellTestDataset (full PTH in RAM) with a
                # lightweight cached version that only retains the per-task
                # metadata the write phase actually consults.  This cuts
                # per-cell RSS growth from ~250 MB to <1 MB, which is the
                # difference between OOM-after-30-cells and a flat sweep.
                cached_dataset = cache_minimal_cell_meta(cell_dataset, predictions)
                all_predictions[cell_name] = predictions
                all_cell_datasets[cell_name] = cached_dataset
                del cell_dataset
            else:
                # Generate lib files per cell (original behavior)
                cell_output_path = os.path.join(output_path, cell_name)
                os.makedirs(cell_output_path, exist_ok=True)

                generate_lib_files_for_cell(
                    cell_name=cell_name,
                    predictions=predictions,
                    cell_dataset=cell_dataset,
                    lib_base_path=args.lib_base_path,
                    output_base_path=cell_output_path,
                    test_folder=args.test_folder,
                    data_type=current_data_type,
                )

            # Per-cell memory cleanup — drops PyTorch graph tensors and Adam
            # optimizer state accumulated during per-task adapt loops, plus
            # the heavy CellTestDataset just freed above.
            try:
                del lib_support_data
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Store predictions for this data_type in combined structure
        if lib_file_mode and args.all_voltages and args.data_type in ('all', 'all_delay', 'all_constraint'):
            # For all_voltages with multi-category data_type, collect all predictions first
            combined_predictions[current_data_type] = all_predictions
            combined_cell_datasets[current_data_type] = all_cell_datasets
            print(f"\n  Stored {len(all_predictions)} cell predictions for {current_data_type}")
        elif lib_file_mode and all_predictions:
            # For single data_type or non-all_voltages mode, generate immediately
            if args.all_voltages:
                # Generate all 61 voltage lib files
                all_voltages_output_dir = os.path.join(output_path, f"all_voltages_{lib_corner}_{lib_temp}")
                tables_count = generate_all_voltage_lib_files(
                    all_predictions=all_predictions,
                    all_cell_datasets=all_cell_datasets,
                    lib_dir=args.lib_dir,
                    corner=lib_corner,
                    temperature=lib_temp,
                    output_dir=all_voltages_output_dir,
                    data_type=current_data_type,
                    num_libs=args.total_points,
                    seq_lib_dir=args.seq_lib_dir,
                    seq_folder_suffix=args.seq_folder_suffix,
                )
                grand_total_tables += tables_count if tables_count else 0
            elif data_type_idx == 0:
                # First data_type: copy original lib file and update
                tables_count = generate_single_lib_file(
                    all_predictions=all_predictions,
                    all_cell_datasets=all_cell_datasets,
                    lib_file_path=args.lib_file,
                    output_path=output_lib_path,
                    data_type=current_data_type,
                    target_lib_idx=target_lib_idx,
                )
                grand_total_tables = tables_count if tables_count else 0
            else:
                # Subsequent data_types: update existing output file
                tables_count = generate_single_lib_file(
                    all_predictions=all_predictions,
                    all_cell_datasets=all_cell_datasets,
                    lib_file_path=output_lib_path,  # Use already generated file
                    output_path=output_lib_path,
                    data_type=current_data_type,
                    target_lib_idx=target_lib_idx,
                )
                grand_total_tables += tables_count if tables_count else 0
        elif args.unified and all_predictions:
            generate_unified_lib_files(
                all_predictions=all_predictions,
                all_cell_datasets=all_cell_datasets,
                lib_base_path=args.lib_base_path,
                output_path=output_path,
                test_folder=args.test_folder,
                data_type=current_data_type,
            )

    # For all_voltages with multi-category data_type: generate lib files ONCE after all predictions collected
    if lib_file_mode and args.all_voltages and args.data_type in ('all', 'all_delay', 'all_constraint') and combined_predictions:
        print(f"\n{'#'*80}")
        print(f"# Generating lib files with combined predictions ({', '.join(combined_predictions.keys())})")
        print(f"{'#'*80}")
        all_voltages_output_dir = os.path.join(output_path, f"all_voltages_{lib_corner}_{lib_temp}")
        tables_count = generate_all_voltage_lib_files_combined(
            combined_predictions=combined_predictions,
            combined_cell_datasets=combined_cell_datasets,
            lib_dir=args.lib_dir,
            corner=lib_corner,
            temperature=lib_temp,
            output_dir=all_voltages_output_dir,
            num_libs=args.total_points,
            seq_lib_dir=args.seq_lib_dir,
            seq_folder_suffix=args.seq_folder_suffix,
        )
        grand_total_tables = tables_count if tables_count else 0

    print(f"\n{'='*80}")
    print(f"Done! Output saved to: {output_path}")
    if lib_file_mode and grand_total_tables > 0:
        if args.all_voltages:
            print(f"Grand Total: {grand_total_tables} timing tables updated across 61 voltage lib files")
        else:
            print(f"Grand Total: {grand_total_tables} timing tables updated (cell + transition)")
    print(f"{'='*80}")

    return 0


def merge_libs(comb_lib_path: str, seq_lib_path: str, output_path: str) -> None:
    """Take the comb lib as the base (it has the library header / template
    definitions / footer).  Extract each `cell (...)` block from the seq lib
    and insert them just before the comb lib's closing `}`.

    Liberty files are well-formed enough that brace counting on each cell
    block boundary works reliably (templates / lu_table_template etc. live in
    the header so the seq lib's cell blocks are self-contained).
    """
    with open(comb_lib_path) as f:
        comb_text = f.read()
    with open(seq_lib_path) as f:
        seq_text = f.read()

    def extract_cell_blocks(text):
        out = []
        i = 0
        while True:
            j = text.find('cell ', i)
            if j < 0: j = text.find('cell(', i)
            if j < 0: break
            line_start = text.rfind('\n', 0, j) + 1
            prefix = text[line_start:j]
            if prefix.strip() != '':
                i = j + 5; continue
            ob = text.find('{', j)
            if ob < 0: break
            depth = 1; k = ob + 1
            while k < len(text) and depth > 0:
                c = text[k]
                if c == '{': depth += 1
                elif c == '}': depth -= 1
                k += 1
            end = k
            if end < len(text) and text[end] == '\n':
                end += 1
            out.append(text[line_start:end])
            i = end
        return out

    seq_cells = extract_cell_blocks(seq_text)
    insert_at = comb_text.rstrip().rfind('}')
    if insert_at < 0:
        out_text = comb_text + '\n'.join(seq_cells)
    else:
        out_text = (comb_text[:insert_at].rstrip() + '\n\n'
                    + '\n'.join(seq_cells).rstrip() + '\n'
                    + comb_text[insert_at:])
    with open(output_path, 'w') as f:
        f.write(out_text)


if __name__ == "__main__":
    sys.exit(main())
