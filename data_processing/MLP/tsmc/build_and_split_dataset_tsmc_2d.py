#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# build_and_split_dataset_separate_cell_types_tsmc.py
import torch
from pathlib import Path
from utils.datasets import libdata
from utils.transform_sample_MAML_tsmc import transform_all_samples
import sys
import re
import argparse
import logging
import importlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_tsmc_folder_name(folder_name):
    """Parse TSMC folder name to extract corner and temperature.

    Supports temperature notations: integer, negative (-25), or `p` for decimal
    point (12p5 → 12.5). Patterns:
    - TSMC_Seq_XX_TEMP (e.g., TSMC_Seq_TT_25)
    - TSMC_XX2_TEMP   (e.g., TSMC_TT2_25, TSMC_SS2_0)
    - TSMC_XX_TEMP    (e.g., TSMC_FF_-25, TSMC_FF_12p5, TSMC_TT_125)
    """
    # Temperature regex piece: optional minus, digits, optional `pN` for decimal
    temp_re = r'(-?\d+(?:p\d+)?)'

    def parse_temp(s: str) -> float:
        return float(s.replace('p', '.'))

    pattern_seq = r'TSMC_Seq_([A-Z]+)_' + temp_re + r'$'
    match = re.match(pattern_seq, folder_name)
    if match:
        return match.group(1), parse_temp(match.group(2))

    pattern_2 = r'TSMC_([A-Z]+)2_' + temp_re + r'$'
    match = re.match(pattern_2, folder_name)
    if match:
        return match.group(1), parse_temp(match.group(2))

    pattern = r'TSMC_([A-Z]+)_' + temp_re + r'$'
    match = re.match(pattern, folder_name)
    if match:
        return match.group(1), parse_temp(match.group(2))

    return None, None

def get_abc_parameters(corner, temperature, param_a_str, param_b_str, param_c_str):
    """
    Map corner and temperature to a,b,c parameters
    All parameters now use nmos/pmos pairs format
    """
    # Parse parameter strings
    param_a_list = [float(x.strip()) for x in param_a_str.split(',')]
    param_b_list = [float(x.strip()) for x in param_b_str.split(',')]
    param_c_list = [float(x.strip()) for x in param_c_str.split(',')]
    
    # Corner to parameter index mapping (for nmos/pmos pairs)
    # All A, B, C parameters are determined by corner, not temperature
    corner_to_idx = {
        'FF': 0,  # FF -> [0,1] pairs
        'TT': 1,  # TT -> [2,3] pairs
        'SS': 2,  # SS -> [4,5] pairs
        'FS': 3,  # FS -> [6,7] pairs
        'SF': 4,  # SF -> [8,9] pairs
    }
    
    # Get parameter index for this corner
    corner_idx = corner_to_idx.get(corner, 0)
    nmos_idx = corner_idx * 2
    pmos_idx = corner_idx * 2 + 1
    
    # Get A parameters (nmos/pmos pairs) - corner based
    a_nmos = param_a_list[nmos_idx] if nmos_idx < len(param_a_list) else param_a_list[0]
    a_pmos = param_a_list[pmos_idx] if pmos_idx < len(param_a_list) else param_a_list[1]
    
    # Get B parameters (nmos/pmos pairs) - corner based  
    b_nmos = param_b_list[nmos_idx] if nmos_idx < len(param_b_list) else param_b_list[0]
    b_pmos = param_b_list[pmos_idx] if pmos_idx < len(param_b_list) else param_b_list[1]
    
    # Get C parameters (nmos/pmos pairs) - corner based
    c_nmos = param_c_list[nmos_idx] if nmos_idx < len(param_c_list) else param_c_list[0]
    c_pmos = param_c_list[pmos_idx] if pmos_idx < len(param_c_list) else param_c_list[1]
    
    return {
        'a_n': a_nmos,
        'a_p': a_pmos,
        'b_n': b_nmos,
        'b_p': b_pmos, 
        'c_n': c_nmos,
        'c_p': c_pmos
    }

def filter_pin_data_by_cell_type(pin_data, target_cell_types):
    """Filter pin data by cell types."""
    test_pins = []
    train_pins = []

    # If target_cell_types contains "NONE", put all data (except INV) in train set
    if "NONE" in target_cell_types or len(target_cell_types) == 0:
        # Filter out INV cells before returning
        filtered_data = [pin for pin in pin_data if 'INV' not in pin.get('cell', '').upper()]
        return [], filtered_data  # Empty test set, all non-INV data goes to train

    for pin in pin_data:
        cell_name = pin.get('cell', '')

        # Skip INV cells entirely
        if 'INV' in cell_name.upper():
            continue

        is_target_cell = False
        
        for target_cell_type in target_cell_types:
            # Extract the cell type with size from full cell name
            cell_prefix = cell_name.split('_')[0] if '_' in cell_name else cell_name
            
            # Direct comparison of cell prefix with target
            if target_cell_type.upper() == cell_prefix.upper():
                is_target_cell = True
                break
            
            # Also check if target is just the logic type without size
            logic_type = re.match(r'([A-Z]+\d*)', cell_name)
            
            if logic_type:
                extracted_type = logic_type.group(1).upper()
                # Check for exact match
                if target_cell_type.upper() == extracted_type:
                    is_target_cell = True
                    break
        
        if is_target_cell:
            test_pins.append(pin)
        else:
            train_pins.append(pin)
    
    return test_pins, train_pins

def process_all_data(data_dirs, output_dir, target_cell_types, param_a, param_b, param_c, delay_type='transition', train_only=False, test_only=False, topology_type='topology_agnostic'):
    """
    Process all TSMC data with abc parameter mapping

    Args:
        delay_type: 'cell' or 'transition' to determine which libdata_extract module to use
    """
    logger.info(f"Processing TSMC data from: {data_dirs}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target cell types: {target_cell_types}")
    logger.info(f"Delay type: {delay_type}")

    # Dynamically import the appropriate libdata_extract module
    if delay_type == 'cell':
        libdata_extract = importlib.import_module('utils.libdata_extract_MAML_cell')
    else:  # transition
        libdata_extract = importlib.import_module('utils.libdata_extract_MAML_transition')

    parse_liberty_pin_blocks = libdata_extract.parse_liberty_pin_blocks
    flatten_pin_data = libdata_extract.flatten_pin_data

    logger.info(f"Using module: libdata_extract_MAML_{delay_type}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect all pin data from all folders
    all_pin_data = []
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning(f"Directory does not exist: {data_dir}")
            continue
            
        # Find all TSMC folders. Temperature suffix may be negative (-25) or
        # decimal with `p` (12p5 = 12.5); the regex must accept both.
        _temp_re = r'-?\d+(?:p\d+)?'
        tsmc_folders = [
            f for f in data_path.iterdir() if f.is_dir() and (
                f.name.startswith('TSMC_Seq_')
                or re.match(rf'TSMC_[A-Z]+2_{_temp_re}$', f.name)
                or re.match(rf'TSMC_[A-Z]+_{_temp_re}$', f.name)
            )
        ]
        logger.info(f"Found {len(tsmc_folders)} TSMC folders in {data_dir}")
        
        # Train-only temperature filter: keep only the canonical 6 train temps
        # to keep the (V × T) plane unseen by test temps (matches GCN 2-D pipeline).
        TRAIN_TEMPS = {-25.0, 12.5, 37.5, 62.5, 87.5, 125.0}
        for folder_path in tsmc_folders:
            folder_name = folder_path.name
            corner, temperature = parse_tsmc_folder_name(folder_name)

            if corner is None or temperature is None:
                logger.warning(f"Skipping folder: {folder_name}")
                continue
            if temperature not in TRAIN_TEMPS:
                logger.info(f"Skipping non-train-temp folder: {folder_name} (temp={temperature})")
                continue

            # Get abc parameters for this corner/temperature
            abc_params = get_abc_parameters(corner, temperature, param_a, param_b, param_c)
            
            logger.info(f"Processing {folder_name}: corner={corner}, temp={temperature}")
            logger.info(f"  Parameters: a_n={abc_params['a_n']:.3f}, a_p={abc_params['a_p']:.3f}, b_n={abc_params['b_n']:.3f}, b_p={abc_params['b_p']:.3f}, c_n={abc_params['c_n']:.3f}, c_p={abc_params['c_p']:.3f}")
            
            # Find .lib files in this folder
            lib_files = list(folder_path.glob("*.lib"))
            logger.info(f"  Found {len(lib_files)} .lib files")
            
            # Store lib file data separately for proper stacking
            folder_lib_data = []
            
            for lib_file in lib_files:
                try:
                    # Read and parse the .lib file
                    with open(lib_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    # Parse pin blocks (same as original)
                    pin_blocks = parse_liberty_pin_blocks(lines)
                    
                    # Flatten pin data (returns tuple: (rows, cap))
                    flattened_pin_data, cap_data = flatten_pin_data(pin_blocks)
                    
                    # Add abc_params and other metadata to each pin data entry
                    for pin_data in flattened_pin_data:
                        # Add the abc parameters
                        pin_data['abc_params'] = abc_params
                        pin_data['corner'] = corner
                        pin_data['temperature_folder'] = temperature
                        pin_data['lib_file'] = lib_file.name  # Add lib file identifier

                        # Set Temperature and other required fields for transform_sample_MAML_invbuf
                        pin_data['Temperature'] = temperature
                        # Voltage is already parsed from lib file in libdata_extract_MAML_cell.py
                        # Keep the parsed value, don't overwrite it
                        # If not present (shouldn't happen), set a fallback value
                        
                        if 'Voltage' not in pin_data or pin_data['Voltage'] == '':
                            pin_data['Voltage'] = 0.8  # Fallback only if missing
                        pin_data['Process'] = 1    # Default process
                        
                        # Ensure required fields exist with dummy values (not used in actual computation)
                        if 'related_pin' not in pin_data:
                            pin_data['related_pin'] = ''  # Not used, dummy value
                        if 'input_port_name' not in pin_data or not isinstance(pin_data.get('input_port_name'), list):
                            pin_data['input_port_name'] = []  # Not used, dummy value
                        if 'size' not in pin_data or pin_data['size'] is None:
                            pin_data['size'] = '1'  # Not used, dummy value
                        if 'input_port_num' not in pin_data or pin_data['input_port_num'] is None:
                            pin_data['input_port_num'] = 1  # Not used, dummy value
                    
                    # Store lib file data separately
                    folder_lib_data.append({
                        'lib_file': lib_file.name,
                        'pin_data': flattened_pin_data,
                        'cap_data': cap_data
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {lib_file}: {e}")
                    continue
            
            # Add folder lib data to global collection
            all_pin_data.append({
                'folder_name': folder_name,
                'corner': corner,
                'temperature': temperature,
                'abc_params': abc_params,
                'lib_files': folder_lib_data
            })
    
    logger.info(f"Collected data from {len(all_pin_data)} folders")

    # NEW APPROACH: Group by condition, then by voltage sweep within each condition
    # Target structure: [samples*conditions, 61_voltage_sweep, features]

    # Sort folders by name for consistent ordering
    all_pin_data.sort(key=lambda x: x['folder_name'])

    # Extract voltage number from lib file name (e.g., TSMC_TT_50_090.lib -> 90)
    def extract_voltage_number(lib_name):
        """Extract voltage number from lib file name for sorting"""
        # Expected format: TSMC_XX_YY_ZZZ.lib where ZZZ is the voltage index
        match = re.search(r'_(\d{3})\.lib$', lib_name)
        if match:
            return int(match.group(1))
        return 0

    # === 2-D V×T grouping: each task is a (cell-sample) at one corner, spanning ALL 6 train temps ===
    # Bucket folders by corner: corner_folders[corner] = list of (temp, folder_data) sorted by temp.
    from collections import defaultdict
    corner_folders = defaultdict(list)
    for folder_data in all_pin_data:
        corner_name = folder_data.get('corner') or parse_tsmc_folder_name(folder_data['folder_name'])[0]
        temp_val   = folder_data.get('temperature') or parse_tsmc_folder_name(folder_data['folder_name'])[1]
        if corner_name is None or temp_val is None:
            logger.warning(f"Skipping unparseable folder: {folder_data['folder_name']}")
            continue
        corner_folders[corner_name].append((temp_val, folder_data))

    # Sort each corner's temps in ascending order so the T axis is well-defined.
    for corner_name in corner_folders:
        corner_folders[corner_name].sort(key=lambda t: t[0])

    # Train uses 6 temps (TSMC train temps). We expect every corner to share the same temp set.
    train_condition_tensors_2d = []   # list of [samples, 61_V, num_temps, 10] tensors (last col = output)
    total_train_samples = 0

    expected_temps = None
    for corner_name in sorted(corner_folders.keys()):
        temp_pairs = corner_folders[corner_name]
        these_temps = [t for (t, _) in temp_pairs]
        if expected_temps is None:
            expected_temps = these_temps
        elif these_temps != expected_temps:
            raise ValueError(
                f"Corner {corner_name} temps {these_temps} != reference {expected_temps}. "
                f"All corners must share the same train temp set for V×T grouping."
            )

        # For each temp, build [samples, 61_V, 10] just like the 1-D builder did.
        # Then stack temps along a new T axis → [samples, 61_V, num_temps, 10].
        per_temp_tensors = []
        for (temp_val, folder_data) in temp_pairs:
            abc_params = folder_data['abc_params']
            lib_files  = folder_data['lib_files']
            lib_files_sorted = sorted(lib_files, key=lambda x: extract_voltage_number(x['lib_file']))

            train_voltage_data = []  # one [samples, 10] tensor per voltage lib

            for lib_file_data in lib_files_sorted:
                lib_name = lib_file_data['lib_file']
                pin_data = lib_file_data['pin_data']
                cap_data = lib_file_data['cap_data']

                if not pin_data:
                    train_voltage_data.append(torch.empty(0, 10, dtype=torch.float32))
                    continue

                _test_pins, train_pins = filter_pin_data_by_cell_type(pin_data, target_cell_types)
                if not train_pins:
                    train_voltage_data.append(torch.empty(0, 10, dtype=torch.float32))
                    continue

                train_datasets = transform_all_samples(
                    train_pins, cap_data, lib_prefix="tsmc", abc_params=abc_params
                )
                if not train_datasets:
                    train_voltage_data.append(torch.empty(0, 10, dtype=torch.float32))
                    continue

                train_inputs  = [s['input']  for s in train_datasets]
                train_outputs = [s['output'] for s in train_datasets]
                lib_train_input  = torch.tensor(train_inputs,  dtype=torch.float32)   # [y, 9]
                lib_train_output = torch.tensor(train_outputs, dtype=torch.float32)   # [y]
                combined_train = torch.cat([lib_train_input, lib_train_output.unsqueeze(1)], dim=1)  # [y, 10]
                train_voltage_data.append(combined_train)

            if not train_voltage_data:
                continue

            max_samples = max(t.shape[0] for t in train_voltage_data)
            if max_samples == 0:
                continue

            padded = []
            for t in train_voltage_data:
                if t.shape[0] < max_samples:
                    pad = torch.zeros(max_samples - t.shape[0], 10, dtype=torch.float32)
                    padded.append(torch.cat([t, pad], dim=0))
                else:
                    padded.append(t)

            # [num_voltages=61, samples, 10] → transpose → [samples, 61, 10]
            per_temp_tensor = torch.stack(padded, dim=0).transpose(0, 1)
            per_temp_tensors.append(per_temp_tensor)

        if len(per_temp_tensors) != len(temp_pairs):
            logger.warning(
                f"Corner {corner_name}: only {len(per_temp_tensors)}/{len(temp_pairs)} temps "
                f"produced data; skipping this corner to keep the V×T grid complete."
            )
            continue

        # Align sample counts across the 6 temps within this corner by truncating to min.
        min_n = min(t.shape[0] for t in per_temp_tensors)
        if min_n == 0:
            continue
        aligned = [t[:min_n] for t in per_temp_tensors]   # each [min_n, 61, 10]
        # Stack along new T axis: [min_n, 61, num_temps, 10]
        corner_tensor = torch.stack(aligned, dim=2)
        train_condition_tensors_2d.append(corner_tensor)
        total_train_samples += min_n
        logger.info(
            f"Corner {corner_name}: aggregated {min_n} samples × 61 V × {len(aligned)} T → "
            f"{corner_tensor.shape}"
        )

    if not train_condition_tensors_2d:
        logger.error("No train tensors produced — aborting.")
        return False

    train_combined_2d = torch.cat(train_condition_tensors_2d, dim=0)   # [N, 61, T, 10]
    train_input  = train_combined_2d[:, :, :, :9]                       # [N, 61, T, 9]
    train_output = train_combined_2d[:, :, :, 9:10]                     # [N, 61, T, 1]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_input_path  = output_dir / f"tsmc_{topology_type}_train_input_{delay_type}_2d.pth"
    train_output_path = output_dir / f"tsmc_{topology_type}_train_output_{delay_type}_2d.pth"
    torch.save(train_input,  train_input_path)
    torch.save(train_output, train_output_path)
    logger.info(f"Saved train (2-D V×T): input {train_input.shape}, output {train_output.shape}")
    logger.info(f"   Input:  {train_input_path}")
    logger.info(f"   Output: {train_output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Build TSMC MLP 2-D V×T dataset (train only)')
    parser.add_argument('--data-dirs', '--data_dirs', dest='data_dirs',
                        nargs='+', required=True, help='TSMC data directories')
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir',
                        type=str, required=True, help='Output directory')
    parser.add_argument('--test-cell-types', '--test_cell_types', dest='test_cell_types',
                        nargs='+', default=[], help='Cell types for test set')
    parser.add_argument('--param-a', '--param_a', dest='param_a',
                        type=str, required=True, help='A parameter values')
    parser.add_argument('--param-b', '--param_b', dest='param_b',
                        type=str, required=True, help='B parameter values')
    parser.add_argument('--param-c', '--param_c', dest='param_c',
                        type=str, required=True, help='C parameter values')
    parser.add_argument('--delay-type', '--delay_type', dest='delay_type',
                        type=str, default='transition', choices=['cell', 'transition'],
                        help='Delay type: cell or transition (default: transition)')
    parser.add_argument('--topology-type', '--topology_type', dest='topology_type',
                        type=str, default='topology_agnostic',
                        choices=['intra_topology', 'topology_agnostic'],
                        help='Output filename topology suffix (default: topology_agnostic)')
    parser.add_argument('--train-only', '--train_only', dest='train_only',
                        action='store_true', help='Create only train dataset (2-D builder is always train-only)')
    parser.add_argument('--test-only', '--test_only', dest='test_only',
                        action='store_true', help='(no-op for 2-D; kept for CLI compatibility)')

    args = parser.parse_args()

    target_cell_types = []
    if args.test_cell_types:
        for cell_type in args.test_cell_types:
            if cell_type != "NONE":
                target_cell_types.append(cell_type)

    success = process_all_data(
        data_dirs=args.data_dirs,
        output_dir=Path(args.output_dir),
        target_cell_types=target_cell_types if target_cell_types else ["NONE"],
        param_a=args.param_a,
        param_b=args.param_b,
        param_c=args.param_c,
        delay_type=args.delay_type,
        train_only=args.train_only,
        test_only=args.test_only,
        topology_type=args.topology_type,
    )
    
    if success:
        logger.info("🎉 Processing completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()