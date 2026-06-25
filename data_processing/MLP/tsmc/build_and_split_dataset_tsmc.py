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

# Train/test temperature split aligned with 1-D and 2-D GNN convention
# (Projects/dataset_all/GNN_dataset_TSMC/train_cell_stage_aware.pth and
#  Projects/dataset_all/gnn_dataset_2d/train_cell_stage_aware_2d.pth).
# Train pool covers the 6 half-integer + corner temps; test pool covers the
# 5 integer temps. The MLP 1-D path historically had only the 5 integer set
# because (a) the regex below was \\d+ (no minus, no 'p5') and (b) the half/
# minus folders were added to TSMC_lib_files months after the MLP dataset was
# first generated. This builder fixes (a) and folds in (b) at build time.
TRAIN_TEMPERATURES = {-25.0, 12.5, 37.5, 62.5, 87.5, 125.0}
TEST_TEMPERATURES  = {0.0, 25.0, 50.0, 75.0, 100.0}


def _parse_temp_token(tok):
    """Convert a temp suffix token like '25', '-25', '12p5' to float."""
    return float(tok.replace('p', '.'))


def parse_tsmc_folder_name(folder_name):
    """Parse TSMC folder name to extract corner and temperature.

    Patched: regex now accepts (i) an optional leading minus sign and
    (ii) the 'p5' half-integer suffix used by ASAP7-style char runs
    (e.g. TSMC_FF_-25, TSMC_FF_12p5, TSMC_FF_125). Temperature is
    returned as float so 12p5 -> 12.5. The unpatched original used
    (\\d+) and silently dropped every minus and 'p5' folder.
    """
    TEMP_RE = r'(-?\d+(?:p\d+)?)'

    # Try TSMC_Seq pattern first
    match = re.match(r'TSMC_Seq_([A-Z]+)_' + TEMP_RE, folder_name)
    if match:
        return match.group(1), _parse_temp_token(match.group(2))

    # Try TSMC_{CORNER}seq_{TEMP} pattern (e.g., TSMC_FFseq_0)
    match = re.match(r'TSMC_([A-Z]+)seq_' + TEMP_RE, folder_name)
    if match:
        return match.group(1), _parse_temp_token(match.group(2))

    # Try TSMC_{CORNER}2_{TEMP} pattern
    match = re.match(r'TSMC_([A-Z]+)2_' + TEMP_RE, folder_name)
    if match:
        return match.group(1), _parse_temp_token(match.group(2))

    match = re.match(r'TSMC_([A-Z]+)_' + TEMP_RE, folder_name)
    if match:
        return match.group(1), _parse_temp_token(match.group(2))

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

def process_all_data(data_dirs, output_dir, target_cell_types, param_a, param_b, param_c, delay_type='transition', train_only=False, test_only=False):
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
            
        # Find all TSMC folders (both TSMC_Seq_ and TSMC_*2_ patterns).
        # Patched: same minus/p5 extension as parse_tsmc_folder_name so the
        # collection step matches what the parsing step will accept.
        tsmc_folders = [f for f in data_path.iterdir() if f.is_dir() and (
            f.name.startswith('TSMC_Seq_')
            or re.match(r'TSMC_[A-Z]+2_-?\d+(?:p\d+)?', f.name)
            or re.match(r'TSMC_[A-Z]+_-?\d+(?:p\d+)?',  f.name)
        )]
        logger.info(f"Found {len(tsmc_folders)} TSMC folders in {data_dir}")
        
        for folder_path in tsmc_folders:
            folder_name = folder_path.name
            corner, temperature = parse_tsmc_folder_name(folder_name)
            
            if corner is None or temperature is None:
                logger.warning(f"Skipping folder: {folder_name}")
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

    # ----------------------------------------------------------------------
    # PATCHED outer-loop (2nd rebuild):
    #   * cell_split temps (-25, 12.5, 37.5, 62.5, 87.5, 125)
    #         -> 44 train cells go to the main train pool
    #         -> 20 test_cell_types are DROPPED entirely (not test, not train).
    #            Only the (test_cell × test_temp) combinations end up in test.
    #   * test_only temps (0, 25, 50, 75, 100)
    #         -> every pin (all cells) is sent to test, grouped by cell name
    #            so each cell receives its own per-cell *.pth files under a
    #            dedicated subdirectory (matches the legacy test_data_dir
    #            layout the validation scripts expect).
    # `transform_sample` (utils/transform_sample_MAML_tsmc.py) now preserves
    # the 'cell' key so we can group post-transform without having to redo
    # parsing per cell.
    # ----------------------------------------------------------------------
    from collections import defaultdict

    train_condition_tensors = []                  # main train pool
    test_condition_tensors_by_cell = defaultdict(list)  # cell_name -> [folder tensors]

    total_train_samples = 0
    total_test_samples = 0

    for folder_data in all_pin_data:
        folder_name = folder_data['folder_name']
        abc_params = folder_data['abc_params']
        lib_files = folder_data['lib_files']
        folder_temp = float(folder_data['temperature'])

        if folder_temp in TEST_TEMPERATURES:
            temp_role = 'test_only'
        elif folder_temp in TRAIN_TEMPERATURES:
            temp_role = 'cell_split'
        else:
            logger.warning(
                f"Folder {folder_name} temp={folder_temp} is in neither train "
                f"nor test temperature set; skipping."
            )
            continue

        logger.info(
            f"Processing folder {folder_name} with {len(lib_files)} lib files... "
            f"(temp={folder_temp}, role={temp_role})"
        )

        # Sort lib files by voltage number — the voltage axis must align
        # across lib files when stacking per-condition tensors below.
        lib_files_sorted = sorted(lib_files, key=lambda x: extract_voltage_number(x['lib_file']))

        train_voltage_data = []  # this folder's main-train per-voltage tensors

        # For test_only folders, we need per-cell, per-voltage tensors. The
        # outer voltage stacking assumes one tensor per voltage step, but
        # each lib file mixes many cells. So we record per-cell entries and
        # interleave with zero placeholders for cells that did not appear at
        # this voltage — then stack on the voltage axis below.
        test_per_cell_voltage_data = defaultdict(list)  # cell -> list aligned by lib file
        all_cells_seen_in_folder = set()

        for vidx, lib_file_data in enumerate(lib_files_sorted):
            lib_name = lib_file_data['lib_file']
            pin_data = lib_file_data['pin_data']
            cap_data = lib_file_data['cap_data']

            if not pin_data:
                logger.warning(f"  No pin data in {lib_name}, skipping...")
                # Preserve alignment with the voltage axis
                if temp_role == 'cell_split' and not test_only:
                    train_voltage_data.append(torch.empty(0, 10, dtype=torch.float32))
                if temp_role == 'test_only' and not train_only:
                    for cell in all_cells_seen_in_folder:
                        test_per_cell_voltage_data[cell].append(torch.empty(0, 10, dtype=torch.float32))
                continue

            if temp_role == 'cell_split':
                # train_pool only — discard the test_cells slice (key change
                # from rebuild #1: those folder×cell pairs are intentionally
                # excluded so the test set carries only the 5 test temps).
                _drop, train_pins = filter_pin_data_by_cell_type(pin_data, target_cell_types)

                if not test_only:
                    if train_pins:
                        ds = transform_all_samples(train_pins, cap_data, lib_prefix="tsmc", abc_params=abc_params)
                        if ds:
                            inp  = torch.tensor([s['input']  for s in ds], dtype=torch.float32)
                            outp = torch.tensor([s['output'] for s in ds], dtype=torch.float32)
                            combined = torch.cat([inp, outp.unsqueeze(1)], dim=1)
                            train_voltage_data.append(combined)
                            total_train_samples += inp.shape[0]
                        else:
                            train_voltage_data.append(torch.empty(0, 10, dtype=torch.float32))
                    else:
                        train_voltage_data.append(torch.empty(0, 10, dtype=torch.float32))

            else:  # temp_role == 'test_only'
                if not train_only:
                    ds = transform_all_samples(pin_data, cap_data, lib_prefix="tsmc", abc_params=abc_params)
                    # Group by cell — transform_sample now preserves 'cell'.
                    per_cell_buf = defaultdict(lambda: ([], []))  # cell -> ([inputs], [outputs])
                    for s in ds:
                        cell_name = s.get('cell', '') or 'UNKNOWN'
                        per_cell_buf[cell_name][0].append(s['input'])
                        per_cell_buf[cell_name][1].append(s['output'])

                    # Append one entry per cell. If a cell was already seen at
                    # an earlier voltage and missed this lib file, we will
                    # back-fill with zero placeholders at the stacking step.
                    cells_at_this_voltage = set(per_cell_buf.keys())
                    new_cells = cells_at_this_voltage - all_cells_seen_in_folder
                    for nc in new_cells:
                        # back-fill earlier voltages with empty tensors
                        test_per_cell_voltage_data[nc].extend(
                            [torch.empty(0, 10, dtype=torch.float32)] * vidx
                        )
                    all_cells_seen_in_folder |= cells_at_this_voltage

                    for cell_name, (inps, outs) in per_cell_buf.items():
                        inp  = torch.tensor(inps, dtype=torch.float32)
                        outp = torch.tensor(outs, dtype=torch.float32)
                        combined = torch.cat([inp, outp.unsqueeze(1)], dim=1)
                        test_per_cell_voltage_data[cell_name].append(combined)
                        total_test_samples += inp.shape[0]

                    # Cells previously seen but absent at this voltage: pad
                    missing = all_cells_seen_in_folder - cells_at_this_voltage
                    for mc in missing:
                        test_per_cell_voltage_data[mc].append(torch.empty(0, 10, dtype=torch.float32))

        # Stack train per-voltage tensors for this folder (cell_split branch).
        if temp_role == 'cell_split' and train_voltage_data and not test_only:
            max_samples = max(t.shape[0] for t in train_voltage_data)
            if max_samples > 0:
                padded = []
                for t in train_voltage_data:
                    if t.shape[0] < max_samples:
                        pad = torch.zeros(max_samples - t.shape[0], 10, dtype=torch.float32)
                        padded.append(torch.cat([t, pad], dim=0))
                    else:
                        padded.append(t)
                # [61_voltages, samples, 10] -> [samples, 61, 10]
                condition_train_tensor = torch.stack(padded, dim=0).transpose(0, 1)
                train_condition_tensors.append(condition_train_tensor)
                logger.info(
                    f"  Train: {condition_train_tensor.shape[0]} samples × "
                    f"{condition_train_tensor.shape[1]} voltages"
                )

        # Stack test per-voltage tensors per cell for this folder (test_only branch).
        if temp_role == 'test_only' and test_per_cell_voltage_data and not train_only:
            for cell_name, voltage_tensors in test_per_cell_voltage_data.items():
                # Some cells may have fewer entries than the number of lib
                # files if they appeared late; pad to full length so the
                # voltage axis matches.
                if len(voltage_tensors) < len(lib_files_sorted):
                    voltage_tensors = voltage_tensors + [
                        torch.empty(0, 10, dtype=torch.float32)
                    ] * (len(lib_files_sorted) - len(voltage_tensors))
                max_samples = max((t.shape[0] for t in voltage_tensors), default=0)
                if max_samples == 0:
                    continue
                padded = []
                for t in voltage_tensors:
                    if t.shape[0] < max_samples:
                        pad = torch.zeros(max_samples - t.shape[0], 10, dtype=torch.float32)
                        padded.append(torch.cat([t, pad], dim=0))
                    else:
                        padded.append(t)
                condition_test_tensor = torch.stack(padded, dim=0).transpose(0, 1)
                test_condition_tensors_by_cell[cell_name].append(condition_test_tensor)

            logger.info(
                f"  Test (per-cell): {len(test_per_cell_voltage_data)} cells in this folder"
            )

    # ---------------------- Save train (main file) ----------------------
    logger.info("Concatenating train conditions...")
    if train_condition_tensors and not test_only:
        train_combined = torch.cat(train_condition_tensors, dim=0)  # [N, 61, 10]
        train_input  = train_combined[:, :, :9]
        train_output = train_combined[:, :, 9:10]
        train_input_path  = output_dir / f"tsmc_topology_agnostic_train_input_{delay_type}.pth"
        train_output_path = output_dir / f"tsmc_topology_agnostic_train_output_{delay_type}.pth"
        torch.save(train_input,  train_input_path)
        torch.save(train_output, train_output_path)
        logger.info(
            f"✅ Saved train dataset: input {tuple(train_input.shape)}, "
            f"output {tuple(train_output.shape)} "
            f"({len(train_condition_tensors)} conditions)"
        )

    # ---------------------- Save test (per-cell subdirs) ----------------------
    logger.info(
        f"Saving per-cell test datasets: {len(test_condition_tensors_by_cell)} cells..."
    )
    if test_condition_tensors_by_cell and not train_only:
        for cell_name, cell_tensors in test_condition_tensors_by_cell.items():
            cell_combined = torch.cat(cell_tensors, dim=0)
            cell_input  = cell_combined[:, :, :9]
            cell_output = cell_combined[:, :, 9:10]
            cell_dir = output_dir / cell_name
            cell_dir.mkdir(parents=True, exist_ok=True)
            cell_input_path  = cell_dir / f"tsmc_merged_test_input_{delay_type}.pth"
            cell_output_path = cell_dir / f"tsmc_merged_test_output_{delay_type}.pth"
            torch.save(cell_input,  cell_input_path)
            torch.save(cell_output, cell_output_path)
            logger.info(
                f"  ✅ {cell_name}: input {tuple(cell_input.shape)}, "
                f"output {tuple(cell_output.shape)} "
                f"({len(cell_tensors)} conditions)"
            )

    
    return True

def main():
    parser = argparse.ArgumentParser(description='Build TSMC dataset with a,b,c parameter mapping')
    parser.add_argument('--data-dirs', nargs='+', required=True, help='TSMC data directories')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--test-cell-types', nargs='+', default=[], help='Cell types for test set')
    parser.add_argument('--param-a', type=str, required=True, help='A parameter values')
    parser.add_argument('--param-b', type=str, required=True, help='B parameter values')
    parser.add_argument('--param-c', type=str, required=True, help='C parameter values')
    parser.add_argument('--delay-type', type=str, default='transition', choices=['cell', 'transition'],
                        help='Delay type: cell or transition (default: transition)')
    parser.add_argument('--train-only', action='store_true', help='Create only train dataset')
    parser.add_argument('--test-only', action='store_true', help='Create only test dataset')
    # Accepted-but-unused: wrappers pass this; saved train/test filenames are
    # hard-coded to topology_agnostic in this builder. Downstream consumers
    # distinguish intra/agnostic via the test_cell_types subset, not via the
    # file name.
    parser.add_argument('--topology-type', type=str, default='agnostic',
                        choices=['intra', 'agnostic'],
                        help=argparse.SUPPRESS)

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
        test_only=args.test_only
    )
    
    if success:
        logger.info("🎉 Processing completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()