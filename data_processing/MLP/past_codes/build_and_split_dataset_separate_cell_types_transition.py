# build_and_split_dataset_separate_cell_types_fixed.py
import torch
from pathlib import Path
from datasets import libdata
from libdata_extract_MAML_transition import parse_liberty_pin_blocks, flatten_pin_data
from transform_sample_MAML_invbuf import transform_all_samples
import sys
import re

def filter_pin_data_by_cell_type(pin_data, target_cell_types):
    """
    Filter pin data by cell types.
    
    Args:
        pin_data: List of pin dictionaries
        target_cell_types: List of target cell types, each can be:
                         - Logic type only (e.g., 'XOR2', 'NAND2')
                         - Full name with size (e.g., 'NAND2xp67', 'INVx2')
    
    Returns:
        test_pins: Pin data for test set (only target cell types)
        train_pins: Pin data for train set (all other cell types)
    """
    test_pins = []
    train_pins = []
    
    # If target_cell_types contains "NONE", put all data in train set
    if "NONE" in target_cell_types or len(target_cell_types) == 0:
        return [], pin_data  # Empty test set, all data goes to train
    
    for pin in pin_data:
        cell_name = pin.get('cell', '')
        
        is_target_cell = False
        
        for target_cell_type in target_cell_types:
            # Extract the cell type with size from full cell name
            cell_prefix = cell_name.split('_')[0] if '_' in cell_name else cell_name
            
            # Direct comparison of cell prefix with target
            if target_cell_type.upper() == cell_prefix.upper():
                is_target_cell = True
                break
            
            # Also check if target is just the logic type without size
            # Extract logic type from cell name (e.g., 'XOR2' from 'XOR2x1_ASAP7_75t_L')
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

def parse_temperature_to_index(temp_str):
    """Map temperature string to index for simple_ folders"""
    # Temperature mapping: '-25', '12p5', '37p5', '62p5', '87p5', '125'
    temp_map = {
        '-25': 0,
        '12p5': 1,
        '37p5': 2,
        '62p5': 3,
        '87p5': 4,
        '125': 5
    }
    if temp_str in temp_map:
        return temp_map[temp_str]
    else:
        # Try to parse as integer for other cases
        try:
            return int(temp_str)
        except ValueError:
            raise ValueError(f"Unknown temperature format: {temp_str}")

def parse_ex2_folder(folder_path):
    """Parse folder name like 'ex2_1_1_1_1', 'simple_0_0_0_125', or 'VDD_0p8_1_1_PTC_130_max' to get a,b,c,d values"""
    folder_name = Path(folder_path).name
    
    # Pattern 1: ex2_<a>_<b>_<c>_<d> or invbuf patterns
    if folder_name.startswith('ex2_') or folder_name.startswith('invbuf'):
        parts = folder_name.split('_')
        # Handle negative temperature case where ex2_0_0_0_-25 becomes ['ex2', '0', '0', '0', '', '25']
        if len(parts) == 6 and parts[4] == '':
            # This is a negative number case
            a = int(parts[1])
            b = int(parts[2])
            c = int(parts[3])
            # For invbuf folders, use temperature mapping
            if folder_name.startswith('invbuf'):
                temp_str = f"-{parts[5]}"
                d = parse_temperature_to_index(temp_str)
            else:
                # For ex2 folders, use negative integer
                d = -int(parts[5])
            return a, b, c, d
        elif len(parts) == 5:
            try:
                a = int(parts[1])
                b = int(parts[2])
                c = int(parts[3])
                # For invbuf folders, handle temperature strings like '125', '12p5', '87p5', etc.
                if folder_name.startswith('invbuf'):
                    temp_str = parts[4]
                    d = parse_temperature_to_index(temp_str)
                else:
                    # For ex2 folders, try to parse as integer
                    d = int(parts[4])
                return a, b, c, d
            except ValueError as e:
                raise ValueError(f"Failed to parse folder '{folder_name}': {e}")
        else:
            raise ValueError(f"Folder name '{folder_name}' doesn't match expected pattern")
    
    # Pattern 2: simple_<a>_<b>_<c>_<d> where d can be like '125', '12p5', '-25', etc.
    elif folder_name.startswith('simple_'):
        parts = folder_name.split('_')
        # Handle negative temperature case
        if len(parts) == 6 and parts[4] == '':
            # This is a negative number case (simple_0_0_0_-25)
            a = int(parts[1])
            b = int(parts[2])
            c = int(parts[3])
            # Map temperature string to index
            temp_str = f"-{parts[5]}"
            d = parse_temperature_to_index(temp_str)
            return a, b, c, d
        elif len(parts) == 5:
            try:
                a = int(parts[1])
                b = int(parts[2])
                c = int(parts[3])
                # Handle temperature strings like '125', '12p5', etc.
                temp_str = parts[4]
                d = parse_temperature_to_index(temp_str)
                return a, b, c, d
            except ValueError as e:
                raise ValueError(f"Failed to parse simple folder '{folder_name}': {e}")
        else:
            raise ValueError(f"Folder name '{folder_name}' doesn't match simple_a_b_c_d pattern")
    
    # Pattern 2: VDD_<voltage>_<a>_<b>_<process>_<temp>_<corner>
    elif folder_name.startswith('VDD_'):
        parts = folder_name.split('_')
        
        if len(parts) < 6:
            raise ValueError(f"Folder name '{folder_name}' doesn't match VDD pattern")
        
        try:
            # Extract voltage (VDD_0p8 -> 0.8)
            voltage_str = parts[1]  # '0p8'
            voltage = float(voltage_str.replace('p', '.'))
            
            # Extract parameters
            a = int(parts[2])  # First number after voltage
            b = int(parts[3])  # Second number
            
            # Process type (PTC, CTC, etc.)
            process = parts[4]
            
            # Temperature
            temp = int(parts[5])
            
            # Map to indices (this mimics the original logic)
            # For ex2 dataset: a and b are direct indices
            c = 0  # Default for process mapping
            d = 0  # Default for temperature mapping
            
            return a, b, c, d
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse VDD folder '{folder_name}': {e}")
    
    else:
        raise ValueError(f"Unknown folder pattern: '{folder_name}'")

def build_split_dataset_by_cell_type(filenames, abc_param_lists, target_cell_types, folder_params,
                                   save_test_input, save_test_output, save_train_input, save_train_output):
    """
    Build dataset with train/test split by cell type using proper lib file stacking
    
    폴더마다 61개 lib file을 개별 처리하고 스택해서 x,61,9 구조 생성
    각 task는 동일한 (slew, load) 조합에서 voltage만 다른 61개 sample로 구성
    """
    
    # Parse folder parameters
    a, b, c, d = folder_params
    
    # Get ABC parameters from folder indices
    # B and C parameters are stored as (nmos, pmos) pairs, so we need to access them correctly
    if (a < len(abc_param_lists[0]) and b*2+1 < len(abc_param_lists[1]) and
        c*2+1 < len(abc_param_lists[2])):
        abc_params = {
            'a': abc_param_lists[0][a],
            'a_n': abc_param_lists[0][a],
            'a_p': abc_param_lists[0][a],
            'b_n': abc_param_lists[1][b*2],     # nmos: even index
            'b_p': abc_param_lists[1][b*2 + 1], # pmos: odd index
            'c_n': abc_param_lists[2][c*2],     # nmos: even index
            'c_p': abc_param_lists[2][c*2 + 1]  # pmos: odd index
        }
        print(f"  📊 ABC parameters: a={abc_params['a']}, b_n={abc_params['b_n']}, b_p={abc_params['b_p']}, c_n={abc_params['c_n']}, c_p={abc_params['c_p']}")
    else:
        print(f"  ❌ Invalid parameter indices: a={a}, b={b}, c={c}")
        return None, None, None, None
    
    # Process each lib file separately to maintain proper structure
    all_test_lib_data = []   # List of tensors from each lib file: [[y,9], [y,9], ...]  
    all_train_lib_data = []  
    all_test_lib_outputs = [] # List of tensors from each lib file: [[y], [y], ...]
    all_train_lib_outputs = []
    
    print(f"  📄 Processing {len(filenames)} lib files individually...")
    
    for i, filename in enumerate(filenames):
        lib_name = Path(filename).name
        print(f"    📄 [{i+1}/{len(filenames)}] {lib_name}")
        
        with open(filename, "r") as f:
            lines = f.readlines()
        
        pin_data = parse_liberty_pin_blocks(lines)
        if not pin_data:
            print(f"      ⚠️ No pin data found, skipping...")
            continue
        
        # Filter by target cell types for this lib file
        test_pins, train_pins = filter_pin_data_by_cell_type(pin_data, target_cell_types)
        
        # Process test data for this lib file
        if test_pins:
            test_flattened, test_cap = flatten_pin_data(test_pins)
            test_datasets = transform_all_samples(test_flattened, test_cap, lib_name, abc_params)
            
            if test_datasets:
                test_inputs = [sample['input'] for sample in test_datasets]
                test_outputs = [sample['output'] for sample in test_datasets]
                lib_test_input = torch.tensor(test_inputs, dtype=torch.float32)  # [y, 9]
                lib_test_output = torch.tensor(test_outputs, dtype=torch.float32)  # [y]
                
                all_test_lib_data.append(lib_test_input)
                all_test_lib_outputs.append(lib_test_output)
                print(f"      ✅ Test: {lib_test_input.shape[0]} samples")
        else:
            print(f"      ⚠️ No test pins found")
        
        # Process train data for this lib file
        if train_pins:
            train_flattened, train_cap = flatten_pin_data(train_pins)
            train_datasets = transform_all_samples(train_flattened, train_cap, lib_name, abc_params)
            
            if train_datasets:
                train_inputs = [sample['input'] for sample in train_datasets]
                train_outputs = [sample['output'] for sample in train_datasets]
                lib_train_input = torch.tensor(train_inputs, dtype=torch.float32)  # [y, 9]
                lib_train_output = torch.tensor(train_outputs, dtype=torch.float32)  # [y]
                
                all_train_lib_data.append(lib_train_input)
                all_train_lib_outputs.append(lib_train_output)
                print(f"      ✅ Train: {lib_train_input.shape[0]} samples")
        else:
            print(f"      ⚠️ No train pins found")
    
    print(f"  📊 Lib file processing summary:")
    print(f"    🎯 Test data from {len(all_test_lib_data)} lib files")
    print(f"    🏋️ Train data from {len(all_train_lib_data)} lib files")
    
    # Stack lib file data to create [samples, lib_files, features] structure
    # 핵심: 각 lib file에서 나온 [y, 9] 텐서들을 dim=1에서 stack하여 [y, 61, 9] 생성
    test_input = None
    test_output = None
    train_input = None
    train_output = None
    
    if all_test_lib_data:
        print(f"  🔗 Stacking test data from {len(all_test_lib_data)} lib files...")
        # Stack along lib file dimension: [y, 9] × N files → [y, N, 9]
        test_input = torch.stack(all_test_lib_data, dim=1)  # [samples, lib_files, features]
        test_output = torch.stack(all_test_lib_outputs, dim=1)  # [samples, lib_files]
        
        # Add output dimension: [samples, lib_files, 1]
        test_output = test_output.unsqueeze(-1)
        
        print(f"    ✅ Test data stacked: {test_input.shape}")
        print(f"    📊 Task structure: {test_input.shape[0]} tasks × {test_input.shape[1]} lib files × {test_input.shape[2]} features")
    
    if all_train_lib_data:
        print(f"  🔗 Stacking train data from {len(all_train_lib_data)} lib files...")
        # Stack along lib file dimension: [y, 9] × N files → [y, N, 9]  
        train_input = torch.stack(all_train_lib_data, dim=1)  # [samples, lib_files, features]
        train_output = torch.stack(all_train_lib_outputs, dim=1)  # [samples, lib_files]
        
        # Add output dimension: [samples, lib_files, 1]
        train_output = train_output.unsqueeze(-1)
        
        print(f"    ✅ Train data stacked: {train_input.shape}")
        print(f"    📊 Task structure: {train_input.shape[0]} tasks × {train_input.shape[1]} lib files × {train_input.shape[2]} features")
    
    # Save individual folder data
    if test_input is not None and test_output is not None:
        torch.save(test_input, save_test_input)
        torch.save(test_output, save_test_output)
        print(f"  💾 Saved test data: {save_test_input}")
    
    if train_input is not None and train_output is not None:
        torch.save(train_input, save_train_input)
        torch.save(train_output, save_train_output)
        print(f"  💾 Saved train data: {save_train_input}")
    
    return test_input, test_output, train_input, train_output

def create_separate_test_sets(folder_paths, abc_param_lists, target_cell_types, output_dir):
    """Create separate test sets for each target cell type"""
    
    for target_cell_type in target_cell_types:
        print(f"\n🎯 Creating test set for cell type: {target_cell_type}")
        
        all_test_inputs = []
        all_test_outputs = []
        all_train_inputs = []
        all_train_outputs = []
        folder_info = []
        
        for folder_path in folder_paths:
            folder_name = Path(folder_path).name
            print(f"📁 Processing folder: {folder_name}")
            
            # Parse folder parameters
            try:
                a, b, c, d = parse_ex2_folder(folder_path)
                print(f"  Folder parameters: a={a}, b={b}, c={c}, d={d}")
            except ValueError as e:
                print(f"  ❌ Error parsing folder name: {e}")
                continue
            
            # Find lib files - resolve symbolic links
            folder_path_obj = Path(folder_path).resolve() if Path(folder_path).is_symlink() else Path(folder_path)
            lib_files = list(folder_path_obj.glob("*.lib"))
            print(f"  Found {len(lib_files)} .lib files")
            
            if len(lib_files) == 0:
                print(f"  ⚠️ No .lib files found, skipping...")
                continue
            
            try:
                # Process folder for this specific cell type
                test_input, test_output, train_input, train_output = build_split_dataset_by_cell_type(
                    filenames=[str(f) for f in sorted(lib_files)],
                    abc_param_lists=abc_param_lists,
                    target_cell_types=[target_cell_type],  # Single cell type
                    folder_params=(a, b, c, d),
                    save_test_input=f"temp_{folder_name}_{target_cell_type}_test_input.pth",
                    save_test_output=f"temp_{folder_name}_{target_cell_type}_test_output.pth",
                    save_train_input=f"temp_{folder_name}_{target_cell_type}_train_input.pth",
                    save_train_output=f"temp_{folder_name}_{target_cell_type}_train_output.pth"
                )
                
                # Add to collections
                if test_input is not None and test_output is not None:
                    all_test_inputs.append(test_input)
                    all_test_outputs.append(test_output)
                    folder_info.append(folder_name)
                
                if train_input is not None and train_output is not None:
                    all_train_inputs.append(train_input)
                    all_train_outputs.append(train_output)
                
                print(f"  ✅ {folder_name}: Processed successfully")
                
                # Clean up temp files
                for temp_file in [f"temp_{folder_name}_{target_cell_type}_test_input.pth", 
                                 f"temp_{folder_name}_{target_cell_type}_test_output.pth",
                                 f"temp_{folder_name}_{target_cell_type}_train_input.pth", 
                                 f"temp_{folder_name}_{target_cell_type}_train_output.pth"]:
                    Path(temp_file).unlink(missing_ok=True)
                
            except Exception as e:
                print(f"  ❌ Error processing folder {folder_name}: {e}")
                continue
        
        # Merge and save test data for this cell type
        if all_test_inputs and all_test_outputs:
            print(f"📊 Merging {len(all_test_inputs)} test tensors for {target_cell_type}:")
            
            try:
                merged_test_input = torch.cat(all_test_inputs, dim=0)
                merged_test_output = torch.cat(all_test_outputs, dim=0)
                
                # Add output dimension: [samples, files, 1]
                if merged_test_output.dim() == 2:
                    merged_test_output = merged_test_output.unsqueeze(-1)
                
                # Save test set for this cell type
                test_input_path = f"{output_dir}/transition_{target_cell_type}_test_input.pth"
                test_output_path = f"{output_dir}/transition_{target_cell_type}_test_output.pth"
                
                torch.save(merged_test_input, test_input_path)
                torch.save(merged_test_output, test_output_path)
                
                print(f"✅ Test set for {target_cell_type} saved:")
                print(f"   Input: {test_input_path} - Shape: {merged_test_input.shape}")
                print(f"   Output: {test_output_path} - Shape: {merged_test_output.shape}")
                
            except Exception as e:
                print(f"❌ Error merging test data for {target_cell_type}: {e}")
        else:
            print(f"⚠️ No test data found for cell type '{target_cell_type}'")

def create_shared_train_set(folder_paths, abc_param_lists, exclude_cell_types, output_dir):
    """
    Create shared train set excluding all test cell types
    각 폴더마다 61개 lib file을 개별 처리하고 스택해서 x,61,9 구조 생성
    """
    print(f"\n🏋️ Creating shared train set (excluding: {exclude_cell_types})")
    
    all_train_inputs = []
    all_train_outputs = []
    
    for folder_path in folder_paths:
        folder_name = Path(folder_path).name
        print(f"📁 Processing folder: {folder_name}")
        
        # Parse folder parameters
        try:
            a, b, c, d = parse_ex2_folder(folder_path)
            print(f"  Folder parameters: a={a}, b={b}, c={c}, d={d}")
        except ValueError as e:
            print(f"  ❌ Error parsing folder name: {e}")
            continue
        
        # Get ABC parameters
        if (a < len(abc_param_lists[0]) and b*2+1 < len(abc_param_lists[1]) and
            c*2+1 < len(abc_param_lists[2])):
            abc_params = {
                'a': abc_param_lists[0][a],
                'a_n': abc_param_lists[0][a],
                'a_p': abc_param_lists[0][a],
                'b_n': abc_param_lists[1][b*2],     # nmos: even index
                'b_p': abc_param_lists[1][b*2 + 1], # pmos: odd index
                'c_n': abc_param_lists[2][c*2],     # nmos: even index
                'c_p': abc_param_lists[2][c*2 + 1]  # pmos: odd index
            }
        else:
            print(f"  ❌ Invalid parameter indices: a={a}, b={b}, c={c}")
            continue
        
        # Find lib files - resolve symbolic links
        folder_path_obj = Path(folder_path).resolve() if Path(folder_path).is_symlink() else Path(folder_path)
        lib_files = list(folder_path_obj.glob("*.lib"))
        print(f"  Found {len(lib_files)} .lib files")
        
        if len(lib_files) == 0:
            continue
        
        try:
            # Process each lib file separately to maintain proper structure
            all_train_lib_data = []
            all_train_lib_outputs = []
            
            print(f"    📄 Processing {len(lib_files)} lib files individually...")
            
            for i, lib_file in enumerate(lib_files):
                lib_name = Path(lib_file).name
                
                with open(lib_file, "r") as f:
                    lines = f.readlines()
                
                pin_data = parse_liberty_pin_blocks(lines)
                if not pin_data:
                    continue
                
                # Filter out all excluded cell types (keep only cells NOT in exclude list)
                train_pins = []
                for pin in pin_data:
                    cell_name = pin.get('cell', '')
                    is_excluded = False
                    
                    for exclude_type in exclude_cell_types:
                        # Use same logic as filter_pin_data_by_cell_type
                        has_size_spec = False
                        if ('x' in exclude_type.lower() or 'p' in exclude_type.lower()) and not exclude_type.upper().startswith(('XOR', 'XNOR')):
                            has_size_spec = True
                        
                        if has_size_spec:
                            cell_prefix = cell_name.split('_')[0] if '_' in cell_name else cell_name
                            if exclude_type.upper() == cell_prefix.upper():
                                is_excluded = True
                                break
                        else:
                            logic_type = re.match(r'([A-Z]+\d*)', cell_name)
                            if logic_type:
                                extracted_type = logic_type.group(1).upper()
                                if (exclude_type.upper() == extracted_type or 
                                    exclude_type.upper() in extracted_type):
                                    is_excluded = True
                                    break
                    
                    if not is_excluded:
                        train_pins.append(pin)
                
                # Process train data for this lib file
                if train_pins:
                    train_flattened, train_cap = flatten_pin_data(train_pins)
                    train_datasets = transform_all_samples(train_flattened, train_cap, lib_name, abc_params)
                    
                    if train_datasets:
                        train_inputs = [sample['input'] for sample in train_datasets]
                        train_outputs = [sample['output'] for sample in train_datasets]
                        lib_train_input = torch.tensor(train_inputs, dtype=torch.float32)  # [y, 9]
                        lib_train_output = torch.tensor(train_outputs, dtype=torch.float32)  # [y]
                        
                        all_train_lib_data.append(lib_train_input)
                        all_train_lib_outputs.append(lib_train_output)
            
            # Stack lib file data to create [samples, lib_files, features] structure for this folder
            if all_train_lib_data:
                # Stack along lib file dimension: [y, 9] × N files → [y, N, 9]
                folder_train_input = torch.stack(all_train_lib_data, dim=1)  # [samples, lib_files, features]
                folder_train_output = torch.stack(all_train_lib_outputs, dim=1)  # [samples, lib_files]
                folder_train_output = folder_train_output.unsqueeze(-1)  # [samples, lib_files, 1]
                
                all_train_inputs.append(folder_train_input)
                all_train_outputs.append(folder_train_output)
                
                print(f"  ✅ {folder_name}: {folder_train_input.shape[0]} tasks × {folder_train_input.shape[1]} lib files")
            else:
                print(f"  ⚠️ {folder_name}: No train data found")
            
        except Exception as e:
            print(f"  ❌ Error processing folder {folder_name}: {e}")
            continue
    
    # Merge and save shared train data
    if all_train_inputs and all_train_outputs:
        print(f"📊 Merging {len(all_train_inputs)} train tensors:")
        for i, tensor in enumerate(all_train_inputs):
            print(f"    Folder {i}: {tensor.shape}")
        
        # Concatenate along task dimension: [tasks1, 61, 9] + [tasks2, 61, 9] → [total_tasks, 61, 9] 
        merged_train_input = torch.cat(all_train_inputs, dim=0)
        merged_train_output = torch.cat(all_train_outputs, dim=0)
        
        print(f"    ✅ Merged shape: input {merged_train_input.shape}, output {merged_train_output.shape}")
        
        # Save shared train data
        train_input_path = f"{output_dir}/transition_topology_agnostic_train_input.pth"
        train_output_path = f"{output_dir}/transition_topology_agnostic_train_output.pth"
        
        torch.save(merged_train_input, train_input_path)
        torch.save(merged_train_output, train_output_path)
        
        print(f"✅ Shared train set saved:")
        print(f"   Input: {train_input_path} - Shape: {merged_train_input.shape}")
        print(f"   Output: {train_output_path} - Shape: {merged_train_output.shape}")
        print(f"   📊 Structure: {merged_train_input.shape[0]} tasks × {merged_train_input.shape[1]} lib files × {merged_train_input.shape[2]} features")
    else:
        print(f"⚠️ No train data to merge")

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build dataset with separate test sets for each cell type")
    parser.add_argument('--data-dirs', nargs='+', required=True, help='Input data directories')
    parser.add_argument('--output-dir', required=True, help='Output directory for datasets')
    parser.add_argument('--test-cell-types', nargs='+', required=True, help='Cell types for test sets')
    parser.add_argument('--param-a', type=str, required=True, help='Comma-separated A parameters')
    parser.add_argument('--param-b', type=str, required=True, help='Comma-separated B parameters')
    parser.add_argument('--param-c', type=str, required=True, help='Comma-separated C parameters')
    parser.add_argument('--train-only', action='store_true', help='Only create shared train dataset')
    
    args = parser.parse_args()
    
    # Parse parameter lists
    param_a_list = [float(x) for x in args.param_a.split(',')]
    param_b_list = [float(x) for x in args.param_b.split(',')]
    param_c_list = [float(x) for x in args.param_c.split(',')]
    
    # ABC parameter lists
    abc_param_lists = [param_a_list, param_b_list, param_c_list]
    
    print(f"🚀 Building separate cell type datasets")
    print(f"📁 Data directories: {args.data_dirs}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"🎯 Target cell types: {args.test_cell_types}")
    print(f"📊 Parameter mappings:")
    print(f"  A parameters: {param_a_list}")
    print(f"  B parameters: {param_b_list}")
    print(f"  C parameters: {param_c_list}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect all folder paths from multiple data directories
    folder_paths = []
    for data_dir in args.data_dirs:
        if not Path(data_dir).exists():
            print(f"⚠️ Directory does not exist: {data_dir}, skipping...")
            continue
        
        # Find folders that match pattern (ex2_*, VDD_*, simple_*, or invbuf_*)
        for p in Path(data_dir).iterdir():
            if p.is_dir() and (p.name.startswith('ex2_') or p.name.startswith('VDD_') or 
                             p.name.startswith('simple_') or p.name.startswith('invbuf_')):
                folder_paths.append(str(p))
    
    folder_paths = sorted(folder_paths)
    print(f"📊 Found {len(folder_paths)} folders total")
    
    if len(folder_paths) == 0:
        print("❌ No valid folders found in any data directory!")
        return
    
    if args.train_only:
        # Only create shared train set
        print(f"🏋️ Creating shared train set only (--train-only mode)")
        create_shared_train_set(folder_paths, abc_param_lists, args.test_cell_types, args.output_dir)
    else:
        # Create separate test sets for each cell type
        create_separate_test_sets(folder_paths, abc_param_lists, args.test_cell_types, args.output_dir)
        
        # Create shared train set (excluding all test cell types)
        #create_shared_train_set(folder_paths, abc_param_lists, args.test_cell_types, args.output_dir)
    
    print(f"\n🎉 Dataset creation completed!")
    print(f"💾 Files saved in: {args.output_dir}")

if __name__ == "__main__":
    main()