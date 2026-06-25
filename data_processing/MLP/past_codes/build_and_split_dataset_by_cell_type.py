# build_and_split_dataset_by_cell_type.py
import torch
from pathlib import Path
from datasets import libdata
from libdata_extract_MAML import parse_liberty_pin_blocks, flatten_pin_data
from transform_sample_MAML_invbuf import transform_all_samples
import sys
import re

def filter_pin_data_by_cell_type(pin_data, target_cell_types):
    """
    Filter pin data to separate specific cell types for test set
    
    Args:
        pin_data: List of pin dictionaries from parse_liberty_pin_blocks
        target_cell_types: List of cell type patterns to put in test set 
                          Can be either:
                          - Logic type only (e.g., ['INV', 'NAND2', 'NOR2'])
                          - Full name with size (e.g., ['NAND2xp67', 'INVx2'])
                          - Mix of both
    
    Returns:
        test_pins: Pin data for test set (target cell types)
        train_pins: Pin data for train set (remaining cell types)
    """
    test_pins = []
    train_pins = []
    
    for pin in pin_data:
        cell_name = pin.get('cell', '')
        
        # Check if this cell matches any target cell type
        is_target_cell = False
        for target_type in target_cell_types:
            # Check if target includes size specification (contains 'x' or 'p' as size indicator, not as part of logic name)
            # XOR2, XNOR2 should be treated as logic types, not size specifications
            has_size_spec = False
            if ('x' in target_type.lower() or 'p' in target_type.lower()) and not target_type.upper().startswith(('XOR', 'XNOR')):
                has_size_spec = True
            
            if has_size_spec:
                # Full name match (e.g., 'NAND2xp67' matches 'NAND2xp67_ASAP7_75t_L')
                cell_prefix = cell_name.split('_')[0] if '_' in cell_name else cell_name
                
                if target_type.upper() == cell_prefix.upper():
                    is_target_cell = True
                    break
            else:
                # Logic type only match (e.g., 'NAND2' matches all NAND2 sizes)
                logic_type = re.match(r'([A-Z]+\d*)', cell_name)
                if logic_type and (target_type.upper() == logic_type.group(1).upper() or 
                                  target_type.upper() in logic_type.group(1).upper()):
                    is_target_cell = True
                    break
        
        if is_target_cell:
            test_pins.append(pin)
        else:
            train_pins.append(pin)
    
    return test_pins, train_pins

def dataextract_filtered(text, lib_prefix="", abc_params_dict=None, target_cell_types=None, folder_params=None):
    """Extract data from lib file with cell type filtering"""
    import torch
    
    with open(text, "r") as f:
        lines = f.readlines()
    
    pin_data = parse_liberty_pin_blocks(lines)
    
    # Extract folder params and create proper abc_params for transform_all_samples
    if abc_params_dict and folder_params:
        a, b, c, d = folder_params
        a = int(a)
        b = int(b) 
        c = int(c)
        
        # Map folder indices to actual parameter values
        # B and C parameters have nmos/pmos pairs
        abc_params = {
            'a': abc_params_dict['a_params'][a] if a < len(abc_params_dict['a_params']) else abc_params_dict['a_params'][0],
            'b_n': abc_params_dict['b_params'][b * 2 + 1] if (b * 2 + 1) < len(abc_params_dict['b_params']) else abc_params_dict['b_params'][1],
            'b_p': abc_params_dict['b_params'][b * 2] if (b * 2) < len(abc_params_dict['b_params']) else abc_params_dict['b_params'][0],
            'c_n': abc_params_dict['c_params'][c * 2 + 1] if (c * 2 + 1) < len(abc_params_dict['c_params']) else abc_params_dict['c_params'][1],
            'c_p': abc_params_dict['c_params'][c * 2] if (c * 2) < len(abc_params_dict['c_params']) else abc_params_dict['c_params'][0],
        }
    else:
        abc_params = None
    
    if target_cell_types:
        # Split pin data by cell type
        test_pins, train_pins = filter_pin_data_by_cell_type(pin_data, target_cell_types)
        
        # Process test data
        test_datasets = None
        if test_pins:
            test_flattened, test_cap = flatten_pin_data(test_pins)
            test_datasets = transform_all_samples(test_flattened, test_cap, lib_prefix + "_test", abc_params)
        
        # Process train data  
        train_datasets = None
        if train_pins:
            train_flattened, train_cap = flatten_pin_data(train_pins)
            train_datasets = transform_all_samples(train_flattened, train_cap, lib_prefix + "_train", abc_params)
        
        return test_datasets, train_datasets, len(test_pins), len(train_pins)
    
    else:
        # No filtering, process all data as train
        flattened, cap = flatten_pin_data(pin_data)
        datasets = transform_all_samples(flattened, cap, lib_prefix, abc_params)
        return None, datasets, 0, len(pin_data)

def aggregate_datasets(datasets):
    """Aggregate list of datasets into single input/output tensors"""
    if not datasets or len(datasets) == 0:
        return None, None
    
    all_inputs = []
    all_outputs = []
    
    for item in datasets:
        if isinstance(item, dict) and 'input' in item and 'output' in item:
            input_data = item['input']
            output_data = item['output']
            
            # Convert to tensor if needed
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            if not isinstance(output_data, torch.Tensor):
                output_data = torch.tensor(output_data, dtype=torch.float32)
            
            # Ensure proper dimensions
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)
            if output_data.dim() == 0:
                output_data = output_data.unsqueeze(0)
                
            all_inputs.append(input_data)
            all_outputs.append(output_data)
    
    if all_inputs and all_outputs:
        # Concatenate all cell data along first dimension
        combined_input = torch.cat(all_inputs, dim=0)
        combined_output = torch.cat(all_outputs, dim=0)
        return combined_input, combined_output
    
    return None, None

def build_split_dataset_by_cell_type(filenames, abc_param_lists, target_cell_types, folder_params=None, 
                                   save_test_input="test_input.pth", save_test_output="test_output.pth",
                                   save_train_input="train_input.pth", save_train_output="train_output.pth"):
    """
    Build dataset with train/test split based on cell types
    
    Args:
        filenames: List of .lib files to process
        abc_param_lists: Parameter lists for ABC values
        target_cell_types: List of cell types to put in test set (e.g., ['INV', 'NAND2'])
        folder_params: Folder parameters (a, b, c, d)
        save_*: Output file paths
    """
    
    test_inputs = []
    test_outputs = []
    train_inputs = []
    train_outputs = []
    
    processed_files = 0
    skipped_files = 0
    total_test_pins = 0
    total_train_pins = 0

    print(f"🎯 Target cell types for test set: {target_cell_types}")
    
    for filename in filenames:
        print(f"  Processing: {Path(filename).name}")
        
        try:
            # Parse filename to extract voltage
            voltage = parse_ex2_filename(filename)
            
            if folder_params is not None:
                # Use folder parameters for a,b,c,d
                a, b, c, d = folder_params
                
                # Create abc_params dict
                abc_params = {
                    'a': abc_param_lists[0][a] if a < len(abc_param_lists[0]) else abc_param_lists[0][0],
                    'b_p': abc_param_lists[1][b * 2] if (b * 2) < len(abc_param_lists[1]) else abc_param_lists[1][0],
                    'b_n': abc_param_lists[1][b * 2 + 1] if (b * 2 + 1) < len(abc_param_lists[1]) else abc_param_lists[1][1],
                    'c_p': abc_param_lists[2][c * 2] if (c * 2) < len(abc_param_lists[2]) else abc_param_lists[2][0],
                    'c_n': abc_param_lists[2][c * 2 + 1] if (c * 2 + 1) < len(abc_param_lists[2]) else abc_param_lists[2][1],
                }
            else:
                # Fallback: use default parameters
                abc_params = {
                    'a': abc_param_lists[0][0],
                    'b_p': abc_param_lists[1][0],
                    'b_n': abc_param_lists[1][1],
                    'c_p': abc_param_lists[2][0],
                    'c_n': abc_param_lists[2][1],
                }
            
            lib_prefix = Path(filename).stem
            
            # Extract with cell type filtering
            test_datasets, train_datasets, test_pin_count, train_pin_count = dataextract_filtered(
                filename, lib_prefix, abc_params, target_cell_types
            )
            
            total_test_pins += test_pin_count
            total_train_pins += train_pin_count
            
            # Process test data
            if test_datasets:
                test_input, test_output = aggregate_datasets(test_datasets)
                if test_input is not None and test_output is not None:
                    test_inputs.append(test_input)
                    test_outputs.append(test_output)
            
            # Process train data
            if train_datasets:
                train_input, train_output = aggregate_datasets(train_datasets)
                if train_input is not None and train_output is not None:
                    train_inputs.append(train_input)
                    train_outputs.append(train_output)
            
            processed_files += 1
            print(f"    ✅ Test pins: {test_pin_count}, Train pins: {train_pin_count}")
                
        except Exception as e:
            print(f"    ⚠️ Error processing {Path(filename).name}: {e}")
            skipped_files += 1
            continue
    
    print(f"  📊 Processed: {processed_files} files, Skipped: {skipped_files} files")
    print(f"  🎯 Total test pins: {total_test_pins}, Total train pins: {total_train_pins}")
    
    # Save test data
    if test_inputs and test_outputs:
        test_input_combined = torch.stack(test_inputs, dim=1)    # [samples, files, features]
        test_output_combined = torch.stack(test_outputs, dim=1)  # [samples, files]
        test_output_combined = test_output_combined.unsqueeze(-1)  # [samples, files, 1]
        
        torch.save(test_input_combined, save_test_input)
        torch.save(test_output_combined, save_test_output)
        print(f"✅ Test data saved: {save_test_input}, {save_test_output}")
        print(f"   Test shape: Input {test_input_combined.shape}, Output {test_output_combined.shape}")
    else:
        print("⚠️ No test data to save")
    
    # Save train data
    if train_inputs and train_outputs:
        train_input_combined = torch.stack(train_inputs, dim=1)    # [samples, files, features]
        train_output_combined = torch.stack(train_outputs, dim=1)  # [samples, files]
        train_output_combined = train_output_combined.unsqueeze(-1)  # [samples, files, 1]
        
        torch.save(train_input_combined, save_train_input)
        torch.save(train_output_combined, save_train_output)
        print(f"✅ Train data saved: {save_train_input}, {save_train_output}")
        print(f"   Train shape: Input {train_input_combined.shape}, Output {train_output_combined.shape}")
    else:
        print("⚠️ No train data to save")
    
    return (test_input_combined if test_inputs else None, test_output_combined if test_outputs else None,
            train_input_combined if train_inputs else None, train_output_combined if train_outputs else None)

def parse_ex2_folder(folder_path):
    """Parse ex2_a_b_c_d or invbuf_a_b_c_d folder name to extract a,b,c,d values
    Also handles folders with prefixes or suffixes due to conflict resolution"""
    folder_name = Path(folder_path).name
    
    # Handle folders with parent directory prefix (e.g., test_processed_invbuf_1_2_3_4)
    # or with hash suffix (e.g., invbuf_1_2_3_4_a1b2c3d4)
    
    # Try to find ex2_ or invbuf_ pattern anywhere in the folder name
    # Pattern for ex2_
    match = re.search(r'ex2_(\d+)_(\d+)_(\d+)_(\d+)', folder_name)
    if match:
        a, b, c, d = match.groups()
        return int(a), int(b), int(c), int(d)
    
    # Pattern for invbuf_
    match = re.search(r'invbuf_(\d+)_(\d+)_(\d+)_(\d+)', folder_name)
    if match:
        a, b, c, d = match.groups()
        return int(a), int(b), int(c), int(d)
    
    # If neither pattern matches
    raise ValueError(f"Invalid folder format (expected ex2_ or invbuf_ pattern): {folder_name}")

def parse_ex2_filename(filename):
    """Parse ex2_a_b_c_d_voltage.lib or invbuf_a_b_c_d_voltage.lib filename to extract voltage only"""
    basename = Path(filename).stem
    
    # Try ex2_ pattern first
    match = re.match(r'ex2_\d+_\d+_\d+_\d+_(\d+)', basename)
    if match:
        voltage = int(match.group(1))
        return voltage
    
    # Try invbuf_ pattern
    match = re.match(r'invbuf_\d+_\d+_\d+_\d+_(\d+)', basename)
    if match:
        voltage = int(match.group(1))
        return voltage
    
    # If neither pattern matches
    raise ValueError(f"Invalid filename format (expected ex2_ or invbuf_): {filename}")

def build_merged_split_dataset(folder_paths, abc_param_lists, target_cell_types, 
                             save_test_input="merged_test_input.pth", save_test_output="merged_test_output.pth",
                             save_train_input="merged_train_input.pth", save_train_output="merged_train_output.pth"):
    """Build merged dataset with train/test split from multiple folders"""
    
    all_test_inputs = []
    all_test_outputs = []
    all_train_inputs = []
    all_train_outputs = []
    folder_info = []
    
    print(f"🔄 Building train/test split dataset based on cell types")
    print(f"🎯 Test cell types: {target_cell_types}")
    print(f"📊 Parameter mappings:")
    print(f"  A parameters: {abc_param_lists[0]}")
    print(f"  B parameters: {abc_param_lists[1]}")  
    print(f"  C parameters: {abc_param_lists[2]}")
    
    for folder_path in folder_paths:
        folder_name = Path(folder_path).name
        print(f"📁 Processing folder: {folder_name}")
        
        # Parse folder name to get a,b,c,d values
        try:
            a, b, c, d = parse_ex2_folder(folder_path)
            print(f"  Folder parameters: a={a}, b={b}, c={c}, d={d}")
        except ValueError as e:
            print(f"  ❌ Error parsing folder name: {e}")
            continue
        
        # Find all .lib files in folder
        lib_files = list(Path(folder_path).glob("*.lib"))
        print(f"  Found {len(lib_files)} .lib files")
        if lib_files:
            print(f"    First few files: {[f.name for f in lib_files[:3]]}")
        
        if len(lib_files) == 0:
            print(f"  ⚠️ No .lib files found, skipping...")
            continue
            
        try:
            # Process this folder with train/test split
            test_input, test_output, train_input, train_output = build_split_dataset_by_cell_type(
                filenames=[str(f) for f in sorted(lib_files)],
                abc_param_lists=abc_param_lists,
                target_cell_types=target_cell_types,
                folder_params=(a, b, c, d),
                save_test_input=f"temp_{folder_name}_test_input.pth",
                save_test_output=f"temp_{folder_name}_test_output.pth",
                save_train_input=f"temp_{folder_name}_train_input.pth",
                save_train_output=f"temp_{folder_name}_train_output.pth"
            )
            
            # Add to collections
            if test_input is not None and test_output is not None:
                all_test_inputs.append(test_input)
                all_test_outputs.append(test_output)
                folder_info.append(folder_name)
            
            if train_input is not None and train_output is not None:
                all_train_inputs.append(train_input)
                all_train_outputs.append(train_output)
            
            folder_info.append(folder_name)
            print(f"  ✅ {folder_name}: Processed successfully")
            
            # Clean up temp files
            for temp_file in [f"temp_{folder_name}_test_input.pth", f"temp_{folder_name}_test_output.pth",
                             f"temp_{folder_name}_train_input.pth", f"temp_{folder_name}_train_output.pth"]:
                Path(temp_file).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"  ❌ Error processing folder {folder_name}: {e}")
            continue
    
    # Merge all test data
    if all_test_inputs and all_test_outputs:
        print(f"📊 Merging {len(all_test_inputs)} test tensors:")
        for i, tensor in enumerate(all_test_inputs):
            folder_name = folder_info[i] if i < len(folder_info) else "unknown"
            print(f"   Tensor {i} ({folder_name}): {tensor.shape}")
        
        # Check if all tensors have the same shape (except dim 0)
        if len(set(tensor.shape[1:] for tensor in all_test_inputs)) > 1:
            print("⚠️ Tensors have different shapes in files dimension - cannot concatenate directly")
            print("   This happens when folders have different numbers of .lib files")
            print("   Folder tensor shapes:")
            for i, tensor in enumerate(all_test_inputs):
                folder_name = folder_info[i] if i < len(folder_info) else "unknown"
                print(f"     {folder_name}: {tensor.shape}")
            print("   Skipping test data merge due to shape mismatch")
            merged_test_input = None
            merged_test_output = None
        else:
            merged_test_input = torch.cat(all_test_inputs, dim=0)
            merged_test_output = torch.cat(all_test_outputs, dim=0)
        
        torch.save(merged_test_input, save_test_input)
        torch.save(merged_test_output, save_test_output)
        print(f"✅ Merged test dataset saved:")
        print(f"   Input: {save_test_input} - Shape: {merged_test_input.shape}")
        print(f"   Output: {save_test_output} - Shape: {merged_test_output.shape}")
    else:
        print("⚠️ No test data to merge")
    
    # Merge all train data
    if all_train_inputs and all_train_outputs:
        merged_train_input = torch.cat(all_train_inputs, dim=0)
        merged_train_output = torch.cat(all_train_outputs, dim=0)
        
        torch.save(merged_train_input, save_train_input)
        torch.save(merged_train_output, save_train_output)
        print(f"✅ Merged train dataset saved:")
        print(f"   Input: {save_train_input} - Shape: {merged_train_input.shape}")
        print(f"   Output: {save_train_output} - Shape: {merged_train_output.shape}")
    else:
        print("⚠️ No train data to merge")
    
    # Save folder mapping for reference
    if folder_info:
        mapping_file = save_train_input.replace('_input.pth', '_folder_mapping.txt')
        with open(mapping_file, 'w') as f:
            f.write(f"Target test cell types: {target_cell_types}\n")
            f.write(f"Processed folders:\n")
            for i, folder in enumerate(folder_info):
                f.write(f"{i}: {folder}\n")
        print(f"📝 Folder mapping saved: {mapping_file}")
    
    return (merged_test_input if all_test_inputs else None,
            merged_test_output if all_test_outputs else None,
            merged_train_input if all_train_inputs else None,
            merged_train_output if all_train_outputs else None)

def main():
    if len(sys.argv) < 7:
        print("Usage: python build_and_split_dataset_by_cell_type.py [--merge] <output_dir> <data_dir> <param_a> <param_b> <param_c> <test_cell_types>")
        print("  test_cell_types: Comma-separated list of cell types for test set (e.g., 'INV,NAND2,NOR2')")
        sys.exit(1)
    
    if sys.argv[1] == "--merge":
        mode = "merge"
        output_dir = sys.argv[2]
        data_dir = sys.argv[3]
        param_a = sys.argv[4]
        param_b = sys.argv[5]
        param_c = sys.argv[6]
        test_cell_types = sys.argv[7].split(',') if len(sys.argv) > 7 else ['INV']
    else:
        mode = "single"
        output_dir = sys.argv[1]
        data_dir = sys.argv[2]
        param_a = sys.argv[3]
        param_b = sys.argv[4]
        param_c = sys.argv[5]
        test_cell_types = sys.argv[6].split(',') if len(sys.argv) > 6 else ['INV']
    
    # Parse parameter lists
    abc_param_lists = [
        [float(x) for x in param_a.split(',')],
        [float(x) for x in param_b.split(',')],
        [float(x) for x in param_c.split(',')]
    ]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if mode == "merge":
        # Find all folders
        data_path = Path(data_dir)
        test_folders = [f for f in data_path.iterdir() if f.is_dir()]
        
        print(f"📁 Found {len(test_folders)} folders:")
        for folder in sorted(test_folders):
            print(f"  {folder.name}")
        
        if not test_folders:
            print("❌ No folders found!")
            sys.exit(1)
        
        # Build merged dataset with train/test split
        build_merged_split_dataset(
            folder_paths=[str(f) for f in sorted(test_folders)],
            abc_param_lists=abc_param_lists,
            target_cell_types=test_cell_types,
            save_test_input=f"{output_dir}/transition_topology_agnostic_test_input.pth",
            save_test_output=f"{output_dir}/transition_topology_agnostic_test_output.pth", 
            save_train_input=f"{output_dir}/transition_topology_agnostic_train_input.pth",
            save_train_output=f"{output_dir}/transition_topology_agnostic_train_output.pth"
        )
        
        print(f"✅ Train/Test split dataset creation completed!")
        print(f"   Test cell types: {test_cell_types}")
        
    else:
        # Single folder processing
        lib_files = list(Path(data_dir).glob("*.lib"))
        print(f"Found {len(lib_files)} .lib files")
        
        if len(lib_files) == 0:
            print("No .lib files found!")
            sys.exit(1)
        
        # Build dataset with train/test split
        build_split_dataset_by_cell_type(
            filenames=[str(f) for f in sorted(lib_files)],
            abc_param_lists=abc_param_lists,
            target_cell_types=test_cell_types,
            save_test_input=f"{output_dir}/transition_test_input.pth",
            save_test_output=f"{output_dir}/transition_test_output.pth",
            save_train_input=f"{output_dir}/transition_train_input.pth", 
            save_train_output=f"{output_dir}/transition_train_output.pth"
        )
        
        print(f"✅ Train/Test split dataset creation completed!")
        print(f"   Test cell types: {test_cell_types}")

if __name__ == "__main__":
    main()