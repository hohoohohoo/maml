# build_and_split_dataset_invbuf_test.py
import torch
from pathlib import Path
from datasets import libdata
from libdata_extract_MAML import parse_liberty_pin_blocks, flatten_pin_data
from transform_sample_MAML_invbuf import transform_all_samples
import sys
import re

def parse_invbuf_filename(filename):
    """
    Parse invbuf_a_b_c_d_e.lib filename to extract a,b,c,d,e values
    Handles formats like:
    - invbuf_0_0_0_m25_049.lib (m25 = -25)
    - invbuf_0_0_0_62p5_049.lib (62p5 = 62.5)
    Returns: (a, b, c, temperature, voltage)
    """
    # Extract basename without .lib extension
    basename = Path(filename).stem
    
    # Pattern: invbuf_a_b_c_temp_voltage
    match = re.match(r'invbuf_(\d+)_(\d+)_(\d+)_([^_]+)_(\d+)', basename)
    if match:
        a, b, c, temp_str, voltage_str = match.groups()
        
        # Parse temperature string
        if temp_str.startswith('m'):
            # Handle negative: m25 -> -25
            temperature = -float(temp_str[1:])
        elif 'p' in temp_str:
            # Handle decimal: 62p5 -> 62.5
            temperature = float(temp_str.replace('p', '.'))
        else:
            temperature = float(temp_str)
        
        voltage = int(voltage_str)
        
        return int(a), int(b), int(c), temperature, voltage
    else:
        raise ValueError(f"Invalid invbuf filename format: {filename}")

def dataextract(text, lib_prefix="", abc_params=None):
    """Extract data from lib file with additional abc parameter mapping"""
    with open(text, "r") as f:
        lines = f.readlines()
    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)
    dataset = transform_all_samples(flattened, cap, lib_prefix, abc_params)
    return dataset

def data_management(text, lib_prefix="", abc_params=None):
    """Process single lib file with abc parameter mapping"""
    from pathlib import Path
    file_path = Path(text)
    dataset = dataextract(file_path, lib_prefix, abc_params)
    dataset = libdata(dataset)
    return dataset

def build_all_data(filenames, abc_param_lists, save_input="full_input_tensor.pth", save_output="full_output_tensor.pth"):
    """
    Build dataset from list of invbuf files
    
    Args:
        filenames: List of invbuf_a_b_c_d_e.lib files
        abc_param_lists: List of 3 lists, each containing 5 values for mapping a,b,c
        save_input: Output file for input tensors
        save_output: Output file for output tensors
    """
    data_input = []
    data_output = []

    for filename in filenames:
        print(f"📥 Processing {filename}")
        
        # Parse a,b,c values from filename
        try:
            a, b, c, d, e = parse_invbuf_filename(filename)
            
            # Create abc_params dict for this file  
            # For each b,c index, we have both nmos and pmos versions
            abc_params = {
                'a': abc_param_lists[0][a] if a < len(abc_param_lists[0]) else 0,
                'b_p': abc_param_lists[1][b * 2] if (b * 2) < len(abc_param_lists[1]) else 0,      # b nmos 
                'b_n': abc_param_lists[1][b * 2 + 1] if (b * 2 + 1) < len(abc_param_lists[1]) else 0,  # b pmos
                'c_p': abc_param_lists[2][c * 2] if (c * 2) < len(abc_param_lists[2]) else 0,      # c nmos
                'c_n': abc_param_lists[2][c * 2 + 1] if (c * 2 + 1) < len(abc_param_lists[2]) else 0,  # c pmos
            }
            
            print(f"  a={a} -> {abc_params['a']}")
            print(f"  b={b} -> b_n:{abc_params['b_n']}, b_p:{abc_params['b_p']}")
            print(f"  c={c} -> c_n:{abc_params['c_n']}, c_p:{abc_params['c_p']}")
            
        except ValueError as e:
            print(f"  ⚠️ Error parsing filename: {e}")
            continue
            
        dataset = data_management(filename, "invbuf_", abc_params)
        data_input.append(dataset.X)
        data_output.append(dataset.Y)
        print(f"  Dataset shape: {dataset.X.size()}")

    if not data_input:
        raise ValueError("No valid datasets were processed")
        
    data_input = torch.stack(data_input, dim=1)    # [samples, files, 9]
    data_output = torch.stack(data_output, dim=1)  # [samples, files]
    
    # Add unsqueeze to match expected output format [samples, files, 1]
    data_output = data_output.unsqueeze(-1)
    torch.save(data_input, save_input)
    torch.save(data_output, save_output)
    print(f"✅ Saved: {save_input}, {save_output}")
    print(f"   Input shape: {data_input.shape}, Output shape: {data_output.shape}")
    return data_input, data_output

def build_merged_data(folder_paths, abc_param_lists, save_input="merged_invbuf_test_input.pth", save_output="merged_invbuf_test_output.pth"):
    """
    Build merged test dataset from all invbuf folders
    
    Args:
        folder_paths: List of folder paths containing invbuf_*.lib files
        abc_param_lists: List of 3 lists, each containing parameter mappings
        save_input: Output file for merged input tensors
        save_output: Output file for merged output tensors
    """
    all_inputs = []
    all_outputs = []
    folder_info = []

    for folder_path in folder_paths:
        folder_name = Path(folder_path).name
        print(f"📁 Processing test folder: {folder_name}")
        
        # Find all .lib files in this folder
        lib_files = list(Path(folder_path).glob("*.lib"))
        if not lib_files:
            print(f"  ⚠️ No .lib files found in {folder_name}, skipping...")
            continue
            
        print(f"  Found {len(lib_files)} .lib files")
        
        # Process all files in this folder
        folder_input = []
        folder_output = []
        
        for lib_file in sorted(lib_files):
            try:
                a, b, c, d, e = parse_invbuf_filename(lib_file.name)
                
                abc_params = {
                    'a': abc_param_lists[0][a] if a < len(abc_param_lists[0]) else 0,
                    'b_p': abc_param_lists[1][b * 2] if (b * 2) < len(abc_param_lists[1]) else 0,
                    'b_n': abc_param_lists[1][b * 2 + 1] if (b * 2 + 1) < len(abc_param_lists[1]) else 0,
                    'c_p': abc_param_lists[2][c * 2] if (c * 2) < len(abc_param_lists[2]) else 0,
                    'c_n': abc_param_lists[2][c * 2 + 1] if (c * 2 + 1) < len(abc_param_lists[2]) else 0,
                }
                
                dataset = data_management(str(lib_file), "invbuf_", abc_params)
                folder_input.append(dataset.X)
                folder_output.append(dataset.Y)
                
            except ValueError as e:
                print(f"    ⚠️ Error processing {lib_file.name}: {e}")
                continue
        
        if folder_input:
            # Stack files from this folder: [samples, files, features]
            folder_input = torch.stack(folder_input, dim=1)
            folder_output = torch.stack(folder_output, dim=1)
            
            all_inputs.append(folder_input)
            all_outputs.append(folder_output)
            folder_info.append(folder_name)
            print(f"  ✅ {folder_name}: Input {folder_input.shape}, Output {folder_output.shape}")
        else:
            print(f"  ❌ No valid files processed in {folder_name}")

    if not all_inputs:
        raise ValueError("No valid folders were processed")

    # Concatenate all folders along samples dimension
    merged_input = torch.cat(all_inputs, dim=0)    # [total_samples, files, features]
    merged_output = torch.cat(all_outputs, dim=0)  # [total_samples, files]
    
    # Add unsqueeze to match expected output format [total_samples, files, 1]
    merged_output = merged_output.unsqueeze(-1)
    
    torch.save(merged_input, save_input)
    torch.save(merged_output, save_output)
    
    print(f"✅ Merged test dataset saved:")
    print(f"   Input: {save_input} - Shape: {merged_input.shape}")
    print(f"   Output: {save_output} - Shape: {merged_output.shape}")
    print(f"   Processed {len(folder_info)} folders: {folder_info}")
    
    return merged_input, merged_output, folder_info

def split_and_save_nodewise(input_path, output_path, output_dir, prefix, start_node, end_node):
    """Split and save node-wise data"""
    Path(output_dir).mkdir(exist_ok=True)
    data_input = torch.load(input_path)     # [samples, files, 9]
    data_output_all = torch.load(output_path)  # [samples, files, 1]
    
    samples_per_file = data_input.shape[0] // data_input.shape[1]  # Should be 49
    
    for i in range(start_node, end_node):
        # Extract samples corresponding to node i from all files
        node_indices = []
        for file_idx in range(data_input.shape[1]):
            sample_idx = file_idx * samples_per_file + i
            if sample_idx < data_input.shape[0]:
                node_indices.append(sample_idx)
        
        if node_indices:
            data_input_node = data_input[node_indices]  # [files, 9]
            data_output_node = data_output_all[node_indices]  # [files, 1]
            
            input_save = f"{output_dir}/transition_{prefix}input_{i+1}nodes_align.pth"
            output_save = f"{output_dir}/transition_{prefix}output_{i+1}nodes_align.pth"
            print(f"Node {i+1}: {data_input_node.size()}, {data_output_node.size()}")
            torch.save(data_input_node, input_save)
            torch.save(data_output_node, output_save)
            print(f"💾 Node {i+1} saved → {input_save}, {output_save}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage modes:")
        print("1. Single folder: python build_and_split_dataset_invbuf_test.py <output_dir> <data_dir> <start_node> <end_node> <param_a_list> [param_b_list] [param_c_list]")
        print("2. Merge folders: python build_and_split_dataset_invbuf_test.py --merge <output_dir> <base_data_dir> <param_a_list> [param_b_list] [param_c_list]")
        print("Example merge: python build_and_split_dataset_invbuf_test.py --merge test_unified_dataset /path/to/test_processed '0.1,0.2,0.3,0.4' '1.0,1.5,2.0,2.5' '10,20,30,40'")
        sys.exit(1)
    
    if sys.argv[1] == "--merge":
        # Merge mode: process all folders and create unified test dataset
        if len(sys.argv) < 5:
            print("❌ Merge mode requires: --merge <output_dir> <base_data_dir> <param_a_list> [param_b_list] [param_c_list]")
            sys.exit(1)
            
        output_dir = sys.argv[2]       # e.g., test_unified_dataset  
        base_data_dir = sys.argv[3]    # Directory containing invbuf_* folders
        param_a_str = sys.argv[4]
        param_b_str = sys.argv[5] if len(sys.argv) > 5 else "0,0,0,0,0,0,0,0"  # Default values
        param_c_str = sys.argv[6] if len(sys.argv) > 6 else "0,0,0,0,0,0,0,0"  # Default values
        
        # Convert string to lists of floats
        abc_param_lists = [
            list(map(float, param_a_str.split(','))),
            list(map(float, param_b_str.split(','))),  
            list(map(float, param_c_str.split(',')))
        ]
        
        print(f"🔄 Test merge mode: Building unified test dataset")
        print(f"📊 Parameter mappings:")
        print(f"  A parameters: {abc_param_lists[0]}")
        print(f"  B parameters: {abc_param_lists[1]}")
        print(f"  C parameters: {abc_param_lists[2]}")
        
        # Find all invbuf_* folders
        base_path = Path(base_data_dir)
        invbuf_folders = [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith('invbuf_')]
        
        if not invbuf_folders:
            print(f"❌ No invbuf_* folders found in {base_data_dir}")
            sys.exit(1)
            
        print(f"📁 Found {len(invbuf_folders)} invbuf test folders:")
        for folder in sorted(invbuf_folders):
            print(f"  {folder.name}")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Build merged test dataset
        merged_input, merged_output, folder_info = build_merged_data(
            folder_paths=[str(f) for f in sorted(invbuf_folders)],
            abc_param_lists=abc_param_lists,
            save_input=f"{output_dir}/transition_merged_invbuf_test_input.pth",
            save_output=f"{output_dir}/transition_merged_invbuf_test_output.pth"
        )
        
        # Save folder mapping info
        info_file = f"{output_dir}/test_folder_mapping.txt"
        with open(info_file, 'w') as f:
            f.write("Test folder mapping for merged dataset:\n")
            for i, folder in enumerate(folder_info):
                f.write(f"{folder}\n")
        
        print(f"📝 Test folder mapping saved: {info_file}")
        print("✅ Unified test dataset creation completed!")
        
    else:
        # Original single folder mode
        if len(sys.argv) < 6:
            print("Usage: python build_and_split_dataset_invbuf_test.py <output_dir> <data_dir> <start_node> <end_node> <param_a_list> [param_b_list] [param_c_list]")
            print("Example: python build_and_split_dataset_invbuf_test.py nodewise_pth data_dir 0 49 '0.1,0.2,0.3,0.4,0.5' '1.0,1.5,2.0,2.5,3.0' '10,20,30,40,50'")
            sys.exit(1)
        
        output_dir = sys.argv[1]       # e.g., nodewise_pth
        data_dir = sys.argv[2]         # Directory containing invbuf_*.lib files
        start_node = int(sys.argv[3])  # 0 
        end_node = int(sys.argv[4])    # 49
        
        # Parse parameter lists from command line arguments
        param_a_str = sys.argv[5]
        param_b_str = sys.argv[6] if len(sys.argv) > 6 else "0,0,0,0,0"  # Default values
        param_c_str = sys.argv[7] if len(sys.argv) > 7 else "0,0,0,0,0"  # Default values
        
        # Convert string to lists of floats
        abc_param_lists = [
            list(map(float, param_a_str.split(','))),
            list(map(float, param_b_str.split(','))),  
            list(map(float, param_c_str.split(',')))
        ]
        
        print(f"📊 Parameter mappings:")
        print(f"  A parameters: {abc_param_lists[0]}")
        print(f"  B parameters: {abc_param_lists[1]}")
        print(f"  C parameters: {abc_param_lists[2]}")
        
        # Find all invbuf_*.lib files in data directory
        data_path = Path(data_dir)
        invbuf_files = list(data_path.glob("invbuf_*.lib"))
        
        if not invbuf_files:
            print(f"❌ No invbuf_*.lib files found in {data_dir}")
            sys.exit(1)
        
        print(f"📁 Found {len(invbuf_files)} invbuf files:")
        for f in sorted(invbuf_files):
            print(f"  {f.name}")
        
        # Step 1: build full dataset from .lib files
        data_input, data_output = build_all_data(
            filenames=[str(f) for f in sorted(invbuf_files)],
            abc_param_lists=abc_param_lists
        )

        # Step 2: Save the full dataset directly
        full_dataset_input = f"{output_dir}/transition_invbuf_test_dataset_input.pth"
        full_dataset_output = f"{output_dir}/transition_invbuf_test_dataset_output.pth"
        torch.save(data_input, full_dataset_input)
        torch.save(data_output, full_dataset_output)
        print(f"💾 Full test dataset saved → {full_dataset_input}, {full_dataset_output}")