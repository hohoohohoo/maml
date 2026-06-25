# build_and_split_dataset_simple_test.py
import torch
from pathlib import Path
from datasets import libdata
from libdata_extract_MAML import parse_liberty_pin_blocks, flatten_pin_data
from transform_sample_MAML_invbuf import transform_all_samples
import sys
import re

def parse_ex2_folder(folder_path):
    """
    Parse ex2_a_b_c_d folder name to extract a,b,c,d values
    Returns: (a, b, c, d)
    """
    folder_name = Path(folder_path).name
    
    # Pattern: ex2_a_b_c_d
    match = re.match(r'ex2_(\d+)_(\d+)_(\d+)_(\d+)', folder_name)
    if match:
        a, b, c, d = match.groups()
        return int(a), int(b), int(c), int(d)
    else:
        raise ValueError(f"Invalid ex2 folder format: {folder_name}")

def parse_ex2_filename(filename):
    """
    Parse ex2_a_b_c_d_voltage.lib filename to extract voltage only
    Returns: voltage
    """
    # Extract basename without .lib extension
    basename = Path(filename).stem
    
    # Pattern: ex2_a_b_c_d_voltage
    match = re.match(r'ex2_\d+_\d+_\d+_\d+_(\d+)', basename)
    if match:
        voltage = int(match.group(1))
        return voltage
    else:
        raise ValueError(f"Invalid ex2 filename format: {filename}")

def dataextract(text, lib_prefix="", abc_params=None):
    """Extract data from lib file with additional abc parameter mapping"""
    import torch
    
    with open(text, "r") as f:
        lines = f.readlines()
    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)
    datasets = transform_all_samples(flattened, cap, lib_prefix, abc_params)
    
    # transform_all_samples returns a list of dicts for multiple cells
    # We need to aggregate them into single input/output tensors
    if isinstance(datasets, list) and len(datasets) > 0:
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
            return (combined_input, combined_output)
    
    return None

def build_all_data_simple(filenames, abc_param_lists, folder_params=None, save_input="merged_simple_test_input.pth", save_output="merged_simple_test_output.pth"):
    """
    Build unified simple test dataset from all .lib files
    """
    data_input = []
    data_output = []
    processed_files = 0
    skipped_files = 0

    for filename in filenames:
        print(f"  Processing: {Path(filename).name}")
        
        try:
            # Parse filename to extract voltage only
            voltage = parse_ex2_filename(filename)
            
            if folder_params is not None:
                # Use folder parameters for a,b,c,d
                a, b, c, d = folder_params
                
                # Create abc_params dict following invbuf_test.py pattern
                # For each b,c index, we have both nmos and pmos versions
                abc_params = {
                    'a': abc_param_lists[0][a] if a < len(abc_param_lists[0]) else abc_param_lists[0][0],
                    'b_p': abc_param_lists[1][b * 2] if (b * 2) < len(abc_param_lists[1]) else abc_param_lists[1][0],      # b nmos 
                    'b_n': abc_param_lists[1][b * 2 + 1] if (b * 2 + 1) < len(abc_param_lists[1]) else abc_param_lists[1][1],  # b pmos
                    'c_p': abc_param_lists[2][c * 2] if (c * 2) < len(abc_param_lists[2]) else abc_param_lists[2][0],      # c nmos
                    'c_n': abc_param_lists[2][c * 2 + 1] if (c * 2 + 1) < len(abc_param_lists[2]) else abc_param_lists[2][1],  # c pmos
                }
                
                if processed_files == 0:  # Print once per folder
                    print(f"  Folder parameters: a={a}, b={b}, c={c}, d={d}")
                    print(f"  Mapped values: a={abc_params['a']}, b_n:{abc_params['b_n']}, b_p:{abc_params['b_p']}, c_n:{abc_params['c_n']}, c_p:{abc_params['c_p']}")
                    
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
            
            dataset = dataextract(filename, lib_prefix, abc_params)
            
            if dataset is not None:
                data_input.append(dataset[0])
                data_output.append(dataset[1])
                processed_files += 1
            else:
                skipped_files += 1
                print(f"    ⚠️ No data extracted from {Path(filename).name}")
                
        except Exception as e:
            print(f"    ⚠️ Error processing {Path(filename).name}: {e}")
            skipped_files += 1
            continue
    
    print(f"  📊 Processed: {processed_files} files, Skipped: {skipped_files} files")
    
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

def build_merged_data_simple(folder_paths, abc_param_lists, save_input="merged_simple_test_input.pth", save_output="merged_simple_test_output.pth"):
    """
    Build merged simple test dataset from multiple folders
    """
    all_inputs = []
    all_outputs = []
    folder_info = []
    
    print(f"🔄 Simple test merge mode: Building unified simple test dataset")
    print(f"📊 Parameter mappings:")
    print(f"  A parameters: {abc_param_lists[0]}")
    print(f"  B parameters: {abc_param_lists[1]}")
    print(f"  C parameters: {abc_param_lists[2]}")
    
    for folder_path in folder_paths:
        folder_name = Path(folder_path).name
        print(f"📁 Processing simple test folder: {folder_name}")
        
        # Parse folder name to get a,b,c,d values
        try:
            a, b, c, d = parse_ex2_folder(folder_path)
            print(f"  Folder parameters: a={a}, b={b}, c={c}, d={d}")
        except ValueError as e:
            print(f"  ❌ Error parsing folder name: {e}")
            continue
        
        # Find all .lib files in folder (not just invbuf, but all simple cells)
        lib_files = list(Path(folder_path).glob("*.lib"))
        print(f"  Found {len(lib_files)} .lib files")
        
        if len(lib_files) == 0:
            print(f"  ⚠️ No .lib files found, skipping...")
            continue
            
        try:
            # Process this folder with folder-level a,b,c,d parameters
            data_input, data_output = build_all_data_simple(
                filenames=[str(f) for f in sorted(lib_files)],
                abc_param_lists=abc_param_lists,
                folder_params=(a, b, c, d),  # Pass folder parameters
                save_input=f"temp_{folder_name}_input.pth",
                save_output=f"temp_{folder_name}_output.pth"
            )
            
            all_inputs.append(data_input)
            all_outputs.append(data_output)
            folder_info.append(folder_name)
            
            print(f"  ✅ {folder_name}: Input {data_input.shape}, Output {data_output.shape}")
            
            # Clean up temp files
            Path(f"temp_{folder_name}_input.pth").unlink(missing_ok=True)
            Path(f"temp_{folder_name}_output.pth").unlink(missing_ok=True)
            
        except Exception as e:
            print(f"  ❌ Error processing folder {folder_name}: {e}")
            continue
    
    if not all_inputs:
        raise ValueError("No valid folders were processed")

    # Concatenate all folders along samples dimension
    merged_input = torch.cat(all_inputs, dim=0)    # [total_samples, files, features]
    merged_output = torch.cat(all_outputs, dim=0)  # [total_samples, files, 1]
    
    torch.save(merged_input, save_input)
    torch.save(merged_output, save_output)
    
    print(f"✅ Merged simple test dataset saved:")
    print(f"   Input: {save_input} - Shape: {merged_input.shape}")
    print(f"   Output: {save_output} - Shape: {merged_output.shape}")
    print(f"   Processed {len(folder_info)} folders: {folder_info}")
    
    # Save folder mapping for reference
    mapping_file = save_input.replace('_input.pth', '_folder_mapping.txt')
    with open(mapping_file, 'w') as f:
        for i, folder in enumerate(folder_info):
            f.write(f"{i}: {folder}\n")
    print(f"📝 Simple test folder mapping saved: {mapping_file}")
    
    return merged_input, merged_output

def main():
    if len(sys.argv) < 6:
        print("Usage: python build_and_split_dataset_simple_test.py [--merge] <output_dir> <data_dir> <param_a> <param_b> <param_c>")
        sys.exit(1)
    
    if sys.argv[1] == "--merge":
        mode = "merge"
        output_dir = sys.argv[2]
        data_dir = sys.argv[3]
        param_a = sys.argv[4]
        param_b = sys.argv[5]
        param_c = sys.argv[6]
    else:
        mode = "single"
        output_dir = sys.argv[1]
        data_dir = sys.argv[2]
        param_a = sys.argv[3]
        param_b = sys.argv[4]
        param_c = sys.argv[5]
    
    # Parse parameter lists
    abc_param_lists = [
        [float(x) for x in param_a.split(',')],
        [float(x) for x in param_b.split(',')],
        [float(x) for x in param_c.split(',')]
    ]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if mode == "merge":
        # Find all folders (can be ex2_* or any other pattern)
        data_path = Path(data_dir)
        test_folders = [f for f in data_path.iterdir() if f.is_dir()]
        
        print(f"📁 Found {len(test_folders)} simple test folders:")
        for folder in sorted(test_folders):
            print(f"  {folder.name}")
        
        if not test_folders:
            print("❌ No folders found!")
            sys.exit(1)
        
        # Build merged dataset from all folders
        data_input, data_output = build_merged_data_simple(
            folder_paths=[str(f) for f in sorted(test_folders)],
            abc_param_lists=abc_param_lists,
            save_input=f"{output_dir}/cell_merged_simple_test_input.pth",
            save_output=f"{output_dir}/cell_merged_simple_test_output.pth"
        )
        
        print(f"✅ Unified SIMPLE TEST dataset creation completed!")
        
    else:
        # Single folder processing
        lib_files = list(Path(data_dir).glob("*.lib"))
        print(f"Found {len(lib_files)} .lib files")
        
        if len(lib_files) == 0:
            print("No .lib files found!")
            sys.exit(1)
        
        # Build dataset from .lib files
        data_input, data_output = build_all_data_simple(
            filenames=[str(f) for f in sorted(lib_files)],
            abc_param_lists=abc_param_lists,
            save_input=f"{output_dir}/cell_merged_simple_test_input.pth",
            save_output=f"{output_dir}/cell_merged_simple_test_output.pth"
        )
        
        full_dataset_input = f"{output_dir}/cell_simple_test_full_input.pth"
        full_dataset_output = f"{output_dir}/cell_simple_test_full_output.pth"
        torch.save(data_input, full_dataset_input)
        torch.save(data_output, full_dataset_output)

if __name__ == "__main__":
    main()