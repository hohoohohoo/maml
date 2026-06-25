#!/usr/bin/env python3
"""
Build merged invbuf dataset from all folders at once
Creates single dataset with shape [total_samples, 61, 9] and [total_samples, 61, 1]
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import re
from datasets import libdata
from libdata_extract_MAML import parse_liberty_pin_blocks, flatten_pin_data
from transform_sample_MAML_invbuf import transform_all_samples

def extract_invbuf_data(lib_file_path, param_mappings):
    """Extract data from a single .lib file"""
    try:
        with open(lib_file_path, "r") as f:
            lines = f.readlines()
        pin_data = parse_liberty_pin_blocks(lines)
        flattened, cap = flatten_pin_data(pin_data)
        
        # Transform with parameter mappings
        dataset = transform_all_samples(flattened, cap, param_mappings)
        return dataset
    except Exception as e:
        print(f"⚠️ Error processing {lib_file_path}: {e}")
        return None

def parse_folder_name(folder_name):
    """Parse invbuf folder name to extract a,b,c,temperature parameters
    Handles formats like:
    - invbuf_0_0_0_m25 (m25 = -25)
    - invbuf_0_0_0_62p5 (62p5 = 62.5)
    - invbuf_0_0_0_125 (125 = 125)
    Returns: (a, b, c, temperature)
    """
    match = re.match(r'invbuf_(\d+)_(\d+)_(\d+)_(.+)', folder_name)
    if match:
        a, b, c, temp_str = match.groups()
        
        # Parse temperature string - same logic as build_batch_invbuf_dataset.py
        if temp_str.startswith('m'):
            # Handle negative: m25 -> -25
            temp_str = '-' + temp_str[1:]
        
        if 'p' in temp_str:
            # Handle decimal: 62p5 -> 62.5
            temp_str = temp_str.replace('p', '.')
        
        try:
            temperature = float(temp_str)
        except ValueError:
            temperature = 25.0  # Default
        
        return int(a), int(b), int(c), temperature
    else:
        return None, None, None, None

def create_parameter_mappings(a_idx, b_idx, c_idx, param_a, param_b, param_c):
    """Create parameter mappings for transform_sample_MAML"""
    param_a_list = [float(x) for x in param_a.split(',')]
    param_b_list = [float(x) for x in param_b.split(',')]
    param_c_list = [float(x) for x in param_c.split(',')]
    
    # Get values for this folder's indices
    a_val = param_a_list[a_idx] if a_idx < len(param_a_list) else param_a_list[0]
    
    # B and C are nmos,pmos pairs - get the pair for this index
    if b_idx * 2 + 1 < len(param_b_list):
        b_nmos = param_b_list[b_idx * 2]
        b_pmos = param_b_list[b_idx * 2 + 1]
    else:
        b_nmos = param_b_list[0]
        b_pmos = param_b_list[1]
        
    if c_idx * 2 + 1 < len(param_c_list):
        c_nmos = param_c_list[c_idx * 2]
        c_pmos = param_c_list[c_idx * 2 + 1]
    else:
        c_nmos = param_c_list[0]
        c_pmos = param_c_list[1]
    
    return {
        'param_a': a_val,
        'param_b_nmos': b_nmos,
        'param_b_pmos': b_pmos,
        'param_c_nmos': c_nmos,
        'param_c_pmos': c_pmos
    }

def build_merged_invbuf_dataset(data_dir, output_dir, start_voltage=40, end_voltage=101, 
                               param_a="0.625,0.875,1.125,1.375",
                               param_b="0.089,0.06,0.091,0.064,0.093,0.068,0.095,0.072",
                               param_c="0.35,0.465,0.37,0.473,0.39,0.478,0.41,0.485"):
    """Build merged dataset from all invbuf folders"""
    
    print("🚀 Building Merged INVBUF Dataset")
    print("=" * 50)
    print(f"📁 Data directory: {data_dir}")
    print(f"📊 Output directory: {output_dir}")
    print(f"⚡ Voltage range: {start_voltage}-{end_voltage}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all invbuf folders
    invbuf_folders = list(Path(data_dir).glob("invbuf_*"))
    invbuf_folders.sort()
    
    print(f"📂 Found {len(invbuf_folders)} invbuf folders")
    
    if len(invbuf_folders) == 0:
        print("❌ No invbuf folders found!")
        return None
        
    # Collect all data
    all_inputs = []
    all_outputs = []
    folder_info = []
    
    for folder_path in invbuf_folders:
        folder_name = folder_path.name
        print(f"\n📁 Processing {folder_name}...")
        
        # Parse folder parameters including temperature
        a_idx, b_idx, c_idx, temperature = parse_folder_name(folder_name)
        if a_idx is None:
            print(f"   ⚠️ Cannot parse folder name: {folder_name}, skipping...")
            continue
        
        print(f"   🔧 Parameters: a={a_idx}, b={b_idx}, c={c_idx}, temp={temperature}°C")
        
        # Create parameter mappings including temperature
        param_mappings = create_parameter_mappings(a_idx, b_idx, c_idx, param_a, param_b, param_c)
        param_mappings['temperature'] = temperature
        print(f"   📊 Mapped values: a={param_mappings['param_a']:.3f}, "
              f"b_nmos={param_mappings['param_b_nmos']:.3f}, b_pmos={param_mappings['param_b_pmos']:.3f}, "
              f"c_nmos={param_mappings['param_c_nmos']:.3f}, c_pmos={param_mappings['param_c_pmos']:.3f}")
        
        # Find .lib files in this folder (only current folder, not subdirs)
        lib_files = list(folder_path.glob("*.lib"))
        lib_files.sort()
        
        print(f"   📚 Found {len(lib_files)} .lib files")
        
        if len(lib_files) == 0:
            print(f"   ❌ No .lib files found in {folder_name}, skipping...")
            continue
        
        # Process files for expected voltage range
        folder_inputs = []
        folder_outputs = []
        
        processed_count = 0
        for voltage in range(start_voltage, end_voltage):
            # Find file with this voltage
            voltage_files = [f for f in lib_files if f"_{voltage:03d}.lib" in f.name]
            
            if not voltage_files:
                print(f"   ⚠️ Missing voltage {voltage:03d}")
                continue
                
            lib_file = voltage_files[0]  # Use first match
            
            # Extract data
            dataset = extract_invbuf_data(lib_file, param_mappings)
            if dataset is None:
                print(f"   ❌ Failed to process {lib_file.name}")
                continue
                
            # Convert to tensors
            input_tensor = dataset['input']    # [samples, 61, 9]
            output_tensor = dataset['output']  # [samples, 61, 1]
            
            folder_inputs.append(input_tensor)
            folder_outputs.append(output_tensor)
            processed_count += 1
        
        print(f"   ✅ Processed {processed_count}/{end_voltage-start_voltage} voltage files")
        
        if len(folder_inputs) > 0:
            # Stack along voltage dimension to get [samples*voltages, 61, 9]
            folder_input_stacked = torch.cat(folder_inputs, dim=0)
            folder_output_stacked = torch.cat(folder_outputs, dim=0)
            
            print(f"   📊 Folder data shape: input={folder_input_stacked.shape}, output={folder_output_stacked.shape}")
            
            all_inputs.append(folder_input_stacked)
            all_outputs.append(folder_output_stacked)
            folder_info.append(folder_name)
        else:
            print(f"   ❌ No valid data from {folder_name}")
    
    if len(all_inputs) == 0:
        print("❌ No valid data collected from any folder!")
        return None
    
    # Merge all folders
    print(f"\n🔄 Merging data from {len(all_inputs)} folders...")
    
    merged_input = torch.cat(all_inputs, dim=0)    # [total_samples, 61, 9]
    merged_output = torch.cat(all_outputs, dim=0)  # [total_samples, 61, 1]
    
    print(f"✅ Final merged dataset shapes:")
    print(f"   Input:  {merged_input.shape}")
    print(f"   Output: {merged_output.shape}")
    
    # Save merged dataset
    input_file = Path(output_dir) / "merged_invbuf_input.pth"
    output_file = Path(output_dir) / "merged_invbuf_output.pth"
    
    torch.save(merged_input, input_file)
    torch.save(merged_output, output_file)
    
    print(f"💾 Merged dataset saved:")
    print(f"   📁 {input_file}")
    print(f"   📁 {output_file}")
    
    # Save folder info
    info_file = Path(output_dir) / "folder_info.txt"
    with open(info_file, 'w') as f:
        f.write("Folders included in merged dataset:\n")
        for i, folder in enumerate(folder_info):
            f.write(f"{i}: {folder}\n")
    
    print(f"📝 Folder info saved: {info_file}")
    
    return {
        'input': merged_input,
        'output': merged_output,
        'folders': folder_info
    }

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python build_merged_invbuf_dataset.py <output_dir> <data_dir> <start_voltage> <end_voltage> <param_a> <param_b> <param_c>")
        print("Example: python build_merged_invbuf_dataset.py merged_output ../dataset_ex2/processed 40 101 '0.625,0.875,1.125,1.375' '0.089,0.06,0.091,0.064' '0.35,0.465,0.37,0.473'")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    data_dir = sys.argv[2] 
    start_voltage = int(sys.argv[3])
    end_voltage = int(sys.argv[4])
    param_a = sys.argv[5]
    param_b = sys.argv[6]
    param_c = sys.argv[7]
    
    result = build_merged_invbuf_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        start_voltage=start_voltage,
        end_voltage=end_voltage,
        param_a=param_a,
        param_b=param_b,
        param_c=param_c
    )
    
    if result:
        print("\n✅ Merged dataset generation completed!")
        print(f"   Total samples: {result['input'].shape[0]}")
        print(f"   Folders processed: {len(result['folders'])}")
    else:
        print("\n❌ Failed to generate merged dataset!")
        sys.exit(1)