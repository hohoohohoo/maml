# build_and_split_dataset.py
import torch
import os
from pathlib import Path
from utils.datasets import libdata
from utils.transform_sample_MAML_5feature import transform_all_samples
import sys
import importlib

def filetered_data(data_input,data_output):
    mask = data_input[..., -1] == 0 # 맨 뒤의 index가 0인 경우의 data만 남김
    #mask = torch.all(data_input[..., -3:] == 0, dim=-1) # 맨 뒤에서부터 3개의 index가 모두 0인 경우의 data만 남김
    filtered_input = data_input[mask]
    filtered_output = data_output[mask]
    return filtered_input,filtered_output

def dataextract(text, lib_prefix="", parse_liberty_fn=None, flatten_pin_fn=None):
    with open(text, "r") as f:
        lines = f.readlines()
    pin_data = parse_liberty_fn(lines)
    flattened,cap = flatten_pin_fn(pin_data)
    dataset= transform_all_samples(flattened,cap,lib_prefix)
    return dataset

def data_management(text, lib_prefix="", parse_liberty_fn=None, flatten_pin_fn=None):
    from pathlib import Path
    file_path = Path(text)
    dataset = dataextract(file_path, lib_prefix, parse_liberty_fn, flatten_pin_fn)
    dataset = libdata(dataset)
    return dataset
    
def build_all_data(start=40, end=101, prefix="OA_LVT_2_25_", save_input="full_input_tensor.pth",
                   save_output="full_output_tensor.pth", data_dir="OA_LVT",
                   parse_liberty_fn=None, flatten_pin_fn=None):
    data_input = []
    data_output = []

    for i in range(start, end):
        v_str = f"{i//100}{i%100:02d}"
        filename = f"{data_dir}/{prefix}{v_str}.lib"
        print(f"📥 Processing {filename}")
        dataset = data_management(filename, prefix, parse_liberty_fn, flatten_pin_fn)
        data_input.append(dataset.X)
        data_output.append(dataset.Y)
        print(i,dataset.X.size())

    data_input = torch.stack(data_input, dim=1)    # [3625, 61, 5]
    data_output = torch.stack(data_output, dim=1)  # [3625, 61, 1]
    torch.save(data_input, save_input)
    torch.save(data_output, save_output)
    print(f"✅ Saved: {save_input}, {save_output}")
    return data_input, data_output

def split_and_save_nodewise(input_path, output_path, output_dir, prefix, start_node, end_node):
    Path(output_dir).mkdir(exist_ok=True)
    data_input = torch.load(input_path)     # [samples, files, 5]
    data_output_all = torch.load(output_path)  # [samples, files, 1]
    
    # Since each sample now corresponds to a specific delay position,
    # we need to group samples by their position in the original 49-element delay array
    # Assuming samples are ordered by delay position (0-48 repeating for each file)
    
    samples_per_file = data_input.shape[0] // data_input.shape[1]  # Should be 49
    
    for i in range(start_node, end_node):
        # Extract samples corresponding to node i from all files
        # Each group of 49 samples represents one file, and position i within each group
        node_indices = []
        for file_idx in range(data_input.shape[1]):
            sample_idx = file_idx * samples_per_file + i
            if sample_idx < data_input.shape[0]:
                node_indices.append(sample_idx)
        
        if node_indices:
            data_input_node = data_input[node_indices]  # [files, 5]
            data_output_node = data_output_all[node_indices]  # [files, 1]
            
            input_save = f"{output_dir}/cell_{prefix}input_{i+1}nodes_align.pth"
            output_save = f"{output_dir}/cell_{prefix}output_{i+1}nodes_align.pth"
            print(f"Node {i+1}: {data_input_node.size()}, {data_output_node.size()}")
            torch.save(data_input_node, input_save)
            torch.save(data_output_node, output_save)
            print(f"💾 Node {i+1} saved → {input_save}, {output_save}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build and split voltage variation dataset')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory containing .lib files')
    parser.add_argument('--prefix', type=str, required=True, help='Lib file prefix (e.g., OA_LVT_2_25_)')
    parser.add_argument('--start', type=int, default=60, help='Start voltage value (default: 60)')
    parser.add_argument('--end', type=int, default=121, help='End voltage value (default: 121)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for saved tensors')
    parser.add_argument('--delay-type', type=str, default='cell', choices=['cell', 'transition'],
                       help='Delay type: cell or transition (default: cell)')
    parser.add_argument('--start-node', type=int, default=0, help='Start node for splitting (default: 0)')
    parser.add_argument('--end-node', type=int, default=49, help='End node for splitting (default: 49)')
    parser.add_argument('--split', action='store_true', help='Split by nodes instead of saving full dataset')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Dynamically import delay-type specific module
    print(f"📚 Loading delay type: {args.delay_type}")
    if args.delay_type == 'cell':
        delay_module = importlib.import_module('utils.libdata_extract_MAML_cell')
    else:  # transition
        delay_module = importlib.import_module('utils.libdata_extract_MAML_transition')

    parse_liberty_fn = delay_module.parse_liberty_pin_blocks
    flatten_pin_fn = delay_module.flatten_pin_data

    # Step 1: build full dataset from .lib files
    print(f"📊 Building dataset from {args.data_dir}")
    print(f"   Delay type: {args.delay_type}")
    print(f"   Voltage range: {args.start} - {args.end}")
    print(f"   Prefix: {args.prefix}")

    save_input = f"{args.output_dir}/full_input_tensor_{args.delay_type}.pth"
    save_output = f"{args.output_dir}/full_output_tensor_{args.delay_type}.pth"

    data_input, data_output = build_all_data(
        start=args.start,
        end=args.end,
        prefix=args.prefix,
        data_dir=args.data_dir,
        save_input=save_input,
        save_output=save_output,
        parse_liberty_fn=parse_liberty_fn,
        flatten_pin_fn=flatten_pin_fn
    )

    # Step 2: Save split by nodes or full dataset
    if args.split:
        print(f"\n📦 Splitting dataset by nodes ({args.start_node} - {args.end_node})")
        split_and_save_nodewise(
            save_input,
            save_output,
            output_dir=args.output_dir,
            prefix=f"{args.delay_type}_{args.prefix}",
            start_node=args.start_node,
            end_node=args.end_node
        )
        # Clean up intermediate files after node-wise splitting
        if os.path.exists(save_input):
            os.remove(save_input)
            print(f"🗑️  Removed intermediate file: {save_input}")
        if os.path.exists(save_output):
            os.remove(save_output)
            print(f"🗑️  Removed intermediate file: {save_output}")
    else:
        # Save the full dataset directly with better naming
        full_dataset_input = f"{args.output_dir}/{args.delay_type}_{args.prefix}dataset_input.pth"
        full_dataset_output = f"{args.output_dir}/{args.delay_type}_{args.prefix}dataset_output.pth"
        torch.save(data_input, full_dataset_input)
        torch.save(data_output, full_dataset_output)
        print(f"💾 Full dataset saved → {full_dataset_input}, {full_dataset_output}")

        # Clean up intermediate files (they are duplicates of the final files)
        if os.path.exists(save_input) and save_input != full_dataset_input:
            os.remove(save_input)
            print(f"🗑️  Removed intermediate file: {save_input}")
        if os.path.exists(save_output) and save_output != full_dataset_output:
            os.remove(save_output)
            print(f"🗑️  Removed intermediate file: {save_output}")
