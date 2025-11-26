#!/usr/bin/env python
# coding: utf-8

"""
ASAP7 Dataset Merger and Train/Test Splitter

Merges cell type data for each VT type and corner combination,
then splits into 80% train / 20% test datasets.

ASAP7 has:
- VT types: LVT, RVT, SLVT, SRAM (4 types)
- Corners: FF(1), TT(2), SS(3) (3 corners)
- Total: 12 combinations

Each combination merges cell types: AO, OA, simple, INVBUF
"""

import torch
import random
import numpy as np
from pathlib import Path

# Set random seeds for reproducibility
def set_random_seed(seed=42):
    """Set random seed for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directory_structure(base_path, vt_type, corner_name):
    """Create directory structure for a specific VT type and corner"""
    task_path = base_path / f"taskdivide_{vt_type.lower()}_{corner_name}"

    # Create directories
    (task_path / "testdatainput").mkdir(parents=True, exist_ok=True)
    (task_path / "testdataoutput").mkdir(parents=True, exist_ok=True)
    (task_path / "traindatainput").mkdir(parents=True, exist_ok=True)
    (task_path / "traindataoutput").mkdir(parents=True, exist_ok=True)

    return task_path

def merge_cell_types(processed_path, vt_type, corner_num, cell_types=['AO', 'OA', 'simple', 'INVBUF']):
    """Merge data for different cell types of the same VT type and corner"""

    data = {
        'cell_input': [],
        'cell_output': [],
        'transition_input': [],
        'transition_output': []
    }

    print(f"  Processing cell types for {vt_type}...")

    for cell_type in cell_types:
        # Cell data files
        cell_input_file = processed_path / f"cell_{cell_type}_{vt_type}_{corner_num}_25_dataset_input.pth"
        cell_output_file = processed_path / f"cell_{cell_type}_{vt_type}_{corner_num}_25_dataset_output.pth"

        # Load cell data if exists
        if cell_input_file.exists() and cell_output_file.exists():
            try:
                cell_input = torch.load(cell_input_file)
                cell_output = torch.load(cell_output_file)
                data['cell_input'].append(cell_input)
                data['cell_output'].append(cell_output)
                print(f"    Loaded cell {cell_type}: input {cell_input.shape}, output {cell_output.shape}")
            except Exception as e:
                print(f"    Error loading cell {cell_type}: {e}")

        # Transition data files
        transition_input_file = processed_path / f"transition_{cell_type}_{vt_type}_{corner_num}_25_dataset_input.pth"
        transition_output_file = processed_path / f"transition_{cell_type}_{vt_type}_{corner_num}_25_dataset_output.pth"

        if transition_input_file.exists() and transition_output_file.exists():
            try:
                transition_input = torch.load(transition_input_file)
                transition_output = torch.load(transition_output_file)
                data['transition_input'].append(transition_input)
                data['transition_output'].append(transition_output)
                print(f"    Loaded transition {cell_type}: input {transition_input.shape}, output {transition_output.shape}")
            except Exception as e:
                print(f"    Error loading transition {cell_type}: {e}")

    return data

def split_train_test(data, test_ratio=0.2):
    """Split data into train and test sets (80:20)"""
    train_data = {
        'cell_input': [],
        'cell_output': [],
        'transition_input': [],
        'transition_output': []
    }
    test_data = {
        'cell_input': [],
        'cell_output': [],
        'transition_input': [],
        'transition_output': []
    }

    # Process input and output data together
    input_types = [('cell_input', 'cell_output'), ('transition_input', 'transition_output')]

    for input_type, output_type in input_types:
        input_data_list = data[input_type]
        output_data_list = data[output_type]

        if input_data_list and output_data_list:
            # Merge all tensors
            merged_input = torch.cat(input_data_list, dim=0)
            merged_output = torch.cat(output_data_list, dim=0)

            total_samples = merged_input.shape[0]
            test_size = int(total_samples * test_ratio)

            print(f"    {input_type}: shape {merged_input.shape}, total {total_samples}, test {test_size}")

            # Random split
            indices = torch.randperm(total_samples)
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]

            if test_size > 0:
                test_data[input_type] = [merged_input[test_indices]]
                test_data[output_type] = [merged_output[test_indices]]

            if len(train_indices) > 0:
                train_data[input_type] = [merged_input[train_indices]]
                train_data[output_type] = [merged_output[train_indices]]

            print(f"    Split: train={len(train_indices)}, test={len(test_indices)}")

    return train_data, test_data

def save_data(data, base_path, split_type):
    """Save train or test data"""
    input_dir = base_path / f"{split_type}datainput"
    output_dir = base_path / f"{split_type}dataoutput"

    # Save cell data
    if data['cell_input']:
        merged_cell_input = data['cell_input'][0]
        torch.save(merged_cell_input, input_dir / f"cell_{split_type}_input.pth")
        print(f"    Saved cell_{split_type}_input.pth: {merged_cell_input.shape}")

    if data['cell_output']:
        merged_cell_output = data['cell_output'][0]
        # Ensure (N, time_steps, 1) format
        if merged_cell_output.dim() == 2:
            merged_cell_output = merged_cell_output.unsqueeze(-1)
        torch.save(merged_cell_output, output_dir / f"cell_{split_type}_output.pth")
        print(f"    Saved cell_{split_type}_output.pth: {merged_cell_output.shape}")

    # Save transition data
    if data['transition_input']:
        merged_transition_input = data['transition_input'][0]
        torch.save(merged_transition_input, input_dir / f"transition_{split_type}_input.pth")
        print(f"    Saved transition_{split_type}_input.pth: {merged_transition_input.shape}")

    if data['transition_output']:
        merged_transition_output = data['transition_output'][0]
        # Ensure (N, time_steps, 1) format
        if merged_transition_output.dim() == 2:
            merged_transition_output = merged_transition_output.unsqueeze(-1)
        torch.save(merged_transition_output, output_dir / f"transition_{split_type}_output.pth")
        print(f"    Saved transition_{split_type}_output.pth: {merged_transition_output.shape}")

def process_combination(base_path, vt_type, corner_name, corner_num):
    """Process a single VT type and corner combination"""
    print(f"\n{'='*80}")
    print(f"Processing {vt_type} - {corner_name} (corner {corner_num})")
    print(f"{'='*80}")

    # Paths
    processed_path = base_path / "processed"

    if not processed_path.exists():
        print(f"  Processed path not found: {processed_path}")
        print(f"  Skipping...")
        return

    # Create output directory structure
    output_path = create_directory_structure(base_path, vt_type, corner_name)

    # Merge cell types
    data = merge_cell_types(processed_path, vt_type, corner_num)

    # Check if any data found
    if not any(data.values()):
        print(f"  No data found for {vt_type} - {corner_name}")
        print(f"  Skipping...")
        return

    # Split into train/test (80:20)
    print(f"  Splitting data (80:20)...")
    train_data, test_data = split_train_test(data, test_ratio=0.2)

    # Save train and test data
    print(f"  Saving train data...")
    save_data(train_data, output_path, "train")

    print(f"  Saving test data...")
    save_data(test_data, output_path, "test")

    print(f"  Completed: {output_path}")

def main():
    # Set random seed
    set_random_seed(42)
    print("Random seed set to 42 for reproducible results\n")

    # Base path
    base_path = Path("/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/temp_dataset_ASAP7_dim5")

    # VT types and corners
    vt_types = ['LVT', 'RVT', 'SLVT', 'SRAM']
    corners = [
        ('FF', 1),
        ('TT', 2),
        ('SS', 3)
    ]

    print(f"ASAP7 Dataset Merger")
    print(f"Base path: {base_path}")
    print(f"VT types: {', '.join(vt_types)}")
    print(f"Corners: {', '.join([c[0] for c in corners])}")
    print(f"Total combinations: {len(vt_types)} x {len(corners)} = {len(vt_types) * len(corners)}")

    # Process all combinations
    for vt_type in vt_types:
        for corner_name, corner_num in corners:
            process_combination(base_path, vt_type, corner_name, corner_num)

    print(f"\n{'='*80}")
    print("All ASAP7 datasets created successfully!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
