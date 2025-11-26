#!/usr/bin/env python
# coding: utf-8

"""
TSMC Dataset Train/Test Splitter

Splits each corner/temperature combination into 80% train / 20% test datasets.

TSMC has:
- Corners: FF, SS, TT (3 corners)
- Temperatures: 0, 25, 50, 75, 100 (5 temps)
- Total: 15 combinations

Each lib_files directory generates one train/test dataset pair.
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

def create_directory_structure(base_path, corner, temp):
    """Create directory structure for a specific corner and temperature"""
    task_path = base_path / f"taskdivide_{corner.lower()}_{temp}"

    # Create directories
    (task_path / "testdatainput").mkdir(parents=True, exist_ok=True)
    (task_path / "testdataoutput").mkdir(parents=True, exist_ok=True)
    (task_path / "traindatainput").mkdir(parents=True, exist_ok=True)
    (task_path / "traindataoutput").mkdir(parents=True, exist_ok=True)

    return task_path

def split_and_save_data(processed_path, output_path, corner, temp, test_ratio=0.2):
    """Load data, split into train/test, and save"""

    # File paths
    cell_input_file = processed_path / f"cell_TSMC_{corner}_{temp}_dataset_input.pth"
    cell_output_file = processed_path / f"cell_TSMC_{corner}_{temp}_dataset_output.pth"
    transition_input_file = processed_path / f"transition_TSMC_{corner}_{temp}_dataset_input.pth"
    transition_output_file = processed_path / f"transition_TSMC_{corner}_{temp}_dataset_output.pth"

    # Process cell data
    if cell_input_file.exists() and cell_output_file.exists():
        print(f"  Processing cell data...")
        try:
            cell_input = torch.load(cell_input_file)
            cell_output = torch.load(cell_output_file)

            print(f"    Loaded: input {cell_input.shape}, output {cell_output.shape}")

            # Split
            total_samples = cell_input.shape[0]
            test_size = int(total_samples * test_ratio)

            indices = torch.randperm(total_samples)
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]

            # Train data
            train_input = cell_input[train_indices]
            train_output = cell_output[train_indices]

            # Ensure (N, time_steps, 1) format
            if train_output.dim() == 2:
                train_output = train_output.unsqueeze(-1)

            torch.save(train_input, output_path / "traindatainput" / "cell_train_input.pth")
            torch.save(train_output, output_path / "traindataoutput" / "cell_train_output.pth")
            print(f"    Saved cell train: input {train_input.shape}, output {train_output.shape}")

            # Test data
            test_input = cell_input[test_indices]
            test_output = cell_output[test_indices]

            # Ensure (N, time_steps, 1) format
            if test_output.dim() == 2:
                test_output = test_output.unsqueeze(-1)

            torch.save(test_input, output_path / "testdatainput" / "cell_test_input.pth")
            torch.save(test_output, output_path / "testdataoutput" / "cell_test_output.pth")
            print(f"    Saved cell test: input {test_input.shape}, output {test_output.shape}")

        except Exception as e:
            print(f"    Error processing cell data: {e}")
    else:
        print(f"  Cell data files not found")

    # Process transition data
    if transition_input_file.exists() and transition_output_file.exists():
        print(f"  Processing transition data...")
        try:
            transition_input = torch.load(transition_input_file)
            transition_output = torch.load(transition_output_file)

            print(f"    Loaded: input {transition_input.shape}, output {transition_output.shape}")

            # Split
            total_samples = transition_input.shape[0]
            test_size = int(total_samples * test_ratio)

            indices = torch.randperm(total_samples)
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]

            # Train data
            train_input = transition_input[train_indices]
            train_output = transition_output[train_indices]

            # Ensure (N, time_steps, 1) format
            if train_output.dim() == 2:
                train_output = train_output.unsqueeze(-1)

            torch.save(train_input, output_path / "traindatainput" / "transition_train_input.pth")
            torch.save(train_output, output_path / "traindataoutput" / "transition_train_output.pth")
            print(f"    Saved transition train: input {train_input.shape}, output {train_output.shape}")

            # Test data
            test_input = transition_input[test_indices]
            test_output = transition_output[test_indices]

            # Ensure (N, time_steps, 1) format
            if test_output.dim() == 2:
                test_output = test_output.unsqueeze(-1)

            torch.save(test_input, output_path / "testdatainput" / "transition_test_input.pth")
            torch.save(test_output, output_path / "testdataoutput" / "transition_test_output.pth")
            print(f"    Saved transition test: input {test_input.shape}, output {test_output.shape}")

        except Exception as e:
            print(f"    Error processing transition data: {e}")
    else:
        print(f"  Transition data files not found")

def process_combination(base_path, corner, temp):
    """Process a single corner and temperature combination"""
    print(f"\n{'='*80}")
    print(f"Processing {corner} - {temp}°C")
    print(f"{'='*80}")

    # Paths
    processed_path = base_path / "processed"

    if not processed_path.exists():
        print(f"  Processed path not found: {processed_path}")
        print(f"  Skipping...")
        return

    # Create output directory structure
    output_path = create_directory_structure(base_path, corner, temp)

    # Split and save data
    split_and_save_data(processed_path, output_path, corner, temp, test_ratio=0.2)

    print(f"  Completed: {output_path}")

def main():
    # Set random seed
    set_random_seed(42)
    print("Random seed set to 42 for reproducible results\n")

    # Base path
    base_path = Path("/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/temp_dataset_TSMC_dim5")

    # Corners and temperatures
    corners = ['FF', 'SS', 'TT','FS','SF']
    temperatures = [0, 25, 50, 75, 100]

    print(f"TSMC Dataset Train/Test Splitter")
    print(f"Base path: {base_path}")
    print(f"Corners: {', '.join(corners)}")
    print(f"Temperatures: {', '.join(map(str, temperatures))}")
    print(f"Total combinations: {len(corners)} x {len(temperatures)} = {len(corners) * len(temperatures)}")

    # Process all combinations
    for corner in corners:
        for temp in temperatures:
            process_combination(base_path, corner, temp)

    print(f"\n{'='*80}")
    print("All TSMC datasets created successfully!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
