import torch
import os
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

def create_dataset_directory_structure(base_path, vt_type):
    """Create directory structure for LVT or RVT datasets"""
    vt_path = Path(base_path) / f"taskdivide_{vt_type.lower()}"
    
    # Create directories
    (vt_path / "testdatainput").mkdir(parents=True, exist_ok=True)
    (vt_path / "testdataoutput").mkdir(parents=True, exist_ok=True) 
    (vt_path / "traindatainput").mkdir(parents=True, exist_ok=True)
    (vt_path / "traindataoutput").mkdir(parents=True, exist_ok=True)
    
    return vt_path

def merge_nodewise_data_by_vt_type(nodewise_path, vt_type, cell_types=['AO', 'OA', 'INVBUF', 'simple']):
    """Merge nodewise data for a specific VT type (LVT, RVT, SLVT, SRAM)"""
    
    # Data containers
    vt_data = {
        'cell_input': [],
        'cell_output': [],
        'transition_input': [],
        'transition_output': []
    }
    
    print(f"Processing {vt_type} data...")
    
    for cell_type in cell_types:
        print(f"  Processing {cell_type}...")
        
        # Cell data files in the actual format
        cell_input_file = nodewise_path / f"cell_{cell_type}_{vt_type}_2_25_dataset_input.pth"
        cell_output_file = nodewise_path / f"cell_{cell_type}_{vt_type}_2_25_dataset_output.pth"
        
        # Load cell data if exists
        if cell_input_file.exists() and cell_output_file.exists():
            try:
                cell_input = torch.load(cell_input_file)
                cell_output = torch.load(cell_output_file)
                vt_data['cell_input'].append(cell_input)
                vt_data['cell_output'].append(cell_output)
                print(f"    Loaded cell {cell_type} data: input shape {cell_input.shape}, output shape {cell_output.shape}")
            except Exception as e:
                print(f"    Error loading cell {cell_type}: {e}")
        else:
            print(f"    Cell {cell_type} files not found")
    
    # Look for transition data (only for OA_RVT based on the files we saw)
        transition_input_file = nodewise_path / f"transition_{cell_type}_{vt_type}_2_25_dataset_input.pth"
        transition_output_file = nodewise_path / f"transition_{cell_type}_{vt_type}_2_25_dataset_output.pth"
        
        if transition_input_file.exists() and transition_output_file.exists():
            try:
                transition_input = torch.load(transition_input_file)
                transition_output = torch.load(transition_output_file)
                vt_data['transition_input'].append(transition_input)
                vt_data['transition_output'].append(transition_output)
                print(f"    Loaded transition OA data: input shape {transition_input.shape}, output shape {transition_output.shape}")
            except Exception as e:
                print(f"    Error loading transition OA: {e}")
    
    return vt_data

def split_train_test(vt_data, test_ratio=0.2):
    """Split data into train and test sets - stratified split by input component values"""
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
    
    # Process input and output data together to maintain correspondence
    input_types = [('cell_input', 'cell_output'), ('transition_input', 'transition_output')]
    
    for input_type, output_type in input_types:
        input_data_list = vt_data[input_type]
        output_data_list = vt_data[output_type]
        
        if input_data_list and output_data_list:
            # Merge all tensors into one
            merged_input = torch.cat(input_data_list, dim=0)
            merged_output = torch.cat(output_data_list, dim=0)
            
            total_samples = merged_input.shape[0]
            test_size = int(total_samples * test_ratio)
            
            print(f"    {input_type} Data shape: {merged_input.shape}")
            print(f"    {input_type} Total samples: {total_samples}, Test size: {test_size}")
            
            # Simple random split
            indices = torch.randperm(total_samples)
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
            
            if test_size > 0:
                test_data[input_type] = [merged_input[test_indices]]
                test_data[output_type] = [merged_output[test_indices]]
            
            if len(train_indices) > 0:
                train_data[input_type] = [merged_input[train_indices]]
                train_data[output_type] = [merged_output[train_indices]]
            
            print(f"    {input_type} Split: Train={len(train_indices)}, Test={len(test_indices)}")
    
    return train_data, test_data

def save_data(data, base_path, split_type):
    """Save data, separating cell and transition data"""
    input_dir = base_path / f"{split_type}datainput"
    output_dir = base_path / f"{split_type}dataoutput"
    
    # Save cell data
    if data['cell_input']:
        merged_cell_input = torch.cat(data['cell_input'], dim=0) if len(data['cell_input']) > 1 else data['cell_input'][0]
        cell_input_filename = f"cell_{split_type}_input.pth"
        torch.save(merged_cell_input, input_dir / cell_input_filename)
        print(f"    Saved {cell_input_filename} with shape {merged_cell_input.shape}")
    
    if data['cell_output']:
        merged_cell_output = torch.cat(data['cell_output'], dim=0) if len(data['cell_output']) > 1 else data['cell_output'][0]
        # Add last dimension to make it (N, time_steps, 1)
        if merged_cell_output.dim() == 2:
            merged_cell_output = merged_cell_output.unsqueeze(-1)
        cell_output_filename = f"cell_{split_type}_output.pth"
        torch.save(merged_cell_output, output_dir / cell_output_filename)
        print(f"    Saved {cell_output_filename} with shape {merged_cell_output.shape}")
    
    # Save transition data
    if data['transition_input']:
        merged_transition_input = torch.cat(data['transition_input'], dim=0) if len(data['transition_input']) > 1 else data['transition_input'][0]
        transition_input_filename = f"transition_{split_type}_input.pth"
        torch.save(merged_transition_input, input_dir / transition_input_filename)
        print(f"    Saved {transition_input_filename} with shape {merged_transition_input.shape}")
    
    if data['transition_output']:
        merged_transition_output = torch.cat(data['transition_output'], dim=0) if len(data['transition_output']) > 1 else data['transition_output'][0]
        # Add last dimension to make it (N, time_steps, 1)
        if merged_transition_output.dim() == 2:
            merged_transition_output = merged_transition_output.unsqueeze(-1)
        transition_output_filename = f"transition_{split_type}_output.pth"
        torch.save(merged_transition_output, output_dir / transition_output_filename)
        print(f"    Saved {transition_output_filename} with shape {merged_transition_output.shape}")

def main():
    # Set random seed for reproducibility
    set_random_seed(42)
    print("Random seed set to 42 for reproducible results")
    
    # Paths
    base_path = Path("/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_test5(dim5)")
    nodewise_path = base_path / "processed"    # Cell types to merge
    cell_types = ['AO', 'OA', 'INVBUF', 'simple']
    vt_types = ['LVT', 'RVT', 'SRAM', 'SLVT']
    
    for vt_type in vt_types:
        print(f"\n=== Processing {vt_type} ===")
        
        # Create directory structure
        vt_path = create_dataset_directory_structure(base_path, vt_type)
        
        # Merge nodewise data for this VT type
        vt_data = merge_nodewise_data_by_vt_type(nodewise_path, vt_type, cell_types)
        
        # Print summary
        print(f"  Found data:")
        print(f"    Cell input: {len(vt_data['cell_input'])} files")
        print(f"    Cell output: {len(vt_data['cell_output'])} files")
        print(f"    Transition input: {len(vt_data['transition_input'])} files")
        print(f"    Transition output: {len(vt_data['transition_output'])} files")
        
        # Skip if no data found
        if not any(vt_data.values()):
            print(f"  No data found for {vt_type}, skipping...")
            continue
        
        # Split into train/test (8:2 ratio)
        train_data, test_data = split_train_test(vt_data, test_ratio=0.2)
        
        # Save train and test data
        save_data(train_data, vt_path, "train")
        save_data(test_data, vt_path, "test")
        
        print(f"  {vt_type} dataset creation completed!")
        print(f"  Created merged cell and transition files (train/test split 8:2)")

if __name__ == "__main__":
    main()