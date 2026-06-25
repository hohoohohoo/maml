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

def create_dataset_directory_structure(base_path):
    """Create directory structure for combined datasets"""
    combined_path = Path(base_path) / "taskdivide_combined"
    
    # Create directories
    (combined_path / "testdatainput").mkdir(parents=True, exist_ok=True)
    (combined_path / "testdataoutput").mkdir(parents=True, exist_ok=True) 
    (combined_path / "traindatainput").mkdir(parents=True, exist_ok=True)
    (combined_path / "traindataoutput").mkdir(parents=True, exist_ok=True)
    
    return combined_path

def load_combined_nodewise_data(nodewise_path, num_nodes=49):
    """Load combined nodewise data from build_and_split_dataset_all.py output"""
    
    # Data containers organized by i value
    i_nodes_data = {}
    
    print(f"Loading combined nodewise data from {nodewise_path}...")
    
    # Initialize containers for each i value (0-48)
    for i in range(num_nodes):
        i_nodes_data[i] = {
            'input': None,
            'output': None
        }
    
    # Load each combined dataset file
    for i in range(num_nodes):
        # File names from build_and_split_dataset_all.py
        input_file = nodewise_path / f"transition_combined_input_{i}nodes_test.pth"  # Note: i starts from 0 but file name uses i
        output_file = nodewise_path / f"transition_combined_output_{i}nodes_test.pth"
        
        print(f"  Loading node {i}: {input_file.name}, {output_file.name}")
        
        # Load input data if exists
        if input_file.exists():
            try:
                input_data = torch.load(input_file)
                i_nodes_data[i]['input'] = input_data
                print(f"    Input shape: {input_data.shape}")
            except Exception as e:
                print(f"    Error loading input {i}: {e}")
        else:
            print(f"    Input file not found: {input_file}")
        
        # Load output data if exists  
        if output_file.exists():
            try:
                output_data = torch.load(output_file)
                i_nodes_data[i]['output'] = output_data
                print(f"    Output shape: {output_data.shape}")
            except Exception as e:
                print(f"    Error loading output {i}: {e}")
        else:
            print(f"    Output file not found: {output_file}")
    
    return i_nodes_data

def split_train_test_by_i(i_nodes_data, test_ratio=0.2, num_nodes=49):
    """Split data into train and test sets for each i value - stratified split by input features"""
    train_data_by_i = {}
    test_data_by_i = {}
    
    for i in range(num_nodes):
        train_data_by_i[i] = {
            'input': None,
            'output': None
        }
        test_data_by_i[i] = {
            'input': None,
            'output': None
        }
        
        input_data = i_nodes_data[i]['input']
        output_data = i_nodes_data[i]['output']
        
        if input_data is not None and output_data is not None:
            print(f"  Processing node {i}: input {input_data.shape}, output {output_data.shape}")
            
            # Check if input has sufficient dimensions for stratification
            if input_data.dim() >= 3 and input_data.shape[2] >= 2:
                # Use second feature (index 1) for stratification like the original script
                component_values = input_data[:, 0, 1]  # Second feature of first time step
                unique_values = torch.unique(component_values)
                
                print(f"    Unique second feature values: {unique_values.tolist()}")
                print(f"    Sample of second feature values: {component_values[:10].tolist()}")
                
                # Use the actual unique values found in data for stratification
                all_train_indices = []
                all_test_indices = []
                
                # Split each unique component value separately
                for component_val in unique_values:
                    indices = torch.where(component_values == component_val)[0]
                    
                    if len(indices) > 0:
                        # Calculate split sizes
                        n_test = max(1, int(len(indices) * test_ratio))
                        n_train = len(indices) - n_test
                        
                        # Random shuffle indices within this component group
                        shuffled_indices = indices[torch.randperm(len(indices))]
                        
                        # Add to respective lists
                        component_train_indices = shuffled_indices[:n_train]
                        component_test_indices = shuffled_indices[n_train:n_train+n_test]
                        
                        all_train_indices.extend(component_train_indices.tolist())
                        all_test_indices.extend(component_test_indices.tolist())
                        
                        print(f"    Feature {component_val.item()}: "
                              f"Total={len(indices)}, Train={len(component_train_indices)}, Test={len(component_test_indices)}")
                
                # Convert to tensor
                all_train_indices = torch.tensor(sorted(all_train_indices))
                all_test_indices = torch.tensor(sorted(all_test_indices))
                
                # Verify no overlap between train and test
                overlap = set(all_train_indices.tolist()) & set(all_test_indices.tolist())
                if overlap:
                    print(f"    ERROR: Found {len(overlap)} overlapping indices!")
                    # Fallback to simple split
                    total_samples = input_data.shape[0]
                    test_size = int(total_samples * test_ratio)
                    
                    if test_size > 0:
                        test_data_by_i[i]['input'] = input_data[-test_size:]
                        test_data_by_i[i]['output'] = output_data[-test_size:]
                        train_data_by_i[i]['input'] = input_data[:-test_size]
                        train_data_by_i[i]['output'] = output_data[:-test_size]
                    else:
                        train_data_by_i[i]['input'] = input_data
                        train_data_by_i[i]['output'] = output_data
                    
                    print(f"    Fallback split: Train={total_samples-test_size}, Test={test_size}")
                    continue
                
                # Split data using stratified indices
                if len(all_train_indices) > 0:
                    train_data_by_i[i]['input'] = input_data[all_train_indices]
                    train_data_by_i[i]['output'] = output_data[all_train_indices]
                
                if len(all_test_indices) > 0:
                    test_data_by_i[i]['input'] = input_data[all_test_indices]
                    test_data_by_i[i]['output'] = output_data[all_test_indices]
                    
                print(f"    Final stratified split: Train={len(all_train_indices)}, Test={len(all_test_indices)}")
            
            else:
                # Fallback to simple split if input doesn't have expected dimensions
                total_samples = input_data.shape[0]
                test_size = int(total_samples * test_ratio)
                
                if test_size > 0:
                    test_data_by_i[i]['input'] = input_data[-test_size:]
                    test_data_by_i[i]['output'] = output_data[-test_size:]
                    train_data_by_i[i]['input'] = input_data[:-test_size]
                    train_data_by_i[i]['output'] = output_data[:-test_size]
                else:
                    train_data_by_i[i]['input'] = input_data
                    train_data_by_i[i]['output'] = output_data
                
                print(f"    Simple split: Train={total_samples-test_size}, Test={test_size}")
        else:
            print(f"  Skipping node {i}: missing data")
    
    return train_data_by_i, test_data_by_i

def save_data_by_i(data_by_i, base_path, split_type, num_nodes=49):
    """Save data organized by i value (0-48)"""
    input_dir = base_path / f"{split_type}datainput"
    output_dir = base_path / f"{split_type}dataoutput"
    
    for i in range(num_nodes):
        # Save input data for this i value
        if data_by_i[i]['input'] is not None:
            input_filename = f"cell_combined_{split_type}_input_{i}_test.pth"
            torch.save(data_by_i[i]['input'], input_dir / input_filename)
            print(f"    Saved {input_filename}: {data_by_i[i]['input'].shape}")
        
        # Save output data for this i value
        if data_by_i[i]['output'] is not None:
            output_filename = f"cell_combined_{split_type}_output_{i}_test.pth"
            torch.save(data_by_i[i]['output'], output_dir / output_filename)
            print(f"    Saved {output_filename}: {data_by_i[i]['output'].shape}")
    
    print(f"  Saved {split_type} data files for i=0 to i={num_nodes-1}")

def main():
    # Set random seed for reproducibility
    set_random_seed(42)
    print("Random seed set to 42 for reproducible results")
    
    # Paths
    base_path = Path("../../dataset_PV/processed")  # Where to save the divided datasets
    nodewise_path = base_path  # Where the combined_input_Xnodes.pth files are located
    
    num_nodes = 49  # Number of nodes (0-48)
    
    print(f"\n=== Processing Combined Datasets ===")
    print(f"Reading from: {nodewise_path}")
    print(f"Saving to: {base_path}")
    
    # Create directory structure
    combined_path = create_dataset_directory_structure(base_path)
    
    # Load combined nodewise data
    i_nodes_data = load_combined_nodewise_data(nodewise_path, num_nodes)
    
    # Print summary
    valid_nodes = 0
    for i in range(num_nodes):
        if i_nodes_data[i]['input'] is not None and i_nodes_data[i]['output'] is not None:
            valid_nodes += 1
            print(f"  Node {i}: input {i_nodes_data[i]['input'].shape}, output {i_nodes_data[i]['output'].shape}")
    
    print(f"  Found data for {valid_nodes} nodes")
    
    if valid_nodes == 0:
        print("  No valid data found. Exiting.")
        return
    
    # Split into train/test (8:2 ratio)
    print(f"\n=== Splitting into Train/Test (8:2 ratio) ===")
    train_data_by_i, test_data_by_i = split_train_test_by_i(i_nodes_data, test_ratio=0.2, num_nodes=num_nodes)
    
    # Save train and test data
    print(f"\n=== Saving Train Data ===")
    save_data_by_i(train_data_by_i, combined_path, "train", num_nodes)
    
    print(f"\n=== Saving Test Data ===")
    save_data_by_i(test_data_by_i, combined_path, "test", num_nodes)
    
    print(f"\n✅ Combined dataset division completed!")
    print(f"Created combined train/test files for i=0 to i={num_nodes-1} (train/test split 8:2)")
    print(f"Output directory: {combined_path}")

if __name__ == "__main__":
    main()