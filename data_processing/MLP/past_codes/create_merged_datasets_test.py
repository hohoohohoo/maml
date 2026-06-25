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

def merge_nodewise_data_by_i(nodewise_path, vt_type, cell_types=['AO', 'OA', 'INVBUF', 'simple']):
    """Merge nodewise data by i value (0-49) for LVT or RVT"""
    
    # Data containers organized by i value
    i_nodes_data = {}
    
    print(f"Processing {vt_type} data by i nodes...")
    
    # Initialize containers for each i value (0-49)
    for i in range(50):
        i_nodes_data[i] = {
            'cell_input': [],
            'cell_output': [],
            'transition_input': [],
            'transition_output': []
        }
    
    for cell_type in cell_types:
        print(f"  Processing {cell_type}...")
        
        # Find files for this cell type and VT type
        for i in range(1, 50):  # 1부터 49까지 (파일명에서는 1-49)
            # Cell data files 
            cell_input_file = nodewise_path / f"cell_{cell_type}_{vt_type}_2_25_input_{i}nodes_align.pth"
            cell_output_file = nodewise_path / f"cell_{cell_type}_{vt_type}_2_25_output_{i}nodes_align.pth"
            
            # Transition data files
            transition_input_file = nodewise_path / f"transition_{cell_type}_{vt_type}_2_25_input_{i}nodes_align.pth"
            transition_output_file = nodewise_path / f"transition_{cell_type}_{vt_type}_2_25_output_{i}nodes_align.pth"
            
            # Load cell data if exists
            if cell_input_file.exists() and cell_output_file.exists():
                try:
                    cell_input = torch.load(cell_input_file)
                    cell_output = torch.load(cell_output_file)
                    i_nodes_data[i]['cell_input'].append(cell_input)
                    i_nodes_data[i]['cell_output'].append(cell_output)
                except Exception as e:
                    print(f"    Error loading cell {cell_type} {i}nodes: {e}")
            
            # Load transition data if exists  
            if transition_input_file.exists() and transition_output_file.exists():
                try:
                    transition_input = torch.load(transition_input_file)
                    transition_output = torch.load(transition_output_file)
                    i_nodes_data[i]['transition_input'].append(transition_input)
                    i_nodes_data[i]['transition_output'].append(transition_output)
                except Exception as e:
                    print(f"    Error loading transition {cell_type} {i}nodes: {e}")
    
    return i_nodes_data

def split_train_test_by_i(i_nodes_data, test_ratio=0.2):
    """Split data into train and test sets for each i value - stratified split by input component values 1,2,3"""
    train_data_by_i = {}
    test_data_by_i = {}
    
    for i in range(49):  # Changed to 0-48
        train_data_by_i[i] = {
            'cell_input': [],
            'cell_output': [],
            'transition_input': [],
            'transition_output': []
        }
        test_data_by_i[i] = {
            'cell_input': [],
            'cell_output': [],
            'transition_input': [],
            'transition_output': []
        }
        
        # Process input and output data together to maintain correspondence
        input_types = [('cell_input', 'cell_output'), ('transition_input', 'transition_output')]
        
        for input_type, output_type in input_types:
            input_data_list = i_nodes_data[i+1][input_type]  # i+1 because original data uses 1-49
            output_data_list = i_nodes_data[i+1][output_type]
            
            if input_data_list and output_data_list:
                # Merge all tensors into one
                merged_input = torch.cat(input_data_list, dim=0)
                merged_output = torch.cat(output_data_list, dim=0)
                
                # Check if input has at least 3 dimensions to access second feature
                if merged_input.dim() >= 3 and merged_input.shape[2] >= 2:
                    # Use second feature (index 1) for stratification
                    component_values = merged_input[:, 0, 1]  # Second feature of first time step
                    unique_values = torch.unique(component_values)
                    
                    print(f"    i={i} {input_type} Data shape: {merged_input.shape}")
                    print(f"    i={i} {input_type} Unique second feature values: {unique_values.tolist()}")
                    print(f"    i={i} {input_type} Sample of second feature values: {component_values[:10].tolist()}")
                    
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
                            
                            print(f"    i={i} {input_type} Second feature {component_val.item()}: "
                                  f"Total={len(indices)}, Train={len(component_train_indices)}, Test={len(component_test_indices)}")
                    
                    # Convert to tensor
                    all_train_indices = torch.tensor(sorted(all_train_indices))
                    all_test_indices = torch.tensor(sorted(all_test_indices))
                    
                    # Verify no overlap between train and test
                    overlap = set(all_train_indices.tolist()) & set(all_test_indices.tolist())
                    if overlap:
                        print(f"    ERROR: Found {len(overlap)} overlapping indices - this should not happen!")
                        # Just split normally if stratification fails
                        total_samples = merged_input.shape[0]
                        test_size = int(total_samples * test_ratio)
                        
                        if test_size > 0:
                            test_data_by_i[i][input_type] = [merged_input[-test_size:]]
                            test_data_by_i[i][output_type] = [merged_output[-test_size:]]
                            train_data_by_i[i][input_type] = [merged_input[:-test_size]]
                            train_data_by_i[i][output_type] = [merged_output[:-test_size]]
                        else:
                            train_data_by_i[i][input_type] = [merged_input]
                            train_data_by_i[i][output_type] = [merged_output]
                        
                        print(f"    i={i} {input_type} Fallback split: Train={total_samples-test_size}, Test={test_size}")
                        continue
                    
                    # Split data using stratified indices
                    if len(all_train_indices) > 0:
                        train_data_by_i[i][input_type] = [merged_input[all_train_indices]]
                        train_data_by_i[i][output_type] = [merged_output[all_train_indices]]
                    
                    if len(all_test_indices) > 0:
                        test_data_by_i[i][input_type] = [merged_input[all_test_indices]]
                        test_data_by_i[i][output_type] = [merged_output[all_test_indices]]
                        
                    print(f"    i={i} {input_type} Final: Train={len(all_train_indices)}, Test={len(all_test_indices)}")
                
                else:
                    # Fallback to simple split if input doesn't have expected dimensions
                    total_samples = merged_input.shape[0]
                    test_size = int(total_samples * test_ratio)
                    
                    if test_size > 0:
                        test_data_by_i[i][input_type] = [merged_input[-test_size:]]
                        test_data_by_i[i][output_type] = [merged_output[-test_size:]]
                        train_data_by_i[i][input_type] = [merged_input[:-test_size]]
                        train_data_by_i[i][output_type] = [merged_output[:-test_size]]
                    else:
                        train_data_by_i[i][input_type] = [merged_input]
                        train_data_by_i[i][output_type] = [merged_output]
    
    return train_data_by_i, test_data_by_i

def save_data_by_i(data_by_i, base_path, split_type):
    """Save data organized by i value (0-48), separating cell and transition data"""
    input_dir = base_path / f"{split_type}datainput"
    output_dir = base_path / f"{split_type}dataoutput"
    
    for i in range(49):  # Changed to 0-48
        # Save cell data for this i value
        if data_by_i[i]['cell_input']:
            merged_cell_input = torch.cat(data_by_i[i]['cell_input'], dim=0) if len(data_by_i[i]['cell_input']) > 1 else data_by_i[i]['cell_input'][0]
            cell_input_filename = f"cell_{split_type}_input_{i}.pth"
            torch.save(merged_cell_input, input_dir / cell_input_filename)
        
        if data_by_i[i]['cell_output']:
            merged_cell_output = torch.cat(data_by_i[i]['cell_output'], dim=0) if len(data_by_i[i]['cell_output']) > 1 else data_by_i[i]['cell_output'][0]
            cell_output_filename = f"cell_{split_type}_output_{i}.pth"
            torch.save(merged_cell_output, output_dir / cell_output_filename)
        
        # Save transition data for this i value
        if data_by_i[i]['transition_input']:
            merged_transition_input = torch.cat(data_by_i[i]['transition_input'], dim=0) if len(data_by_i[i]['transition_input']) > 1 else data_by_i[i]['transition_input'][0]
            transition_input_filename = f"transition_{split_type}_input_{i}.pth"
            torch.save(merged_transition_input, input_dir / transition_input_filename)
        
        if data_by_i[i]['transition_output']:
            merged_transition_output = torch.cat(data_by_i[i]['transition_output'], dim=0) if len(data_by_i[i]['transition_output']) > 1 else data_by_i[i]['transition_output'][0]
            transition_output_filename = f"transition_{split_type}_output_{i}.pth"
            torch.save(merged_transition_output, output_dir / transition_output_filename)
    
    print(f"  Saved {split_type} cell and transition data files for i=0 to i=48")

def main():
    # Set random seed for reproducibility
    set_random_seed(42)
    print("Random seed set to 42 for reproducible results")
    
    # Paths
    base_path = Path("/mnt/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_test3(fixed)")
    nodewise_path = base_path / "processed"
    
    # Cell types to merge
    cell_types = ['AO', 'OA', 'INVBUF', 'simple']
    vt_types = ['LVT', 'RVT', 'SRAM', 'SLVT']
    
    for vt_type in vt_types:
        print(f"\n=== Processing {vt_type} ===")
        
        # Create directory structure
        vt_path = create_dataset_directory_structure(base_path, vt_type)
        
        # Merge nodewise data by i value
        i_nodes_data = merge_nodewise_data_by_i(nodewise_path, vt_type, cell_types)
        
        # Print summary
        total_files = 0
        for i in range(50):
            i_total = (len(i_nodes_data[i]['cell_input']) + len(i_nodes_data[i]['cell_output']) + 
                      len(i_nodes_data[i]['transition_input']) + len(i_nodes_data[i]['transition_output']))
            if i_total > 0:
                total_files += 1
                print(f"  i={i}: {len(i_nodes_data[i]['cell_input'])} cell_in, {len(i_nodes_data[i]['cell_output'])} cell_out, "
                      f"{len(i_nodes_data[i]['transition_input'])} trans_in, {len(i_nodes_data[i]['transition_output'])} trans_out")
        
        print(f"  Found data for {total_files} different i values")
        
        # Split into train/test (8:2 ratio)
        train_data_by_i, test_data_by_i = split_train_test_by_i(i_nodes_data, test_ratio=0.2)
        
        # Save train and test data
        save_data_by_i(train_data_by_i, vt_path, "train")
        save_data_by_i(test_data_by_i, vt_path, "test")
        
        print(f"  {vt_type} dataset creation completed!")
        print(f"  Created cell and transition files for i=0 to i=48 (train/test split 8:2)")

if __name__ == "__main__":
    main()