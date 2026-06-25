#!/usr/bin/env python3
"""
Add adjacency matrix to existing GNN dataset
"""

import torch
from pathlib import Path

def add_adjacency_matrix_to_dataset(input_file, output_file):
    """
    Add adjacency matrix to existing GNN dataset samples
    """
    print(f"📥 Loading dataset from: {input_file}")
    data = torch.load(input_file)
    
    print(f"   Found {len(data)} samples")
    
    # Process each sample
    updated_data = []
    
    for i, sample in enumerate(data):
        if i % 1000 == 0:
            print(f"   Processing sample {i}/{len(data)}")
            
        # Create adjacency matrix based on existing edge_index
        num_nodes = sample['node_features'].shape[0]
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        
        edge_index = sample['edge_index']
        if edge_index.shape[1] > 0:
            # Fill adjacency matrix from edge_index
            for j in range(edge_index.shape[1]):
                src = edge_index[0, j].item()
                dst = edge_index[1, j].item()
                adjacency_matrix[src, dst] = 1.0
        else:
            # If no edges, create identity matrix (self-loops for all nodes)
            adjacency_matrix = torch.eye(num_nodes, dtype=torch.float32)
        
        # Add adjacency matrix to sample
        updated_sample = sample.copy()
        updated_sample['adjacency_matrix'] = adjacency_matrix
        updated_data.append(updated_sample)
    
    # Save updated dataset
    print(f"💾 Saving updated dataset to: {output_file}")
    torch.save(updated_data, output_file)
    print(f"✅ Successfully added adjacency matrices to {len(updated_data)} samples")
    
    return updated_data

def main():
    # Add adjacency matrix to existing datasets
    base_dir = Path("../../dataset_gnn/processed/graph_data")
    
    files_to_update = [
        ("transition_graph_input.pth", "transition_graph_input_with_adj.pth"),
        ("transition_test_graph_input.pth", "transition_test_graph_input_with_adj.pth")
    ]
    
    for input_file, output_file in files_to_update:
        input_path = base_dir / input_file
        output_path = base_dir / output_file
        
        if input_path.exists():
            print(f"\n🔄 Processing {input_file}...")
            add_adjacency_matrix_to_dataset(input_path, output_path)
        else:
            print(f"⚠️ File not found: {input_path}")
    
    print("\n🎉 All datasets updated with adjacency matrices!")

if __name__ == "__main__":
    main()