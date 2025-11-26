#!/usr/bin/env python
"""
Unified GNN Training Pipeline
Combines all cell types for selected process condition and corner combination
12 combinations: 4 process (RVT, LVT, SLVT, SRAM) × 3 corner (TT, FF, SS)
"""

import os
import torch
from torch import optim
import torch.nn as nn
import sys
import time
import random
from torch_geometric.data import Data, Batch

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# Add paths
sys.path.append('./')
sys.path.append('tools/data_processing')

from gnn_maml_optimized import (
    create_gcn_model,
    create_graphsage_model, 
    create_gat_model
)

class UnifiedDatasetLoader:
    """
    통합 데이터셋 로더 - 여러 cell type을 하나로 합침
    """
    
    def __init__(self, process_type, corner_type, train_ratio=0.8, seed=42):
        """
        Args:
            process_type: "RVT", "LVT", "SLVT", or "SRAM"
            corner_type: "TT", "FF", or "SS"
            train_ratio: Train/test split ratio
            seed: Random seed
        """
        self.process_type = process_type
        self.corner_type = corner_type
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"🔍 Loading unified dataset: {process_type}_{corner_type}")
        
        # Find all datasets matching process and corner
        self.matching_datasets = self._find_matching_datasets()
        
        if not self.matching_datasets:
            raise ValueError(f"No datasets found for {process_type}_{corner_type}")
        
        print(f"   Found {len(self.matching_datasets)} matching datasets:")
        for dataset in self.matching_datasets:
            print(f"     {dataset}")
        
        # Load and combine all data
        self.all_samples, self.all_outputs = self._load_and_combine_data()
        
        # Split into train/test
        self.train_samples, self.test_samples, self.train_outputs, self.test_outputs = self._split_data()
        
        # Compute normalization stats from training data
        self.norm_stats = self._compute_normalization_stats()
        
        print(f"✅ Unified dataset ready:")
        print(f"   Total samples: {len(self.all_samples)}")
        print(f"   Train: {len(self.train_samples)} samples")
        print(f"   Test: {len(self.test_samples)} samples")
    
    def _find_matching_datasets(self):
        """
        Find all datasets matching process and corner
        """
        base_path = "dataset_gnn/processed_batch"
        matching = []
        
        if not os.path.exists(base_path):
            return matching
        
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if not os.path.isdir(item_path):
                continue
            
            data_file = os.path.join(item_path, "graph_data", "all_graph_data.pth")
            if not os.path.exists(data_file):
                continue
            
            # Check if dataset matches process and corner
            parts = item.split('_')
            
            # Extract process type from dataset name
            dataset_process = None
            for part in parts:
                if part in ['RVT', 'LVT', 'SLVT', 'SRAM']:
                    dataset_process = part
                    break
            
            if dataset_process != self.process_type:
                continue
            
            # Extract corner type from dataset name
            if self.corner_type == "TT":
                # TT datasets don't have _FF or _SS suffix
                if item.endswith(('_FF', '_SS')):
                    continue
            elif self.corner_type == "FF":
                if not item.endswith('_FF'):
                    continue
            elif self.corner_type == "SS":
                if not item.endswith('_SS'):
                    continue
            
            matching.append(item)
        
        return sorted(matching)  # Sort for consistent ordering
    
    def _load_and_combine_data(self):
        """
        Load and combine data from all matching datasets
        """
        print(f"📊 Loading and combining data...")
        
        all_samples = []
        all_outputs = []
        
        for dataset_name in self.matching_datasets:
            dataset_path = f"dataset_gnn/processed_batch/{dataset_name}/graph_data/all_graph_data.pth"
            
            print(f"   Loading: {dataset_name}")
            data = torch.load(dataset_path, weights_only=False)
            
            graph_data_per_file = data['graph_data_per_file']
            stacked_outputs = data['stacked_outputs']
            
            dataset_samples = 0
            
            # Flatten all samples from this dataset
            for lib_idx in range(len(graph_data_per_file)):
                lib_samples = graph_data_per_file[lib_idx]
                
                for sample_idx in range(len(lib_samples)):
                    graph_sample = lib_samples[sample_idx]
                    output_value = stacked_outputs[sample_idx, lib_idx]
                    
                    # Add dataset identifier to sample
                    graph_sample['dataset_name'] = dataset_name
                    
                    all_samples.append(graph_sample)
                    all_outputs.append(output_value.item())
                    dataset_samples += 1
            
            print(f"     Added {dataset_samples} samples")
        
        print(f"   Combined total: {len(all_samples)} samples")
        return all_samples, torch.tensor(all_outputs, dtype=torch.float32)
    
    def _split_data(self):
        """
        Split data into train/test sets
        """
        total_samples = len(self.all_samples)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_size = int(total_samples * self.train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_samples = [self.all_samples[i] for i in train_indices]
        train_outputs = self.all_outputs[train_indices]
        
        test_samples = [self.all_samples[i] for i in test_indices]
        test_outputs = self.all_outputs[test_indices]
        
        return train_samples, test_samples, train_outputs, test_outputs
    
    def _compute_normalization_stats(self, sample_size=2000):
        """
        Compute normalization statistics from training data
        """
        print(f"📊 Computing normalization statistics from {min(sample_size, len(self.train_samples))} training samples...")
        
        all_voltages = []
        all_input_slews = []
        all_output_loads = []
        
        num_samples = min(sample_size, len(self.train_samples))
        sample_indices = torch.randperm(len(self.train_samples))[:num_samples]
        
        for idx in sample_indices:
            graph = self.train_samples[idx]
            node_features = graph['node_features']
            
            # Extract features from all nodes
            # Feature structure: [is_power, is_circuit, trans_type, width, voltage, input_slew, output_load]
            voltages = node_features[:, 4]
            input_slews = node_features[:, 5]
            output_loads = node_features[:, 6]
            
            all_voltages.extend(voltages[voltages != 0].tolist())
            all_input_slews.extend(input_slews[input_slews != 0].tolist())
            all_output_loads.extend(output_loads[output_loads != 0].tolist())
        
        # Compute statistics
        stats = {
            'voltage': {
                'mean': torch.tensor(all_voltages).mean().item() if all_voltages else 0.7,
                'std': torch.tensor(all_voltages).std().item() if all_voltages else 0.1
            },
            'input_slew': {
                'mean': torch.tensor(all_input_slews).mean().item() if all_input_slews else 40.0,
                'std': torch.tensor(all_input_slews).std().item() if all_input_slews else 20.0
            },
            'output_load': {
                'mean': torch.tensor(all_output_loads).mean().item() if all_output_loads else 5.0,
                'std': torch.tensor(all_output_loads).std().item() if all_output_loads else 3.0
            }
        }
        
        # Avoid division by zero
        for feature in stats:
            if stats[feature]['std'] < 1e-8:
                stats[feature]['std'] = 1.0
        
        print(f"   Voltage: mean={stats['voltage']['mean']:.4f}, std={stats['voltage']['std']:.4f}")
        print(f"   Input slew: mean={stats['input_slew']['mean']:.4f}, std={stats['input_slew']['std']:.4f}")
        print(f"   Output load: mean={stats['output_load']['mean']:.4f}, std={stats['output_load']['std']:.4f}")
        
        return stats
    
    def get_train_batch(self, batch_size=32):
        """
        Get a batch of training data
        """
        batch_indices = torch.randperm(len(self.train_samples))[:batch_size]
        batch_samples = [self.train_samples[i] for i in batch_indices]
        batch_outputs = self.train_outputs[batch_indices]
        
        return batch_samples, batch_outputs
    
    def get_test_batch(self, batch_size=32, start_idx=0):
        """
        Get a batch of test data
        """
        end_idx = min(start_idx + batch_size, len(self.test_samples))
        batch_samples = self.test_samples[start_idx:end_idx]
        batch_outputs = self.test_outputs[start_idx:end_idx]
        
        return batch_samples, batch_outputs

def normalize_node_features(node_features, norm_stats):
    """
    Normalize node features using computed statistics
    """
    normalized = node_features.clone()
    
    # Normalize voltage (column 4)
    voltage_mask = normalized[:, 4] != 0
    normalized[voltage_mask, 4] = (normalized[voltage_mask, 4] - norm_stats['voltage']['mean']) / norm_stats['voltage']['std']
    
    # Normalize input_slew (column 5)
    slew_mask = normalized[:, 5] != 0
    normalized[slew_mask, 5] = (normalized[slew_mask, 5] - norm_stats['input_slew']['mean']) / norm_stats['input_slew']['std']
    
    # Normalize output_load (column 6)
    load_mask = normalized[:, 6] != 0
    normalized[load_mask, 6] = (normalized[load_mask, 6] - norm_stats['output_load']['mean']) / norm_stats['output_load']['std']
    
    return normalized

def create_pyg_data_with_adj_matrix(graph_sample, norm_stats):
    """
    Create PyTorch Geometric Data object with adjacency matrix multiplication and normalization
    """
    # Extract components
    node_features = graph_sample['node_features']
    adjacency_matrix = graph_sample['adjacency_matrix']
    edge_index = graph_sample['edge_index']
    
    # Apply normalization
    normalized_features = normalize_node_features(node_features, norm_stats)
    
    # Apply adjacency matrix multiplication (A × X)
    aggregated_features = torch.matmul(adjacency_matrix, normalized_features)
    
    # Create PyG Data object
    data = Data(
        x=aggregated_features,
        edge_index=edge_index,
        dataset_name=graph_sample.get('dataset_name', 'unknown'),
        cell_name=graph_sample.get('cell_name', 'unknown')
    )
    
    return data

def train_epoch(model, dataset_loader, optimizer, criterion, batch_size=32):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Calculate number of batches
    total_batches = len(dataset_loader.train_samples) // batch_size
    
    for batch_idx in range(total_batches):
        batch_samples, batch_outputs = dataset_loader.get_train_batch(batch_size)
        
        # Convert to PyG Data objects with normalization and A×X
        batch_data = []
        for graph in batch_samples:
            data = create_pyg_data_with_adj_matrix(graph, dataset_loader.norm_stats)
            batch_data.append(data)
        
        # Create batch
        batch = Batch.from_data_list(batch_data).to(device)
        targets = batch_outputs.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs.squeeze(), targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"   Batch {batch_idx}/{total_batches}: Loss = {loss.item():.6f}")
    
    return total_loss / num_batches

# evaluate function removed - will be done separately using saved norm_stats

def select_process_corner_combination():
    """
    Interactive selection of process and corner combination
    """
    print("🎯 Available Process-Corner Combinations:")
    print("=" * 50)
    
    processes = ["RVT", "LVT", "SLVT", "SRAM"]
    corners = ["TT", "FF", "SS"]
    
    combinations = []
    for i, process in enumerate(processes):
        for j, corner in enumerate(corners):
            combo_idx = i * 3 + j + 1
            combination = f"{process}_{corner}"
            combinations.append((process, corner))
            print(f"{combo_idx:2d}. {combination:10s} ({process} process, {corner} corner)")
    
    print(f"\n📊 Total: {len(combinations)} combinations available")
    
    while True:
        try:
            choice = input(f"\nEnter combination number (1-{len(combinations)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(combinations):
                selected_process, selected_corner = combinations[choice_idx]
                print(f"✅ Selected: {selected_process}_{selected_corner}")
                return selected_process, selected_corner
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(combinations)}")
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Selection cancelled")
            return None, None

def main():
    """
    Main training function
    """
    print("🚀 Unified GNN Training Pipeline")
    print("12 Process-Corner Combinations")
    print("=" * 60)
    
    # Interactive selection
    process_type, corner_type = select_process_corner_combination()
    
    if process_type is None or corner_type is None:
        print("No combination selected. Exiting...")
        return
    
    # Configuration
    num_epochs = 50
    learning_rate = 0.001
    hidden_dim = 40
    gnn_type = "GCN"
    batch_size = 32
    
    print(f"\n📋 Training Configuration:")
    print(f"   Process-Corner: {process_type}_{corner_type}")
    print(f"   Model: {gnn_type}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")
    
    # Load unified dataset
    try:
        dataset_loader = UnifiedDatasetLoader(process_type, corner_type, train_ratio=0.8)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Create model
    node_features = 7  # Stage-aware features
    
    if gnn_type == "GCN":
        model = create_gcn_model(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=3,
            pooling='mean'
        ).to(device)
    elif gnn_type == "GraphSAGE":
        model = create_graphsage_model(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=3,
            pooling='mean'
        ).to(device)
    elif gnn_type == "GAT":
        model = create_gat_model(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=3,
            heads=4,
            pooling='mean'
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model: {total_params:,} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop (training only - evaluation done separately)
    print(f"\n🎯 Starting training (evaluation will be done separately)...")
    print("=" * 60)
    
    train_losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train only
        train_loss = train_epoch(model, dataset_loader, optimizer, criterion, batch_size)
        train_losses.append(train_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss = {train_loss:.6f}, "
              f"Time = {epoch_time:.2f}s")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"pretrained_models/unified/{process_type}_{corner_type}_{gnn_type}_epoch{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'norm_stats': dataset_loader.norm_stats,
                'config': {
                    'process_type': process_type,
                    'corner_type': corner_type,
                    'gnn_type': gnn_type,
                    'hidden_dim': hidden_dim,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'node_features': node_features,
                    'combined_datasets': dataset_loader.matching_datasets,
                    'normalized': True
                }
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved at epoch {epoch+1}")
    
    total_time = time.time() - start_time
    
    # Final model save
    final_path = f"pretrained_models/unified/{process_type}_{corner_type}_{gnn_type}_final.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'norm_stats': dataset_loader.norm_stats,
        'config': {
            'process_type': process_type,
            'corner_type': corner_type,
            'gnn_type': gnn_type,
            'hidden_dim': hidden_dim,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'node_features': node_features,
            'combined_datasets': dataset_loader.matching_datasets,
            'normalized': True
        }
    }, final_path)
    
    print(f"\n🎉 Training Complete!")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Final train loss: {train_losses[-1]:.6f}")
    print(f"   Combined datasets: {dataset_loader.matching_datasets}")
    print(f"   Final model: {final_path}")
    print(f"   📊 Norm stats saved in model - use for evaluation!")
    
    # Save normalization stats separately for easy access
    import json
    norm_path = final_path.replace('.pth', '_norm_stats.json')
    with open(norm_path, 'w') as f:
        json.dump(dataset_loader.norm_stats, f, indent=2)
    print(f"   Normalization stats JSON: {norm_path}")
    
    print(f"\n📖 Next Steps:")
    print(f"   1. For evaluation: Use saved norm_stats from {final_path}")
    print(f"   2. Test data available: {len(dataset_loader.test_samples)} samples") 
    print(f"   3. Use separate evaluation script with saved model + norm_stats")

if __name__ == "__main__":
    main()