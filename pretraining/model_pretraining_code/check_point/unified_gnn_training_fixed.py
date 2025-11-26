#!/usr/bin/env python
"""
Fixed Unified GNN Training Pipeline with Output Scaling
Addresses high loss issues by scaling outputs and adjusting training parameters
"""

import os
import torch
from torch import optim
import torch.nn as nn
import sys
import time
import random
from torch_geometric.data import Data, Batch
import numpy as np

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
    통합 데이터셋 로더 - 여러 cell type을 하나로 합침 (High Loss Fixed Version)
    """
    
    def __init__(self, process_type, corner_type, train_ratio=0.8, seed=42, output_scale_factor=1000.0):
        """
        Args:
            process_type: "RVT", "LVT", "SLVT", or "SRAM"
            corner_type: "TT", "FF", or "SS"
            train_ratio: Train/test split ratio
            seed: Random seed
            output_scale_factor: Scale down outputs to prevent high loss (default: 1000.0)
        """
        self.process_type = process_type
        self.corner_type = corner_type
        self.train_ratio = train_ratio
        self.seed = seed
        self.output_scale_factor = output_scale_factor
        
        # Set random seeds
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        self.base_path = "dataset_gnn/processed_batch"
        self.matching_datasets = self._find_matching_datasets()
        
        print(f"🎯 Unified Dataset: {process_type}_{corner_type}")
        print(f"   Found datasets: {self.matching_datasets}")
        print(f"   Output scaling: ÷{output_scale_factor}")
        
        if not self.matching_datasets:
            raise ValueError(f"No datasets found for {process_type}_{corner_type}")
    
    def _find_matching_datasets(self):
        """
        Find all datasets matching the process-corner combination
        """
        matching = []
        
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if not os.path.isdir(item_path):
                continue
            
            data_file = os.path.join(item_path, "graph_data", "all_graph_data.pth")
            if not os.path.exists(data_file):
                continue
            
            # Check if dataset matches process and corner
            parts = item.split('_')
            
            dataset_process = None
            for part in parts:
                if part in ['RVT', 'LVT', 'SLVT', 'SRAM']:
                    dataset_process = part
                    break
            
            if dataset_process != self.process_type:
                continue
            
            if self.corner_type == "TT":
                if item.endswith(('_FF', '_SS')):
                    continue
            elif self.corner_type == "FF":
                if not item.endswith('_FF'):
                    continue
            elif self.corner_type == "SS":
                if not item.endswith('_SS'):
                    continue
            
            matching.append(item)
        
        return matching
    
    def load_combined_data(self):
        """
        Load and combine all matching datasets with improved memory management
        """
        print(f"\n📊 Loading combined data...")
        
        all_train_samples = []
        all_train_outputs = []
        all_test_samples = []
        all_test_outputs = []
        
        # Track normalization stats across all datasets
        all_voltage_values = []
        all_slew_values = []
        all_load_values = []
        
        for dataset_name in sorted(self.matching_datasets):
            dataset_path = f"{self.base_path}/{dataset_name}/graph_data/all_graph_data.pth"
            print(f"   📂 Processing: {dataset_name}")
            
            # Load dataset
            data = torch.load(dataset_path, weights_only=False, map_location='cpu')
            graph_data_per_file = data['graph_data_per_file']
            stacked_outputs = data['stacked_outputs']
            
            # Flatten all samples from this dataset
            dataset_samples = []
            dataset_outputs = []
            
            for lib_idx in range(len(graph_data_per_file)):
                lib_samples = graph_data_per_file[lib_idx]
                for sample_idx in range(len(lib_samples)):
                    graph_sample = lib_samples[sample_idx]
                    output_value = stacked_outputs[sample_idx, lib_idx]
                    
                    # Add dataset name for tracking
                    graph_sample['dataset_name'] = dataset_name
                    
                    dataset_samples.append(graph_sample)
                    dataset_outputs.append(output_value.item())
            
            print(f"      Samples: {len(dataset_samples)}")
            
            # Check output range and apply scaling
            outputs_array = np.array(dataset_outputs)
            print(f"      Output range: [{outputs_array.min():.2f}, {outputs_array.max():.2f}]")
            
            if outputs_array.max() > 1000:
                print(f"      ⚠️ Large outputs detected - applying scaling (÷{self.output_scale_factor})")
                dataset_outputs = [x / self.output_scale_factor for x in dataset_outputs]
            
            # Split dataset
            total_samples = len(dataset_samples)
            indices = list(range(total_samples))
            random.shuffle(indices)
            
            train_size = int(total_samples * self.train_ratio)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            # Split samples
            train_samples = [dataset_samples[i] for i in train_indices]
            train_outputs = [dataset_outputs[i] for i in train_indices]
            test_samples = [dataset_samples[i] for i in test_indices]
            test_outputs = [dataset_outputs[i] for i in test_indices]
            
            # Collect normalization statistics from training data only
            for i in train_indices:
                sample = dataset_samples[i]
                node_features = sample['node_features']
                
                # Extract features for normalization
                voltage_values = node_features[:, 4]
                slew_values = node_features[:, 5]
                load_values = node_features[:, 6]
                
                # Only non-zero values
                all_voltage_values.extend(voltage_values[voltage_values != 0].tolist())
                all_slew_values.extend(slew_values[slew_values != 0].tolist())
                all_load_values.extend(load_values[load_values != 0].tolist())
            
            all_train_samples.extend(train_samples)
            all_train_outputs.extend(train_outputs)
            all_test_samples.extend(test_samples)
            all_test_outputs.extend(test_outputs)
            
            print(f"      Train: {len(train_samples)}, Test: {len(test_samples)}")
        
        # Calculate normalization statistics
        norm_stats = self._calculate_norm_stats(
            all_voltage_values, all_slew_values, all_load_values
        )
        
        print(f"\n📈 Final Combined Dataset:")
        print(f"   Total train: {len(all_train_samples)}")
        print(f"   Total test: {len(all_test_samples)}")
        print(f"   Output scaling: ÷{self.output_scale_factor}")
        
        return (
            all_train_samples, torch.tensor(all_train_outputs, dtype=torch.float32),
            all_test_samples, torch.tensor(all_test_outputs, dtype=torch.float32),
            norm_stats
        )
    
    def _calculate_norm_stats(self, voltage_values, slew_values, load_values):
        """
        Calculate normalization statistics with improved handling
        """
        def safe_stats(values, name):
            if len(values) == 0:
                print(f"   ⚠️ No {name} values found")
                return {'mean': 0.0, 'std': 1.0}
            
            values_tensor = torch.tensor(values, dtype=torch.float32)
            mean_val = values_tensor.mean().item()
            std_val = values_tensor.std().item()
            
            # Handle very small std
            if std_val < 1e-8:
                print(f"   ⚠️ {name} std too small ({std_val:.2e}), using 1.0")
                std_val = 1.0
            
            print(f"   {name}: mean={mean_val:.4f}, std={std_val:.4f}, count={len(values)}")
            return {'mean': mean_val, 'std': std_val}
        
        print(f"\n📊 Normalization Statistics:")
        norm_stats = {
            'voltage': safe_stats(voltage_values, 'Voltage'),
            'input_slew': safe_stats(slew_values, 'Input Slew'),
            'output_load': safe_stats(load_values, 'Output Load')
        }
        
        # Add scaling factor to stats
        norm_stats['output_scale_factor'] = self.output_scale_factor
        
        return norm_stats

def normalize_node_features(node_features, norm_stats):
    """
    Normalize node features using provided statistics
    """
    normalized = node_features.clone()
    
    # Normalize voltage (column 4)
    voltage_mask = normalized[:, 4] != 0
    if voltage_mask.any():
        normalized[voltage_mask, 4] = (
            normalized[voltage_mask, 4] - norm_stats['voltage']['mean']
        ) / norm_stats['voltage']['std']
    
    # Normalize input_slew (column 5)
    slew_mask = normalized[:, 5] != 0
    if slew_mask.any():
        normalized[slew_mask, 5] = (
            normalized[slew_mask, 5] - norm_stats['input_slew']['mean']
        ) / norm_stats['input_slew']['std']
    
    # Normalize output_load (column 6)
    load_mask = normalized[:, 6] != 0
    if load_mask.any():
        normalized[load_mask, 6] = (
            normalized[load_mask, 6] - norm_stats['output_load']['mean']
        ) / norm_stats['output_load']['std']
    
    return normalized

def create_pyg_data_with_adj_matrix(graph_sample, norm_stats):
    """
    Create PyTorch Geometric Data object with normalization and adjacency matrix multiplication
    """
    node_features = graph_sample['node_features']
    adjacency_matrix = graph_sample['adjacency_matrix']
    edge_index = graph_sample['edge_index']
    
    # Apply normalization
    normalized_features = normalize_node_features(node_features, norm_stats)
    
    # Apply adjacency matrix multiplication (A × X)
    aggregated_features = torch.matmul(adjacency_matrix, normalized_features)
    
    data = Data(
        x=aggregated_features,
        edge_index=edge_index,
        dataset_name=graph_sample.get('dataset_name', 'unknown'),
        cell_name=graph_sample.get('cell_name', 'unknown')
    )
    
    return data

def train_unified_gnn(process_type, corner_type, gnn_type="GCN", 
                     hidden_dim=40, num_epochs=100, batch_size=64, 
                     learning_rate=1e-4, output_scale_factor=1000.0):  # Lower LR
    """
    Train GNN with improved loss handling
    """
    print(f"\n🚀 Training Unified GNN: {process_type}_{corner_type}_{gnn_type}")
    print("=" * 60)
    
    # Load data with output scaling
    loader = UnifiedDatasetLoader(
        process_type=process_type, 
        corner_type=corner_type,
        output_scale_factor=output_scale_factor
    )
    train_samples, train_outputs, test_samples, test_outputs, norm_stats = loader.load_combined_data()
    
    # Create model
    node_features = 7
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
    
    # Training setup with improvements
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Model: {gnn_type}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Output Scaling: ÷{output_scale_factor}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch Size: {batch_size}")
    
    # Training loop with gradient clipping
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle training data
        indices = torch.randperm(len(train_samples))
        
        for batch_start in range(0, len(train_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(train_samples))
            batch_indices = indices[batch_start:batch_end]
            
            # Create batch
            batch_data = []
            batch_targets = []
            
            for idx in batch_indices:
                graph = train_samples[idx.item()]
                target = train_outputs[idx.item()]
                
                data = create_pyg_data_with_adj_matrix(graph, norm_stats)
                batch_data.append(data)
                batch_targets.append(target)
            
            batch = Batch.from_data_list(batch_data).to(device)
            targets = torch.tensor(batch_targets, dtype=torch.float32).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch < 5:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.2e}")
        
        if patience_counter >= max_patience:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    # Save model with configuration and normalization stats
    model_dir = "pretrained_models/unified_fixed"
    os.makedirs(model_dir, exist_ok=True)
    
    model_filename = f"{process_type}_{corner_type}_{gnn_type}_fixed_final.pth"
    model_path = os.path.join(model_dir, model_filename)
    
    # Save comprehensive checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_stats': norm_stats,
        'config': {
            'process_type': process_type,
            'corner_type': corner_type,
            'gnn_type': gnn_type,
            'hidden_dim': hidden_dim,
            'node_features': node_features,
            'combined_datasets': loader.matching_datasets,
            'output_scale_factor': output_scale_factor,
            'learning_rate': learning_rate,
            'final_loss': best_loss
        },
        'train_info': {
            'num_train_samples': len(train_samples),
            'num_test_samples': len(test_samples),
            'epochs_trained': epoch + 1,
            'best_loss': best_loss
        }
    }, model_path)
    
    print(f"\n✅ Training Complete!")
    print(f"   Final Loss: {best_loss:.6f}")
    print(f"   Model saved: {model_path}")
    print(f"   Normalization stats included for evaluation")
    print(f"   Output scaling factor: {output_scale_factor}")

def main():
    """
    Main training function with interactive selection
    """
    print("🚀 Fixed Unified GNN Training")
    print("=" * 60)
    
    # Process and corner options
    process_options = ["RVT", "LVT", "SLVT", "SRAM"]
    corner_options = ["TT", "FF", "SS"]
    gnn_options = ["GCN", "GraphSAGE", "GAT"]
    
    print(f"📊 Available combinations:")
    combination_id = 1
    combinations = []
    
    for process in process_options:
        for corner in corner_options:
            print(f"   {combination_id:2d}. {process}_{corner}")
            combinations.append((process, corner))
            combination_id += 1
    
    # Interactive selection
    while True:
        try:
            choice = input(f"\nSelect combination (1-{len(combinations)}): ")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(combinations):
                process_type, corner_type = combinations[choice_idx]
                break
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(combinations)}")
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Training cancelled")
            return
    
    # GNN architecture selection
    print(f"\n🏗️ Select GNN Architecture:")
    for i, gnn in enumerate(gnn_options, 1):
        print(f"   {i}. {gnn}")
    
    while True:
        try:
            gnn_choice = input(f"Select architecture (1-{len(gnn_options)}): ")
            gnn_idx = int(gnn_choice) - 1
            
            if 0 <= gnn_idx < len(gnn_options):
                gnn_type = gnn_options[gnn_idx]
                break
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(gnn_options)}")
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Training cancelled")
            return
    
    # Output scaling selection
    print(f"\n⚖️ Select Output Scaling (to fix high loss):")
    print(f"   1. Scale ÷1000 (recommended for timing data)")
    print(f"   2. Scale ÷10000 (for very large values)")
    print(f"   3. No scaling")
    
    scale_factors = [1000.0, 10000.0, 1.0]
    
    while True:
        try:
            scale_choice = input("Select scaling (1-3): ")
            scale_idx = int(scale_choice) - 1
            
            if 0 <= scale_idx < len(scale_factors):
                output_scale_factor = scale_factors[scale_idx]
                break
            else:
                print("❌ Invalid choice. Please enter 1-3")
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Training cancelled")
            return
    
    # Start training
    train_unified_gnn(
        process_type=process_type,
        corner_type=corner_type,
        gnn_type=gnn_type,
        output_scale_factor=output_scale_factor,
        learning_rate=1e-4,  # Lower learning rate
        num_epochs=100,
        batch_size=64
    )

if __name__ == "__main__":
    main()