#!/usr/bin/env python
"""
MAML GNN Training with OptimizedMAML structure - FIXED VERSION
Using the same meta-loop structure as datacheck_SLVT_test5_FF.py

Key Features:
- Per-task output normalization for consistent learning across different scales
- K=5 support set sampling as requested
- OptimizedMAML structure with chunked training
- Inner loop adaptation with fast weights
- Parallel/Sequential fallback processing
"""

import os
import torch
from torch import optim
import torch.nn as nn
import sys
import random
from torch_geometric.data import Data, Batch
import numpy as np
import time
import argparse

# GPU 최적화 설정 (GPU device will be set in main() from args)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name())
else:
    print('❌ CUDA is not available!')

# Add paths
sys.path.append('./')
sys.path.append('tools/data_processing')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model_code'))

from gnn_maml import (
    MAML_GNN_Model,
    create_maml_gcn_model
)

def functional_forward(model, x, fast_weights):
    """Perform forward pass with fast weights using functional approach"""
    # Simple approach: temporarily replace model parameters
    original_params = []
    param_iter = iter(fast_weights)
    
    # Store original parameters and apply fast weights
    for param in model.parameters():
        original_params.append(param.data.clone())
        try:
            fast_weight = next(param_iter)
            if fast_weight is not None:
                param.data = fast_weight
        except StopIteration:
            break
    
    # Forward pass
    output = model(x)
    
    # Restore original parameters
    for param, original in zip(model.parameters(), original_params):
        param.data = original
    
    return output

class GNNOptimizedMAML:
    """
    GNN version of OptimizedMAML - same structure as maml_optimized.py
    """
    
    def __init__(self, model, inner_lr, meta_lr, K=5, inner_steps=1, 
                 graph_data_per_file=None, stacked_outputs=None, norm_stats=None,
                 tasks_per_meta_batch=16):
        
        # Dataset (correct structure)
        self.graph_data_per_file = graph_data_per_file  # [lib_files][samples]
        self.stacked_outputs = stacked_outputs          # [samples, lib_files]
        self.norm_stats = norm_stats
        
        self.num_tasks = stacked_outputs.shape[0] if stacked_outputs is not None else 0      # Number of input conditions
        self.lib_files_per_task = stacked_outputs.shape[1] if stacked_outputs is not None else 0  # Number of process variations
        
        # Pre-normalize all task outputs
        self.task_norm_stats = {}  # Initialize BEFORE normalization
        self.normalized_outputs = self.normalize_all_task_outputs(stacked_outputs)
        
        # Important objects
        self.model = model
        self.weights = list(model.parameters())
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)
        
        # Hyperparameters (same as OptimizedMAML)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K  # Support set size
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch
        
        # Metrics (same as OptimizedMAML)
        self.plot_every = 10
        self.print_every = 200
        self.meta_losses = []
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            print(f"✅ Model moved to {device}")
        else:
            print("⚠️ Model running on CPU")
        
        print(f"🎯 GNN OptimizedMAML Configuration:")
        print(f"   Tasks: {self.num_tasks} (input conditions)")
        print(f"   Lib files per task: {self.lib_files_per_task}")
        print(f"   Support set size (K): {self.K}")
        print(f"   Inner steps: {self.inner_steps}")
        print(f"   Tasks per meta batch: {self.tasks_per_meta_batch}")
    
    def normalize_all_task_outputs(self, stacked_outputs):
        """Pre-normalize all task outputs before training"""
        if stacked_outputs is None:
            return None
            
        normalized = torch.zeros_like(stacked_outputs)
        
        print(f"📊 Pre-normalizing outputs for {self.num_tasks} tasks...")
        
        for task_idx in range(self.num_tasks):
            task_outputs = stacked_outputs[task_idx]  # Shape: [lib_files]
            
            # Calculate per-task statistics
            task_mean = task_outputs.mean().item()
            task_std = task_outputs.std().item()
            
            # Store normalization stats for each task
            self.task_norm_stats[task_idx] = {
                'mean': task_mean,
                'std': task_std
            }
            
            # Safe normalization
            if task_std > 1e-8:
                normalized[task_idx] = (task_outputs - task_mean) / task_std
            else:
                normalized[task_idx] = task_outputs - task_mean
                
            # Log some examples
            # if task_idx < 5 or task_idx % 1000 == 0:
            #     print(f"   Task {task_idx}: mean={task_mean:.6f}, std={task_std:.6f}")
        
        print(f"✅ Output normalization complete")
        return normalized
    
    def normalize_node_features(self, node_features):
        """Normalize node features using saved statistics"""
        if self.norm_stats is None:
            return node_features
            
        normalized = node_features.clone()
        
        # Normalize voltage (column 4)
        voltage_mask = normalized[:, 4] != 0
        if voltage_mask.any():
            normalized[voltage_mask, 4] = (
                normalized[voltage_mask, 4] - self.norm_stats['node_features']['voltage']['mean']
            ) / self.norm_stats['node_features']['voltage']['std']
        
        # Normalize input_slew (column 5)
        slew_mask = normalized[:, 5] != 0
        if slew_mask.any():
            normalized[slew_mask, 5] = (
                normalized[slew_mask, 5] - self.norm_stats['node_features']['input_slew']['mean']
            ) / self.norm_stats['node_features']['input_slew']['std']
        
        # Normalize output_load (column 6)
        load_mask = normalized[:, 6] != 0
        if load_mask.any():
            normalized[load_mask, 6] = (
                normalized[load_mask, 6] - self.norm_stats['node_features']['output_load']['mean']
            ) / self.norm_stats['node_features']['output_load']['std']
        
        return normalized
    
    def create_pyg_data_with_adj_matrix(self, graph_sample):
        """Create PyTorch Geometric Data object with normalization"""
        node_features = graph_sample['node_features']
        adjacency_matrix = graph_sample['adjacency_matrix']
        edge_index = graph_sample['edge_index']
        
        # Apply normalization
        normalized_features = self.normalize_node_features(node_features)
        
        # Apply adjacency matrix multiplication (A × X)
        aggregated_features = torch.matmul(adjacency_matrix, normalized_features)
        
        data = Data(
            x=aggregated_features,
            edge_index=edge_index
        )
        
        return data
    
    def get_task_data(self, task_id):
        """
        Get data for a specific task (same input condition across lib files)
        Same logic as OptimizedMAML but for graphs
        """
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} out of range (max: {self.num_tasks-1})")
        
        graphs = []
        outputs = []
        
        # Collect data from all lib files for this task
        for lib_idx in range(self.lib_files_per_task):
            if task_id < len(self.graph_data_per_file[lib_idx]):
                graph = self.graph_data_per_file[lib_idx][task_id]
                # Use pre-normalized outputs
                output = self.normalized_outputs[task_id, lib_idx].item()
                
                graphs.append(graph)
                outputs.append(output)
        
        return graphs, outputs
    
    def inner_loop_single_task(self, task_idx):
        """
        Inner loop for single task - same structure as OptimizedMAML
        """
        temp_weights = [w.clone() for w in self.weights]
        
        # Get task data (already normalized)
        graphs, outputs = self.get_task_data(task_idx)
        
        # Log normalization stats occasionally (from pre-computed stats)
        if random.random() < 0.01 and task_idx in self.task_norm_stats:  # Log 1% of tasks
            stats = self.task_norm_stats[task_idx]
            #print(f"   Task {task_idx} (pre-normalized): mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        
        # Sample K graphs for support set
        total_libs = len(graphs)
        if total_libs < self.K:
            support_indices = list(range(total_libs))
        else:
            support_indices = random.sample(range(total_libs), self.K)
        
        support_graphs = [graphs[i] for i in support_indices]
        support_outputs = [outputs[i] for i in support_indices]
        
        for step in range(self.inner_steps):
            # Convert to PyG batch
            batch_data = []
            for graph in support_graphs:
                data = self.create_pyg_data_with_adj_matrix(graph)
                batch_data.append(data)
            
            if not batch_data:
                return torch.tensor(0.0, requires_grad=True).to(device)
            
            X = Batch.from_data_list(batch_data).to(device)
            y = torch.tensor(support_outputs, dtype=torch.float32).to(device).view(-1, 1)
            
            # Mixed precision for stability (like original)
            with torch.cuda.amp.autocast():
                predictions = functional_forward(self.model, X, temp_weights)
                loss = self.criterion(predictions, y + 1e-6) / self.K
            
            grad = torch.autograd.grad(loss, temp_weights, create_graph=True, allow_unused=True)
            temp_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(temp_weights, grad)]
        
        # Meta-update loss calculation (query set)
        # Use different samples for query
        if total_libs > self.K:
            remaining_indices = [i for i in range(total_libs) if i not in support_indices]
            if remaining_indices:
                query_indices = random.sample(remaining_indices, min(self.K, len(remaining_indices)))
            else:
                query_indices = support_indices  # Fallback
        else:
            query_indices = support_indices
        
        query_graphs = [graphs[i] for i in query_indices]
        query_outputs = [outputs[i] for i in query_indices]  # Already normalized
        
        # Convert query to PyG batch
        batch_data = []
        for graph in query_graphs:
            data = self.create_pyg_data_with_adj_matrix(graph)
            batch_data.append(data)
        
        X = Batch.from_data_list(batch_data).to(device)
        y = torch.tensor(query_outputs, dtype=torch.float32).to(device).view(-1, 1)
        
        with torch.cuda.amp.autocast():
            predictions = functional_forward(self.model, X, temp_weights)
            loss = self.criterion(predictions, y + 1e-6) / len(query_indices)
        
        return loss
    
    def main_loop_optimized(self, num_iterations):
        """
        Main MAML loop - optimized version (same as OptimizedMAML)
        """
        from concurrent.futures import ThreadPoolExecutor
        
        epoch_loss = 0
        
        for iteration in range(1, num_iterations + 1):
            meta_losses = []
            
            # Parallel task processing (like original)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for _ in range(self.tasks_per_meta_batch):
                    task_idx = random.randint(0, self.num_tasks - 1)
                    future = executor.submit(self.inner_loop_single_task, task_idx)
                    futures.append(future)
                
                for future in futures:
                    try:
                        meta_losses.append(future.result())
                    except Exception as e:
                        print(f"⚠️ Task failed with error: {e}")
                        continue
            
            if not meta_losses:
                continue
            
            meta_loss = sum(meta_losses) / len(meta_losses)
            
            # Meta gradient computation and update
            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()
            
            # Logging (same as original)
            epoch_loss += meta_loss.item()
            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.print_every:.6f}")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0
    
    def main_loop_sequential(self, num_iterations):
        """
        Sequential processing version - most stable (same as OptimizedMAML)
        """
        epoch_loss = 0
        
        for iteration in range(1, num_iterations + 1):
            meta_losses = []
            
            # Sequential task processing
            for _ in range(self.tasks_per_meta_batch):
                task_idx = random.randint(0, self.num_tasks - 1)
                try:
                    loss = self.inner_loop_single_task(task_idx)
                    meta_losses.append(loss)
                except Exception as e:
                    print(f"⚠️ Sequential task {task_idx} failed: {e}")
                    continue
            
            if not meta_losses:
                continue
            
            meta_loss = sum(meta_losses) / len(meta_losses)
            
            # Meta gradient computation and update
            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()
            
            # Logging
            epoch_loss += meta_loss.item()
            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.print_every:.6f}")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

def calculate_norm_stats_from_data(graph_data_per_file):
    """
    Calculate normalization statistics from actual loaded data
    """
    print(f"   🔍 Calculating norm_stats from {len(graph_data_per_file)} lib files...")
    
    all_voltages = []
    all_input_slews = []
    all_output_loads = []
    all_delays = []
    all_process_values = []
    all_temperatures = []
    
    # Sample from multiple lib files and tasks to get representative stats
    sample_count = 0
    for lib_idx, lib_graphs in enumerate(graph_data_per_file):
        if lib_idx % 10 == 0:  # Sample every 10th lib file
            for task_idx in range(0, min(len(lib_graphs), 1000), 10):  # Sample every 10th task
                graph = lib_graphs[task_idx]
                
                if 'node_features' in graph:
                    features = graph['node_features']
                    
                    # Column 4: voltage
                    voltage_values = features[:, 4]
                    voltage_values = voltage_values[voltage_values != 0]
                    if len(voltage_values) > 0:
                        all_voltages.extend(voltage_values.tolist())
                    
                    # Column 5: input_slew 
                    slew_values = features[:, 5]
                    slew_values = slew_values[slew_values != 0]
                    if len(slew_values) > 0:
                        all_input_slews.extend(slew_values.tolist())
                    
                    # Column 6: output_load
                    load_values = features[:, 6]
                    load_values = load_values[load_values != 0]
                    if len(load_values) > 0:
                        all_output_loads.extend(load_values.tolist())
                    
                    # Column 3: delay or other feature
                    delay_values = features[:, 3]
                    delay_values = delay_values[delay_values != 0]
                    if len(delay_values) > 0:
                        all_delays.extend(delay_values.tolist())
                
                # Check for global attributes as well
                if 'voltage' in graph:
                    all_voltages.append(float(graph['voltage']))
                if 'input_slew' in graph:
                    all_input_slews.append(float(graph['input_slew']))
                if 'output_load' in graph:
                    all_output_loads.append(float(graph['output_load']))
                
                sample_count += 1
    
    print(f"   📊 Sampled {sample_count} graphs")
    
    # Calculate statistics with safe handling
    def safe_stats(values, name):
        if len(values) == 0:
            print(f"     ⚠️ No {name} values found, using defaults")
            return {'mean': 1.0, 'std': 0.1}
        
        values_array = np.array(values)
        mean_val = values_array.mean()
        std_val = values_array.std()
        
        # Handle very small std
        if std_val < 1e-8:
            print(f"     ⚠️ {name} std too small ({std_val:.2e}), using 0.1")
            std_val = 0.1
        
        print(f"     {name}: mean={mean_val:.6f}, std={std_val:.6f} (n={len(values)})")
        return {'mean': float(mean_val), 'std': float(std_val)}
    
    norm_stats = {
        'node_features': {
            'voltage': safe_stats(all_voltages, 'Voltage'),
            'input_slew': safe_stats(all_input_slews, 'Input Slew'),
            'output_load': safe_stats(all_output_loads, 'Output Load'),
            'delay': safe_stats(all_delays, 'Delay'),
            'process': {'mean': 1.0, 'std': 0.1},  # Default
            'temperature': {'mean': 25.0, 'std': 10.0}  # Default
        }
    }
    
    return norm_stats

def load_gnn_data_for_maml(process_type, corner_type, data_type='cell', graph_mode='stage_aware'):
    """
    Load GNN train data from train_test_split folders and merge multiple matching datasets

    Args:
        process_type: Process type (RVT, LVT, SLVT, SRAM)
        corner_type: Corner type (TT, FF, SS)
        data_type: Data type ('cell' or 'transition')
        graph_mode: Graph mode ('stage_aware' or 'full_graph')
    """
    print(f"🎯 Loading GNN train data for MAML: {process_type}_{corner_type}")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")

    base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp"
    matching_folders = []

    # Determine train indices filename based on data_type and graph_mode
    train_indices_filename = f"train_indices_{data_type}_{graph_mode}.pth"

    # Find all matching folders for this condition
    for item in os.listdir(base_path):
        parts = item.split('_')
        dataset_process = None
        for part in parts:
            if part in ['RVT', 'LVT', 'SLVT', 'SRAM']:
                dataset_process = part
                break

        if dataset_process != process_type:
            continue

        # Check corner type matching
        corner_match = False
        if corner_type == "TT" and not item.endswith(('_FF', '_SS')):
            corner_match = True
        elif corner_type == "FF" and item.endswith('_FF'):
            corner_match = True
        elif corner_type == "SS" and item.endswith('_SS'):
            corner_match = True

        if corner_match:
            train_data_path = f"{base_path}/{item}/train_test_split/{train_indices_filename}"
            if os.path.exists(train_data_path):
                matching_folders.append((item, train_data_path))
            else:
                print(f"   ⚠️ Train indices not found: {train_data_path}")

    if matching_folders:
        print(f"   Using train indices: {train_indices_filename}")
    if not matching_folders:
        raise ValueError(f"No train data found for {process_type}_{corner_type}")
    
    print(f"   📂 Found {len(matching_folders)} matching datasets:")
    for folder, _ in matching_folders:
        print(f"     - {folder}")
    
    # Load and merge all matching datasets
    all_graph_data_per_file = []
    all_stacked_outputs = []

    for folder, train_indices_path in matching_folders:
        print(f"   📥 Loading: {folder}")

        # Load train indices
        train_meta = torch.load(train_indices_path, weights_only=False, map_location='cpu')
        sample_indices = train_meta['sample_indices']  # List of sample indices
        data_file = train_meta['data_file']  # Path to cell_all_graph_data.pth

        # Load full dataset
        print(f"     Loading full dataset from: {data_file}")
        full_data = torch.load(data_file, weights_only=False, map_location='cpu')
        graph_data_per_file = full_data['graph_data_per_file']  # [num_libs][num_samples]
        full_outputs = full_data['stacked_outputs']  # [num_samples, num_libs]

        # Extract train subset: only selected samples, keep all libs
        train_outputs = full_outputs[sample_indices, :]  # [num_train_samples, num_libs]

        print(f"     Full dataset: {full_outputs.shape[0]} samples, {full_outputs.shape[1]} lib files")
        print(f"     Train subset: {train_outputs.shape[0]} samples, {train_outputs.shape[1]} lib files")

        stacked_outputs = train_outputs
        
        if not all_graph_data_per_file:
            # First dataset - initialize
            all_graph_data_per_file = [[] for _ in range(len(graph_data_per_file))]

        # Merge graph data: extract only train samples for each lib
        for lib_idx, lib_graphs in enumerate(graph_data_per_file):
            # Extract only the samples in sample_indices
            train_graphs = [lib_graphs[idx] for idx in sample_indices]
            all_graph_data_per_file[lib_idx].extend(train_graphs)
        
        # Collect stacked outputs
        all_stacked_outputs.append(stacked_outputs)
    
    # Concatenate all outputs
    merged_stacked_outputs = torch.cat(all_stacked_outputs, dim=0)
    
    print(f"   ✅ Merged data:")
    print(f"     Total tasks: {merged_stacked_outputs.shape[0]} (input conditions)")
    print(f"     Lib files per task: {merged_stacked_outputs.shape[1]} (process variations)")

    # Calculate norm_stats from merged data (including output statistics)
    print(f"\n🔧 Calculating normalization statistics (including output values)...")
    output_values = merged_stacked_outputs.flatten().numpy()
    all_output_values = output_values[output_values > 0].tolist()  # Remove zeros/negatives

    norm_stats = calculate_norm_stats_from_data(all_graph_data_per_file)

    # Add output statistics
    if len(all_output_values) > 0:
        output_mean = np.mean(all_output_values)
        output_std = np.std(all_output_values)
        if output_std < 1e-8:
            output_std = 1.0
        norm_stats['output'] = {'mean': float(output_mean), 'std': float(output_std)}
        print(f"   Output values: mean={output_mean:.6f}, std={output_std:.6f} (n={len(all_output_values)})")
    else:
        norm_stats['output'] = {'mean': 100.0, 'std': 50.0}
        print(f"   ⚠️ No output values found, using defaults")

    return all_graph_data_per_file, merged_stacked_outputs, norm_stats

def parse_arguments():
    """Parse command-line arguments for MAML GNN training parameters"""
    parser = argparse.ArgumentParser(
        description='MAML GNN Training with OptimizedMAML structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python maml_gnn_training_optimized.py --process SRAM --corner FF
  python maml_gnn_training_optimized.py --process LVT --corner TT --innerdiv 10 --meta_lr 0.0001
  python maml_gnn_training_optimized.py --process RVT --corner SS --conv_hidden_dim 256 --fc_hidden_dim 256 --inner_steps 2
  python maml_gnn_training_optimized.py --process SLVT --corner FF --innerdiv 20 --meta_lr 0.00005 --conv_hidden_dim 128 --fc_hidden_dim 128 --inner_steps 1

Note: inner_lr is calculated as 0.001 / innerdiv
  innerdiv=10  -> inner_lr=0.0001
  innerdiv=20  -> inner_lr=0.00005
  innerdiv=100 -> inner_lr=0.00001
"""
    )

    # Required arguments
    parser.add_argument('--process', type=str, required=True,
                       choices=['RVT', 'LVT', 'SLVT', 'SRAM'],
                       help='Process type: RVT, LVT, SLVT, or SRAM')
    parser.add_argument('--corner', type=str, required=True,
                       choices=['TT', 'FF', 'SS'],
                       help='Process corner: TT, FF, or SS')

    # Training hyperparameters
    parser.add_argument('--innerdiv', type=int, default=10,
                       help='Inner learning rate divisor (inner_lr = 0.001 / innerdiv, default: 10 -> 0.0001)')
    parser.add_argument('--meta_lr', type=float, default=0.0001,
                       help='Meta learning rate (default: 0.0001)')
    parser.add_argument('--inner_steps', type=int, default=1,
                       help='Number of inner loop adaptation steps (default: 1)')

    # Model architecture parameters
    parser.add_argument('--conv_hidden_dim', type=int, default=128,
                       help='Convolution layer hidden dimension (default: 128)')
    parser.add_argument('--num_conv_layers', type=int, default=3,
                       help='Number of GCN convolutional layers (default: 3)')
    parser.add_argument('--fc_hidden_dim', type=int, default=40,
                       help='FC layer hidden dimension (default: 40)')
    parser.add_argument('--num_fc_layers', type=int, default=2, choices=[1, 2, 3],
                       help='Number of FC layers (default: 2)')

    # Training configuration
    parser.add_argument('--total_iterations', type=int, default=100000,
                       help='Total number of training iterations (default: 100000)')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size for saving checkpoints (default: 10000)')
    parser.add_argument('--K', type=int, default=5,
                       help='Support set size (default: 5)')
    parser.add_argument('--tasks_per_meta_batch', type=int, default=16,
                       help='Number of tasks per meta batch (default: 16)')

    # GPU configuration
    parser.add_argument('--gpu', type=str, default='2',
                       help='GPU device ID (default: 2)')

    # Data type
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type: cell or transition (default: cell)')

    # Graph mode
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode: stage_aware or full_graph (default: stage_aware)')

    return parser.parse_args()

def main():
    """
    Main function - same structure as datacheck_SLVT_test5_FF.py
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Extract parameters from args
    total_iterations = args.total_iterations
    chunk_size = args.chunk_size
    num_chunks = total_iterations // chunk_size
    process = args.process
    corner_type = args.corner

    # Model architecture parameters
    conv_hidden_dim = args.conv_hidden_dim
    num_conv_layers = args.num_conv_layers
    fc_hidden_dim = args.fc_hidden_dim
    num_fc_layers = args.num_fc_layers
    inner_steps = args.inner_steps
    innerdiv = args.innerdiv

    # Calculate inner_lr from innerdiv
    base_lr = 0.001
    inner_lr = base_lr / innerdiv

    meta_lr = args.meta_lr
    K = args.K
    tasks_per_meta_batch = args.tasks_per_meta_batch
    data_type = args.data_type
    graph_mode = args.graph_mode

    start_time = time.time()

    print(f"\n🚀 MAML GNN Training with OptimizedMAML structure")
    print(f"📋 Configuration:")
    print(f"   Process: {process}")
    print(f"   Corner: {corner_type}")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")
    print(f"   Inner div: {innerdiv} (inner_lr = {base_lr} / {innerdiv} = {inner_lr})")
    print(f"   Meta LR: {meta_lr}")
    print(f"   Conv hidden dim: {conv_hidden_dim}")
    print(f"   Num conv layers: {num_conv_layers}")
    print(f"   FC hidden dim: {fc_hidden_dim}")
    print(f"   Num FC layers: {num_fc_layers}")
    print(f"   Inner steps: {inner_steps}")
    print(f"   K (support set): {K}")
    print(f"   Tasks per meta batch: {tasks_per_meta_batch}")
    print(f"   Total iterations: {total_iterations}")
    print(f"   Chunk size: {chunk_size}")
    print(f"   GPU: {args.gpu}")

    # Load GNN data
    print("📊 Loading GNN data...")
    graph_data_per_file, stacked_outputs, norm_stats = load_gnn_data_for_maml(process, corner_type, data_type, graph_mode)

    # Create GNN MAML model (same pattern as original)
    print("🤖 Creating GNN MAML model...")
    gnn_maml = GNNOptimizedMAML(
        model=create_maml_gcn_model(
                node_features=7,
                pooling='mean',
                output_dim=1,
                dropout=0.0,
                conv_hidden_dim=conv_hidden_dim,
                num_conv_layers=num_conv_layers,
                fc_hidden_dim=fc_hidden_dim,
                num_fc_layers=num_fc_layers
            
        ),
        graph_data_per_file=graph_data_per_file,
        stacked_outputs=stacked_outputs,
        norm_stats=norm_stats,
        inner_lr=inner_lr,
        meta_lr=meta_lr,
        inner_steps=inner_steps,
        K=K,
        tasks_per_meta_batch=tasks_per_meta_batch
    )
    
    # Training in chunks (same pattern as original)
    for chunk in range(num_chunks):
        print(f"\n📦 Processing chunk {chunk+1}/{num_chunks}")
        chunk_start_time = time.time()
        
        try:
            # Try optimized version first
            gnn_maml.main_loop_optimized(num_iterations=chunk_size)
        except Exception as e:
            print(f"⚠️ 병렬 처리 실패, 순차적 처리로 전환: {e}")
            try:
                gnn_maml.main_loop_sequential(num_iterations=chunk_size)
            except Exception as e2:
                print(f"⚠️ 순차 처리도 실패: {e2}")
                print("⚠️ 학습률을 더 낮춰서 재시도...")
                gnn_maml.inner_lr *= 0.5
                gnn_maml.meta_lr *= 0.5
                gnn_maml.main_loop_sequential(num_iterations=chunk_size//2)
        
        # GPU synchronization and timing
        torch.cuda.synchronize()
        chunk_end_time = time.time()
        
        chunk_time = chunk_end_time - chunk_start_time
        print(f"⏱️ Chunk {chunk+1} completed in {chunk_time:.2f}s")
        print(f"📈 Average time per iteration: {chunk_time/chunk_size:.4f}s")
        
        # Memory monitoring
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # Save checkpoint with iteration number
        checkpoint_dir = "../../../pretrained_models/gnn_maml_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        iterations_completed = (chunk + 1) * chunk_size

        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"
        checkpoint_path = f"{checkpoint_dir}/gnn_maml_{process}_{corner_type}_{data_type}_{graph_mode}_innerdiv{innerdiv}_iter{iterations_completed}_inner{inner_steps}{arch_suffix}.pth"

        torch.save({
            'model_state_dict': gnn_maml.model.state_dict(),
            'norm_stats': norm_stats,
            'config': {
                'process_type': process,
                'corner_type': corner_type,
                'data_type': data_type,
                'graph_mode': graph_mode,
                'conv_hidden_dim': conv_hidden_dim,
                'num_conv_layers': num_conv_layers,
                'fc_hidden_dim': fc_hidden_dim,
                'num_fc_layers': num_fc_layers,
                'inner_steps': inner_steps,
                'inner_lr': inner_lr,
                'meta_lr': meta_lr,
                'K': K,
                'innerdiv': innerdiv,
                'iterations_completed': iterations_completed,
                'meta_losses': gnn_maml.meta_losses
            }
        }, checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_dir = "../../../pretrained_models/gnn_maml_final"
    os.makedirs(final_model_dir, exist_ok=True)

    arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"
    final_model_path = f"{final_model_dir}/gnn_maml_{process}_{corner_type}_{data_type}_{graph_mode}_innerdiv{innerdiv}_iter{total_iterations}_inner{inner_steps}{arch_suffix}.pth"

    torch.save({
        'model_state_dict': gnn_maml.model.state_dict(),
        'norm_stats': norm_stats,
        'config': {
            'process_type': process,
            'corner_type': corner_type,
            'data_type': data_type,
            'graph_mode': graph_mode,
            'conv_hidden_dim': conv_hidden_dim,
                'num_conv_layers': num_conv_layers,
                'fc_hidden_dim': fc_hidden_dim,
                'num_fc_layers': num_fc_layers,
            'inner_steps': inner_steps,
            'inner_lr': inner_lr,
            'innerdiv': innerdiv,
            'meta_lr': meta_lr,
            'K': K,
            'total_iterations': total_iterations,
            'meta_losses': gnn_maml.meta_losses
        }
    }, final_model_path)
    
    print(f"🏁 Training complete. Model saved to: {final_model_path}")
    
    # GPU memory cleanup
    del gnn_maml, graph_data_per_file, stacked_outputs
    torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time:.2f}s")

if __name__ == "__main__":
    main()