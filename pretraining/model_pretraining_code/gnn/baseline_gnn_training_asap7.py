#!/usr/bin/env python
"""
Baseline GNN Training with Unified Dataset Format - mmap Loading

Uses pre-split and pre-processed datasets from split_gnn_dataset.py
No preprocessing needed - data is already filtered and norm_stats are pre-computed.
Memory-mapped loading (mmap=True) to avoid loading entire file into RAM.

Key Features:
- Loads unified train file with mmap (train_cell_full_graph.pth)
- Topology reconstruction from cache on-the-fly
- Standard mini-batch training (NOT MAML)
- Per-task output normalization (on-the-fly)
- Adam optimizer with weight decay
"""

import os
import sys

# Parse GPU argument before importing torch
# This is necessary because CUDA_VISIBLE_DEVICES must be set before torch import
def get_gpu_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return '0'  # default

os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_from_args()

import torch
from torch import optim
import torch.nn as nn
import sys
import random
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
import numpy as np
import time
import argparse
import gc

# GPU optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name())
else:
    print('CUDA is not available!')

# Add paths
sys.path.append('./')
sys.path.append('tools/data_processing')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model_code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from gnn_maml import (
    MAML_GNN_Model,
    create_maml_gcn_model
)
from gnn_data_preprocessing_utils import (
    normalize_node_features_safe,
    normalize_task_outputs
)


class UnifiedBaselineDataset(Dataset):
    """
    PyTorch Dataset for Baseline GNN training with true mmap loading.
    Uses tensor-based storage format for efficient memory-mapped access.

    Data format:
    - node_features: [num_libs, total_nodes, num_features] - mmap tensor
    - outputs: [num_libs, num_tasks] - mmap tensor
    - node_slices: [num_tasks + 1] - cumulative node indices
    - cell_names: [num_tasks] - for topology cache lookup

    edge_index is loaded from topology_cache using cell_name.
    """

    def __init__(self, unified_train_path, graph_mode='full_graph'):
        """
        Args:
            unified_train_path: Path to unified train file (train_cell_full_graph.pth)
            graph_mode: 'full_graph' or 'stage_aware'
        """
        self.unified_train_path = unified_train_path
        self.graph_mode = graph_mode

        # Data references (mmap)
        self._data = None
        self._node_features = None
        self._outputs = None
        self._node_slices = None
        self._cell_names = None
        self._delay_types = None
        self._output_names = None

        # Metadata
        self._topology_cache = None
        self._norm_stats = None
        self._num_tasks = None
        self._num_libs = None

        # Load data with mmap (large, memory-mapped)
        # This also extracts cache_path for topology cache loading
        self._load_data()

        print(f"UnifiedBaselineDataset initialized (tensor format, mmap):")
        print(f"   Train file: {unified_train_path}")
        print(f"   Tasks: {self._num_tasks}")
        print(f"   Libs: {self._num_libs}")
        print(f"   Graph mode: {graph_mode}")
        print(f"   node_features: {self._node_features.shape}")
        print(f"   outputs: {self._outputs.shape}")
        print(f"   topology_cache: {len(self._topology_cache)} cells")

    def _load_data(self):
        """Load data using memory mapping (mmap) and topology cache"""
        print(f"   Loading with mmap=True (memory-mapped tensors)")
        data = torch.load(self.unified_train_path, weights_only=False, map_location='cpu', mmap=True)

        # Check format
        if data.get('format') != 'tensor':
            raise ValueError(f"Expected tensor format, got: {data.get('format', 'legacy')}")

        # Store mmap tensor references
        self._node_features = data['node_features']
        self._outputs = data['outputs']
        self._node_slices = data['node_slices']
        self._cell_names = data['cell_names']
        self._delay_types = data.get('delay_types', None)
        self._output_names = data.get('output_names', None)

        # Metadata
        self._num_libs = data['num_libs']
        self._num_tasks = data['num_tasks']
        self._norm_stats = data.get('norm_stats', None)

        # Load topology cache from cache_path in data file
        cache_path = data.get('cache_path', None)
        if cache_path:
            self._load_topology_cache(cache_path)

        # Keep reference to prevent garbage collection
        self._data = data

    def _load_topology_cache(self, cache_path):
        """Load topology cache (small file, load entirely into memory)"""
        # Handle path variations (/mnt/home vs /home)
        if cache_path.startswith('/mnt/home/'):
            cache_path = cache_path.replace('/mnt/home/', '/home/')

        # Resolve relative path if needed
        if not os.path.isabs(cache_path) or not os.path.exists(cache_path):
            cache_filename = os.path.basename(cache_path)
            cache_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Topology cache not found: {cache_path}")

        print(f"   Loading topology cache: {cache_path}")
        self._topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu')
        print(f"   Loaded {len(self._topology_cache)} cells")

    @property
    def topology_cache(self):
        """Get topology cache"""
        return self._topology_cache

    @property
    def norm_stats(self):
        return self._norm_stats

    @property
    def num_libs(self):
        return self._num_libs

    @property
    def num_tasks(self):
        return self._num_tasks

    def __len__(self):
        return self._num_tasks

    def __getitem__(self, task_idx):
        """Get data for a specific task"""
        if task_idx >= self._num_tasks:
            raise IndexError(f"Task index {task_idx} out of range (max: {self._num_tasks - 1})")

        # Get cell name and metadata from stored data
        cell_name = self._cell_names[task_idx] if self._cell_names else f'task_{task_idx}'
        delay_type = self._delay_types[task_idx] if self._delay_types else 'rise'
        output_name = self._output_names[task_idx] if self._output_names else ''

        # Get edge_index from topology cache based on graph_mode
        edge_index = None
        if self.topology_cache and cell_name in self.topology_cache:
            cell_cache = self.topology_cache[cell_name]

            if self.graph_mode == 'stage_aware' and 'output_topologies' in cell_cache:
                # Stage-aware: use delay_type and output_name to get correct adjacency matrix
                if output_name in cell_cache['output_topologies']:
                    output_topo = cell_cache['output_topologies'][output_name]
                    if 'rise' in delay_type:
                        adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
                    else:
                        adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
                    edge_index = adjacency_matrix.nonzero().t()
                else:
                    # Fallback to full graph if output_name not found
                    if 'edge_index' in cell_cache:
                        edge_index = cell_cache['edge_index']
                    elif 'adjacency_matrix' in cell_cache:
                        edge_index = cell_cache['adjacency_matrix'].nonzero().t()
            else:
                # Full graph mode
                if 'edge_index' in cell_cache:
                    edge_index = cell_cache['edge_index']
                elif 'adjacency_matrix' in cell_cache:
                    edge_index = cell_cache['adjacency_matrix'].nonzero().t()

        # Get slice indices for node_features
        node_start = self._node_slices[task_idx].item()
        node_end = self._node_slices[task_idx + 1].item()

        # Get outputs for all libs
        task_outputs = self._outputs[:, task_idx]

        # Build minimal_samples list for compatibility
        minimal_samples = []
        for lib_idx in range(self._num_libs):
            # Extract node features for this lib and task
            task_node_features = self._node_features[lib_idx, node_start:node_end, :]

            sample = {
                'node_features': task_node_features,
                'edge_index': edge_index,
                'output': task_outputs[lib_idx].item(),
                'cell_name': cell_name,
                'delay_type': delay_type,
                'output_name': output_name,
            }
            minimal_samples.append(sample)

        return {
            'minimal_samples': minimal_samples,
            'outputs': task_outputs.tolist(),
            'task_idx': task_idx,
            'cell_name': cell_name,
            'delay_type': delay_type,
            'output_name': output_name
        }


class GNN_Baseline_Unified:
    """
    GNN Baseline training with unified dataset (mmap loading).
    Standard mini-batch training (NOT MAML).

    Data organization:
    - Dataset organized into tasks (same input condition across lib files)
    - Each task has num_libs samples (different lib files)
    - Each iteration: randomly select 1 task, then randomly sample batch_size samples
    """

    def __init__(self, model, lr=2e-3, wd=5e-3,
                 dataset=None, iteration=100000, batch_size=5):
        """
        Args:
            model: GNN model
            lr: Learning rate
            wd: Weight decay
            dataset: UnifiedBaselineDataset instance
            iteration: Number of training iterations
            batch_size: Number of samples per task in mini-batch
        """
        self.lr = lr
        self.wd = wd
        self.iteration = iteration
        self.batch_size = batch_size

        # Dataset
        self.dataset = dataset
        self.topology_cache = dataset.topology_cache if dataset else None
        self.cache_type = dataset.graph_mode if dataset else 'full_graph'
        self.norm_stats = dataset.norm_stats if dataset else None

        self.num_tasks = dataset.num_tasks if dataset else 0
        self.lib_files_per_task = dataset.num_libs if dataset else 0

        # Task output normalization (computed on-the-fly)
        self.task_norm_stats = {}

        # Model and optimizer
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            print(f"Model moved to {device}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        print(f"\nGNN Baseline Unified Configuration:")
        print(f"  Cache type: {self.cache_type}")
        print(f"  Number of tasks: {self.num_tasks}")
        print(f"  Lib files per task: {self.lib_files_per_task}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Weight decay: {self.wd}")
        print(f"  Iterations: {self.iteration}")
        print(f"  Memory efficient: mmap loading enabled")

    def normalize_node_features(self, node_features):
        """Normalize node features using saved statistics"""
        if self.norm_stats is None:
            return node_features

        normalized, _ = normalize_node_features_safe(
            node_features,
            norm_stats=self.norm_stats['node_features']
        )
        return normalized

    def normalize_outputs(self, outputs, task_idx):
        """Normalize outputs for a task (with caching)"""
        if task_idx not in self.task_norm_stats:
            outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
            mean = outputs_tensor.mean().item()
            std = outputs_tensor.std().item()
            if std < 1e-8:
                std = 1.0
            self.task_norm_stats[task_idx] = {'mean': mean, 'std': std}

        stats = self.task_norm_stats[task_idx]
        return [(o - stats['mean']) / stats['std'] for o in outputs]

    def get_adjacency_matrix_from_cache(self, minimal_sample):
        """Load pre-computed adjacency matrix from topology cache"""
        cell_name = minimal_sample['cell_name']

        if cell_name not in self.topology_cache:
            raise ValueError(f"Cell {cell_name} not found in topology cache")

        cell_cache = self.topology_cache[cell_name]

        if self.cache_type == 'stage_aware':
            output_name = minimal_sample['output_name']
            delay_type = minimal_sample['delay_type']

            if output_name not in cell_cache['output_topologies']:
                raise ValueError(f"Output {output_name} not found for cell {cell_name}")

            output_topo = cell_cache['output_topologies'][output_name]

            if 'rise' in delay_type:
                adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
            else:
                adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
        else:
            adjacency_matrix = cell_cache['adjacency_matrix']

        return adjacency_matrix

    def create_pyg_data_with_adj_matrix(self, minimal_sample):
        """Create PyTorch Geometric Data object from minimal sample"""
        node_features = minimal_sample['node_features']
        adjacency_matrix = self.get_adjacency_matrix_from_cache(minimal_sample)
        normalized_features = self.normalize_node_features(node_features)
        edge_index = adjacency_matrix.nonzero().t()

        data = Data(
            x=normalized_features,
            edge_index=edge_index
        )
        return data

    def get_task_data(self, task_idx):
        """Get data for a specific task"""
        task_data = self.dataset[task_idx]
        minimal_samples = task_data['minimal_samples']
        raw_outputs = task_data['outputs']

        # Normalize outputs on-the-fly
        outputs = self.normalize_outputs(raw_outputs, task_idx)

        return minimal_samples, outputs

    def loop(self, checkpoint_dir='checkpoints'):
        """
        Training loop with task-based mini-batch sampling

        Each iteration:
        1. Select a random task
        2. Within that task, randomly sample batch_size samples
        3. Train using standard SGD

        Returns:
            Average training loss
        """
        running_loss = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

        for i in range(self.iteration):
            if i % 1000 == 0:
                avg_loss = running_loss / max(1, i)
                print(f"Iteration {i}/{self.iteration}, Avg Loss: {avg_loss:.6f}")

            # Select random task
            task_idx = random.randint(0, self.num_tasks - 1)

            # Get task data
            minimal_samples, outputs = self.get_task_data(task_idx)

            # Sample random indices
            total_samples = len(minimal_samples)
            if total_samples < self.batch_size:
                sample_indices = list(range(total_samples))
            else:
                sample_indices = random.sample(range(total_samples), self.batch_size)

            # Create mini-batch
            batch_data = []
            batch_outputs = []

            for idx in sample_indices:
                minimal_sample = minimal_samples[idx]
                output = outputs[idx]
                data = self.create_pyg_data_with_adj_matrix(minimal_sample)
                batch_data.append(data)
                batch_outputs.append(output)

            # Create PyG batch
            X = Batch.from_data_list(batch_data).to(device)
            y = torch.tensor(batch_outputs, dtype=torch.float32).to(device).view(-1, 1)

            # Training step
            self.optimizer.zero_grad()
            self.model.train()
            y_pred = self.model(X)

            the_loss = nn.functional.mse_loss(y_pred, y)
            the_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += the_loss.item()

        return float(running_loss / self.iteration)


def get_unified_train_path(process_type, corner_type, data_type='cell', graph_mode='full_graph'):
    """Get path to unified train file"""
    base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp"
    unified_dir = os.path.join(base_path, f"unified_{process_type}_{corner_type}")
    train_file = os.path.join(unified_dir, f"train_{data_type}_{graph_mode}.pth")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Unified train file not found: {train_file}\n"
                               f"Run split_gnn_dataset.py --process {process_type} --corner {corner_type} first.")

    return train_file


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Baseline GNN Training with Unified Dataset (mmap Loading)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single configuration
  python baseline_gnn_training_unified.py --process LVT --corner FF --graph_mode full_graph

  # Architecture sweep
  python baseline_gnn_training_unified.py --process LVT --corner FF \\
      --conv_hidden_dim 64 128 256 \\
      --num_conv_layers 2 3 4

  # Run all 12 process-corner combinations
  python baseline_gnn_training_unified.py --run_all --graph_mode full_graph
"""
    )

    # Run all option
    parser.add_argument('--run_all', action='store_true',
                       help='Run all 12 combinations (4 process x 3 corners)')

    # Single run arguments
    parser.add_argument('--process', type=str,
                       choices=['RVT', 'LVT', 'SLVT', 'SRAM'],
                       help='Process type (required unless --run_all)')
    parser.add_argument('--corner', type=str,
                       choices=['TT', 'FF', 'SS'],
                       help='Process corner (required unless --run_all)')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-3,
                       help='Weight decay (default: 5e-3)')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Mini-batch size (default: 5)')

    # Model architecture parameters
    parser.add_argument('--conv_hidden_dim', type=int, nargs='+', default=[128],
                       help='Convolution layer hidden dimension(s) (default: 128)')
    parser.add_argument('--num_conv_layers', type=int, nargs='+', default=[3],
                       help='Number of GCN convolutional layers (default: 3)')
    parser.add_argument('--fc_hidden_dim', type=int, nargs='+', default=[40],
                       help='FC layer hidden dimension(s) (default: 40)')
    parser.add_argument('--num_fc_layers', type=int, nargs='+', default=[2],
                       help='Number of FC layers (default: 2)')

    # Training configuration
    parser.add_argument('--total_iterations', type=int, default=100000,
                       help='Total iterations (default: 100000)')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size (default: 10000)')

    # GPU configuration
    parser.add_argument('--gpu', type=str, default='2',
                       help='GPU device ID (default: 2)')

    # Data type
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')

    # Graph mode
    parser.add_argument('--graph_mode', type=str, default='full_graph',
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode (default: full_graph)')

    return parser.parse_args()


def train_single_config(process, corner_type, args):
    """Train a single process-corner configuration"""
    import itertools

    total_iterations = args.total_iterations
    chunk_size = args.chunk_size
    lr = args.lr
    wd = args.wd
    batch_size = args.batch_size
    data_type = args.data_type
    graph_mode = args.graph_mode

    # Generate architecture combinations
    arch_combinations = list(itertools.product(
        args.conv_hidden_dim,
        args.num_conv_layers,
        args.fc_hidden_dim,
        args.num_fc_layers
    ))

    num_combinations = len(arch_combinations)
    is_sweep = num_combinations > 1

    print(f"\n{'#'*80}")
    print(f"# Baseline GNN Training with Unified Dataset (mmap Loading)")
    print(f"{'#'*80}")
    print(f"Process: {process}")
    print(f"Corner: {corner_type}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {wd}")
    print(f"Batch size: {batch_size}")
    print(f"Total iterations: {total_iterations}")

    if is_sweep:
        print(f"\nArchitecture Sweep: {num_combinations} combinations")

    # Get unified train file path
    train_path = get_unified_train_path(process, corner_type, data_type, graph_mode)
    print(f"\nUnified train file: {train_path}")

    # Create dataset with mmap loading
    print("\nCreating dataset (mmap loading)...")
    data_load_start = time.time()

    dataset = UnifiedBaselineDataset(
        unified_train_path=train_path,
        graph_mode=graph_mode
    )

    data_load_time = time.time() - data_load_start
    print(f"Dataset initialized in {data_load_time:.2f} seconds")

    # Get norm_stats from dataset
    norm_stats = dataset.norm_stats

    # Create checkpoint directory
    checkpoint_dir = "../../../pretrained_models/gnn_baseline_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train each architecture combination
    trained_models = []
    sweep_start_time = time.time()

    for idx, (conv_hidden_dim, num_conv_layers, fc_hidden_dim, num_fc_layers) in enumerate(arch_combinations, 1):
        if is_sweep:
            print(f"\n{'='*80}")
            print(f"Architecture {idx}/{num_combinations}")
            print(f"{'='*80}")

        arch_start_time = time.time()

        print(f"Architecture:")
        print(f"  Conv hidden dim: {conv_hidden_dim}")
        print(f"  Num conv layers: {num_conv_layers}")
        print(f"  FC hidden dim: {fc_hidden_dim}")
        print(f"  Num FC layers: {num_fc_layers}")

        num_chunks = total_iterations // chunk_size

        # Create GNN baseline model
        print("Creating GNN Baseline model...")

        gnn_baseline = GNN_Baseline_Unified(
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
            lr=lr,
            wd=wd,
            dataset=dataset,
            iteration=chunk_size,
            batch_size=batch_size
        )

        # Training in chunks
        for chunk in range(num_chunks):
            print(f"\nProcessing chunk {chunk+1}/{num_chunks}")
            chunk_start_time = time.time()

            avg_loss = gnn_baseline.loop(checkpoint_dir=checkpoint_dir)

            torch.cuda.synchronize()
            chunk_end_time = time.time()

            chunk_time = chunk_end_time - chunk_start_time
            print(f"Chunk {chunk+1} completed in {chunk_time:.2f}s")
            print(f"Average loss: {avg_loss:.6f}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

            # Save checkpoint
            iterations_completed = (chunk + 1) * chunk_size
            arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

            checkpoint_path = f"{checkpoint_dir}/gnn_baseline_{process}_{corner_type}_{data_type}_{graph_mode}_iter{iterations_completed}{arch_suffix}.pth"

            torch.save({
                'model_state_dict': gnn_baseline.model.state_dict(),
                'optimizer_state_dict': gnn_baseline.optimizer.state_dict(),
                'norm_stats': norm_stats,
                'task_norm_stats': gnn_baseline.task_norm_stats,
                'config': {
                    'process_type': process,
                    'corner_type': corner_type,
                    'data_type': data_type,
                    'graph_mode': graph_mode,
                    'conv_hidden_dim': conv_hidden_dim,
                    'num_conv_layers': num_conv_layers,
                    'fc_hidden_dim': fc_hidden_dim,
                    'num_fc_layers': num_fc_layers,
                    'lr': lr,
                    'wd': wd,
                    'batch_size': batch_size,
                    'iterations_completed': iterations_completed,
                    'avg_loss': avg_loss
                }
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_model_dir = "../../../pretrained_models/gnn_baseline_final"
        os.makedirs(final_model_dir, exist_ok=True)

        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"
        final_model_path = f"{final_model_dir}/gnn_baseline_{process}_{corner_type}_{data_type}_{graph_mode}_iter{total_iterations}{arch_suffix}.pth"

        torch.save({
            'model_state_dict': gnn_baseline.model.state_dict(),
            'optimizer_state_dict': gnn_baseline.optimizer.state_dict(),
            'norm_stats': norm_stats,
            'task_norm_stats': gnn_baseline.task_norm_stats,
            'config': {
                'process_type': process,
                'corner_type': corner_type,
                'data_type': data_type,
                'graph_mode': graph_mode,
                'conv_hidden_dim': conv_hidden_dim,
                'num_conv_layers': num_conv_layers,
                'fc_hidden_dim': fc_hidden_dim,
                'num_fc_layers': num_fc_layers,
                'lr': lr,
                'wd': wd,
                'batch_size': batch_size,
                'total_iterations': total_iterations
            }
        }, final_model_path)

        print(f"\nSaved final model: {final_model_path}")

        # Cleanup model-specific data
        del gnn_baseline.task_norm_stats
        del gnn_baseline.optimizer
        del gnn_baseline.model
        del gnn_baseline

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        arch_time = time.time() - arch_start_time
        print(f"\nArchitecture {idx}/{num_combinations} completed in {arch_time/3600:.2f} hours")

        trained_models.append({
            'path': final_model_path,
            'arch': (conv_hidden_dim, num_conv_layers, fc_hidden_dim, num_fc_layers),
            'time': arch_time
        })

    # Cleanup dataset
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print summary
    sweep_total_time = time.time() - sweep_start_time
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Process: {process}")
    print(f"Corner: {corner_type}")
    print(f"Total architectures trained: {len(trained_models)}")
    print(f"Total time: {sweep_total_time/3600:.2f} hours")
    print(f"\nTrained models:")
    for i, model_info in enumerate(trained_models, 1):
        conv_h, conv_l, fc_h, fc_l = model_info['arch']
        print(f"  {i}. conv{conv_h}x{conv_l}_fc{fc_h}x{fc_l} - {model_info['time']/3600:.2f}h")
        print(f"     {model_info['path']}")

    return trained_models[0]['path'] if trained_models else None


def main():
    """Main function"""
    args = parse_arguments()

    # Validate arguments
    if not args.run_all and (not args.process or not args.corner):
        print("Error: --process and --corner are required unless --run_all is specified")
        print("Use --help for usage information")
        return

    # GPU is set at the top of file before torch import
    print(f"Using GPU: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")

    if args.run_all:
        # Run all 12 combinations
        all_processes = ['RVT', 'LVT', 'SLVT', 'SRAM']
        all_corners = ['TT', 'FF', 'SS']

        total_configs = len(all_processes) * len(all_corners)
        completed = 0
        failed = []

        print(f"\n{'='*80}")
        print(f"RUNNING ALL {total_configs} CONFIGURATIONS (mmap Loading)")
        print(f"{'='*80}")
        print(f"Processes: {', '.join(all_processes)}")
        print(f"Corners: {', '.join(all_corners)}")
        print(f"Graph mode: {args.graph_mode}")
        print(f"Data type: {args.data_type}")

        overall_start = time.time()

        for process in all_processes:
            for corner in all_corners:
                config_num = completed + len(failed) + 1
                print(f"\n{'='*80}")
                print(f"CONFIG {config_num}/{total_configs}: {process}_{corner}")
                print(f"{'='*80}\n")

                try:
                    model_path = train_single_config(process, corner, args)
                    completed += 1
                    print(f"\nConfig {config_num}/{total_configs} completed: {process}_{corner}")
                except Exception as e:
                    failed.append(f"{process}_{corner}")
                    print(f"\nConfig {config_num}/{total_configs} failed: {process}_{corner}")
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

        overall_time = time.time() - overall_start

        print(f"\n{'='*80}")
        print(f"ALL CONFIGURATIONS COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {overall_time/3600:.2f} hours")
        print(f"Completed: {completed}/{total_configs}")
        if failed:
            print(f"Failed: {len(failed)}")
            for config in failed:
                print(f"  - {config}")
    else:
        # Run single configuration
        train_single_config(args.process, args.corner, args)


if __name__ == "__main__":
    main()
