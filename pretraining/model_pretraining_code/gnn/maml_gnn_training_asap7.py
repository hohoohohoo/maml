#!/usr/bin/env python
"""
MAML GNN Training with Unified Dataset Format - Lazy Loading Version

Uses pre-split and pre-processed datasets from split_gnn_dataset.py
No preprocessing needed - data is already filtered and norm_stats are pre-computed.
Lazy loading: data is loaded on-demand to reduce memory usage.

Key Features:
- Loads unified train file lazily (train_cell_full_graph.pth)
- Topology reconstruction from cache on-the-fly
- Full graph: cell_name -> topology lookup
- Stage-aware: cell_name + output_name + delay_type -> topology lookup
- Per-task output normalization (on-the-fly)
- K=5 support set sampling
- Inner loop adaptation with fast weights
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


def functional_forward(model, x, fast_weights):
    """Perform forward pass with fast weights using functional approach"""
    original_params = []
    param_iter = iter(fast_weights)

    for param in model.parameters():
        original_params.append(param.data.clone())
        try:
            fast_weight = next(param_iter)
            if fast_weight is not None:
                param.data = fast_weight
        except StopIteration:
            break

    output = model(x)

    for param, original in zip(model.parameters(), original_params):
        param.data = original

    return output


class UnifiedMAMLTaskDataset(Dataset):
    """
    PyTorch Dataset for MAML task data with true mmap loading.
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

        print(f"UnifiedMAMLTaskDataset initialized (tensor format, mmap):")
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
        """Get norm_stats"""
        return self._norm_stats

    @property
    def num_libs(self):
        """Get number of lib files"""
        return self._num_libs

    def __len__(self):
        return self._num_tasks

    def __getitem__(self, task_idx):
        """
        Get data for a specific task.

        Returns:
            dict: {
                'minimal_samples': list of minimal sample dicts (one per lib file),
                'outputs': list of output values,
                'task_idx': original task index,
                'cell_name': cell name string
            }
        """
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

    def get_task_batch(self, task_indices):
        """Get data for multiple tasks at once"""
        return [self[idx] for idx in task_indices]


class GNNUnifiedLazyMAML:
    """
    GNN MAML with unified dataset and lazy loading.
    Uses UnifiedMAMLTaskDataset for memory-efficient data access.
    """

    def __init__(self, model, inner_lr, meta_lr, K=5, inner_steps=1,
                 dataset=None, tasks_per_meta_batch=16):
        """
        Args:
            model: GNN model
            inner_lr: Inner loop learning rate
            meta_lr: Meta learning rate
            K: Support set size
            inner_steps: Number of inner loop steps
            dataset: UnifiedMAMLTaskDataset instance
            tasks_per_meta_batch: Number of tasks per meta-batch
        """
        # Dataset
        self.dataset = dataset
        self.topology_cache = dataset.topology_cache if dataset else None
        self.cache_type = dataset.graph_mode if dataset else 'full_graph'
        self.norm_stats = dataset.norm_stats if dataset else None

        self.num_tasks = len(dataset) if dataset else 0

        # Task output normalization (computed on-the-fly)
        self.task_norm_stats = {}

        # Model and optimizer
        self.model = model
        self.weights = list(model.parameters())
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # Hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # Metrics
        self.plot_every = 10
        self.print_every = 200
        self.meta_losses = []

        if torch.cuda.is_available():
            self.model = self.model.to(device)
            print(f"Model moved to {device}")

        print(f"\nGNN Unified Lazy MAML Configuration:")
        print(f"  Cache type: {self.cache_type}")
        print(f"  Tasks: {self.num_tasks}")
        print(f"  Libs per task: {dataset.num_libs if dataset else 'N/A'}")
        print(f"  Support set size (K): {self.K}")
        print(f"  Inner steps: {self.inner_steps}")
        print(f"  Tasks per meta batch: {self.tasks_per_meta_batch}")
        print(f"  Memory efficient: Lazy loading enabled")

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

    def inner_loop_single_task(self, task_idx):
        """Inner loop for single task with lazy loading"""
        temp_weights = [w.clone() for w in self.weights]

        # Get task data from dataset (lazy load)
        task_data = self.dataset[task_idx]
        minimal_samples = task_data['minimal_samples']
        raw_outputs = task_data['outputs']

        # Normalize outputs on-the-fly
        outputs = self.normalize_outputs(raw_outputs, task_idx)

        # Sample K samples for support set
        total_libs = len(minimal_samples)
        if total_libs < self.K:
            support_indices = list(range(total_libs))
        else:
            support_indices = random.sample(range(total_libs), self.K)

        support_samples = [minimal_samples[i] for i in support_indices]
        support_outputs = [outputs[i] for i in support_indices]

        for step in range(self.inner_steps):
            batch_data = []
            for minimal_sample in support_samples:
                data = self.create_pyg_data_with_adj_matrix(minimal_sample)
                batch_data.append(data)

            if not batch_data:
                return torch.tensor(0.0, requires_grad=True).to(device)

            X = Batch.from_data_list(batch_data).to(device)
            y = torch.tensor(support_outputs, dtype=torch.float32).to(device).view(-1, 1)

            with torch.amp.autocast('cuda'):
                predictions = functional_forward(self.model, X, temp_weights)
                loss = self.criterion(predictions, y + 1e-6) / self.K

            grad = torch.autograd.grad(loss, temp_weights, create_graph=True, allow_unused=True)
            grad = [torch.clamp(g, -1.0, 1.0) if g is not None else g for g in grad]
            temp_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(temp_weights, grad)]

        # Query set
        if total_libs > self.K:
            remaining_indices = [i for i in range(total_libs) if i not in support_indices]
            if remaining_indices:
                query_indices = random.sample(remaining_indices, min(self.K, len(remaining_indices)))
            else:
                query_indices = support_indices
        else:
            query_indices = support_indices

        query_samples = [minimal_samples[i] for i in query_indices]
        query_outputs = [outputs[i] for i in query_indices]

        batch_data = []
        for minimal_sample in query_samples:
            data = self.create_pyg_data_with_adj_matrix(minimal_sample)
            batch_data.append(data)

        X = Batch.from_data_list(batch_data).to(device)
        y = torch.tensor(query_outputs, dtype=torch.float32).to(device).view(-1, 1)

        with torch.amp.autocast('cuda'):
            predictions = functional_forward(self.model, X, temp_weights)
            loss = self.criterion(predictions, y + 1e-6) / len(query_indices)

        return loss

    def main_loop_sequential(self, num_iterations):
        """Sequential MAML training loop with lazy loading"""
        print(f"\nStarting Lazy MAML Training (Sequential)")
        print(f"   Total iterations: {num_iterations}")

        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):
            meta_losses = []

            for _ in range(self.tasks_per_meta_batch):
                task_idx = random.randint(0, self.num_tasks - 1)
                try:
                    loss = self.inner_loop_single_task(task_idx)
                    meta_losses.append(loss)
                except Exception as e:
                    print(f"Task {task_idx} failed: {e}")
                    continue

            if not meta_losses:
                continue

            meta_loss = sum(meta_losses) / len(meta_losses)

            if torch.isnan(meta_loss) or torch.isinf(meta_loss):
                print(f"WARNING: NaN/Inf detected at iteration {iteration}")
                continue

            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            meta_grads = [torch.clamp(g, -1.0, 1.0) for g in meta_grads]

            has_nan_grad = False
            for i, (w, g) in enumerate(zip(self.weights, meta_grads)):
                if torch.isnan(g).any() or torch.isinf(g).any():
                    has_nan_grad = True
                w.grad = g

            if has_nan_grad:
                self.meta_optimiser.zero_grad()
                continue

            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()

            epoch_loss += meta_loss.item()
            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.print_every:.6f}")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0


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
        description='MAML GNN Training with Unified Dataset (Lazy Loading)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single configuration
  python maml_gnn_training_unified.py --process LVT --corner FF --graph_mode full_graph

  # Architecture sweep
  python maml_gnn_training_unified.py --process LVT --corner FF \\
      --conv_hidden_dim 64 128 256 \\
      --num_conv_layers 2 3 4

  # Run all 12 process-corner combinations
  python maml_gnn_training_unified.py --run_all --graph_mode full_graph
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
    parser.add_argument('--innerdiv', type=int, default=10,
                       help='Inner learning rate divisor (default: 10)')
    parser.add_argument('--meta_lr', type=float, default=0.0001,
                       help='Meta learning rate (default: 0.0001)')
    parser.add_argument('--inner_steps', type=int, default=1,
                       help='Number of inner loop steps (default: 1)')

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
    parser.add_argument('--K', type=int, default=5,
                       help='Support set size (default: 5)')
    parser.add_argument('--tasks_per_meta_batch', type=int, default=16,
                       help='Tasks per meta batch (default: 16)')

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
    """Train MAML for a single process-corner configuration"""
    import itertools

    total_iterations = args.total_iterations
    chunk_size = args.chunk_size
    inner_steps = args.inner_steps
    innerdiv = args.innerdiv
    meta_lr = args.meta_lr
    K = args.K
    tasks_per_meta_batch = args.tasks_per_meta_batch
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
    print(f"# MAML GNN Training with Unified Dataset (mmap Loading)")
    print(f"{'#'*80}")
    print(f"Process: {process}")
    print(f"Corner: {corner_type}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Inner div: {innerdiv}")
    print(f"Meta LR: {meta_lr}")
    print(f"Inner steps: {inner_steps}")
    print(f"K: {K}")
    print(f"Tasks per meta batch: {tasks_per_meta_batch}")
    print(f"Total iterations: {total_iterations}")

    if is_sweep:
        print(f"\nArchitecture Sweep: {num_combinations} combinations")

    # Get unified train file path
    train_path = get_unified_train_path(process, corner_type, data_type, graph_mode)
    print(f"\nUnified train file: {train_path}")

    # Create lazy loading dataset (data loaded on first access)
    print("\nCreating lazy loading dataset...")
    data_load_start = time.time()

    dataset = UnifiedMAMLTaskDataset(
        unified_train_path=train_path,
        graph_mode=graph_mode
    )

    data_load_time = time.time() - data_load_start
    print(f"Dataset initialized in {data_load_time:.2f} seconds")

    # Get norm_stats from dataset
    norm_stats = dataset.norm_stats

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

        # Calculate inner_lr
        base_lr = 0.001
        inner_lr = base_lr / innerdiv

        num_chunks = total_iterations // chunk_size

        # Create GNN MAML model with lazy loading
        print("Creating GNN Lazy MAML model...")

        gnn_maml = GNNUnifiedLazyMAML(
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
            dataset=dataset,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            inner_steps=inner_steps,
            K=K,
            tasks_per_meta_batch=tasks_per_meta_batch
        )

        # Training in chunks
        for chunk in range(num_chunks):
            print(f"\nProcessing chunk {chunk+1}/{num_chunks}")
            chunk_start_time = time.time()

            try:
                gnn_maml.main_loop_sequential(num_iterations=chunk_size)
            except Exception as e:
                print(f"Training failed: {e}")
                print("Reducing learning rates and retrying...")
                gnn_maml.inner_lr *= 0.5
                gnn_maml.meta_lr *= 0.5
                gnn_maml.main_loop_sequential(num_iterations=chunk_size // 2)

            torch.cuda.synchronize()
            chunk_end_time = time.time()

            chunk_time = chunk_end_time - chunk_start_time
            print(f"Chunk {chunk+1} completed in {chunk_time:.2f}s")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

            # Save checkpoint
            checkpoint_dir = "../../../pretrained_models/gnn_maml_checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            iterations_completed = (chunk + 1) * chunk_size

            arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

            checkpoint_path = f"{checkpoint_dir}/gnn_maml_{process}_{corner_type}_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{iterations_completed}_inner{inner_steps}{arch_suffix}.pth"

            torch.save({
                'model_state_dict': gnn_maml.model.state_dict(),
                'norm_stats': norm_stats,
                'task_norm_stats': gnn_maml.task_norm_stats,
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
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_model_dir = "../../../pretrained_models/gnn_maml_final"
        os.makedirs(final_model_dir, exist_ok=True)

        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

        final_model_path = f"{final_model_dir}/gnn_maml_{process}_{corner_type}_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{total_iterations}_inner{inner_steps}{arch_suffix}.pth"

        torch.save({
            'model_state_dict': gnn_maml.model.state_dict(),
            'norm_stats': norm_stats,
            'task_norm_stats': gnn_maml.task_norm_stats,
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

        print(f"\nSaved final model: {final_model_path}")

        # Cleanup model-specific data
        del gnn_maml.task_norm_stats
        del gnn_maml.weights
        del gnn_maml.meta_optimiser
        del gnn_maml.model
        del gnn_maml

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
        all_processes = ['LVT', 'RVT', 'SLVT', 'SRAM']
        all_corners = ['FF' , 'TT' , 'SS']

        total_configs = len(all_processes) * len(all_corners)
        completed = 0
        failed = []

        print(f"\n{'='*80}")
        print(f"RUNNING ALL {total_configs} CONFIGURATIONS (Lazy Loading)")
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
