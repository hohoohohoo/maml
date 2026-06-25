#!/usr/bin/env python
"""
MAML GAT Training for TSMC Process Dataset - Unified 3D Format

Uses unified datasets from build_gnn_dataset_tsmc_unified.py
Memory-mapped loading (mmap) for large datasets via torch.load(..., mmap=True).

Key Features:
- GAT (Graph Attention Network) with multi-head attention
- Node features: 11D (7 base + 4 process params: param_a, param_b, param_c, temperature)
- Memory-mapped .pth tensors for efficient loading
- MAML meta-learning with inner/outer loop
- K=5 support set sampling
- Per-task output normalization (on-the-fly)

Dataset Structure (unified 3D format):
- train_{data_type}_{graph_mode}.pth
  - node_features: [num_libs, total_nodes, 11] torch.Tensor
  - outputs: [num_libs, num_tasks] torch.Tensor
  - node_slices: [num_tasks + 1] torch.Tensor
  - cell_names, delay_types, output_names, norm_stats, etc.
  - format: 'unified_3d'
"""

import os
import sys
import glob
import re

# Parse GPU argument before importing torch
def get_gpu_from_args():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return '0'

os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_from_args()

import torch
from torch import optim
import torch.nn as nn
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model_code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from gnn_maml import create_maml_gat_model
from gnn_data_preprocessing_utils import normalize_node_features_safe


def extract_iteration_from_gat_filename(filename):
    """Extract iteration number from GAT model filename"""
    basename = os.path.basename(filename)
    # Pattern: iter{number}
    match = re.search(r'iter(\d+)', basename)
    if match:
        return int(match.group(1))
    return 0


def find_pretrained_gat_model(model_dir, data_type, graph_mode, innerdiv, meta,
                               inner_steps, conv_hidden_dim, num_conv_layers,
                               fc_hidden_dim, num_fc_layers, heads, pooling='mean'):
    """
    Find the latest pretrained GAT model matching the given parameters.

    Returns:
        tuple: (model_path, iteration) or (None, 0) if not found
    """
    if not os.path.exists(model_dir):
        return None, 0

    # Build expected filename pattern
    pool_suffix = f"_pool{pooling}" if pooling != 'mean' else ""
    arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}_heads{heads}{pool_suffix}"

    # Search pattern
    pattern = f"gat_maml_tsmc_process_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{meta}_iter*_inner{inner_steps}{arch_suffix}.pth"

    search_path = os.path.join(model_dir, pattern)
    matching_files = glob.glob(search_path)

    if not matching_files:
        print(f"No matching pretrained models found with pattern: {pattern}")
        return None, 0

    # Find the one with highest iteration
    best_file = None
    best_iteration = 0

    for f in matching_files:
        iteration = extract_iteration_from_gat_filename(f)
        if iteration > best_iteration:
            best_iteration = iteration
            best_file = f

    if best_file:
        print(f"Found pretrained model: {os.path.basename(best_file)}")
        print(f"   Iteration: {best_iteration}")
        return best_file, best_iteration

    return None, 0


def load_pretrained_gat_model(model, checkpoint_path, device):
    """
    Load pretrained GAT model from checkpoint.

    Returns:
        bool: True if loaded successfully, False otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's a direct state dict
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        print(f"Loaded pretrained model from: {os.path.basename(checkpoint_path)}")

        # Print config if available
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            if 'iterations_completed' in config:
                print(f"   Iterations completed: {config['iterations_completed']}")

        return True
    except Exception as e:
        print(f"Failed to load pretrained model: {e}")
        return False


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


class TSMCProcessMAMLDataset(Dataset):
    """
    PyTorch Dataset for TSMC Process MAML training with mmap loading.
    Uses tensor-based storage format (same as ASAP7 unified format).

    Data format (3D tensor):
    - node_features: [num_libs, total_nodes, num_features] - mmap tensor
    - outputs: [num_libs, num_tasks] - mmap tensor
    - node_slices: [num_tasks + 1] - cumulative node indices
    - cell_names: [num_tasks] - for topology cache lookup

    Each task has num_libs samples (different lib files from same directory).
    """

    def __init__(self, train_path, graph_mode='stage_aware', cache_path_override=None):
        """
        Args:
            train_path: Path to train file (train_cell_stage_aware.pth)
            graph_mode: 'full_graph' or 'stage_aware'
            cache_path_override: Override topology cache path (None = use path in dataset file)
        """
        self.train_path = train_path
        self.graph_mode = graph_mode
        self.cache_path_override = cache_path_override

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

        # Load data with mmap
        self._load_data()

        print(f"TSMCProcessMAMLDataset initialized (tensor format, mmap):")
        print(f"   Train file: {train_path}")
        print(f"   Tasks: {self._num_tasks}")
        print(f"   Libs: {self._num_libs}")
        print(f"   Graph mode: {graph_mode}")
        print(f"   node_features: {self._node_features.shape}")
        print(f"   outputs: {self._outputs.shape}")
        print(f"   topology_cache: {len(self._topology_cache)} cells")

    def _load_data(self):
        """Load data using memory mapping (mmap) and topology cache"""
        print(f"   Loading with mmap=True (memory-mapped tensors)")
        data = torch.load(self.train_path, weights_only=False, map_location='cpu', mmap=True)

        # Check format (accept both 'tensor' and 'unified_3d')
        data_format = data.get('format', 'legacy')
        if data_format not in ['tensor', 'unified_3d']:
            raise ValueError(f"Expected tensor or unified_3d format, got: {data_format}")

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

        # Load topology cache (use override if provided, otherwise from data file)
        if self.cache_path_override:
            cache_path = self.cache_path_override
            print(f"   Using cache_path override: {cache_path}")
        else:
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

        # If cache file still not found, try to find matching file based on graph_mode
        if not os.path.exists(cache_path):
            cache_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache"
            cache_filename = os.path.basename(cache_path)

            # Try to find cache file matching current graph_mode
            if self.graph_mode == 'full_graph':
                # Try full_graph version first
                if 'stage_aware_' in cache_filename:
                    alt_filename = cache_filename.replace('stage_aware_', 'full_graph_')
                    alt_path = os.path.join(cache_dir, alt_filename)
                    if os.path.exists(alt_path):
                        cache_path = alt_path
                        print(f"   Using full_graph cache: {alt_filename}")
            elif self.graph_mode == 'stage_aware':
                # Try stage_aware version first
                if 'full_graph_' in cache_filename:
                    alt_filename = cache_filename.replace('full_graph_', 'stage_aware_')
                    alt_path = os.path.join(cache_dir, alt_filename)
                    if os.path.exists(alt_path):
                        cache_path = alt_path
                        print(f"   Using stage_aware cache: {alt_filename}")

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Topology cache not found: {cache_path}")

        print(f"   Loading topology cache: {cache_path}")
        self._topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu')
        print(f"   Loaded {len(self._topology_cache)} cells")

    @property
    def topology_cache(self):
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
        node_end = self._node_slices[task_idx + 1].item() if task_idx + 1 < len(self._node_slices) else self._node_features.shape[1]

        # Get outputs for all libs
        task_outputs = self._outputs[:, task_idx]

        # Build minimal_samples list for compatibility
        minimal_samples = []
        for lib_idx in range(self._num_libs):
            # Extract node features for this lib and task
            # node_features shape: [num_libs, total_nodes, num_features]
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


class GAT_MAML_TSMCProcess:
    """
    GAT MAML training with TSMC Process dataset (11D features).
    Meta-learning with inner/outer loop optimization.
    """

    def __init__(self, model, inner_lr, meta_lr, K=5, inner_steps=1,
                 dataset=None, tasks_per_meta_batch=16):
        # Dataset
        self.dataset = dataset
        self.topology_cache = dataset.topology_cache if dataset else None
        self.cache_type = dataset.graph_mode if dataset else 'stage_aware'
        self.norm_stats = dataset.norm_stats if dataset else None

        self.num_tasks = len(dataset) if dataset else 0

        # Task output normalization cache
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

        print(f"\nGAT MAML TSMC Process Configuration:")
        print(f"  Cache type: {self.cache_type}")
        print(f"  Tasks: {self.num_tasks}")
        print(f"  Libs per task: {dataset.num_libs if dataset else 'N/A'}")
        print(f"  Support set size (K): {self.K}")
        print(f"  Inner steps: {self.inner_steps}")
        print(f"  Inner LR: {self.inner_lr}")
        print(f"  Meta LR: {self.meta_lr}")
        print(f"  Tasks per meta batch: {self.tasks_per_meta_batch}")
        print(f"  Node features: 11D (with process params)")

    def normalize_node_features(self, node_features):
        """Normalize node features using saved statistics"""
        if self.norm_stats is None:
            return node_features

        normalized, _ = normalize_node_features_safe(
            node_features,
            norm_stats=self.norm_stats.get('node_features', self.norm_stats)
        )
        return normalized

    def normalize_outputs(self, outputs, task_idx):
        """Normalize outputs for a task"""
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
        """Load adjacency matrix from topology cache"""
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

    def get_output_node_idx(self, minimal_sample):
        """Get output node index from topology cache"""
        cell_name = minimal_sample['cell_name']
        cell_cache = self.topology_cache[cell_name]

        if self.cache_type == 'stage_aware':
            output_name = minimal_sample['output_name']
            delay_type = minimal_sample['delay_type']

            if output_name in cell_cache['output_topologies']:
                output_topo = cell_cache['output_topologies'][output_name]
                if 'rise' in delay_type:
                    all_nodes = output_topo['pull_up'].get('all_nodes', [])
                else:
                    all_nodes = output_topo['pull_down'].get('all_nodes', [])

                # Find output node index in node list
                if output_name in all_nodes:
                    return all_nodes.index(output_name)

        # Fallback for full_graph or if not found
        all_nodes = cell_cache.get('all_nodes', [])
        output_nodes = cell_cache.get('output_nodes', [])
        if output_nodes and output_nodes[0] in all_nodes:
            return all_nodes.index(output_nodes[0])

        # Default fallback (VDD=0, VSS=1, output=2)
        return 2

    def create_pyg_data_with_adj_matrix(self, minimal_sample):
        """Create PyTorch Geometric Data object"""
        node_features = minimal_sample['node_features']
        adjacency_matrix = self.get_adjacency_matrix_from_cache(minimal_sample)
        normalized_features = self.normalize_node_features(node_features)
        edge_index = adjacency_matrix.nonzero().t()

        # Get dynamic output node index
        output_node_idx = self.get_output_node_idx(minimal_sample)

        data = Data(
            x=normalized_features,
            edge_index=edge_index,
            output_node_idx=torch.tensor(output_node_idx, dtype=torch.long)
        )
        return data

    def inner_loop_single_task(self, task_idx):
        """Inner loop for single task"""
        temp_weights = [w.clone() for w in self.weights]

        # Get task data
        task_data = self.dataset[task_idx]
        minimal_samples = task_data['minimal_samples']
        raw_outputs = task_data['outputs']

        # Normalize outputs
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
        """Sequential MAML training loop"""
        print(f"\nStarting TSMC Process GAT MAML Training")
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
                print(f"WARNING: NaN/Inf at iteration {iteration}")
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


# Default dataset path
DEFAULT_DATASET_DIR = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_GNN_unified"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='MAML GAT Training for TSMC Process Dataset (11D features)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single training run
  python maml_gat_training_tsmc_process.py --graph_mode stage_aware

  # Architecture sweep
  python maml_gat_training_tsmc_process.py \\
      --conv_hidden_dim 32 64 128 \\
      --num_conv_layers 2 3

  # Custom hyperparameters with attention heads
  python maml_gat_training_tsmc_process.py --innerdiv 10 --meta_lr 0.0001 --K 5 --heads 4
"""
    )

    # Dataset path
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET_DIR,
                       help=f'Dataset directory (default: {DEFAULT_DATASET_DIR})')

    # Training hyperparameters
    parser.add_argument('--innerdiv', type=int, default=10,
                       help='Inner LR divisor (default: 10, inner_lr = 0.001/innerdiv)')
    parser.add_argument('--meta_lr', type=float, default=0.0001,
                       help='Meta learning rate (default: 0.0001)')
    parser.add_argument('--inner_steps', type=int, default=1,
                       help='Inner loop steps (default: 1)')
    parser.add_argument('--K', type=int, default=5,
                       help='Support set size (default: 5)')
    parser.add_argument('--tasks_per_meta_batch', type=int, default=16,
                       help='Tasks per meta batch (default: 16)')

    # Model architecture
    parser.add_argument('--conv_hidden_dim', type=int, nargs='+', default=[32],
                       help='Conv hidden dim(s) (default: 32)')
    parser.add_argument('--num_conv_layers', type=int, nargs='+', default=[2],
                       help='Number of conv layers (default: 2)')
    parser.add_argument('--fc_hidden_dim', type=int, nargs='+', default=[64],
                       help='FC hidden dim(s) (default: 64)')
    parser.add_argument('--num_fc_layers', type=int, nargs='+', default=[2],
                       help='Number of FC layers (2 or 3) (default: 2)')
    parser.add_argument('--heads', type=int, default=4,
                       help='Number of attention heads for GAT (default: 4)')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'add', 'output'],
                       help='Pooling method: mean, max, add, or output (output-node-only) (default: mean)')
    parser.add_argument('--output_node_idx', type=int, default=2,
                       help='Fallback index of output node (default: 2). Normally auto-detected from topology cache.')

    # Training configuration
    parser.add_argument('--total_iterations', type=int, default=300000,
                       help='Total iterations (default: 300000)')
    parser.add_argument('--chunk_size', type=int, default=100000,
                       help='Chunk size (default: 100000)')

    # GPU
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID (default: 0)')

    # Data type and graph mode
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode (default: stage_aware)')
    parser.add_argument('--voltage_mode', type=str, default='all_nodes',
                       choices=['all_nodes', 'vdd_only'],
                       help='Voltage feature mode: all_nodes (default) or vdd_only')
    parser.add_argument('--cache_path', type=str, default=None,
                       help='Override topology cache path (default: use cache_path stored in dataset file)')
    parser.add_argument('--related_pin_only', action='store_true',
                       help='Use related_pin_only dataset (adds _relpin suffix). Input slew assigned only to triggering input port.')

    # Resume training options
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to specific checkpoint file to resume training from')
    parser.add_argument('--auto_resume', action='store_true',
                       help='Automatically find and resume from latest checkpoint')
    parser.add_argument('--additional_iterations', type=int, default=None,
                       help='Additional iterations to train when resuming (overrides total_iterations)')

    return parser.parse_args()


def train_single_config(args):
    """Train a single configuration"""
    import itertools

    total_iterations = args.total_iterations
    chunk_size = args.chunk_size
    innerdiv = args.innerdiv
    meta_lr = args.meta_lr
    inner_steps = args.inner_steps
    K = args.K
    tasks_per_meta_batch = args.tasks_per_meta_batch
    data_type = args.data_type
    graph_mode = args.graph_mode
    voltage_mode = args.voltage_mode
    dataset_dir = args.dataset_dir
    cache_path = args.cache_path
    heads = args.heads

    # Calculate inner_lr
    base_lr = 0.001
    inner_lr = base_lr / innerdiv

    # Architecture combinations
    arch_combinations = list(itertools.product(
        args.conv_hidden_dim,
        args.num_conv_layers,
        args.fc_hidden_dim,
        args.num_fc_layers
    ))

    num_combinations = len(arch_combinations)
    is_sweep = num_combinations > 1

    print(f"\n{'#'*80}")
    print(f"# MAML GAT Training - TSMC Process (11D features)")
    print(f"{'#'*80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Voltage mode: {voltage_mode}")
    print(f"Inner div: {innerdiv} (inner_lr = {inner_lr})")
    print(f"Meta LR: {meta_lr}")
    print(f"Inner steps: {inner_steps}")
    print(f"K: {K}")
    print(f"Tasks per meta batch: {tasks_per_meta_batch}")
    print(f"Total iterations: {total_iterations}")
    print(f"Attention heads: {heads}")
    print(f"Node features: 11D (7 base + 4 process params)")
    if args.related_pin_only:
        print(f"Related pin only: enabled (slew assigned to triggering input only)")

    if is_sweep:
        print(f"\nArchitecture Sweep: {num_combinations} combinations")

    # Create dataset
    print("\nCreating dataset (mmap loading)...")
    data_load_start = time.time()

    # Construct train file path (voltage_mode suffix only for non-default)
    voltage_suffix = f"_{voltage_mode}" if voltage_mode != 'all_nodes' else ""
    slew_suffix = "_relpin" if args.related_pin_only else ""
    train_file = f"train_{data_type}_{graph_mode}{voltage_suffix}{slew_suffix}.pth"
    train_path = os.path.join(dataset_dir, train_file)

    # Cache suffix for gate control edges
    cache_suffix = ""
    if cache_path and "gatectrl" in cache_path:
        cache_suffix = "_gatectrl"

    if not os.path.exists(train_path):
        print(f"Train file not found: {train_path}")
        print(f"Looking in dataset_dir: {dataset_dir}")
        raise FileNotFoundError(f"Train file not found: {train_path}")

    dataset = TSMCProcessMAMLDataset(
        train_path=train_path,
        graph_mode=graph_mode,
        cache_path_override=cache_path
    )

    data_load_time = time.time() - data_load_start
    print(f"Dataset initialized in {data_load_time:.2f} seconds")

    norm_stats = dataset.norm_stats

    # Checkpoint directories (include voltage_mode suffix for vdd_only, cache_suffix for gatectrl, slew_suffix for relpin)
    checkpoint_dir = f"../../../pretrained_models/gat_maml_tsmc_process_checkpoints{voltage_suffix}{cache_suffix}{slew_suffix}"
    final_model_dir = f"../../../pretrained_models/gat_maml_tsmc_process_final{voltage_suffix}{cache_suffix}{slew_suffix}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    # Train each architecture
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
        print(f"  Attention heads: {heads}")

        # Create model with 11D input features
        pooling_mode = args.pooling
        output_node_idx = args.output_node_idx
        print(f"Creating GAT MAML model (11D input, pooling={pooling_mode}, heads={heads})...")

        gat_model = create_maml_gat_model(
            node_features=11,  # 11D features!
            pooling=pooling_mode,
            output_dim=1,
            dropout=0.0,
            conv_hidden_dim=conv_hidden_dim,
            num_conv_layers=num_conv_layers,
            fc_hidden_dim=fc_hidden_dim,
            num_fc_layers=num_fc_layers,
            heads=heads,
            output_node_idx=output_node_idx
        )

        # Resume training logic
        start_iteration = 0
        loaded_model = False

        if args.resume:
            # Load from specific checkpoint file
            if os.path.exists(args.resume):
                start_iteration = extract_iteration_from_gat_filename(args.resume)
                loaded_model = load_pretrained_gat_model(gat_model, args.resume, device)
            else:
                print(f"Specified checkpoint not found: {args.resume}")
                print("   Starting from scratch...")
        elif args.auto_resume:
            # Auto-find latest checkpoint
            # Search in both checkpoint and final model directories
            search_dirs = [checkpoint_dir, final_model_dir]
            best_path, best_iter = None, 0

            for search_dir in search_dirs:
                model_path, iteration = find_pretrained_gat_model(
                    search_dir, data_type, graph_mode, innerdiv,
                    tasks_per_meta_batch, inner_steps,
                    conv_hidden_dim, num_conv_layers,
                    fc_hidden_dim, num_fc_layers, heads, pooling_mode
                )
                if iteration > best_iter:
                    best_path, best_iter = model_path, iteration

            if best_path:
                start_iteration = best_iter
                loaded_model = load_pretrained_gat_model(gat_model, best_path, device)

        # Calculate training iterations
        if loaded_model:
            if args.additional_iterations:
                additional_iterations = args.additional_iterations
            else:
                additional_iterations = total_iterations
            target_iteration = start_iteration + additional_iterations
            print(f"Resuming training from iteration: {start_iteration}")
            print(f"   Additional iterations to train: {additional_iterations}")
            print(f"   Target final iteration: {target_iteration}")
        else:
            additional_iterations = total_iterations
            target_iteration = total_iterations
            print(f"Starting training from scratch")
            print(f"   Total iterations to train: {total_iterations}")

        num_chunks = additional_iterations // chunk_size
        if additional_iterations % chunk_size > 0:
            num_chunks += 1  # Add partial chunk if needed

        gat_maml = GAT_MAML_TSMCProcess(
            model=gat_model,
            inner_lr=inner_lr,
            meta_lr=meta_lr,
            K=K,
            inner_steps=inner_steps,
            dataset=dataset,
            tasks_per_meta_batch=tasks_per_meta_batch
        )

        # Training in chunks
        iterations_trained = 0
        for chunk in range(num_chunks):
            # Calculate iterations for this chunk
            remaining = additional_iterations - iterations_trained
            chunk_iters = min(chunk_size, remaining)
            if chunk_iters <= 0:
                break
            print(f"\nProcessing chunk {chunk+1}/{num_chunks} ({chunk_iters} iterations)")
            chunk_start_time = time.time()

            try:
                gat_maml.main_loop_sequential(num_iterations=chunk_iters)
            except Exception as e:
                print(f"Training failed: {e}")
                print("Reducing learning rates and retrying...")
                gat_maml.inner_lr *= 0.5
                gat_maml.meta_lr *= 0.5
                gat_maml.main_loop_sequential(num_iterations=chunk_iters // 2)

            iterations_trained += chunk_iters
            torch.cuda.synchronize()
            chunk_time = time.time() - chunk_start_time

            print(f"Chunk {chunk+1} completed in {chunk_time:.2f}s")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save checkpoint with cumulative iteration count
            cumulative_iterations = start_iteration + iterations_trained
            pool_suffix = f"_pool{pooling_mode}" if pooling_mode != 'mean' else ""
            arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}_heads{heads}{pool_suffix}"

            checkpoint_path = f"{checkpoint_dir}/gat_maml_tsmc_process_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{cumulative_iterations}_inner{inner_steps}{arch_suffix}.pth"

            torch.save({
                'model_state_dict': gat_maml.model.state_dict(),
                'norm_stats': norm_stats,
                'task_norm_stats': gat_maml.task_norm_stats,
                'config': {
                    'model_type': 'GAT',
                    'data_type': data_type,
                    'graph_mode': graph_mode,
                    'conv_hidden_dim': conv_hidden_dim,
                    'num_conv_layers': num_conv_layers,
                    'fc_hidden_dim': fc_hidden_dim,
                    'num_fc_layers': num_fc_layers,
                    'heads': heads,
                    'pooling': pooling_mode,
                    'output_node_idx': output_node_idx,
                    'inner_steps': inner_steps,
                    'inner_lr': inner_lr,
                    'meta_lr': meta_lr,
                    'K': K,
                    'innerdiv': innerdiv,
                    'iterations_completed': cumulative_iterations,
                    'start_iteration': start_iteration,
                    'additional_iterations_trained': iterations_trained,
                    'meta_losses': gat_maml.meta_losses,
                    'node_features': 11
                }
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            print(f"   Cumulative iteration: {cumulative_iterations}")

        # Save final model with cumulative iteration count
        final_iteration = start_iteration + iterations_trained
        pool_suffix = f"_pool{pooling_mode}" if pooling_mode != 'mean' else ""
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}_heads{heads}{pool_suffix}"
        final_model_path = f"{final_model_dir}/gat_maml_tsmc_process_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{final_iteration}_inner{inner_steps}{arch_suffix}.pth"

        torch.save({
            'model_state_dict': gat_maml.model.state_dict(),
            'norm_stats': norm_stats,
            'task_norm_stats': gat_maml.task_norm_stats,
            'config': {
                'model_type': 'GAT',
                'data_type': data_type,
                'graph_mode': graph_mode,
                'conv_hidden_dim': conv_hidden_dim,
                'num_conv_layers': num_conv_layers,
                'fc_hidden_dim': fc_hidden_dim,
                'num_fc_layers': num_fc_layers,
                'heads': heads,
                'pooling': pooling_mode,
                'output_node_idx': output_node_idx,
                'inner_steps': inner_steps,
                'inner_lr': inner_lr,
                'innerdiv': innerdiv,
                'meta_lr': meta_lr,
                'K': K,
                'total_iterations': final_iteration,
                'start_iteration': start_iteration,
                'additional_iterations_trained': iterations_trained,
                'meta_losses': gat_maml.meta_losses,
                'node_features': 11
            }
        }, final_model_path)

        print(f"\nTraining complete for architecture {idx}!")
        print(f"   Final model: {final_model_path}")
        print(f"   Final cumulative iteration: {final_iteration}")

        # Cleanup
        del gat_maml.task_norm_stats
        del gat_maml.weights
        del gat_maml.meta_optimiser
        del gat_maml.model
        del gat_maml

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    # Summary
    sweep_total_time = time.time() - sweep_start_time
    print(f"\n{'='*80}")
    print(f"GAT TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Total architectures trained: {len(trained_models)}")
    print(f"Total time: {sweep_total_time/3600:.2f} hours")
    print(f"\nTrained models:")
    for i, model_info in enumerate(trained_models, 1):
        conv_h, conv_l, fc_h, fc_l = model_info['arch']
        print(f"  {i}. conv{conv_h}x{conv_l}_fc{fc_h}x{fc_l}_heads{heads} - {model_info['time']/3600:.2f}h")
        print(f"     {model_info['path']}")

    return trained_models[0]['path'] if trained_models else None


def main():
    args = parse_arguments()
    print(f"Using GPU: {args.gpu}")
    train_single_config(args)


if __name__ == "__main__":
    main()
