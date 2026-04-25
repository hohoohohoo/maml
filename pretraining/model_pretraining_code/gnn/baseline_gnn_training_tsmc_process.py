#!/usr/bin/env python
"""
Baseline GNN Training for TSMC Process Dataset - Unified 3D Format

Uses unified datasets from build_gnn_dataset_tsmc_unified.py
Memory-mapped loading (mmap) for large datasets via torch.load(..., mmap=True).

Key Features:
- Node features: 11D (7 base + 4 process params: param_a, param_b, param_c, temperature)
- Memory-mapped .pth tensors for efficient loading
- Standard mini-batch training (NOT MAML)
- Per-task output normalization (on-the-fly)
- Adam optimizer with weight decay

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
import json

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

from gnn_maml import create_maml_gcn_model
from gnn_data_preprocessing_utils import normalize_node_features_safe


class TSMCProcessDataset(Dataset):
    """
    PyTorch Dataset for TSMC Process GNN training.
    Supports both unified 3D tensor format (.pth) and legacy numpy format.

    Unified format (train_cell_stage_aware.pth):
    - node_features: [num_libs, total_nodes, 11] - 3D tensor
    - outputs: [num_libs, num_tasks] - 2D tensor
    - node_slices: [num_tasks + 1] - cumulative indices

    Legacy format (numpy arrays):
    - train_cell_all_graph_data_{graph_mode}_node_features.npy
    - train_cell_all_graph_data_{graph_mode}_outputs.npy
    - train_cell_all_graph_data_{graph_mode}_slices.npy
    """

    def __init__(self, train_path, graph_mode='full_graph'):
        """
        Args:
            train_path: Path to train file (.pth for unified, directory for legacy)
            graph_mode: 'full_graph' or 'stage_aware'
        """
        self.train_path = train_path
        self.graph_mode = graph_mode

        # Data references
        self._data = None
        self._node_features = None
        self._outputs = None
        self._slices = None

        # Metadata
        self._cell_names = None
        self._delay_types = None
        self._output_names = None
        self._topology_cache = None
        self._norm_stats = None
        self._num_tasks = None
        self._num_libs = None
        self._is_unified_format = False

        # Load data
        self._load_data()

        print(f"TSMCProcessDataset initialized (11D features):")
        print(f"   Train path: {train_path}")
        print(f"   Format: {'unified' if self._is_unified_format else 'legacy'}")
        print(f"   Graph mode: {graph_mode}")
        print(f"   Tasks: {self._num_tasks}")
        print(f"   Libs: {self._num_libs}")
        print(f"   Node features shape: {self._node_features.shape}")
        print(f"   Topology cache: {len(self._topology_cache)} cells")

    def _load_data(self):
        """Load data - auto-detect format"""
        if self.train_path.endswith('.pth'):
            self._load_unified_format()
        else:
            self._load_legacy_format()

    def _load_unified_format(self):
        """Load unified 3D tensor format (.pth file)"""
        print(f"   Loading unified format (mmap): {self.train_path}")
        self._is_unified_format = True

        data = torch.load(self.train_path, weights_only=False, map_location='cpu', mmap=True)

        # Check format (accept both 'tensor' and 'unified_3d')
        data_format = data.get('format', 'legacy')
        if data_format not in ['tensor', 'unified_3d']:
            raise ValueError(f"Expected tensor or unified_3d format, got: {data_format}")

        self._node_features = data['node_features']
        self._outputs = data['outputs']
        self._slices = data['node_slices']
        self._cell_names = data.get('cell_names', [])
        self._delay_types = data.get('delay_types', [])
        self._output_names = data.get('output_names', [])
        self._num_libs = data['num_libs']
        self._num_tasks = data['num_tasks']
        self._norm_stats = data.get('norm_stats', None)

        # Load topology cache
        cache_path = data.get('cache_path', None)
        if cache_path:
            self._load_topology_cache(cache_path)

        self._data = data

    def _load_legacy_format(self):
        """Load legacy numpy array format (from directory)"""
        print(f"   Loading legacy format: {self.train_path}")
        self._is_unified_format = False

        dataset_dir = self.train_path
        data_type = 'cell'  # default

        prefix = f"train_{data_type}_all_graph_data_{self.graph_mode}"
        metadata_path = os.path.join(dataset_dir, f"{prefix}_metadata.pth")
        node_features_path = os.path.join(dataset_dir, f"{prefix}_node_features.npy")
        outputs_path = os.path.join(dataset_dir, f"{prefix}_outputs.npy")
        slices_path = os.path.join(dataset_dir, f"{prefix}_slices.npy")

        # Load metadata
        metadata = torch.load(metadata_path, weights_only=False, map_location='cpu')
        self._cell_names = metadata.get('cell_names', [])
        self._delay_types = metadata.get('delay_types', [])
        self._output_names = metadata.get('output_names', [])
        self._num_tasks = metadata.get('num_tasks', len(self._cell_names))
        self._num_libs = metadata.get('num_libs', 1)
        self._norm_stats = metadata.get('norm_stats', None)

        # Load numpy arrays
        self._node_features = np.load(node_features_path, mmap_mode='r')
        self._outputs = np.load(outputs_path, mmap_mode='r')
        self._slices = np.load(slices_path)

        # Load topology cache
        cache_path = metadata.get('cache_path', None)
        if cache_path:
            self._load_topology_cache(cache_path)

    def _load_topology_cache(self, cache_path):
        """Load topology cache"""
        # Handle path variations
        if cache_path.startswith('/mnt/home/'):
            cache_path = cache_path.replace('/mnt/home/', '/home/')

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
        """Get data for a specific task"""
        if task_idx >= self._num_tasks:
            raise IndexError(f"Task index {task_idx} out of range")

        cell_name = self._cell_names[task_idx] if self._cell_names else f'task_{task_idx}'
        delay_type = self._delay_types[task_idx] if self._delay_types else 'rise'
        output_name = self._output_names[task_idx] if self._output_names else ''

        # Get edge_index from topology cache
        edge_index = None
        if self._topology_cache and cell_name in self._topology_cache:
            cell_cache = self._topology_cache[cell_name]

            if self.graph_mode == 'stage_aware' and 'output_topologies' in cell_cache:
                if output_name in cell_cache['output_topologies']:
                    output_topo = cell_cache['output_topologies'][output_name]
                    if 'rise' in delay_type:
                        adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
                    else:
                        adjacency_matrix = output_topo['pull_down']['adjacency_matrix']
                    edge_index = adjacency_matrix.nonzero().t()
                else:
                    if 'edge_index' in cell_cache:
                        edge_index = cell_cache['edge_index']
                    elif 'adjacency_matrix' in cell_cache:
                        edge_index = cell_cache['adjacency_matrix'].nonzero().t()
            else:
                if 'edge_index' in cell_cache:
                    edge_index = cell_cache['edge_index']
                elif 'adjacency_matrix' in cell_cache:
                    edge_index = cell_cache['adjacency_matrix'].nonzero().t()

        # Get slice indices
        if isinstance(self._slices, torch.Tensor):
            node_start = self._slices[task_idx].item()
            node_end = self._slices[task_idx + 1].item() if task_idx + 1 < len(self._slices) else self._node_features.shape[1]
        else:
            node_start = int(self._slices[task_idx])
            node_end = int(self._slices[task_idx + 1])

        # Get outputs for all libs
        task_outputs = self._outputs[:, task_idx]

        # Build minimal_samples list
        minimal_samples = []
        for lib_idx in range(self._num_libs):
            # Handle both tensor and numpy array formats
            if isinstance(self._node_features, torch.Tensor):
                task_node_features = self._node_features[lib_idx, node_start:node_end, :].clone()
            else:
                task_node_features = torch.from_numpy(
                    self._node_features[lib_idx, node_start:node_end, :].copy()
                ).float()

            # Get output value
            if isinstance(task_outputs, torch.Tensor):
                output_val = task_outputs[lib_idx].item()
            else:
                output_val = float(task_outputs[lib_idx])

            sample = {
                'node_features': task_node_features,
                'edge_index': edge_index,
                'output': output_val,
                'cell_name': cell_name,
                'delay_type': delay_type,
                'output_name': output_name,
            }
            minimal_samples.append(sample)

        # Handle output list conversion
        if isinstance(task_outputs, torch.Tensor):
            outputs_list = task_outputs.tolist()
        else:
            outputs_list = task_outputs.tolist()

        return {
            'minimal_samples': minimal_samples,
            'outputs': outputs_list,
            'task_idx': task_idx,
            'cell_name': cell_name,
            'delay_type': delay_type,
            'output_name': output_name
        }


class GNN_Baseline_TSMCProcess:
    """
    GNN Baseline training with TSMC Process dataset (11D features).
    Standard mini-batch training (NOT MAML).
    """

    def __init__(self, model, lr=2e-3, wd=5e-3,
                 dataset=None, iteration=100000, batch_size=5,
                 loss_logging_config=None):
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

        # Task output normalization cache
        self.task_norm_stats = {}

        # Model and optimizer
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            print(f"Model moved to {device}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        # Loss logging configuration
        self.loss_logging_config = loss_logging_config or {}
        self.enable_loss_logging = self.loss_logging_config.get('enabled', False)
        self.loss_log_every = self.loss_logging_config.get('log_every', 1000)
        self.loss_log_dir = self.loss_logging_config.get('save_dir', None)
        self.iteration_loss_log = []  # List of (iteration, loss) tuples

        print(f"\nGNN Baseline TSMC Process Configuration:")
        print(f"  Cache type: {self.cache_type}")
        print(f"  Number of tasks: {self.num_tasks}")
        print(f"  Lib files per task: {self.lib_files_per_task}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Weight decay: {self.wd}")
        print(f"  Iterations: {self.iteration}")
        print(f"  Node features: 11D (with process params)")
        if self.enable_loss_logging:
            print(f"  Loss logging: enabled (every {self.loss_log_every} iterations)")

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

    def create_pyg_data_with_adj_matrix(self, minimal_sample):
        """Create PyTorch Geometric Data object"""
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

        outputs = self.normalize_outputs(raw_outputs, task_idx)
        return minimal_samples, outputs

    def loop(self, checkpoint_dir='checkpoints', start_iteration=0):
        """Training loop

        Args:
            checkpoint_dir: Directory for checkpoints
            start_iteration: Starting iteration number (for cumulative tracking when resuming)
        """
        running_loss = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

        for i in range(self.iteration):
            cumulative_iteration = start_iteration + i + 1

            if i % 1000 == 0:
                avg_loss = running_loss / max(1, i)
                print(f"Iteration {i}/{self.iteration}, Avg Loss: {avg_loss:.6f}")

            task_idx = random.randint(0, self.num_tasks - 1)
            minimal_samples, outputs = self.get_task_data(task_idx)

            total_samples = len(minimal_samples)
            if total_samples < self.batch_size:
                sample_indices = list(range(total_samples))
            else:
                sample_indices = random.sample(range(total_samples), self.batch_size)

            batch_data = []
            batch_outputs = []

            for idx in sample_indices:
                minimal_sample = minimal_samples[idx]
                output = outputs[idx]
                data = self.create_pyg_data_with_adj_matrix(minimal_sample)
                batch_data.append(data)
                batch_outputs.append(output)

            X = Batch.from_data_list(batch_data).to(device)
            y = torch.tensor(batch_outputs, dtype=torch.float32).to(device).view(-1, 1)

            self.optimizer.zero_grad()
            self.model.train()
            y_pred = self.model(X)

            the_loss = nn.functional.mse_loss(y_pred, y)
            the_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            current_loss = the_loss.item()
            running_loss += current_loss

            # Loss logging at specified intervals
            if self.enable_loss_logging and cumulative_iteration % self.loss_log_every == 0:
                self.iteration_loss_log.append({
                    'iteration': cumulative_iteration,
                    'loss': current_loss
                })

        return float(running_loss / self.iteration)

    def save_loss_log(self, save_path):
        """Save iteration loss log to JSON file

        Args:
            save_path: Path to save the loss log JSON file
        """
        if not self.iteration_loss_log:
            print("No loss log entries to save")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        log_data = {
            'config': {
                'lr': self.lr,
                'wd': self.wd,
                'batch_size': self.batch_size,
                'loss_log_every': self.loss_log_every
            },
            'loss_log': self.iteration_loss_log
        }

        with open(save_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"Loss log saved: {save_path} ({len(self.iteration_loss_log)} entries)")


# Default dataset path
DEFAULT_DATASET_DIR = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/GNN_dataset_TSMC"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Baseline GNN Training for TSMC Process Dataset (11D features)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single training run
  python baseline_gnn_training_tsmc_process.py --graph_mode full_graph

  # Architecture sweep
  python baseline_gnn_training_tsmc_process.py \\
      --conv_hidden_dim 32 64 128 \\
      --num_conv_layers 2 3

  # Stage-aware mode
  python baseline_gnn_training_tsmc_process.py --graph_mode stage_aware
"""
    )

    # Dataset path
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET_DIR,
                       help=f'Dataset directory (default: {DEFAULT_DATASET_DIR})')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-3,
                       help='Weight decay (default: 5e-3)')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Mini-batch size (default: 5)')

    # Model architecture
    parser.add_argument('--conv_hidden_dim', type=int, nargs='+', default=[32],
                       help='Conv hidden dim(s) (default: 32)')
    parser.add_argument('--num_conv_layers', type=int, nargs='+', default=[2],
                       help='Number of conv layers (default: 2)')
    parser.add_argument('--fc_hidden_dim', type=int, nargs='+', default=[64],
                       help='FC hidden dim(s) (default: 64)')
    parser.add_argument('--num_fc_layers', type=int, nargs='+', default=[2],
                       help='Number of FC layers (default: 2)')

    # Training configuration
    parser.add_argument('--total_iterations', type=int, default=100000,
                       help='Total iterations (default: 100000)')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size (default: 10000)')

    # GPU
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device ID (default: 0)')

    # Data type and graph mode
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str, default='full_graph',
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode (default: full_graph)')

    # Pooling method
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'output'],
                       help='Pooling method: mean (global mean pooling) or output (output node only) (default: mean)')

    # Voltage mode
    parser.add_argument('--voltage_mode', type=str, default='all_nodes',
                       choices=['all_nodes', 'vdd_only'],
                       help='Voltage mode: all_nodes (voltage on all nodes) or vdd_only (voltage only on VDD) (default: all_nodes)')

    # Related pin only
    parser.add_argument('--related_pin_only', action='store_true',
                       help='Use related_pin_only slew assignment (adds _relpin suffix to paths)')

    # Loss logging options
    parser.add_argument('--enable_loss_logging', action='store_true',
                       help='Enable training loss logging at specified intervals')
    parser.add_argument('--loss_log_every', type=int, default=1000,
                       help='Log training loss every N iterations (default: 1000)')
    parser.add_argument('--loss_log_dir', type=str, default=None,
                       help='Directory to save loss logs (default: loss_logs/)')

    return parser.parse_args()


def train_single_config(args):
    """Train a single configuration"""
    import itertools

    total_iterations = args.total_iterations
    chunk_size = args.chunk_size
    lr = args.lr
    wd = args.wd
    batch_size = args.batch_size
    data_type = args.data_type
    graph_mode = args.graph_mode
    dataset_dir = args.dataset_dir
    pooling = args.pooling
    voltage_mode = args.voltage_mode
    related_pin_only = args.related_pin_only

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
    print(f"# Baseline GNN Training - TSMC Process (11D features)")
    print(f"{'#'*80}")
    print(f"Dataset: {dataset_dir}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Pooling: {pooling}")
    print(f"Voltage mode: {voltage_mode}")
    print(f"Related pin only: {related_pin_only}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {wd}")
    print(f"Batch size: {batch_size}")
    print(f"Total iterations: {total_iterations}")
    print(f"Node features: 11D (7 base + 4 process params)")

    if is_sweep:
        print(f"\nArchitecture Sweep: {num_combinations} combinations")

    # Create dataset
    print("\nCreating dataset (mmap loading)...")
    data_load_start = time.time()

    # Build suffixes for voltage_mode and related_pin_only
    # Dataset files use "_vdd_only" (with underscore), checkpoints use "_vddonly"
    voltage_suffix_data = "_vdd_only" if voltage_mode == "vdd_only" else ""
    relpin_suffix_data = "_relpin" if related_pin_only else ""
    voltage_suffix_ckpt = "_vdd_only" if voltage_mode == "vdd_only" else ""
    relpin_suffix_ckpt = "_relpin" if related_pin_only else ""

    # Construct train file path - check both unified and legacy formats
    train_file_unified = f"train_{data_type}_{graph_mode}{voltage_suffix_data}{relpin_suffix_data}.pth"
    train_path_unified = os.path.join(dataset_dir, train_file_unified)

    if os.path.exists(train_path_unified):
        # Use unified format
        train_path = train_path_unified
        print(f"Using unified format: {train_path}")
    else:
        # Use legacy directory format
        train_path = dataset_dir
        print(f"Using legacy format: {train_path}")

    dataset = TSMCProcessDataset(
        train_path=train_path,
        graph_mode=graph_mode
    )

    data_load_time = time.time() - data_load_start
    print(f"Dataset initialized in {data_load_time:.2f} seconds")

    norm_stats = dataset.norm_stats

    # Checkpoint directories (with voltage_mode and related_pin_only suffixes)
    dir_suffix = f"{voltage_suffix_ckpt}{relpin_suffix_ckpt}"
    checkpoint_dir = f"../../../pretrained_models/gnn_baseline_tsmc_process_checkpoints{dir_suffix}"
    final_model_dir = f"../../../pretrained_models/gnn_baseline_tsmc_process_final{dir_suffix}"
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

        num_chunks = total_iterations // chunk_size

        # Build loss logging configuration
        loss_logging_config = {
            'enabled': args.enable_loss_logging,
            'log_every': args.loss_log_every,
            'save_dir': args.loss_log_dir
        }

        # Create model with 11D input features
        print("Creating GNN Baseline model (11D input)...")

        gnn_baseline = GNN_Baseline_TSMCProcess(
            model=create_maml_gcn_model(
                node_features=11,  # 11D features!
                pooling=pooling,
                output_dim=1,
                dropout=0.3,
                conv_hidden_dim=conv_hidden_dim,
                num_conv_layers=num_conv_layers,
                fc_hidden_dim=fc_hidden_dim,
                num_fc_layers=num_fc_layers
            ),
            lr=lr,
            wd=wd,
            dataset=dataset,
            iteration=chunk_size,
            batch_size=batch_size,
            loss_logging_config=loss_logging_config
        )

        # Training in chunks
        iterations_trained = 0
        for chunk in range(num_chunks):
            print(f"\nProcessing chunk {chunk+1}/{num_chunks}")
            chunk_start_time = time.time()

            # Calculate starting iteration for this chunk (for loss logging)
            chunk_start_iteration = iterations_trained

            avg_loss = gnn_baseline.loop(checkpoint_dir=checkpoint_dir, start_iteration=chunk_start_iteration)
            iterations_trained += chunk_size

            torch.cuda.synchronize()
            chunk_time = time.time() - chunk_start_time

            print(f"Chunk {chunk+1} completed in {chunk_time:.2f}s")
            print(f"Average loss: {avg_loss:.6f}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save checkpoint
            iterations_completed = (chunk + 1) * chunk_size
            arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"
            pool_suffix = f"_pool{pooling}" if pooling != 'mean' else ""

            checkpoint_path = f"{checkpoint_dir}/gnn_baseline_tsmc_process_{data_type}_{graph_mode}_iter{iterations_completed}{arch_suffix}{pool_suffix}.pth"

            torch.save({
                'model_state_dict': gnn_baseline.model.state_dict(),
                'optimizer_state_dict': gnn_baseline.optimizer.state_dict(),
                'norm_stats': norm_stats,
                'task_norm_stats': gnn_baseline.task_norm_stats,
                'config': {
                    'data_type': data_type,
                    'graph_mode': graph_mode,
                    'pooling': pooling,
                    'voltage_mode': voltage_mode,
                    'related_pin_only': related_pin_only,
                    'conv_hidden_dim': conv_hidden_dim,
                    'num_conv_layers': num_conv_layers,
                    'fc_hidden_dim': fc_hidden_dim,
                    'num_fc_layers': num_fc_layers,
                    'lr': lr,
                    'wd': wd,
                    'batch_size': batch_size,
                    'iterations_completed': iterations_completed,
                    'avg_loss': avg_loss,
                    'node_features': 11
                }
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save final model
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"
        pool_suffix = f"_pool{pooling}" if pooling != 'mean' else ""
        final_model_path = f"{final_model_dir}/gnn_baseline_tsmc_process_{data_type}_{graph_mode}_iter{total_iterations}{arch_suffix}{pool_suffix}.pth"

        torch.save({
            'model_state_dict': gnn_baseline.model.state_dict(),
            'optimizer_state_dict': gnn_baseline.optimizer.state_dict(),
            'norm_stats': norm_stats,
            'task_norm_stats': gnn_baseline.task_norm_stats,
            'config': {
                'data_type': data_type,
                'graph_mode': graph_mode,
                'pooling': pooling,
                'voltage_mode': voltage_mode,
                'related_pin_only': related_pin_only,
                'conv_hidden_dim': conv_hidden_dim,
                'num_conv_layers': num_conv_layers,
                'fc_hidden_dim': fc_hidden_dim,
                'num_fc_layers': num_fc_layers,
                'lr': lr,
                'wd': wd,
                'batch_size': batch_size,
                'total_iterations': total_iterations,
                'node_features': 11
            }
        }, final_model_path)

        print(f"\nSaved final model: {final_model_path}")

        # Save loss log if enabled
        if args.enable_loss_logging and gnn_baseline.iteration_loss_log:
            loss_log_dir = args.loss_log_dir or f"../../../pretrained_models/loss_logs_baseline{dir_suffix}"
            os.makedirs(loss_log_dir, exist_ok=True)
            loss_log_filename = f"loss_log_baseline_{data_type}_{graph_mode}_iter{total_iterations}{arch_suffix}{pool_suffix}.json"
            loss_log_path = os.path.join(loss_log_dir, loss_log_filename)
            gnn_baseline.save_loss_log(loss_log_path)

        # Cleanup
        del gnn_baseline.task_norm_stats
        del gnn_baseline.iteration_loss_log
        del gnn_baseline.optimizer
        del gnn_baseline.model
        del gnn_baseline

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
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Total architectures trained: {len(trained_models)}")
    print(f"Total time: {sweep_total_time/3600:.2f} hours")
    print(f"\nTrained models:")
    for i, model_info in enumerate(trained_models, 1):
        conv_h, conv_l, fc_h, fc_l = model_info['arch']
        print(f"  {i}. conv{conv_h}x{conv_l}_fc{fc_h}x{fc_l} - {model_info['time']/3600:.2f}h")
        print(f"     {model_info['path']}")

    return trained_models[0]['path'] if trained_models else None


def main():
    args = parse_arguments()
    print(f"Using GPU: {args.gpu}")
    train_single_config(args)


if __name__ == "__main__":
    main()
