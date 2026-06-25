#!/usr/bin/env python
"""
Baseline GNN Training with Cached Topology - PROCESS-AWARE VERSION
Standard mini-batch training approach (NOT MAML) with cached topology
Uses 11D node features (7 base + 4 process parameters)

Process parameters added to base features:
- param_a, param_b (PMOS/NMOS), param_c (PMOS/NMOS), temperature

Key Features:
- Mini-batch training similar to MLP_pretraining in networks.py
- Minimal dataset: only node_features + metadata stored
- Topology reconstruction from cache on-the-fly
- Full graph: cell_name -> topology lookup
- Stage-aware: cell_name + output_name + delay_type -> topology lookup
- Adam optimizer with weight decay
- Simple MSE loss
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
import gc

# GPU optimization settings
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..','model_code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from gnn_maml import (
    MAML_GNN_Model,
    create_maml_gcn_model
)
from gnn_data_preprocessing_utils import (
    preprocess_gnn_minimal_data,
    normalize_node_features_safe,
    normalize_task_outputs
)


class GNN_Baseline_Pretraining_Process:
    """
    GNN training wrapper with task-based mini-batch sampling (similar to MAML data organization)
    Process-aware version with 11D node features
    Uses standard SGD training (no MAML inner/outer loop)

    Data organization:
    - Dataset organized into tasks (same input condition across lib files)
    - Each task has multiple samples (different lib files)
    - Each iteration: randomly select 1 task, then randomly sample batch_size samples from that task

    Args:
        lr: Learning rate (default: 2e-3)
        wd: Weight decay (default: 5e-3)
        minimal_data_per_file: Minimal dataset [num_libs][num_samples]
        topology_cache: Pre-computed topology cache
        cache_type: 'full_graph' or 'stage_aware'
        norm_stats: Normalization statistics
        iteration: Number of training iterations (default: 100000)
        hidden_size: Hidden layer size (default: 128)
        batch_size: Number of samples per task to use in mini-batch (default: 5)
        normalized_outputs: Pre-computed normalized outputs (optional, if None will compute)
        task_norm_stats: Pre-computed task normalization stats (optional, if None will compute)
    """
    def __init__(self, model, lr=2e-3, wd=5e-3,
                 minimal_data_per_file=None, topology_cache=None,
                 cache_type='full_graph', norm_stats=None,
                 iteration=100000, hidden_size=128, batch_size=5,
                 normalized_outputs=None, task_norm_stats=None):

        self.lr = lr
        self.wd = wd
        self.iteration = iteration
        self.batch_size = batch_size

        # Dataset (minimal format)
        self.minimal_data_per_file = minimal_data_per_file  # [lib_files][samples]
        self.topology_cache = topology_cache
        self.cache_type = cache_type  # 'full_graph' or 'stage_aware'
        self.norm_stats = norm_stats

        # Build task data (organize by input conditions)
        # If pre-computed normalized_outputs provided, use them
        if normalized_outputs is not None and task_norm_stats is not None:
            print(f"Using pre-computed normalized outputs and task stats...")
            self.normalized_outputs = normalized_outputs
            self.task_norm_stats = task_norm_stats
            self.num_tasks = normalized_outputs.shape[0]
            self.lib_files_per_task = normalized_outputs.shape[1]
            print(f"   Loaded {self.num_tasks} tasks x {self.lib_files_per_task} lib files per task")
        else:
            self.num_tasks = self._build_task_data()

        # Model and optimizer
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            print(f"Model moved to {device}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        print(f"\nGNN Baseline Pretraining Configuration (Process-aware):")
        print(f"   Node features: 11D (7 base + 4 process)")
        print(f"   Cache type: {self.cache_type}")
        print(f"   Number of tasks: {self.num_tasks}")
        print(f"   Lib files per task: {self.lib_files_per_task}")
        print(f"   Batch size (samples per task): {self.batch_size}")
        print(f"   Learning rate: {self.lr}")
        print(f"   Weight decay: {self.wd}")
        print(f"   Iterations: {self.iteration}")

    def _build_task_data(self):
        """
        Organize data into tasks (similar to MAML)
        Each task has multiple samples (different lib files with same input condition)

        Normalizes outputs PER TASK (not globally) to match MAML_topology_pretraining behavior

        Returns:
            num_tasks: Number of tasks
        """
        print(f"Organizing data into tasks...")

        num_libs = len(self.minimal_data_per_file)
        num_tasks = len(self.minimal_data_per_file[0])

        self.num_tasks = num_tasks
        self.lib_files_per_task = num_libs

        # Pre-compute normalized outputs: [num_tasks, num_libs]
        # Build stacked outputs first
        stacked_outputs = torch.zeros(num_tasks, num_libs, dtype=torch.float32)
        for task_idx in range(num_tasks):
            for lib_idx in range(num_libs):
                sample = self.minimal_data_per_file[lib_idx][task_idx]
                stacked_outputs[task_idx, lib_idx] = sample['output']

        # Use safe normalization from utils (with NaN/Inf protection)
        self.normalized_outputs, self.task_norm_stats = normalize_task_outputs(
            stacked_outputs,
            min_std_threshold=1e-8
        )

        print(f"   Organized into {num_tasks} tasks x {num_libs} lib files per task")

        return num_tasks

    def normalize_node_features(self, node_features):
        """Normalize node features using saved statistics (with NaN/Inf protection)"""
        if self.norm_stats is None:
            return node_features

        # Use safe normalization from utils
        normalized, _ = normalize_node_features_safe(
            node_features,
            norm_stats=self.norm_stats['node_features']
        )

        return normalized

    def get_adjacency_matrix_from_cache(self, minimal_sample):
        """
        Load pre-computed adjacency matrix from topology cache based on sample metadata

        For full_graph: Uses cell_name only
        For stage_aware: Uses cell_name + output_name + delay_type to select pull-up/pull-down path

        Args:
            minimal_sample: dict with {node_features, cell_name, output_name, delay_type, output}

        Returns:
            torch.Tensor: Pre-computed adjacency matrix from cache
        """
        cell_name = minimal_sample['cell_name']

        if cell_name not in self.topology_cache:
            raise ValueError(f"Cell {cell_name} not found in topology cache")

        cell_cache = self.topology_cache[cell_name]

        if self.cache_type == 'stage_aware':
            # Stage-aware: lookup by cell_name + output_name + delay_type
            output_name = minimal_sample['output_name']
            delay_type = minimal_sample['delay_type']

            if output_name not in cell_cache['output_topologies']:
                raise ValueError(f"Output {output_name} not found for cell {cell_name}")

            output_topo = cell_cache['output_topologies'][output_name]

            # Select pull-up (rise) or pull-down (fall) path
            if 'rise' in delay_type:
                adjacency_matrix = output_topo['pull_up']['adjacency_matrix']
            else:
                adjacency_matrix = output_topo['pull_down']['adjacency_matrix']

        else:  # full_graph
            # Full graph: lookup by cell_name only
            adjacency_matrix = cell_cache['adjacency_matrix']

        return adjacency_matrix

    def create_pyg_data_with_adj_matrix(self, minimal_sample):
        """
        Create PyTorch Geometric Data object from minimal sample
        Loads pre-computed adjacency matrix from cache and applies graph convolution

        Args:
            minimal_sample: dict with {node_features, cell_name, output_name, delay_type, output}

        Returns:
            PyG Data object with aggregated features
        """
        # Get node features from minimal sample (11D for process-aware)
        node_features = minimal_sample['node_features']

        # Load pre-computed adjacency matrix from cache
        adjacency_matrix = self.get_adjacency_matrix_from_cache(minimal_sample)

        # Apply normalization to node features
        normalized_features = self.normalize_node_features(node_features)

        # Create edge_index from adjacency matrix for GCN convolution
        # GCNConv will perform A x X internally
        edge_index = adjacency_matrix.nonzero().t()

        data = Data(
            x=normalized_features,
            edge_index=edge_index
        )

        return data

    def get_task_data(self, task_id):
        """
        Get data for a specific task (same input condition across lib files)

        Args:
            task_id: Task index

        Returns:
            minimal_samples: List of minimal samples for this task
            outputs: List of normalized outputs for this task
        """
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} out of range (max: {self.num_tasks-1})")

        minimal_samples = []
        outputs = []

        # Collect minimal samples from all lib files for this task
        for lib_idx in range(self.lib_files_per_task):
            if task_id < len(self.minimal_data_per_file[lib_idx]):
                minimal_sample = self.minimal_data_per_file[lib_idx][task_id]

                # Use pre-normalized outputs
                output = self.normalized_outputs[task_id, lib_idx].item()

                minimal_samples.append(minimal_sample)
                outputs.append(output)

        return minimal_samples, outputs

    def loop(self, checkpoint_dir='checkpoints'):
        """
        Training loop with task-based mini-batch sampling

        Each iteration:
        1. Select a random task (x tasks total, each with multiple lib files)
        2. Within that task, randomly sample batch_size samples
        3. Train on these samples using standard SGD (no MAML inner/outer loop)

        Args:
            checkpoint_dir: Directory to save checkpoints (default: 'checkpoints')

        Returns:
            Average training loss
        """
        running_loss = 0.0

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        for i in range(self.iteration):
            if i % 1000 == 0:
                avg_loss = running_loss / max(1, i)
                print(f"Iteration {i}/{self.iteration}, Avg Loss: {avg_loss:.6f}")

            # Save checkpoint every 10000 iterations
            if i > 0 and i % 10000 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{i}.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'iteration': i,
                    'loss': running_loss / i,
                    'task_norm_stats': self.task_norm_stats,  # Per-task normalization stats
                    'norm_stats': self.norm_stats  # Input normalization stats
                }, checkpoint_path)
                print(f"Checkpoint saved at iteration {i}: {checkpoint_path}")

            # Select random task
            task_idx = random.randint(0, self.num_tasks - 1)

            # Get task data
            minimal_samples, outputs = self.get_task_data(task_idx)

            # Sample random indices from this task
            total_samples = len(minimal_samples)
            if total_samples < self.batch_size:
                sample_indices = list(range(total_samples))
            else:
                sample_indices = random.sample(range(total_samples), self.batch_size)

            # Get mini-batch data from selected samples
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

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            self.model.train()
            y_pred = self.model(X)

            # Compute loss
            the_loss = nn.functional.mse_loss(y_pred, y)

            # Backward pass and optimization
            the_loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += the_loss.item()

        return float(running_loss / self.iteration)


def load_cached_gnn_data_for_baseline_process(data_type='cell', graph_mode='stage_aware'):
    """
    Load process-aware GNN datasets from dataset_temp_process directory
    Excludes test datasets, uses all other datasets
    NO train/test split - loads full datasets

    Args:
        data_type: 'cell' or 'transition'
        graph_mode: 'stage_aware' or 'full_graph'

    Returns:
        minimal_data_per_file: Merged minimal data
        topology_cache: Shared topology cache
        norm_stats: Normalization statistics
    """

    base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp_process"

    print(f"\nLoading process-aware datasets from {base_path}")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")
    print(f"   Note: Loading FULL datasets (no train/test split)")

    # Find all non-test dataset directories
    all_items = os.listdir(base_path)
    matching_folders = []

    data_filename = f"{data_type}_all_graph_data_{graph_mode}.pth"

    for item in sorted(all_items):
        item_path = os.path.join(base_path, item)

        # Skip if not directory
        if not os.path.isdir(item_path):
            continue

        # Skip test datasets
        if 'test' in item.lower():
            continue

        # Check if data file exists in graph_data directory
        data_file_path = f"{base_path}/{item}/graph_data/{data_filename}"
        if os.path.exists(data_file_path):
            matching_folders.append((item, data_file_path))
        else:
            print(f"   Warning: Data file not found: {data_file_path}")

    if not matching_folders:
        raise ValueError(f"No data found in {base_path}")

    print(f"   Found {len(matching_folders)} non-test datasets")

    # Load and merge all matching datasets
    all_minimal_data_per_file = []
    topology_cache = None

    for folder, data_file_path in matching_folders:
        print(f"   Loading: {folder}")

        # Load full dataset
        print(f"     Loading full dataset from: {data_file_path}")
        full_data = torch.load(data_file_path, weights_only=False, map_location='cpu')
        minimal_data_per_file = full_data['minimal_data_per_file']
        cache_path = full_data.get('cache_path', None)

        print(f"     Dataset: {len(minimal_data_per_file[0])} samples, {len(minimal_data_per_file)} lib files")

        # Convert cache_path to absolute path if needed
        if cache_path and not os.path.isabs(cache_path):
            if not os.path.exists(cache_path):
                cache_filename = os.path.basename(cache_path)
                cache_file = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"
                if os.path.exists(cache_file):
                    cache_path = cache_file
                    print(f"     Resolved cache_path to: {cache_path}")
                else:
                    print(f"     Warning: Could not find cache: {cache_path}")

        # Load topology cache (only once)
        if topology_cache is None and cache_path:
            print(f"     Loading topology cache from: {cache_path}")
            topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu')
            print(f"     Loaded topology cache for {len(topology_cache)} cells")

        if not all_minimal_data_per_file:
            # First dataset - initialize
            all_minimal_data_per_file = [[] for _ in range(len(minimal_data_per_file))]

        # Merge minimal data: use ALL samples (no train/test split)
        for lib_idx, lib_samples in enumerate(minimal_data_per_file):
            all_minimal_data_per_file[lib_idx].extend(lib_samples)

    print(f"   Merged data:")
    print(f"     Total tasks: {len(all_minimal_data_per_file[0])} (input conditions)")
    print(f"     Lib files per task: {len(all_minimal_data_per_file)} (process variations)")

    # Apply preprocessing pipeline (filtering + normalization with NaN/Inf detection)
    print(f"\nApplying data preprocessing pipeline...")
    preprocessed_data, norm_stats, preprocessing_stats = preprocess_gnn_minimal_data(
        all_minimal_data_per_file,
        min_std_threshold=1e-6,
        enable_filtering=True,
        verbose=True
    )

    print(f"\nPreprocessing Summary:")
    print(f"   Valid tasks after filtering: {preprocessing_stats['filtering']['valid_tasks']}")
    print(f"   Filter ratio: {preprocessing_stats['filtering']['filter_ratio']:.1f}%")

    return preprocessed_data, topology_cache, norm_stats


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Baseline GNN Training with Cached Topology (Process-Aware)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline_gnn_training_cached_with_process.py --graph_mode stage_aware --data_type cell
  python baseline_gnn_training_cached_with_process.py --graph_mode full_graph --data_type transition --lr 0.001
"""
    )

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-3,
                       help='Weight decay (default: 5e-3)')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Mini-batch size (default: 5)')

    # Model architecture parameters
    parser.add_argument('--conv_hidden_dim', type=int, default=128,
                       help='Convolution layer hidden dimension (default: 128)')
    parser.add_argument('--num_conv_layers', type=int, default=3,
                       help='Number of GCN convolutional layers (default: 3)')
    parser.add_argument('--fc_hidden_dim', type=int, default=40,
                       help='FC layer hidden dimension (default: 40)')
    parser.add_argument('--num_fc_layers', type=int, default=3, choices=[1, 2, 3],
                       help='Number of FC layers (default: 3)')

    # Training configuration
    parser.add_argument('--total_iterations', type=int, default=100000,
                       help='Total iterations (default: 100000)')
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size for checkpoint saving (default: 10000)')

    # GPU configuration
    parser.add_argument('--gpu', type=str, default='2',
                       help='GPU device ID (default: 2)')

    # Data type
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')

    # Graph mode
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode (default: stage_aware)')

    # Use merged dataset (much faster loading)
    parser.add_argument('--use_merged', action='store_true',
                       help='Use pre-merged dataset (recommended for faster loading)')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Extract parameters
    total_iterations = args.total_iterations
    chunk_size = args.chunk_size
    num_chunks = total_iterations // chunk_size
    lr = args.lr
    wd = args.wd
    batch_size = args.batch_size
    data_type = args.data_type
    graph_mode = args.graph_mode

    # Model architecture parameters
    conv_hidden_dim = args.conv_hidden_dim
    num_conv_layers = args.num_conv_layers
    fc_hidden_dim = args.fc_hidden_dim
    num_fc_layers = args.num_fc_layers

    start_time = time.time()

    print(f"\nBaseline GNN Training with Cached Topology (Process-Aware)")
    print(f"Configuration:")
    print(f"   Node features: 11D (7 base + 4 process)")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")
    print(f"   Learning rate: {lr}")
    print(f"   Weight decay: {wd}")
    print(f"   Batch size: {batch_size}")
    print(f"   Conv hidden dim: {conv_hidden_dim}")
    print(f"   Num conv layers: {num_conv_layers}")
    print(f"   FC hidden dim: {fc_hidden_dim}")
    print(f"   Num FC layers: {num_fc_layers}")
    print(f"   Total iterations: {total_iterations}")
    print(f"   GPU: {args.gpu}")

    # Load cached GNN data (process-aware datasets)
    print("Loading process-aware cached GNN data...")

    if args.use_merged:
        # Use pre-merged dataset (much faster!)
        from load_merged_process_data import load_merged_process_data
        print("   Using pre-merged dataset for faster loading...")
        minimal_data_per_file, topology_cache, _ = load_merged_process_data(
            data_type, graph_mode
        )

        # Apply preprocessing to compute norm_stats
        print("\nApplying data preprocessing pipeline...")
        from gnn_data_preprocessing_utils import preprocess_gnn_minimal_data
        minimal_data_per_file, norm_stats, preprocessing_stats = preprocess_gnn_minimal_data(
            minimal_data_per_file,
            min_std_threshold=1e-6,
            enable_filtering=True,
            verbose=True
        )
        print(f"\nPreprocessing Summary:")
        print(f"   Valid tasks after filtering: {preprocessing_stats['filtering']['valid_tasks']}")
        print(f"   Filter ratio: {preprocessing_stats['filtering']['filter_ratio']:.1f}%")

    else:
        # Load from individual directories (slower)
        print("   Loading from individual directories (slower)...")
        print("   Tip: Use --use_merged flag for faster loading after running merge_process_datasets.py")
        minimal_data_per_file, topology_cache, norm_stats = load_cached_gnn_data_for_baseline_process(
            data_type, graph_mode
        )

    # Create checkpoint directory
    checkpoint_dir = "../../../pretrained_models/gnn_baseline_checkpoints_process"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Pre-compute per-task normalization ONCE (before chunking)
    print("\nPre-computing per-task normalization (one-time)...")
    num_libs = len(minimal_data_per_file)
    num_tasks = len(minimal_data_per_file[0])

    normalized_outputs = torch.zeros(num_tasks, num_libs, dtype=torch.float32)
    task_norm_stats = {}

    print(f"Normalizing outputs for {num_tasks} tasks...")
    for task_idx in range(num_tasks):
        # Collect outputs for this task
        task_outputs = []
        for lib_idx in range(num_libs):
            sample = minimal_data_per_file[lib_idx][task_idx]
            task_outputs.append(sample['output'])

        # Calculate per-task statistics
        task_outputs_tensor = torch.tensor(task_outputs, dtype=torch.float32)
        task_mean = task_outputs_tensor.mean().item()
        task_std = task_outputs_tensor.std().item()

        # Store task normalization stats
        task_norm_stats[task_idx] = {
            'mean': task_mean,
            'std': task_std
        }

        # Normalize this task's outputs
        if task_std > 1e-8:
            normalized_outputs[task_idx] = (task_outputs_tensor - task_mean) / task_std
        else:
            normalized_outputs[task_idx] = task_outputs_tensor - task_mean

        # Progress indicator every 1000 tasks
        if (task_idx + 1) % 1000 == 0:
            print(f"   Processed {task_idx + 1}/{num_tasks} tasks")

    print(f"   Per-task normalization complete")
    print(f"   Organized into {num_tasks} tasks x {num_libs} lib files per task")

    # Create GNN baseline model (11D node features for process-aware version)
    print("Creating GNN baseline model...")
    gnn_baseline = GNN_Baseline_Pretraining_Process(
        model=create_maml_gcn_model(
            node_features=11,  # 11D: 7 base + 4 process parameters
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
        minimal_data_per_file=minimal_data_per_file,
        topology_cache=topology_cache,
        cache_type=graph_mode,
        norm_stats=norm_stats,
        iteration=chunk_size,  # Iterations per chunk
        hidden_size=conv_hidden_dim,
        batch_size=batch_size,
        normalized_outputs=normalized_outputs,  # Pass pre-computed
        task_norm_stats=task_norm_stats  # Pass pre-computed
    )

    # Training in chunks (reuse same model instance)
    for chunk in range(num_chunks):
        print(f"\nProcessing chunk {chunk+1}/{num_chunks}")
        chunk_start_time = time.time()

        # Train for this chunk (no model deletion/recreation)
        avg_loss = gnn_baseline.loop(checkpoint_dir=checkpoint_dir)

        # GPU synchronization and timing
        torch.cuda.synchronize()
        chunk_end_time = time.time()

        chunk_time = chunk_end_time - chunk_start_time
        print(f"Chunk {chunk+1} completed in {chunk_time:.2f}s")
        print(f"Average loss: {avg_loss:.6f}")

        # Memory cleanup after each chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Memory monitoring
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        # Save checkpoint for this chunk
        iterations_completed = (chunk + 1) * chunk_size
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"
        checkpoint_path = f"{checkpoint_dir}/gnn_baseline_process_{data_type}_{graph_mode}_iter{iterations_completed}{arch_suffix}.pth"

        torch.save({
            'model_state_dict': gnn_baseline.model.state_dict(),
            'optimizer_state_dict': gnn_baseline.optimizer.state_dict(),
            'norm_stats': norm_stats,
            'task_norm_stats': task_norm_stats,  # Use pre-computed task normalization stats
            'config': {
                'data_type': data_type,
                'graph_mode': graph_mode,
                'node_features': 11,  # Process-aware
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
    final_model_dir = "../../../pretrained_models/gnn_baseline_final_process"
    os.makedirs(final_model_dir, exist_ok=True)

    # Build final model path with architecture suffix
    arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"
    final_model_path = f"{final_model_dir}/gnn_baseline_process_{data_type}_{graph_mode}_iter{total_iterations}{arch_suffix}.pth"

    torch.save({
        'model_state_dict': gnn_baseline.model.state_dict(),
        'optimizer_state_dict': gnn_baseline.optimizer.state_dict(),
        'norm_stats': norm_stats,
        'task_norm_stats': task_norm_stats,
        'config': {
            'data_type': data_type,
            'graph_mode': graph_mode,
            'node_features': 11,  # Process-aware
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

    end_time = time.time()
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {(end_time - start_time)/3600:.2f} hours")
    print(f"Final model saved: {final_model_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
