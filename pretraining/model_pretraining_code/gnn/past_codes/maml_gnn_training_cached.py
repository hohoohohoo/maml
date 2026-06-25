#!/usr/bin/env python
"""
MAML GNN Training with Cached Topology - MINIMAL DATASET VERSION
Supports both full_graph and stage_aware modes with topology cache

Key Features:
- Minimal dataset: only node_features + metadata stored
- Topology reconstruction from cache on-the-fly
- Full graph: cell_name -> topology lookup
- Stage-aware: cell_name + output_name + delay_type -> topology lookup
- Per-task output normalization
- K=5 support set sampling
- Inner loop adaptation with fast weights
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

# GPU 최적화 설정
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
from torch.utils.data import Dataset


class MAMLTaskDataset(Dataset):
    """
    PyTorch Dataset for MAML task data with lazy loading.

    Instead of loading all data into memory, this dataset:
    1. Stores only index metadata (file paths, sample indices)
    2. Loads actual data from disk when __getitem__ is called
    3. Caches recently accessed data for efficiency

    This significantly reduces memory usage for large datasets.
    """

    def __init__(self, data_file_paths, sample_indices_list, topology_cache_path,
                 graph_mode='full_graph', cache_size=1000):
        """
        Args:
            data_file_paths: List of paths to data files (one per dataset folder)
            sample_indices_list: List of (folder_idx, sample_indices) for train samples
            topology_cache_path: Path to topology cache file
            graph_mode: 'full_graph' or 'stage_aware'
            cache_size: Number of recently accessed tasks to cache in memory
        """
        self.data_file_paths = data_file_paths
        self.sample_indices_list = sample_indices_list  # List of tuples: [(folder_idx, [sample_indices])]
        self.graph_mode = graph_mode
        self.cache_size = cache_size

        # Build task index: maps task_id -> (folder_idx, local_sample_idx)
        self.task_index = []
        for folder_idx, sample_indices in sample_indices_list:
            for local_idx, global_sample_idx in enumerate(sample_indices):
                self.task_index.append((folder_idx, global_sample_idx))

        self.num_tasks = len(self.task_index)

        # Lazy load topology cache (load once when first needed)
        self.topology_cache_path = topology_cache_path
        self._topology_cache = None

        # LRU cache for recently accessed data files
        self._data_cache = {}  # folder_idx -> loaded_data
        self._cache_order = []  # Track access order for LRU eviction

        # Store lib file count per folder (will be populated on first access)
        self._lib_counts = {}

        print(f"📦 MAMLTaskDataset initialized:")
        print(f"   Total tasks: {self.num_tasks}")
        print(f"   Data files: {len(data_file_paths)}")
        print(f"   Graph mode: {graph_mode}")
        print(f"   Cache size: {cache_size} tasks")

    @property
    def topology_cache(self):
        """Lazy load topology cache on first access"""
        if self._topology_cache is None:
            print(f"   Loading topology cache from: {self.topology_cache_path}")
            self._topology_cache = torch.load(self.topology_cache_path, weights_only=False, map_location='cpu')
            print(f"   ✓ Loaded topology cache for {len(self._topology_cache)} cells")
        return self._topology_cache

    def _load_data_file(self, folder_idx):
        """Load a data file with LRU caching"""
        if folder_idx in self._data_cache:
            # Move to end of cache order (most recently used)
            if folder_idx in self._cache_order:
                self._cache_order.remove(folder_idx)
            self._cache_order.append(folder_idx)
            return self._data_cache[folder_idx]

        # Load new file
        data_path = self.data_file_paths[folder_idx]
        data = torch.load(data_path, weights_only=False, map_location='cpu')

        # Cache eviction if needed
        while len(self._data_cache) >= max(1, self.cache_size // 100):  # Keep ~10 files cached
            if self._cache_order:
                oldest = self._cache_order.pop(0)
                if oldest in self._data_cache:
                    del self._data_cache[oldest]

        self._data_cache[folder_idx] = data
        self._cache_order.append(folder_idx)

        # Store lib count
        if folder_idx not in self._lib_counts:
            self._lib_counts[folder_idx] = len(data['minimal_data_per_file'])

        return data

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, task_idx):
        """
        Get data for a specific task.

        Returns:
            dict: {
                'minimal_samples': list of minimal sample dicts (one per lib file),
                'outputs': list of output values,
                'task_idx': original task index
            }
        """
        if task_idx >= self.num_tasks:
            raise IndexError(f"Task index {task_idx} out of range (max: {self.num_tasks - 1})")

        folder_idx, sample_idx = self.task_index[task_idx]

        # Load data file (with caching)
        data = self._load_data_file(folder_idx)
        minimal_data_per_file = data['minimal_data_per_file']

        # Collect samples from all lib files for this task
        minimal_samples = []
        outputs = []

        for lib_idx in range(len(minimal_data_per_file)):
            if sample_idx < len(minimal_data_per_file[lib_idx]):
                sample = minimal_data_per_file[lib_idx][sample_idx]
                minimal_samples.append(sample)
                outputs.append(sample['output'])

        return {
            'minimal_samples': minimal_samples,
            'outputs': outputs,
            'task_idx': task_idx
        }

    def get_task_batch(self, task_indices):
        """
        Get data for multiple tasks at once (batch loading).
        More efficient than calling __getitem__ multiple times.

        Args:
            task_indices: List of task indices

        Returns:
            List of task data dicts
        """
        # Group by folder for efficient loading
        folder_tasks = {}
        for task_idx in task_indices:
            folder_idx, sample_idx = self.task_index[task_idx]
            if folder_idx not in folder_tasks:
                folder_tasks[folder_idx] = []
            folder_tasks[folder_idx].append((task_idx, sample_idx))

        # Load and collect data
        results = {}
        for folder_idx, tasks in folder_tasks.items():
            data = self._load_data_file(folder_idx)
            minimal_data_per_file = data['minimal_data_per_file']

            for task_idx, sample_idx in tasks:
                minimal_samples = []
                outputs = []

                for lib_idx in range(len(minimal_data_per_file)):
                    if sample_idx < len(minimal_data_per_file[lib_idx]):
                        sample = minimal_data_per_file[lib_idx][sample_idx]
                        minimal_samples.append(sample)
                        outputs.append(sample['output'])

                results[task_idx] = {
                    'minimal_samples': minimal_samples,
                    'outputs': outputs,
                    'task_idx': task_idx
                }

        # Return in original order
        return [results[idx] for idx in task_indices]

    def get_num_libs(self, task_idx=0):
        """Get number of lib files for a task"""
        folder_idx, _ = self.task_index[task_idx]
        if folder_idx not in self._lib_counts:
            data = self._load_data_file(folder_idx)
            self._lib_counts[folder_idx] = len(data['minimal_data_per_file'])
        return self._lib_counts[folder_idx]


def functional_forward(model, x, fast_weights):
    """Perform forward pass with fast weights using functional approach"""
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

class GNNCachedMAML:
    """
    GNN MAML with cached topology - reconstructs graphs on-the-fly
    """

    def __init__(self, model, inner_lr, meta_lr, K=5, inner_steps=1,
                 minimal_data_per_file=None, topology_cache=None,
                 cache_type='full_graph', norm_stats=None,
                 tasks_per_meta_batch=16):

        # Dataset (minimal format)
        self.minimal_data_per_file = minimal_data_per_file  # [lib_files][samples]
        self.topology_cache = topology_cache
        self.cache_type = cache_type  # 'full_graph' or 'stage_aware'
        self.norm_stats = norm_stats

        # Build stacked outputs from minimal data
        self.stacked_outputs = self._build_stacked_outputs()

        self.num_tasks = self.stacked_outputs.shape[0]  # Number of input conditions
        self.lib_files_per_task = self.stacked_outputs.shape[1]  # Number of process variations

        # Pre-normalize all task outputs
        self.task_norm_stats = {}
        self.normalized_outputs = self.normalize_all_task_outputs(self.stacked_outputs)

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
            print(f"✅ Model moved to {device}")

        print(f"🎯 GNN Cached MAML Configuration:")
        print(f"   Cache type: {self.cache_type}")
        print(f"   Tasks: {self.num_tasks} (input conditions)")
        print(f"   Lib files per task: {self.lib_files_per_task}")
        print(f"   Support set size (K): {self.K}")
        print(f"   Inner steps: {self.inner_steps}")
        print(f"   Tasks per meta batch: {self.tasks_per_meta_batch}")

    def _build_stacked_outputs(self):
        """Build stacked_outputs tensor from minimal data"""
        print(f"📊 Building stacked_outputs from minimal data...")

        num_libs = len(self.minimal_data_per_file)
        num_samples = len(self.minimal_data_per_file[0])

        stacked_outputs = torch.zeros(num_samples, num_libs)

        for lib_idx in range(num_libs):
            for sample_idx in range(num_samples):
                sample = self.minimal_data_per_file[lib_idx][sample_idx]
                stacked_outputs[sample_idx, lib_idx] = sample['output']

        print(f"   ✓ Built stacked_outputs: {stacked_outputs.shape}")
        return stacked_outputs

    def normalize_all_task_outputs(self, stacked_outputs):
        """Pre-normalize all task outputs before training (with NaN/Inf protection)"""
        if stacked_outputs is None:
            return None

        # Use safe normalization from utils
        normalized, task_norm_stats = normalize_task_outputs(
            stacked_outputs,
            min_std_threshold=1e-8
        )

        # Update task_norm_stats
        self.task_norm_stats = task_norm_stats

        return normalized

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
        # Get node features from minimal sample
        node_features = minimal_sample['node_features']

        # Load pre-computed adjacency matrix from cache
        adjacency_matrix = self.get_adjacency_matrix_from_cache(minimal_sample)

        # Apply normalization to node features
        normalized_features = self.normalize_node_features(node_features)

        # Apply adjacency matrix multiplication (A × X) for graph convolution
        #aggregated_features = torch.matmul(adjacency_matrix, normalized_features)

        # Create edge_index from adjacency matrix for PyG compatibility
        edge_index = adjacency_matrix.nonzero().t()

        data = Data(
            x=normalized_features,
            edge_index=edge_index
        )

        return data

    def get_task_data(self, task_id):
        """
        Get data for a specific task (same input condition across lib files)
        Loads minimal samples and their corresponding adjacency matrices from cache
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

    def inner_loop_single_task(self, task_idx):
        """Inner loop for single task"""
        temp_weights = [w.clone() for w in self.weights]

        # Get task data (minimal samples + normalized outputs)
        minimal_samples, outputs = self.get_task_data(task_idx)

        # Sample K samples for support set
        total_libs = len(minimal_samples)
        if total_libs < self.K:
            support_indices = list(range(total_libs))
        else:
            support_indices = random.sample(range(total_libs), self.K)

        support_samples = [minimal_samples[i] for i in support_indices]
        support_outputs = [outputs[i] for i in support_indices]

        for step in range(self.inner_steps):
            # Convert minimal samples to PyG batch (loads adjacency matrix from cache)
            batch_data = []
            for minimal_sample in support_samples:
                data = self.create_pyg_data_with_adj_matrix(minimal_sample)
                batch_data.append(data)

            if not batch_data:
                return torch.tensor(0.0, requires_grad=True).to(device)

            X = Batch.from_data_list(batch_data).to(device)
            y = torch.tensor(support_outputs, dtype=torch.float32).to(device).view(-1, 1)

            # Mixed precision for stability
            with torch.cuda.amp.autocast():
                predictions = functional_forward(self.model, X, temp_weights)
                loss = self.criterion(predictions, y + 1e-6) / self.K

            grad = torch.autograd.grad(loss, temp_weights, create_graph=True, allow_unused=True)

            # Gradient clipping to prevent explosion
            grad = [torch.clamp(g, -1.0, 1.0) if g is not None else g for g in grad]
            temp_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(temp_weights, grad)]

        # Meta-update loss calculation (query set)
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

        # Convert query samples to PyG batch (loads adjacency matrix from cache)
        batch_data = []
        for minimal_sample in query_samples:
            data = self.create_pyg_data_with_adj_matrix(minimal_sample)
            batch_data.append(data)

        X = Batch.from_data_list(batch_data).to(device)
        y = torch.tensor(query_outputs, dtype=torch.float32).to(device).view(-1, 1)

        with torch.cuda.amp.autocast():
            predictions = functional_forward(self.model, X, temp_weights)
            loss = self.criterion(predictions, y + 1e-6) / len(query_indices)

        return loss

    def main_loop_optimized(self, num_iterations):
        """Main MAML loop - optimized version"""
        from concurrent.futures import ThreadPoolExecutor

        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):
            meta_losses = []

            # Parallel task processing
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

            # Check for NaN/Inf in meta_loss
            if torch.isnan(meta_loss) or torch.isinf(meta_loss):
                print(f"⚠️ WARNING: NaN/Inf detected in meta_loss at iteration {iteration}")
                print(f"   Meta losses: {[l.item() if hasattr(l, 'item') else l for l in meta_losses]}")
                print(f"   Skipping this iteration...")
                continue

            # Meta gradient computation and update
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # Check for NaN/Inf in gradients
            has_nan_grad = False
            for i, (w, g) in enumerate(zip(self.weights, meta_grads)):
                if torch.isnan(g).any() or torch.isinf(g).any():
                    print(f"⚠️ WARNING: NaN/Inf in gradient {i} at iteration {iteration}")
                    has_nan_grad = True
                w.grad = g

            if has_nan_grad:
                print(f"   Skipping optimizer step due to NaN gradients")
                self.meta_optimiser.zero_grad()
                continue

            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()

            # Logging
            epoch_loss += meta_loss.item()
            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.print_every:.6f}")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

    def main_loop_sequential(self, num_iterations):
        """Sequential processing version - most stable"""
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

            # Check for NaN/Inf in meta_loss
            if torch.isnan(meta_loss) or torch.isinf(meta_loss):
                print(f"⚠️ WARNING: NaN/Inf detected in meta_loss at iteration {iteration}")
                print(f"   Meta losses: {[l.item() if hasattr(l, 'item') else l for l in meta_losses]}")
                print(f"   Skipping this iteration...")
                continue

            # Meta gradient computation and update
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # Gradient clipping to prevent explosion
            max_grad_norm = 1.0
            meta_grads = [torch.clamp(g, -max_grad_norm, max_grad_norm) for g in meta_grads]

            # Check for NaN/Inf in gradients
            has_nan_grad = False
            for i, (w, g) in enumerate(zip(self.weights, meta_grads)):
                if torch.isnan(g).any() or torch.isinf(g).any():
                    print(f"⚠️ WARNING: NaN/Inf in gradient {i} at iteration {iteration}")
                    has_nan_grad = True
                w.grad = g

            if has_nan_grad:
                print(f"   Skipping optimizer step due to NaN gradients")
                self.meta_optimiser.zero_grad()
                continue

            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()

            # Logging
            epoch_loss += meta_loss.item()
            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.print_every:.6f}")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

    def main_loop_multi_gpu(self, num_iterations, gpu_ids=None):
        """
        Multi-GPU MAML training - each GPU processes different tasks in parallel

        Args:
            num_iterations: Number of meta-training iterations
            gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3]). If None, uses all available GPUs.

        Architecture:
            - Master GPU (gpu_ids[0]): Holds main model, aggregates gradients, performs meta-update
            - Worker GPUs: Each processes a subset of tasks in the meta-batch
            - Gradient synchronization: All task losses collected on master, single backward pass
        """
        import torch.multiprocessing as mp
        from torch.nn.parallel import DataParallel

        # Determine available GPUs
        if gpu_ids is None:
            num_gpus = torch.cuda.device_count()
            gpu_ids = list(range(num_gpus))
        else:
            num_gpus = len(gpu_ids)

        if num_gpus < 2:
            print(f"⚠️ Only {num_gpus} GPU(s) available. Falling back to sequential mode.")
            return self.main_loop_sequential(num_iterations)

        print(f"\n🚀 Multi-GPU MAML Training")
        print(f"   GPUs: {gpu_ids}")
        print(f"   Tasks per meta-batch: {self.tasks_per_meta_batch}")
        print(f"   Tasks per GPU: {self.tasks_per_meta_batch // num_gpus}")

        master_device = torch.device(f'cuda:{gpu_ids[0]}')
        epoch_loss = 0

        # Move main model to master GPU
        self.model = self.model.to(master_device)
        self.weights = list(self.model.parameters())

        # Create model replicas for each GPU
        model_replicas = {}
        for gpu_id in gpu_ids:
            if gpu_id == gpu_ids[0]:
                model_replicas[gpu_id] = self.model
            else:
                # Clone model to other GPUs
                replica = type(self.model)(
                    node_features=self.model.node_features if hasattr(self.model, 'node_features') else 7,
                    pooling=self.model.pooling if hasattr(self.model, 'pooling') else 'mean',
                    output_dim=self.model.output_dim if hasattr(self.model, 'output_dim') else 1,
                    dropout=self.model.dropout_rate if hasattr(self.model, 'dropout_rate') else 0.0,
                    conv_hidden_dim=self.model.conv_hidden_dim if hasattr(self.model, 'conv_hidden_dim') else 128,
                    num_conv_layers=self.model.num_conv_layers if hasattr(self.model, 'num_conv_layers') else 3,
                    fc_hidden_dim=self.model.fc_hidden_dim if hasattr(self.model, 'fc_hidden_dim') else 40,
                    num_fc_layers=self.model.num_fc_layers if hasattr(self.model, 'num_fc_layers') else 2
                ).to(f'cuda:{gpu_id}')
                replica.load_state_dict(self.model.state_dict())
                model_replicas[gpu_id] = replica

        print(f"   ✓ Model replicas created on {num_gpus} GPUs")

        for iteration in range(1, num_iterations + 1):
            # Sync weights to all replicas at start of each iteration
            master_state = self.model.state_dict()
            for gpu_id in gpu_ids[1:]:
                model_replicas[gpu_id].load_state_dict(master_state)

            # Distribute tasks across GPUs
            tasks_per_gpu = self.tasks_per_meta_batch // num_gpus
            remaining_tasks = self.tasks_per_meta_batch % num_gpus

            # Sample all task indices
            all_task_indices = [random.randint(0, self.num_tasks - 1)
                               for _ in range(self.tasks_per_meta_batch)]

            # Process tasks on each GPU
            all_losses = []
            task_offset = 0

            for gpu_idx, gpu_id in enumerate(gpu_ids):
                # Determine number of tasks for this GPU
                n_tasks = tasks_per_gpu + (1 if gpu_idx < remaining_tasks else 0)
                gpu_task_indices = all_task_indices[task_offset:task_offset + n_tasks]
                task_offset += n_tasks

                if not gpu_task_indices:
                    continue

                gpu_device = torch.device(f'cuda:{gpu_id}')
                gpu_model = model_replicas[gpu_id]
                gpu_weights = list(gpu_model.parameters())

                # Process tasks on this GPU
                for task_idx in gpu_task_indices:
                    try:
                        loss = self._inner_loop_on_device(
                            task_idx, gpu_model, gpu_weights, gpu_device
                        )
                        # Move loss to master device for aggregation
                        all_losses.append(loss.to(master_device))
                    except Exception as e:
                        print(f"⚠️ Task {task_idx} on GPU {gpu_id} failed: {e}")
                        continue

            if not all_losses:
                continue

            # Aggregate losses on master GPU
            meta_loss = sum(all_losses) / len(all_losses)

            # Check for NaN/Inf
            if torch.isnan(meta_loss) or torch.isinf(meta_loss):
                print(f"⚠️ WARNING: NaN/Inf in meta_loss at iteration {iteration}")
                continue

            # Meta gradient computation on master model
            meta_grads = torch.autograd.grad(meta_loss, self.weights, allow_unused=True)

            # Gradient clipping
            max_grad_norm = 1.0
            meta_grads = [torch.clamp(g, -max_grad_norm, max_grad_norm) if g is not None
                         else torch.zeros_like(w) for g, w in zip(meta_grads, self.weights)]

            # Check for NaN/Inf in gradients
            has_nan_grad = False
            for i, (w, g) in enumerate(zip(self.weights, meta_grads)):
                if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                    print(f"⚠️ WARNING: NaN/Inf in gradient {i}")
                    has_nan_grad = True
                if g is not None:
                    w.grad = g

            if has_nan_grad:
                self.meta_optimiser.zero_grad()
                continue

            # Meta update on master model
            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()

            # Logging
            epoch_loss += meta_loss.item()
            if iteration % self.print_every == 0:
                print(f"{iteration}/{num_iterations} | loss: {epoch_loss / self.print_every:.6f} [Multi-GPU: {num_gpus}]")
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

        # Cleanup replicas
        for gpu_id in gpu_ids[1:]:
            del model_replicas[gpu_id]
        torch.cuda.empty_cache()

    def _inner_loop_on_device(self, task_idx, model, weights, device):
        """
        Inner loop executed on a specific GPU device

        Args:
            task_idx: Task index to process
            model: Model replica on target device
            weights: Model weights on target device
            device: Target CUDA device
        """
        temp_weights = [w.clone() for w in weights]

        # Get task data
        minimal_samples, outputs = self.get_task_data(task_idx)

        # Sample K samples for support set
        total_libs = len(minimal_samples)
        if total_libs < self.K:
            support_indices = list(range(total_libs))
        else:
            support_indices = random.sample(range(total_libs), self.K)

        support_samples = [minimal_samples[i] for i in support_indices]
        support_outputs = [outputs[i] for i in support_indices]

        # Inner loop adaptation
        for step in range(self.inner_steps):
            batch_data = []
            for minimal_sample in support_samples:
                data = self.create_pyg_data_with_adj_matrix(minimal_sample)
                batch_data.append(data)

            if not batch_data:
                return torch.tensor(0.0, requires_grad=True, device=device)

            X = Batch.from_data_list(batch_data).to(device)
            y = torch.tensor(support_outputs, dtype=torch.float32, device=device).view(-1, 1)

            with torch.cuda.amp.autocast():
                predictions = functional_forward(model, X, temp_weights)
                loss = self.criterion(predictions, y + 1e-6) / self.K

            grad = torch.autograd.grad(loss, temp_weights, create_graph=True, allow_unused=True)
            grad = [torch.clamp(g, -1.0, 1.0) if g is not None else g for g in grad]
            temp_weights = [w - self.inner_lr * g if g is not None else w
                          for w, g in zip(temp_weights, grad)]

        # Query set evaluation
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
        y = torch.tensor(query_outputs, dtype=torch.float32, device=device).view(-1, 1)

        with torch.cuda.amp.autocast():
            predictions = functional_forward(model, X, temp_weights)
            loss = self.criterion(predictions, y + 1e-6) / len(query_indices)

        return loss


class MultiGPUMAMLWorker:
    """
    Worker class for true multi-process multi-GPU MAML

    Uses torch.multiprocessing to spawn separate processes for each GPU,
    avoiding GIL limitations and enabling true parallelism.
    """

    def __init__(self, gpu_id, model_config, data_config):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}')
        self.model_config = model_config
        self.data_config = data_config

    @staticmethod
    def worker_process(gpu_id, model_state_dict, task_indices, data_queue, result_queue,
                       model_config, inner_lr, inner_steps, K):
        """
        Worker process function - runs on a separate process for each GPU

        Args:
            gpu_id: GPU ID for this worker
            model_state_dict: Shared model state dict
            task_indices: List of task indices to process
            data_queue: Queue containing task data
            result_queue: Queue to put results (losses)
            model_config: Model configuration dict
            inner_lr: Inner loop learning rate
            inner_steps: Number of inner loop steps
            K: Support set size
        """
        import torch
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')

        # Create local model
        from gnn_maml import create_maml_gcn_model
        model = create_maml_gcn_model(**model_config).to(device)
        model.load_state_dict(model_state_dict)

        criterion = torch.nn.MSELoss()
        weights = list(model.parameters())

        losses = []

        for task_idx in task_indices:
            try:
                # Get task data from queue
                task_data = data_queue.get()
                support_data, support_y, query_data, query_y = task_data

                # Move data to device
                support_data = support_data.to(device)
                support_y = support_y.to(device)
                query_data = query_data.to(device)
                query_y = query_y.to(device)

                # Inner loop
                temp_weights = [w.clone() for w in weights]

                for step in range(inner_steps):
                    with torch.cuda.amp.autocast():
                        predictions = functional_forward(model, support_data, temp_weights)
                        loss = criterion(predictions, support_y) / K

                    grad = torch.autograd.grad(loss, temp_weights, create_graph=True, allow_unused=True)
                    grad = [torch.clamp(g, -1.0, 1.0) if g is not None else g for g in grad]
                    temp_weights = [w - inner_lr * g if g is not None else w
                                   for w, g in zip(temp_weights, grad)]

                # Query evaluation
                with torch.cuda.amp.autocast():
                    predictions = functional_forward(model, query_data, temp_weights)
                    query_loss = criterion(predictions, query_y) / len(query_y)

                losses.append(query_loss.detach().cpu())

            except Exception as e:
                print(f"Worker {gpu_id}: Task {task_idx} failed: {e}")
                continue

        # Put results back
        result_queue.put((gpu_id, losses))


class GNNLazyMAML:
    """
    GNN MAML with lazy loading - loads data from disk on demand.
    Significantly reduces memory usage compared to loading all data upfront.
    """

    def __init__(self, model, inner_lr, meta_lr, K=5, inner_steps=1,
                 dataset=None, norm_stats=None, tasks_per_meta_batch=16):
        """
        Args:
            model: GNN model
            inner_lr: Inner loop learning rate
            meta_lr: Meta learning rate
            K: Support set size
            inner_steps: Number of inner loop steps
            dataset: MAMLTaskDataset instance for lazy loading
            norm_stats: Normalization statistics for node features
            tasks_per_meta_batch: Number of tasks per meta-batch
        """
        # Dataset
        self.dataset = dataset
        self.topology_cache = dataset.topology_cache if dataset else None
        self.cache_type = dataset.graph_mode if dataset else 'full_graph'
        self.norm_stats = norm_stats

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
            print(f"✅ Model moved to {device}")

        print(f"🎯 GNN Lazy MAML Configuration:")
        print(f"   Cache type: {self.cache_type}")
        print(f"   Tasks: {self.num_tasks}")
        print(f"   Support set size (K): {self.K}")
        print(f"   Inner steps: {self.inner_steps}")
        print(f"   Tasks per meta batch: {self.tasks_per_meta_batch}")
        print(f"   💾 Memory efficient: Lazy loading enabled")

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

            with torch.cuda.amp.autocast():
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

        with torch.cuda.amp.autocast():
            predictions = functional_forward(self.model, X, temp_weights)
            loss = self.criterion(predictions, y + 1e-6) / len(query_indices)

        return loss

    def main_loop_sequential(self, num_iterations):
        """Sequential MAML training loop with lazy loading"""
        print(f"\n🚀 Starting Lazy MAML Training (Sequential)")
        print(f"   Total iterations: {num_iterations}")

        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):
            # Sample random tasks
            task_indices = random.sample(range(self.num_tasks), min(self.tasks_per_meta_batch, self.num_tasks))

            meta_losses = []
            for task_idx in task_indices:
                try:
                    task_loss = self.inner_loop_single_task(task_idx)
                    meta_losses.append(task_loss)
                except Exception as e:
                    print(f"⚠️ Task {task_idx} failed: {e}")
                    continue

            if not meta_losses:
                continue

            meta_loss = sum(meta_losses) / len(meta_losses)

            if torch.isnan(meta_loss) or torch.isinf(meta_loss):
                print(f"⚠️ WARNING: NaN/Inf detected at iteration {iteration}")
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


def prepare_lazy_loading_data(process_type, corner_type, data_type='cell', graph_mode='stage_aware'):
    """
    Prepare data for lazy loading mode.
    Returns only metadata (file paths, indices) without loading actual data.

    Args:
        process_type: Process type (RVT, LVT, SLVT, SRAM)
        corner_type: Corner type (TT, FF, SS)
        data_type: Data type ('cell' or 'transition')
        graph_mode: Graph mode ('stage_aware' or 'full_graph')

    Returns:
        tuple: (data_file_paths, sample_indices_list, topology_cache_path, norm_stats)
    """
    print(f"🎯 Preparing lazy loading data for: {process_type}_{corner_type}")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")

    base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp"
    train_indices_filename = f"train_indices_{data_type}_{graph_mode}.pth"

    data_file_paths = []
    sample_indices_list = []
    topology_cache_path = None

    folder_idx = 0
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

        if not corner_match:
            continue

        train_data_path = f"{base_path}/{item}/train_test_split/{train_indices_filename}"
        if not os.path.exists(train_data_path):
            print(f"   ⚠️ Train indices not found: {train_data_path}")
            continue

        print(f"   📂 Found: {item}")

        # Load only train indices metadata
        train_meta = torch.load(train_data_path, weights_only=False, map_location='cpu')
        sample_indices = train_meta['sample_indices']
        data_file = train_meta['data_file']
        cache_path = train_meta.get('cache_path', None)

        # Fix path if needed
        if data_file.startswith('/mnt/home/'):
            data_file = data_file.replace('/mnt/home/', '/home/')

        if not os.path.isabs(data_file):
            dataset_dir = train_meta.get('dataset_dir', '')
            if dataset_dir:
                data_filename = os.path.basename(data_file)
                data_file = os.path.join(dataset_dir, 'graph_data', data_filename)

        if not os.path.exists(data_file):
            print(f"   ⚠️ Data file not found: {data_file}")
            continue

        data_file_paths.append(data_file)
        sample_indices_list.append((folder_idx, sample_indices))
        folder_idx += 1

        # Get topology cache path (only need one)
        if topology_cache_path is None and cache_path:
            if not os.path.isabs(cache_path) and not os.path.exists(cache_path):
                cache_filename = os.path.basename(cache_path)
                cache_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"
            if os.path.exists(cache_path):
                topology_cache_path = cache_path

    if not data_file_paths:
        raise ValueError(f"No data found for {process_type}_{corner_type}")

    total_tasks = sum(len(indices) for _, indices in sample_indices_list)
    print(f"   ✅ Prepared {len(data_file_paths)} data files")
    print(f"   ✅ Total tasks: {total_tasks}")

    # Load norm_stats from first data file (small overhead, needed for normalization)
    print(f"   📊 Loading normalization stats...")
    first_data = torch.load(data_file_paths[0], weights_only=False, map_location='cpu')
    minimal_data_sample = first_data['minimal_data_per_file'][0][:100]  # Small sample
    _, norm_stats, _ = preprocess_gnn_minimal_data(
        [minimal_data_sample],
        min_std_threshold=1e-6,
        enable_filtering=False,
        verbose=False
    )
    del first_data
    gc.collect()

    return data_file_paths, sample_indices_list, topology_cache_path, norm_stats


def load_cached_gnn_data_for_maml(process_type, corner_type, data_type='cell', graph_mode='stage_aware'):
    """
    Load cached GNN train data from train_test_split folders

    Args:
        process_type: Process type (RVT, LVT, SLVT, SRAM)
        corner_type: Corner type (TT, FF, SS)
        data_type: Data type ('cell' or 'transition')
        graph_mode: Graph mode ('stage_aware' or 'full_graph')
    """
    print(f"🎯 Loading cached GNN train data for MAML: {process_type}_{corner_type}")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")

    base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp"
    matching_folders = []

    # Determine train indices filename
    train_indices_filename = f"train_indices_{data_type}_{graph_mode}.pth"

    # Find all matching folders
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

    if not matching_folders:
        raise ValueError(f"No train data found for {process_type}_{corner_type}")

    print(f"   📂 Found {len(matching_folders)} matching datasets:")
    for folder, _ in matching_folders:
        print(f"     - {folder}")

    # Load and merge all matching datasets
    all_minimal_data_per_file = []
    topology_cache = None

    for folder, train_indices_path in matching_folders:
        print(f"   📥 Loading: {folder}")

        # Load train indices
        train_meta = torch.load(train_indices_path, weights_only=False, map_location='cpu')
        sample_indices = train_meta['sample_indices']
        data_file = train_meta['data_file']
        cache_path = train_meta.get('cache_path', None)

        # Convert /mnt/home/ to /home/ if needed (fix path mismatch)
        if data_file.startswith('/mnt/home/'):
            data_file = data_file.replace('/mnt/home/', '/home/')
            print(f"     Converted /mnt/home/ → /home/")

        # Convert data_file to absolute path if it's relative (for backward compatibility)
        if not os.path.isabs(data_file):
            # Check if file exists as-is first
            if not os.path.exists(data_file):
                # Try resolving relative to dataset_dir instead of train_indices location
                dataset_dir = train_meta.get('dataset_dir', '')
                if dataset_dir:
                    # Extract filename from data_file path
                    data_filename = os.path.basename(data_file)
                    # Construct path: dataset_dir/graph_data/filename
                    data_file = os.path.join(dataset_dir, 'graph_data', data_filename)
                    print(f"     Resolved data_file to: {data_file}")
                else:
                    print(f"     ⚠️ Warning: Could not resolve relative path: {data_file}")

        # Verify data_file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file not found: {data_file}")

        # Convert cache_path to absolute path if needed
        if cache_path and not os.path.isabs(cache_path):
            if not os.path.exists(cache_path):
                # Default cache location
                cache_filename = os.path.basename(cache_path)
                cache_file = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"
                if os.path.exists(cache_file):
                    cache_path = cache_file
                    print(f"     Resolved cache_path to: {cache_path}")
                else:
                    print(f"     ⚠️ Warning: Could not find cache: {cache_path}")

        # Load topology cache (only once)
        if topology_cache is None and cache_path:
            print(f"     Loading topology cache from: {cache_path}")
            topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu')
            print(f"     ✓ Loaded topology cache for {len(topology_cache)} cells")

        # Load full dataset
        print(f"     Loading full dataset from: {data_file}")
        full_data = torch.load(data_file, weights_only=False, map_location='cpu')
        minimal_data_per_file = full_data['minimal_data_per_file']  # [num_libs][num_samples]

        print(f"     Full dataset: {len(minimal_data_per_file[0])} samples, {len(minimal_data_per_file)} lib files")
        print(f"     Train subset: {len(sample_indices)} samples")

        if not all_minimal_data_per_file:
            # First dataset - initialize
            all_minimal_data_per_file = [[] for _ in range(len(minimal_data_per_file))]

        # Merge minimal data: extract only train samples for each lib
        for lib_idx, lib_samples in enumerate(minimal_data_per_file):
            train_samples = [lib_samples[idx] for idx in sample_indices]
            all_minimal_data_per_file[lib_idx].extend(train_samples)

        # Clean up full_data to free memory
        del full_data
        del minimal_data_per_file
        gc.collect()

    print(f"   ✅ Merged data:")
    print(f"     Total tasks: {len(all_minimal_data_per_file[0])} (input conditions)")
    print(f"     Lib files per task: {len(all_minimal_data_per_file)} (process variations)")

    # Apply preprocessing pipeline (filtering + normalization with NaN/Inf detection)
    print(f"\n🔧 Applying data preprocessing pipeline...")
    preprocessed_data, norm_stats, preprocessing_stats = preprocess_gnn_minimal_data(
        all_minimal_data_per_file,
        min_std_threshold=1e-6,
        enable_filtering=True,
        verbose=True
    )

    print(f"\n📊 Preprocessing Summary:")
    print(f"   Valid tasks after filtering: {preprocessing_stats['filtering']['valid_tasks']}")
    print(f"   Filter ratio: {preprocessing_stats['filtering']['filter_ratio']:.1f}%")

    return preprocessed_data, topology_cache, norm_stats

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='MAML GNN Training with Cached Topology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single architecture
  python maml_gnn_training_cached.py --process SRAM --corner FF --graph_mode stage_aware

  # Architecture sweep (data loaded once)
  python maml_gnn_training_cached.py --process LVT --corner FF \\
      --conv_hidden_dim 64 128 256 \\
      --num_conv_layers 2 3 4 \\
      --fc_hidden_dim 40 64 128 \\
      --num_fc_layers 2 3 4

  # Run all 12 process-corner combinations
  python maml_gnn_training_cached.py --run_all --graph_mode stage_aware

  # Multi-GPU mode (each GPU processes different tasks in parallel)
  python maml_gnn_training_cached.py --process RVT --corner FF --multi_gpu 0,1,2,3

  # Lazy loading mode (memory efficient - loads data on demand)
  python maml_gnn_training_cached.py --process RVT --corner FF --lazy_load
"""
    )

    # Run all option
    parser.add_argument('--run_all', action='store_true',
                       help='Run all 12 combinations (4 process × 3 corners)')

    # Single run arguments (required unless --run_all is set)
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

    # Model architecture parameters (can accept multiple values for sweep)
    parser.add_argument('--conv_hidden_dim', type=int, nargs='+', default=[128],
                       help='Convolution layer hidden dimension(s) (default: 128). Multiple values for sweep.')
    parser.add_argument('--num_conv_layers', type=int, nargs='+', default=[3],
                       help='Number of GCN convolutional layers (default: 3). Multiple values for sweep.')
    parser.add_argument('--fc_hidden_dim', type=int, nargs='+', default=[40],
                       help='FC layer hidden dimension(s) (default: 40). Multiple values for sweep.')
    parser.add_argument('--num_fc_layers', type=int, nargs='+', default=[2],
                       help='Number of FC layers (default: 2). Multiple values for sweep.')

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
    parser.add_argument('--multi_gpu', type=str, default=None,
                       help='Multi-GPU mode: comma-separated GPU IDs (e.g., "0,1,2,3"). Each GPU processes different tasks in parallel.')

    # Memory configuration
    parser.add_argument('--lazy_load', action='store_true',
                       help='Enable lazy loading mode for memory efficiency. Loads data from disk on demand instead of loading all into memory.')

    # Data type
    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')

    # Graph mode
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode (default: stage_aware)')

    return parser.parse_args()

def train_single_config(process, corner_type, args):
    """Train MAML for a single process-corner configuration with architecture sweep support"""
    import itertools

    # Extract parameters
    total_iterations = args.total_iterations
    chunk_size = args.chunk_size
    inner_steps = args.inner_steps
    innerdiv = args.innerdiv
    meta_lr = args.meta_lr
    K = args.K
    tasks_per_meta_batch = args.tasks_per_meta_batch
    data_type = args.data_type
    graph_mode = args.graph_mode

    # Generate architecture combinations (cartesian product)
    arch_combinations = list(itertools.product(
        args.conv_hidden_dim,
        args.num_conv_layers,
        args.fc_hidden_dim,
        args.num_fc_layers
    ))

    num_combinations = len(arch_combinations)
    is_sweep = num_combinations > 1

    print(f"\n🚀 MAML GNN Training with Cached Topology")
    print(f"📋 Base Configuration:")
    print(f"   Process: {process}")
    print(f"   Corner: {corner_type}")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")
    print(f"   Inner div: {innerdiv}")
    print(f"   Meta LR: {meta_lr}")
    print(f"   Inner steps: {inner_steps}")
    print(f"   K: {K}")
    print(f"   Tasks per meta batch: {tasks_per_meta_batch}")
    print(f"   Total iterations: {total_iterations}")
    print(f"   GPU: {args.gpu}")

    # Parse multi-GPU configuration
    multi_gpu_ids = None
    if args.multi_gpu:
        multi_gpu_ids = [int(g.strip()) for g in args.multi_gpu.split(',')]
        print(f"   Multi-GPU mode: {multi_gpu_ids} ({len(multi_gpu_ids)} GPUs)")

    # Lazy loading mode
    lazy_load = getattr(args, 'lazy_load', False)
    if lazy_load:
        print(f"   💾 Lazy loading: ENABLED (memory efficient)")

    if is_sweep:
        print(f"\n🔄 Architecture Sweep Mode: {num_combinations} combinations")
        print(f"   conv_hidden_dim: {args.conv_hidden_dim}")
        print(f"   num_conv_layers: {args.num_conv_layers}")
        print(f"   fc_hidden_dim: {args.fc_hidden_dim}")
        print(f"   num_fc_layers: {args.num_fc_layers}")
    else:
        print(f"\n📐 Architecture:")
        print(f"   Conv hidden dim: {args.conv_hidden_dim[0]}")
        print(f"   Num conv layers: {args.num_conv_layers[0]}")
        print(f"   FC hidden dim: {args.fc_hidden_dim[0]}")
        print(f"   Num FC layers: {args.num_fc_layers[0]}")

    # Load data based on mode (lazy vs full load)
    data_load_start = time.time()

    if lazy_load:
        # Lazy loading mode - only load metadata, actual data loaded on demand
        print("\n📊 Preparing lazy loading data (memory efficient)...")
        data_file_paths, sample_indices_list, topology_cache_path, norm_stats = prepare_lazy_loading_data(
            process, corner_type, data_type, graph_mode
        )
        # Create dataset (doesn't load actual data yet)
        dataset = MAMLTaskDataset(
            data_file_paths=data_file_paths,
            sample_indices_list=sample_indices_list,
            topology_cache_path=topology_cache_path,
            graph_mode=graph_mode
        )
        minimal_data_per_file = None
        topology_cache = None
    else:
        # Full load mode - load all data into memory
        print("\n📊 Loading cached GNN data (once for all architectures)...")
        minimal_data_per_file, topology_cache, norm_stats = load_cached_gnn_data_for_maml(
            process, corner_type, data_type, graph_mode
        )
        dataset = None

    data_load_time = time.time() - data_load_start
    print(f"✅ Data {'prepared' if lazy_load else 'loaded'} in {data_load_time/60:.2f} minutes")

    # Train each architecture combination
    trained_models = []
    sweep_start_time = time.time()

    for idx, (conv_hidden_dim, num_conv_layers, fc_hidden_dim, num_fc_layers) in enumerate(arch_combinations, 1):
        if is_sweep:
            print(f"\n{'='*80}")
            print(f"🔧 Architecture {idx}/{num_combinations}")
            print(f"{'='*80}")

        arch_start_time = time.time()

        print(f"📐 Current Architecture:")
        print(f"   Conv hidden dim: {conv_hidden_dim}")
        print(f"   Num conv layers: {num_conv_layers}")
        print(f"   FC hidden dim: {fc_hidden_dim}")
        print(f"   Num FC layers: {num_fc_layers}")

        # Calculate inner_lr
        base_lr = 0.001
        inner_lr = base_lr / innerdiv

        # Calculate num_chunks
        num_chunks = total_iterations // chunk_size

        # Create GNN MAML model based on loading mode
        print("🤖 Creating GNN MAML model...")

        if lazy_load:
            # Lazy loading mode - use GNNLazyMAML
            gnn_maml = GNNLazyMAML(
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
                norm_stats=norm_stats,
                inner_lr=inner_lr,
                meta_lr=meta_lr,
                inner_steps=inner_steps,
                K=K,
                tasks_per_meta_batch=tasks_per_meta_batch
            )
        else:
            # Full load mode - use GNNCachedMAML
            gnn_maml = GNNCachedMAML(
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
                minimal_data_per_file=minimal_data_per_file,
                topology_cache=topology_cache,
                cache_type=graph_mode,
                norm_stats=norm_stats,
                inner_lr=inner_lr,
                meta_lr=meta_lr,
                inner_steps=inner_steps,
                K=K,
                tasks_per_meta_batch=tasks_per_meta_batch
            )

        # Training in chunks
        for chunk in range(num_chunks):
            print(f"\n📦 Processing chunk {chunk+1}/{num_chunks}")
            chunk_start_time = time.time()

            try:
                if multi_gpu_ids and len(multi_gpu_ids) > 1 and not lazy_load:
                    # Multi-GPU mode: each GPU processes different tasks in parallel
                    # Note: Multi-GPU not yet supported with lazy loading
                    gnn_maml.main_loop_multi_gpu(num_iterations=chunk_size, gpu_ids=multi_gpu_ids)
                else:
                    gnn_maml.main_loop_sequential(num_iterations=chunk_size)
            except Exception as e:
                print(f"⚠️ Training failed: {e}")
                print("⚠️ Falling back to sequential mode...")
                try:
                    gnn_maml.main_loop_sequential(num_iterations=chunk_size)
                except Exception as e2:
                    print(f"⚠️ Sequential processing failed: {e2}")
                    print("⚠️ Reducing learning rates and retrying...")
                    gnn_maml.inner_lr *= 0.5
                    gnn_maml.meta_lr *= 0.5
                    gnn_maml.main_loop_sequential(num_iterations=chunk_size//2)

            # GPU synchronization and timing
            torch.cuda.synchronize()
            chunk_end_time = time.time()

            chunk_time = chunk_end_time - chunk_start_time
            print(f"⏱️ Chunk {chunk+1} completed in {chunk_time:.2f}s")

            # Memory cleanup after each chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Memory monitoring
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"💾 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

            # Save checkpoint
            checkpoint_dir = "../../../pretrained_models/gnn_maml_checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            iterations_completed = (chunk + 1) * chunk_size

            # Build model architecture suffix for filename
            arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

            checkpoint_path = f"{checkpoint_dir}/gnn_maml_{process}_{corner_type}_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{iterations_completed}_inner{inner_steps}{arch_suffix}.pth"

            torch.save({
                'model_state_dict': gnn_maml.model.state_dict(),
                'norm_stats': norm_stats,
                'task_norm_stats': gnn_maml.task_norm_stats,  # Per-task output normalization
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

        # Build model architecture suffix for final model filename
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

        final_model_path = f"{final_model_dir}/gnn_maml_{process}_{corner_type}_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{total_iterations}_inner{inner_steps}{arch_suffix}.pth"

        torch.save({
            'model_state_dict': gnn_maml.model.state_dict(),
            'norm_stats': norm_stats,
            'task_norm_stats': gnn_maml.task_norm_stats,  # Per-task output normalization
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

        print(f"\n🧹 Cleaning up memory after this architecture...")

        # GPU memory cleanup - delete model-specific data structures
        del gnn_maml.stacked_outputs
        del gnn_maml.normalized_outputs
        del gnn_maml.task_norm_stats
        del gnn_maml.weights
        del gnn_maml.meta_optimiser
        del gnn_maml.model
        del gnn_maml

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"   ✓ CUDA cache cleared")

        print(f"   ✓ Memory cleaned up")

        arch_time = time.time() - arch_start_time
        print(f"\n⏱️ Architecture {idx}/{num_combinations} completed in {arch_time/3600:.2f} hours")

        # Store trained model info
        trained_models.append({
            'path': final_model_path,
            'arch': (conv_hidden_dim, num_conv_layers, fc_hidden_dim, num_fc_layers),
            'time': arch_time
        })

    # All architectures completed - cleanup shared data
    print(f"\n🧹 Cleaning up shared data...")
    del minimal_data_per_file
    del topology_cache
    del norm_stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"   ✓ Shared data cleaned up")

    # Print sweep summary
    sweep_total_time = time.time() - sweep_start_time
    print(f"\n{'='*80}")
    print(f"🎉 ARCHITECTURE SWEEP COMPLETED")
    print(f"{'='*80}")
    print(f"Process: {process}")
    print(f"Corner: {corner_type}")
    print(f"Total architectures trained: {len(trained_models)}")
    print(f"Total time: {sweep_total_time/3600:.2f} hours")
    print(f"Average time per architecture: {sweep_total_time/len(trained_models)/3600:.2f} hours")
    print(f"\nTrained models:")
    for i, model_info in enumerate(trained_models, 1):
        conv_h, conv_l, fc_h, fc_l = model_info['arch']
        print(f"  {i}. conv{conv_h}x{conv_l}_fc{fc_h}x{fc_l} - {model_info['time']/3600:.2f}h")
        print(f"     {model_info['path']}")
    print(f"{'='*80}")

    return trained_models[0]['path'] if trained_models else None

def main():
    """Main function"""
    args = parse_arguments()

    # Validate arguments
    if not args.run_all and (not args.process or not args.corner):
        print("❌ Error: --process and --corner are required unless --run_all is specified")
        print("Use --help for usage information")
        return

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args.run_all:
        # Run all 12 combinations
        all_processes = ['RVT', 'SLVT', 'SRAM' , 'LVT' ]
        all_corners = ['TT', 'FF', 'SS']

        total_configs = len(all_processes) * len(all_corners)
        completed = 0
        failed = []

        print(f"\n{'='*80}")
        print(f"🚀 RUNNING ALL {total_configs} CONFIGURATIONS")
        print(f"{'='*80}")
        print(f"Processes: {', '.join(all_processes)}")
        print(f"Corners: {', '.join(all_corners)}")
        print(f"Graph mode: {args.graph_mode}")
        print(f"Data type: {args.data_type}")
        print(f"{'='*80}\n")

        overall_start = time.time()

        for process in all_processes:
            for corner in all_corners:
                config_num = completed + 1
                print(f"\n{'='*80}")
                print(f"📦 CONFIG {config_num}/{total_configs}: {process}_{corner}")
                print(f"{'='*80}\n")

                try:
                    model_path = train_single_config(process, corner, args)
                    completed += 1
                    print(f"\n✅ Config {config_num}/{total_configs} completed: {process}_{corner}")
                    print(f"   Model: {model_path}")
                except Exception as e:
                    failed.append(f"{process}_{corner}")
                    print(f"\n❌ Config {config_num}/{total_configs} failed: {process}_{corner}")
                    print(f"   Error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Clear GPU memory between configurations (comprehensive cleanup)
                    print(f"\n🧹 Cleaning up memory between configurations...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3
                        print(f"   💾 GPU Memory after cleanup: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                    print(f"   ✓ Memory cleaned up")

        overall_time = time.time() - overall_start

        print(f"\n{'='*80}")
        print(f"🎉 ALL CONFIGURATIONS COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {overall_time/3600:.2f} hours")
        print(f"Completed: {completed}/{total_configs}")
        if failed:
            print(f"Failed: {len(failed)}")
            for config in failed:
                print(f"  - {config}")
        print(f"{'='*80}\n")
    else:
        # Run single configuration
        train_single_config(args.process, args.corner, args)

if __name__ == "__main__":
    main()
