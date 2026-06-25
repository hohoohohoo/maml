#!/usr/bin/env python
"""
MAML GNN Training with Cached Topology - PROCESS-AWARE VERSION
Supports both full_graph and stage_aware modes with topology cache
Uses 11D node features (7 base + 4 process parameters)

Process parameters added to base features:
- param_a, param_b (PMOS/NMOS), param_c (PMOS/NMOS), temperature

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
    Process-aware version with 11D node features
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

        # Normalize outputs
        self.normalized_outputs = self.normalize_all_task_outputs(self.stacked_outputs)

        self.num_tasks = self.stacked_outputs.shape[0]  # Number of input conditions
        self.lib_files_per_task = self.stacked_outputs.shape[1]  # Number of process variations

        # MAML hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # Logging
        self.print_every = 100
        self.plot_every = 100
        self.meta_losses = []

        # Meta-learner
        self.model = model.to(device)
        self.weights = list(self.model.parameters())
        self.criterion = nn.MSELoss()
        self.meta_optimiser = optim.Adam(self.weights, lr=self.meta_lr)

        print(f"\n✅ GNN MAML Initialized (Process-aware version)")
        print(f"   Node features: 11D (7 base + 4 process)")
        print(f"   Cache type: {self.cache_type}")
        print(f"   Tasks: {self.num_tasks}")
        print(f"   Lib files per task: {self.lib_files_per_task}")
        print(f"   Inner LR: {self.inner_lr}")
        print(f"   Meta LR: {self.meta_lr}")
        print(f"   K: {self.K}")
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
        # Get node features from minimal sample (11D for process-aware)
        node_features = minimal_sample['node_features']

        # Load pre-computed adjacency matrix from cache
        adjacency_matrix = self.get_adjacency_matrix_from_cache(minimal_sample)

        # Apply normalization to node features
        normalized_features = self.normalize_node_features(node_features)

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

def load_cached_gnn_data_for_maml_process(data_type='cell', graph_mode='stage_aware'):
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

    print(f"\n📂 Loading process-aware datasets from {base_path}")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")
    print(f"   ⚠️  Note: Loading FULL datasets (no train/test split)")

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
            print(f"   ⚠️ Data file not found: {data_file_path}")

    if not matching_folders:
        raise ValueError(f"No data found in {base_path}")

    print(f"   📂 Found {len(matching_folders)} non-test datasets")

    # Load and merge all matching datasets
    all_minimal_data_per_file = []
    topology_cache = None

    for folder, data_file_path in matching_folders:
        print(f"   📥 Loading: {folder}")

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
                    print(f"     ⚠️ Warning: Could not find cache: {cache_path}")

        # Load topology cache (only once)
        if topology_cache is None and cache_path:
            print(f"     Loading topology cache from: {cache_path}")
            topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu')
            print(f"     ✓ Loaded topology cache for {len(topology_cache)} cells")

        if not all_minimal_data_per_file:
            # First dataset - initialize
            all_minimal_data_per_file = [[] for _ in range(len(minimal_data_per_file))]

        # Merge minimal data: use ALL samples (no train/test split)
        for lib_idx, lib_samples in enumerate(minimal_data_per_file):
            all_minimal_data_per_file[lib_idx].extend(lib_samples)

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


def load_merged_data_direct(merged_file_path, apply_preprocessing=True):
    """
    Load pre-merged dataset directly from a single .pth file.
    Skips the merge process entirely - much faster initialization.

    Args:
        merged_file_path: Path to merged .pth file (e.g., merged_cell_stage_aware.pth)
                          If not absolute path, will look in default base directory.
        apply_preprocessing: Whether to apply preprocessing (filtering + normalization)

    Returns:
        minimal_data_per_file: Merged minimal data
        topology_cache: Shared topology cache (loaded from cache_path)
        norm_stats: Normalization statistics (if apply_preprocessing=True)
    """
    # Default base path for merged files
    DEFAULT_BASE_PATH = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp_process"

    # If not absolute path, prepend default base path
    if not os.path.isabs(merged_file_path):
        merged_file_path = os.path.join(DEFAULT_BASE_PATH, merged_file_path)

    print(f"\n📂 Loading pre-merged dataset directly")
    print(f"   File: {merged_file_path}")

    if not os.path.exists(merged_file_path):
        raise FileNotFoundError(f"Merged file not found: {merged_file_path}")

    # Load merged data
    print(f"   Loading merged data...")
    merged_data = torch.load(merged_file_path, weights_only=False, map_location='cpu')

    minimal_data_per_file = merged_data['minimal_data_per_file']
    cache_path = merged_data.get('cache_path', None)
    metadata = merged_data.get('metadata', {})

    print(f"   ✓ Loaded minimal_data_per_file")
    print(f"     Tasks: {len(minimal_data_per_file[0])}")
    print(f"     Lib files per task: {len(minimal_data_per_file)}")

    if metadata:
        print(f"   📋 Metadata:")
        print(f"     Data type: {metadata.get('data_type', 'N/A')}")
        print(f"     Graph mode: {metadata.get('graph_mode', 'N/A')}")
        print(f"     Num datasets merged: {metadata.get('num_datasets', 'N/A')}")
        print(f"     Sample ratio: {metadata.get('sample_ratio', 'N/A')}")

    # Load topology cache from cache_path
    topology_cache = None
    if cache_path:
        # Handle relative path
        if not os.path.isabs(cache_path):
            cache_filename = os.path.basename(cache_path)
            possible_paths = [
                cache_path,
                f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    cache_path = p
                    break

        if os.path.exists(cache_path):
            print(f"   Loading topology cache from: {cache_path}")
            topology_cache = torch.load(cache_path, weights_only=False, map_location='cpu')
            print(f"   ✓ Loaded topology cache for {len(topology_cache)} cells")
        else:
            print(f"   ⚠️ Warning: Topology cache not found at {cache_path}")
    else:
        print(f"   ⚠️ Warning: No cache_path in merged file")

    # Apply preprocessing if requested
    norm_stats = None
    if apply_preprocessing:
        print(f"\n🔧 Applying data preprocessing pipeline...")
        minimal_data_per_file, norm_stats, preprocessing_stats = preprocess_gnn_minimal_data(
            minimal_data_per_file,
            min_std_threshold=1e-6,
            enable_filtering=True,
            verbose=True
        )
        print(f"\n📊 Preprocessing Summary:")
        print(f"   Valid tasks after filtering: {preprocessing_stats['filtering']['valid_tasks']}")
        print(f"   Filter ratio: {preprocessing_stats['filtering']['filter_ratio']:.1f}%")

    print(f"\n✅ Direct loading complete!")
    return minimal_data_per_file, topology_cache, norm_stats


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='MAML GNN Training with Cached Topology (Process-Aware)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (loads from individual directories - slowest)
  python maml_gnn_training_cached_with_process.py --graph_mode stage_aware --data_type cell

  # Use pre-merged file with auto-detected path (faster)
  python maml_gnn_training_cached_with_process.py --graph_mode stage_aware --data_type cell --use_merged

  # Use pre-merged file with explicit path (fastest, recommended)
  python maml_gnn_training_cached_with_process.py --merged_file /path/to/merged_cell_stage_aware.pth

  # With sampled merged file
  python maml_gnn_training_cached_with_process.py --merged_file /path/to/merged_cell_stage_aware_sampled10.pth

Data Loading Options:
  --merged_file : Direct path to pre-merged .pth file (fastest, skips merge process)
  --use_merged  : Auto-detect merged file based on data_type and graph_mode
  (none)        : Load from individual directories and merge (slowest)

Note: Run merge_process_datasets.py first to create merged files.
"""
    )

    # Training hyperparameters
    parser.add_argument('--innerdiv', type=int, default=10,
                       help='Inner learning rate divisor (default: 10)')
    parser.add_argument('--meta_lr', type=float, default=0.0001,
                       help='Meta learning rate (default: 0.0001)')
    parser.add_argument('--inner_steps', type=int, default=1,
                       help='Number of inner loop steps (default: 1)')

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
    parser.add_argument('--graph_mode', type=str, default='stage_aware',
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode (default: stage_aware)')

    # Use merged dataset (much faster loading)
    parser.add_argument('--use_merged', action='store_true',
                       help='Use pre-merged dataset (recommended for faster loading)')

    # Direct merged file path (new option - skips merge process entirely)
    parser.add_argument('--merged_file', type=str, default=None,
                       help='Path to pre-merged .pth file (e.g., merged_cell_stage_aware.pth). '
                            'When specified, loads data directly without merge process.')

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
    inner_steps = args.inner_steps
    innerdiv = args.innerdiv

    # Model architecture parameters
    conv_hidden_dim = args.conv_hidden_dim
    num_conv_layers = args.num_conv_layers
    fc_hidden_dim = args.fc_hidden_dim
    num_fc_layers = args.num_fc_layers

    # Calculate inner_lr
    base_lr = 0.001
    inner_lr = base_lr / innerdiv

    meta_lr = args.meta_lr
    K = args.K
    tasks_per_meta_batch = args.tasks_per_meta_batch
    data_type = args.data_type
    graph_mode = args.graph_mode

    start_time = time.time()

    print(f"\n🚀 MAML GNN Training with Cached Topology (Process-Aware)")
    print(f"📋 Configuration:")
    print(f"   Node features: 11D (7 base + 4 process)")
    print(f"   Data type: {data_type}")
    print(f"   Graph mode: {graph_mode}")
    print(f"   Inner div: {innerdiv} (inner_lr = {inner_lr})")
    print(f"   Meta LR: {meta_lr}")
    print(f"   Conv hidden dim: {conv_hidden_dim}")
    print(f"   Num conv layers: {num_conv_layers}")
    print(f"   FC hidden dim: {fc_hidden_dim}")
    print(f"   Num FC layers: {num_fc_layers}")
    print(f"   Inner steps: {inner_steps}")
    print(f"   K: {K}")
    print(f"   Tasks per meta batch: {tasks_per_meta_batch}")
    print(f"   Total iterations: {total_iterations}")
    print(f"   GPU: {args.gpu}")

    # Load cached GNN data (process-aware datasets)
    print("📊 Loading process-aware cached GNN data...")

    if args.merged_file:
        # Option 1: Load directly from pre-merged file (fastest, no merge process)
        print("   Using --merged_file option (direct loading, no merge process)")
        minimal_data_per_file, topology_cache, norm_stats = load_merged_data_direct(
            args.merged_file,
            apply_preprocessing=True
        )

    elif args.use_merged:
        # Option 2: Use pre-merged dataset with default path (legacy option)
        # Auto-detect merged file path based on data_type and graph_mode
        base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp_process"
        merged_file = os.path.join(base_path, f"merged_{data_type}_{graph_mode}.pth")

        if os.path.exists(merged_file):
            print(f"   Using pre-merged dataset: {merged_file}")
            minimal_data_per_file, topology_cache, norm_stats = load_merged_data_direct(
                merged_file,
                apply_preprocessing=True
            )
        else:
            print(f"   ⚠️ Merged file not found: {merged_file}")
            print(f"   Falling back to individual directory loading...")
            print(f"   💡 Tip: Run merge_process_datasets.py first to create merged file")
            minimal_data_per_file, topology_cache, norm_stats = load_cached_gnn_data_for_maml_process(
                data_type, graph_mode
            )

    else:
        # Option 3: Load from individual directories (slowest, original behavior)
        print("   Loading from individual directories (slower)...")
        print("   💡 Tip: Use --merged_file or --use_merged for faster loading")
        minimal_data_per_file, topology_cache, norm_stats = load_cached_gnn_data_for_maml_process(
            data_type, graph_mode
        )

    # Create GNN MAML model (11D node features for process-aware version)
    print("🤖 Creating GNN MAML model...")
    gnn_maml = GNNCachedMAML(
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
            gnn_maml.main_loop_optimized(num_iterations=chunk_size)
        except Exception as e:
            print(f"⚠️ Parallel processing failed, switching to sequential: {e}")
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

        # Memory monitoring
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        # Save checkpoint
        checkpoint_dir = "../../../pretrained_models/gnn_maml_checkpoints_process"
        os.makedirs(checkpoint_dir, exist_ok=True)
        iterations_completed = (chunk + 1) * chunk_size

        # Build model architecture suffix for filename
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

        checkpoint_path = f"{checkpoint_dir}/gnn_maml_process_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{iterations_completed}_inner{inner_steps}{arch_suffix}.pth"

        torch.save({
            'model_state_dict': gnn_maml.model.state_dict(),
            'norm_stats': norm_stats,
            'task_norm_stats': gnn_maml.task_norm_stats,  # Per-task output normalization
            'config': {
                'data_type': data_type,
                'graph_mode': graph_mode,
                'node_features': 11,  # Process-aware
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
    final_model_dir = "../../../pretrained_models/gnn_maml_final_process"
    os.makedirs(final_model_dir, exist_ok=True)

    # Build model architecture suffix for final model filename
    arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

    final_model_path = f"{final_model_dir}/gnn_maml_process_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{total_iterations}_inner{inner_steps}{arch_suffix}.pth"

    torch.save({
        'model_state_dict': gnn_maml.model.state_dict(),
        'norm_stats': norm_stats,
        'task_norm_stats': gnn_maml.task_norm_stats,  # Per-task output normalization
        'config': {
            'data_type': data_type,
            'graph_mode': graph_mode,
            'node_features': 11,  # Process-aware
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

    end_time = time.time()
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {(end_time - start_time)/3600:.2f} hours")
    print(f"Final model saved: {final_model_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
