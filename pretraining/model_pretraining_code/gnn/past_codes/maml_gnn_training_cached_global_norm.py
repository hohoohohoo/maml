#!/usr/bin/env python
"""
MAML GNN Training with Cached Topology - MINIMAL DATASET VERSION (Global Normalization)
Supports both full_graph and stage_aware modes with topology cache

Key Features:
- Minimal dataset: only node_features + metadata stored
- Topology reconstruction from cache on-the-fly
- Full graph: cell_name -> topology lookup
- Stage-aware: cell_name + output_name + delay_type -> topology lookup
- GLOBAL output normalization (NOT per-task)
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

class GNNCachedMAML_GlobalNorm:
    """
    GNN MAML with cached topology - reconstructs graphs on-the-fly
    Uses GLOBAL output normalization (NOT per-task)
    """

    def __init__(self, model, inner_lr, meta_lr, K=5, inner_steps=1,
                 minimal_data_per_file=None, topology_cache=None,
                 cache_type='full_graph', norm_stats=None,
                 tasks_per_meta_batch=16,
                 global_output_mean=None, global_output_std=None):

        # Dataset (minimal format)
        self.minimal_data_per_file = minimal_data_per_file  # [lib_files][samples]
        self.topology_cache = topology_cache
        self.cache_type = cache_type  # 'full_graph' or 'stage_aware'
        self.norm_stats = norm_stats

        # Build stacked outputs from minimal data
        self.stacked_outputs = self._build_stacked_outputs()

        self.num_tasks = self.stacked_outputs.shape[0]  # Number of input conditions
        self.lib_files_per_task = self.stacked_outputs.shape[1]  # Number of process variations

        # Pre-normalize all outputs using GLOBAL statistics
        if global_output_mean is not None and global_output_std is not None:
            print(f"📊 Using pre-computed global normalization stats...")
            self.global_output_mean = global_output_mean
            self.global_output_std = global_output_std
            self.normalized_outputs = self._normalize_with_global_stats(self.stacked_outputs)
            print(f"   ✓ Global output normalization: mean={self.global_output_mean:.6f}, std={self.global_output_std:.6f}")
        else:
            self.normalized_outputs = self.normalize_all_outputs_globally(self.stacked_outputs)

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

        print(f"🎯 GNN Cached MAML Configuration (Global Normalization):")
        print(f"   Cache type: {self.cache_type}")
        print(f"   Tasks: {self.num_tasks} (input conditions)")
        print(f"   Lib files per task: {self.lib_files_per_task}")
        print(f"   Support set size (K): {self.K}")
        print(f"   Inner steps: {self.inner_steps}")
        print(f"   Tasks per meta batch: {self.tasks_per_meta_batch}")
        print(f"   Output normalization: GLOBAL (all tasks)")

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

    def _normalize_with_global_stats(self, stacked_outputs):
        """Normalize outputs using pre-computed global statistics"""
        if stacked_outputs is None:
            return None

        if self.global_output_std > 1e-8:
            normalized = (stacked_outputs - self.global_output_mean) / self.global_output_std
        else:
            normalized = stacked_outputs - self.global_output_mean

        return normalized

    def normalize_all_outputs_globally(self, stacked_outputs):
        """
        Pre-normalize all outputs using GLOBAL statistics
        This is the key difference from per-task normalization
        """
        if stacked_outputs is None:
            return None

        print(f"📊 Computing GLOBAL normalization statistics...")

        # Calculate global mean and std across ALL outputs
        self.global_output_mean = stacked_outputs.mean().item()
        self.global_output_std = stacked_outputs.std().item()

        print(f"   📊 Global output statistics:")
        print(f"      Mean: {self.global_output_mean:.6f}")
        print(f"      Std: {self.global_output_std:.6f}")
        print(f"      Total samples: {stacked_outputs.numel()}")

        # Normalize all outputs using global statistics
        if self.global_output_std > 1e-8:
            normalized = (stacked_outputs - self.global_output_mean) / self.global_output_std
        else:
            # If std is too small, just center the data
            normalized = stacked_outputs - self.global_output_mean
            print(f"   ⚠️ Global std too small, using centering only")

        print(f"✅ Global output normalization complete")
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
        Uses globally-normalized outputs
        """
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} out of range (max: {self.num_tasks-1})")

        minimal_samples = []
        outputs = []

        # Collect minimal samples from all lib files for this task
        for lib_idx in range(self.lib_files_per_task):
            if task_id < len(self.minimal_data_per_file[lib_idx]):
                minimal_sample = self.minimal_data_per_file[lib_idx][task_id]

                # Use globally-normalized outputs
                output = self.normalized_outputs[task_id, lib_idx].item()

                minimal_samples.append(minimal_sample)
                outputs.append(output)

        return minimal_samples, outputs

    def inner_loop_single_task(self, task_idx):
        """Inner loop for single task"""
        temp_weights = [w.clone() for w in self.weights]

        # Get task data (minimal samples + globally-normalized outputs)
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

            # Meta gradient computation and update
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # Gradient clipping to prevent explosion
            max_grad_norm = 1.0
            meta_grads = [torch.clamp(g, -max_grad_norm, max_grad_norm) for g in meta_grads]

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

        # Convert /mnt/home/ to /home/ for cache_path if needed
        if cache_path and cache_path.startswith('/mnt/home/'):
            cache_path = cache_path.replace('/mnt/home/', '/home/')
            print(f"     Converted cache /mnt/home/ → /home/")

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
        description='MAML GNN Training with Cached Topology (Global Normalization)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python maml_gnn_training_cached_global_norm.py --process SRAM --corner FF --graph_mode stage_aware
  python maml_gnn_training_cached_global_norm.py --process LVT --corner TT --graph_mode full_graph
"""
    )

    # Run all combinations option
    parser.add_argument('--run_all', action='store_true',
                       help='Run all 12 combinations (4 process × 3 corners)')

    # Required arguments (unless --run_all is used)
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
    """Train a single process-corner configuration with architecture sweep support

    Args:
        process: Process type (RVT, LVT, SLVT, SRAM)
        corner_type: Process corner (TT, FF, SS)
        args: Command-line arguments

    Returns:
        Path to the saved final model
    """
    import itertools

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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

    print(f"\n🚀 MAML GNN Training with Cached Topology (Global Normalization)")
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
    print(f"   ⚠️  Output normalization: GLOBAL (NOT per-task)")

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

    # Load cached GNN data ONCE (shared across all architecture combinations)
    print("\n📊 Loading cached GNN data (once for all architectures)...")
    data_load_start = time.time()
    minimal_data_per_file, topology_cache, norm_stats = load_cached_gnn_data_for_maml(
        process, corner_type, data_type, graph_mode
    )
    data_load_time = time.time() - data_load_start
    print(f"✅ Data loaded in {data_load_time/60:.2f} minutes")

    # Pre-compute GLOBAL normalization statistics ONCE
    print("\n📊 Pre-computing GLOBAL normalization statistics...")
    num_libs = len(minimal_data_per_file)
    num_tasks = len(minimal_data_per_file[0])

    # Collect ALL outputs
    all_outputs = []
    for lib_idx in range(num_libs):
        for task_idx in range(num_tasks):
            sample = minimal_data_per_file[lib_idx][task_idx]
            all_outputs.append(sample['output'])

    # Calculate global statistics
    all_outputs_tensor = torch.tensor(all_outputs, dtype=torch.float32)
    global_output_mean = all_outputs_tensor.mean().item()
    global_output_std = all_outputs_tensor.std().item()

    if global_output_std < 1e-8:
        global_output_std = 1.0
        print(f"   ⚠️ Global output std too small, using 1.0")

    print(f"   📊 Global output statistics:")
    print(f"      Mean: {global_output_mean:.6f}")
    print(f"      Std: {global_output_std:.6f}")
    print(f"      Total samples: {len(all_outputs)}")
    print(f"   ✅ Global normalization stats computed")

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

        # Create GNN MAML model
        print("🤖 Creating GNN MAML model (Global Normalization)...")
        gnn_maml = GNNCachedMAML_GlobalNorm(
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
            tasks_per_meta_batch=tasks_per_meta_batch,
            global_output_mean=global_output_mean,
            global_output_std=global_output_std
        )

        # Training in chunks
        for chunk in range(num_chunks):
            print(f"\n📦 Processing chunk {chunk+1}/{num_chunks}")
            chunk_start_time = time.time()

            try:
                gnn_maml.main_loop_sequential(num_iterations=chunk_size)
            except Exception as e:
                print(f"⚠️ Sequential processing failed: {e}")
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
            checkpoint_dir = "../../../pretrained_models/gnn_maml_checkpoints_global_norm"
            os.makedirs(checkpoint_dir, exist_ok=True)
            iterations_completed = (chunk + 1) * chunk_size

            # Build model architecture suffix for filename
            arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

            checkpoint_path = f"{checkpoint_dir}/gnn_maml_global_norm_{process}_{corner_type}_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{iterations_completed}_inner{inner_steps}{arch_suffix}.pth"

            torch.save({
                'model_state_dict': gnn_maml.model.state_dict(),
                'norm_stats': norm_stats,
                'global_output_mean': global_output_mean,
                'global_output_std': global_output_std,
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
                    'meta_losses': gnn_maml.meta_losses,
                    'normalization_type': 'global'
                }
            }, checkpoint_path)
            print(f"✅ Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_model_dir = "../../../pretrained_models/gnn_maml_final_global_norm"
        os.makedirs(final_model_dir, exist_ok=True)

        # Build model architecture suffix for final model filename
        arch_suffix = f"_conv{conv_hidden_dim}x{num_conv_layers}_fc{fc_hidden_dim}x{num_fc_layers}"

        final_model_path = f"{final_model_dir}/gnn_maml_global_norm_{process}_{corner_type}_{data_type}_{graph_mode}_innerdiv{innerdiv}_meta{tasks_per_meta_batch}_iter{total_iterations}_inner{inner_steps}{arch_suffix}.pth"

        torch.save({
            'model_state_dict': gnn_maml.model.state_dict(),
            'norm_stats': norm_stats,
            'global_output_mean': global_output_mean,
            'global_output_std': global_output_std,
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
                'meta_losses': gnn_maml.meta_losses,
                'normalization_type': 'global'
            }
        }, final_model_path)

        print(f"\n🧹 Cleaning up memory after this architecture...")

        # GPU memory cleanup - delete model-specific data structures
        del gnn_maml.stacked_outputs
        del gnn_maml.normalized_outputs
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
    del all_outputs
    del all_outputs_tensor
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
    """Main function - handles both single run and batch run modes"""
    args = parse_arguments()

    # Validate arguments
    if not args.run_all:
        if not args.process or not args.corner:
            print("Error: --process and --corner are required unless --run_all is used")
            sys.exit(1)

    if args.run_all:
        # Run all 12 combinations
        all_processes = ['RVT', 'LVT', 'SLVT', 'SRAM']
        all_corners = ['TT', 'FF', 'SS']

        total_configs = len(all_processes) * len(all_corners)
        completed = 0
        failed = []

        print(f"\n{'='*80}")
        print(f"🚀 Running ALL {total_configs} configurations (MAML Global Normalization)")
        print(f"{'='*80}")
        print(f"Processes: {', '.join(all_processes)}")
        print(f"Corners: {', '.join(all_corners)}")
        print(f"{'='*80}\n")

        overall_start = time.time()

        for process in all_processes:
            for corner in all_corners:
                config_num = completed + len(failed) + 1
                print(f"\n{'#'*80}")
                print(f"# Configuration {config_num}/{total_configs}: {process} + {corner}")
                print(f"{'#'*80}\n")

                try:
                    model_path = train_single_config(process, corner, args)
                    completed += 1
                    print(f"\n✅ [{config_num}/{total_configs}] Completed: {process}_{corner}")
                    print(f"   Saved to: {model_path}")

                except Exception as e:
                    failed.append(f"{process}_{corner}")
                    print(f"\n❌ [{config_num}/{total_configs}] Failed: {process}_{corner}")
                    print(f"   Error: {str(e)}")
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

        overall_end = time.time()

        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 BATCH TRAINING SUMMARY (MAML Global Normalization)")
        print(f"{'='*80}")
        print(f"Total configurations: {total_configs}")
        print(f"✅ Completed: {completed}")
        print(f"❌ Failed: {len(failed)}")
        if failed:
            print(f"\nFailed configurations:")
            for config in failed:
                print(f"  - {config}")
        print(f"\n⏱️ Total time: {(overall_end - overall_start)/3600:.2f} hours")
        print(f"{'='*80}\n")

    else:
        # Run single configuration
        train_single_config(args.process, args.corner, args)


if __name__ == "__main__":
    main()
