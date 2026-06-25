#!/usr/bin/env python
"""
Multi-file GNN DataLoader for process-condition datasets

Designed for large datasets that are stored as multiple files.
Supports mmap loading and random task sampling across files.

Usage:
    from multifile_gnn_dataloader import MultiFileGNNDataLoader

    # Load from manifest
    loader = MultiFileGNNDataLoader(
        manifest_path="dataset_temp_process/manifest_cell_full_graph.json",
        split='train',
        batch_size=32
    )

    # Sample tasks for MAML
    for batch in loader:
        node_features, outputs, metadata = batch
        # node_features: [batch_size, num_libs, max_nodes, num_features]
        # outputs: [batch_size, num_libs]
"""

import torch
import json
import os
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path


class MultiFileGNNDataLoader:
    """
    DataLoader for multi-file GNN datasets with lazy loading and mmap support.

    Features:
    - Lazy loading: files are loaded only when needed
    - LRU cache: keeps recently used files in memory
    - mmap support: memory-mapped loading for large files
    - Random sampling: samples tasks uniformly across all files
    - Normalization: applies global normalization stats
    """

    def __init__(self, manifest_path: str, split: str = 'train',
                 batch_size: int = 32, cache_size: int = 5,
                 use_mmap: bool = True, normalize_output: bool = True,
                 topology_cache: Optional[Dict] = None,
                 shuffle: bool = True, seed: int = 42):
        """
        Initialize the DataLoader.

        Args:
            manifest_path: Path to manifest JSON file
            split: 'train' or 'test'
            batch_size: Number of tasks per batch
            cache_size: Number of files to keep in LRU cache
            use_mmap: Use memory-mapped loading
            normalize_output: Whether to normalize outputs
            topology_cache: Pre-loaded topology cache (optional)
            shuffle: Whether to shuffle files
            seed: Random seed
        """
        self.manifest_path = manifest_path
        self.split = split
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.use_mmap = use_mmap
        self.normalize_output = normalize_output
        self.topology_cache = topology_cache
        self.shuffle = shuffle

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.global_stats = self.manifest.get('global_norm_stats', {})
        self.data_type = self.manifest.get('data_type', 'cell')
        self.graph_mode = self.manifest.get('graph_mode', 'full_graph')

        # Build file list with task counts
        self._build_file_index()

        # LRU cache for loaded files
        self._file_cache = {}
        self._cache_order = []

        print(f"MultiFileGNNDataLoader initialized:")
        print(f"  Split: {split}")
        print(f"  Total files: {len(self.file_list)}")
        print(f"  Total tasks: {self.total_tasks}")
        print(f"  Batch size: {batch_size}")
        print(f"  Cache size: {cache_size}")
        print(f"  Use mmap: {use_mmap}")

    def _build_file_index(self):
        """Build index of files and task counts."""
        self.file_list = []
        self.task_counts = []
        self.cumulative_tasks = [0]

        file_key = 'train_file' if self.split == 'train' else 'test_file'

        for file_info in self.manifest['files']:
            file_path = file_info.get(file_key)
            if file_path and os.path.exists(file_path):
                # Quick load to get task count
                try:
                    data = torch.load(file_path, weights_only=False)
                    num_tasks = data.get('num_tasks', 0)

                    if num_tasks > 0:
                        self.file_list.append({
                            'path': file_path,
                            'folder_name': file_info.get('folder_name', ''),
                            'num_tasks': num_tasks
                        })
                        self.task_counts.append(num_tasks)
                        self.cumulative_tasks.append(self.cumulative_tasks[-1] + num_tasks)

                    del data

                except Exception as e:
                    print(f"  Warning: Could not load {file_path}: {e}")
                    continue

        self.total_tasks = sum(self.task_counts)

        if self.shuffle:
            # Create shuffled order
            self.file_order = list(range(len(self.file_list)))
            random.shuffle(self.file_order)
        else:
            self.file_order = list(range(len(self.file_list)))

    def _load_file(self, file_idx: int) -> Dict:
        """Load a file with caching."""
        if file_idx in self._file_cache:
            # Move to end of cache order (most recently used)
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._file_cache[file_idx]

        # Load file
        file_info = self.file_list[file_idx]
        file_path = file_info['path']

        if self.use_mmap:
            data = torch.load(file_path, weights_only=False, map_location='cpu')
        else:
            data = torch.load(file_path, weights_only=False)

        # Add to cache
        self._file_cache[file_idx] = data
        self._cache_order.append(file_idx)

        # Evict oldest if cache is full
        while len(self._cache_order) > self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            if oldest_idx in self._file_cache:
                del self._file_cache[oldest_idx]

        return data

    def _get_task(self, file_idx: int, task_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single task's data.

        Args:
            file_idx: Index of file in file_list
            task_idx: Index of task within file

        Returns:
            Tuple of (node_features, outputs, metadata)
            - node_features: [num_libs, num_nodes, num_features]
            - outputs: [num_libs]
            - metadata: dict with cell_name, delay_type, etc.
        """
        data = self._load_file(file_idx)

        # Get slices
        node_slices = data['node_slices']
        node_start = node_slices[task_idx].item()
        node_end = node_slices[task_idx + 1].item()

        # Extract node features for this task: [num_libs, num_nodes, num_features]
        node_features = data['node_features'][:, node_start:node_end, :]

        # Extract outputs for this task: [num_libs]
        outputs = data['outputs'][:, task_idx]

        # Normalize outputs if requested
        if self.normalize_output and self.global_stats:
            output_stats = self.global_stats.get('output_stats', {})
            mean = output_stats.get('mean', 0.0)
            std = output_stats.get('std', 1.0)
            if std > 0:
                outputs = (outputs - mean) / std

        # Metadata
        metadata = {
            'cell_name': data['cell_names'][task_idx],
            'delay_type': data['delay_types'][task_idx],
            'output_name': data['output_names'][task_idx],
            'num_nodes': node_end - node_start,
            'file_idx': file_idx,
            'task_idx': task_idx,
            'folder_name': self.file_list[file_idx]['folder_name']
        }

        return node_features, outputs, metadata

    def sample_batch(self, batch_size: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
        """
        Sample a random batch of tasks from across all files.

        Args:
            batch_size: Number of tasks (default: self.batch_size)

        Returns:
            Tuple of lists: (node_features_list, outputs_list, metadata_list)
        """
        if batch_size is None:
            batch_size = self.batch_size

        node_features_list = []
        outputs_list = []
        metadata_list = []

        # Sample random tasks
        for _ in range(batch_size):
            # Random file (weighted by task count for uniform task sampling)
            file_idx = random.choices(
                range(len(self.file_list)),
                weights=self.task_counts,
                k=1
            )[0]

            # Random task within file
            num_tasks = self.file_list[file_idx]['num_tasks']
            task_idx = random.randint(0, num_tasks - 1)

            # Get task data
            node_features, outputs, metadata = self._get_task(file_idx, task_idx)

            node_features_list.append(node_features)
            outputs_list.append(outputs)
            metadata_list.append(metadata)

        return node_features_list, outputs_list, metadata_list

    def get_task_by_global_idx(self, global_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get task by global index (across all files).

        Args:
            global_idx: Global task index

        Returns:
            Tuple of (node_features, outputs, metadata)
        """
        # Find which file this task belongs to
        file_idx = 0
        for i, cum in enumerate(self.cumulative_tasks[1:], 1):
            if global_idx < cum:
                file_idx = i - 1
                break

        # Local task index within file
        local_idx = global_idx - self.cumulative_tasks[file_idx]

        return self._get_task(file_idx, local_idx)

    def __len__(self) -> int:
        """Return number of batches."""
        return (self.total_tasks + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        """Iterate over batches."""
        # For deterministic iteration, go through files in order
        for file_idx in self.file_order:
            file_info = self.file_list[file_idx]
            num_tasks = file_info['num_tasks']

            # Batch tasks within this file
            task_indices = list(range(num_tasks))
            if self.shuffle:
                random.shuffle(task_indices)

            for batch_start in range(0, num_tasks, self.batch_size):
                batch_indices = task_indices[batch_start:batch_start + self.batch_size]

                node_features_list = []
                outputs_list = []
                metadata_list = []

                for task_idx in batch_indices:
                    node_features, outputs, metadata = self._get_task(file_idx, task_idx)
                    node_features_list.append(node_features)
                    outputs_list.append(outputs)
                    metadata_list.append(metadata)

                yield node_features_list, outputs_list, metadata_list

    def get_topology_for_task(self, metadata: Dict) -> Optional[Dict]:
        """
        Get topology cache entry for a task.

        Args:
            metadata: Task metadata dict

        Returns:
            Topology cache entry or None
        """
        if self.topology_cache is None:
            return None

        cell_name = metadata.get('cell_name')
        if cell_name and cell_name in self.topology_cache:
            return self.topology_cache[cell_name]

        return None

    def clear_cache(self):
        """Clear file cache."""
        self._file_cache.clear()
        self._cache_order.clear()

    def get_output_denorm_params(self) -> Tuple[float, float]:
        """Get parameters for denormalizing outputs."""
        output_stats = self.global_stats.get('output_stats', {})
        mean = output_stats.get('mean', 0.0)
        std = output_stats.get('std', 1.0)
        return mean, std


class MAMLMultiFileDataLoader(MultiFileGNNDataLoader):
    """
    MAML-specific DataLoader that samples support and query sets.
    """

    def __init__(self, manifest_path: str, split: str = 'train',
                 n_way: int = 1, k_shot: int = 5, q_query: int = 5,
                 **kwargs):
        """
        Initialize MAML DataLoader.

        Args:
            manifest_path: Path to manifest JSON
            split: 'train' or 'test'
            n_way: Number of classes (usually 1 for regression)
            k_shot: Number of support samples
            q_query: Number of query samples
            **kwargs: Additional args for parent class
        """
        super().__init__(manifest_path, split, batch_size=1, **kwargs)

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # For regression MAML, we need k_shot + q_query voltage points
        # Each "task" is one cell, and we sample voltage points as support/query
        self.total_samples_per_task = k_shot + q_query

    def sample_maml_task(self) -> Dict:
        """
        Sample a single MAML task with support and query sets.

        For voltage-based MAML:
        - Each task is one (cell, delay_type) combination
        - Support set: k_shot voltage points
        - Query set: q_query voltage points

        Returns:
            Dict with support and query data
        """
        # Sample random task
        file_idx = random.choices(
            range(len(self.file_list)),
            weights=self.task_counts,
            k=1
        )[0]

        num_tasks = self.file_list[file_idx]['num_tasks']
        task_idx = random.randint(0, num_tasks - 1)

        # Get task data
        node_features, outputs, metadata = self._get_task(file_idx, task_idx)

        # node_features: [num_libs, num_nodes, num_features]
        # outputs: [num_libs]
        num_libs = node_features.shape[0]

        # Sample voltage indices for support and query
        if num_libs < self.total_samples_per_task:
            # If not enough voltage points, use all with replacement
            all_indices = list(range(num_libs))
            support_indices = random.choices(all_indices, k=self.k_shot)
            query_indices = random.choices(all_indices, k=self.q_query)
        else:
            # Sample without replacement
            all_indices = list(range(num_libs))
            random.shuffle(all_indices)
            support_indices = all_indices[:self.k_shot]
            query_indices = all_indices[self.k_shot:self.k_shot + self.q_query]

        # Build support set
        support_x = node_features[support_indices]  # [k_shot, num_nodes, num_features]
        support_y = outputs[support_indices]  # [k_shot]

        # Build query set
        query_x = node_features[query_indices]  # [q_query, num_nodes, num_features]
        query_y = outputs[query_indices]  # [q_query]

        return {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y,
            'metadata': metadata,
            'support_indices': support_indices,
            'query_indices': query_indices
        }

    def sample_maml_batch(self, batch_size: int) -> List[Dict]:
        """
        Sample a batch of MAML tasks.

        Args:
            batch_size: Number of tasks

        Returns:
            List of task dicts
        """
        return [self.sample_maml_task() for _ in range(batch_size)]


def load_topology_cache(cache_path: str) -> Dict:
    """Load topology cache."""
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    return {}


# Test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True,
                       help="Path to manifest JSON file")
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'test'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batches', type=int, default=3)

    args = parser.parse_args()

    print(f"\nTesting MultiFileGNNDataLoader...")
    print(f"Manifest: {args.manifest}")
    print(f"Split: {args.split}")

    loader = MultiFileGNNDataLoader(
        manifest_path=args.manifest,
        split=args.split,
        batch_size=args.batch_size
    )

    print(f"\nSampling {args.test_batches} random batches...")
    for i in range(args.test_batches):
        node_features, outputs, metadata = loader.sample_batch()

        print(f"\nBatch {i+1}:")
        print(f"  Tasks: {len(node_features)}")
        for j, (nf, out, meta) in enumerate(zip(node_features, outputs, metadata)):
            print(f"    Task {j}: {meta['cell_name']} ({meta['folder_name']})")
            print(f"      node_features: {nf.shape}")
            print(f"      outputs: {out.shape}")

    print("\n\nTesting MAMLMultiFileDataLoader...")
    maml_loader = MAMLMultiFileDataLoader(
        manifest_path=args.manifest,
        split=args.split,
        k_shot=5,
        q_query=5
    )

    print(f"\nSampling MAML tasks...")
    for i in range(2):
        task = maml_loader.sample_maml_task()
        print(f"\nMAML Task {i+1}:")
        print(f"  Cell: {task['metadata']['cell_name']}")
        print(f"  Support X: {task['support_x'].shape}")
        print(f"  Support Y: {task['support_y'].shape}")
        print(f"  Query X: {task['query_x'].shape}")
        print(f"  Query Y: {task['query_y'].shape}")

    print("\nDone!")
