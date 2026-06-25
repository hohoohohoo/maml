#!/usr/bin/env python
"""
Merge all process-aware datasets into a single file for efficient loading

This script merges 384×2 directories (for stage_aware and full_graph) into:
- dataset_temp_process/merged_cell_stage_aware.pth
- dataset_temp_process/merged_cell_full_graph.pth

Benefits:
- Load once instead of 384×2 times
- Much faster training initialization
- Easier to manage
"""

import os
import torch
import gc
import random
from tqdm import tqdm
import argparse

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def merge_datasets(data_type='cell', graph_mode='stage_aware', output_dir=None,
                   sample_ratio=1.0, seed=42):
    """
    Merge all non-test datasets in dataset_temp_process

    Args:
        data_type: 'cell' or 'transition'
        graph_mode: 'stage_aware' or 'full_graph'
        output_dir: Output directory (default: same as source)
        sample_ratio: Ratio of samples to keep (0.0-1.0, default: 1.0 = all)
        seed: Random seed for reproducibility
    """

    base_path = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_temp_process"

    if output_dir is None:
        output_dir = base_path

    # Set random seed for reproducibility
    random.seed(seed)

    print(f"\n{'='*80}")
    print(f"MERGING PROCESS-AWARE DATASETS")
    print(f"{'='*80}")
    print(f"Source directory: {base_path}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Sample ratio: {sample_ratio:.1%}" + (" (random sampling)" if sample_ratio < 1.0 else " (all samples)"))
    print(f"Random seed: {seed}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Find all non-test dataset directories
    all_items = os.listdir(base_path)
    matching_folders = []

    data_filename = f"{data_type}_all_graph_data_{graph_mode}.pth"

    print("🔍 Scanning for datasets...")
    for item in sorted(all_items):
        item_path = os.path.join(base_path, item)

        # Skip if not directory
        if not os.path.isdir(item_path):
            continue

        # Skip test datasets
        if 'test' in item.lower():
            print(f"   ⏭️  Skipping test dataset: {item}")
            continue

        # Check if data file exists
        data_file_path = f"{base_path}/{item}/graph_data/{data_filename}"
        if os.path.exists(data_file_path):
            matching_folders.append((item, data_file_path))
        else:
            print(f"   ⚠️  Data file not found: {item}/{data_filename}")

    if not matching_folders:
        raise ValueError(f"No data found in {base_path}")

    print(f"\n✅ Found {len(matching_folders)} non-test datasets to merge")
    print(f"{'='*80}\n")

    # Merge datasets
    all_minimal_data_per_file = []
    cache_path = None
    total_samples = 0

    print("📦 Loading and merging datasets...")
    print("💡 Memory-efficient mode: Loading cache_path only (not topology_cache itself)")

    for folder, data_file_path in tqdm(matching_folders, desc="Merging"):
        # Load dataset (memory-efficient)
        full_data = torch.load(data_file_path, weights_only=False, map_location='cpu')

        # Extract only what we need
        minimal_data_per_file = full_data['minimal_data_per_file']

        # Apply random sampling if sample_ratio < 1.0
        if sample_ratio < 1.0:
            sampled_minimal_data = []
            for lib_samples in minimal_data_per_file:
                num_to_keep = max(1, int(len(lib_samples) * sample_ratio))
                sampled = random.sample(lib_samples, num_to_keep)
                sampled_minimal_data.append(sampled)
            minimal_data_per_file = sampled_minimal_data

        # Store cache_path (only once, don't load topology_cache)
        if cache_path is None:
            cache_path = full_data.get('cache_path', None)

            # Convert to absolute path if needed
            if cache_path and not os.path.isabs(cache_path):
                cache_filename = os.path.basename(cache_path)
                cache_file = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/{cache_filename}"
                if os.path.exists(cache_file):
                    cache_path = cache_file
                    print(f"   📌 Topology cache path: {cache_path}")

        # Initialize if first dataset
        if not all_minimal_data_per_file:
            all_minimal_data_per_file = [[] for _ in range(len(minimal_data_per_file))]

        # Merge minimal data
        num_samples = len(minimal_data_per_file[0])
        for lib_idx, lib_samples in enumerate(minimal_data_per_file):
            all_minimal_data_per_file[lib_idx].extend(lib_samples)

        total_samples += num_samples

        # Aggressive memory cleanup
        del full_data
        del minimal_data_per_file
        gc.collect()

        # Progress update every 50 files to reduce overhead
        if PSUTIL_AVAILABLE and (len(matching_folders) > 50) and ((total_samples // num_samples) % 50 == 0):
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"   💾 Memory usage: {mem_mb:.1f} MB")

    # Final memory report
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"\n💾 Final memory usage: {mem_mb:.1f} MB")

    print(f"\n{'='*80}")
    print(f"MERGE SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets merged: {len(matching_folders)}")
    print(f"Total samples (tasks): {len(all_minimal_data_per_file[0])}")
    print(f"Lib files per task: {len(all_minimal_data_per_file)}")
    print(f"Topology cache path: {cache_path if cache_path else 'Not found'}")
    print(f"{'='*80}\n")

    # Save merged dataset
    if sample_ratio < 1.0:
        ratio_str = f"_sampled{int(sample_ratio*100)}"
        output_filename = f"merged_{data_type}_{graph_mode}{ratio_str}.pth"
    else:
        output_filename = f"merged_{data_type}_{graph_mode}.pth"
    output_path = os.path.join(output_dir, output_filename)

    print(f"💾 Saving merged dataset to: {output_path}")

    merged_data = {
        'minimal_data_per_file': all_minimal_data_per_file,
        'cache_path': cache_path,
        'metadata': {
            'data_type': data_type,
            'graph_mode': graph_mode,
            'num_datasets': len(matching_folders),
            'num_samples': len(all_minimal_data_per_file[0]),
            'num_lib_files': len(all_minimal_data_per_file),
            'source_datasets': [folder for folder, _ in matching_folders],
            'sample_ratio': sample_ratio,
            'random_seed': seed
        }
    }

    torch.save(merged_data, output_path)

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"✅ Merged dataset saved!")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Path: {output_path}")
    print(f"\n{'='*80}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Merge process-aware datasets into single files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge cell data for stage_aware mode (all samples)
  python merge_process_datasets.py --data_type cell --graph_mode stage_aware

  # Merge with 10%% random sampling (reduces file size by 10x)
  python merge_process_datasets.py --data_type cell --graph_mode stage_aware --sample_ratio 0.1

  # Merge with 20%% sampling and custom seed
  python merge_process_datasets.py --graph_mode full_graph --sample_ratio 0.2 --seed 123

  # Merge all modes with 10%% sampling
  python merge_process_datasets.py --all --sample_ratio 0.1
"""
    )

    parser.add_argument('--data_type', type=str, default='cell',
                       choices=['cell', 'transition'],
                       help='Data type (default: cell)')
    parser.add_argument('--graph_mode', type=str,
                       choices=['stage_aware', 'full_graph'],
                       help='Graph mode')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (default: same as source)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Ratio of samples to keep per file (0.0-1.0, default: 1.0 = all). '
                            'Use 0.1 for 10%% sampling to reduce file size.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--all', action='store_true',
                       help='Merge both stage_aware and full_graph')

    args = parser.parse_args()

    # Validate sample_ratio
    if args.sample_ratio <= 0.0 or args.sample_ratio > 1.0:
        print(f"Error: --sample_ratio must be between 0.0 (exclusive) and 1.0 (inclusive)")
        return

    if args.all:
        # Merge both modes
        for graph_mode in ['stage_aware', 'full_graph']:
            print(f"\n{'#'*80}")
            print(f"# Processing {graph_mode}")
            print(f"{'#'*80}\n")
            merge_datasets(
                data_type=args.data_type,
                graph_mode=graph_mode,
                output_dir=args.output_dir,
                sample_ratio=args.sample_ratio,
                seed=args.seed
            )
    else:
        if not args.graph_mode:
            print("Error: --graph_mode is required unless --all is specified")
            print("Use --help for usage information")
            return

        merge_datasets(
            data_type=args.data_type,
            graph_mode=args.graph_mode,
            output_dir=args.output_dir,
            sample_ratio=args.sample_ratio,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
