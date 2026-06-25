#!/usr/bin/env python
"""
Unified cached version of build_gnn_dataset
Supports both full_graph and stage_aware modes
Pre-computed cell topology를 사용하여 빠르게 dataset 생성

최소 저장 방식:
    - node_features (matrix): 각 sample의 node feature matrix
    - cell_name: topology cache에서 graph structure를 가져올 key
    - output: target value (delay or slew)

사용법:
    1. 먼저 topology cache 생성:
       Full-graph: python precompute_cell_topology.py --cdl_path <cdl> --output <cache.pth>
       Stage-aware: python precompute_stage_aware_topology.py --cdl_path <cdl> --output <cache.pth>

    2. Dataset 생성 (cached version - auto-detects cache type):
       python build_gnn_dataset_cached.py --cache_path <cache.pth> --lib_dir <lib_files>

데이터셋 구조:
    {
        'cache_path': str,  # topology cache 파일 경로
        'cache_type': str,  # 'full_graph' or 'stage_aware'
        'minimal_data_per_file': [
            [
                {
                    'node_features': torch.Tensor,  # [num_nodes, 7]
                    'cell_name': str,  # topology 조회용
                    'output': float  # target value
                },
                ...
            ],
            ...
        ],
        'num_lib_files': int,
        'data_type': str  # 'cell' or 'transition'
    }
"""

import torch
from pathlib import Path
import sys
import os
import argparse

# Add utils path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import cache loaders and apply function for node feature generation
from precompute_full_graph_topology import load_cell_topology_cache, apply_topology_to_sample


def detect_cache_type(topology_cache):
    """
    Detect whether cache is full_graph or stage_aware type

    Returns:
        str: 'full_graph' or 'stage_aware'
    """
    if not topology_cache:
        raise ValueError("Empty topology cache")

    # Get first cell to check structure
    first_cell_name = next(iter(topology_cache))
    first_cell = topology_cache[first_cell_name]

    # Stage-aware has 'output_topologies' key
    if 'output_topologies' in first_cell:
        return 'stage_aware'
    # Full-graph has 'all_nodes' key directly
    elif 'all_nodes' in first_cell:
        return 'full_graph'
    else:
        raise ValueError(f"Unknown cache structure for cell {first_cell_name}")


def dataextract_gnn_cached(lib_file_path, topology_cache, lib_prefix="", data_type="cell", cache_type=None):
    """
    Extract node features from .lib file using cached topology
    Only stores: node_features (matrix), cell_name (for topology lookup)

    Args:
        lib_file_path: Path to .lib file
        topology_cache: Pre-computed cell topology cache
        lib_prefix: Library file prefix
        data_type: 'cell' or 'transition' to select parser
        cache_type: 'full_graph' or 'stage_aware' (auto-detected if None)

    Returns:
        list: List of dicts with node_features and cell_name only
    """

    # Detect cache type if not provided
    if cache_type is None:
        cache_type = detect_cache_type(topology_cache)
        print(f"   🔍 Detected cache type: {cache_type}")

    # Import appropriate parser based on data_type
    if data_type == "cell":
        from libdata_extract_MAML_cell import parse_liberty_pin_blocks, flatten_pin_data
    elif data_type == "transition":
        from libdata_extract_MAML_transition import parse_liberty_pin_blocks, flatten_pin_data
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'cell' or 'transition'")

    print(f"   📂 Reading .lib file: {lib_file_path}")

    with open(lib_file_path, "r") as f:
        lines = f.readlines()

    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)

    print(f"   ✓ Parsed {len(flattened)} timing entries")

    # Generate node features using apply_topology_to_sample
    # Store only node_features + cell_name for minimal dataset
    minimal_samples = []
    skipped_count = 0
    cached_count = 0
    skipped_cells = set()  # Track unique skipped cell names

    for sample in flattened:
        cell_name = sample['cell']
        voltage = sample['Voltage']
        delay_type = sample['delay_type']
        input_port_names = sample['input_port_name']
        output_port_name = sample.get('output_port_name', 'Y')

        # Check if cell exists in cache
        if cell_name not in topology_cache:
            skipped_count += 1
            skipped_cells.add(cell_name)  # Track this skipped cell
            continue

        cached_count += 1

        # Get index dimensions
        input_slews = sample['index_1']
        output_loads = sample['index_2']
        timing_values = sample['values']

        # Handle values array size mismatch
        actual_rows = len(timing_values) if isinstance(timing_values, list) else 0
        actual_cols = len(timing_values[0]) if actual_rows > 0 and isinstance(timing_values[0], list) else 0

        effective_rows = min(len(input_slews), actual_rows) if actual_rows > 0 else len(input_slews)
        effective_cols = min(len(output_loads), actual_cols) if actual_cols > 0 else len(output_loads)

        # Generate node features for each runtime parameter set
        for row_idx in range(effective_rows):
            for col_idx in range(effective_cols):
                input_slew = input_slews[row_idx]
                output_load = output_loads[col_idx]
                output_value = timing_values[row_idx][col_idx]

                # Call apply_topology_to_sample to generate full graph
                graph_sample = apply_topology_to_sample(
                    topology_cache,
                    cell_name,
                    voltage,
                    input_slew,
                    output_load,
                    output_value,
                    input_port_names
                )

                # Store only node features and cell name (minimal dataset)
                minimal_samples.append({
                    'node_features': graph_sample['node_features'],  # Node feature matrix
                    'cell_name': cell_name,  # For topology cache lookup
                    'output': output_value,  # Target value
                    'delay_type' : delay_type
                })

    print(f"   ✓ Generated {len(minimal_samples)} node feature samples ({cached_count} cells cached, {skipped_count} cells skipped)")

    # Print skipped cells if any
    if skipped_cells:
        print(f"   ⚠️  Skipped {len(skipped_cells)} unique cells not in topology cache:")
        for cell in sorted(skipped_cells):
            print(f"      - {cell}")

    return minimal_samples


def build_all_gnn_data_cached(
        cache_path,
        start=40,
        end=101,
        prefix="OA_LVT_2_25_",
        save_input="graph_data_output.pth",  # Will be modified based on data_type
        save_output="graph_output_data_cached.pth",  # Deprecated, not used
        data_dir="OA_LVT",
        lib_base_path="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/voltage_variation",
        data_type="cell"):
    """
    Build GNN graph dataset from multiple .lib files using cached topology

    Args:
        cache_path: Path to pre-computed topology cache (.pth)
        start, end: Range of voltage indices
        prefix: Library file prefix
        save_input, save_output: Output file paths
        data_dir: Directory containing lib files
        lib_base_path: Base path to lib files
        data_type: 'cell' or 'transition' to select parser
    """

    print("=" * 80)
    print("BUILD GNN DATASET (CACHED TOPOLOGY)")
    print("=" * 80)
    print(f"Topology cache: {cache_path}")
    print(f"Lib files: {lib_base_path}/{data_dir}/{prefix}[{start:03d}-{end-1:03d}].lib")
    print(f"Data type: {data_type}")
    print("=" * 80)

    # Load topology cache (try to auto-detect cache type)
    print(f"\n📂 Loading topology cache...")
    try:
        # Try loading as torch.load first to detect type
        raw_cache = torch.load(cache_path, weights_only=False)
        cache_type = detect_cache_type(raw_cache)
        topology_cache = raw_cache
        print(f"   ✓ Loaded {len(topology_cache)} cells (cache type: {cache_type})")
    except Exception as e:
        print(f"   ❌ Error loading cache: {e}")
        raise

    # Store only minimal samples (node features + cell name)
    minimal_data_per_file = []

    for i in range(start, end):
        v_str = f"{i//100}{i%100:02d}"
        filename = f"{lib_base_path}/{data_dir}/{prefix}{v_str}.lib"

        print(f"\n📥 Processing [{i-start+1}/{end-start}]: {filename}")

        try:
            # Extract minimal samples (node features + cell name)
            minimal_samples = dataextract_gnn_cached(filename, topology_cache, prefix, data_type, cache_type)

            if minimal_samples:
                minimal_data_per_file.append(minimal_samples)
                print(f"   ✓ File {i-start+1}: {len(minimal_samples)} samples")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save dataset with minimal data (node features + cell name only)
    if minimal_data_per_file:
        print(f"\n🔄 Finalizing dataset...")
        print(f"   Minimal samples: {len(minimal_data_per_file)} lib files")

        # 저장할 데이터 구조 (최소화 버전: node_features + cell_name + output만)
        dataset = {
            'cache_path': cache_path,  # Reference to topology cache
            'cache_type': cache_type,  # full_graph or stage_aware
            'minimal_data_per_file': minimal_data_per_file,  # Only node_features + cell_name + output
            'num_lib_files': len(minimal_data_per_file),
            'data_type': data_type
        }

        print(f"\n💾 Saving dataset...")
        print(f"   Output file: {save_input}")

        torch.save(dataset, save_input)

        print(f"\n✅ Dataset saved!")
        print(f"   Cache type: {cache_type}")
        print(f"   Cache reference: {cache_path}")
        print(f"   Minimal data: {len(minimal_data_per_file)} lib files")
        print("=" * 80)

        return dataset
    else:
        print("❌ No valid minimal data found!")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GNN dataset using full-graph topology cache")
    parser.add_argument("--cache_path", type=str, required=True,
                       help="Path to full-graph topology cache")
    parser.add_argument("--lib_base_path", type=str, required=True,
                       help="Base path to library files")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory name (folder containing .lib files)")
    parser.add_argument("--prefix", type=str, required=True,
                       help="Prefix for .lib files (e.g., 'INVBUF_LVT_FF_')")
    parser.add_argument("--start", type=int, default=40,
                       help="Start voltage index")
    parser.add_argument("--end", type=int, default=141,
                       help="End voltage index (exclusive)")
    parser.add_argument("--data_type", type=str, default="cell",
                       choices=['cell', 'transition'],
                       help="Data type: cell (delay) or transition (slew)")
    parser.add_argument("--save_input", type=str, required=True,
                       help="Path to save the dataset")

    args = parser.parse_args()

    # Automatically adjust filename based on data_type
    save_path = Path(args.save_input)
    filename = save_path.name

    # If filename contains 'cell_all_graph_data' or 'transition_all_graph_data', replace with correct data_type
    if 'cell_all_graph_data' in filename:
        new_filename = filename.replace('cell_all_graph_data', f'{args.data_type}_all_graph_data')
    elif 'transition_all_graph_data' in filename:
        new_filename = filename.replace('transition_all_graph_data', f'{args.data_type}_all_graph_data')
    else:
        # If filename doesn't follow the pattern, use it as-is
        new_filename = filename

    save_input = str(save_path.parent / new_filename)

    # Ensure output directory exists
    Path(save_input).parent.mkdir(parents=True, exist_ok=True)

    print(f"📝 Output filename: {save_input}")
    if new_filename != filename:
        print(f"   (Modified from: {filename})")

    # Build dataset using dataextract_gnn_cached (libdata_extract_MAML parser)
    build_all_gnn_data_cached(
        cache_path=args.cache_path,
        start=args.start,
        end=args.end,
        prefix=args.prefix,
        save_input=save_input,
        data_dir=args.data_dir,
        lib_base_path=args.lib_base_path,
        data_type=args.data_type
    )
