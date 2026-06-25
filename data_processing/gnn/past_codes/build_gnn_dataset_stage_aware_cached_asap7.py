#!/usr/bin/env python
"""
Build GNN dataset using pre-computed stage-aware topology cache
Stage-aware mode에서는 pull-up/pull-down path에 따라 topology가 다르므로,
각 sample의 node features + cell name + delay_type만 저장

최소 저장 방식:
    - node_features (matrix): 각 sample의 node feature matrix
    - cell_name: topology cache에서 graph structure를 가져올 key
    - output_name: output port name (e.g., 'Y', 'CON', 'SN')
    - delay_type: 'rise_transition' or 'fall_transition' (pull-up/pull-down 구분)
    - output: target value (delay or slew)

데이터셋 구조:
    {
        'cache_path': str,  # topology cache 파일 경로
        'cache_type': 'stage_aware',
        'minimal_data_per_file': [
            [
                {
                    'node_features': torch.Tensor,  # [num_nodes, 7]
                    'cell_name': str,  # topology 조회용
                    'output_name': str,  # output port name
                    'delay_type': str,  # rise_transition or fall_transition
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from precompute_stage_aware_topology import apply_stage_aware_topology


def dataextract_gnn_stage_aware_cached(lib_file_path, topology_cache, lib_prefix="", data_type="cell"):
    """
    Extract node features from .lib file using stage-aware topology cache
    Only stores: node_features (matrix), cell_name, output_name, delay_type (for topology lookup)

    Args:
        lib_file_path: Path to .lib file
        topology_cache: Pre-computed stage-aware topology cache
        lib_prefix: Library file prefix
        data_type: 'cell' or 'transition' to select parser

    Returns:
        list: List of dicts with node_features, cell_name, output_name, delay_type
    """

    # Import appropriate parser based on data_type
    if data_type == 'cell':
        from libdata_extract_MAML_cell import parse_liberty_pin_blocks, flatten_pin_data
    else:  # transition
        from libdata_extract_MAML_transition import parse_liberty_pin_blocks, flatten_pin_data

    # Parse liberty file
    with open(lib_file_path, "r") as f:
        lines = f.readlines()

    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)

    print(f"   ✓ Parsed {len(flattened)} timing entries")

    # Generate node features using apply_stage_aware_topology
    # Store only node_features + cell_name + output_name + delay_type for minimal dataset
    minimal_samples = []
    skipped_count = 0
    cached_count = 0
    skipped_cells = set()  # Track unique skipped cell names

    for sample in flattened:
        cell_name = sample['cell']
        voltage = sample['Voltage']
        delay_type = sample['delay_type']  # 'cell_rise', 'cell_fall', etc
        input_port_names = sample['input_port_name']
        output_port_name = sample.get('pin_name', 'Y')  # Output port name from pin_name field

        # Check if cell is in cache
        if cell_name not in topology_cache:
            skipped_count += 1
            skipped_cells.add(cell_name)  # Track this skipped cell
            continue

        cached_count += 1

        # Convert delay_type to rise/fall_transition for stage-aware lookup
        # cell_rise/cell_fall -> rise_transition/fall_transition
        if 'rise' in delay_type:
            stage_delay_type = 'rise_transition'
        else:
            stage_delay_type = 'fall_transition'

        # Get timing data
        input_slews = sample.get('index_1', [40.0])  # index_1 = input slew
        output_loads = sample.get('index_2', [5.76])  # index_2 = output load
        timing_values = sample.get('values', [[0.0]])

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

                # Call apply_stage_aware_topology to generate node features
                try:
                    graph_sample = apply_stage_aware_topology(
                        topology_cache,
                        cell_name,
                        output_port_name,  # output_name (e.g., 'Y', 'CON', 'SN')
                        stage_delay_type,  # 'rise_transition' or 'fall_transition'
                        voltage,
                        input_slew,
                        output_load,
                        input_port_names
                    )

                    # Store only node features and metadata (minimal dataset)
                    minimal_samples.append({
                        'node_features': graph_sample['node_features'],  # Node feature matrix
                        'cell_name': cell_name,        # For topology cache lookup
                        'output_name': output_port_name,  # Output port name
                        'delay_type': stage_delay_type,  # rise_transition or fall_transition
                        'output': output_value,        # Target value
                    })
                except Exception as e:
                    print(f"   ⚠️  Error processing {cell_name} ({output_port_name}, {stage_delay_type}): {e}")
                    continue

    print(f"   ✓ Generated {len(minimal_samples)} node feature samples ({cached_count} cells cached, {skipped_count} cells skipped)")

    # Print skipped cells if any
    if skipped_cells:
        print(f"   ⚠️  Skipped {len(skipped_cells)} unique cells not in topology cache:")
        for cell in sorted(skipped_cells):
            print(f"      - {cell}")

    return minimal_samples


def build_all_gnn_data_stage_aware_cached(
        cache_path,
        start=40,
        end=101,
        prefix="OA_LVT_2_25_",
        save_input="graph_input_data_stage_aware_cached.pth",
        data_dir="OA_LVT",
        lib_base_path="/home/tkdgn2907/Deepsets_test/MAML/Projects/Dataset_All",
        data_type="cell"
):
    """
    Build stage-aware GNN dataset using cached topology
    Stores only minimal data: node_features + cell_name + output_name + delay_type

    Args:
        cache_path: Path to stage-aware topology cache
        start: Start voltage index
        end: End voltage index (exclusive)
        prefix: Library file prefix
        save_input: Path to save dataset
        data_dir: Data directory name
        lib_base_path: Base path to library files
        data_type: 'cell' or 'transition'

    Returns:
        Dataset dict with minimal data
    """

    print("=" * 80)
    print("BUILDING STAGE-AWARE GNN DATASET (CACHED VERSION)")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Data type: {data_type}")
    print(f"Lib files: {lib_base_path}/{data_dir}/{prefix}XXX.lib")
    print(f"Voltage range: {start} to {end-1}")
    print("=" * 80)

    # Load topology cache
    print(f"\n📦 Loading stage-aware topology cache...")
    try:
        topology_cache = torch.load(cache_path, weights_only=False)
        print(f"   ✓ Loaded {len(topology_cache)} cells (cache type: stage_aware)")
    except Exception as e:
        print(f"   ❌ Error loading cache: {e}")
        raise

    # Store only minimal samples (node features + cell name + output_name + delay_type)
    minimal_data_per_file = []

    for i in range(start, end):
        v_str = f"{i//100}{i%100:02d}"
        filename = f"{lib_base_path}/{data_dir}/{prefix}{v_str}.lib"

        print(f"\n📥 Processing [{i-start+1}/{end-start}]: {filename}")

        try:
            # Extract minimal samples (node features + cell name + output_name + delay_type)
            minimal_samples = dataextract_gnn_stage_aware_cached(filename, topology_cache, prefix, data_type)

            if minimal_samples:
                minimal_data_per_file.append(minimal_samples)
                print(f"   ✓ File {i-start+1}: {len(minimal_samples)} samples")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save dataset with minimal data (node features + cell name + output_name + delay_type only)
    if minimal_data_per_file:
        print(f"\n🔄 Finalizing dataset...")
        print(f"   Minimal samples: {len(minimal_data_per_file)} lib files")

        # 저장할 데이터 구조 (최소화 버전)
        dataset = {
            'cache_path': cache_path,  # Reference to topology cache
            'cache_type': 'stage_aware',
            'minimal_data_per_file': minimal_data_per_file,  # Only node_features + metadata
            'num_lib_files': len(minimal_data_per_file),
            'data_type': data_type
        }

        print(f"\n💾 Saving dataset...")
        print(f"   Output file: {save_input}")

        torch.save(dataset, save_input)

        print(f"\n✅ Dataset saved!")
        print(f"   Cache type: stage_aware")
        print(f"   Cache reference: {cache_path}")
        print(f"   Minimal data: {len(minimal_data_per_file)} lib files")
        print("=" * 80)

        return dataset
    else:
        print("❌ No valid minimal data found!")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build GNN dataset using stage-aware topology cache")
    parser.add_argument("--cache_path", type=str, required=True,
                       help="Path to stage-aware topology cache")
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

    # Build dataset using dataextract_gnn_stage_aware_cached
    build_all_gnn_data_stage_aware_cached(
        cache_path=args.cache_path,
        start=args.start,
        end=args.end,
        prefix=args.prefix,
        save_input=save_input,
        data_dir=args.data_dir,
        lib_base_path=args.lib_base_path,
        data_type=args.data_type
    )
