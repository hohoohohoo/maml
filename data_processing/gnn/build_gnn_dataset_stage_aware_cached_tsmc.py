#!/usr/bin/env python
"""
Build GNN dataset using pre-computed TSMC stage-aware topology cache.

Stage-aware mode: pull-up/pull-down path topology varies based on delay type.
Stores minimal data: node_features + cell_name + output_name + delay_type

Dataset structure:
    {
        'cache_path': str,
        'cache_type': 'stage_aware',
        'minimal_data_per_file': [
            [
                {
                    'node_features': torch.Tensor,  # [num_nodes, 7]
                    'cell_name': str,
                    'output_name': str,
                    'delay_type': str,  # rise_transition or fall_transition
                    'output': float
                },
                ...
            ],
            ...
        ],
        'num_lib_files': int,
        'data_type': str  # 'cell' or 'transition'
    }

Usage:
    python build_gnn_dataset_stage_aware_cached_tsmc.py \
        --cache_path ./topology_cache/stage_aware_tsmc.pth \
        --lib_base_path /path/to/TSMC_lib_files \
        --data_dir TSMC_FF_25 \
        --prefix TSMC_FF_25_ \
        --start 60 --end 121 \
        --data_type cell \
        --save_input ./output/cell_all_graph_data_stage_aware.pth
"""

import torch
from pathlib import Path
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from precompute_stage_aware_topology import apply_stage_aware_topology


def dataextract_gnn_stage_aware_cached_tsmc(lib_file_path, topology_cache, lib_prefix="", data_type="cell"):
    """
    Extract node features from TSMC .lib file using stage-aware topology cache.
    Only stores: node_features, cell_name, output_name, delay_type (for topology lookup)

    Args:
        lib_file_path: Path to .lib file
        topology_cache: Pre-computed TSMC stage-aware topology cache
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
    print(f"   Reading .lib file: {lib_file_path}")

    with open(lib_file_path, "r") as f:
        lines = f.readlines()

    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)

    print(f"   Parsed {len(flattened)} timing entries")

    # Generate node features using apply_stage_aware_topology_tsmc
    minimal_samples = []
    skipped_count = 0
    cached_count = 0
    skipped_cells = set()

    for sample in flattened:
        cell_name = sample['cell']
        voltage = sample['Voltage']
        delay_type = sample['delay_type']  # 'cell_rise', 'cell_fall', etc
        input_port_names = sample['input_port_name']
        output_port_name = sample.get('pin_name', 'Z')  # TSMC typically uses 'Z' or 'ZN'

        # Check if cell is in cache
        if cell_name not in topology_cache:
            skipped_count += 1
            skipped_cells.add(cell_name)
            continue

        cached_count += 1

        # Convert delay_type to rise/fall_transition for stage-aware lookup
        if 'rise' in delay_type:
            stage_delay_type = 'rise_transition'
        else:
            stage_delay_type = 'fall_transition'

        # Get timing data
        input_slews = sample.get('index_1', [40.0])
        output_loads = sample.get('index_2', [5.76])
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

                try:
                    graph_sample = apply_stage_aware_topology(
                        topology_cache,
                        cell_name,
                        output_port_name,
                        stage_delay_type,
                        voltage,
                        input_slew,
                        output_load,
                        input_port_names
                    )

                    # Store only node features and metadata (minimal dataset)
                    minimal_samples.append({
                        'node_features': graph_sample['node_features'],
                        'cell_name': cell_name,
                        'output_name': output_port_name,
                        'delay_type': stage_delay_type,
                        'output': output_value,
                    })
                except Exception as e:
                    # Try alternative output name (Z vs ZN)
                    alt_output = 'ZN' if output_port_name == 'Z' else 'Z'
                    try:
                        graph_sample = apply_stage_aware_topology(
                            topology_cache,
                            cell_name,
                            alt_output,
                            stage_delay_type,
                            voltage,
                            input_slew,
                            output_load,
                            input_port_names
                        )
                        minimal_samples.append({
                            'node_features': graph_sample['node_features'],
                            'cell_name': cell_name,
                            'output_name': alt_output,
                            'delay_type': stage_delay_type,
                            'output': output_value,
                        })
                    except Exception as e2:
                        pass  # Skip this sample

    print(f"   Generated {len(minimal_samples)} samples ({cached_count} cells cached, {skipped_count} skipped)")

    if skipped_cells:
        print(f"   Skipped {len(skipped_cells)} unique cells not in cache:")
        for cell in sorted(skipped_cells)[:10]:
            print(f"      - {cell}")
        if len(skipped_cells) > 10:
            print(f"      ... and {len(skipped_cells) - 10} more")

    return minimal_samples


def build_all_gnn_data_stage_aware_cached_tsmc(
        cache_path,
        start=60,
        end=121,
        prefix="TSMC_FF_25_",
        save_input="graph_data_stage_aware_output.pth",
        data_dir="TSMC_FF_25",
        lib_base_path="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/TSMC_lib_files",
        data_type="cell"
):
    """
    Build stage-aware GNN dataset for TSMC using cached topology.
    Stores only minimal data: node_features + cell_name + output_name + delay_type

    Args:
        cache_path: Path to TSMC stage-aware topology cache
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
    print("BUILD STAGE-AWARE GNN DATASET (TSMC - CACHED)")
    print("=" * 80)
    print(f"Topology cache: {cache_path}")
    print(f"Data type: {data_type}")
    print(f"Lib files: {lib_base_path}/{data_dir}/{prefix}XXX.lib")
    print(f"Voltage range: {start} to {end-1}")
    print("=" * 80)

    # Load topology cache
    print(f"\n Loading TSMC stage-aware topology cache...")
    try:
        topology_cache = torch.load(cache_path, weights_only=False)
        print(f"   Loaded {len(topology_cache)} cells (cache type: stage_aware)")
    except Exception as e:
        print(f"   Error loading cache: {e}")
        raise

    # Store only minimal samples
    minimal_data_per_file = []

    for i in range(start, end):
        v_str = f"{i:03d}"
        filename = f"{lib_base_path}/{data_dir}/{prefix}{v_str}.lib"

        print(f"\n Processing [{i-start+1}/{end-start}]: {filename}")

        try:
            minimal_samples = dataextract_gnn_stage_aware_cached_tsmc(
                filename, topology_cache, prefix, data_type
            )

            if minimal_samples:
                minimal_data_per_file.append(minimal_samples)
                print(f"   File {i-start+1}: {len(minimal_samples)} samples")

        except Exception as e:
            print(f"   Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save dataset
    if minimal_data_per_file:
        print(f"\n Finalizing dataset...")
        print(f"   Minimal samples: {len(minimal_data_per_file)} lib files")

        dataset = {
            'cache_path': cache_path,
            'cache_type': 'stage_aware',
            'minimal_data_per_file': minimal_data_per_file,
            'num_lib_files': len(minimal_data_per_file),
            'data_type': data_type
        }

        print(f"\n Saving dataset...")
        print(f"   Output file: {save_input}")

        Path(save_input).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, save_input)

        print(f"\n Dataset saved!")
        print(f"   Cache type: stage_aware")
        print(f"   Cache reference: {cache_path}")
        print(f"   Minimal data: {len(minimal_data_per_file)} lib files")
        print("=" * 80)

        return dataset
    else:
        print(" No valid minimal data found!")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TSMC GNN dataset using stage-aware topology cache")
    parser.add_argument("--cache_path", type=str, required=True,
                       help="Path to TSMC stage-aware topology cache")
    parser.add_argument("--lib_base_path", type=str, required=True,
                       help="Base path to TSMC library files")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory name (folder containing .lib files)")
    parser.add_argument("--prefix", type=str, required=True,
                       help="Prefix for .lib files (e.g., 'TSMC_FF_25_')")
    parser.add_argument("--start", type=int, default=60,
                       help="Start voltage index")
    parser.add_argument("--end", type=int, default=121,
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

    if 'cell_all_graph_data' in filename:
        new_filename = filename.replace('cell_all_graph_data', f'{args.data_type}_all_graph_data')
    elif 'transition_all_graph_data' in filename:
        new_filename = filename.replace('transition_all_graph_data', f'{args.data_type}_all_graph_data')
    else:
        new_filename = filename

    save_input = str(save_path.parent / new_filename)

    # Ensure output directory exists
    Path(save_input).parent.mkdir(parents=True, exist_ok=True)

    print(f" Output filename: {save_input}")
    if new_filename != filename:
        print(f"   (Modified from: {filename})")

    # Build dataset
    build_all_gnn_data_stage_aware_cached_tsmc(
        cache_path=args.cache_path,
        start=args.start,
        end=args.end,
        prefix=args.prefix,
        save_input=save_input,
        data_dir=args.data_dir,
        lib_base_path=args.lib_base_path,
        data_type=args.data_type
    )
