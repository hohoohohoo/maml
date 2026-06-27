#!/usr/bin/env python
"""
Build GNN dataset with process conditions using topology cache (cached version)
Similar to build_gnn_dataset_stage_aware_cached.py but adds process parameters to node features

Node features: 7D → 11D (7 base + 4 process parameters)
Process parameters: param_a, param_b, param_c, temperature

Dataset flow:
1. Load pre-computed topology cache (from precompute_cell_topology.py or precompute_stage_aware_topology.py)
2. Parse .lib files and extract timing data
3. Parse process conditions from lib filename
4. Generate node features with process parameters using apply_topology_with_process()
5. Store minimal dataset: node_features (11D) + cell_name + output_name + delay_type + output

Usage:
    python build_gnn_dataset_process_cached_asap7.py \\
        --cache_path topology_cache/cell_topology_cache_L.pth \\
        --lib_base_path /path/to/processed_libs \\
        --data_dir simple_0_0_0_ \\
        --prefix invbuf_0_0_0_ \\
        --start 40 --end 101 \\
        --data_type cell \\
        --graph_mode full_graph \\
        --save_input dataset/graph_data/{data_type}_all_graph_data_{graph_mode}_process.pth

    Note: The script will automatically use data_type in the filename.
    Example filenames:
        - cell data: cell_all_graph_data_full_graph_process.pth
        - transition data: transition_all_graph_data_stage_aware_process.pth
"""

import torch
import numpy as np
from pathlib import Path
import sys
import os
import argparse
from typing import Dict, Any
import gc
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'precompute'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from precompute_full_graph_topology import apply_topology_to_sample
from precompute_stage_aware_topology import apply_stage_aware_topology


# Intra topology cells - these cells are used for intra-topology evaluation
# and are excluded from training data
INTRA_TOPOLOGY_CELLS = [
    'AND2x6',
    'NAND3x2',
    'NOR2xp67',
    'OR2x6',
]


def parse_process_conditions_from_filename(lib_prefix: str, is_test: bool = False) -> Dict[str, Any]:
    """
    Parse process condition parameters from library filename

    Format: {cell_type}_{param_a_idx}_{param_b_idx}_{param_c_idx}_{temperature}_
    Example: invbuf_0_0_0_12p5_ → param_a=0.625, param_b=[pmos, nmos], param_c=[pmos, nmos], temp=12.5

    Args:
        lib_prefix: Library filename prefix (e.g., 'invbuf_0_0_0_')
        is_test: Whether this is test dataset (different parameter ranges)

    Returns:
        dict with keys: 'param_a', 'param_b_pmos', 'param_b_nmos', 'param_c_pmos', 'param_c_nmos', 'temperature'
    """
    # Default parameter value ranges
    if is_test:
        # Test dataset ranges
        param_a_values = [0.75, 1.0, 1.25]
        param_b_values = [(0.09, 0.062), (0.092, 0.066), (0.094, 0.07)]
        param_c_values = [(0.36, 0.47), (0.38, 0.475), (0.40, 0.48)]
    else:
        # Training dataset ranges (different values)
        param_a_values = [0.625, 0.875, 1.125, 1.375]
        param_b_values = [(0.089, 0.06), (0.091, 0.064), (0.093, 0.068), (0.095, 0.072)]
        param_c_values = [(0.35, 0.465), (0.37, 0.473), (0.39, 0.478), (0.41, 0.485)]

    # Parse filename pattern: {cell_type}_{a_idx}_{b_idx}_{c_idx}_{temp}_
    parts = lib_prefix.rstrip('_').split('_')

    result = {
        'param_a': 1.0,
        'param_b_pmos': 0.092,
        'param_b_nmos': 0.066,
        'param_c_pmos': 0.38,
        'param_c_nmos': 0.475,
        'temperature': 25.0
    }

    try:
        if len(parts) >= 5:
            # Extract indices
            a_idx = int(parts[-4])
            b_idx = int(parts[-3])
            c_idx = int(parts[-2])
            temp_str = parts[-1].replace('p', '.')

            # Map indices to values
            if 0 <= a_idx < len(param_a_values):
                result['param_a'] = param_a_values[a_idx]

            if 0 <= b_idx < len(param_b_values):
                result['param_b_pmos'] = param_b_values[b_idx][0]
                result['param_b_nmos'] = param_b_values[b_idx][1]

            if 0 <= c_idx < len(param_c_values):
                result['param_c_pmos'] = param_c_values[c_idx][0]
                result['param_c_nmos'] = param_c_values[c_idx][1]

            result['temperature'] = float(temp_str)

    except (ValueError, IndexError) as e:
        print(f"   ⚠️  Warning: Could not parse process parameters from '{lib_prefix}': {e}")
        print(f"      Using default values")

    return result


def apply_topology_with_process(topology_cache, cache_type, cell_name, output_name, delay_type,
                                voltage, input_slew, output_load, input_port_names, process_params,
                                slew_mode='all', related_pin=None, voltage_mode='all_nodes'):
    """
    Apply topology from cache and add process parameters to node features

    Args:
        topology_cache: Pre-computed topology cache
        cache_type: 'full_graph' or 'stage_aware'
        cell_name: Cell name
        output_name: Output port name (for stage_aware)
        delay_type: Delay type (for stage_aware: 'rise_transition' or 'fall_transition')
        voltage: Voltage value
        input_slew: Input slew
        output_load: Output load
        input_port_names: List of input port names
        process_params: Dict with process parameters
        slew_mode: 'all' (apply to all inputs) or 'related_pin_only' (apply to related_pin only)
        related_pin: Related pin name from timing arc (used when slew_mode='related_pin_only')
        voltage_mode: 'all_nodes' (voltage on all), 'vdd_only' (voltage on VDD only), 'vdd_mos' (VDD+MOS)

    Returns:
        dict: Graph sample with 11D node features
    """
    # Get base topology and node features (7D)
    if cache_type == 'stage_aware':
        graph_sample = apply_stage_aware_topology(
            topology_cache, cell_name, output_name, delay_type,
            voltage, input_slew, output_load, input_port_names,
            voltage_mode=voltage_mode, slew_mode=slew_mode, related_pin=related_pin
        )
    else:  # full_graph
        graph_sample = apply_topology_to_sample(
            topology_cache, cell_name, voltage, input_slew, output_load,
            output_value=0.0,  # Placeholder
            input_port_names=input_port_names,
            voltage_mode=voltage_mode
        )

    # Get base node features (7D)
    base_node_features = graph_sample['node_features']  # [num_nodes, 7]
    num_nodes = base_node_features.shape[0]

    # Create process parameter features (4D) for each node
    # Different values for NMOS vs PMOS transistors
    process_features = torch.zeros(num_nodes, 4, dtype=torch.float32)

    # Get transistor info from cache
    # Note: transistor_info is always at cell level, not inside output_topologies
    cell_cache = topology_cache[cell_name]
    transistor_info = cell_cache['transistor_info']

    # Process parameters are only added to MOS (transistor) nodes
    # Non-transistor nodes (VDD, GND, internal nets) remain with zeros
    for node_idx, node_name in enumerate(graph_sample.get('all_nodes', [])):
        if node_name in transistor_info:
            trans_info = transistor_info[node_name]
            trans_type = trans_info['type']  # 1.0 for NMOS, -1.0 for PMOS

            # param_a: same for all transistors
            process_features[node_idx, 0] = process_params['param_a']

            # param_b, param_c: different for NMOS/PMOS
            if trans_type > 0:  # NMOS
                process_features[node_idx, 1] = process_params['param_b_nmos']
                process_features[node_idx, 2] = process_params['param_c_nmos']
            else:  # PMOS
                process_features[node_idx, 1] = process_params['param_b_pmos']
                process_features[node_idx, 2] = process_params['param_c_pmos']

            # temperature: same for all MOS nodes
            process_features[node_idx, 3] = process_params['temperature']
        # Non-MOS nodes: process_features remain zeros (initialized above)

    # Concatenate base features (7D) with process features (4D) → 11D
    enhanced_node_features = torch.cat([base_node_features, process_features], dim=1)

    # Update graph sample
    graph_sample['node_features'] = enhanced_node_features
    graph_sample['process_params'] = process_params

    return graph_sample


def dataextract_gnn_cached_with_process(lib_file_path, topology_cache, cache_type, lib_prefix="",
                                       data_type="cell", is_test=False, slew_mode='all',
                                       voltage_mode='all_nodes'):
    """
    Extract node features from .lib file using cached topology WITH process parameters

    Args:
        lib_file_path: Path to .lib file
        topology_cache: Pre-computed topology cache
        cache_type: 'full_graph' or 'stage_aware'
        lib_prefix: Library file prefix (for parsing process conditions)
        data_type: 'cell' or 'transition'
        is_test: Whether this is test dataset
        slew_mode: 'all' (apply to all inputs) or 'related_pin_only' (apply to related_pin only)
        voltage_mode: 'all_nodes', 'vdd_only', or 'vdd_mos'

    Returns:
        list: List of minimal samples with 11D node features
    """
    # Import appropriate parser
    if data_type == 'cell':
        from libdata_extract_MAML_cell import parse_liberty_pin_blocks, flatten_pin_data
    else:
        from libdata_extract_MAML_transition import parse_liberty_pin_blocks, flatten_pin_data

    # Parse liberty file
    with open(lib_file_path, "r") as f:
        lines = f.readlines()

    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)

    print(f"   ✓ Parsed {len(flattened)} timing entries")

    # Parse process conditions from filename
    process_params = parse_process_conditions_from_filename(lib_prefix, is_test)
    print(f"   📊 Process parameters: A={process_params['param_a']:.3f}, "
          f"B_p={process_params['param_b_pmos']:.3f}, B_n={process_params['param_b_nmos']:.3f}, "
          f"C_p={process_params['param_c_pmos']:.3f}, C_n={process_params['param_c_nmos']:.3f}, "
          f"T={process_params['temperature']:.1f}°C")

    # Generate minimal samples with process-aware node features
    minimal_samples = []
    skipped_count = 0
    cached_count = 0
    skipped_cells = set()

    for sample in flattened:
        cell_name = sample['cell']
        voltage = sample['Voltage']
        delay_type = sample['delay_type']
        input_port_names = sample['input_port_name']
        output_port_name = sample.get('pin_name', 'Y')
        related_pin = sample.get('related_pin', None)

        # Check if cell is in cache
        if cell_name not in topology_cache:
            skipped_count += 1
            skipped_cells.add(cell_name)
            continue

        cached_count += 1

        # Convert delay_type for stage-aware mode
        if cache_type == 'stage_aware':
            if 'rise' in delay_type:
                stage_delay_type = 'rise_transition'
            else:
                stage_delay_type = 'fall_transition'
        else:
            stage_delay_type = delay_type

        # Get timing data
        input_slews = sample.get('index_1', [40.0])
        output_loads = sample.get('index_2', [5.76])
        timing_values = sample.get('values', [[0.0]])

        # Handle values array size mismatch
        actual_rows = len(timing_values) if isinstance(timing_values, list) else 0
        actual_cols = len(timing_values[0]) if actual_rows > 0 and isinstance(timing_values[0], list) else 0

        effective_rows = min(len(input_slews), actual_rows) if actual_rows > 0 else len(input_slews)
        effective_cols = min(len(output_loads), actual_cols) if actual_cols > 0 else len(output_loads)

        # Generate node features with process parameters
        for row_idx in range(effective_rows):
            for col_idx in range(effective_cols):
                input_slew = input_slews[row_idx]
                output_load = output_loads[col_idx]
                output_value = timing_values[row_idx][col_idx]

                try:
                    # Apply topology with process parameters → 11D features
                    graph_sample = apply_topology_with_process(
                        topology_cache, cache_type, cell_name, output_port_name, stage_delay_type,
                        voltage, input_slew, output_load, input_port_names, process_params,
                        slew_mode=slew_mode, related_pin=related_pin, voltage_mode=voltage_mode
                    )

                    # Verify feature dimension
                    if graph_sample['node_features'].shape[1] != 11:
                        print(f"   ⚠️  Warning: Expected 11D features, got {graph_sample['node_features'].shape[1]}D")
                        continue

                    # Store minimal sample
                    minimal_sample = {
                        'node_features': graph_sample['node_features'],  # 11D features
                        'cell_name': cell_name,
                        'output': output_value,
                        'delay_type': stage_delay_type if cache_type == 'stage_aware' else delay_type
                    }

                    # Add output_name for stage_aware mode
                    if cache_type == 'stage_aware':
                        minimal_sample['output_name'] = output_port_name

                    minimal_samples.append(minimal_sample)

                except Exception as e:
                    print(f"   ⚠️  Error processing {cell_name} ({output_port_name}, {stage_delay_type}): {e}")
                    continue

    print(f"   ✓ Generated {len(minimal_samples)} samples with 11D features ({cached_count} cells cached, {skipped_count} cells skipped)")

    if skipped_cells:
        print(f"   ⚠️  Skipped {len(skipped_cells)} unique cells not in topology cache:")
        for cell in sorted(skipped_cells):
            print(f"      - {cell}")

    return minimal_samples


def build_all_gnn_data_cached_with_process(
        cache_path,
        cache_type,
        start=40,
        end=101,
        prefix="invbuf_0_0_0_",
        save_input="graph_data/output.pth",  # Will be modified based on data_type
        data_dir="simple",
        lib_base_path="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed_simple",
        data_type="cell",
        is_test=False,
        max_samples_per_cell=None,
        max_test_tasks=None,
        max_train_tasks=None,
        slew_mode='all',
        voltage_mode='all_nodes'
):
    """
    Build GNN dataset with process conditions using cached topology

    Args:
        cache_path: Path to topology cache
        cache_type: 'full_graph' or 'stage_aware'
        start: Start voltage index
        end: End voltage index (exclusive)
        prefix: Library file prefix
        save_input: Path to save dataset
        data_dir: Data directory name
        lib_base_path: Base path to library files
        data_type: 'cell' or 'transition'
        is_test: Whether this is test dataset
        max_samples_per_cell: [DEPRECATED] Use max_test_tasks instead.
        max_test_tasks: Maximum number of total test tasks (random sampling across all cells). Only used with is_test=True.
        max_train_tasks: Maximum number of tasks for training (random sampling). Only used with is_test=False.
        slew_mode: 'all' (apply to all inputs) or 'related_pin_only' (apply to related_pin only)

    Returns:
        Dataset dict with minimal data (11D features)
    """
    print("=" * 80)
    print(f"BUILDING GNN DATASET WITH PROCESS CONDITIONS (CACHED, {cache_type.upper()})")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Cache type: {cache_type}")
    print(f"Data type: {data_type}")
    print(f"Lib files: {lib_base_path}/{data_dir}/{prefix}XXX.lib")
    print(f"Voltage range: {start} to {end-1}")
    print(f"Node features: 11D (7 base + 4 process parameters)")
    print(f"Test dataset: {is_test}")
    if max_samples_per_cell:
        print(f"Max samples per cell: {max_samples_per_cell}")
    if max_train_tasks:
        print(f"Max train tasks: {max_train_tasks}")
    print("=" * 80)

    # Load topology cache
    print(f"\n📦 Loading topology cache...")
    try:
        topology_cache = torch.load(cache_path, weights_only=False)
        print(f"   ✓ Loaded {len(topology_cache)} cells (cache type: {cache_type})")
    except Exception as e:
        print(f"   ❌ Error loading cache: {e}")
        raise

    # Process all lib files
    minimal_data_per_file = []

    for i in range(start, end):
        v_str = f"{i//100}{i%100:02d}"
        filename = f"{lib_base_path}/{data_dir}/{prefix}{v_str}.lib"

        print(f"\n📥 Processing [{i-start+1}/{end-start}]: {filename}")

        minimal_samples = None
        try:
            minimal_samples = dataextract_gnn_cached_with_process(
                filename, topology_cache, cache_type, prefix, data_type, is_test,
                slew_mode=slew_mode, voltage_mode=voltage_mode
            )

            if minimal_samples:
                minimal_data_per_file.append(minimal_samples)
                print(f"   ✓ File {i-start+1}: {len(minimal_samples)} samples (11D features)")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up intermediate variables
            if minimal_samples is not None:
                del minimal_samples
            gc.collect()

            # Periodic memory cleanup every 10 files
            if (i - start + 1) % 10 == 0:
                print(f"   🧹 Periodic memory cleanup (every 10 files)...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save dataset
    if minimal_data_per_file:
        print(f"\n🔄 Finalizing dataset...")
        print(f"   Minimal samples: {len(minimal_data_per_file)} lib files")

        # Ensure output directory exists
        Path(save_input).parent.mkdir(parents=True, exist_ok=True)

        if is_test:
            # For test data: save separate files per cell_name
            print(f"\n📦 Test dataset mode: Grouping samples by cell_name...")

            # Group samples by cell_name across all lib files
            cell_name_groups = {}
            for lib_samples in minimal_data_per_file:
                for sample in lib_samples:
                    cell_name = sample['cell_name']
                    if cell_name not in cell_name_groups:
                        cell_name_groups[cell_name] = []
                    cell_name_groups[cell_name].append(sample)

            print(f"   Found {len(cell_name_groups)} unique cell types")

            # Calculate total tasks before sampling
            total_tasks_before = sum(len(samples) for samples in cell_name_groups.values())
            print(f"   Total tasks: {total_tasks_before}")

            # Apply random sampling if max_test_tasks is set (sample from total, not per cell)
            if max_test_tasks is not None and max_test_tasks > 0 and total_tasks_before > max_test_tasks:
                print(f"\n🎲 Applying random sampling (max {max_test_tasks} total tasks)...")
                random.seed(42)  # For reproducibility

                # Flatten all samples with cell info
                all_samples_flat = []
                for cell_name, samples in cell_name_groups.items():
                    for sample in samples:
                        all_samples_flat.append((cell_name, sample))

                # Random sample from total
                selected_samples = random.sample(all_samples_flat, max_test_tasks)
                print(f"   Selected {len(selected_samples)} tasks from {total_tasks_before} total")

                # Regroup by cell name
                cell_name_groups = {}
                for cell_name, sample in selected_samples:
                    if cell_name not in cell_name_groups:
                        cell_name_groups[cell_name] = []
                    cell_name_groups[cell_name].append(sample)

                # Print per-cell counts after sampling
                print(f"   Tasks per cell after sampling:")
                for cell_name, samples in sorted(cell_name_groups.items()):
                    print(f"      {cell_name}: {len(samples)} tasks")

            # [DEPRECATED] Legacy max_samples_per_cell support
            elif max_samples_per_cell is not None and max_samples_per_cell > 0:
                print(f"\n⚠️  [DEPRECATED] Using max_samples_per_cell. Consider using --max_test_tasks instead.")
                print(f"🎲 Applying random sampling (max {max_samples_per_cell} samples per cell)...")
                random.seed(42)  # For reproducibility
                for cell_name in cell_name_groups:
                    original_count = len(cell_name_groups[cell_name])
                    if original_count > max_samples_per_cell:
                        cell_name_groups[cell_name] = random.sample(
                            cell_name_groups[cell_name], max_samples_per_cell
                        )
                        print(f"   {cell_name}: {original_count} → {max_samples_per_cell} samples")

            # Save separate file for each cell_name
            save_dir = Path(save_input).parent
            cache_type_suffix = cache_type  # 'full_graph' or 'stage_aware'

            saved_files = []
            for cell_name, samples in cell_name_groups.items():
                cell_dataset = {
                    'cache_path': cache_path,
                    'cache_type': cache_type,
                    'minimal_data_per_file': [samples],  # Single "lib" containing all samples for this cell
                    'num_lib_files': 1,
                    'data_type': data_type,
                    'cell_name': cell_name
                }

                # File path: {save_dir}/{data_type}_{cell_name}_graph_data_{cache_type}.pth
                cell_file_path = save_dir / f"{data_type}_{cell_name}_graph_data_{cache_type_suffix}.pth"

                print(f"\n💾 Saving {cell_name}...")
                print(f"   Samples: {len(samples)}")
                print(f"   Output: {cell_file_path}")

                torch.save(cell_dataset, cell_file_path)
                saved_files.append((cell_name, str(cell_file_path), len(samples)))

            print(f"\n✅ Test dataset saved (cell-separated)!")
            print(f"   Cache type: {cache_type}")
            print(f"   Cache reference: {cache_path}")
            print(f"   Total cells: {len(cell_name_groups)}")
            print(f"   Node features: 11D (7 base + 4 process)")
            print(f"\n📋 Saved files:")
            for cell_name, file_path, num_samples in saved_files:
                print(f"   - {cell_name}: {num_samples} samples → {file_path}")
            print("=" * 80)

            # Clean up memory
            print(f"\n🧹 Cleaning up memory...")
            del topology_cache
            del minimal_data_per_file
            del cell_name_groups
            del saved_files
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"   ✓ CUDA cache cleared")
            print(f"   ✓ Memory cleaned up")

            # Return None to avoid keeping reference to data
            return None

        else:
            # For training data: save all cells in one file (original behavior)

            # Calculate normalization statistics (but DO NOT apply normalization)
            # Columns to normalize: 4 (voltage), 5 (input_slew), 6 (output_load), 10 (temperature)
            print(f"\n📊 Calculating normalization statistics...")
            all_node_features = []
            for lib_samples in minimal_data_per_file:
                for sample in lib_samples:
                    all_node_features.append(sample['node_features'])

            if all_node_features:
                all_features = torch.cat(all_node_features, dim=0)  # [total_nodes, 11]
                # Column mapping: 4=voltage, 5=input_slew, 6=output_load, 10=temperature
                col_name_map = {4: 'voltage', 5: 'input_slew', 6: 'output_load', 10: 'temperature'}

                norm_stats = {}
                for col, name in col_name_map.items():
                    col_data = all_features[:, col]
                    norm_stats[name] = {
                        'mean': col_data.mean().item(),
                        'std': col_data.std().item()
                    }
                    print(f"   {name} (col {col}): mean={norm_stats[name]['mean']:.6f}, std={norm_stats[name]['std']:.6f}")

                # Clean up temporary tensor
                del all_features
                del all_node_features
                gc.collect()
            else:
                norm_stats = None
                print(f"   ⚠️  No samples to calculate norm_stats")

            # NOTE: Normalization is NOT applied here - only stats are saved
            # Normalization should be applied during training/validation using saved norm_stats

            # Apply task-level random sampling if max_train_tasks is set
            if max_train_tasks is not None and max_train_tasks > 0:
                # Each lib file has the same number of samples (tasks)
                # Same index across lib files = same task with different voltages
                total_tasks = len(minimal_data_per_file[0]) if minimal_data_per_file else 0

                if total_tasks > max_train_tasks:
                    print(f"\n🎲 Applying task-level random sampling...")
                    print(f"   Total tasks: {total_tasks} → {max_train_tasks}")

                    random.seed(42)  # For reproducibility
                    selected_indices = sorted(random.sample(range(total_tasks), max_train_tasks))

                    # Filter samples by selected indices for each lib file
                    sampled_data_per_file = []
                    for lib_samples in minimal_data_per_file:
                        sampled_lib = [lib_samples[i] for i in selected_indices if i < len(lib_samples)]
                        sampled_data_per_file.append(sampled_lib)

                    minimal_data_per_file = sampled_data_per_file
                    print(f"   ✓ Selected {max_train_tasks} tasks (indices sampled randomly)")
                else:
                    print(f"\n📊 Total tasks ({total_tasks}) <= max_train_tasks ({max_train_tasks}), using all tasks")

            dataset = {
                'cache_path': cache_path,
                'cache_type': cache_type,
                'minimal_data_per_file': minimal_data_per_file,
                'num_lib_files': len(minimal_data_per_file),
                'data_type': data_type,
                'norm_stats': {
                    'node_features': norm_stats  # {'voltage': {...}, 'input_slew': {...}, ...}
                }  # Normalization statistics (not applied, just saved)
            }

            print(f"\n💾 Saving dataset...")
            print(f"   Output file: {save_input}")

            torch.save(dataset, save_input)

            # Calculate total samples for summary
            total_samples = sum(len(lib_samples) for lib_samples in minimal_data_per_file)

            print(f"\n✅ Dataset saved!")
            print(f"   Cache type: {cache_type}")
            print(f"   Cache reference: {cache_path}")
            print(f"   Lib files: {len(minimal_data_per_file)}")
            print(f"   Tasks per lib file: {len(minimal_data_per_file[0]) if minimal_data_per_file else 0}")
            print(f"   Total samples: {total_samples}")
            print(f"   Node features: 11D (7 base + 4 process)")
            print(f"   Norm stats: {'Included (cols 4,5,6,10)' if norm_stats else 'Not available'}")
            print("=" * 80)

            # Clean up memory
            print(f"\n🧹 Cleaning up memory...")
            del topology_cache
            del minimal_data_per_file
            del dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"   ✓ CUDA cache cleared")
            print(f"   ✓ Memory cleaned up")

            return None  # Return None to avoid keeping reference to dataset
    else:
        print("❌ No valid minimal data found!")

        # Clean up memory even on failure
        del topology_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return None


def build_batch_test_data(
        cache_path,
        cache_type,
        test_folders,  # List of (folder_path, prefix, start, end) tuples
        output_dir,
        data_type="cell",
        max_tasks_per_cell=None,
        filter_cells=None,
        slew_mode='all',
        voltage_mode='all_nodes'
):
    """
    Build test dataset using TASK-BASED reservoir sampling with stratified folder distribution.

    A task = 61 datapoints (same cell/slew/load across all voltage points).
    Memory-efficient: processes folder by folder, applies reservoir sampling at task level.
    Stratified: each folder contributes equally to the final sample.

    Args:
        cache_path: Path to topology cache
        cache_type: 'full_graph' or 'stage_aware'
        test_folders: List of tuples (folder_path, lib_base_path, prefix, start, end)
        output_dir: Output directory for cell-based .pth files
        data_type: 'cell' or 'transition'
        max_tasks_per_cell: Maximum TASKS per cell (each task = 61 datapoints)
        filter_cells: Set of cell names to filter (only these cells will be processed)
        slew_mode: 'all' (apply to all inputs) or 'related_pin_only' (apply to related_pin only)

    Returns:
        None
    """
    from collections import defaultdict

    print("=" * 80)
    print(f"BUILDING BATCH TEST DATASET (TASK-BASED RESERVOIR SAMPLING)")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Cache type: {cache_type}")
    print(f"Data type: {data_type}")
    print(f"Test folders: {len(test_folders)}")
    print(f"Max TASKS per cell: {max_tasks_per_cell if max_tasks_per_cell else 'No limit'}")
    print(f"Filter cells: {len(filter_cells)} cells" if filter_cells else "Filter cells: All cells")
    print("=" * 80)

    # Load topology cache
    print(f"\n📦 Loading topology cache...")
    try:
        topology_cache = torch.load(cache_path, weights_only=False)
        print(f"   ✓ Loaded {len(topology_cache)} cells (cache type: {cache_type})")
    except Exception as e:
        print(f"   ❌ Error loading cache: {e}")
        raise

    # Task-based reservoir sampling structures
    # cell_tasks: stores up to max_tasks_per_cell tasks per cell
    # cell_task_counts: tracks total tasks seen per cell (for reservoir sampling)
    # Each task = list of 61 samples (one per voltage point)
    cell_tasks = defaultdict(list)       # cell_name -> list of tasks
    cell_task_counts = defaultdict(int)  # cell_name -> total tasks seen

    # Set random seed for reproducibility
    random.seed(42)

    # Determine if we're using reservoir sampling
    use_reservoir = max_tasks_per_cell is not None and max_tasks_per_cell > 0

    # Calculate tasks per folder for stratified sampling
    num_folders = len(test_folders)
    if use_reservoir:
        tasks_per_folder = max(1, max_tasks_per_cell // num_folders)
        print(f"\n🎯 Stratified sampling: ~{tasks_per_folder} tasks/cell per folder")
    else:
        tasks_per_folder = None

    print(f"\n🚀 Starting task-based streaming processing...")

    for folder_idx, (folder_path, lib_base_path, prefix, start, end) in enumerate(test_folders):
        print(f"\n[{folder_idx+1}/{len(test_folders)}] Processing: {folder_path}")

        num_voltages = end - start  # Should be 61

        # Step 1: Collect all lib files for this folder
        # minimal_data_per_file[voltage_idx] = list of samples at that voltage
        minimal_data_per_file = []

        for i in range(start, end):
            v_str = f"{i//100}{i%100:02d}"
            filename = f"{lib_base_path}/{folder_path}/{prefix}{v_str}.lib"

            try:
                minimal_samples = dataextract_gnn_cached_with_process(
                    filename, topology_cache, cache_type, prefix, data_type, is_test=True,
                    slew_mode=slew_mode, voltage_mode=voltage_mode
                )

                if minimal_samples:
                    # Add folder/voltage info to each sample
                    for sample in minimal_samples:
                        sample['source_folder'] = folder_path
                        sample['source_voltage'] = i
                    minimal_data_per_file.append(minimal_samples)
                else:
                    minimal_data_per_file.append([])

            except Exception as e:
                print(f"   ⚠️ Error processing {filename}: {e}")
                minimal_data_per_file.append([])
                continue

        # Step 2: Build tasks from position indices
        # Task = samples at same index across all 61 voltage files
        if len(minimal_data_per_file) != num_voltages:
            print(f"   ⚠️ Warning: Expected {num_voltages} lib files, got {len(minimal_data_per_file)}")
            continue

        # Get number of tasks (samples in first non-empty lib file)
        num_tasks_in_folder = 0
        for lib_samples in minimal_data_per_file:
            if lib_samples:
                num_tasks_in_folder = len(lib_samples)
                break

        if num_tasks_in_folder == 0:
            print(f"   ⚠️ No samples found in folder")
            continue

        print(f"   Found {num_tasks_in_folder} tasks × {num_voltages} voltages")

        # Step 3: Group by cell and apply reservoir sampling per folder
        folder_tasks_by_cell = defaultdict(list)  # cell_name -> list of tasks in this folder

        for task_idx in range(num_tasks_in_folder):
            # Collect 61 datapoints for this task
            task_samples = []
            cell_name = None

            for voltage_idx in range(num_voltages):
                if voltage_idx < len(minimal_data_per_file) and task_idx < len(minimal_data_per_file[voltage_idx]):
                    sample = minimal_data_per_file[voltage_idx][task_idx]
                    task_samples.append(sample)
                    if cell_name is None:
                        cell_name = sample['cell_name']

            # Only keep complete tasks (all 61 datapoints)
            if len(task_samples) == num_voltages and cell_name:
                # Apply cell filter if specified
                if filter_cells is None or cell_name in filter_cells:
                    folder_tasks_by_cell[cell_name].append(task_samples)

        # Step 4: Apply reservoir sampling for this folder's contribution
        for cell_name, folder_cell_tasks in folder_tasks_by_cell.items():
            for task in folder_cell_tasks:
                cell_task_counts[cell_name] += 1
                n = cell_task_counts[cell_name]

                if use_reservoir:
                    if n <= max_tasks_per_cell:
                        # Still filling the reservoir
                        cell_tasks[cell_name].append(task)
                    else:
                        # Reservoir is full - replace with probability k/n
                        j = random.randint(1, n)
                        if j <= max_tasks_per_cell:
                            cell_tasks[cell_name][j - 1] = task
                else:
                    # No limit - just append
                    cell_tasks[cell_name].append(task)

        # Report folder stats
        folder_task_count = sum(len(tasks) for tasks in folder_tasks_by_cell.values())
        print(f"   Collected {folder_task_count} complete tasks from {len(folder_tasks_by_cell)} cells")

        # Memory cleanup after each folder
        del minimal_data_per_file
        del folder_tasks_by_cell
        gc.collect()

        # Periodic memory status
        if (folder_idx + 1) % 20 == 0:
            current_tasks = sum(len(tasks) for tasks in cell_tasks.values())
            print(f"   📊 Memory: {len(cell_tasks)} cells, {current_tasks} tasks in reservoir")

    # Final statistics
    print(f"\n📊 Task-based streaming complete!")
    print(f"   Unique cells: {len(cell_tasks)}")
    total_tasks_seen = sum(cell_task_counts.values())
    total_tasks_kept = sum(len(tasks) for tasks in cell_tasks.values())
    print(f"   Total tasks seen: {total_tasks_seen}")
    print(f"   Total tasks kept: {total_tasks_kept}")
    print(f"   Total datapoints kept: {total_tasks_kept * (end - start)}")
    if use_reservoir and total_tasks_seen > 0:
        print(f"   Task reduction: {(1 - total_tasks_kept/total_tasks_seen)*100:.1f}%")

    # Save per cell
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving test data by cell (task-based format)...")
    print(f"   Output directory: {output_path}")

    saved_files = []
    num_voltages = end - start  # Number of voltage points per task

    for cell_name, tasks in cell_tasks.items():
        if len(tasks) == 0:
            continue

        # Convert task-based structure to minimal_data_per_file format
        # minimal_data_per_file[voltage_idx] = list of samples at that voltage
        # Each task has 61 samples (one per voltage)
        minimal_data_per_file = [[] for _ in range(num_voltages)]

        for task in tasks:
            for voltage_idx, sample in enumerate(task):
                minimal_data_per_file[voltage_idx].append(sample)

        # Collect source folders from all samples
        all_source_folders = set()
        for task in tasks:
            for sample in task:
                if 'source_folder' in sample:
                    all_source_folders.add(sample['source_folder'])

        cell_dataset = {
            'cache_path': cache_path,
            'cache_type': cache_type,
            'minimal_data_per_file': minimal_data_per_file,  # [num_voltages][num_tasks]
            'num_lib_files': num_voltages,
            'data_type': data_type,
            'cell_name': cell_name,
            'num_tasks': len(tasks),
            'num_datapoints': len(tasks) * num_voltages,
            'source_folders': list(all_source_folders),
            'total_tasks_seen': cell_task_counts[cell_name]
        }

        cell_file_path = output_path / f"{data_type}_{cell_name}_graph_data_{cache_type}.pth"
        torch.save(cell_dataset, cell_file_path)
        saved_files.append((cell_name, str(cell_file_path), len(tasks), cell_task_counts[cell_name]))

    print(f"\n✅ Batch test dataset saved (task-based)!")
    print(f"   Cache type: {cache_type}")
    print(f"   Total cells: {len(saved_files)}")
    print(f"   Node features: 11D (7 base + 4 process)")
    print(f"   Voltages per task: {num_voltages}")
    print(f"\n📋 Saved files:")
    for cell_name, _file_path, num_tasks, total_seen in sorted(saved_files):
        if use_reservoir and total_seen > num_tasks:
            print(f"   - {cell_name}: {num_tasks} tasks ({num_tasks * num_voltages} datapoints, sampled from {total_seen} tasks)")
        else:
            print(f"   - {cell_name}: {num_tasks} tasks ({num_tasks * num_voltages} datapoints)")
    print("=" * 80)

    # Clean up
    del topology_cache
    del cell_tasks
    del cell_task_counts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


def build_batch_test_data_streaming(
        cache_path,
        cache_type,
        test_folders,
        output_dir,
        data_type="cell",
        max_tasks_per_cell=None,
        filter_cells=None,
        slew_mode='all',
        voltage_mode='all_nodes'
):
    """
    Build test dataset using TRUE STREAMING - saves to disk after EACH folder.

    Memory-efficient: Only one folder's data is in memory at a time.
    After processing each folder, data is immediately appended to per-cell files on disk.

    Args:
        cache_path: Path to topology cache
        cache_type: 'full_graph' or 'stage_aware'
        test_folders: List of tuples (folder_path, lib_base_path, prefix, start, end)
        output_dir: Output directory for cell-based .pth files
        data_type: 'cell' or 'transition'
        max_tasks_per_cell: Maximum TASKS per cell (applied at the end via sampling)
        filter_cells: Set of cell names to filter (only these cells will be processed)
        slew_mode: 'all' (apply to all inputs) or 'related_pin_only' (apply to related_pin only)
    """
    import pickle
    import tempfile
    import shutil
    from collections import defaultdict

    print("=" * 80)
    print(f"BUILDING BATCH TEST DATASET (TRUE STREAMING MODE)")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Cache type: {cache_type}")
    print(f"Data type: {data_type}")
    print(f"Test folders: {len(test_folders)}")
    print(f"Max TASKS per cell: {max_tasks_per_cell if max_tasks_per_cell else 'No limit'}")
    print(f"Filter cells: {len(filter_cells)} cells" if filter_cells else "Filter cells: All cells")
    print("=" * 80)

    # Load topology cache
    print(f"\n📦 Loading topology cache...")
    try:
        topology_cache = torch.load(cache_path, weights_only=False)
        print(f"   ✓ Loaded {len(topology_cache)} cells (cache type: {cache_type})")
    except Exception as e:
        print(f"   ❌ Error loading cache: {e}")
        raise

    # Create temp directory for streaming data
    temp_dir = Path(output_dir) / "_temp_streaming"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Temp directory: {temp_dir}")

    # Track task counts per cell (for final stats)
    cell_task_counts = defaultdict(int)
    cell_source_folders = defaultdict(set)

    # Determine num_voltages from first folder
    first_folder = test_folders[0]
    num_voltages = first_folder[4] - first_folder[3]  # end - start

    print(f"\n🚀 Starting TRUE STREAMING processing...")
    print(f"   Each folder will be saved to disk immediately after processing")

    for folder_idx, (folder_path, lib_base_path, prefix, start, end) in enumerate(test_folders):
        print(f"\n[{folder_idx+1}/{len(test_folders)}] Processing: {folder_path}")

        # Step 1: Collect all lib files for this folder
        minimal_data_per_file = []

        for i in range(start, end):
            v_str = f"{i//100}{i%100:02d}"
            filename = f"{lib_base_path}/{folder_path}/{prefix}{v_str}.lib"

            try:
                minimal_samples = dataextract_gnn_cached_with_process(
                    filename, topology_cache, cache_type, prefix, data_type, is_test=True,
                    slew_mode=slew_mode, voltage_mode=voltage_mode
                )

                if minimal_samples:
                    for sample in minimal_samples:
                        sample['source_folder'] = folder_path
                        sample['source_voltage'] = i
                    minimal_data_per_file.append(minimal_samples)
                else:
                    minimal_data_per_file.append([])

            except Exception as e:
                print(f"   ⚠️ Error processing {filename}: {e}")
                minimal_data_per_file.append([])
                continue

        # Step 2: Build tasks and group by cell
        if len(minimal_data_per_file) != num_voltages:
            print(f"   ⚠️ Warning: Expected {num_voltages} lib files, got {len(minimal_data_per_file)}")
            continue

        # Get number of tasks
        num_tasks_in_folder = 0
        for lib_samples in minimal_data_per_file:
            if lib_samples:
                num_tasks_in_folder = len(lib_samples)
                break

        if num_tasks_in_folder == 0:
            print(f"   ⚠️ No samples found in folder")
            continue

        print(f"   Found {num_tasks_in_folder} tasks × {num_voltages} voltages")

        # Group by cell for this folder
        folder_tasks_by_cell = defaultdict(list)

        for task_idx in range(num_tasks_in_folder):
            task_samples = []
            cell_name = None

            for voltage_idx in range(num_voltages):
                if voltage_idx < len(minimal_data_per_file) and task_idx < len(minimal_data_per_file[voltage_idx]):
                    sample = minimal_data_per_file[voltage_idx][task_idx]
                    task_samples.append(sample)
                    if cell_name is None:
                        cell_name = sample['cell_name']

            if len(task_samples) == num_voltages and cell_name:
                # Apply cell filter if specified
                if filter_cells is None or cell_name in filter_cells:
                    folder_tasks_by_cell[cell_name].append(task_samples)

        # Step 3: IMMEDIATELY save to per-cell temp files (append mode)
        for cell_name, tasks in folder_tasks_by_cell.items():
            cell_task_counts[cell_name] += len(tasks)
            cell_source_folders[cell_name].add(folder_path)

            # Append to temp file
            temp_file = temp_dir / f"{cell_name}.pkl"

            # Load existing tasks if file exists
            existing_tasks = []
            if temp_file.exists():
                try:
                    with open(temp_file, 'rb') as f:
                        existing_tasks = pickle.load(f)
                except:
                    existing_tasks = []

            # Append new tasks
            existing_tasks.extend(tasks)

            # Save back
            with open(temp_file, 'wb') as f:
                pickle.dump(existing_tasks, f)

        # Report and cleanup
        folder_task_count = sum(len(tasks) for tasks in folder_tasks_by_cell.values())
        print(f"   ✓ Saved {folder_task_count} tasks to {len(folder_tasks_by_cell)} cell files")

        # CRITICAL: Clear memory after each folder
        del minimal_data_per_file
        del folder_tasks_by_cell
        gc.collect()

        # Periodic status
        if (folder_idx + 1) % 20 == 0:
            print(f"   📊 Progress: {folder_idx+1}/{len(test_folders)} folders processed")

    # Step 4: Convert temp pickle files to final .pth files
    print(f"\n📦 Converting temp files to final .pth format...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    total_tasks_saved = 0

    # Set random seed for sampling
    random.seed(42)

    for temp_file in sorted(temp_dir.glob("*.pkl")):
        cell_name = temp_file.stem

        try:
            with open(temp_file, 'rb') as f:
                all_tasks = pickle.load(f)
        except Exception as e:
            print(f"   ⚠️ Error loading {temp_file}: {e}")
            continue

        if len(all_tasks) == 0:
            continue

        # Apply sampling if max_tasks_per_cell is set
        original_count = len(all_tasks)
        if max_tasks_per_cell and len(all_tasks) > max_tasks_per_cell:
            all_tasks = random.sample(all_tasks, max_tasks_per_cell)
            print(f"   {cell_name}: sampled {len(all_tasks)} from {original_count} tasks")

        # Convert to minimal_data_per_file format
        minimal_data_per_file = [[] for _ in range(num_voltages)]

        for task in all_tasks:
            for voltage_idx, sample in enumerate(task):
                minimal_data_per_file[voltage_idx].append(sample)

        # Create cell dataset
        cell_dataset = {
            'cache_path': cache_path,
            'cache_type': cache_type,
            'minimal_data_per_file': minimal_data_per_file,
            'num_lib_files': num_voltages,
            'data_type': data_type,
            'cell_name': cell_name,
            'num_tasks': len(all_tasks),
            'num_datapoints': len(all_tasks) * num_voltages,
            'source_folders': list(cell_source_folders[cell_name]),
            'total_tasks_seen': cell_task_counts[cell_name]
        }

        # Save final .pth file
        cell_file_path = output_path / f"{data_type}_{cell_name}_graph_data_{cache_type}.pth"
        torch.save(cell_dataset, cell_file_path)

        saved_files.append((cell_name, len(all_tasks), cell_task_counts[cell_name]))
        total_tasks_saved += len(all_tasks)

        # Clear memory after each cell
        del all_tasks
        del minimal_data_per_file
        del cell_dataset
        gc.collect()

    # Cleanup temp directory
    print(f"\n🧹 Cleaning up temp directory...")
    shutil.rmtree(temp_dir)

    # Final stats
    print(f"\n✅ Streaming test dataset complete!")
    print(f"   Cache type: {cache_type}")
    print(f"   Total cells: {len(saved_files)}")
    print(f"   Total tasks saved: {total_tasks_saved}")
    print(f"   Total datapoints: {total_tasks_saved * num_voltages}")
    print(f"   Node features: 11D (7 base + 4 process)")

    if max_tasks_per_cell:
        total_seen = sum(cell_task_counts.values())
        print(f"   Tasks seen: {total_seen}, kept: {total_tasks_saved} ({100*total_tasks_saved/total_seen:.1f}%)")

    print(f"\n📋 Saved files ({len(saved_files)} cells):")
    for cell_name, num_tasks, total_seen in sorted(saved_files)[:10]:
        if max_tasks_per_cell and total_seen > num_tasks:
            print(f"   - {cell_name}: {num_tasks} tasks (sampled from {total_seen})")
        else:
            print(f"   - {cell_name}: {num_tasks} tasks")
    if len(saved_files) > 10:
        print(f"   ... and {len(saved_files) - 10} more cells")

    print("=" * 80)

    # Final cleanup
    del topology_cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


# ============== PARALLEL PROCESSING FUNCTIONS ==============

def _process_single_folder_worker_streaming(args):
    """
    Worker function for processing a single folder in parallel.
    Saves results to temp pickle file (memory-efficient).
    Returns (folder_idx, temp_file_path, cell_task_counts, folder_task_count).
    """
    folder_idx, folder_info, cache_path, cache_type, data_type, num_voltages, temp_dir, slew_mode, voltage_mode = args
    folder_path, lib_base_path, prefix, start, end = folder_info

    import pickle
    from collections import defaultdict

    # Load topology cache (each worker loads its own copy)
    topology_cache = torch.load(cache_path, weights_only=False)

    # Step 1: Collect all lib files for this folder
    minimal_data_per_file = []

    for i in range(start, end):
        v_str = f"{i//100}{i%100:02d}"
        filename = f"{lib_base_path}/{folder_path}/{prefix}{v_str}.lib"

        try:
            minimal_samples = dataextract_gnn_cached_with_process(
                filename, topology_cache, cache_type, prefix, data_type, is_test=True,
                slew_mode=slew_mode, voltage_mode=voltage_mode
            )

            if minimal_samples:
                for sample in minimal_samples:
                    sample['source_folder'] = folder_path
                    sample['source_voltage'] = i
                minimal_data_per_file.append(minimal_samples)
            else:
                minimal_data_per_file.append([])

        except Exception as e:
            minimal_data_per_file.append([])
            continue

    # Clean up topology cache early
    del topology_cache
    gc.collect()

    # Step 2: Build tasks and group by cell
    if len(minimal_data_per_file) != num_voltages:
        return folder_idx, None, {}, 0

    # Get number of tasks
    num_tasks_in_folder = 0
    for lib_samples in minimal_data_per_file:
        if lib_samples:
            num_tasks_in_folder = len(lib_samples)
            break

    if num_tasks_in_folder == 0:
        del minimal_data_per_file
        gc.collect()
        return folder_idx, None, {}, 0

    # Group by cell for this folder
    folder_tasks_by_cell = defaultdict(list)

    for task_idx in range(num_tasks_in_folder):
        task_samples = []
        cell_name = None

        for voltage_idx in range(num_voltages):
            if voltage_idx < len(minimal_data_per_file) and task_idx < len(minimal_data_per_file[voltage_idx]):
                sample = minimal_data_per_file[voltage_idx][task_idx]
                task_samples.append(sample)
                if cell_name is None:
                    cell_name = sample['cell_name']

        if len(task_samples) == num_voltages and cell_name:
            folder_tasks_by_cell[cell_name].append(task_samples)

    # Calculate counts before saving
    cell_task_counts = {cell: len(tasks) for cell, tasks in folder_tasks_by_cell.items()}
    folder_task_count = sum(cell_task_counts.values())

    # Step 3: Save to temp pickle file (one file per folder)
    temp_file_path = os.path.join(temp_dir, f"folder_{folder_idx:04d}.pkl")
    with open(temp_file_path, 'wb') as f:
        pickle.dump(dict(folder_tasks_by_cell), f)

    # Clean up
    del minimal_data_per_file
    del folder_tasks_by_cell
    gc.collect()

    return folder_idx, temp_file_path, cell_task_counts, folder_task_count


def build_batch_test_data_parallel(
        cache_path,
        cache_type,
        test_folders,
        output_dir,
        data_type="cell",
        max_tasks_per_cell=None,
        num_workers=4,
        filter_cells=None,
        slew_mode='all',
        voltage_mode='all_nodes'
):
    """
    Build test dataset using PARALLEL processing with STREAMING saves.

    Memory-efficient: Each worker saves to temp file, main process merges incrementally.

    Args:
        cache_path: Path to topology cache
        cache_type: 'full_graph' or 'stage_aware'
        test_folders: List of tuples (folder_path, lib_base_path, prefix, start, end)
        output_dir: Output directory for cell-based .pth files
        data_type: 'cell' or 'transition'
        max_tasks_per_cell: Maximum TASKS per cell (applied at the end via sampling)
        num_workers: Number of parallel workers (default: 4)
        filter_cells: Set of cell names to filter (only these cells will be processed)
        slew_mode: 'all' (apply to all inputs) or 'related_pin_only' (apply to related_pin only)
    """
    import multiprocessing as mp
    import pickle
    import shutil
    from collections import defaultdict

    print("=" * 80)
    print(f"BUILDING BATCH TEST DATASET (PARALLEL + STREAMING MODE - {num_workers} workers)")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Cache type: {cache_type}")
    print(f"Data type: {data_type}")
    print(f"Test folders: {len(test_folders)}")
    print(f"Max TASKS per cell: {max_tasks_per_cell if max_tasks_per_cell else 'No limit'}")
    print(f"Filter cells: {len(filter_cells)} cells" if filter_cells else "Filter cells: All cells")
    print(f"Workers: {num_workers}")
    print("=" * 80)

    # Determine num_voltages from first folder
    first_folder = test_folders[0]
    num_voltages = first_folder[4] - first_folder[3]  # end - start

    # Create temp directory for worker outputs
    temp_dir = os.path.join(output_dir, "_temp_parallel")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"\n📁 Temp directory: {temp_dir}")

    # Prepare worker arguments
    worker_args = [
        (folder_idx, folder_info, cache_path, cache_type, data_type, num_voltages, temp_dir, slew_mode, voltage_mode)
        for folder_idx, folder_info in enumerate(test_folders)
    ]

    print(f"\n🚀 Starting PARALLEL processing with {num_workers} workers...")
    print(f"   Each worker saves to temp file (memory-efficient)")

    # Track results
    temp_files = []
    cell_task_counts = defaultdict(int)
    cell_source_folders = defaultdict(set)
    completed = 0
    total = len(test_folders)

    # Use multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        for folder_idx, temp_file, counts, task_count in pool.imap_unordered(
            _process_single_folder_worker_streaming, worker_args
        ):
            completed += 1

            if temp_file:
                temp_files.append(temp_file)
                folder_path = test_folders[folder_idx][0]
                for cell_name, count in counts.items():
                    cell_task_counts[cell_name] += count
                    cell_source_folders[cell_name].add(folder_path)

            # Progress report
            if completed % 20 == 0 or completed == total:
                print(f"   📊 Progress: {completed}/{total} folders ({100*completed/total:.1f}%)")

    print(f"\n✅ Parallel processing complete!")
    print(f"   Temp files: {len(temp_files)}")
    print(f"   Total cells found: {len(cell_task_counts)}")

    # Step 2: Merge temp files into per-cell data (batch streaming merge)
    print(f"\n📦 Merging temp files (batch size=10)...")

    # Create per-cell temp directory
    cell_temp_dir = os.path.join(output_dir, "_temp_cells")
    os.makedirs(cell_temp_dir, exist_ok=True)

    # Batch processing: accumulate 10 temp files before writing to reduce I/O
    BATCH_SIZE = 10
    cell_batch_data = defaultdict(list)
    sorted_temp_files = sorted(temp_files)
    total_temp_files = len(sorted_temp_files)

    for temp_idx, temp_file in enumerate(sorted_temp_files):
        try:
            with open(temp_file, 'rb') as f:
                folder_data = pickle.load(f)
        except Exception as e:
            print(f"   ⚠️ Error loading {temp_file}: {e}")
            continue

        # Accumulate to batch (in memory)
        for cell_name, tasks in folder_data.items():
            # Apply cell filter if specified
            if filter_cells is not None and cell_name not in filter_cells:
                continue
            cell_batch_data[cell_name].extend(tasks)

        del folder_data

        # Flush batch to files every BATCH_SIZE or at the end
        if (temp_idx + 1) % BATCH_SIZE == 0 or temp_idx == total_temp_files - 1:
            print(f"   Merging temp files {temp_idx + 2 - min(BATCH_SIZE, (temp_idx % BATCH_SIZE) + 1)}-{temp_idx + 1}/{total_temp_files} ({len(cell_batch_data)} cells)...")

            for cell_name, tasks in cell_batch_data.items():
                cell_temp_file = os.path.join(cell_temp_dir, f"{cell_name}.pkl")

                existing_tasks = []
                if os.path.exists(cell_temp_file):
                    try:
                        with open(cell_temp_file, 'rb') as f:
                            existing_tasks = pickle.load(f)
                    except:
                        existing_tasks = []

                existing_tasks.extend(tasks)

                with open(cell_temp_file, 'wb') as f:
                    pickle.dump(existing_tasks, f)

            # Clear batch memory
            cell_batch_data.clear()
            gc.collect()

    # Clean up folder temp files
    print(f"\n🧹 Cleaning up folder temp files...")
    shutil.rmtree(temp_dir)

    # Step 3: Convert per-cell temp files to final .pth files
    print(f"\n📦 Saving final .pth files...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    total_tasks_saved = 0

    # Set random seed for sampling
    random.seed(42)

    cell_temp_files = sorted(Path(cell_temp_dir).glob("*.pkl"))
    for cell_temp_file in cell_temp_files:
        cell_name = cell_temp_file.stem

        try:
            with open(cell_temp_file, 'rb') as f:
                all_tasks = pickle.load(f)
        except Exception as e:
            print(f"   ⚠️ Error loading {cell_temp_file}: {e}")
            continue

        if len(all_tasks) == 0:
            continue

        # Apply sampling if max_tasks_per_cell is set
        original_count = len(all_tasks)
        if max_tasks_per_cell and len(all_tasks) > max_tasks_per_cell:
            all_tasks = random.sample(all_tasks, max_tasks_per_cell)
            print(f"   {cell_name}: sampled {len(all_tasks)} from {original_count} tasks")

        # Convert to minimal_data_per_file format
        minimal_data_per_file = [[] for _ in range(num_voltages)]

        for task in all_tasks:
            for voltage_idx, sample in enumerate(task):
                minimal_data_per_file[voltage_idx].append(sample)

        # Create cell dataset
        cell_dataset = {
            'cache_path': cache_path,
            'cache_type': cache_type,
            'minimal_data_per_file': minimal_data_per_file,
            'num_lib_files': num_voltages,
            'data_type': data_type,
            'cell_name': cell_name,
            'num_tasks': len(all_tasks),
            'num_datapoints': len(all_tasks) * num_voltages,
            'source_folders': list(cell_source_folders[cell_name]),
            'total_tasks_seen': cell_task_counts[cell_name]
        }

        # Save final .pth file
        cell_file_path = output_path / f"{data_type}_{cell_name}_graph_data_{cache_type}.pth"
        torch.save(cell_dataset, cell_file_path)

        saved_files.append((cell_name, len(all_tasks), cell_task_counts[cell_name]))
        total_tasks_saved += len(all_tasks)

        # Clear memory
        del all_tasks
        del minimal_data_per_file
        del cell_dataset
        gc.collect()

    # Clean up cell temp directory
    print(f"\n🧹 Cleaning up cell temp files...")
    shutil.rmtree(cell_temp_dir)

    # Final stats
    print(f"\n✅ Parallel + Streaming test dataset complete!")
    print(f"   Cache type: {cache_type}")
    print(f"   Total cells: {len(saved_files)}")
    print(f"   Total tasks saved: {total_tasks_saved}")
    print(f"   Total datapoints: {total_tasks_saved * num_voltages}")
    print(f"   Node features: 11D (7 base + 4 process)")

    if max_tasks_per_cell:
        total_seen = sum(cell_task_counts.values())
        print(f"   Tasks seen: {total_seen}, kept: {total_tasks_saved} ({100*total_tasks_saved/total_seen:.1f}%)")

    print(f"\n📋 Saved files ({len(saved_files)} cells):")
    for cell_name, num_tasks, total_seen in sorted(saved_files)[:10]:
        if max_tasks_per_cell and total_seen > num_tasks:
            print(f"   - {cell_name}: {num_tasks} tasks (sampled from {total_seen})")
        else:
            print(f"   - {cell_name}: {num_tasks} tasks")
    if len(saved_files) > 10:
        print(f"   ... and {len(saved_files) - 10} more cells")

    print("=" * 80)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


def build_batch_train_data(
        cache_path,
        cache_type,
        train_folders,  # List of (folder_path, lib_base_path, prefix, start, end) tuples
        output_path,
        data_type="cell",
        max_train_tasks=None,
        sample_ratio_per_dir=None,  # e.g., 0.1 for 10% sampling per directory
        slew_mode='all',
        voltage_mode='all_nodes'
):
    """
    Build unified train dataset by aggregating data from ALL train folders.
    Similar to TSMC version: creates [num_libs, total_nodes, 11] 3D tensor format.

    Args:
        cache_path: Path to topology cache
        cache_type: 'full_graph' or 'stage_aware'
        train_folders: List of tuples (folder_path, lib_base_path, prefix, start, end)
        output_path: Output .pth file path
        data_type: 'cell' or 'transition'
        max_train_tasks: Maximum number of tasks (random sampling from total)
        sample_ratio_per_dir: Sampling ratio per directory (e.g., 0.1 for 10%)
        slew_mode: 'all' (apply to all inputs) or 'related_pin_only' (apply to related_pin only)

    Returns:
        None
    """
    print("=" * 80)
    print(f"BUILDING UNIFIED TRAIN DATASET (ASAP7)")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Cache type: {cache_type}")
    print(f"Data type: {data_type}")
    print(f"Train folders: {len(train_folders)}")
    if sample_ratio_per_dir is not None:
        print(f"Sample ratio per dir: {sample_ratio_per_dir*100:.1f}%")
    elif max_train_tasks is not None:
        print(f"Max train tasks: {max_train_tasks}")
    else:
        print(f"Sampling: No limit (all tasks)")
    print("=" * 80)

    # Load topology cache
    print(f"\n📦 Loading topology cache...")
    try:
        topology_cache = torch.load(cache_path, weights_only=False)
        print(f"   ✓ Loaded {len(topology_cache)} cells (cache type: {cache_type})")
    except Exception as e:
        print(f"   ❌ Error loading cache: {e}")
        raise

    # Collect all tasks from all directories
    # Each task: {'samples_by_lib': {lib_idx: sample}, 'metadata': {...}}
    all_train_tasks = []
    num_libs = None  # Will be determined from first folder

    for folder_idx, (folder_path, lib_base_path, prefix, start, end) in enumerate(train_folders):
        folder_full_path = Path(lib_base_path) / folder_path
        print(f"\n[{folder_idx+1}/{len(train_folders)}] Processing: {folder_path}")

        # Check number of lib files
        current_num_libs = end - start
        if num_libs is None:
            num_libs = current_num_libs
            print(f"   Number of lib files (voltages): {num_libs}")
        elif current_num_libs != num_libs:
            print(f"   ⚠️  Warning: Expected {num_libs} lib files, found {current_num_libs}")

        # Process lib files and collect samples per lib
        samples_per_lib = []  # List of list of samples, indexed by lib_idx
        for lib_idx, i in enumerate(range(start, end)):
            v_str = f"{i//100}{i%100:02d}"
            filename = f"{folder_full_path}/{prefix}{v_str}.lib"

            try:
                # is_test=False for training parameter ranges
                minimal_samples = dataextract_gnn_cached_with_process(
                    filename, topology_cache, cache_type, prefix, data_type, is_test=False,
                    slew_mode=slew_mode, voltage_mode=voltage_mode
                )
                samples_per_lib.append(minimal_samples if minimal_samples else [])
            except Exception as e:
                print(f"      ⚠️  Error processing {filename}: {e}")
                samples_per_lib.append([])

        # Create tasks (same index across lib files = same task)
        if samples_per_lib and samples_per_lib[0]:
            num_tasks_in_folder = len(samples_per_lib[0])
            folder_tasks = []

            # Exclude INTRA_TOPOLOGY_CELLS from training data
            excluded_count = 0
            for task_idx in range(num_tasks_in_folder):
                samples_by_lib = {}
                valid = True
                for lib_idx in range(num_libs):
                    if lib_idx < len(samples_per_lib) and task_idx < len(samples_per_lib[lib_idx]):
                        samples_by_lib[lib_idx] = samples_per_lib[lib_idx][task_idx]
                    else:
                        valid = False
                        break

                if valid and len(samples_by_lib) == num_libs:
                    # Check if cell is in INTRA_TOPOLOGY_CELLS - exclude from train
                    cell_name = samples_by_lib[0]['cell_name']
                    if cell_name in INTRA_TOPOLOGY_CELLS:
                        excluded_count += 1
                        continue  # Skip this cell for training

                    folder_tasks.append({
                        'folder': folder_path,
                        'samples_by_lib': samples_by_lib
                    })

            if excluded_count > 0:
                print(f"   ⚠️  Excluded {excluded_count} tasks from INTRA_TOPOLOGY_CELLS")

            # Apply per-directory sampling if sample_ratio_per_dir is set
            if sample_ratio_per_dir is not None and sample_ratio_per_dir > 0 and sample_ratio_per_dir < 1.0:
                original_count = len(folder_tasks)
                sample_count = max(1, int(original_count * sample_ratio_per_dir))
                if sample_count < original_count:
                    random.seed(42 + folder_idx)  # Different seed per folder for variety
                    folder_tasks = random.sample(folder_tasks, sample_count)
                    print(f"   ✓ Sampled {sample_count}/{original_count} tasks ({sample_ratio_per_dir*100:.0f}%)")
                else:
                    print(f"   ✓ Added all {len(folder_tasks)} tasks (no sampling needed)")
            else:
                print(f"   ✓ Added {len(folder_tasks)} tasks from {num_libs} lib files")

            all_train_tasks.extend(folder_tasks)

        gc.collect()

    print(f"\n📊 Total tasks collected: {len(all_train_tasks)}")
    print(f"   Number of libs (voltages): {num_libs}")

    if len(all_train_tasks) == 0:
        print("❌ No valid tasks found!")
        del topology_cache
        gc.collect()
        return None

    # Apply random sampling if max_train_tasks is set
    if max_train_tasks is not None and max_train_tasks > 0 and len(all_train_tasks) > max_train_tasks:
        print(f"\n🎲 Applying random sampling: {len(all_train_tasks)} → {max_train_tasks} tasks")
        random.seed(42)
        all_train_tasks = random.sample(all_train_tasks, max_train_tasks)

    # Calculate total nodes and validate consistency
    print(f"\n🔧 Validating task consistency...")
    task_node_counts = []
    cell_names = []
    delay_types = []
    output_names = []
    for task_info in all_train_tasks:
        sample = task_info['samples_by_lib'][0]
        task_node_counts.append(sample['node_features'].shape[0])  # Get num_nodes from tensor shape
        cell_names.append(sample['cell_name'])
        delay_types.append(sample.get('delay_type', 'rise'))
        output_names.append(sample.get('output_name', ''))

    total_nodes = sum(task_node_counts)
    num_tasks = len(all_train_tasks)
    num_features = 11

    print(f"   Total tasks: {num_tasks}")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Num libs: {num_libs}")
    print(f"   Features: {num_features}")

    # Create node slices
    node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
    node_slices[1:] = np.cumsum(task_node_counts)

    # Allocate 3D tensors: [num_libs, total_nodes, num_features]
    print(f"\n📦 Allocating tensors: [{num_libs}, {total_nodes}, {num_features}]")
    node_features_3d = torch.zeros(num_libs, total_nodes, num_features, dtype=torch.float32)
    outputs_2d = torch.zeros(num_libs, num_tasks, dtype=torch.float32)

    # Fill tensors
    print(f"📝 Filling tensors...")
    for task_idx, task_info in enumerate(all_train_tasks):
        start_idx = node_slices[task_idx]
        end_idx = node_slices[task_idx + 1]

        for lib_idx in range(num_libs):
            sample = task_info['samples_by_lib'][lib_idx]
            node_features_3d[lib_idx, start_idx:end_idx, :] = sample['node_features']
            outputs_2d[lib_idx, task_idx] = sample['output']

        if (task_idx + 1) % 10000 == 0:
            print(f"   Processed {task_idx + 1}/{num_tasks} tasks...")

    # Calculate normalization statistics
    print(f"\n📊 Calculating normalization statistics...")
    col_name_map = {4: 'voltage', 5: 'input_slew', 6: 'output_load', 10: 'temperature'}
    norm_stats = {}

    for col, name in col_name_map.items():
        col_data = node_features_3d[:, :, col].reshape(-1)
        norm_stats[name] = {
            'mean': col_data.mean().item(),
            'std': col_data.std().item()
        }
        print(f"   {name} (col {col}): mean={norm_stats[name]['mean']:.6f}, std={norm_stats[name]['std']:.6f}")

    # Create unified dataset
    unified_dataset = {
        'node_features': node_features_3d,  # [num_libs, total_nodes, 11]
        'outputs': outputs_2d,              # [num_libs, num_tasks]
        'node_slices': node_slices,         # [num_tasks + 1]
        'cell_names': cell_names,           # List of cell names
        'delay_types': delay_types,         # List of delay types (for stage_aware)
        'output_names': output_names,       # List of output names (for stage_aware)
        'norm_stats': norm_stats,           # Normalization statistics
        'cache_path': cache_path,
        'cache_type': cache_type,
        'data_type': data_type,
        'num_libs': num_libs,
        'num_tasks': num_tasks,
        'total_nodes': total_nodes,
        'num_conditions': len(train_folders),
        'format': 'unified_3d',             # Dataset format identifier
    }

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Saving unified train dataset...")
    print(f"   Output: {output_path}")
    torch.save(unified_dataset, output_path)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {file_size:.2f} MB")

    print(f"\n✅ Unified train dataset saved!")
    print(f"   Format: node_features [{num_libs}, {total_nodes}, {num_features}]")
    print(f"   Format: outputs [{num_libs}, {num_tasks}]")
    print(f"   Tasks: {num_tasks}")
    print(f"   Conditions: {len(train_folders)}")
    print(f"   Norm stats: Included")
    print("=" * 80)

    # Clean up
    del topology_cache
    del all_train_tasks
    del node_features_3d
    del outputs_2d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


# ============== PARALLEL TRAIN PROCESSING FUNCTIONS ==============

def _process_train_folder_worker(args):
    """
    Worker function for processing a single train folder in parallel.
    Saves results to temp pickle file (memory-efficient).
    Returns (folder_idx, temp_file_path, metadata).
    """
    folder_idx, folder_info, cache_path, cache_type, data_type, num_libs, temp_dir, sample_ratio, slew_mode, voltage_mode = args
    folder_path, lib_base_path, prefix, start, end = folder_info

    import pickle
    from collections import defaultdict

    # Load topology cache (each worker loads its own copy)
    topology_cache = torch.load(cache_path, weights_only=False)

    folder_full_path = Path(lib_base_path) / folder_path

    # Process lib files and collect samples per lib
    samples_per_lib = []
    for lib_idx, i in enumerate(range(start, end)):
        v_str = f"{i//100}{i%100:02d}"
        filename = f"{folder_full_path}/{prefix}{v_str}.lib"

        try:
            minimal_samples = dataextract_gnn_cached_with_process(
                filename, topology_cache, cache_type, prefix, data_type, is_test=False,
                slew_mode=slew_mode, voltage_mode=voltage_mode
            )
            samples_per_lib.append(minimal_samples if minimal_samples else [])
        except Exception as e:
            samples_per_lib.append([])

    # Clean up topology cache early
    del topology_cache
    gc.collect()

    # Create tasks (same index across lib files = same task)
    folder_tasks = []
    metadata = {
        'folder_path': folder_path,
        'num_tasks': 0,
        'node_counts': [],
        'excluded_count': 0
    }

    if samples_per_lib and samples_per_lib[0]:
        num_tasks_in_folder = len(samples_per_lib[0])
        excluded_count = 0

        for task_idx in range(num_tasks_in_folder):
            samples_by_lib = {}
            valid = True
            for lib_idx in range(num_libs):
                if lib_idx < len(samples_per_lib) and task_idx < len(samples_per_lib[lib_idx]):
                    samples_by_lib[lib_idx] = samples_per_lib[lib_idx][task_idx]
                else:
                    valid = False
                    break

            if valid and len(samples_by_lib) == num_libs:
                cell_name = samples_by_lib[0]['cell_name']
                if cell_name in INTRA_TOPOLOGY_CELLS:
                    excluded_count += 1
                    continue

                folder_tasks.append({
                    'folder': folder_path,
                    'samples_by_lib': samples_by_lib
                })

        metadata['excluded_count'] = excluded_count

        # Apply per-directory sampling
        if sample_ratio is not None and sample_ratio > 0 and sample_ratio < 1.0:
            original_count = len(folder_tasks)
            sample_count = max(1, int(original_count * sample_ratio))
            if sample_count < original_count:
                random.seed(42 + folder_idx)
                folder_tasks = random.sample(folder_tasks, sample_count)

    # Calculate metadata
    metadata['num_tasks'] = len(folder_tasks)
    metadata['node_counts'] = [task['samples_by_lib'][0]['node_features'].shape[0] for task in folder_tasks]

    # Save to temp file
    temp_file_path = os.path.join(temp_dir, f"train_folder_{folder_idx:04d}.pkl")
    with open(temp_file_path, 'wb') as f:
        pickle.dump(folder_tasks, f)

    # Clean up
    del samples_per_lib
    del folder_tasks
    gc.collect()

    return folder_idx, temp_file_path, metadata


def build_batch_train_data_parallel(
        cache_path,
        cache_type,
        train_folders,
        output_path,
        data_type="cell",
        max_train_tasks=None,
        sample_ratio_per_dir=None,
        num_workers=4,
        slew_mode='all',
        voltage_mode='all_nodes'
):
    """
    Build unified train dataset using PARALLEL processing with STREAMING saves.

    Memory-efficient: Each worker saves to temp file, main process merges incrementally.

    Args:
        cache_path: Path to topology cache
        cache_type: 'full_graph' or 'stage_aware'
        train_folders: List of tuples (folder_path, lib_base_path, prefix, start, end)
        output_path: Output .pth file path
        data_type: 'cell' or 'transition'
        max_train_tasks: Maximum number of tasks (random sampling from total)
        sample_ratio_per_dir: Sampling ratio per directory (e.g., 0.1 for 10%)
        num_workers: Number of parallel workers
        slew_mode: 'all' or 'related_pin_only'
    """
    import multiprocessing as mp
    import pickle
    import shutil

    print("=" * 80)
    print(f"BUILDING UNIFIED TRAIN DATASET (PARALLEL + STREAMING - {num_workers} workers)")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Cache type: {cache_type}")
    print(f"Data type: {data_type}")
    print(f"Train folders: {len(train_folders)}")
    if sample_ratio_per_dir is not None:
        print(f"Sample ratio per dir: {sample_ratio_per_dir*100:.1f}%")
    if max_train_tasks is not None:
        print(f"Max train tasks: {max_train_tasks}")
    print(f"Workers: {num_workers}")
    print("=" * 80)

    # Determine num_libs from first folder
    first_folder = train_folders[0]
    num_libs = first_folder[4] - first_folder[3]
    print(f"\nNumber of lib files (voltages): {num_libs}")

    # Create temp directory
    output_dir = Path(output_path).parent
    temp_dir = output_dir / "_temp_train_parallel"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Temp directory: {temp_dir}")

    # Prepare worker arguments
    worker_args = [
        (folder_idx, folder_info, cache_path, cache_type, data_type, num_libs, str(temp_dir), sample_ratio_per_dir, slew_mode, voltage_mode)
        for folder_idx, folder_info in enumerate(train_folders)
    ]

    print(f"\n🚀 Starting PARALLEL processing with {num_workers} workers...")

    # Track results
    temp_files = []
    all_metadata = []
    completed = 0
    total = len(train_folders)

    # Use multiprocessing pool
    with mp.Pool(processes=num_workers) as pool:
        for folder_idx, temp_file, metadata in pool.imap_unordered(
            _process_train_folder_worker, worker_args
        ):
            completed += 1
            metadata['folder_idx'] = folder_idx  # Track folder index in metadata
            temp_files.append((folder_idx, temp_file))
            all_metadata.append(metadata)

            if completed % 50 == 0 or completed == total:
                print(f"   📊 Progress: {completed}/{total} folders ({100*completed/total:.1f}%)")

    print(f"\n✅ Parallel processing complete!")
    print(f"   Temp files: {len(temp_files)}")

    # Sort temp files AND metadata by folder index for consistent ordering
    temp_files.sort(key=lambda x: x[0])
    all_metadata.sort(key=lambda x: x['folder_idx'])

    # Calculate total statistics (now in correct folder order)
    total_tasks_before_sampling = sum(m['num_tasks'] for m in all_metadata)
    all_node_counts = []
    for m in all_metadata:
        all_node_counts.extend(m['node_counts'])

    print(f"\n📊 Statistics:")
    print(f"   Total tasks collected: {total_tasks_before_sampling}")
    print(f"   Total excluded (INTRA_TOPOLOGY): {sum(m['excluded_count'] for m in all_metadata)}")

    # Apply global max_train_tasks sampling if needed
    selected_indices = None
    if max_train_tasks is not None and max_train_tasks > 0 and total_tasks_before_sampling > max_train_tasks:
        print(f"\n🎲 Applying global random sampling: {total_tasks_before_sampling} → {max_train_tasks} tasks")
        random.seed(42)
        selected_indices = set(random.sample(range(total_tasks_before_sampling), max_train_tasks))
        # Recalculate node counts for selected indices only
        all_node_counts = [all_node_counts[i] for i in sorted(selected_indices)]

    # Calculate final dimensions
    num_tasks = len(all_node_counts)
    total_nodes = sum(all_node_counts)
    num_features = 11

    print(f"\n📦 Final dataset dimensions:")
    print(f"   Tasks: {num_tasks}")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Libs: {num_libs}")
    print(f"   Features: {num_features}")

    # Create node slices
    node_slices = np.zeros(num_tasks + 1, dtype=np.int64)
    node_slices[1:] = np.cumsum(all_node_counts)

    # Allocate 3D tensors
    print(f"\n📦 Allocating tensors: [{num_libs}, {total_nodes}, {num_features}]")
    tensor_size_gb = (num_libs * total_nodes * num_features * 4) / (1024**3)
    print(f"   Estimated size: {tensor_size_gb:.2f} GB")

    node_features_3d = torch.zeros(num_libs, total_nodes, num_features, dtype=torch.float32)
    outputs_2d = torch.zeros(num_libs, num_tasks, dtype=torch.float32)

    # Metadata arrays
    cell_names = []
    delay_types = []
    output_names = []

    # Fill tensors by loading temp files one at a time (streaming merge)
    print(f"\n📝 Filling tensors (streaming merge)...")

    global_task_idx = 0
    filled_task_idx = 0

    for folder_idx, temp_file in temp_files:
        try:
            with open(temp_file, 'rb') as f:
                folder_tasks = pickle.load(f)
        except Exception as e:
            print(f"   ⚠️ Error loading {temp_file}: {e}")
            continue

        for task in folder_tasks:
            # Check if this task is selected (for global sampling)
            if selected_indices is not None and global_task_idx not in selected_indices:
                global_task_idx += 1
                continue

            # Get node range
            start_idx = node_slices[filled_task_idx]
            end_idx = node_slices[filled_task_idx + 1]

            # Fill data
            for lib_idx in range(num_libs):
                sample = task['samples_by_lib'][lib_idx]
                expected_nodes = end_idx - start_idx
                actual_nodes = sample['node_features'].shape[0]

                # Debug: check for shape mismatch before assignment
                if expected_nodes != actual_nodes:
                    print(f"\n❌ SHAPE MISMATCH DETECTED!")
                    print(f"   Folder: {task.get('folder', 'unknown')}")
                    print(f"   Folder idx: {folder_idx}")
                    print(f"   Global task idx: {global_task_idx}")
                    print(f"   Filled task idx: {filled_task_idx}")
                    print(f"   Lib idx: {lib_idx}")
                    print(f"   Cell: {sample.get('cell_name', 'unknown')}")
                    print(f"   Delay type: {sample.get('delay_type', 'unknown')}")
                    print(f"   Output: {sample.get('output_name', 'unknown')}")
                    print(f"   Expected nodes: {expected_nodes}")
                    print(f"   Actual nodes: {actual_nodes}")
                    print(f"   node_features shape: {sample['node_features'].shape}")
                    # Also check lib_idx 0
                    sample0 = task['samples_by_lib'][0]
                    print(f"   Lib 0 cell: {sample0.get('cell_name', 'unknown')}")
                    print(f"   Lib 0 nodes: {sample0['node_features'].shape[0]}")
                    sys.exit(1)

                node_features_3d[lib_idx, start_idx:end_idx, :] = sample['node_features']
                outputs_2d[lib_idx, filled_task_idx] = sample['output']

            # Collect metadata from first lib
            sample = task['samples_by_lib'][0]
            cell_names.append(sample['cell_name'])
            delay_types.append(sample.get('delay_type', 'rise'))
            output_names.append(sample.get('output_name', ''))

            filled_task_idx += 1
            global_task_idx += 1

        # Clean up this batch
        del folder_tasks
        gc.collect()

        if (folder_idx + 1) % 100 == 0:
            print(f"   Processed {folder_idx + 1}/{len(temp_files)} temp files, {filled_task_idx} tasks filled...")

    print(f"   ✓ Filled {filled_task_idx} tasks")

    # Clean up temp directory
    print(f"\n🧹 Cleaning up temp files...")
    shutil.rmtree(temp_dir)

    # Calculate normalization statistics
    print(f"\n📊 Calculating normalization statistics...")
    col_name_map = {4: 'voltage', 5: 'input_slew', 6: 'output_load', 10: 'temperature'}
    norm_stats = {}

    for col, name in col_name_map.items():
        col_data = node_features_3d[:, :, col].reshape(-1)
        norm_stats[name] = {
            'mean': col_data.mean().item(),
            'std': col_data.std().item()
        }
        print(f"   {name} (col {col}): mean={norm_stats[name]['mean']:.6f}, std={norm_stats[name]['std']:.6f}")

    # Create unified dataset
    unified_dataset = {
        'node_features': node_features_3d,
        'outputs': outputs_2d,
        'node_slices': node_slices,
        'cell_names': cell_names,
        'delay_types': delay_types,
        'output_names': output_names,
        'norm_stats': norm_stats,
        'cache_path': cache_path,
        'cache_type': cache_type,
        'data_type': data_type,
        'num_libs': num_libs,
        'num_tasks': num_tasks,
        'total_nodes': total_nodes,
        'num_conditions': len(train_folders),
        'format': 'unified_3d',
    }

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Saving unified train dataset...")
    print(f"   Output: {output_path}")
    torch.save(unified_dataset, output_path)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {file_size:.2f} MB")

    print(f"\n✅ Parallel train dataset complete!")
    print(f"   Format: node_features [{num_libs}, {total_nodes}, {num_features}]")
    print(f"   Format: outputs [{num_libs}, {num_tasks}]")
    print(f"   Tasks: {num_tasks}")
    print(f"   Conditions: {len(train_folders)}")
    print("=" * 80)

    # Clean up
    del node_features_3d
    del outputs_2d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build GNN dataset with process conditions using topology cache"
    )
    parser.add_argument("--cache_path", type=str, required=True,
                       help="Path to topology cache")
    parser.add_argument("--cache_type", type=str, required=True,
                       choices=['full_graph', 'stage_aware'],
                       help="Cache type: full_graph or stage_aware")
    parser.add_argument("--lib_base_path", type=str, default=None,
                       help="Base path to library files (required for single folder mode)")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Data directory name (required for single folder mode)")
    parser.add_argument("--prefix", type=str, default=None,
                       help="Prefix for .lib files (required for single folder mode)")
    parser.add_argument("--start", type=int, default=40,
                       help="Start voltage index")
    parser.add_argument("--end", type=int, default=101,
                       help="End voltage index (exclusive)")
    parser.add_argument("--data_type", type=str, default="cell",
                       choices=['cell', 'transition'],
                       help="Data type: cell (delay) or transition (slew)")
    parser.add_argument("--save_input", type=str, default=None,
                       help="Path to save the dataset (required for single folder mode)")
    parser.add_argument("--is_test", action='store_true',
                       help="Whether this is test dataset (different parameter ranges)")
    parser.add_argument("--max_samples_per_cell", type=int, default=None,
                       help="Maximum samples per cell (for batch test mode or single folder test)")
    parser.add_argument("--max_test_tasks", type=int, default=None,
                       help="[DEPRECATED] Use --max_samples_per_cell. Maximum total test tasks.")
    parser.add_argument("--max_train_tasks", type=int, default=None,
                       help="Maximum number of tasks for training (random sampling). Only used without --is_test.")

    # Batch test mode arguments (auto-detected if test_folders_file + output_dir provided)
    parser.add_argument("--test_folders_file", type=str, default=None,
                       help="Path to file containing test folder info (one per line: folder_path,lib_base_path,prefix,start,end)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for batch test mode (cell-based .pth files)")

    # Batch train mode arguments (auto-detected if train_folders_file + train_output provided)
    parser.add_argument("--train_folders_file", type=str, default=None,
                       help="Path to file containing train folder info (one per line: folder_path,lib_base_path,prefix,start,end)")
    parser.add_argument("--train_output", type=str, default=None,
                       help="Output .pth file path for unified train dataset")
    parser.add_argument("--sample_ratio_per_dir", type=float, default=None,
                       help="Sampling ratio per directory (e.g., 0.1 for 10%%). Applies before max_train_tasks.")
    parser.add_argument("--streaming", action='store_true',
                       help="Use streaming mode for test data (saves to disk after each folder, memory-efficient)")
    parser.add_argument("--parallel", action='store_true',
                       help="Use parallel processing for test data (faster than streaming)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers (default: 4, only used with --parallel)")
    parser.add_argument("--filter_cells", type=str, default=None,
                       help="Comma-separated list of cell names to process (e.g., 'AND2x6,NAND3x2'). Only these cells will be saved.")
    parser.add_argument("--slew_mode", type=str, default="all",
                       choices=['all', 'related_pin_only'],
                       help="Input slew assignment mode: 'all' (apply to all input ports) or 'related_pin_only' (apply only to related_pin)")
    parser.add_argument("--voltage_mode", type=str, default="all_nodes",
                       choices=['all_nodes', 'vdd_only', 'vdd_mos'],
                       help="Voltage feature mode: 'all_nodes' (voltage on all), 'vdd_only' (voltage on VDD only), 'vdd_mos' (voltage on VDD+MOS)")

    args = parser.parse_args()

    # Parse filter_cells if provided
    if args.filter_cells:
        args.filter_cells_set = set(args.filter_cells.split(','))
        print(f"📋 Filtering cells: {len(args.filter_cells_set)} cells")
    else:
        args.filter_cells_set = None

    # Auto-detect mode based on provided arguments:
    # - train_folders_file + train_output → batch train mode
    # - test_folders_file + output_dir → batch test mode
    # - otherwise → single folder mode (legacy)

    batch_train_mode = args.train_folders_file is not None and args.train_output is not None
    batch_test_mode = args.test_folders_file is not None and args.output_dir is not None

    # Validate: cannot have both modes
    if batch_train_mode and batch_test_mode:
        print("❌ Error: Cannot specify both train and test batch mode arguments")
        sys.exit(1)

    # Check for batch train mode
    if batch_train_mode:

        # Parse train folders file
        train_folders = []
        with open(args.train_folders_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 5:
                    folder_path = parts[0].strip()
                    lib_base_path = parts[1].strip()
                    prefix = parts[2].strip()
                    start = int(parts[3].strip())
                    end = int(parts[4].strip())
                    train_folders.append((folder_path, lib_base_path, prefix, start, end))

        if not train_folders:
            print("❌ Error: No valid train folders found in file")
            sys.exit(1)

        print(f"📂 Batch train mode: {len(train_folders)} folders")

        # Build batch train dataset
        if args.parallel:
            print(f"🚀 Using PARALLEL mode ({args.num_workers} workers)")
            build_batch_train_data_parallel(
                cache_path=args.cache_path,
                cache_type=args.cache_type,
                train_folders=train_folders,
                output_path=args.train_output,
                data_type=args.data_type,
                max_train_tasks=args.max_train_tasks,
                sample_ratio_per_dir=args.sample_ratio_per_dir,
                num_workers=args.num_workers,
                slew_mode=args.slew_mode,
                voltage_mode=args.voltage_mode
            )
        else:
            build_batch_train_data(
                cache_path=args.cache_path,
                cache_type=args.cache_type,
                train_folders=train_folders,
                output_path=args.train_output,
                data_type=args.data_type,
                max_train_tasks=args.max_train_tasks,
                sample_ratio_per_dir=args.sample_ratio_per_dir,
                slew_mode=args.slew_mode,
                voltage_mode=args.voltage_mode
            )

    # Check for batch test mode
    elif batch_test_mode:
        # Parse test folders file
        test_folders = []
        with open(args.test_folders_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    folder_path = parts[0].strip()
                    lib_base_path = parts[1].strip()
                    prefix = parts[2].strip()
                    start = int(parts[3].strip())
                    end = int(parts[4].strip()) if len(parts) > 4 else 101
                    test_folders.append((folder_path, lib_base_path, prefix, start, end))

        if not test_folders:
            print("❌ Error: No valid test folders found in file")
            sys.exit(1)

        print(f"📂 Batch test mode: {len(test_folders)} folders")

        # Build batch test dataset
        if args.parallel:
            print(f"🚀 Using PARALLEL mode ({args.num_workers} workers)")
            build_batch_test_data_parallel(
                cache_path=args.cache_path,
                cache_type=args.cache_type,
                test_folders=test_folders,
                output_dir=args.output_dir,
                data_type=args.data_type,
                max_tasks_per_cell=args.max_samples_per_cell,
                num_workers=args.num_workers,
                filter_cells=args.filter_cells_set,
                slew_mode=args.slew_mode,
                voltage_mode=args.voltage_mode
            )
        elif args.streaming:
            print("🌊 Using STREAMING mode (memory-efficient)")
            build_batch_test_data_streaming(
                cache_path=args.cache_path,
                cache_type=args.cache_type,
                test_folders=test_folders,
                output_dir=args.output_dir,
                data_type=args.data_type,
                max_tasks_per_cell=args.max_samples_per_cell,
                filter_cells=args.filter_cells_set,
                slew_mode=args.slew_mode,
                voltage_mode=args.voltage_mode
            )
        else:
            build_batch_test_data(
                cache_path=args.cache_path,
                cache_type=args.cache_type,
                test_folders=test_folders,
                output_dir=args.output_dir,
                data_type=args.data_type,
                max_tasks_per_cell=args.max_samples_per_cell,
                filter_cells=args.filter_cells_set,
                slew_mode=args.slew_mode,
                voltage_mode=args.voltage_mode
            )
    else:
        # Training mode: process single folder
        # Validate required arguments
        if not args.lib_base_path or not args.data_dir or not args.prefix or not args.save_input:
            print("❌ Error: --lib_base_path, --data_dir, --prefix, --save_input are required for training mode")
            sys.exit(1)

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
        Path(save_input).parent.mkdir(parents=True, exist_ok=True)

        print(f"📝 Output filename: {save_input}")

        # Build training dataset
        build_all_gnn_data_cached_with_process(
            cache_path=args.cache_path,
            cache_type=args.cache_type,
            start=args.start,
            end=args.end,
            prefix=args.prefix,
            save_input=save_input,
            data_dir=args.data_dir,
            lib_base_path=args.lib_base_path,
            data_type=args.data_type,
            is_test=False,  # Training mode only
            max_samples_per_cell=None,
            max_test_tasks=None,
            max_train_tasks=args.max_train_tasks,
            slew_mode=args.slew_mode,
            voltage_mode=args.voltage_mode
        )
