#!/usr/bin/env python

"""
Stage-Aware Transform with Process Conditions
기존 transform_sample_MAML_stage_aware.py + process condition parameters
"""

import torch
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Import existing transformers
from transform_sample_MAML_stage_aware import transform_sample_stage_aware as original_transform
from stage_aware_extractor_delay_aware import DelayAwareStageExtractor, StageInfo
from cdl_loader import CDLLoader


def parse_process_conditions_from_filename(lib_prefix: str, is_test: bool = False) -> Dict[str, Any]:
    """
    Parse process conditions from lib file prefix

    Format: {cell_type}_{param_a_idx}_{param_b_idx}_{param_c_idx}_{temperature}_
    Example: invbuf_0_0_0_12p5_ → param_a=0.625, param_b=[pmos, nmos], param_c=[pmos, nmos], temp=12.5

    Args:
        lib_prefix: Library file prefix (e.g., "invbuf_0_0_0_12p5_")
        is_test: Whether this is from test dataset

    Returns:
        dict with keys: 'param_a', 'param_b_pmos', 'param_b_nmos', 'param_c_pmos', 'param_c_nmos', 'temperature', 'b_idx', 'c_idx'
    """
    # Default values (will be overwritten if parsing succeeds)
    result = {
        'param_a': 0.625,      # Process parameter A (same for NMOS/PMOS)
        'param_b_pmos': 0.089, # Process parameter B for PMOS
        'param_b_nmos': 0.06,  # Process parameter B for NMOS
        'param_c_pmos': 0.35,  # Process parameter C for PMOS
        'param_c_nmos': 0.465, # Process parameter C for NMOS
        'temperature': 25.0,   # Temperature (°C)
        'b_idx': 0,            # B parameter index
        'c_idx': 0             # C parameter index
    }

    # Parameter ranges based on test/train split
    # Format: [pmos_value, nmos_value] pairs for each index
    if is_test:
        param_a_values = [0.75, 1.0, 1.25]
        # b_idx: 0=[0.09, 0.062], 1=[0.092, 0.066], 2=[0.094, 0.07]
        param_b_values = [(0.09, 0.062), (0.092, 0.066), (0.094, 0.07)]
        # c_idx: 0=[0.36, 0.47], 1=[0.38, 0.475], 2=[0.40, 0.48]
        param_c_values = [(0.36, 0.47), (0.38, 0.475), (0.40, 0.48)]
    else:
        param_a_values = [0.625, 0.875, 1.125, 1.375]
        # b_idx: 0=[0.089, 0.06], 1=[0.091, 0.064], 2=[0.093, 0.068], 3=[0.095, 0.072]
        param_b_values = [(0.089, 0.06), (0.091, 0.064), (0.093, 0.068), (0.095, 0.072)]
        # c_idx: 0=[0.35, 0.465], 1=[0.37, 0.473], 2=[0.39, 0.478], 3=[0.41, 0.485]
        param_c_values = [(0.35, 0.465), (0.37, 0.473), (0.39, 0.478), (0.41, 0.485)]

    # Parse filename pattern: {cell_type}_{a_idx}_{b_idx}_{c_idx}_{temp}_
    # Example: invbuf_0_0_0_12p5_ or simple_0_0_0_12.5_
    pattern = r'.*_(\d+)_(\d+)_(\d+)_([\d\-mp\.]+)_?'
    match = re.search(pattern, lib_prefix)

    if match:
        a_idx = int(match.group(1))
        b_idx = int(match.group(2))
        c_idx = int(match.group(3))
        temp_str = match.group(4)

        # Store indices
        result['b_idx'] = b_idx
        result['c_idx'] = c_idx

        # Extract parameter A (same for NMOS/PMOS)
        if 0 <= a_idx < len(param_a_values):
            result['param_a'] = param_a_values[a_idx]

        # Extract parameter B (different for NMOS/PMOS)
        if 0 <= b_idx < len(param_b_values):
            result['param_b_pmos'] = param_b_values[b_idx][0]  # First value for PMOS
            result['param_b_nmos'] = param_b_values[b_idx][1]  # Second value for NMOS

        # Extract parameter C (different for NMOS/PMOS)
        if 0 <= c_idx < len(param_c_values):
            result['param_c_pmos'] = param_c_values[c_idx][0]  # First value for PMOS
            result['param_c_nmos'] = param_c_values[c_idx][1]  # Second value for NMOS

        # Parse temperature: 12p5 → 12.5, 12.5 → 12.5, m25 → -25, -25 → -25
        if 'p' in temp_str:
            # Replace 'p' with '.' (e.g., 12p5 → 12.5)
            temp_str = temp_str.replace('p', '.')
            result['temperature'] = float(temp_str)
        elif temp_str.startswith('m'):
            # m25 → -25
            result['temperature'] = -float(temp_str[1:])
        else:
            # Direct number (handles both integers and floats)
            result['temperature'] = float(temp_str)

    return result


def add_process_conditions_to_node_features(node_features: torch.Tensor,
                                            process_params: Dict[str, Any]) -> torch.Tensor:
    """
    Add process condition parameters to node features

    MOSFET nodes: Add actual process/temperature values (NMOS/PMOS specific for b, c)
    Circuit nodes: Add zeros for process/temperature features

    Original node features (7D): [is_power, is_port, is_transistor, width, voltage, input_slew, output_load]
    New node features (11D): [is_power, is_port, is_transistor, width, voltage, input_slew, output_load,
                              param_a, param_b, param_c, temperature]

    Args:
        node_features: Original node features tensor [num_nodes, 7]
        process_params: Dict with 'param_a', 'param_b_pmos', 'param_b_nmos', 'param_c_pmos', 'param_c_nmos', 'temperature'

    Returns:
        Enhanced node features tensor [num_nodes, 11]
    """
    num_nodes = node_features.shape[0]

    # Extract process parameters
    param_a = process_params.get('param_a', 0.625)
    param_b_pmos = process_params.get('param_b_pmos', 0.089)
    param_b_nmos = process_params.get('param_b_nmos', 0.06)
    param_c_pmos = process_params.get('param_c_pmos', 0.35)
    param_c_nmos = process_params.get('param_c_nmos', 0.465)
    temperature = process_params.get('temperature', 25.0)

    # Create process parameter tensor for each node based on node type
    # Node features format: [is_power, is_port, is_transistor, width, voltage, input_slew, output_load]
    # is_transistor is at index 2: 1.0 for NMOS, -1.0 for PMOS
    process_tensor_list = []

    for i in range(num_nodes):
        # Check if this is a transistor node by looking at:
        # 1. is_transistor feature (index 2) != 0
        # 2. width feature (index 3) != 0 (transistors have width)
        transistor_type = node_features[i, 2].item()  # 1.0 for NMOS, -1.0 for PMOS, 0.0 for circuit nodes
        has_width = node_features[i, 3].item() != 0.0

        if transistor_type != 0.0 and has_width:
            # MOSFET node: add actual process/temperature values
            # Select b and c parameters based on transistor type
            if transistor_type > 0:  # NMOS (transistor_type == 1.0)
                param_b = param_b_nmos
                param_c = param_c_nmos
            else:  # PMOS (transistor_type == -1.0)
                param_b = param_b_pmos
                param_c = param_c_pmos

            process_tensor_list.append([param_a, param_b, param_c, temperature])
        else:
            # Circuit node: add zeros
            process_tensor_list.append([0.0, 0.0, 0.0, 0.0])

    process_tensor = torch.tensor(process_tensor_list, dtype=torch.float32)  # [num_nodes, 4]

    # Concatenate original features + process parameters
    enhanced_features = torch.cat([node_features, process_tensor], dim=1)  # [num_nodes, 11]

    return enhanced_features


def transform_sample_with_process(sample: Dict[str, Any],
                                  cap: List[Dict[str, Any]],
                                  transformer: CDLLoader,
                                  lib_prefix: str = "",
                                  graph_mode: str = "stage_aware",
                                  is_test: bool = False) -> List[Dict[str, Any]]:
    """
    Transform sample with process conditions added to node features

    Args:
        sample: Liberty file timing sample
        cap: Capacitance info
        transformer: SPICE topology transformer
        lib_prefix: Library prefix (e.g., "invbuf_0_0_0_12p5_")
        graph_mode: "stage_aware" or "full_graph"
        is_test: Whether from test dataset

    Returns:
        List of graph samples with enhanced node features (11D instead of 7D)
    """
    # 1. Parse process conditions from filename
    process_params = parse_process_conditions_from_filename(lib_prefix, is_test)

    print(f"   🔬 Process conditions: a={process_params['param_a']:.3f}, "
          f"b_pmos={process_params['param_b_pmos']:.3f}, b_nmos={process_params['param_b_nmos']:.3f}, "
          f"c_pmos={process_params['param_c_pmos']:.3f}, c_nmos={process_params['param_c_nmos']:.3f}, "
          f"temp={process_params['temperature']:.1f}°C")

    # 2. Use original transform to get base graph samples
    graph_samples = original_transform(sample, cap, transformer, lib_prefix, graph_mode)

    # 3. Add process conditions to each graph sample
    for graph_sample in graph_samples:
        # Enhance node features: 7D → 11D
        original_features = graph_sample['node_features']
        enhanced_features = add_process_conditions_to_node_features(original_features, process_params)
        graph_sample['node_features'] = enhanced_features

        # Store process parameters in metadata
        graph_sample['process_params'] = process_params

    return graph_samples


def transform_all_logic_samples_with_process(flattened: List[Dict[str, Any]],
                                             cap: List[Dict[str, Any]],
                                             lib_prefix: str = "",
                                             graph_mode: str = "stage_aware",
                                             is_test: bool = False) -> List[Dict[str, Any]]:
    """
    Transform all logic samples with process conditions

    Args:
        flattened: List of timing samples
        cap: Capacitance info
        lib_prefix: Library prefix
        graph_mode: "stage_aware" or "full_graph"
        is_test: Whether from test dataset

    Returns:
        List of all graph samples with process conditions
    """
    # Load CDL transformer
    print(f"\n📂 Loading CDL transformer...")
    from cdl_loader import CDLLoader
    import os

    # CDL file paths
    base_path = '/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files'
    cdl_files = []
    cdl_names = ['asap7sc7p5t_28_L.cdl', 'asap7sc7p5t_28_R.cdl', 'asap7sc7p5t_28_SL.cdl', 'asap7sc7p5t_28_SRAM.cdl']

    for cdl_name in cdl_names:
        # Try relative paths first
        relative_paths = [f'../../cdl_files/{cdl_name}', f'../cdl_files/{cdl_name}', cdl_name, f'./{cdl_name}']
        absolute_path = os.path.join(base_path, cdl_name)

        for rel_path in relative_paths:
            if os.path.exists(rel_path):
                cdl_files.append(rel_path)
                break
        else:
            # Fallback to absolute path
            if os.path.exists(absolute_path):
                cdl_files.append(absolute_path)

    # Load first available CDL file
    transformer = None
    print(f"   🔍 Found {len(cdl_files)} CDL files to load")

    for cdl_file in cdl_files:
        try:
            transformer = CDLLoader(cdl_file)
            break
        except Exception as e:
            print(f"   ⚠️ Failed to load {cdl_file}: {e}")
            continue

    if transformer is None:
        raise ValueError(f"No CDL files could be loaded from: {cdl_files}")

    # Merge additional CDL files
    for cdl_file in cdl_files[1:]:
        try:
            transformer.merge_cdl(cdl_file)
        except Exception as e:
            print(f"   ⚠️ Could not merge {cdl_file}: {e}")

    print(f"   ✅ Total logic cells available: {len(transformer.all_logic_cells)}")

    # Allowed cell list - Only process cells in this list
    # If None, process all timing cells (no filtering)
    allowed_cells = None  # Set to None to process all cells

    # Example: allowed_cells = ['NAND', 'INV', 'BUF', 'OA', 'AO']

    transformed_list = []
    processed_count = 0

    for i, s in enumerate(flattened):
        # Check if this is a timing entry
        if s.get("type") != 'timing':
            continue

        cell_name = s.get('cell', '')

        # Filter by allowed_cells list if provided
        if allowed_cells is not None:
            cell_upper = cell_name.upper()
            # Check if cell name contains any of the allowed patterns
            if not any(allowed_pattern.upper() in cell_upper for allowed_pattern in allowed_cells):
                continue

        # Process this cell
        try:
            transformed_samples = transform_sample_with_process(s, cap, transformer, lib_prefix, graph_mode, is_test)
            if transformed_samples:
                transformed_list.extend(transformed_samples)
                processed_count += 1
        except Exception as e:
            print(f"   ⚠️  Error processing {cell_name}: {e}")
            continue

    print(f"\n✅ Stage-Aware transformation with process conditions complete:")
    print(f"   Processed: {processed_count} logic samples")
    print(f"   Generated: {len(transformed_list)} stage-aware graphs with process params")
    print(f"   Node feature dimension: 11 (7 base + 4 process)")

    return transformed_list
