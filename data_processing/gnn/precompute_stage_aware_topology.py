#!/usr/bin/env python
"""
Pre-compute multi-stage topology for both pull-up and pull-down paths.
Supports both ASAP7 CDL and TSMC SPI formats.

Multi-stage mode: For each cell, compute 2 path types:
1. Pull-up path (rise transition)
2. Pull-down path (fall transition)

Supports complex cells (XOR, XNOR, etc.) with more than 2 stages.
Stage detection continues until all gate nodes are external inputs.

Pre-computing these topologies significantly reduces dataset size.

Usage:
    # ASAP7 CDL:
    python precompute_stage_aware_topology.py \
        --cdl_path /path/to/asap7.cdl \
        --output ./topology_cache/stage_aware_cache.pth

    # TSMC SPI:
    python precompute_stage_aware_topology.py \
        --spi_path /path/to/tsmc.spi \
        --output ./topology_cache/stage_aware_cache_tsmc.pth
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.append('/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn')

from cdl_loader import CDLLoader
from stage_aware_extractor_asap7 import ASAP7StageAwareExtractor


# ============================================================================
# Helper Functions
# ============================================================================

def _multi_stage_to_dict(multi_stage_info, external_inputs=None):
    """
    Convert MultiStageInfo dataclass to serializable dictionary.

    Args:
        multi_stage_info: MultiStageInfo object
        external_inputs: List of external input names to exclude from intermediate gates

    Returns:
        dict: Serializable dictionary with stage information
    """
    if external_inputs is None:
        external_inputs = []
    external_inputs_set = set(external_inputs)

    stages_data = []
    all_transistors = []
    all_intermediate_gates = []

    for stage in multi_stage_info.stages:
        stage_dict = {
            'stage_num': stage.stage_num,
            'mos_type': stage.mos_type,
            'power_node': stage.power_node,
            'target_nodes': stage.target_nodes,
            'transistors': stage.transistors,
            'gate_nodes': stage.gate_nodes,
            'paths': stage.paths
        }
        stages_data.append(stage_dict)
        all_transistors.extend(stage.transistors)

        # Gate nodes that are not external inputs are intermediate gates
        for gate_node in stage.gate_nodes:
            if gate_node not in external_inputs_set:
                all_intermediate_gates.append(gate_node)

    # For backward compatibility with old stage_type format
    num_stages = multi_stage_info.num_stages
    if num_stages == 1:
        stage_type = 'one_stage'
    elif num_stages == 2:
        stage_type = 'two_stage'
    else:
        stage_type = f'{num_stages}_stage'

    return {
        'num_stages': num_stages,
        'stage_type': stage_type,
        'stages': stages_data,
        'all_transistors': list(dict.fromkeys(all_transistors)),
        'intermediate_gates': list(dict.fromkeys(all_intermediate_gates)),
        'all_intermediate_nodes': multi_stage_info.all_intermediate_nodes
    }


# ============================================================================
# ASAP7 CDL Multi-Stage Topology
# ============================================================================

def precompute_stage_aware_topology_asap7(cdl_path: str, output_path: str, logic_keywords=None):
    """
    Pre-compute multi-stage topology from ASAP7 CDL file.

    Supports complex cells (XOR, XNOR, etc.) with more than 2 stages.

    Args:
        cdl_path: CDL file path
        output_path: Output cache file path (.pth)
        logic_keywords: Logic cell keywords
    """

    if logic_keywords is None:
        logic_keywords = [
            'AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR',
            'INV', 'BUF', 'MUX', 'AO', 'OA', 'AOI', 'OAI',
            'MAJ', 'FA', 'HA', 'MAJI'
        ]

    print("=" * 80)
    print("PRE-COMPUTING MULTI-STAGE TOPOLOGY (ASAP7 CDL)")
    print("=" * 80)
    print(f"CDL file: {cdl_path}")
    print(f"Output cache: {output_path}")
    print(f"Logic keywords: {logic_keywords}")
    print("=" * 80)

    # Load CDL file
    print("\n Loading CDL file...")
    transformer = CDLLoader(cdl_path)
    extractor = ASAP7StageAwareExtractor(cdl_path)

    print(f"   Loaded {len(transformer.all_logic_cells)} logic cells")

    # Pre-compute topology for each logic cell
    stage_aware_cache = {}

    # Skip problematic cells (too many transistors, causing timeout)
    skip_cells = {'nd4d3bwp30p140','nr4d3bwp30p140'}

    for cell_name, spice_cell in transformer.all_logic_cells.items():
        # Filter logic cells only
        is_logic_cell = any(keyword in cell_name for keyword in logic_keywords)

        if not is_logic_cell:
            continue

        # Skip problematic cells
        if cell_name in skip_cells:
            print(f"\n ⚠️  Skipping {cell_name} (known timeout issue)")
            continue

        print(f"\n Processing: {cell_name}")

        # Auto-detect output nodes from cell ports
        power_nodes = ['VDD', 'VSS']
        all_ports = spice_cell.ports

        # Identify external inputs and outputs from ports
        potential_inputs = []
        potential_outputs = []

        for port in all_ports:
            if port not in power_nodes:
                is_used_as_gate = any(t.gate == port for t in spice_cell.transistors)
                is_used_as_output = any(port in [t.source, t.drain] for t in spice_cell.transistors)

                if is_used_as_output:
                    potential_outputs.append(port)
                elif is_used_as_gate:
                    potential_inputs.append(port)

        # Final output nodes
        if potential_outputs:
            output_nodes = potential_outputs
        else:
            output_nodes = [port for port in all_ports
                           if port not in power_nodes and port not in potential_inputs]

        if not output_nodes:
            output_nodes = ['Y']
            print(f"   No outputs detected, using default: {output_nodes}")

        external_inputs = sorted(potential_inputs)

        print(f"   Inputs: {external_inputs}")
        print(f"   Outputs: {output_nodes}")

        # Get transistor information
        transistor_info = {}
        transistor_nodes = []

        for trans in spice_cell.transistors:
            trans_type = 1.0 if 'nmos' in trans.type.lower() else -1.0
            trans_width = trans.width / 1000.0  # nm to um
            transistor_info[trans.name] = {
                'type': trans_type,
                'width': trans_width,
                'gate': trans.gate,
                'source': trans.source,
                'drain': trans.drain
            }
            transistor_nodes.append(trans.name)

        # For each output node, compute pull-up and pull-down paths
        output_topologies = {}

        for output_node in output_nodes:
            print(f"\n   Output: {output_node}")

            # 1. Pull-up path (rise transition) - Multi-stage
            print(f"      Computing pull-up path (rise)...")
            rise_multi_stage = extractor.extract_multi_stage_paths(
                spice_cell, external_inputs, 'rise_transition', output_nodes=[output_node]
            )

            # Collect nodes from all stages
            rise_nodes = power_nodes + [output_node]
            rise_nodes += rise_multi_stage.all_intermediate_nodes
            for stage in rise_multi_stage.stages:
                rise_nodes += stage.transistors
            rise_nodes = list(dict.fromkeys(rise_nodes))

            rise_edges, rise_edge_attrs = extractor.create_multi_stage_edges(rise_multi_stage, rise_nodes)
            rise_edge_index = torch.tensor(rise_edges, dtype=torch.int64).T if rise_edges else torch.empty((2, 0), dtype=torch.int64)
            rise_edge_attr = torch.tensor(rise_edge_attrs, dtype=torch.float32) if rise_edge_attrs else torch.empty((0, 5), dtype=torch.float32)

            # Pre-compute adjacency matrix
            rise_num_nodes = len(rise_nodes)
            rise_adjacency_matrix = torch.zeros(rise_num_nodes, rise_num_nodes, dtype=torch.float32)
            if rise_edges:
                for i in range(rise_edge_index.shape[1]):
                    src = rise_edge_index[0][i]
                    dst = rise_edge_index[1][i]
                    rise_adjacency_matrix[src][dst] = 1.0

            # 2. Pull-down path (fall transition) - Multi-stage
            print(f"      Computing pull-down path (fall)...")
            fall_multi_stage = extractor.extract_multi_stage_paths(
                spice_cell, external_inputs, 'fall_transition', output_nodes=[output_node]
            )

            # Collect nodes from all stages
            fall_nodes = power_nodes + [output_node]
            fall_nodes += fall_multi_stage.all_intermediate_nodes
            for stage in fall_multi_stage.stages:
                fall_nodes += stage.transistors
            fall_nodes = list(dict.fromkeys(fall_nodes))

            fall_edges, fall_edge_attrs = extractor.create_multi_stage_edges(fall_multi_stage, fall_nodes)
            fall_edge_index = torch.tensor(fall_edges, dtype=torch.int64).T if fall_edges else torch.empty((2, 0), dtype=torch.int64)
            fall_edge_attr = torch.tensor(fall_edge_attrs, dtype=torch.float32) if fall_edge_attrs else torch.empty((0, 5), dtype=torch.float32)

            # Pre-compute adjacency matrix
            fall_num_nodes = len(fall_nodes)
            fall_adjacency_matrix = torch.zeros(fall_num_nodes, fall_num_nodes, dtype=torch.float32)
            if fall_edges:
                for i in range(fall_edge_index.shape[1]):
                    src = fall_edge_index[0][i]
                    dst = fall_edge_index[1][i]
                    fall_adjacency_matrix[src][dst] = 1.0

            # Convert multi-stage info to serializable format (filter external inputs)
            rise_stage_info = _multi_stage_to_dict(rise_multi_stage, external_inputs)
            fall_stage_info = _multi_stage_to_dict(fall_multi_stage, external_inputs)

            # Store topology for this output
            output_topologies[output_node] = {
                'pull_up': {
                    'all_nodes': rise_nodes,
                    'edge_index': rise_edge_index,
                    'edge_attr': rise_edge_attr,
                    'adjacency_matrix': rise_adjacency_matrix,
                    'stage_info': rise_stage_info,
                    'num_nodes': len(rise_nodes),
                    'num_edges': len(rise_edges) if rise_edges else 0
                },
                'pull_down': {
                    'all_nodes': fall_nodes,
                    'edge_index': fall_edge_index,
                    'edge_attr': fall_edge_attr,
                    'adjacency_matrix': fall_adjacency_matrix,
                    'stage_info': fall_stage_info,
                    'num_nodes': len(fall_nodes),
                    'num_edges': len(fall_edges) if fall_edges else 0
                }
            }

            print(f"      Pull-up: {len(rise_nodes)} nodes, {len(rise_edges) if rise_edges else 0} edges ({rise_multi_stage.num_stages}-stage)")
            print(f"      Pull-down: {len(fall_nodes)} nodes, {len(fall_edges) if fall_edges else 0} edges ({fall_multi_stage.num_stages}-stage)")

        # Store cell cache
        stage_aware_cache[cell_name] = {
            'external_inputs': external_inputs,
            'power_nodes': power_nodes,
            'output_nodes': output_nodes,
            'transistor_info': transistor_info,
            'transistor_nodes': transistor_nodes,
            'output_topologies': output_topologies
        }

    # Save cache
    print(f"\n Saving stage-aware topology cache...")
    print(f"   Total cells cached: {len(stage_aware_cache)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stage_aware_cache, output_path)

    print(f"   Saved to: {output_path}")

    # Print summary
    _print_cache_summary(stage_aware_cache)

    return stage_aware_cache


# ============================================================================
# TSMC SPI Multi-Stage Topology
# ============================================================================

def precompute_stage_aware_topology_tsmc(spi_path: str, output_path: str, logic_keywords=None):
    """
    Pre-compute multi-stage topology from TSMC SPI file.

    Supports complex cells (XOR, XNOR, etc.) with more than 2 stages.

    Args:
        spi_path: TSMC SPI file path
        output_path: Output cache file path (.pth)
        logic_keywords: Logic cell keywords
    """
    from spi_parser import SPIParser
    from stage_aware_extractor_tsmc import TSMCStageAwareExtractor

    if logic_keywords is None:
        logic_keywords = [
            'INV', 'ND', 'AN', 'OR', 'XOR', 'NR', 'XNOR',
            'MUX', 'BUF', 'HA', 'FA', 'MAJ',
            'AOI', 'OAI', 'AO', 'OA',
            'CKND', 'CKAN', 'CKNR',
            'DEL', 'DF', 'SDF'
        ]

    print("=" * 80)
    print("PRE-COMPUTING MULTI-STAGE TOPOLOGY (TSMC SPI)")
    print("=" * 80)
    print(f"SPI file: {spi_path}")
    print(f"Output cache: {output_path}")
    print(f"Logic keywords: {logic_keywords}")
    print("=" * 80)

    # Load SPI file
    print("\n Loading TSMC SPI file...")
    parser = SPIParser(spi_path)
    extractor = TSMCStageAwareExtractor(spi_path)

    print(f"   Loaded {len(parser.logic_cells)} logic cells")

    # Pre-compute topology for each logic cell
    stage_aware_cache = {}

    for cell_name, cell in parser.logic_cells.items():
        print(f"\n Processing: {cell_name}")

        # Get ports
        power_nodes = ['VDD', 'VSS']
        all_ports = cell.ports

        # Identify inputs and outputs from ports
        potential_inputs = []
        potential_outputs = []

        for port in all_ports:
            if port in power_nodes:
                continue

            is_used_as_gate = False
            is_used_as_output = False

            for trans in cell.transistors:
                mos_name = trans.name.replace('X', '')
                gate_term = f"{mos_name}:GATE"
                drain_term = f"{mos_name}:DRN"
                source_term = f"{mos_name}:SRC"

                if gate_term in cell.connections:
                    connected = extractor._resolve_node_connection(cell, gate_term)
                    if connected == port:
                        is_used_as_gate = True

                for term in [drain_term, source_term]:
                    if term in cell.connections:
                        connected = extractor._resolve_node_connection(cell, term)
                        if connected == port:
                            is_used_as_output = True

            if is_used_as_output and not is_used_as_gate:
                potential_outputs.append(port)
            elif is_used_as_gate:
                potential_inputs.append(port)

        # Finalize outputs
        if potential_outputs:
            output_nodes = potential_outputs
        else:
            output_nodes = [p for p in all_ports if p in ['Z', 'ZN', 'Y', 'YN', 'CO', 'S']]
            if not output_nodes:
                output_nodes = ['Z']

        external_inputs = sorted(potential_inputs)

        print(f"   Inputs: {external_inputs}")
        print(f"   Outputs: {output_nodes}")

        # Get transistor information
        transistor_info = {}
        transistor_nodes = []

        for trans in cell.transistors:
            trans_type = 1.0 if 'nmos' in trans.type.lower() else -1.0
            trans_width = round(trans.width / 1000.0, 4) if trans.width else 0.14

            transistor_info[trans.name] = {
                'type': trans_type,
                'width': trans_width,
                'gate': trans.gate,
                'source': trans.source,
                'drain': trans.drain
            }
            transistor_nodes.append(trans.name)

        # For each output node, compute pull-up and pull-down paths (multi-stage)
        output_topologies = {}

        for output_node in output_nodes:
            print(f"\n   Output: {output_node}")

            # 1. Pull-up path (rise transition) - Multi-stage
            print(f"      Computing pull-up path (rise)...")
            try:
                rise_multi_stage = extractor.classify_multi_stage_structure(
                    cell_name, external_inputs, 'rise_transition', output_nodes=[output_node]
                )

                # Collect nodes from all stages
                rise_nodes = power_nodes + [output_node]
                rise_nodes += rise_multi_stage.all_intermediate_nodes
                for stage in rise_multi_stage.stages:
                    rise_nodes += stage.transistors
                rise_nodes = list(dict.fromkeys(rise_nodes))

                rise_edges, rise_edge_attrs = extractor.create_multi_stage_edges(rise_multi_stage, rise_nodes)
                rise_edge_index = torch.tensor(rise_edges, dtype=torch.int64).T if rise_edges else torch.empty((2, 0), dtype=torch.int64)
                rise_edge_attr = torch.tensor(rise_edge_attrs, dtype=torch.float32) if rise_edge_attrs else torch.empty((0, 5), dtype=torch.float32)

                # Pre-compute adjacency matrix
                rise_num_nodes = len(rise_nodes)
                rise_adjacency_matrix = torch.zeros(rise_num_nodes, rise_num_nodes, dtype=torch.float32)
                if rise_edges:
                    for i in range(rise_edge_index.shape[1]):
                        src = rise_edge_index[0][i]
                        dst = rise_edge_index[1][i]
                        rise_adjacency_matrix[src][dst] = 1.0

            except Exception as e:
                print(f"      Warning: Error computing pull-up path: {e}")
                rise_nodes = power_nodes + [output_node] + external_inputs
                rise_edge_index = torch.empty((2, 0), dtype=torch.int64)
                rise_edge_attr = torch.empty((0, 5), dtype=torch.float32)
                rise_adjacency_matrix = torch.zeros(len(rise_nodes), len(rise_nodes), dtype=torch.float32)
                # Create dummy multi-stage info
                from stage_aware_extractor_tsmc import MultiStageInfo
                rise_multi_stage = MultiStageInfo(num_stages=1, stages=[], all_intermediate_nodes=[])

            # 2. Pull-down path (fall transition) - Multi-stage
            print(f"      Computing pull-down path (fall)...")
            try:
                fall_multi_stage = extractor.classify_multi_stage_structure(
                    cell_name, external_inputs, 'fall_transition', output_nodes=[output_node]
                )

                # Collect nodes from all stages
                fall_nodes = power_nodes + [output_node]
                fall_nodes += fall_multi_stage.all_intermediate_nodes
                for stage in fall_multi_stage.stages:
                    fall_nodes += stage.transistors
                fall_nodes = list(dict.fromkeys(fall_nodes))

                fall_edges, fall_edge_attrs = extractor.create_multi_stage_edges(fall_multi_stage, fall_nodes)
                fall_edge_index = torch.tensor(fall_edges, dtype=torch.int64).T if fall_edges else torch.empty((2, 0), dtype=torch.int64)
                fall_edge_attr = torch.tensor(fall_edge_attrs, dtype=torch.float32) if fall_edge_attrs else torch.empty((0, 5), dtype=torch.float32)

                # Pre-compute adjacency matrix
                fall_num_nodes = len(fall_nodes)
                fall_adjacency_matrix = torch.zeros(fall_num_nodes, fall_num_nodes, dtype=torch.float32)
                if fall_edges:
                    for i in range(fall_edge_index.shape[1]):
                        src = fall_edge_index[0][i]
                        dst = fall_edge_index[1][i]
                        fall_adjacency_matrix[src][dst] = 1.0

            except Exception as e:
                print(f"      Warning: Error computing pull-down path: {e}")
                fall_nodes = power_nodes + [output_node] + external_inputs
                fall_edge_index = torch.empty((2, 0), dtype=torch.int64)
                fall_edge_attr = torch.empty((0, 5), dtype=torch.float32)
                fall_adjacency_matrix = torch.zeros(len(fall_nodes), len(fall_nodes), dtype=torch.float32)
                # Create dummy multi-stage info
                from stage_aware_extractor_tsmc import MultiStageInfo
                fall_multi_stage = MultiStageInfo(num_stages=1, stages=[], all_intermediate_nodes=[])

            # Convert multi-stage info to serializable format (filter external inputs)
            rise_stage_info = _multi_stage_to_dict(rise_multi_stage, external_inputs)
            fall_stage_info = _multi_stage_to_dict(fall_multi_stage, external_inputs)

            # Store topology for this output
            output_topologies[output_node] = {
                'pull_up': {
                    'all_nodes': rise_nodes,
                    'edge_index': rise_edge_index,
                    'edge_attr': rise_edge_attr,
                    'adjacency_matrix': rise_adjacency_matrix,
                    'stage_info': rise_stage_info,
                    'num_nodes': len(rise_nodes),
                    'num_edges': len(rise_edges) if rise_edges else 0
                },
                'pull_down': {
                    'all_nodes': fall_nodes,
                    'edge_index': fall_edge_index,
                    'edge_attr': fall_edge_attr,
                    'adjacency_matrix': fall_adjacency_matrix,
                    'stage_info': fall_stage_info,
                    'num_nodes': len(fall_nodes),
                    'num_edges': len(fall_edges) if fall_edges else 0
                }
            }

            # Compute intermediate gate widths for TSMC (CMOS logic)
            # For each intermediate gate, sum widths of transistors controlled by that gate
            # Note: external inputs are already filtered in _multi_stage_to_dict()
            all_intermediate_gates = set(rise_stage_info.get('intermediate_gates', []) +
                                         fall_stage_info.get('intermediate_gates', []))

            intermediate_gate_widths = {}

            if all_intermediate_gates:
                # Build gate -> transistors mapping using resolved gate connections
                for gate_node in all_intermediate_gates:
                    width_sum = 0.0
                    controlled_count = 0

                    for trans in cell.transistors:
                        # Resolve transistor gate to actual net
                        mos_name = trans.name.replace('X', '')
                        gate_terminal = f"{mos_name}:GATE"
                        resolved_gate = extractor._resolve_node_connection(cell, gate_terminal)

                        # Check if this transistor is controlled by the current gate_node
                        if resolved_gate == gate_node:
                            trans_width = round(trans.width / 1000.0, 4) if trans.width else 0.14
                            width_sum += trans_width
                            controlled_count += 1

                    intermediate_gate_widths[gate_node] = round(width_sum, 4)
                    if controlled_count > 0:
                        print(f"         {gate_node}: width={width_sum:.4f} ({controlled_count} transistors)")

                # Store intermediate_gate_widths in stage_info for both pull_up and pull_down
                output_topologies[output_node]['pull_up']['stage_info']['intermediate_gate_widths'] = intermediate_gate_widths
                output_topologies[output_node]['pull_down']['stage_info']['intermediate_gate_widths'] = intermediate_gate_widths

            print(f"      Pull-up: {len(rise_nodes)} nodes, {len(rise_edges) if rise_edges else 0} edges ({rise_multi_stage.num_stages}-stage)")
            print(f"      Pull-down: {len(fall_nodes)} nodes, {len(fall_edges) if fall_edges else 0} edges ({fall_multi_stage.num_stages}-stage)")

        # Compute input_connected_transistors - transistors whose gate is connected to external input
        # These transistors should receive input_slew in apply_stage_aware_topology
        input_connected_transistors = []
        external_inputs_set = set(external_inputs)

        for trans in cell.transistors:
            # Resolve transistor gate to actual net
            mos_name = trans.name.replace('X', '')
            gate_terminal = f"{mos_name}:GATE"
            resolved_gate = extractor._resolve_node_connection(cell, gate_terminal)

            # Check if this transistor's gate is connected to an external input
            if resolved_gate in external_inputs_set:
                input_connected_transistors.append(trans.name)

        if input_connected_transistors:
            print(f"   Input-connected transistors: {input_connected_transistors}")

        # Store cell cache
        stage_aware_cache[cell_name] = {
            'external_inputs': external_inputs,
            'power_nodes': power_nodes,
            'output_nodes': output_nodes,
            'transistor_info': transistor_info,
            'transistor_nodes': transistor_nodes,
            'input_connected_transistors': input_connected_transistors,  # For input_slew assignment
            'output_topologies': output_topologies
        }

    # Save cache
    print(f"\n Saving stage-aware topology cache...")
    print(f"   Total cells cached: {len(stage_aware_cache)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stage_aware_cache, output_path)

    print(f"   Saved to: {output_path}")

    # Print summary
    _print_cache_summary(stage_aware_cache)

    return stage_aware_cache


# ============================================================================
# Common Utilities
# ============================================================================

def _print_cache_summary(stage_aware_cache):
    """Print summary of cached cells with multi-stage support."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Count cells by number of stages (dynamically)
    stage_counts = {}

    for cell_name, cache in stage_aware_cache.items():
        num_outputs = len(cache['output_nodes'])
        print(f"  {cell_name:40s}: {num_outputs} outputs, {len(cache['external_inputs'])} inputs")
        for out_name, out_topo in cache['output_topologies'].items():
            up_nodes = out_topo['pull_up']['num_nodes']
            down_nodes = out_topo['pull_down']['num_nodes']

            # Get num_stages from stage_info
            up_stage_info = out_topo['pull_up'].get('stage_info', {})
            down_stage_info = out_topo['pull_down'].get('stage_info', {})

            up_num_stages = up_stage_info.get('num_stages', 1)
            down_num_stages = down_stage_info.get('num_stages', 1)

            # Convert num_stages to readable format
            up_stage_str = f"{up_num_stages}-stg"
            down_stage_str = f"{down_num_stages}-stg"

            print(f"    {out_name}: pull-up={up_nodes} nodes ({up_stage_str}), pull-down={down_nodes} nodes ({down_stage_str})")

            # Count stage types
            stage_counts[up_num_stages] = stage_counts.get(up_num_stages, 0) + 1
            stage_counts[down_num_stages] = stage_counts.get(down_num_stages, 0) + 1

    print("=" * 80)
    # Print stage statistics sorted by stage number
    stage_stats = ", ".join([f"{k}-stage={v}" for k, v in sorted(stage_counts.items())])
    print(f"Stage statistics: {stage_stats}")
    print("Multi-stage pre-computation complete!")
    print("=" * 80)


def load_stage_aware_topology_cache(cache_path: str):
    """
    Load pre-computed stage-aware topology cache.

    Args:
        cache_path: Path to cache file (.pth)

    Returns:
        dict: Stage-aware topology cache
    """
    return torch.load(cache_path, weights_only=False)


def apply_stage_aware_topology(topology_cache, cell_name, output_name, delay_type,
                               voltage, input_slew, output_load, external_inputs):
    """
    Apply voltage/slew/load to pre-computed stage-aware topology.
    Works for both ASAP7 (MM*) and TSMC (XM*) transistor naming.

    Args:
        topology_cache: Pre-computed topology cache
        cell_name: Cell name
        output_name: Output port name (e.g., 'Y', 'Z', 'CON', 'SN')
        delay_type: 'rise_transition' or 'fall_transition'
        voltage: Voltage value
        input_slew: Input slew
        output_load: Output load
        external_inputs: List of input port names

    Returns:
        dict: Node features tensor and all_nodes list
    """

    if cell_name not in topology_cache:
        raise ValueError(f"Cell {cell_name} not found in topology cache")

    cell_cache = topology_cache[cell_name]

    if output_name not in cell_cache['output_topologies']:
        raise ValueError(f"Output {output_name} not found for cell {cell_name}")

    output_topo = cell_cache['output_topologies'][output_name]

    # Select pull-up or pull-down based on delay_type
    if 'rise' in delay_type:
        path_cache = output_topo['pull_up']
    else:
        path_cache = output_topo['pull_down']

    all_nodes = path_cache['all_nodes']
    transistor_info = cell_cache['transistor_info']
    transistor_nodes = cell_cache.get('transistor_nodes', [])
    cached_external_inputs = cell_cache['external_inputs']
    cached_external_inputs_set = set(cached_external_inputs)  # For faster lookup
    power_nodes = cell_cache['power_nodes']
    output_nodes = cell_cache['output_nodes']

    # Get input_connected_transistors (TSMC: pre-computed, ASAP7: fallback to gate check)
    input_connected_transistors = set(cell_cache.get('input_connected_transistors', []))

    # Create node features
    # Feature format: [is_power, is_port, trans_type, width, voltage, input_slew, output_load]
    # Note: input_slew is assigned to transistors whose gate is connected to external input
    node_features = []

    for node in all_nodes:
        # Check if this is a transistor node (MM* for ASAP7, XM* for TSMC)
        is_transistor = node in transistor_nodes or node in transistor_info

        if is_transistor:
            if node in transistor_info:
                info = transistor_info[node]
                # Check if this transistor's gate is connected to an external input
                # TSMC: use pre-computed input_connected_transistors
                # ASAP7: fallback to direct gate name check
                gate_is_external = (node in input_connected_transistors) or (info.get('gate') in cached_external_inputs_set)
                slew_value = input_slew if gate_is_external else 0.0
                # Transistor: [0, 0, trans_type, width, voltage, input_slew_if_gate_external, 0]
                node_features.append([0.0, 0.0, info['type'], info['width'], voltage, slew_value, 0.0])
            else:
                # Fallback
                node_features.append([0.0, 0.0, -1.0, 0.1, voltage, 0.0, 0.0])

        else:  # Circuit node
            if node in power_nodes:
                # Power rails: [1, 0, 0, 0, voltage, 0, 0]
                node_features.append([1.0, 0.0, 0.0, 0.0, voltage, 0.0, 0.0])
            elif node in output_nodes:
                # Output port: [0, 1, 0, 0, voltage, 0, output_load]
                node_features.append([0.0, 1.0, 0.0, 0.0, voltage, 0.0, output_load])
            else:
                # Intermediate gate node - sum of all controlled transistor widths
                gate_width_sum = 0.0

                # Check if pre-computed intermediate_gate_widths exists (TSMC)
                stage_info = path_cache.get('stage_info', {})
                intermediate_gate_widths = stage_info.get('intermediate_gate_widths', {})

                if node in intermediate_gate_widths:
                    # Use pre-computed width (TSMC - accounts for all transistors on same net)
                    gate_width_sum = intermediate_gate_widths[node]
                else:
                    # Fallback: compute from transistor_info (ASAP7 style)
                    for trans_name, trans_info in transistor_info.items():
                        if trans_info.get('gate') == node:
                            gate_width_sum += trans_info['width']

                # Intermediate node: [0, 1, 0, gate_width_sum, voltage, 0, 0]
                node_features.append([0.0, 1.0, 0.0, gate_width_sum, voltage, 0.0, 0.0])

    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

    return {
        'node_features': node_features_tensor,
        'all_nodes': all_nodes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute stage-aware topology for pull-up/pull-down paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ASAP7 CDL:
  python precompute_stage_aware_topology.py \\
      --cdl_path /path/to/asap7.cdl \\
      --output ./topology_cache/stage_aware_cache.pth

  # TSMC SPI:
  python precompute_stage_aware_topology.py \\
      --spi_path /path/to/tsmc.spi \\
      --output ./topology_cache/stage_aware_cache_tsmc.pth
"""
    )
    parser.add_argument("--cdl_path", type=str, default=None,
                       help="Path to ASAP7 CDL file")
    parser.add_argument("--spi_path", type=str, default=None,
                       help="Path to TSMC SPI file")
    parser.add_argument("--output", type=str,
                       default="/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/stage_aware_topology_cache.pth",
                       help="Output cache file path")
    parser.add_argument("--logic_keywords", type=str, nargs='+',
                       default=None,
                       help="Logic cell keywords")

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Determine which format to use
    if args.spi_path:
        # TSMC SPI format
        precompute_stage_aware_topology_tsmc(args.spi_path, args.output, args.logic_keywords)
    elif args.cdl_path:
        # ASAP7 CDL format
        precompute_stage_aware_topology_asap7(args.cdl_path, args.output, args.logic_keywords)
    else:
        # Default to ASAP7 CDL
        default_cdl = "/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files/asap7sc7p5t_28_L.cdl"
        print(f"No input file specified, using default CDL: {default_cdl}")
        precompute_stage_aware_topology_asap7(default_cdl, args.output, args.logic_keywords)
