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

from utils.cdl_loader import CDLLoader
from utils.stage_aware_extractor_asap7 import ASAP7StageAwareExtractor


# ============================================================================
# Weighted Adjacency Matrix Utilities
# ============================================================================

def build_resistance_map_tsmc(spi_cell):
    """
    TSMC SPI cell에서 저항값 맵 생성.

    Args:
        spi_cell: SPIParser의 LogicCell (resistors 정보 포함)

    Returns:
        dict: {(node1, node2): resistance_value, ...}
              양방향 키 모두 저장 (node1, node2) 및 (node2, node1)
    """
    resistance_map = {}

    for resistor in spi_cell.resistors:
        node1, node2 = resistor.node1, resistor.node2
        value = resistor.value

        # 양방향으로 저장
        resistance_map[(node1, node2)] = value
        resistance_map[(node2, node1)] = value

    return resistance_map


def get_base_node_name(node_name):
    """
    노드 이름에서 기본 이름 추출 (variant suffix 제거).

    예: VDD:1 -> VDD, N_7:2 -> N_7, A -> A

    Args:
        node_name: 원본 노드 이름

    Returns:
        기본 노드 이름 (colon suffix 제거)
    """
    if ':' in node_name:
        parts = node_name.split(':')
        # 마지막 부분이 숫자인 경우에만 variant로 처리
        if parts[-1].isdigit():
            return ':'.join(parts[:-1])
    return node_name


def build_node_capacitance_map_tsmc(spi_cell):
    """
    TSMC SPI cell에서 노드별 기생 캐패시턴스 합산 맵 생성.

    각 노드에 연결된 모든 capacitor의 값을 합산.
    Variant 노드 (예: VDD:1, VDD:2)의 capacitance는 기본 노드 (VDD)로 합산.

    Args:
        spi_cell: SPIParser의 LogicCell (capacitors 정보 포함)

    Returns:
        dict: {base_node_name: total_capacitance, ...}
    """
    node_cap_map = {}

    for cap in spi_cell.capacitors:
        node1, node2 = cap.node1, cap.node2
        value = cap.value

        # Variant 노드를 기본 노드로 변환 (VDD:1 -> VDD)
        base_node1 = get_base_node_name(node1)
        base_node2 = get_base_node_name(node2)

        # 양쪽 노드에 모두 추가 (기본 노드 기준)
        if base_node1 not in node_cap_map:
            node_cap_map[base_node1] = 0.0
        if base_node2 not in node_cap_map:
            node_cap_map[base_node2] = 0.0

        node_cap_map[base_node1] += value
        node_cap_map[base_node2] += value

    return node_cap_map


def normalize_adjacency_weights(adjacency_matrix):
    """
    Adjacency matrix의 non-zero 값들을 정규화.

    Min-Max normalization을 사용하여 0이 아닌 값들을 [0.1, 1.0] 범위로 스케일링.
    저항이 작을수록 강한 연결 = 높은 weight

    Args:
        adjacency_matrix: torch.Tensor [num_nodes, num_nodes]

    Returns:
        normalized adjacency_matrix
    """
    # Non-zero 값들만 추출
    non_zero_mask = adjacency_matrix != 0
    non_zero_values = adjacency_matrix[non_zero_mask]

    if len(non_zero_values) == 0:
        return adjacency_matrix

    min_val = non_zero_values.min()
    max_val = non_zero_values.max()

    # 모든 값이 같으면 1.0으로 설정
    if max_val == min_val:
        normalized = adjacency_matrix.clone()
        normalized[non_zero_mask] = 1.0
        return normalized

    # 저항의 역수로 변환 (저항 작을수록 큰 weight)
    normalized = adjacency_matrix.clone()
    inv_values = 1.0 / non_zero_values
    inv_min = inv_values.min()
    inv_max = inv_values.max()

    if inv_max == inv_min:
        normalized[non_zero_mask] = 1.0
    else:
        # [0.1, 1.0] 범위로 정규화
        scaled = 0.1 + 0.9 * (inv_values - inv_min) / (inv_max - inv_min)
        normalized[non_zero_mask] = scaled

    return normalized


def get_internal_series_resistance(base_net, resistance_map):
    """
    Calculate total internal series resistance for parasitic RC nodes.

    For nodes like N_7:1 -> N_7:2 -> N_7:3, sum up the resistances
    between consecutive nodes in the same series.

    Args:
        base_net: Base net name (e.g., "N_7")
        resistance_map: 저항값 맵

    Returns:
        Total internal series resistance (0.0 if no internal resistors found)
    """
    if resistance_map is None:
        return 0.0

    # Find all N_x:y nodes belonging to this base net
    series_nodes = set()
    for (n1, n2) in resistance_map.keys():
        for node in [n1, n2]:
            if ':' in node:
                parts = node.split(':')
                if parts[0] == base_net and parts[1].isdigit():
                    series_nodes.add(node)

    if len(series_nodes) < 2:
        return 0.0

    # Sum up resistances between nodes in the same series
    internal_resistance = 0.0
    series_list = list(series_nodes)

    for i in range(len(series_list)):
        for j in range(i + 1, len(series_list)):
            n1, n2 = series_list[i], series_list[j]
            if (n1, n2) in resistance_map:
                internal_resistance += resistance_map[(n1, n2)]

    return internal_resistance


def find_resistance_with_variants(node1, node2, resistance_map):
    """
    Find resistance value between two nodes, handling N_x:y variant lookups.

    Args:
        node1, node2: Node names
        resistance_map: 저항값 맵

    Returns:
        Resistance value (1.0 if not found)
    """
    if resistance_map is None:
        return 1.0

    # Direct lookup
    if (node1, node2) in resistance_map:
        return resistance_map[(node1, node2)]

    # Try N_x:y variant lookups if one node is a base net (e.g., N_7)
    for (n1, n2), value in resistance_map.items():
        # Check if n1 matches node1 and n2 is a variant of node2
        if n1 == node1 and ':' in n2:
            parts = n2.split(':')
            if parts[0] == node2 and parts[1].isdigit():
                return value
        if n2 == node1 and ':' in n1:
            parts = n1.split(':')
            if parts[0] == node2 and parts[1].isdigit():
                return value
        # Reverse direction
        if n1 == node2 and ':' in n2:
            parts = n2.split(':')
            if parts[0] == node1 and parts[1].isdigit():
                return value
        if n2 == node2 and ':' in n1:
            parts = n1.split(':')
            if parts[0] == node1 and parts[1].isdigit():
                return value

    return 1.0


def normalize_adjacency_weights_per_cell(adjacency_matrix):
    """
    Adjacency matrix의 non-zero 값들을 cell 단위 min-max로 정규화.

    저항값에 대해 conductance (1/R)로 변환 후 [0.1, 1.0] 범위로 정규화.

    Args:
        adjacency_matrix: torch.Tensor with raw resistance values

    Returns:
        normalized adjacency_matrix (values in [0.1, 1.0])
    """
    import torch
    non_zero_mask = adjacency_matrix != 0
    non_zero_values = adjacency_matrix[non_zero_mask]

    if len(non_zero_values) == 0:
        return adjacency_matrix

    normalized = adjacency_matrix.clone()

    # Convert to conductance (1/R) - 작은 저항 = 큰 conductance = 강한 연결
    conductance = 1.0 / non_zero_values

    # Per-cell min-max normalization to [0.1, 1.0]
    min_val = conductance.min()
    max_val = conductance.max()

    if max_val > min_val:
        normalized_values = 0.1 + 0.9 * (conductance - min_val) / (max_val - min_val)
    else:
        normalized_values = torch.ones_like(conductance)  # All same value -> 1.0

    normalized[non_zero_mask] = normalized_values

    return normalized


def normalize_capacitance_per_cell(node_capacitance_map):
    """
    Node capacitance를 cell 단위 min-max로 정규화.

    [0, 1] 범위로 정규화 (0인 노드는 0 유지)

    Args:
        node_capacitance_map: dict {node_name: capacitance_value}

    Returns:
        Normalized capacitance map (values in [0, 1])
    """
    import numpy as np
    CAP_SCALE = 1e18

    # Scale and collect non-zero values
    scaled_values = {node: cap * CAP_SCALE for node, cap in node_capacitance_map.items()}
    non_zero_vals = [v for v in scaled_values.values() if v > 0]

    if not non_zero_vals:
        return scaled_values

    # Per-cell min-max normalization to [0, 1]
    min_val = min(non_zero_vals)
    max_val = max(non_zero_vals)

    normalized_map = {}
    for node, scaled_cap in scaled_values.items():
        if scaled_cap <= 0:
            normalized_map[node] = 0.0
        elif max_val > min_val:
            normalized_map[node] = (scaled_cap - min_val) / (max_val - min_val)
        else:
            normalized_map[node] = 1.0  # All same value

    return normalized_map


def add_gate_control_edges(adjacency_matrix, all_nodes, stage_info, transistor_info,
                           external_inputs, gate_control_weight=0.5):
    """
    Add gate control edges to adjacency matrix.

    Gate control edges connect intermediate gate nodes to the transistors they control.
    This allows GNN to learn the gate-transistor control relationship.

    Args:
        adjacency_matrix: torch.Tensor [num_nodes, num_nodes] - will be modified in place
        all_nodes: List of node names
        stage_info: Stage information dict with 'intermediate_gates'
        transistor_info: Dict of transistor information
        external_inputs: List of external input names (to exclude from intermediate gates)
        gate_control_weight: Weight value for gate control edges (default 0.5)

    Returns:
        int: Number of gate control edges added
    """
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    external_inputs_set = set(external_inputs)

    # Get intermediate gates (gates that are not external inputs)
    intermediate_gates = stage_info.get('intermediate_gates', [])

    gate_control_count = 0

    for gate_node in intermediate_gates:
        if gate_node in external_inputs_set:
            continue  # Skip external inputs

        if gate_node not in node_to_idx:
            continue  # Skip if gate node is not in the node list

        gate_idx = node_to_idx[gate_node]

        # Find transistors controlled by this gate
        for trans_name, trans_info in transistor_info.items():
            if trans_name not in node_to_idx:
                continue

            # Check if this transistor's gate matches the intermediate gate
            trans_gate = trans_info.get('gate', '')
            if trans_gate == gate_node:
                trans_idx = node_to_idx[trans_name]
                # Add gate control edge: gate_node -> transistor
                adjacency_matrix[gate_idx][trans_idx] = gate_control_weight
                gate_control_count += 1

    return gate_control_count


def add_input_port_edges(adjacency_matrix, all_nodes, transistor_info,
                          external_inputs, input_port_weight=0.5):
    """
    Add input port edges to adjacency matrix.

    Input port edges connect external input nodes (A, B, C, etc.) to the transistors
    whose gate is connected to that input. This allows GNN to learn the input-transistor
    control relationship, similar to full_graph topology.

    Args:
        adjacency_matrix: torch.Tensor [num_nodes, num_nodes] - will be modified in place
        all_nodes: List of node names (must include external_inputs)
        transistor_info: Dict of transistor information with 'gate' field
        external_inputs: List of external input names (A, B, C, etc.)
        input_port_weight: Weight value for input port edges (default 0.5)

    Returns:
        int: Number of input port edges added
    """
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    external_inputs_set = set(external_inputs)

    input_port_count = 0

    for input_node in external_inputs:
        if input_node not in node_to_idx:
            continue  # Skip if input node is not in the node list

        input_idx = node_to_idx[input_node]

        # Find transistors whose gate is connected to this input
        for trans_name, trans_info in transistor_info.items():
            if trans_name not in node_to_idx:
                continue

            # Check if this transistor's gate matches the external input
            trans_gate = trans_info.get('gate', '')
            if trans_gate == input_node:
                trans_idx = node_to_idx[trans_name]
                # Add input port edge: input_node -> transistor (unidirectional, like gate_control)
                adjacency_matrix[input_idx][trans_idx] = input_port_weight
                input_port_count += 1

    return input_port_count


def get_edge_resistance(edge_index, node_list, resistance_map, spi_cell):
    """
    Edge에 대한 저항값 계산.

    Args:
        edge_index: tensor [2, num_edges]
        node_list: 노드 이름 리스트
        resistance_map: 저항값 맵
        spi_cell: SPI cell 정보

    Returns:
        list of resistance values for each edge
    """
    edge_weights = []

    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0][i].item()
        dst_idx = edge_index[1][i].item()

        src_node = node_list[src_idx]
        dst_node = node_list[dst_idx]

        # Try to find resistance between nodes
        # Check direct connection first
        key = (src_node, dst_node)
        if key in resistance_map:
            edge_weights.append(resistance_map[key])
        else:
            # For transistor nodes (XM1, XM2), try M1:XXX format
            found = False
            for suffix in ['DRN', 'SRC', 'GATE']:
                src_term = f"{src_node.replace('X', '')}:{suffix}" if src_node.startswith('XM') else src_node
                dst_term = f"{dst_node.replace('X', '')}:{suffix}" if dst_node.startswith('XM') else dst_node

                if (src_term, dst_node) in resistance_map:
                    edge_weights.append(resistance_map[(src_term, dst_node)])
                    found = True
                    break
                if (src_node, dst_term) in resistance_map:
                    edge_weights.append(resistance_map[(src_node, dst_term)])
                    found = True
                    break

            if not found:
                # Try N_x:y variant lookup with internal series resistance
                base_resistance = find_resistance_with_variants(src_node, dst_node, resistance_map)

                # Add internal series resistance if dst_node is an intermediate net
                if dst_node.startswith('N_') and ':' not in dst_node:
                    internal_r = get_internal_series_resistance(dst_node, resistance_map)
                    base_resistance += internal_r
                elif src_node.startswith('N_') and ':' not in src_node:
                    internal_r = get_internal_series_resistance(src_node, resistance_map)
                    base_resistance += internal_r

                edge_weights.append(base_resistance)

    return edge_weights


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

def precompute_stage_aware_topology_asap7(cdl_path: str, output_path: str, logic_keywords=None,
                                          gate_control_weight=0.0, input_port_weight=0.0,
                                          bidirection=False):
    """
    Pre-compute multi-stage topology from ASAP7 CDL file.

    Supports complex cells (XOR, XNOR, etc.) with more than 2 stages.

    Args:
        cdl_path: CDL file path
        output_path: Output cache file path (.pth)
        logic_keywords: Logic cell keywords
        gate_control_weight: Weight for gate control edges (default 0.0 = disabled).
                            If > 0, adds edges from intermediate gates to controlled transistors.
        input_port_weight: Weight for input port edges (default 0.0 = disabled).
                          If > 0, adds input port nodes and edges to controlled transistors.
        bidirection: If True, makes all edges bidirectional (adds reverse edges).
    """

    if logic_keywords is None:
        logic_keywords = [
            'AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR',
            'INV', 'BUF', 'MUX', 'AO', 'OA', 'AOI', 'OAI',
            'MAJ', 'FA', 'HA', 'MAJI',
            'A2O1', 'O2A1'  # A2O1A1O1I, O2A1O1I series
        ]

    print("=" * 80)
    print("PRE-COMPUTING MULTI-STAGE TOPOLOGY (ASAP7 CDL)")
    print("=" * 80)
    print(f"CDL file: {cdl_path}")
    print(f"Output cache: {output_path}")
    print(f"Logic keywords: {logic_keywords}")
    print(f"Gate control edges: {'enabled (weight={})'.format(gate_control_weight) if gate_control_weight > 0 else 'disabled'}")
    print(f"Input port edges: {'enabled (weight={})'.format(input_port_weight) if input_port_weight > 0 else 'disabled'}")
    print(f"Bidirectional edges: {'enabled' if bidirection else 'disabled'}")
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
            # Add external input nodes if input_port_weight > 0 (similar to full_graph)
            if input_port_weight > 0:
                rise_nodes += external_inputs
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
                    if bidirection:
                        rise_adjacency_matrix[dst][src] = 1.0  # Add reverse edge

            # Add gate control edges if enabled
            if gate_control_weight > 0:
                rise_stage_info_temp = _multi_stage_to_dict(rise_multi_stage, external_inputs)
                rise_gate_ctrl_count = add_gate_control_edges(
                    rise_adjacency_matrix, rise_nodes, rise_stage_info_temp,
                    transistor_info, external_inputs, gate_control_weight
                )
                if rise_gate_ctrl_count > 0:
                    print(f"      Added {rise_gate_ctrl_count} gate control edges (weight={gate_control_weight})")

            # Add input port edges if enabled
            if input_port_weight > 0:
                rise_input_port_count = add_input_port_edges(
                    rise_adjacency_matrix, rise_nodes, transistor_info,
                    external_inputs, input_port_weight
                )
                if rise_input_port_count > 0:
                    print(f"      Added {rise_input_port_count} input port edges (weight={input_port_weight})")

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
            # Add external input nodes if input_port_weight > 0 (similar to full_graph)
            if input_port_weight > 0:
                fall_nodes += external_inputs
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
                    if bidirection:
                        fall_adjacency_matrix[dst][src] = 1.0  # Add reverse edge

            # Add gate control edges if enabled
            if gate_control_weight > 0:
                fall_stage_info_temp = _multi_stage_to_dict(fall_multi_stage, external_inputs)
                fall_gate_ctrl_count = add_gate_control_edges(
                    fall_adjacency_matrix, fall_nodes, fall_stage_info_temp,
                    transistor_info, external_inputs, gate_control_weight
                )
                if fall_gate_ctrl_count > 0:
                    print(f"      Added {fall_gate_ctrl_count} gate control edges (weight={gate_control_weight})")

            # Add input port edges if enabled
            if input_port_weight > 0:
                fall_input_port_count = add_input_port_edges(
                    fall_adjacency_matrix, fall_nodes, transistor_info,
                    external_inputs, input_port_weight
                )
                if fall_input_port_count > 0:
                    print(f"      Added {fall_input_port_count} input port edges (weight={input_port_weight})")

            # Convert multi-stage info to serializable format (filter external inputs)
            rise_stage_info = _multi_stage_to_dict(rise_multi_stage, external_inputs)
            fall_stage_info = _multi_stage_to_dict(fall_multi_stage, external_inputs)

            # Compute intermediate gate widths for ASAP7 (similar to TSMC logic)
            # For each intermediate gate, sum widths of transistors controlled by that gate
            all_intermediate_gates = set(rise_stage_info.get('intermediate_gates', []) +
                                         fall_stage_info.get('intermediate_gates', []))

            intermediate_gate_widths = {}

            if all_intermediate_gates:
                # Build gate -> transistors mapping
                for gate_node in all_intermediate_gates:
                    width_sum = 0.0
                    controlled_count = 0

                    for trans in spice_cell.transistors:
                        # In ASAP7 CDL, trans.gate is the direct gate net name
                        if trans.gate == gate_node:
                            trans_width = round(trans.width / 1000.0, 4) if trans.width else 0.054
                            width_sum += trans_width
                            controlled_count += 1

                    intermediate_gate_widths[gate_node] = round(width_sum, 4)
                    if controlled_count > 0:
                        print(f"         {gate_node}: width={width_sum:.4f} ({controlled_count} transistors)")

                # Store intermediate_gate_widths in stage_info for both pull_up and pull_down
                rise_stage_info['intermediate_gate_widths'] = intermediate_gate_widths
                fall_stage_info['intermediate_gate_widths'] = intermediate_gate_widths

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

        # Compute input_connected_transistors - transistors whose gate is connected to external input
        # These transistors should receive input_slew in apply_stage_aware_topology
        input_connected_transistors = []
        external_inputs_set = set(external_inputs)

        for trans in spice_cell.transistors:
            # In ASAP7 CDL, trans.gate is the direct gate net name
            if trans.gate in external_inputs_set:
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
# TSMC SPI Multi-Stage Topology
# ============================================================================

def precompute_stage_aware_topology_tsmc(spi_path: str, output_path: str, logic_keywords=None,
                                          weighted=False, gate_control_weight=0.0, input_port_weight=0.0,
                                          bidirection=False):
    """
    Pre-compute multi-stage topology from TSMC SPI file.

    Supports complex cells (XOR, XNOR, etc.) with more than 2 stages.

    Args:
        spi_path: TSMC SPI file path
        output_path: Output cache file path (.pth)
        logic_keywords: Logic cell keywords
        weighted: True면 저항값 기반 weighted adjacency matrix 생성
        gate_control_weight: Weight for gate control edges (default 0.0 = disabled).
                            If > 0, adds edges from intermediate gates to controlled transistors.
                            Typical value: 0.5 (distinguishes from current flow edges with weight 1.0)
        input_port_weight: Weight for input port edges (default 0.0 = disabled).
                          If > 0, adds input port nodes (A, B, C, etc.) and edges to controlled transistors.
                          Similar to full_graph topology. Typical value: 0.5
        bidirection: If True, makes all edges bidirectional (adds reverse edges).
    """
    from utils.spi_parser import SPIParser
    from utils.stage_aware_extractor_tsmc import TSMCStageAwareExtractor

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
    print(f"Weighted adjacency: {weighted}")
    print(f"Gate control edges: {'enabled (weight={})'.format(gate_control_weight) if gate_control_weight > 0 else 'disabled'}")
    print(f"Input port edges: {'enabled (weight={})'.format(input_port_weight) if input_port_weight > 0 else 'disabled'}")
    print(f"Bidirectional edges: {'enabled' if bidirection else 'disabled'}")
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

            # Resolve gate terminal to actual net (e.g., M1:GATE -> I or A1)
            mos_name = trans.name.replace('X', '')
            gate_terminal = f"{mos_name}:GATE"
            resolved_gate = extractor._resolve_node_connection(cell, gate_terminal) if gate_terminal in cell.connections else trans.gate

            transistor_info[trans.name] = {
                'type': trans_type,
                'width': trans_width,
                'gate': resolved_gate,  # Store resolved gate (actual net name)
                'gate_raw': trans.gate,  # Keep raw gate terminal for debugging
                'source': trans.source,
                'drain': trans.drain
            }
            transistor_nodes.append(trans.name)

        # Build resistance and capacitance maps if weighted mode
        resistance_map = None
        node_capacitance_map = None
        if weighted:
            resistance_map = build_resistance_map_tsmc(cell)
            node_capacitance_map = build_node_capacitance_map_tsmc(cell)
            print(f"   Built resistance map: {len(resistance_map) // 2} unique connections")
            print(f"   Built capacitance map: {len(node_capacitance_map)} nodes")

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
                # Add external input nodes if input_port_weight > 0 (similar to full_graph)
                if input_port_weight > 0:
                    rise_nodes += external_inputs
                rise_nodes = list(dict.fromkeys(rise_nodes))

                rise_edges, rise_edge_attrs = extractor.create_multi_stage_edges(rise_multi_stage, rise_nodes)
                rise_edge_index = torch.tensor(rise_edges, dtype=torch.int64).T if rise_edges else torch.empty((2, 0), dtype=torch.int64)
                rise_edge_attr = torch.tensor(rise_edge_attrs, dtype=torch.float32) if rise_edge_attrs else torch.empty((0, 5), dtype=torch.float32)

                # Pre-compute adjacency matrix (always binary: 0 or 1)
                rise_num_nodes = len(rise_nodes)
                rise_adjacency_matrix = torch.zeros(rise_num_nodes, rise_num_nodes, dtype=torch.float32)
                rise_weighted_adjacency_matrix = None

                if rise_edges:
                    # Binary adjacency matrix
                    for i in range(rise_edge_index.shape[1]):
                        src = rise_edge_index[0][i]
                        dst = rise_edge_index[1][i]
                        rise_adjacency_matrix[src][dst] = 1.0
                        if bidirection:
                            rise_adjacency_matrix[dst][src] = 1.0  # Add reverse edge

                    if weighted and resistance_map:
                        # Weighted adjacency matrix (separate from binary)
                        rise_edge_weights = get_edge_resistance(rise_edge_index, rise_nodes, resistance_map, cell)
                        rise_weighted_raw = torch.zeros(rise_num_nodes, rise_num_nodes, dtype=torch.float32)
                        for i in range(rise_edge_index.shape[1]):
                            src = rise_edge_index[0][i]
                            dst = rise_edge_index[1][i]
                            rise_weighted_raw[src][dst] = rise_edge_weights[i]
                            if bidirection:
                                rise_weighted_raw[dst][src] = rise_edge_weights[i]  # Same weight for reverse
                        # Normalize the weighted adjacency matrix
                        rise_weighted_adjacency_matrix = normalize_adjacency_weights_per_cell(rise_weighted_raw)

                # Add gate control edges if enabled
                if gate_control_weight > 0:
                    rise_stage_info_temp = _multi_stage_to_dict(rise_multi_stage, external_inputs)
                    rise_gate_ctrl_count = add_gate_control_edges(
                        rise_adjacency_matrix, rise_nodes, rise_stage_info_temp,
                        transistor_info, external_inputs, gate_control_weight
                    )
                    if rise_gate_ctrl_count > 0:
                        print(f"      Added {rise_gate_ctrl_count} gate control edges (weight={gate_control_weight})")

                # Add input port edges if enabled (connects external inputs to controlled transistors)
                if input_port_weight > 0:
                    rise_input_port_count = add_input_port_edges(
                        rise_adjacency_matrix, rise_nodes, transistor_info,
                        external_inputs, input_port_weight
                    )
                    if rise_input_port_count > 0:
                        print(f"      Added {rise_input_port_count} input port edges (weight={input_port_weight})")

            except Exception as e:
                print(f"      Warning: Error computing pull-up path: {e}")
                rise_nodes = power_nodes + [output_node] + external_inputs
                rise_edge_index = torch.empty((2, 0), dtype=torch.int64)
                rise_edge_attr = torch.empty((0, 5), dtype=torch.float32)
                rise_adjacency_matrix = torch.zeros(len(rise_nodes), len(rise_nodes), dtype=torch.float32)
                rise_weighted_adjacency_matrix = None
                # Create dummy multi-stage info
                from utils.stage_aware_extractor_tsmc import MultiStageInfo
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
                # Add external input nodes if input_port_weight > 0 (similar to full_graph)
                if input_port_weight > 0:
                    fall_nodes += external_inputs
                fall_nodes = list(dict.fromkeys(fall_nodes))

                fall_edges, fall_edge_attrs = extractor.create_multi_stage_edges(fall_multi_stage, fall_nodes)
                fall_edge_index = torch.tensor(fall_edges, dtype=torch.int64).T if fall_edges else torch.empty((2, 0), dtype=torch.int64)
                fall_edge_attr = torch.tensor(fall_edge_attrs, dtype=torch.float32) if fall_edge_attrs else torch.empty((0, 5), dtype=torch.float32)

                # Pre-compute adjacency matrix (always binary: 0 or 1)
                fall_num_nodes = len(fall_nodes)
                fall_adjacency_matrix = torch.zeros(fall_num_nodes, fall_num_nodes, dtype=torch.float32)
                fall_weighted_adjacency_matrix = None

                if fall_edges:
                    # Binary adjacency matrix
                    for i in range(fall_edge_index.shape[1]):
                        src = fall_edge_index[0][i]
                        dst = fall_edge_index[1][i]
                        fall_adjacency_matrix[src][dst] = 1.0
                        if bidirection:
                            fall_adjacency_matrix[dst][src] = 1.0  # Add reverse edge

                    if weighted and resistance_map:
                        # Weighted adjacency matrix (separate from binary)
                        fall_edge_weights = get_edge_resistance(fall_edge_index, fall_nodes, resistance_map, cell)
                        fall_weighted_raw = torch.zeros(fall_num_nodes, fall_num_nodes, dtype=torch.float32)
                        for i in range(fall_edge_index.shape[1]):
                            src = fall_edge_index[0][i]
                            dst = fall_edge_index[1][i]
                            fall_weighted_raw[src][dst] = fall_edge_weights[i]
                            if bidirection:
                                fall_weighted_raw[dst][src] = fall_edge_weights[i]  # Same weight for reverse
                        # Normalize the weighted adjacency matrix
                        fall_weighted_adjacency_matrix = normalize_adjacency_weights_per_cell(fall_weighted_raw)

                # Add gate control edges if enabled
                if gate_control_weight > 0:
                    fall_stage_info_temp = _multi_stage_to_dict(fall_multi_stage, external_inputs)
                    fall_gate_ctrl_count = add_gate_control_edges(
                        fall_adjacency_matrix, fall_nodes, fall_stage_info_temp,
                        transistor_info, external_inputs, gate_control_weight
                    )
                    if fall_gate_ctrl_count > 0:
                        print(f"      Added {fall_gate_ctrl_count} gate control edges (weight={gate_control_weight})")

                # Add input port edges if enabled (connects external inputs to controlled transistors)
                if input_port_weight > 0:
                    fall_input_port_count = add_input_port_edges(
                        fall_adjacency_matrix, fall_nodes, transistor_info,
                        external_inputs, input_port_weight
                    )
                    if fall_input_port_count > 0:
                        print(f"      Added {fall_input_port_count} input port edges (weight={input_port_weight})")

            except Exception as e:
                print(f"      Warning: Error computing pull-down path: {e}")
                fall_nodes = power_nodes + [output_node] + external_inputs
                fall_edge_index = torch.empty((2, 0), dtype=torch.int64)
                fall_edge_attr = torch.empty((0, 5), dtype=torch.float32)
                fall_adjacency_matrix = torch.zeros(len(fall_nodes), len(fall_nodes), dtype=torch.float32)
                fall_weighted_adjacency_matrix = None
                # Create dummy multi-stage info
                from utils.stage_aware_extractor_tsmc import MultiStageInfo
                fall_multi_stage = MultiStageInfo(num_stages=1, stages=[], all_intermediate_nodes=[])

            # Convert multi-stage info to serializable format (filter external inputs)
            rise_stage_info = _multi_stage_to_dict(rise_multi_stage, external_inputs)
            fall_stage_info = _multi_stage_to_dict(fall_multi_stage, external_inputs)

            # Store topology for this output
            pull_up_data = {
                'all_nodes': rise_nodes,
                'edge_index': rise_edge_index,
                'edge_attr': rise_edge_attr,
                'adjacency_matrix': rise_adjacency_matrix,
                'stage_info': rise_stage_info,
                'num_nodes': len(rise_nodes),
                'num_edges': len(rise_edges) if rise_edges else 0
            }
            pull_down_data = {
                'all_nodes': fall_nodes,
                'edge_index': fall_edge_index,
                'edge_attr': fall_edge_attr,
                'adjacency_matrix': fall_adjacency_matrix,
                'stage_info': fall_stage_info,
                'num_nodes': len(fall_nodes),
                'num_edges': len(fall_edges) if fall_edges else 0
            }

            # Add weighted adjacency matrices if available
            if rise_weighted_adjacency_matrix is not None:
                pull_up_data['weighted_adjacency_matrix'] = rise_weighted_adjacency_matrix
            if fall_weighted_adjacency_matrix is not None:
                pull_down_data['weighted_adjacency_matrix'] = fall_weighted_adjacency_matrix

            output_topologies[output_node] = {
                'pull_up': pull_up_data,
                'pull_down': pull_down_data
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
        cell_cache = {
            'external_inputs': external_inputs,
            'power_nodes': power_nodes,
            'output_nodes': output_nodes,
            'transistor_info': transistor_info,
            'transistor_nodes': transistor_nodes,
            'input_connected_transistors': input_connected_transistors,  # For input_slew assignment
            'output_topologies': output_topologies
        }

        # Add node capacitance map if available (weighted mode) - apply per-cell normalization
        if node_capacitance_map is not None:
            normalized_cap = normalize_capacitance_per_cell(node_capacitance_map)
            cell_cache['node_capacitance'] = normalized_cap

        stage_aware_cache[cell_name] = cell_cache

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
                               voltage, input_slew, output_load, external_inputs,
                               voltage_mode='all_nodes', slew_mode='all',
                               related_pin=None):
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
        voltage_mode: 'all_nodes' (default) - voltage applied to all nodes
                      'vdd_only' - voltage only on VDD node, 0 for others
        slew_mode: 'all' (default) - input_slew applied to all input ports/connected MOS
                   'related_pin_only' - input_slew only to related_pin node/connected MOS
        related_pin: The specific input pin that triggered the timing arc (used when slew_mode='related_pin_only')

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

    # Check if input port nodes are included in all_nodes (input_port mode)
    # If input ports are in node list, input_slew goes to input port nodes, not to transistors
    input_ports_in_graph = any(inp in all_nodes for inp in cached_external_inputs)

    # Build mapping from input port to connected transistors (for related_pin_only mode)
    # This maps each external input to the set of transistors whose gate is connected to it
    input_to_transistors = {}
    if slew_mode == 'related_pin_only':
        for trans_name, trans_info in transistor_info.items():
            gate = trans_info.get('gate')
            if gate in cached_external_inputs_set:
                if gate not in input_to_transistors:
                    input_to_transistors[gate] = set()
                input_to_transistors[gate].add(trans_name)

    # Create node features
    # Feature format: [is_power, is_port, trans_type, width, voltage, input_slew, output_load]
    # Note: input_slew assignment depends on slew_mode and whether input ports are in the graph:
    #   - slew_mode='all' (default):
    #     - If input_ports_in_graph: input_slew on ALL input port nodes, 0 on transistors
    #     - Otherwise: input_slew on transistors whose gate is connected to ANY external input
    #   - slew_mode='related_pin_only':
    #     - If input_ports_in_graph: input_slew ONLY on related_pin node, 0 on other inputs
    #     - Otherwise: input_slew ONLY on transistors connected to related_pin
    node_features = []

    for node in all_nodes:
        # Check if this is a transistor node (MM* for ASAP7, XM* for TSMC)
        is_transistor = node in transistor_nodes or node in transistor_info

        # Determine voltage value based on voltage_mode
        if voltage_mode == 'vdd_only':
            node_voltage = voltage if node == 'VDD' else 0.0
        elif voltage_mode == 'vdd_mos':
            # Apply voltage to VDD and MOS transistor nodes only
            node_voltage = voltage if (node == 'VDD' or is_transistor) else 0.0
        else:  # 'all_nodes' (default)
            node_voltage = voltage

        if is_transistor:
            if node in transistor_info:
                info = transistor_info[node]
                # input_slew assignment for transistors depends on slew_mode:
                # - If input_ports_in_graph: always 0 (input_slew is on input port nodes)
                # - slew_mode='all': input_slew if gate is connected to ANY external input
                # - slew_mode='related_pin_only': input_slew only if gate is connected to related_pin
                if input_ports_in_graph:
                    slew_value = 0.0  # input_slew is on input port nodes, not transistors
                elif slew_mode == 'related_pin_only' and related_pin:
                    # Only apply slew to transistors connected to the specific related_pin
                    connected_to_related = (
                        info.get('gate') == related_pin or
                        node in input_to_transistors.get(related_pin, set())
                    )
                    slew_value = input_slew if connected_to_related else 0.0
                else:
                    # Default: apply slew to all transistors with gate connected to external input
                    gate_is_external = (node in input_connected_transistors) or (info.get('gate') in cached_external_inputs_set)
                    slew_value = input_slew if gate_is_external else 0.0
                # Transistor: [0, 0, trans_type, width, voltage, input_slew, 0]
                node_features.append([0.0, 0.0, info['type'], info['width'], node_voltage, slew_value, 0.0])
            else:
                # Fallback
                node_features.append([0.0, 0.0, -1.0, 0.1, node_voltage, 0.0, 0.0])

        else:  # Circuit node
            if node in power_nodes:
                # Power rails: [1, 0, 0, 0, voltage, 0, 0]
                node_features.append([1.0, 0.0, 0.0, 0.0, node_voltage, 0.0, 0.0])
            elif node in output_nodes:
                # Output port: [0, 1, 0, 0, voltage, 0, output_load]
                node_features.append([0.0, 1.0, 0.0, 0.0, node_voltage, 0.0, output_load])
            elif node in cached_external_inputs_set:
                # Input port node (A, B, C, A1, A2, etc.) - like full_graph
                # slew_mode='related_pin_only': only related_pin gets input_slew
                # slew_mode='all': all input ports get input_slew
                if slew_mode == 'related_pin_only' and related_pin:
                    port_slew = input_slew if node == related_pin else 0.0
                else:
                    port_slew = input_slew
                # Input port: [0, 1, 0, 0, voltage, input_slew, 0]
                node_features.append([0.0, 1.0, 0.0, 0.0, node_voltage, port_slew, 0.0])
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
                node_features.append([0.0, 1.0, 0.0, gate_width_sum, node_voltage, 0.0, 0.0])

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

  # TSMC SPI with weighted adjacency (uses resistance values):
  python precompute_stage_aware_topology.py \\
      --spi_path /path/to/tsmc.spi \\
      --output ./topology_cache/stage_aware_cache_tsmc_weighted.pth --weighted

  # TSMC SPI with gate control edges (weight 0.5):
  python precompute_stage_aware_topology.py \\
      --spi_path /path/to/tsmc.spi \\
      --output ./topology_cache/stage_aware_cache_tsmc_gate_ctrl.pth --gate_control 0.5

  # TSMC SPI with input port nodes (weight 0.5, similar to full_graph):
  python precompute_stage_aware_topology.py \\
      --spi_path /path/to/tsmc.spi \\
      --output ./topology_cache/stage_aware_cache_tsmc_input_port.pth --input_ports 0.5

  # TSMC SPI with both gate control and input port edges:
  python precompute_stage_aware_topology.py \\
      --spi_path /path/to/tsmc.spi \\
      --output ./topology_cache/stage_aware_cache_tsmc_full.pth --gate_control 0.5 --input_ports 0.5

Note: --weighted only works with TSMC SPI files. --gate_control and --input_ports work with both.
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
    parser.add_argument("--weighted", action="store_true",
                       help="Use resistance-based weighted adjacency matrix (TSMC SPI only)")
    parser.add_argument("--gate_control", type=float, default=0.0,
                       help="Add gate control edges with specified weight (default 0.0 = disabled). "
                            "Typical value: 0.5. These edges connect intermediate gates to "
                            "the transistors they control, enabling GNN to learn gate-transistor relationships.")
    parser.add_argument("--input_ports", type=float, default=0.0,
                       help="Add input port nodes and edges with specified weight (default 0.0 = disabled). "
                            "Typical value: 0.5. These edges connect external input nodes (A, B, C, etc.) to "
                            "the transistors they control, similar to full_graph topology. "
                            "Enables GNN to learn input-transistor relationships.")
    parser.add_argument("--bidirection", action="store_true",
                       help="Make all edges bidirectional. Adds reverse edges for every edge in the graph.")

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Determine which format to use
    if args.spi_path:
        # TSMC SPI format
        precompute_stage_aware_topology_tsmc(args.spi_path, args.output, args.logic_keywords,
                                              weighted=args.weighted, gate_control_weight=args.gate_control,
                                              input_port_weight=args.input_ports,
                                              bidirection=args.bidirection)
    elif args.cdl_path:
        # ASAP7 CDL format
        precompute_stage_aware_topology_asap7(args.cdl_path, args.output, args.logic_keywords,
                                              gate_control_weight=args.gate_control,
                                              input_port_weight=args.input_ports,
                                              bidirection=args.bidirection)
    else:
        # Default to ASAP7 CDL
        default_cdl = "/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files/asap7sc7p5t_28_L.cdl"
        print(f"No input file specified, using default CDL: {default_cdl}")
        precompute_stage_aware_topology_asap7(default_cdl, args.output, args.logic_keywords,
                                              gate_control_weight=args.gate_control,
                                              input_port_weight=args.input_ports,
                                              bidirection=args.bidirection)
