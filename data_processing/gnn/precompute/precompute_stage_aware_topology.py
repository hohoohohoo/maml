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
# Shared multi-stage topology helpers
# ============================================================================

def _collect_path_nodes(multi_stage, power_nodes, output_node, external_inputs, include_input_ports):
    """Collect deduplicated node list for one path direction."""
    nodes = power_nodes + [output_node]
    nodes += multi_stage.all_intermediate_nodes
    for stage in multi_stage.stages:
        nodes += stage.transistors
    if include_input_ports:
        nodes += external_inputs
    return list(dict.fromkeys(nodes))


def _edges_to_index_attr(edges, edge_attrs):
    """Convert raw edge lists to PyG edge_index [2,E] / edge_attr [E,5] tensors."""
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.int64).T
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        edge_attr = torch.empty((0, 5), dtype=torch.float32)
    return edge_index, edge_attr


def _build_binary_adjacency(edge_index, num_nodes, bidirection):
    """Binary adjacency from edge_index, optionally bidirectional."""
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    if edge_index.numel() == 0:
        return adjacency
    for i in range(edge_index.shape[1]):
        src = edge_index[0][i]
        dst = edge_index[1][i]
        adjacency[src][dst] = 1.0
        if bidirection:
            adjacency[dst][src] = 1.0
    return adjacency


def _build_weighted_adjacency_normalized(edge_index, nodes, resistance_map, cell, bidirection):
    """Resistance-weighted adjacency (TSMC only), normalized per cell."""
    num_nodes = len(nodes)
    edge_weights = get_edge_resistance(edge_index, nodes, resistance_map, cell)
    weighted_raw = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i in range(edge_index.shape[1]):
        src = edge_index[0][i]
        dst = edge_index[1][i]
        weighted_raw[src][dst] = edge_weights[i]
        if bidirection:
            weighted_raw[dst][src] = edge_weights[i]
    return normalize_adjacency_weights_per_cell(weighted_raw)


def _apply_extra_edges(adjacency, nodes, multi_stage, transistor_info, external_inputs,
                       gate_control_weight, input_port_weight):
    """In-place: optionally add gate-control + input-port edges; print counts."""
    if gate_control_weight > 0:
        stage_info_temp = _multi_stage_to_dict(multi_stage, external_inputs)
        gate_ctrl_count = add_gate_control_edges(
            adjacency, nodes, stage_info_temp,
            transistor_info, external_inputs, gate_control_weight,
        )
        if gate_ctrl_count > 0:
            print(f"      Added {gate_ctrl_count} gate control edges (weight={gate_control_weight})")
    if input_port_weight > 0:
        input_port_count = add_input_port_edges(
            adjacency, nodes, transistor_info,
            external_inputs, input_port_weight,
        )
        if input_port_count > 0:
            print(f"      Added {input_port_count} input port edges (weight={input_port_weight})")


def _build_directional_path(extractor, multi_stage, power_nodes, output_node, external_inputs,
                            transistor_info, gate_control_weight, input_port_weight, bidirection,
                            weighted=False, resistance_map=None, cell=None):
    """
    Build full path data for one direction (rise OR fall).

    Returns dict with keys: nodes, edges, edge_index, edge_attr, adjacency,
    weighted_adjacency (or None).
    """
    nodes = _collect_path_nodes(
        multi_stage, power_nodes, output_node, external_inputs,
        include_input_ports=input_port_weight > 0,
    )
    edges, edge_attrs = extractor.create_multi_stage_edges(multi_stage, nodes)
    edge_index, edge_attr = _edges_to_index_attr(edges, edge_attrs)

    adjacency = _build_binary_adjacency(edge_index, len(nodes), bidirection)
    weighted_adjacency = None
    if edges and weighted and resistance_map and cell is not None:
        weighted_adjacency = _build_weighted_adjacency_normalized(
            edge_index, nodes, resistance_map, cell, bidirection,
        )

    _apply_extra_edges(
        adjacency, nodes, multi_stage, transistor_info, external_inputs,
        gate_control_weight, input_port_weight,
    )

    return {
        'nodes': nodes,
        'edges': edges,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'adjacency': adjacency,
        'weighted_adjacency': weighted_adjacency,
    }


def _empty_path_fallback(power_nodes, output_node, external_inputs):
    """Fallback path data used by TSMC except blocks (empty edges/adjacency)."""
    nodes = power_nodes + [output_node] + external_inputs
    return {
        'nodes': nodes,
        'edges': [],
        'edge_index': torch.empty((2, 0), dtype=torch.int64),
        'edge_attr': torch.empty((0, 5), dtype=torch.float32),
        'adjacency': torch.zeros(len(nodes), len(nodes), dtype=torch.float32),
        'weighted_adjacency': None,
    }


def _make_path_cache_entry(path, stage_info):
    """Cache dict entry for one direction (pull_up or pull_down)."""
    entry = {
        'all_nodes': path['nodes'],
        'edge_index': path['edge_index'],
        'edge_attr': path['edge_attr'],
        'adjacency_matrix': path['adjacency'],
        'stage_info': stage_info,
        'num_nodes': len(path['nodes']),
        'num_edges': len(path['edges']) if path['edges'] else 0,
    }
    if path['weighted_adjacency'] is not None:
        entry['weighted_adjacency_matrix'] = path['weighted_adjacency']
    return entry


def _save_stage_aware_cache(stage_aware_cache, output_path):
    """mkdir parent + torch.save + summary print."""
    print(f"\n Saving stage-aware topology cache...")
    print(f"   Total cells cached: {len(stage_aware_cache)}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(stage_aware_cache, output_path)
    print(f"   Saved to: {output_path}")
    _print_cache_summary(stage_aware_cache)


# ----- ASAP7-specific helpers -----

def _detect_ports_stage_aware_asap7(spice_cell, power_nodes):
    """ASAP7: (output_nodes, external_inputs) from cell ports."""
    potential_inputs = []
    potential_outputs = []
    for port in spice_cell.ports:
        if port in power_nodes:
            continue
        is_used_as_gate = any(t.gate == port for t in spice_cell.transistors)
        is_used_as_output = any(port in [t.source, t.drain] for t in spice_cell.transistors)
        if is_used_as_output:
            potential_outputs.append(port)
        elif is_used_as_gate:
            potential_inputs.append(port)

    if potential_outputs:
        output_nodes = potential_outputs
    else:
        output_nodes = [
            p for p in spice_cell.ports
            if p not in power_nodes and p not in potential_inputs
        ]
    if not output_nodes:
        output_nodes = ['Y']
        print(f"   No outputs detected, using default: {output_nodes}")
    return output_nodes, sorted(potential_inputs)


def _build_transistor_info_stage_aware_asap7(spice_cell):
    """ASAP7: ({trans_name: {type, width, gate, source, drain}}, transistor_nodes)."""
    transistor_info = {}
    transistor_nodes = []
    for trans in spice_cell.transistors:
        trans_type = 1.0 if 'nmos' in trans.type.lower() else -1.0
        trans_width = trans.width / 1000.0
        transistor_info[trans.name] = {
            'type': trans_type,
            'width': trans_width,
            'gate': trans.gate,
            'source': trans.source,
            'drain': trans.drain,
        }
        transistor_nodes.append(trans.name)
    return transistor_info, transistor_nodes


def _compute_intermediate_gate_widths_asap7(spice_cell, gate_set):
    """ASAP7: per-gate sum of widths of transistors whose gate equals each gate."""
    if not gate_set:
        return {}
    widths = {}
    for gate_node in gate_set:
        width_sum = 0.0
        controlled_count = 0
        for trans in spice_cell.transistors:
            if trans.gate == gate_node:
                trans_width = round(trans.width / 1000.0, 4) if trans.width else 0.054
                width_sum += trans_width
                controlled_count += 1
        widths[gate_node] = round(width_sum, 4)
        if controlled_count > 0:
            print(f"         {gate_node}: width={width_sum:.4f} ({controlled_count} transistors)")
    return widths


def _compute_input_connected_transistors_asap7(spice_cell, external_inputs):
    """ASAP7: transistor names whose gate is an external input."""
    external_inputs_set = set(external_inputs)
    return [t.name for t in spice_cell.transistors if t.gate in external_inputs_set]


# ----- TSMC-specific helpers -----

def _detect_ports_stage_aware_tsmc(cell, extractor, power_nodes):
    """TSMC: (output_nodes, external_inputs) using the extractor's resolver."""
    potential_inputs = []
    potential_outputs = []
    for port in cell.ports:
        if port in power_nodes:
            continue
        is_used_as_gate = False
        is_used_as_output = False
        for trans in cell.transistors:
            mos_name = trans.name.replace('X', '')
            for term, role in (
                (f"{mos_name}:GATE", 'gate'),
                (f"{mos_name}:DRN", 'output'),
                (f"{mos_name}:SRC", 'output'),
            ):
                if term not in cell.connections:
                    continue
                connected = extractor._resolve_node_connection(cell, term)
                if connected != port:
                    continue
                if role == 'gate':
                    is_used_as_gate = True
                else:
                    is_used_as_output = True

        if is_used_as_output and not is_used_as_gate:
            potential_outputs.append(port)
        elif is_used_as_gate:
            potential_inputs.append(port)

    if potential_outputs:
        output_nodes = potential_outputs
    else:
        output_nodes = [p for p in cell.ports if p in ['Z', 'ZN', 'Y', 'YN', 'CO', 'S']]
        if not output_nodes:
            output_nodes = ['Z']
    return output_nodes, sorted(potential_inputs)


def _build_transistor_info_stage_aware_tsmc(cell, extractor):
    """TSMC: transistor_info with resolved gate net + gate_raw terminal."""
    transistor_info = {}
    transistor_nodes = []
    for trans in cell.transistors:
        trans_type = 1.0 if 'nmos' in trans.type.lower() else -1.0
        trans_width = round(trans.width / 1000.0, 4) if trans.width else 0.14
        mos_name = trans.name.replace('X', '')
        gate_terminal = f"{mos_name}:GATE"
        resolved_gate = (
            extractor._resolve_node_connection(cell, gate_terminal)
            if gate_terminal in cell.connections else trans.gate
        )
        transistor_info[trans.name] = {
            'type': trans_type,
            'width': trans_width,
            'gate': resolved_gate,
            'gate_raw': trans.gate,
            'source': trans.source,
            'drain': trans.drain,
        }
        transistor_nodes.append(trans.name)
    return transistor_info, transistor_nodes


def _compute_intermediate_gate_widths_tsmc(cell, extractor, gate_set):
    """TSMC: per-gate sum of widths of transistors whose resolved gate equals each gate."""
    if not gate_set:
        return {}
    widths = {}
    for gate_node in gate_set:
        width_sum = 0.0
        controlled_count = 0
        for trans in cell.transistors:
            mos_name = trans.name.replace('X', '')
            gate_terminal = f"{mos_name}:GATE"
            resolved_gate = extractor._resolve_node_connection(cell, gate_terminal)
            if resolved_gate == gate_node:
                trans_width = round(trans.width / 1000.0, 4) if trans.width else 0.14
                width_sum += trans_width
                controlled_count += 1
        widths[gate_node] = round(width_sum, 4)
        if controlled_count > 0:
            print(f"         {gate_node}: width={width_sum:.4f} ({controlled_count} transistors)")
    return widths


def _compute_input_connected_transistors_tsmc(cell, extractor, external_inputs):
    """TSMC: transistor names whose resolved gate is an external input."""
    external_inputs_set = set(external_inputs)
    result = []
    for trans in cell.transistors:
        mos_name = trans.name.replace('X', '')
        gate_terminal = f"{mos_name}:GATE"
        resolved_gate = extractor._resolve_node_connection(cell, gate_terminal)
        if resolved_gate in external_inputs_set:
            result.append(trans.name)
    return result


# ============================================================================
# ASAP7 CDL Multi-Stage Topology
# ============================================================================

def precompute_stage_aware_topology_asap7(cdl_path: str, output_path: str, logic_keywords=None,
                                          gate_control_weight=0.0, input_port_weight=0.0,
                                          bidirection=False):
    """
    Pre-compute multi-stage (rise + fall) topology cache for ASAP7 from a CDL file.

    Supports complex cells (XOR, XNOR, etc.) with more than 2 stages.

    Args:
        cdl_path: CDL file path.
        output_path: Output cache file (.pth).
        logic_keywords: Substring matchers identifying logic cells (default list provided).
        gate_control_weight: When > 0, add intermediate-gate→controlled-transistor edges.
        input_port_weight: When > 0, add input-port nodes + edges to controlled transistors.
        bidirection: When True, make every edge bidirectional in the adjacency matrix.

    Returns:
        stage_aware_cache: {cell_name: per-cell topology dict} (also saved to disk).
    """
    if logic_keywords is None:
        logic_keywords = [
            'AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR',
            'INV', 'BUF', 'MUX', 'AO', 'OA', 'AOI', 'OAI',
            'MAJ', 'FA', 'HA', 'MAJI',
            'A2O1', 'O2A1',
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

    print("\n Loading CDL file...")
    transformer = CDLLoader(cdl_path)
    extractor = ASAP7StageAwareExtractor(cdl_path)
    print(f"   Loaded {len(transformer.all_logic_cells)} logic cells")

    power_nodes = ['VDD', 'VSS']
    # Known timeouts due to large transistor counts; safe to skip.
    skip_cells = {'nd4d3bwp30p140', 'nr4d3bwp30p140'}
    stage_aware_cache = {}

    for cell_name, spice_cell in transformer.all_logic_cells.items():
        if not any(keyword in cell_name for keyword in logic_keywords):
            continue
        if cell_name in skip_cells:
            print(f"\n ⚠️  Skipping {cell_name} (known timeout issue)")
            continue

        print(f"\n Processing: {cell_name}")
        output_nodes, external_inputs = _detect_ports_stage_aware_asap7(spice_cell, power_nodes)
        print(f"   Inputs: {external_inputs}")
        print(f"   Outputs: {output_nodes}")

        transistor_info, transistor_nodes = _build_transistor_info_stage_aware_asap7(spice_cell)

        output_topologies = {}
        for output_node in output_nodes:
            print(f"\n   Output: {output_node}")

            print(f"      Computing pull-up path (rise)...")
            rise_multi = extractor.extract_multi_stage_paths(
                spice_cell, external_inputs, 'rise_transition', output_nodes=[output_node],
            )
            rise = _build_directional_path(
                extractor, rise_multi, power_nodes, output_node, external_inputs,
                transistor_info, gate_control_weight, input_port_weight, bidirection,
            )

            print(f"      Computing pull-down path (fall)...")
            fall_multi = extractor.extract_multi_stage_paths(
                spice_cell, external_inputs, 'fall_transition', output_nodes=[output_node],
            )
            fall = _build_directional_path(
                extractor, fall_multi, power_nodes, output_node, external_inputs,
                transistor_info, gate_control_weight, input_port_weight, bidirection,
            )

            rise_stage_info = _multi_stage_to_dict(rise_multi, external_inputs)
            fall_stage_info = _multi_stage_to_dict(fall_multi, external_inputs)

            gate_set = set(rise_stage_info.get('intermediate_gates', [])
                           + fall_stage_info.get('intermediate_gates', []))
            widths = _compute_intermediate_gate_widths_asap7(spice_cell, gate_set)
            if widths:
                rise_stage_info['intermediate_gate_widths'] = widths
                fall_stage_info['intermediate_gate_widths'] = widths

            output_topologies[output_node] = {
                'pull_up': _make_path_cache_entry(rise, rise_stage_info),
                'pull_down': _make_path_cache_entry(fall, fall_stage_info),
            }
            print(f"      Pull-up: {len(rise['nodes'])} nodes, {len(rise['edges'])} edges ({rise_multi.num_stages}-stage)")
            print(f"      Pull-down: {len(fall['nodes'])} nodes, {len(fall['edges'])} edges ({fall_multi.num_stages}-stage)")

        input_connected_transistors = _compute_input_connected_transistors_asap7(spice_cell, external_inputs)
        if input_connected_transistors:
            print(f"   Input-connected transistors: {input_connected_transistors}")

        stage_aware_cache[cell_name] = {
            'external_inputs': external_inputs,
            'power_nodes': power_nodes,
            'output_nodes': output_nodes,
            'transistor_info': transistor_info,
            'transistor_nodes': transistor_nodes,
            'input_connected_transistors': input_connected_transistors,
            'output_topologies': output_topologies,
        }

    _save_stage_aware_cache(stage_aware_cache, output_path)
    return stage_aware_cache


# ============================================================================
# TSMC SPI Multi-Stage Topology
# ============================================================================

def precompute_stage_aware_topology_tsmc(spi_path: str, output_path: str, logic_keywords=None,
                                          weighted=False, gate_control_weight=0.0, input_port_weight=0.0,
                                          bidirection=False, only_cells=None):
    """
    Pre-compute multi-stage (rise + fall) topology cache for TSMC from a SPI file.

    Supports complex cells (XOR, XNOR, etc.) with more than 2 stages. Loop-closing is
    always applied: cross-coupled storage feedback inside FF cells (DFCNQD1, SDFSNQD0,
    SDFCSNQD1) is preserved as an explicit cycle in the graph; for purely combinational
    cells the loop-closing pass is a no-op.

    Args:
        spi_path: TSMC SPI file path.
        output_path: Output cache file (.pth).
        logic_keywords: Substring matchers identifying logic cells (default list provided).
        weighted: When True, additionally build a resistance-weighted adjacency matrix.
        gate_control_weight: When > 0, add intermediate-gate→controlled-transistor edges
            (typical value 0.5 — distinguishes from current-flow edges with weight 1.0).
        input_port_weight: When > 0, add input-port nodes and edges to controlled
            transistors (similar to full_graph topology, typical value 0.5).
        bidirection: When True, make every edge bidirectional in the adjacency matrix.
        only_cells: Optional whitelist of cell names to process.

    Returns:
        stage_aware_cache: {cell_name: per-cell topology dict} (also saved to disk).
    """
    from utils.spi_parser import SPIParser
    from utils.stage_aware_extractor_tsmc import TSMCStageAwareExtractor, MultiStageInfo

    if logic_keywords is None:
        logic_keywords = [
            'INV', 'ND', 'AN', 'OR', 'XOR', 'NR', 'XNOR',
            'MUX', 'BUF', 'HA', 'FA', 'MAJ',
            'AOI', 'OAI', 'AO', 'OA',
            'CKND', 'CKAN', 'CKNR',
            'DEL', 'DF', 'SDF',
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

    print("\n Loading TSMC SPI file...")
    parser = SPIParser(spi_path)
    extractor = TSMCStageAwareExtractor(spi_path)
    print(f"   Loaded {len(parser.logic_cells)} logic cells")

    power_nodes = ['VDD', 'VSS']
    stage_aware_cache = {}

    for cell_name, cell in parser.logic_cells.items():
        if only_cells is not None and cell_name not in only_cells:
            continue
        print(f"\n Processing: {cell_name}")

        output_nodes, external_inputs = _detect_ports_stage_aware_tsmc(cell, extractor, power_nodes)
        print(f"   Inputs: {external_inputs}")
        print(f"   Outputs: {output_nodes}")

        transistor_info, transistor_nodes = _build_transistor_info_stage_aware_tsmc(cell, extractor)

        resistance_map = None
        node_capacitance_map = None
        if weighted:
            resistance_map = build_resistance_map_tsmc(cell)
            node_capacitance_map = build_node_capacitance_map_tsmc(cell)
            print(f"   Built resistance map: {len(resistance_map) // 2} unique connections")
            print(f"   Built capacitance map: {len(node_capacitance_map)} nodes")

        output_topologies = {}
        for output_node in output_nodes:
            print(f"\n   Output: {output_node}")

            print(f"      Computing pull-up path (rise)...")
            try:
                rise_multi = extractor.classify_multi_stage_structure(
                    cell_name, external_inputs, 'rise_transition', output_nodes=[output_node],
                )
                rise = _build_directional_path(
                    extractor, rise_multi, power_nodes, output_node, external_inputs,
                    transistor_info, gate_control_weight, input_port_weight, bidirection,
                    weighted=weighted, resistance_map=resistance_map, cell=cell,
                )
            except Exception as e:
                print(f"      Warning: Error computing pull-up path: {e}")
                rise = _empty_path_fallback(power_nodes, output_node, external_inputs)
                rise_multi = MultiStageInfo(num_stages=1, stages=[], all_intermediate_nodes=[])

            print(f"      Computing pull-down path (fall)...")
            try:
                fall_multi = extractor.classify_multi_stage_structure(
                    cell_name, external_inputs, 'fall_transition', output_nodes=[output_node],
                )
                fall = _build_directional_path(
                    extractor, fall_multi, power_nodes, output_node, external_inputs,
                    transistor_info, gate_control_weight, input_port_weight, bidirection,
                    weighted=weighted, resistance_map=resistance_map, cell=cell,
                )
            except Exception as e:
                print(f"      Warning: Error computing pull-down path: {e}")
                fall = _empty_path_fallback(power_nodes, output_node, external_inputs)
                fall_multi = MultiStageInfo(num_stages=1, stages=[], all_intermediate_nodes=[])

            rise_stage_info = _multi_stage_to_dict(rise_multi, external_inputs)
            fall_stage_info = _multi_stage_to_dict(fall_multi, external_inputs)

            gate_set = set(rise_stage_info.get('intermediate_gates', [])
                           + fall_stage_info.get('intermediate_gates', []))
            widths = _compute_intermediate_gate_widths_tsmc(cell, extractor, gate_set)
            if widths:
                rise_stage_info['intermediate_gate_widths'] = widths
                fall_stage_info['intermediate_gate_widths'] = widths

            output_topologies[output_node] = {
                'pull_up': _make_path_cache_entry(rise, rise_stage_info),
                'pull_down': _make_path_cache_entry(fall, fall_stage_info),
            }
            print(f"      Pull-up: {len(rise['nodes'])} nodes, {len(rise['edges'])} edges ({rise_multi.num_stages}-stage)")
            print(f"      Pull-down: {len(fall['nodes'])} nodes, {len(fall['edges'])} edges ({fall_multi.num_stages}-stage)")

        input_connected_transistors = _compute_input_connected_transistors_tsmc(cell, extractor, external_inputs)
        if input_connected_transistors:
            print(f"   Input-connected transistors: {input_connected_transistors}")

        cell_cache = {
            'external_inputs': external_inputs,
            'power_nodes': power_nodes,
            'output_nodes': output_nodes,
            'transistor_info': transistor_info,
            'transistor_nodes': transistor_nodes,
            'input_connected_transistors': input_connected_transistors,
            'output_topologies': output_topologies,
        }
        if node_capacitance_map is not None:
            cell_cache['node_capacitance'] = normalize_capacitance_per_cell(node_capacitance_map)
        stage_aware_cache[cell_name] = cell_cache

    _save_stage_aware_cache(stage_aware_cache, output_path)
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


def _select_path_cache(cell_cache, cell_name, output_name, delay_type):
    """Pick pull_up or pull_down topology cache based on delay_type."""
    if output_name not in cell_cache['output_topologies']:
        raise ValueError(f"Output {output_name} not found for cell {cell_name}")
    output_topo = cell_cache['output_topologies'][output_name]
    return output_topo['pull_up' if 'rise' in delay_type else 'pull_down']


def _build_input_to_transistors(transistor_info, cached_external_inputs_set):
    """Map external input port -> {transistors whose gate is that port}."""
    mapping = {}
    for trans_name, trans_info in transistor_info.items():
        gate = trans_info.get('gate')
        if gate in cached_external_inputs_set:
            mapping.setdefault(gate, set()).add(trans_name)
    return mapping


def _node_voltage(node, is_transistor, voltage, voltage_mode):
    """Resolve per-node voltage according to voltage_mode."""
    if voltage_mode == 'vdd_only':
        return voltage if node == 'VDD' else 0.0
    if voltage_mode == 'vdd_mos':
        return voltage if (node == 'VDD' or is_transistor) else 0.0
    return voltage  # 'all_nodes' default


def _transistor_slew(node, info, input_slew, slew_mode, related_pin,
                     input_ports_in_graph, input_connected_transistors,
                     cached_external_inputs_set, input_to_transistors):
    """
    Per-transistor input_slew, depending on whether input ports live in the graph
    and whether we're in related_pin_only mode.
    """
    if input_ports_in_graph:
        # Slew already lives on input port nodes; transistors get 0.
        return 0.0
    if slew_mode == 'related_pin_only' and related_pin:
        connected_to_related = (
            info.get('gate') == related_pin
            or node in input_to_transistors.get(related_pin, set())
        )
        return input_slew if connected_to_related else 0.0
    gate_is_external = (
        node in input_connected_transistors
        or info.get('gate') in cached_external_inputs_set
    )
    return input_slew if gate_is_external else 0.0


def _input_port_slew(node, input_slew, slew_mode, related_pin):
    """Per-input-port input_slew: all-of vs only-related-pin."""
    if slew_mode == 'related_pin_only' and related_pin:
        return input_slew if node == related_pin else 0.0
    return input_slew


def _intermediate_gate_width(node, path_cache, transistor_info):
    """
    Width feature for an intermediate gate node:
    prefer pre-computed `intermediate_gate_widths` (TSMC), else sum
    widths of transistors whose gate matches this node (ASAP7 fallback).
    """
    intermediate_gate_widths = path_cache.get('stage_info', {}).get('intermediate_gate_widths', {})
    if node in intermediate_gate_widths:
        return intermediate_gate_widths[node]
    return sum(
        info['width']
        for info in transistor_info.values()
        if info.get('gate') == node
    )


def apply_stage_aware_topology(topology_cache, cell_name, output_name, delay_type,
                               voltage, input_slew, output_load, external_inputs,
                               voltage_mode='all_nodes', slew_mode='all',
                               related_pin=None):
    """
    Apply voltage/slew/load to a pre-computed stage-aware topology cache and
    return a node_features tensor + node list. Works for both ASAP7 (MM*) and
    TSMC (XM*) transistor naming.

    Args:
        topology_cache: Pre-computed topology cache.
        cell_name: Cell name.
        output_name: Output port name (e.g., 'Y', 'Z', 'CON', 'SN').
        delay_type: 'rise_transition' or 'fall_transition'.
        voltage: Voltage value.
        input_slew: Input slew.
        output_load: Output load.
        external_inputs: List of input port names (kept for API compatibility).
        voltage_mode:
            - 'all_nodes' (default): voltage applied to all nodes.
            - 'vdd_only': voltage only on VDD node, 0 for others.
            - 'vdd_mos': voltage on VDD + transistor nodes only.
        slew_mode:
            - 'all' (default): input_slew applied to all input ports / connected MOS.
            - 'related_pin_only': input_slew only to the related_pin node /
              connected MOS.
        related_pin: Input pin that triggered the timing arc (used when
            slew_mode='related_pin_only').

    Feature format per node: [is_power, is_port, trans_type, width, voltage,
    input_slew, output_load].

    Returns:
        {'node_features': [N,7] tensor, 'all_nodes': list}.
    """
    if cell_name not in topology_cache:
        raise ValueError(f"Cell {cell_name} not found in topology cache")
    cell_cache = topology_cache[cell_name]
    path_cache = _select_path_cache(cell_cache, cell_name, output_name, delay_type)

    all_nodes = path_cache['all_nodes']
    transistor_info = cell_cache['transistor_info']
    transistor_nodes = cell_cache.get('transistor_nodes', [])
    cached_external_inputs = cell_cache['external_inputs']
    cached_external_inputs_set = set(cached_external_inputs)
    power_nodes = cell_cache['power_nodes']
    output_nodes = cell_cache['output_nodes']
    input_connected_transistors = set(cell_cache.get('input_connected_transistors', []))

    # When input ports are in the node list, input_slew lives on the port nodes;
    # otherwise it has to live on the gate-driven transistors.
    input_ports_in_graph = any(inp in all_nodes for inp in cached_external_inputs)
    input_to_transistors = (
        _build_input_to_transistors(transistor_info, cached_external_inputs_set)
        if slew_mode == 'related_pin_only' else {}
    )

    node_features = []
    for node in all_nodes:
        is_transistor = node in transistor_nodes or node in transistor_info
        node_voltage = _node_voltage(node, is_transistor, voltage, voltage_mode)

        if is_transistor:
            info = transistor_info.get(node)
            if info is None:
                # Fallback when transistor list and transistor_info disagree.
                node_features.append([0.0, 0.0, -1.0, 0.1, node_voltage, 0.0, 0.0])
                continue
            slew_value = _transistor_slew(
                node, info, input_slew, slew_mode, related_pin,
                input_ports_in_graph, input_connected_transistors,
                cached_external_inputs_set, input_to_transistors,
            )
            node_features.append([0.0, 0.0, info['type'], info['width'], node_voltage, slew_value, 0.0])
        elif node in power_nodes:
            node_features.append([1.0, 0.0, 0.0, 0.0, node_voltage, 0.0, 0.0])
        elif node in output_nodes:
            node_features.append([0.0, 1.0, 0.0, 0.0, node_voltage, 0.0, output_load])
        elif node in cached_external_inputs_set:
            port_slew = _input_port_slew(node, input_slew, slew_mode, related_pin)
            node_features.append([0.0, 1.0, 0.0, 0.0, node_voltage, port_slew, 0.0])
        else:
            gate_width_sum = _intermediate_gate_width(node, path_cache, transistor_info)
            node_features.append([0.0, 1.0, 0.0, gate_width_sum, node_voltage, 0.0, 0.0])

    return {
        'node_features': torch.tensor(node_features, dtype=torch.float32),
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
