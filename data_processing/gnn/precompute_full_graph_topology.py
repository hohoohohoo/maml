#!/usr/bin/env python
"""
Pre-compute cell topology for full_graph mode
Full graph mode에서는 cell의 topology가 고정되어 있으므로,
CDL/SPI에서 logic cell들의 graph structure를 미리 계산하여 캐싱합니다.

Supports both:
- ASAP7 CDL files (.cdl)
- TSMC SPI files (.spi)

사용법:
    # ASAP7 CDL
    python precompute_cell_topology.py --cdl_path <path_to_cdl> --output <cache_file.pth>

    # TSMC SPI
    python precompute_cell_topology.py --spi_path <path_to_spi> --output <cache_file.pth>
"""

import torch
import pickle
import argparse
from pathlib import Path
import sys

sys.path.append('/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn')

from cdl_loader import CDLLoader


def create_full_graph_edges_asap7(cell, all_nodes):
    """
    ASAP7 CDL cell에 대한 full graph edges 생성

    전체 cell에 대한 complete adjacency matrix 생성 (baseline용)
    Intermediate node 없이 transistor 간 직접 연결만 표현
    Gate, Source, Drain 연결을 모두 동일한 edge로 처리

    Edge attributes:
    - [1,0,0]: All connections (gate/source/drain 구분 없이 동일)

    Args:
        cell: CDLLoader의 SpiceCell (transistors 정보 포함)
        all_nodes: 전체 노드 리스트

    Returns:
        edges: edge 리스트 [[src, dst], ...]
        edge_attrs: edge attribute 리스트 [[1,0,0], ...]
    """
    edges = []
    edge_attrs = []

    # Node index mapping
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    print(f"   🔗 Creating full graph edges (ASAP7 - no intermediate nodes):")

    # Net을 통한 transistor 간 연결 맵 생성
    net_connections = {}

    # 모든 transistor의 연결 정보 수집
    for trans in cell.transistors:
        trans_name = trans.name

        # Transistor가 node list에 있는 경우만 처리
        if trans_name not in node_to_idx:
            continue

        # Source, Drain, Gate 연결 수집
        for terminal_type, terminal_node in [('source', trans.source), ('drain', trans.drain), ('gate', trans.gate)]:
            # Terminal이 node list에 있으면 직접 연결
            if terminal_node in node_to_idx:
                trans_idx = node_to_idx[trans_name]
                terminal_idx = node_to_idx[terminal_node]

                # Bidirectional edge (무방향 그래프)
                edges.append([trans_idx, terminal_idx])
                edge_attrs.append([1.0, 0.0, 0.0])
                edges.append([terminal_idx, trans_idx])
                edge_attrs.append([1.0, 0.0, 0.0])

                # Input/output node를 공유하는 transistor들도 연결하기 위해 추가
                if terminal_node not in net_connections:
                    net_connections[terminal_node] = []
                net_connections[terminal_node].append(trans_name)

            else:
                # Terminal이 intermediate net이면 나중에 처리
                if terminal_node not in net_connections:
                    net_connections[terminal_node] = []
                net_connections[terminal_node].append(trans_name)

    # Intermediate nets를 통한 transistor 간 직접 연결
    for net, connected_trans in net_connections.items():
        print(f"      Via {net}: connecting {len(connected_trans)} transistors")
        # 같은 net에 연결된 모든 transistor 쌍을 연결
        for i in range(len(connected_trans)):
            for j in range(i + 1, len(connected_trans)):
                trans1 = connected_trans[i]
                trans2 = connected_trans[j]

                if trans1 in node_to_idx and trans2 in node_to_idx:
                    idx1 = node_to_idx[trans1]
                    idx2 = node_to_idx[trans2]

                    # Bidirectional edge
                    edges.append([idx1, idx2])
                    edge_attrs.append([1.0, 0.0, 0.0])
                    edges.append([idx2, idx1])
                    edge_attrs.append([1.0, 0.0, 0.0])

    print(f"   📊 Total edges: {len(edges)} (all connections treated equally)")

    return edges, edge_attrs


def precompute_cell_topology_asap7(cdl_path: str, output_path: str, logic_keywords=None):
    """
    CDL 파일에서 logic cell들의 topology를 미리 계산하여 저장

    Args:
        cdl_path: CDL 파일 경로
        output_path: 출력 캐시 파일 경로 (.pth)
        logic_keywords: Logic cell을 구분하는 키워드 리스트 (기본값 사용 가능)
    """

    if logic_keywords is None:
        # Default logic keywords (build_gnn_dataset_no_split.py와 동일)
        logic_keywords = [
            'AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR',
            'INV', 'BUF', 'MUX', 'AO', 'OA', 'AOI', 'OAI',
            'MAJ', 'FA', 'HA','MAJI'
        ]

    print("=" * 80)
    print("PRE-COMPUTING CELL TOPOLOGY FOR FULL_GRAPH MODE")
    print("=" * 80)
    print(f"CDL file: {cdl_path}")
    print(f"Output cache: {output_path}")
    print(f"Logic keywords: {logic_keywords}")
    print("=" * 80)

    # Load CDL file
    print("\n📂 Loading CDL file...")
    transformer = CDLLoader(cdl_path)

    print(f"   ✓ Loaded {len(transformer.all_logic_cells)} logic cells")

    # Pre-compute topology for each logic cell
    cell_topology_cache = {}

    for cell_name, spice_cell in transformer.all_logic_cells.items():
        # Filter logic cells only
        is_logic_cell = any(keyword in cell_name for keyword in logic_keywords)

        if not is_logic_cell:
            continue

        print(f"\n🔄 Processing: {cell_name}")

        # Node list (full_graph mode: no intermediate nodes)
        power_nodes = ['VDD', 'VSS']
        transistor_nodes = [trans.name for trans in spice_cell.transistors]

        # Auto-detect output nodes from cell ports (same as transform_sample_MAML_stage_aware.py)
        # Output nodes = all ports that are not power nodes and not inputs
        all_ports = spice_cell.ports

        # First pass: collect all nets to identify potential inputs
        all_nets = set()
        for trans in spice_cell.transistors:
            all_nets.add(trans.gate)
            all_nets.add(trans.source)
            all_nets.add(trans.drain)

        # Identify external inputs and outputs from ports
        # More accurate heuristic for multi-output cells (e.g., Full Adder)
        potential_inputs = []
        potential_outputs = []

        for port in all_ports:
            if port not in power_nodes:
                is_used_as_gate = any(t.gate == port for t in spice_cell.transistors)
                is_used_as_output = any(port in [t.source, t.drain] for t in spice_cell.transistors)

                # Output characteristic: primarily appears as source/drain (driven by transistors)
                # Input characteristic: primarily appears as gate only
                if is_used_as_output:
                    # This port is driven by transistors → likely output
                    potential_outputs.append(port)
                elif is_used_as_gate:
                    # This port is only used as gate → pure input
                    potential_inputs.append(port)

        # Final output nodes (prefer potential_outputs, fallback to non-input ports)
        if potential_outputs:
            output_nodes = potential_outputs
        else:
            # Fallback: ports that are NOT power and NOT inputs
            output_nodes = [port for port in all_ports
                           if port not in power_nodes and port not in potential_inputs]

        # If no outputs detected, fallback to 'Y'
        if not output_nodes:
            output_nodes = ['Y']
            print(f"   ⚠️  No outputs detected, using default: {output_nodes}")

        # External inputs (extract from CDL)
        external_inputs = []
        intermediate_nets = set()

        for trans in spice_cell.transistors:
            # Collect all nets
            for net in [trans.gate, trans.source, trans.drain]:
                if net not in power_nodes and net:
                    # Check if it's an external input or intermediate net
                    # Simple heuristic: if it's a gate of a transistor and appears as source/drain of another, it's intermediate
                    is_gate = any(t.gate == net for t in spice_cell.transistors)
                    is_terminal = any(net in [t.source, t.drain] for t in spice_cell.transistors)

                    # Intermediate net: used as both gate and terminal (source/drain)
                    if is_gate and is_terminal:
                        intermediate_nets.add(net)
                    elif is_gate or (not is_terminal):
                        if net not in external_inputs:
                            external_inputs.append(net)

        # Sort for consistency
        external_inputs = sorted(external_inputs)

        # Full graph mode: all nodes except intermediate nets
        all_nodes = power_nodes + output_nodes + external_inputs + transistor_nodes
        all_nodes = list(dict.fromkeys(all_nodes))  # Remove duplicates

        print(f"   Nodes: {len(all_nodes)} (power={len(power_nodes)}, output={len(output_nodes)}, "
              f"inputs={len(external_inputs)}, transistors={len(transistor_nodes)})")
        print(f"   Intermediate nets (excluded): {sorted(intermediate_nets)}")

        # Create full graph edges
        edges, edge_attrs = create_full_graph_edges_asap7(spice_cell, all_nodes)

        # Convert to tensor format
        edge_index_tensor = torch.tensor(edges, dtype=torch.int64).T  # Transpose for PyG format [2, num_edges]
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)

        # Pre-compute adjacency matrix from edge_index
        num_nodes = len(all_nodes)
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        for i in range(edge_index_tensor.shape[1]):
            src = edge_index_tensor[0][i]
            dst = edge_index_tensor[1][i]
            adjacency_matrix[src][dst] = 1.0

        # Get transistor information for node features (type and width)
        # IMPORTANT: NMOS = 1.0, PMOS = -1.0 (same as transform_sample_MAML_stage_aware.py)
        transistor_info = {}
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

        # Store topology cache
        cell_topology_cache[cell_name] = {
            'all_nodes': all_nodes,
            'external_inputs': external_inputs,
            'power_nodes': power_nodes,
            'output_nodes': output_nodes,
            'transistor_nodes': transistor_nodes,
            'intermediate_nets': list(intermediate_nets),
            'edge_index': edge_index_tensor,  # [2, num_edges]
            'edge_attr': edge_attr_tensor,    # [num_edges, 3]
            'adjacency_matrix': adjacency_matrix,  # [num_nodes, num_nodes] - PRE-COMPUTED
            'transistor_info': transistor_info,
            'num_nodes': len(all_nodes),
            'num_edges': len(edges)
        }

        print(f"   ✓ Cached: {len(all_nodes)} nodes, {len(edges)} edges, adjacency_matrix: {adjacency_matrix.shape}")

    # Save cache
    print(f"\n💾 Saving topology cache...")
    print(f"   Total cells cached: {len(cell_topology_cache)}")

    torch.save(cell_topology_cache, output_path)

    print(f"   ✓ Saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for cell_name, cache in cell_topology_cache.items():
        print(f"  {cell_name:30s}: {cache['num_nodes']:3d} nodes, {cache['num_edges']:4d} edges")

    print("=" * 80)
    print("✅ Pre-computation complete!")
    print("=" * 80)

    return cell_topology_cache


def precompute_cell_topology_tsmc(spi_path: str, output_path: str, logic_keywords=None):
    """
    TSMC SPI 파일에서 logic cell들의 topology를 미리 계산하여 저장
    결과 형식은 CDL 버전과 동일

    Args:
        spi_path: TSMC SPI 파일 경로
        output_path: 출력 캐시 파일 경로 (.pth)
        logic_keywords: Logic cell을 구분하는 키워드 리스트 (기본값 사용 가능)
    """
    from spi_loader import SPILoader

    if logic_keywords is None:
        # TSMC cell naming conventions
        logic_keywords = [
            'AN', 'ND', 'OR', 'NR', 'XOR', 'XNOR',
            'INV', 'BUF', 'MUX', 'AO', 'OA', 'AOI', 'OAI',
            'MAJ', 'FA', 'HA', 'DEL', 'CKND', 'CKAN', 'CKNR' , 'DFCNQD' , 'SDFSNQD'
        ]

    print("=" * 80)
    print("PRE-COMPUTING CELL TOPOLOGY FOR FULL_GRAPH MODE (TSMC SPI)")
    print("=" * 80)
    print(f"SPI file: {spi_path}")
    print(f"Output cache: {output_path}")
    print(f"Logic keywords: {logic_keywords}")
    print("=" * 80)

    # Load SPI file
    print("\n📂 Loading TSMC SPI file...")
    loader = SPILoader(spi_path, verbose=False)

    print(f"   ✓ Loaded {len(loader.all_logic_cells)} logic cells")

    # Pre-compute topology for each logic cell
    cell_topology_cache = {}

    for cell_name, spi_cell in loader.all_logic_cells.items():
        # Filter logic cells only
        is_logic_cell = any(keyword in cell_name.upper() for keyword in logic_keywords)

        if not is_logic_cell:
            continue

        print(f"\n🔄 Processing: {cell_name}")

        # Get transistor connectivity (resolves resistance-based connections)
        connectivity = loader.get_transistor_connectivity(cell_name)

        # Node list (full_graph mode)
        power_nodes = ['VDD', 'VSS']

        # Transistor nodes: use original names (XM1, XM2, etc.)
        # No need to normalize - apply_topology_to_sample uses transistor_nodes list directly
        transistor_nodes = [trans.name for trans in spi_cell.transistors]
        transistor_name_map = {name: name for name in transistor_nodes}  # Identity mapping

        # Get ports (filter out power)
        all_ports = spi_cell.ports

        # Identify external inputs and outputs from connectivity
        potential_inputs = []
        potential_outputs = []

        for port in all_ports:
            if port in power_nodes:
                continue

            # Check if port is used as gate (input) or source/drain (output)
            is_used_as_gate = False
            is_used_as_output = False

            for trans_name, conn_info in connectivity.items():
                # Check gate connections
                for node in conn_info['gate']:
                    if port == node or (port in node if ':' not in node else False):
                        is_used_as_gate = True
                # Check drain/source connections
                for node in conn_info['drain'] | conn_info['source']:
                    if port == node or (port in node if ':' not in node else False):
                        is_used_as_output = True

            if is_used_as_output and not is_used_as_gate:
                potential_outputs.append(port)
            elif is_used_as_gate:
                potential_inputs.append(port)

        # Final output nodes
        if potential_outputs:
            output_nodes = potential_outputs
        else:
            # TSMC often uses 'Z' or 'ZN' for output
            output_nodes = [p for p in all_ports if p in ['Z', 'ZN', 'Y', 'CO', 'S']]
            if not output_nodes:
                output_nodes = ['Z'] if 'Z' in all_ports else ['ZN'] if 'ZN' in all_ports else ['Y']

        # External inputs
        external_inputs = sorted([p for p in all_ports if p not in power_nodes and p not in output_nodes])

        # Intermediate nets (internal nodes not in ports)
        intermediate_nets = set()

        # Full graph mode: all nodes
        all_nodes = power_nodes + output_nodes + external_inputs + transistor_nodes
        all_nodes = list(dict.fromkeys(all_nodes))  # Remove duplicates

        print(f"   Nodes: {len(all_nodes)} (power={len(power_nodes)}, output={len(output_nodes)}, "
              f"inputs={len(external_inputs)}, transistors={len(transistor_nodes)})")
        print(f"   Output nodes: {output_nodes}")
        print(f"   Input nodes: {external_inputs}")

        # Create edges from TSMC SPI connectivity
        edges, edge_attrs = create_full_graph_edges_tsmc(
            spi_cell, connectivity, all_nodes, transistor_name_map
        )

        # Convert to tensor format
        edge_index_tensor = torch.tensor(edges, dtype=torch.int64).T if edges else torch.zeros(2, 0, dtype=torch.int64)
        edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32) if edge_attrs else torch.zeros(0, 3, dtype=torch.float32)

        # Pre-compute adjacency matrix
        num_nodes = len(all_nodes)
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        if edges:
            for i in range(edge_index_tensor.shape[1]):
                src = edge_index_tensor[0][i]
                dst = edge_index_tensor[1][i]
                adjacency_matrix[src][dst] = 1.0

        # Get transistor information (normalized names)
        # IMPORTANT: NMOS = 1.0, PMOS = -1.0 (same as CDL version)
        transistor_info = {}
        for i, trans in enumerate(spi_cell.transistors):
            normalized_name = transistor_name_map[trans.name]
            trans_type = 1.0 if trans.type == 'nmos' else -1.0
            trans_width = round(trans.width / 1000.0, 4) if trans.width else 0.1  # nm to um

            # Get actual connected nodes from connectivity
            conn = connectivity.get(trans.name, {})

            # Find the actual gate/source/drain nodes (filter out internal M:XXX nodes)
            def find_external_node(node_set, default=''):
                for n in node_set:
                    if ':' not in n and not n.startswith('N_'):
                        return n
                return default

            gate_node = find_external_node(conn.get('gate', set()), '')
            source_node = find_external_node(conn.get('source', set()), '')
            drain_node = find_external_node(conn.get('drain', set()), '')

            transistor_info[normalized_name] = {
                'type': trans_type,
                'width': trans_width,
                'gate': gate_node,
                'source': source_node,
                'drain': drain_node
            }

        # Store topology cache
        cell_topology_cache[cell_name] = {
            'all_nodes': all_nodes,
            'external_inputs': external_inputs,
            'power_nodes': power_nodes,
            'output_nodes': output_nodes,
            'transistor_nodes': transistor_nodes,
            'intermediate_nets': list(intermediate_nets),
            'edge_index': edge_index_tensor,
            'edge_attr': edge_attr_tensor,
            'adjacency_matrix': adjacency_matrix,
            'transistor_info': transistor_info,
            'num_nodes': len(all_nodes),
            'num_edges': len(edges)
        }

        print(f"   ✓ Cached: {len(all_nodes)} nodes, {len(edges)} edges, adjacency_matrix: {adjacency_matrix.shape}")

    # Save cache
    print(f"\n💾 Saving topology cache...")
    print(f"   Total cells cached: {len(cell_topology_cache)}")

    torch.save(cell_topology_cache, output_path)

    print(f"   ✓ Saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for cell_name, cache in cell_topology_cache.items():
        print(f"  {cell_name:30s}: {cache['num_nodes']:3d} nodes, {cache['num_edges']:4d} edges")

    print("=" * 80)
    print("✅ Pre-computation complete!")
    print("=" * 80)

    return cell_topology_cache


def create_full_graph_edges_tsmc(spi_cell, connectivity, all_nodes, transistor_name_map):
    """
    TSMC SPI cell에 대한 full graph edges 생성

    TSMC SPI는 저항 기반 연결 정보를 사용하므로,
    connectivity 정보를 활용하여 실제 연결을 추출

    Args:
        spi_cell: SPIParser의 LogicCell
        connectivity: get_transistor_connectivity() 결과
        all_nodes: 모든 노드 리스트
        transistor_name_map: XMx -> MMy 매핑

    Returns:
        edges: [[src, dst], ...]
        edge_attrs: [[1,0,0], ...]
    """
    edges = []
    edge_attrs = []

    # Node index mapping
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    print(f"   🔗 Creating full graph edges (TSMC SPI):")

    # Track connections via intermediate nodes
    net_connections = {}  # net -> [connected transistors/nodes]

    for trans in spi_cell.transistors:
        trans_name = trans.name
        normalized_name = transistor_name_map.get(trans_name)

        if normalized_name not in node_to_idx:
            continue

        trans_idx = node_to_idx[normalized_name]

        # Get connectivity info
        conn = connectivity.get(trans_name, {})

        # Process each terminal type
        for terminal_type in ['gate', 'source', 'drain']:
            connected_nodes = conn.get(terminal_type, set())

            for node in connected_nodes:
                # Skip internal M:XXX nodes
                if ':' in node:
                    continue

                # Skip intermediate nodes (N_xx)
                if node.startswith('N_'):
                    # Track for transistor-transistor connections
                    if node not in net_connections:
                        net_connections[node] = []
                    net_connections[node].append(normalized_name)
                    continue

                # Direct connection to external node
                if node in node_to_idx:
                    terminal_idx = node_to_idx[node]

                    # Bidirectional edge
                    edges.append([trans_idx, terminal_idx])
                    edge_attrs.append([1.0, 0.0, 0.0])
                    edges.append([terminal_idx, trans_idx])
                    edge_attrs.append([1.0, 0.0, 0.0])

    # Connect transistors via intermediate nets (N_xx nodes)
    for net, connected_trans in net_connections.items():
        if len(connected_trans) >= 2:
            print(f"      Via {net}: connecting {connected_trans}")
            for i in range(len(connected_trans)):
                for j in range(i + 1, len(connected_trans)):
                    trans1 = connected_trans[i]
                    trans2 = connected_trans[j]

                    if trans1 in node_to_idx and trans2 in node_to_idx:
                        idx1 = node_to_idx[trans1]
                        idx2 = node_to_idx[trans2]

                        # Bidirectional edge
                        edges.append([idx1, idx2])
                        edge_attrs.append([1.0, 0.0, 0.0])
                        edges.append([idx2, idx1])
                        edge_attrs.append([1.0, 0.0, 0.0])

    # Remove duplicate edges
    unique_edges = []
    unique_attrs = []
    seen = set()
    for edge, attr in zip(edges, edge_attrs):
        key = (edge[0], edge[1])
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)
            unique_attrs.append(attr)

    print(f"      📊 Total edges: {len(unique_edges)} (deduplicated from {len(edges)})")

    return unique_edges, unique_attrs


def load_cell_topology_cache(cache_path: str):
    """
    Load pre-computed cell topology cache

    Args:
        cache_path: Path to cache file (.pth)

    Returns:
        dict: Cell topology cache
    """
    return torch.load(cache_path)


def apply_topology_to_sample(topology_cache, cell_name: str, voltage: float,
                             input_slew: float, output_load: float, output_value: float,
                             input_port_names: list = None):
    """
    Apply voltage/slew/load to pre-computed topology to create a graph sample

    Args:
        topology_cache: Pre-computed topology cache
        cell_name: Cell name
        voltage: Voltage value
        input_slew: Input slew
        output_load: Output load
        output_value: Output timing value (delay or transition)
        input_port_names: List of input port names (optional, for validation)

    Returns:
        dict: Graph sample ready for GNN training
    """

    if cell_name not in topology_cache:
        raise ValueError(f"Cell {cell_name} not found in topology cache")

    cache = topology_cache[cell_name]

    # Validate input ports if provided
    if input_port_names is not None:
        expected_inputs = set(cache['external_inputs'])
        provided_inputs = set(input_port_names)
        if expected_inputs != provided_inputs:
            print(f"   ⚠️ Warning: Input port mismatch for {cell_name}")
            print(f"      Expected: {expected_inputs}")
            print(f"      Provided: {provided_inputs}")

    # Create node features using cached topology + runtime parameters
    node_features = []
    circuit_nodes = []
    transistor_node_list = []

    all_nodes = cache['all_nodes']
    transistor_info = cache['transistor_info']
    external_inputs = cache['external_inputs']
    output_nodes = cache['output_nodes']  # Get output nodes from cache
    transistor_nodes = cache.get('transistor_nodes', [])  # Get transistor node list from cache

    for node in all_nodes:
        # Check if node is a transistor (use transistor_nodes list for robustness)
        is_transistor = node in transistor_nodes or node in transistor_info

        if is_transistor:
            if node in transistor_info:
                info = transistor_info[node]
                # Transistor: [0, 0, trans_type, width, voltage, 0, 0]
                # Match transform_sample_MAML_stage_aware.py line 218
                node_features.append([0.0, 0.0, info['type'], info['width'], voltage, 0.0, 0.0])
                transistor_node_list.append(node)
            else:
                # Fallback (should not happen)
                node_features.append([0.0, 0.0, -1.0, 0.1, voltage, 0.0, 0.0])
                transistor_node_list.append(node)

        else:  # Circuit node
            circuit_nodes.append(node)

            if node in ['VDD', 'VSS']:
                # Power rails: [1, 0, 0, 0, voltage, 0, 0]
                # Match transform_sample_MAML_stage_aware.py line 230
                node_features.append([1.0, 0.0, 0.0, 0.0, voltage, 0.0, 0.0])
            elif node in output_nodes:
                # Output port: [0, 1, 0, 0, voltage, 0, output_load]
                # Match transform_sample_MAML_stage_aware.py line 234
                # Handles standard Y output and non-standard outputs like CON, SN (FA/HA cells)
                node_features.append([0.0, 1.0, 0.0, 0.0, voltage, 0.0, output_load])
            elif node in external_inputs:
                # Input port: [0, 1, 0, 0, voltage, input_slew, 0]
                # Match transform_sample_MAML_stage_aware.py line 237
                node_features.append([0.0, 1.0, 0.0, 0.0, voltage, input_slew, 0.0])
            else:
                # Intermediate gate node (should not happen in full_graph mode normally)
                # 해당 gate로 제어되는 모든 transistor의 width 합을 사용
                gate_width_sum = 0.0
                # Find all transistors that use this node as gate
                for trans_name, trans_info in transistor_info.items():
                    if trans_info['gate'] == node:
                        gate_width_sum += trans_info['width']

                # Intermediate node: [0, 1, 0, gate_width_sum, voltage, 0, 0]
                node_features.append([0.0, 1.0, 0.0, gate_width_sum, voltage, 0.0, 0.0])

    # Convert to tensors
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)


    # Create graph sample
    graph_sample = {
        # 'cell': cell_name,
        # 'voltage': voltage,
        # 'output': output_value,
        'all_nodes': all_nodes,  # IMPORTANT: Required for process parameter assignment
        # 'circuit_nodes': circuit_nodes,
        # 'transistor_nodes': transistor_node_list,
        'node_features': node_features_tensor,
        # 'edge_index': edge_index_tensor,
        # 'edge_attr': edge_attr_tensor,
        #'adjacency_matrix': adjacency_matrix,
        # 'total_node_count': num_nodes,
        # 'input_slew': input_slew,
        # 'output_load': output_load,
        # 'graph_mode': 'full_graph'
    }

    return graph_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute cell topology for full_graph mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ASAP7 CDL file
  python precompute_cell_topology.py --cdl_path /path/to/asap7.cdl --output cache_asap7.pth

  # TSMC SPI file
  python precompute_cell_topology.py --spi_path /path/to/tsmc.spi --output cache_tsmc.pth

Note: Specify either --cdl_path (ASAP7) or --spi_path (TSMC), not both.
"""
    )
    parser.add_argument("--cdl_path", type=str, default=None,
                       help="Path to ASAP7 CDL file")
    parser.add_argument("--spi_path", type=str, default=None,
                       help="Path to TSMC SPI file")
    parser.add_argument("--output", type=str,
                       default=None,
                       help="Output cache file path")
    parser.add_argument("--logic_keywords", type=str, nargs='+',
                       default=None,
                       help="Logic cell keywords (default: AND, NAND, OR, etc.)")

    args = parser.parse_args()

    # Determine which file type to process
    if args.spi_path:
        # TSMC SPI file
        if args.output is None:
            args.output = "/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/cell_topology_cache_tsmc.pth"

        print("🔧 Processing TSMC SPI file...")
        precompute_cell_topology_tsmc(args.spi_path, args.output, args.logic_keywords)

    elif args.cdl_path:
        # ASAP7 CDL file
        if args.output is None:
            args.output = "/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/cell_topology_cache_asap7.pth"

        print("🔧 Processing ASAP7 CDL file...")
        precompute_cell_topology_asap7(args.cdl_path, args.output, args.logic_keywords)

    else:
        # Default: ASAP7 CDL
        default_cdl = "/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files/asap7sc7p5t_28_L.cdl"
        default_output = "/home/tkdgn2907/Deepsets_test/MAML/Projects/data_processing/gnn/topology_cache/cell_topology_cache_asap7.pth"

        print("⚠️  No input file specified. Using default ASAP7 CDL...")
        print(f"   CDL: {default_cdl}")
        precompute_cell_topology_asap7(default_cdl, default_output, args.logic_keywords)
