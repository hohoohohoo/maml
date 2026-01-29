#!/usr/bin/env python

"""
TSMC Stage-Aware Path Extractor
Stage-aware current path extraction for TSMC SPI format.

Key differences from ASAP7 version:
- Uses spi_parser.py instead of cdl_loader.py
- Transistor names: XMx (not MMx)
- Type: nch_mac/pch_mac (normalized to nmos/pmos by parser)
- Connections resolved via resistors (handled by spi_parser)
"""

from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
import re
from spi_parser import SPIParser


@dataclass
class StageInfo:
    """Stage information class (legacy 2-stage)"""
    stage_type: str  # "one_stage" or "two_stage"
    intermediate_gates: List[str]  # Intermediate gate nodes
    stage1_paths: List[Tuple[str, str]]  # (source, destination) for first stage
    stage2_paths: List[Tuple[str, str]]  # (source, destination) for second stage
    stage1_transistors: List[str]  # First stage controlled transistors
    stage2_transistors: List[str]  # Second stage controlled transistors


@dataclass
class StageData:
    """Single stage data"""
    stage_num: int  # 1, 2, 3, ... (1 = closest to external input)
    mos_type: str  # 'pmos' or 'nmos'
    power_node: str  # 'VDD' or 'VSS'
    target_nodes: List[str]  # output or intermediate nodes this stage drives
    paths: List[Tuple[str, str]]  # edge list
    transistors: List[str]  # transistor names in this stage
    gate_nodes: List[str]  # gate nodes controlling this stage


@dataclass
class MultiStageInfo:
    """Multi-stage information for complex cells (XOR, XNOR, etc.)"""
    num_stages: int
    stages: List[StageData]  # ordered from stage 1 (input side) to stage N (output side)
    all_intermediate_nodes: List[str]  # all intermediate nodes between stages


class TSMCStageAwareExtractor:
    """
    Stage-aware path extractor for TSMC SPI format.

    - Rise transition: VDD (pull-up) paths only
    - Fall transition: VSS (pull-down) paths only
    """

    def __init__(self, spi_file_path: str):
        self.spi_file_path = spi_file_path
        self.parser = SPIParser(spi_file_path)
        # Cache for resolved transistor connections
        self._trans_conn_cache = {}
        # Equivalence map for collapsing parasitic nodes (VDD:1 -> VDD, etc.)
        self._equiv_map = {}

    def _build_equivalence_map(self, cell):
        """
        Build equivalence map for parasitic RC nodes.
        Nodes like VDD:1, VDD:2, VDD:3 are collapsed to VDD.
        Nodes like ZN:1, ZN:2 are collapsed to ZN.
        Nodes like N_2:1, N_2:2 are collapsed to N_2.

        This prevents exponential path explosion in DFS due to parasitic resistor networks.
        """
        self._equiv_map = {}

        # Get all nodes from connections
        all_nodes = set(cell.connections.keys()) if hasattr(cell, 'connections') else set()

        # Also check for nodes in connection values
        for connected_list in cell.connections.values():
            all_nodes.update(connected_list)

        for node in all_nodes:
            if ':' in node:
                # Split by ':' and check if it's a numbered variant
                parts = node.split(':')
                base_name = parts[0]
                suffix = parts[1] if len(parts) > 1 else ''

                # Check if suffix is a number (e.g., VDD:1, VDD:2, ZN:1, N_2:1)
                if suffix.isdigit():
                    # Power nodes: VDD:x -> VDD, VSS:x -> VSS
                    if base_name in ['VDD', 'VSS']:
                        self._equiv_map[node] = base_name
                    # Output/port nodes: ZN:x -> ZN (if ZN is a port)
                    elif base_name in cell.ports:
                        self._equiv_map[node] = base_name
                    # Internal net nodes: N_x:y -> N_x (parasitic resistor network)
                    elif base_name.startswith('N_'):
                        self._equiv_map[node] = base_name

        if self._equiv_map:
            print(f"      [Equivalence] Collapsed {len(self._equiv_map)} parasitic nodes")

    def _apply_equivalence(self, node: str) -> str:
        """Apply equivalence mapping to a node name."""
        return self._equiv_map.get(node, node)

    def _get_cached_transistor_connections(self, cell, trans):
        """Get cached transistor connections for performance."""
        cache_key = (cell.name if hasattr(cell, 'name') else id(cell), trans.name)
        if cache_key not in self._trans_conn_cache:
            self._trans_conn_cache[cache_key] = self._get_transistor_connections(cell, trans)
        return self._trans_conn_cache[cache_key]

    def _find_stage2_transistors(self, cell, power_node: str, output_nodes: set, mos_type: str) -> set:
        """
        Find transistors that are in the Stage 2 current path (power -> output).
        Uses DFS with shared visited set to avoid exponential exploration.
        Connection-aware matching: N_9와 N_9:1 같이 연결된 노드도 매칭
        """
        stage2_trans = set()
        visited = set()  # Shared visited set

        def is_connected_to_node(node1: str, node2: str) -> bool:
            """Check if two nodes are the same or connected via resistor map"""
            if node1 == node2:
                return True
            return self._is_connected_via_map(cell, node1, {node2}) is not None

        def dfs(current_node: str) -> bool:
            """
            Returns True if this node leads to output (directly or through transistors).
            Uses shared visited set to prevent re-exploration.
            """
            if current_node in visited:
                return False

            # Check if reached output (direct or via multi-hop connections map)
            if current_node in output_nodes:
                return True
            matched = self._is_connected_via_map(cell, current_node, output_nodes)
            if matched is not None:
                return True

            visited.add(current_node)

            # Track if ANY branch leads to output
            any_path_found = False

            # Trace through transistors of the specified type
            for trans in cell.transistors:
                if trans.type != mos_type:
                    continue

                conn = self._get_cached_transistor_connections(cell, trans)

                next_node = None
                if is_connected_to_node(conn['source'], current_node):
                    next_node = conn['drain']
                elif is_connected_to_node(conn['drain'], current_node):
                    next_node = conn['source']

                if next_node and next_node not in visited:
                    if dfs(next_node):
                        stage2_trans.add(trans.name)
                        any_path_found = True

            return any_path_found

        dfs(power_node)
        return stage2_trans

    def _is_connected_via_map(self, cell, node: str, target_nodes: set) -> Optional[str]:
        """
        Check if node is connected to any target_node via connections map (multi-hop BFS).
        Returns the matched target_node if connected, None otherwise.

        Applies equivalence mapping to collapse parasitic nodes (VDD:1 -> VDD, etc.)
        """
        # Apply equivalence to input node
        node_eq = self._apply_equivalence(node)

        # Also build equivalence-mapped target set for faster lookup
        target_nodes_eq = {self._apply_equivalence(t) for t in target_nodes}

        if node_eq in target_nodes_eq:
            return node_eq

        if not hasattr(cell, 'connections'):
            return None

        # BFS to find connection through multiple hops
        visited = set()
        queue = [node]

        while queue:
            current = queue.pop(0)
            current_eq = self._apply_equivalence(current)

            if current_eq in visited:
                continue
            visited.add(current_eq)

            if current_eq in target_nodes_eq:
                return current_eq

            # Search from both original and equivalence-mapped node
            for search_node in [current, current_eq]:
                if search_node in cell.connections:
                    for connected in cell.connections[search_node]:
                        connected_eq = self._apply_equivalence(connected)
                        if connected_eq not in visited:
                            queue.append(connected)

        return None

    def trace_pmos_paths_tsmc(self, cell, start_node: str, end_nodes: set) -> List[List[str]]:
        """
        PMOS transistor만을 따라 pull-up path 추적 (DFS)
        end_nodes에 직접 도달하거나 connections map을 통해 연결되어 있으면 path 완료
        Connection-aware matching: N_9와 N_9:1 같이 연결된 노드도 매칭
        """
        paths = []

        def is_connected_to_node(node1: str, node2: str) -> bool:
            """Check if two nodes are the same or connected via resistor map"""
            if node1 == node2:
                return True
            return self._is_connected_via_map(cell, node1, {node2}) is not None

        def dfs_pmos(current_node: str, path: List[str], visited: set):
            if current_node in visited:
                return

            # Check if current_node is connected to any end_node (direct or via connections map)
            matched_end = self._is_connected_via_map(cell, current_node, end_nodes)
            if matched_end:
                paths.append(path + [matched_end])
                return

            visited.add(current_node)

            # PMOS transistor만 탐색
            for trans in cell.transistors:
                if trans.type != 'pmos':
                    continue

                conn = self._get_cached_transistor_connections(cell, trans)

                next_node = None
                # Connection-aware matching (N_9 == N_9:1 if connected via resistor)
                if is_connected_to_node(conn['source'], current_node):
                    next_node = conn['drain']
                elif is_connected_to_node(conn['drain'], current_node):
                    next_node = conn['source']

                if next_node and next_node not in visited:
                    dfs_pmos(next_node, path + [current_node, trans.name], visited.copy())

        dfs_pmos(start_node, [], set())
        return paths

    def trace_nmos_paths_tsmc(self, cell, start_node: str, end_nodes: set) -> List[List[str]]:
        """
        NMOS transistor만을 따라 pull-down path 추적 (DFS)
        end_nodes에 직접 도달하거나 connections map을 통해 연결되어 있으면 path 완료
        Connection-aware matching: N_9와 N_9:1 같이 연결된 노드도 매칭
        """
        paths = []

        def is_connected_to_node(node1: str, node2: str) -> bool:
            """Check if two nodes are the same or connected via resistor map"""
            if node1 == node2:
                return True
            return self._is_connected_via_map(cell, node1, {node2}) is not None

        def dfs_nmos(current_node: str, path: List[str], visited: set):
            if current_node in visited:
                return

            # Check if current_node is connected to any end_node (direct or via connections map)
            matched_end = self._is_connected_via_map(cell, current_node, end_nodes)
            if matched_end:
                paths.append(path + [matched_end])
                return

            visited.add(current_node)

            # NMOS transistor만 탐색
            for trans in cell.transistors:
                if trans.type != 'nmos':
                    continue

                conn = self._get_cached_transistor_connections(cell, trans)

                next_node = None
                # Connection-aware matching (N_9 == N_9:1 if connected via resistor)
                if is_connected_to_node(conn['source'], current_node):
                    next_node = conn['drain']
                elif is_connected_to_node(conn['drain'], current_node):
                    next_node = conn['source']

                if next_node and next_node not in visited:
                    dfs_nmos(next_node, path + [current_node, trans.name], visited.copy())

        dfs_nmos(start_node, [], set())
        return paths

    def _paths_to_edges_tsmc(self, paths: List[List[str]], stage_name: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Path list를 edge list로 변환 (ASAP7 방식과 동일)"""
        edges = []
        transistors = []

        for path in paths:
            print(f"     {stage_name} path: {' -> '.join(path)}")

            # Path를 연속된 edge로 변환
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                edges.append((src, dst))

                # Transistor 이름 수집 (XM으로 시작)
                if src.startswith('XM') and src not in transistors:
                    transistors.append(src)
                if dst.startswith('XM') and dst not in transistors:
                    transistors.append(dst)

        return edges, transistors

    def _resolve_node_connection(self, cell, terminal):
        """
        Resolve TSMC terminal (e.g., M1:DRN) to actual connected port/net.

        TSMC SPI uses M1:DRN, M2:SRC notation for transistor terminals.
        We need to find what external port/net this terminal connects to
        using the resistor-based connection map.

        Args:
            cell: LogicCell from spi_parser
            terminal: Terminal name (e.g., "M1:DRN")

        Returns:
            str: The resolved port/net name (e.g., "Z", "VDD", "N_14")
        """
        if terminal not in cell.connections:
            return terminal

        # Find connected nodes via BFS
        visited = set()
        queue = [terminal]
        found_intermediate_net = None  # Track intermediate net if found

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            # Check if this is a port (VDD, VSS, Z, A1, A2, etc.)
            if node in cell.ports or node in ['VDD', 'VSS']:
                return node

            # Check if this is a simple net (N_xx) - intermediate net
            if node.startswith('N_'):
                # Save the first intermediate net found
                if found_intermediate_net is None:
                    found_intermediate_net = node
                # Continue searching in case it connects to a port
                if node in cell.connections:
                    for neighbor in cell.connections[node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                continue

            # Continue BFS through connections
            if node in cell.connections:
                for neighbor in cell.connections[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        # If no port found, return intermediate net if found, else terminal itself
        if found_intermediate_net:
            result = found_intermediate_net
        else:
            result = terminal

        # Apply equivalence mapping (VDD:1 -> VDD, ZN:1 -> ZN, etc.)
        return self._apply_equivalence(result)

    def _get_transistor_connections(self, cell, trans):
        """
        Get resolved connections for a transistor.

        Args:
            cell: LogicCell from spi_parser
            trans: Transistor namedtuple

        Returns:
            dict: {'drain': str, 'gate': str, 'source': str}
        """
        # Get MOS prefix for terminal resolution (XM1 -> M1)
        mos_name = trans.name.replace('X', '')

        # Find connected nodes for each terminal
        drain_term = f"{mos_name}:DRN"
        gate_term = f"{mos_name}:GATE"
        source_term = f"{mos_name}:SRC"

        # Resolve to actual ports/nets
        drain = self._resolve_node_connection(cell, drain_term)
        gate = self._resolve_node_connection(cell, gate_term)
        source = self._resolve_node_connection(cell, source_term)

        return {
            'drain': drain,
            'gate': gate,
            'source': source
        }

    def _is_connected_to_intermediate_gate(self, cell, trans, intermediate_gate):
        """
        Check if transistor's source or drain is connected to intermediate_gate.
        Connection can be direct or via N_xx intermediate nets.

        Args:
            cell: LogicCell from spi_parser
            trans: Transistor
            intermediate_gate: Intermediate gate terminal (e.g., 'M6:GATE')

        Returns:
            bool: True if connected
        """
        mos_name = trans.name.replace('X', '')
        drain_term = f"{mos_name}:DRN"
        source_term = f"{mos_name}:SRC"

        # Check both drain and source terminals
        for terminal in [drain_term, source_term]:
            if terminal not in cell.connections:
                continue

            # BFS to find if terminal connects to intermediate_gate
            visited = set()
            queue = [terminal]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)

                # Found intermediate_gate!
                if node == intermediate_gate:
                    return True

                # Only follow N_xx nets (intermediate internal nets)
                if node.startswith('N_'):
                    if node in cell.connections:
                        for neighbor in cell.connections[node]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                # Also follow from the starting terminal
                elif node == terminal:
                    if node in cell.connections:
                        for neighbor in cell.connections[node]:
                            if neighbor not in visited:
                                queue.append(neighbor)

        return False

    def _check_terminal_connection(self, cell, terminal, target_node):
        """
        Check if a specific terminal is connected to target_node via N_xx nets.

        Args:
            cell: LogicCell from spi_parser
            terminal: Terminal name (e.g., 'M1:SRC')
            target_node: Target node to find (e.g., 'M6:GATE')

        Returns:
            bool: True if connected
        """
        if terminal not in cell.connections:
            return False

        visited = set()
        queue = [terminal]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node == target_node:
                return True

            # Only follow N_xx nets
            if node.startswith('N_'):
                if node in cell.connections:
                    for neighbor in cell.connections[node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            elif node == terminal:
                if node in cell.connections:
                    for neighbor in cell.connections[node]:
                        if neighbor not in visited:
                            queue.append(neighbor)

        return False

    def classify_stage_structure(self, cell_name: str, external_inputs: List[str],
                                  delay_type: str = "rise_transition",
                                  output_nodes: Optional[List[str]] = None) -> StageInfo:
        """
        Classify cell as one-stage or two-stage and extract path information.

        Args:
            cell_name: Name of cell in cache
            external_inputs: List of input port names
            delay_type: 'rise_transition' or 'fall_transition'
            output_nodes: Output node list (e.g., ['Z'] or ['CON', 'SN'])

        Returns:
            StageInfo with path information
        """
        if cell_name not in self.parser.logic_cells:
            raise ValueError(f"Cell {cell_name} not found in SPI file")

        cell = self.parser.logic_cells[cell_name]

        # Clear cache for fresh resolution
        self._trans_conn_cache = {}

        # Build equivalence map for parasitic node collapsing (VDD:1 -> VDD, etc.)
        self._build_equivalence_map(cell)

        # Default output nodes
        if output_nodes is None:
            output_nodes = ['Z']

        # 1. Determine Stage 2 MOS type and power based on delay_type
        # Rise: Stage 2 = PMOS (VDD -> output)
        # Fall: Stage 2 = NMOS (VSS -> output)
        if 'rise' in delay_type:
            stage2_mos_type = 'pmos'
            stage2_power = 'VDD'
        else:
            stage2_mos_type = 'nmos'
            stage2_power = 'VSS'

        power_nets = {'VDD', 'VSS'}
        output_nets = set(output_nodes)
        external_nets = set(external_inputs)

        # 2. First, find Stage 2 transistors (those in the path from power to output)
        # Use DFS to find which transistors are actually in Stage 2 current path
        stage2_transistor_names = self._find_stage2_transistors(cell, stage2_power, output_nets, stage2_mos_type)

        print(f"   Delay type: {delay_type}")
        print(f"   Stage 2 MOS type: {stage2_mos_type}")
        print(f"   Stage 2 transistors (in {stage2_power} -> output path): {stage2_transistor_names}")

        # 3. Find intermediate gates from Stage 2 transistors only
        # Intermediate node = Stage 2 transistor의 gate 중 external input이 아닌 것
        # Gate terminal을 실제 연결된 net으로 resolve하여 같은 net 공유시 자동 병합
        intermediate_gates = set()
        for trans in cell.transistors:
            if trans.name not in stage2_transistor_names:
                continue
            gate_terminal = trans.gate
            # Resolve gate terminal to actual connected net (e.g., M5:GATE -> N_5)
            resolved_gate = self._resolve_node_connection(cell, gate_terminal)
            # Gate가 external input이 아니면 intermediate node
            if resolved_gate and resolved_gate not in power_nets and resolved_gate not in external_nets:
                intermediate_gates.add(resolved_gate)

        intermediate_gates = [gate for gate in intermediate_gates if gate and gate.strip()]

        print(f"   Intermediate gates (from Stage 2 transistor gates): {intermediate_gates}")

        # 3. Stage classification
        if intermediate_gates:
            return self._analyze_two_stage_structure(cell, external_inputs, intermediate_gates,
                                                      delay_type, output_nodes)
        else:
            return self._analyze_one_stage_structure(cell, external_inputs, delay_type, output_nodes)

    def _analyze_one_stage_structure(self, cell, external_inputs: List[str],
                                      delay_type: str, output_nodes: List[str]) -> StageInfo:
        """One-stage structure analysis using DFS (ASAP7 방식과 동일)"""

        print(f"   ONE-STAGE structure detected")

        # Determine target power rail based on delay_type
        target_power = 'VDD' if 'rise' in delay_type else 'VSS'
        path_type_name = "pull-up" if target_power == 'VDD' else "pull-down"

        print(f"   Target: {target_power} ({path_type_name} paths)")

        # DFS를 사용하여 complete current path 추적
        output_node_set = set(output_nodes)

        if 'rise' in delay_type:
            # Pull-up: VDD -> output via PMOS
            pmos_paths = self.trace_pmos_paths_tsmc(cell, 'VDD', output_node_set)
            print(f"   Found {len(pmos_paths)} PMOS paths")
            stage1_paths, stage1_transistors = self._paths_to_edges_tsmc(pmos_paths, "One-stage PMOS")
        else:
            # Pull-down: VSS -> output via NMOS
            nmos_paths = self.trace_nmos_paths_tsmc(cell, 'VSS', output_node_set)
            print(f"   Found {len(nmos_paths)} NMOS paths")
            stage1_paths, stage1_transistors = self._paths_to_edges_tsmc(nmos_paths, "One-stage NMOS")

        return StageInfo(
            stage_type="one_stage",
            intermediate_gates=[],
            stage1_paths=stage1_paths,
            stage2_paths=[],
            stage1_transistors=stage1_transistors,
            stage2_transistors=[]
        )

    def _analyze_two_stage_structure(self, cell, external_inputs: List[str],
                                      intermediate_gates: List[str], delay_type: str,
                                      output_nodes: List[str]) -> StageInfo:
        """Two-stage structure analysis using DFS (ASAP7 방식과 동일)"""

        print(f"   TWO-STAGE structure detected")
        print(f"   Intermediate gate(s): {intermediate_gates}")

        # Two-stage inverting logic:
        # Rise: First stage (pull-down/NMOS) -> Second stage (pull-up/PMOS)
        # Fall: First stage (pull-up/PMOS) -> Second stage (pull-down/NMOS)
        if 'rise' in delay_type:
            stage1_power = 'VSS'
            stage2_power = 'VDD'
            stage1_type = "pull-down"
            stage2_type = "pull-up"
        else:
            stage1_power = 'VDD'
            stage2_power = 'VSS'
            stage1_type = "pull-up"
            stage2_type = "pull-down"

        print(f"   Two-stage inverting logic:")
        print(f"     Stage 1: {stage1_power} ({stage1_type}) -> intermediate gates")
        print(f"     Stage 2: {stage2_power} ({stage2_type}) -> output")

        # Convert to sets for DFS
        intermediate_gate_set = set(intermediate_gates)
        output_node_set = set(output_nodes)

        # Stage 1: Power -> intermediate gates (DFS)
        print(f"   Stage 1 Analysis ({stage1_power} -> intermediate gates):")
        if 'rise' in delay_type:
            # Rise: Stage 1 uses NMOS (VSS -> intermediate gate)
            stage1_paths_raw = self.trace_nmos_paths_tsmc(cell, stage1_power, intermediate_gate_set)
        else:
            # Fall: Stage 1 uses PMOS (VDD -> intermediate gate)
            stage1_paths_raw = self.trace_pmos_paths_tsmc(cell, stage1_power, intermediate_gate_set)

        print(f"   Found {len(stage1_paths_raw)} Stage 1 paths")
        stage1_paths, stage1_transistors = self._paths_to_edges_tsmc(stage1_paths_raw, f"Stage1 {stage1_type}")

        # Stage 2: Power -> output (DFS)
        print(f"   Stage 2 Analysis ({stage2_power} -> output):")
        if 'rise' in delay_type:
            # Rise: Stage 2 uses PMOS (VDD -> output)
            stage2_paths_raw = self.trace_pmos_paths_tsmc(cell, stage2_power, output_node_set)
        else:
            # Fall: Stage 2 uses NMOS (VSS -> output)
            stage2_paths_raw = self.trace_nmos_paths_tsmc(cell, stage2_power, output_node_set)

        print(f"   Found {len(stage2_paths_raw)} Stage 2 paths")
        stage2_paths, stage2_transistors = self._paths_to_edges_tsmc(stage2_paths_raw, f"Stage2 {stage2_type}")

        return StageInfo(
            stage_type="two_stage",
            intermediate_gates=intermediate_gates,
            stage1_paths=stage1_paths,
            stage2_paths=stage2_paths,
            stage1_transistors=stage1_transistors,
            stage2_transistors=stage2_transistors
        )

    def classify_multi_stage_structure(self, cell_name: str, external_inputs: List[str],
                                        delay_type: str = "rise_transition",
                                        output_nodes: Optional[List[str]] = None,
                                        max_stages: int = 10) -> MultiStageInfo:
        """
        Classify cell with multi-stage support (for XOR, XNOR, complex cells).

        Iteratively finds stages by tracing back from output until all gate nodes
        are external inputs.

        Args:
            cell_name: Name of cell
            external_inputs: List of external input port names
            delay_type: 'rise_transition' or 'fall_transition'
            output_nodes: Output node list
            max_stages: Maximum stages to prevent infinite loops

        Returns:
            MultiStageInfo with all stages
        """
        if cell_name not in self.parser.logic_cells:
            raise ValueError(f"Cell {cell_name} not found in SPI file")

        cell = self.parser.logic_cells[cell_name]
        self._trans_conn_cache = {}

        # Build equivalence map for parasitic node collapsing (VDD:1 -> VDD, etc.)
        self._build_equivalence_map(cell)

        if output_nodes is None:
            output_nodes = ['Z']

        power_nets = {'VDD', 'VSS'}
        external_nets = set(external_inputs)

        # Determine initial MOS type based on delay_type
        # Rise: final stage is PMOS (pull-up), Fall: final stage is NMOS (pull-down)
        if 'rise' in delay_type:
            final_mos_type = 'pmos'
            final_power = 'VDD'
        else:
            final_mos_type = 'nmos'
            final_power = 'VSS'

        stages_reversed = []  # Will build from output backwards, then reverse
        all_intermediate_nodes = []
        visited_target_nodes = set()  # Track visited targets to prevent cycles
        visited_transistors = set()   # Track transistors already assigned to a stage

        current_target_nodes = set(output_nodes)
        current_mos_type = final_mos_type
        current_power = final_power
        stage_count = 0

        print(f"\n{'='*60}")
        print(f"Multi-Stage Analysis: {cell_name} ({delay_type})")
        print(f"{'='*60}")

        while stage_count < max_stages:
            stage_count += 1
            print(f"\n--- Analyzing Stage {stage_count} (from output side) ---")
            print(f"    MOS type: {current_mos_type}, Power: {current_power}")
            print(f"    Target nodes: {current_target_nodes}")

            # 1. Find transistors in path: power -> current_target_nodes
            stage_transistors = self._find_stage_transistors(
                cell, current_power, current_target_nodes, current_mos_type
            )

            # Filter out transistors already used in previous stages
            stage_transistors = stage_transistors - visited_transistors

            if not stage_transistors:
                print(f"    No new transistors found for this stage, stopping.")
                break

            print(f"    Found transistors: {stage_transistors}")
            visited_transistors.update(stage_transistors)

            # 2. Get paths for this stage
            if current_mos_type == 'pmos':
                paths_raw = self.trace_pmos_paths_tsmc(cell, current_power, current_target_nodes)
            else:
                paths_raw = self.trace_nmos_paths_tsmc(cell, current_power, current_target_nodes)

            paths, transistor_list = self._paths_to_edges_tsmc(
                paths_raw, f"Stage-{stage_count} {current_mos_type.upper()}"
            )

            # 3. Find gate nodes of transistors in this stage
            # Use transistor_list from path extraction (more complete than stage_transistors)
            gate_nodes = set()
            transistor_set = set(transistor_list)
            for trans in cell.transistors:
                if trans.name in transistor_set:
                    resolved_gate = self._resolve_node_connection(cell, trans.gate)
                    if resolved_gate and resolved_gate not in power_nets:
                        gate_nodes.add(resolved_gate)

            gate_nodes_list = list(gate_nodes)
            print(f"    Gate nodes: {gate_nodes_list}")

            # 4. Check if all gates are external inputs (termination condition)
            # Also filter out gates that were already visited as target nodes (cycle detection)
            non_external_gates = [g for g in gate_nodes_list
                                  if g not in external_nets and g not in visited_target_nodes]
            all_gates_external = len(non_external_gates) == 0

            print(f"    Non-external gates (new): {non_external_gates}")
            print(f"    All gates external or already visited? {all_gates_external}")

            # Store this stage
            stage_data = StageData(
                stage_num=0,  # Will be assigned after reversing
                mos_type=current_mos_type,
                power_node=current_power,
                target_nodes=list(current_target_nodes),
                paths=paths,
                transistors=transistor_list,
                gate_nodes=gate_nodes_list
            )
            stages_reversed.append(stage_data)

            # Mark current targets as visited
            visited_target_nodes.update(current_target_nodes)

            # 5. Termination check
            if all_gates_external:
                print(f"    All gates are external inputs or already visited. Stage search complete!")
                break

            # 6. Prepare for next stage (go backwards)
            # Non-external gates become the new target nodes
            all_intermediate_nodes.extend(non_external_gates)
            current_target_nodes = set(non_external_gates)

            # Alternate MOS type and power
            if current_mos_type == 'pmos':
                current_mos_type = 'nmos'
                current_power = 'VSS'
            else:
                current_mos_type = 'pmos'
                current_power = 'VDD'

        # Reverse stages so stage 1 is closest to input
        stages_reversed.reverse()
        for i, stage in enumerate(stages_reversed):
            stage.stage_num = i + 1

        print(f"\n{'='*60}")
        print(f"Result: {len(stages_reversed)} stages found")
        for stage in stages_reversed:
            print(f"  Stage {stage.stage_num}: {stage.mos_type.upper()} "
                  f"({stage.power_node} -> {stage.target_nodes})")
            print(f"    Transistors: {stage.transistors}")
            print(f"    Gates: {stage.gate_nodes}")
        print(f"{'='*60}\n")

        return MultiStageInfo(
            num_stages=len(stages_reversed),
            stages=stages_reversed,
            all_intermediate_nodes=list(set(all_intermediate_nodes))
        )

    def _find_stage_transistors(self, cell, power_node: str, target_nodes: set,
                                 mos_type: str) -> set:
        """
        Find all transistors in the current path from power to target_nodes.
        Reuses _find_stage2_transistors logic with connection-aware matching.
        """
        return self._find_stage2_transistors(cell, power_node, target_nodes, mos_type)

    def _analyze_stage2_pull_up_paths_tsmc(self, cell, intermediate_gate, direct_controlled,
                                            output_nodes: List[str], target_power: str):
        """
        Stage 2 pull-up path analysis: VDD -> output paths after intermediate gate
        """
        result = {
            'transistors': [],
            'paths': [],
            'descriptions': []
        }

        target_mos_type = 'pmos'  # Pull-up uses PMOS

        # 1. Transistors directly controlled by intermediate gate (PMOS for pull-up)
        for trans, conn in direct_controlled:
            result['transistors'].append(trans.name)

            if conn['source'] == target_power:
                # VDD -> transistor -> output/net
                result['paths'].append((target_power, trans.name))
                result['paths'].append((trans.name, conn['drain']))
                result['descriptions'].append(
                    f"Stage 2 PMOS (direct): {target_power} -> {trans.name} -> {conn['drain']} (controlled by {intermediate_gate})"
                )
            elif conn['drain'] == target_power:
                # VDD -> transistor -> output/net
                result['paths'].append((target_power, trans.name))
                result['paths'].append((trans.name, conn['source']))
                result['descriptions'].append(
                    f"Stage 2 PMOS (direct): {target_power} -> {trans.name} -> {conn['source']} (controlled by {intermediate_gate})"
                )
            elif conn['drain'] in output_nodes:
                # intermediate_net -> transistor -> output
                # Need to trace from VDD to this transistor's source
                series_path = self._trace_to_power_tsmc(cell, trans, conn['source'], target_power,
                                                         target_mos_type, [trans.name])
                if series_path:
                    # Add series transistors to result (series_path already includes trans.name)
                    for series_trans in series_path:
                        if series_trans not in result['transistors']:
                            result['transistors'].append(series_trans)
                    # Build path: VDD -> series transistors -> output
                    # Note: series_path[-1] is already trans.name
                    result['paths'].append((target_power, series_path[0]))
                    for i in range(len(series_path) - 1):
                        result['paths'].append((series_path[i], series_path[i+1]))
                    result['paths'].append((series_path[-1], conn['drain']))
                    result['descriptions'].append(
                        f"Stage 2 PMOS (series to output): {target_power} -> {' -> '.join(series_path)} -> {conn['drain']} (controlled by {intermediate_gate})"
                    )
                else:
                    # No series path found, just add direct connection
                    result['paths'].append((conn['source'], trans.name))
                    result['paths'].append((trans.name, conn['drain']))
                    result['descriptions'].append(
                        f"Stage 2 PMOS (to output): {conn['source']} -> {trans.name} -> {conn['drain']} (controlled by {intermediate_gate})"
                    )
            elif conn['source'] in output_nodes:
                # intermediate_net -> transistor -> output
                # Need to trace from VDD to this transistor's drain
                series_path = self._trace_to_power_tsmc(cell, trans, conn['drain'], target_power,
                                                         target_mos_type, [trans.name])
                if series_path:
                    # series_path already includes trans.name
                    for series_trans in series_path:
                        if series_trans not in result['transistors']:
                            result['transistors'].append(series_trans)
                    result['paths'].append((target_power, series_path[0]))
                    for i in range(len(series_path) - 1):
                        result['paths'].append((series_path[i], series_path[i+1]))
                    result['paths'].append((series_path[-1], conn['source']))
                    result['descriptions'].append(
                        f"Stage 2 PMOS (series to output): {target_power} -> {' -> '.join(series_path)} -> {conn['source']} (controlled by {intermediate_gate})"
                    )
                else:
                    result['paths'].append((conn['drain'], trans.name))
                    result['paths'].append((trans.name, conn['source']))
                    result['descriptions'].append(
                        f"Stage 2 PMOS (to output): {conn['drain']} -> {trans.name} -> {conn['source']} (controlled by {intermediate_gate})"
                    )
            else:
                # General case: try to trace to power from both ends
                source_path = self._trace_to_power_tsmc(cell, trans, conn['source'], target_power,
                                                          target_mos_type, [trans.name])
                drain_path = self._trace_to_power_tsmc(cell, trans, conn['drain'], target_power,
                                                         target_mos_type, [trans.name])

                if source_path:
                    # source_path already includes trans.name
                    for series_trans in source_path:
                        if series_trans not in result['transistors']:
                            result['transistors'].append(series_trans)
                    result['paths'].append((target_power, source_path[0]))
                    for i in range(len(source_path) - 1):
                        result['paths'].append((source_path[i], source_path[i+1]))
                    result['paths'].append((source_path[-1], conn['drain']))
                    result['descriptions'].append(
                        f"Stage 2 PMOS (series): {target_power} -> {' -> '.join(source_path)} -> {conn['drain']} (controlled by {intermediate_gate})"
                    )
                elif drain_path:
                    # drain_path already includes trans.name
                    for series_trans in drain_path:
                        if series_trans not in result['transistors']:
                            result['transistors'].append(series_trans)
                    result['paths'].append((target_power, drain_path[0]))
                    for i in range(len(drain_path) - 1):
                        result['paths'].append((drain_path[i], drain_path[i+1]))
                    result['paths'].append((drain_path[-1], conn['source']))
                    result['descriptions'].append(
                        f"Stage 2 PMOS (series): {target_power} -> {' -> '.join(drain_path)} -> {conn['source']} (controlled by {intermediate_gate})"
                    )
                else:
                    # Fallback: add both directions without power tracing
                    result['paths'].append((conn['source'], trans.name))
                    result['paths'].append((trans.name, conn['drain']))
                    result['descriptions'].append(
                        f"Stage 2 PMOS: {conn['source']} -> {trans.name} -> {conn['drain']} (controlled by {intermediate_gate})"
                    )

        return result

    def _analyze_stage2_pull_down_paths_tsmc(self, cell, intermediate_gate, direct_controlled,
                                               output_nodes: List[str], target_power: str):
        """
        Stage 2 pull-down path analysis: VSS -> output paths after intermediate gate
        """
        result = {
            'transistors': [],
            'paths': [],
            'descriptions': []
        }

        # 1. Transistors directly controlled by intermediate gate (NMOS for pull-down)
        for trans, conn in direct_controlled:
            result['transistors'].append(trans.name)

            if conn['source'] == target_power:
                # VSS -> transistor -> output/net
                result['paths'].append((target_power, trans.name))
                result['paths'].append((trans.name, conn['drain']))
                result['descriptions'].append(
                    f"Stage 2 NMOS (direct): {target_power} -> {trans.name} -> {conn['drain']} (controlled by {intermediate_gate})"
                )
            elif conn['drain'] == target_power:
                # VSS -> transistor -> output/net
                result['paths'].append((target_power, trans.name))
                result['paths'].append((trans.name, conn['source']))
                result['descriptions'].append(
                    f"Stage 2 NMOS (direct): {target_power} -> {trans.name} -> {conn['source']} (controlled by {intermediate_gate})"
                )
            elif conn['drain'] in output_nodes:
                # intermediate_net -> transistor -> output
                result['paths'].append((conn['source'], trans.name))
                result['paths'].append((trans.name, conn['drain']))
                result['descriptions'].append(
                    f"Stage 2 NMOS (to output): {conn['source']} -> {trans.name} -> {conn['drain']} (controlled by {intermediate_gate})"
                )
            elif conn['source'] in output_nodes:
                # intermediate_net -> transistor -> output
                result['paths'].append((conn['drain'], trans.name))
                result['paths'].append((trans.name, conn['source']))
                result['descriptions'].append(
                    f"Stage 2 NMOS (to output): {conn['drain']} -> {trans.name} -> {conn['source']} (controlled by {intermediate_gate})"
                )
            else:
                # General case: add both directions
                result['paths'].append((conn['source'], trans.name))
                result['paths'].append((trans.name, conn['drain']))
                result['descriptions'].append(
                    f"Stage 2 NMOS: {conn['source']} -> {trans.name} -> {conn['drain']} (controlled by {intermediate_gate})"
                )

        return result

    def _trace_series_transistors_tsmc(self, cell, first_trans, first_net, target_power,
                                         target_mos_type, paths, transistors,
                                         path_type, output_nodes):
        """Trace series-connected transistors."""
        current_net = first_net
        processed_nets = set()
        prev_trans = first_trans

        while current_net not in output_nodes and current_net not in processed_nets:
            processed_nets.add(current_net)
            next_trans_found = None

            for next_trans in cell.transistors:
                if next_trans == prev_trans or next_trans.name in transistors:
                    continue
                if next_trans.type != target_mos_type:
                    continue

                conn = self._get_transistor_connections(cell, next_trans)

                if current_net in [conn['source'], conn['drain']]:
                    next_trans_found = next_trans
                    break

            if next_trans_found:
                conn = self._get_transistor_connections(cell, next_trans_found)
                paths.append((prev_trans.name, next_trans_found.name))
                transistors.append(next_trans_found.name)

                next_other = conn['drain'] if conn['source'] == current_net else conn['source']

                if next_other in output_nodes:
                    paths.append((next_trans_found.name, next_other))
                    print(f"     Series {path_type}: {prev_trans.name} -> {next_trans_found.name} -> {next_other}")
                    break
                else:
                    print(f"     Series {path_type}: {prev_trans.name} -> {next_trans_found.name}")
                    current_net = next_other
                    prev_trans = next_trans_found
            else:
                break

    def _trace_to_power_tsmc(self, cell, start_trans, start_net, target_power,
                              target_mos_type, visited):
        """Trace series transistor chain to power rail."""
        for trans in cell.transistors:
            if trans.name in visited:
                continue
            if trans.type != target_mos_type:
                continue

            conn = self._get_transistor_connections(cell, trans)

            if start_net in [conn['source'], conn['drain']]:
                other = conn['drain'] if conn['source'] == start_net else conn['source']

                if other == target_power:
                    return [trans.name] + [start_trans.name]
                else:
                    deeper_path = self._trace_to_power_tsmc(cell, trans, other, target_power,
                                                             target_mos_type, visited + [trans.name])
                    if deeper_path:
                        return deeper_path + [start_trans.name]

        return None

    def _trace_series_to_output_tsmc(self, cell, first_trans, first_net, target_mos_type,
                                       paths, transistors, path_type, output_nodes):
        """Trace series transistors from intermediate net to output."""
        current_net = first_net
        processed_nets = set()
        prev_trans = first_trans

        while current_net not in output_nodes and current_net not in processed_nets:
            processed_nets.add(current_net)
            next_trans_found = None

            for next_trans in cell.transistors:
                if next_trans == prev_trans or next_trans.name in transistors:
                    continue
                if next_trans.type != target_mos_type:
                    continue

                conn = self._get_transistor_connections(cell, next_trans)

                if current_net in [conn['source'], conn['drain']]:
                    next_trans_found = next_trans
                    break

            if next_trans_found:
                conn = self._get_transistor_connections(cell, next_trans_found)
                paths.append((prev_trans.name, next_trans_found.name))
                transistors.append(next_trans_found.name)

                next_other = conn['drain'] if conn['source'] == current_net else conn['source']

                if next_other in output_nodes:
                    paths.append((next_trans_found.name, next_other))
                    print(f"     Series {path_type} to output: -> {next_trans_found.name} -> {next_other}")
                    break
                else:
                    current_net = next_other
                    prev_trans = next_trans_found
            else:
                break

    def create_stage_aware_edges(self, stage_info: StageInfo, all_nodes: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Create edge_index and edge_attr from stage information.

        Edge attributes:
        - [1,0,0]: Stage 1 source-drain connection
        - [0,1,0]: Stage 2 source-drain connection
        - [0,0,1]: Gate control connection
        """
        edges = []
        edge_attrs = []

        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

        print(f"   Creating stage-aware edges:")

        def compress_paths(paths, stage_name):
            """Convert paths with intermediate nets to direct transistor connections."""
            compressed = set()  # Use set to avoid duplicates
            net_connections = {}
            direct_edges = set()  # Use set to avoid duplicates

            for src, dst in paths:
                src_in_nodes = src in node_to_idx
                dst_in_nodes = dst in node_to_idx

                if src_in_nodes and dst_in_nodes:
                    direct_edges.add((src, dst))
                elif src_in_nodes and not dst_in_nodes:
                    if dst not in net_connections:
                        net_connections[dst] = (set(), set())  # Use sets
                    net_connections[dst][0].add(src)
                elif not src_in_nodes and dst_in_nodes:
                    if src not in net_connections:
                        net_connections[src] = (set(), set())  # Use sets
                    net_connections[src][1].add(dst)

            compressed.update(direct_edges)

            for net, (incoming, outgoing) in net_connections.items():
                for src in incoming:
                    for dst in outgoing:
                        if (src, dst) not in compressed:
                            compressed.add((src, dst))
                            print(f"     {stage_name} (via {net}): {src} -> {dst}")

            return list(compressed)

        if stage_info.stage_type == "one_stage":
            compressed_paths = compress_paths(stage_info.stage1_paths, "One-stage")

            for src, dst in compressed_paths:
                if src in node_to_idx and dst in node_to_idx:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edges.append([src_idx, dst_idx])
                    edge_attrs.append([1.0, 0.0, 0.0])
                    print(f"     One-stage: {src}[{src_idx}] -> {dst}[{dst_idx}] [1,0,0]")

        else:  # two_stage
            # Stage 1 paths
            compressed_stage1 = compress_paths(stage_info.stage1_paths, "Stage 1")

            for src, dst in compressed_stage1:
                if src in node_to_idx and dst in node_to_idx:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edges.append([src_idx, dst_idx])
                    edge_attrs.append([1.0, 0.0, 0.0])
                    print(f"     Stage 1: {src}[{src_idx}] -> {dst}[{dst_idx}] [1,0,0]")

            # Stage 2 paths
            compressed_stage2 = compress_paths(stage_info.stage2_paths, "Stage 2")

            for src, dst in compressed_stage2:
                if src in node_to_idx and dst in node_to_idx:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edges.append([src_idx, dst_idx])
                    edge_attrs.append([0.0, 1.0, 0.0])
                    print(f"     Stage 2: {src}[{src_idx}] -> {dst}[{dst_idx}] [0,1,0]")

        print(f"     Total edges: {len(edges)} ({len([a for a in edge_attrs if a[0] == 1.0])} stage1, "
              f"{len([a for a in edge_attrs if a[1] == 1.0])} stage2)")

        return edges, edge_attrs

    def create_multi_stage_edges(self, multi_stage_info: MultiStageInfo,
                                  all_nodes: List[str],
                                  max_attr_dim: int = 5) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Create edge_index and edge_attr from multi-stage information.

        Edge attributes are one-hot encoded by stage number:
        - Stage 1: [1, 0, 0, 0, 0]
        - Stage 2: [0, 1, 0, 0, 0]
        - Stage 3: [0, 0, 1, 0, 0]
        - etc.

        Args:
            multi_stage_info: MultiStageInfo from classify_multi_stage_structure
            all_nodes: List of all node names
            max_attr_dim: Maximum dimension for edge attributes (default 5 for up to 5 stages)

        Returns:
            Tuple of (edge_index, edge_attr)
        """
        edges = []
        edge_attrs = []

        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

        print(f"   Creating multi-stage edges ({multi_stage_info.num_stages} stages):")

        def compress_paths(paths, stage_name):
            """Convert paths with intermediate nets to direct transistor connections."""
            compressed = set()  # Use set to avoid duplicates
            net_connections = {}
            direct_edges = set()  # Use set to avoid duplicates

            for src, dst in paths:
                src_in_nodes = src in node_to_idx
                dst_in_nodes = dst in node_to_idx

                if src_in_nodes and dst_in_nodes:
                    direct_edges.add((src, dst))
                elif src_in_nodes and not dst_in_nodes:
                    if dst not in net_connections:
                        net_connections[dst] = (set(), set())  # Use sets
                    net_connections[dst][0].add(src)
                elif not src_in_nodes and dst_in_nodes:
                    if src not in net_connections:
                        net_connections[src] = (set(), set())  # Use sets
                    net_connections[src][1].add(dst)

            compressed.update(direct_edges)

            for net, (incoming, outgoing) in net_connections.items():
                for src in incoming:
                    for dst in outgoing:
                        if (src, dst) not in compressed:
                            compressed.add((src, dst))
                            print(f"     {stage_name} (via {net}): {src} -> {dst}")

            return list(compressed)

        # Process each stage
        stage_edge_counts = []
        for stage in multi_stage_info.stages:
            stage_name = f"Stage {stage.stage_num} ({stage.mos_type.upper()})"
            compressed_paths = compress_paths(stage.paths, stage_name)

            stage_edge_count = 0
            for src, dst in compressed_paths:
                if src in node_to_idx and dst in node_to_idx:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edges.append([src_idx, dst_idx])

                    # Create one-hot edge attribute for this stage
                    attr = [0.0] * max_attr_dim
                    stage_idx = stage.stage_num - 1  # 0-indexed
                    if stage_idx < max_attr_dim:
                        attr[stage_idx] = 1.0
                    edge_attrs.append(attr)

                    print(f"     {stage_name}: {src}[{src_idx}] -> {dst}[{dst_idx}] {attr[:multi_stage_info.num_stages]}")
                    stage_edge_count += 1

            stage_edge_counts.append(stage_edge_count)

        # Print summary
        summary = ", ".join([f"stage{i+1}={count}" for i, count in enumerate(stage_edge_counts)])
        print(f"     Total edges: {len(edges)} ({summary})")

        return edges, edge_attrs


if __name__ == "__main__":
    print("TSMC Stage-Aware Path Extractor")
    print("=" * 50)
    print("Rise transition: VDD (pull-up) paths only")
    print("Fall transition: VSS (pull-down) paths only")
    print("Stage information in edge attributes [stage1, stage2, gate_control]")
