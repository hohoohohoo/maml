#!/usr/bin/env python

"""
ASAP7 Stage-Aware Path Extractor
- Complete Current Path Extractor (PMOS/NMOS path tracing, multi-stage support)
- Delay-Type Aware Stage Extractor (Rise/Fall에 따라 pull-up/pull-down path 선택)
"""

from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
import re


# ============================================================================
# Complete Current Path Extractor Classes
# ============================================================================

@dataclass
class CompleteStageInfo:
    """Complete stage 정보 (legacy 2-stage)"""
    stage_type: str
    all_intermediate_nodes: List[str]  # 모든 intermediate nodes
    stage1_current_paths: List[Tuple[str, str]]  # Complete current paths for stage 1
    stage2_current_paths: List[Tuple[str, str]]  # Complete current paths for stage 2
    stage1_transistors: List[str]
    stage2_transistors: List[str]


@dataclass
class StageData:
    """Single stage data for multi-stage support"""
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


class CompleteCurrentPathExtractor:
    """Complete current path extraction using PMOS/NMOS tracing"""

    def __init__(self, spice_file_path: str):
        self.spice_file_path = spice_file_path

    def find_intermediate_gate_nodes(self, cell, external_inputs: List[str], output_nodes: Optional[List[str]] = None) -> List[str]:
        """
        Intermediate gate node 찾기: gate로 사용되지만 input pin이 아닌 node
        이것만이 진짜 intermediate node이고 stage 구분의 기준이 됨
        """
        # 모든 gate nets 수집
        all_gate_nets = set(t.gate for t in cell.transistors)

        # Power rails, output, external inputs 제외
        power_nets = {'VDD', 'VSS'}
        if output_nodes is None:
            output_nodes = ['Y']  # Default fallback
        output_nets = set(output_nodes)
        external_nets = set(external_inputs)

        # Gate로 사용되면서 input pin이 아닌 것만 intermediate
        intermediate_gates = all_gate_nets - power_nets - output_nets - external_nets
        intermediate_gates = [gate for gate in intermediate_gates if gate and gate.strip()]

        return intermediate_gates

    def trace_pmos_paths(self, cell, start_node: str, end_nodes: Set[str]) -> List[List[str]]:
        """PMOS transistor만을 따라 pull-up path 추적"""
        paths = []

        def dfs_pmos(current_node: str, path: List[str], visited: Set[str]):
            if current_node in visited:
                return

            if current_node in end_nodes:
                paths.append(path + [current_node])
                return

            visited.add(current_node)

            # PMOS transistor만 탐색
            for trans in cell.transistors:
                if 'pmos' not in trans.type.lower():
                    continue

                next_node = None
                if trans.source == current_node:
                    next_node = trans.drain
                elif trans.drain == current_node:
                    next_node = trans.source

                if next_node and next_node not in visited:
                    dfs_pmos(next_node, path + [current_node, trans.name], visited.copy())

        dfs_pmos(start_node, [], set())
        return paths

    def trace_nmos_paths(self, cell, start_node: str, end_nodes: Set[str]) -> List[List[str]]:
        """NMOS transistor만을 따라 pull-down path 추적"""
        paths = []

        def dfs_nmos(current_node: str, path: List[str], visited: Set[str]):
            if current_node in visited:
                return

            if current_node in end_nodes:
                paths.append(path + [current_node])
                return

            visited.add(current_node)

            # NMOS transistor만 탐색
            for trans in cell.transistors:
                if 'nmos' not in trans.type.lower():
                    continue

                next_node = None
                if trans.source == current_node:
                    next_node = trans.drain
                elif trans.drain == current_node:
                    next_node = trans.source

                if next_node and next_node not in visited:
                    dfs_nmos(next_node, path + [current_node, trans.name], visited.copy())

        dfs_nmos(start_node, [], set())
        return paths

    def extract_complete_current_paths(self, cell, external_inputs: List[str], delay_type: str, output_nodes: Optional[List[str]] = None) -> CompleteStageInfo:
        """Complete current path extraction based on intermediate gate nodes"""

        # Default output nodes if not provided
        if output_nodes is None:
            output_nodes = ['Y']

        # 1. Intermediate gate nodes만 찾기 (stage 구분의 기준)
        intermediate_gates = self.find_intermediate_gate_nodes(cell, external_inputs, output_nodes)
        print(f"🔍 Intermediate gate nodes (stage dividers): {intermediate_gates}")

        if not intermediate_gates:
            # One-stage logic
            return self._extract_one_stage_paths(cell, delay_type, output_nodes)

        # 2. Two-stage logic: current path 구성 (intermediate gate 기준)
        stage1_paths = []
        stage2_paths = []
        stage1_transistors = []
        stage2_transistors = []

        print(f"📊 Two-stage logic detected with intermediate gates: {intermediate_gates}")

        # Convert output_nodes to set for path tracing
        output_node_set = set(output_nodes)

        if 'rise' in delay_type:
            # Rise: Stage1(pull-down to intermediate gate) → Stage2(pull-up to output)
            print(f"   Rise timing: Stage1(NMOS → intermediate gate) → Stage2(PMOS → {output_nodes})")

            # Stage 1: VSS에서 intermediate gates까지 NMOS path (생성하는 경로)
            stage1_nmos_paths = self.trace_nmos_paths(cell, 'VSS', set(intermediate_gates))
            print(f"   Stage 1 NMOS paths (to intermediate): {len(stage1_nmos_paths)} found")

            # Stage 2: VDD에서 output까지 PMOS path (intermediate gate에 의해 제어되는 경로 포함)
            stage2_pmos_paths = self.trace_pmos_paths(cell, 'VDD', output_node_set)
            # 추가로 intermediate gate에서 output까지 직접 경로도 포함
            for gate in intermediate_gates:
                for output_node in output_nodes:
                    gate_to_out_paths = self._find_paths_controlled_by_gate(cell, gate, 'pmos', output_node)
                    stage2_pmos_paths.extend(gate_to_out_paths)
            print(f"   Stage 2 PMOS paths (to output): {len(stage2_pmos_paths)} found")

            # Convert paths to edges
            stage1_paths, stage1_transistors = self._paths_to_edges(stage1_nmos_paths, "Stage1 NMOS")
            stage2_paths, stage2_transistors = self._paths_to_edges(stage2_pmos_paths, "Stage2 PMOS")

        else:  # fall
            # Fall: Stage1(pull-up to intermediate gate) → Stage2(pull-down to output)
            print(f"   Fall timing: Stage1(PMOS → intermediate gate) → Stage2(NMOS → {output_nodes})")

            # Stage 1: VDD에서 intermediate gates까지 PMOS path
            stage1_pmos_paths = self.trace_pmos_paths(cell, 'VDD', set(intermediate_gates))
            print(f"   Stage 1 PMOS paths (to intermediate): {len(stage1_pmos_paths)} found")

            # Stage 2: VSS에서 output까지 NMOS path
            stage2_nmos_paths = self.trace_nmos_paths(cell, 'VSS', output_node_set)
            # 추가로 intermediate gate에서 output까지 직접 경로도 포함
            for gate in intermediate_gates:
                for output_node in output_nodes:
                    gate_to_out_paths = self._find_paths_controlled_by_gate(cell, gate, 'nmos', output_node)
                    stage2_nmos_paths.extend(gate_to_out_paths)
            print(f"   Stage 2 NMOS paths (to output): {len(stage2_nmos_paths)} found")

            # Convert paths to edges
            stage1_paths, stage1_transistors = self._paths_to_edges(stage1_pmos_paths, "Stage1 PMOS")
            stage2_paths, stage2_transistors = self._paths_to_edges(stage2_nmos_paths, "Stage2 NMOS")

        return CompleteStageInfo(
            stage_type="two_stage",
            all_intermediate_nodes=intermediate_gates,
            stage1_current_paths=stage1_paths,
            stage2_current_paths=stage2_paths,
            stage1_transistors=stage1_transistors,
            stage2_transistors=stage2_transistors
        )

    def _paths_to_edges(self, paths: List[List[str]], stage_name: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Path list를 edge list로 변환"""
        edges = []
        transistors = []

        for path in paths:
            print(f"     {stage_name} path: {' → '.join(path)}")

            # Path를 연속된 edge로 변환
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                edges.append((src, dst))

                # Transistor 이름 수집 (MM으로 시작)
                if src.startswith('MM') and src not in transistors:
                    transistors.append(src)
                if dst.startswith('MM') and dst not in transistors:
                    transistors.append(dst)

        return edges, transistors

    def _find_paths_controlled_by_gate(self, cell, gate_node: str, transistor_type: str, target: str) -> List[List[str]]:
        """특정 gate node에 의해 제어되는 transistor를 통한 path 찾기

        주의: Complete current path는 반드시 power rail(VDD/VSS)에서 시작해야 하므로
        intermediate gate에 의해 제어되는 transistor만 찾고,
        실제 path 추적은 main trace 함수에서 power rail부터 시작
        """
        # 이 함수는 사용하지 않음 - power rail에서 시작하지 않는 path는 current path가 아님
        return []

    def _extract_one_stage_paths(self, cell, delay_type: str, output_nodes: List[str]) -> CompleteStageInfo:
        """One-stage logic path extraction"""

        # Convert output_nodes to set for path tracing
        output_node_set = set(output_nodes)

        if 'rise' in delay_type:
            # Direct VDD → output pull-up paths
            pmos_paths = self.trace_pmos_paths(cell, 'VDD', output_node_set)
            stage1_paths, stage1_transistors = self._paths_to_edges(pmos_paths, "One-stage PMOS")
            stage2_paths, stage2_transistors = [], []
        else:
            # Direct VSS → output pull-down paths
            nmos_paths = self.trace_nmos_paths(cell, 'VSS', output_node_set)
            stage1_paths, stage1_transistors = self._paths_to_edges(nmos_paths, "One-stage NMOS")
            stage2_paths, stage2_transistors = [], []

        return CompleteStageInfo(
            stage_type="one_stage",
            all_intermediate_nodes=[],
            stage1_current_paths=stage1_paths,
            stage2_current_paths=stage2_paths,
            stage1_transistors=stage1_transistors,
            stage2_transistors=stage2_transistors
        )

    def _find_stage_transistors(self, cell, power_node: str, target_nodes: Set[str],
                                 mos_type: str) -> Set[str]:
        """
        Find all transistors in the current path from power to target_nodes.
        Uses DFS to find ALL parallel paths (no early return).
        """
        stage_trans = set()

        def dfs(current_node: str, visited: Set[str]) -> bool:
            if current_node in visited:
                return False

            if current_node in target_nodes:
                return True

            visited.add(current_node)
            any_path_found = False

            for trans in cell.transistors:
                if mos_type not in trans.type.lower():
                    continue

                next_node = None
                if trans.source == current_node:
                    next_node = trans.drain
                elif trans.drain == current_node:
                    next_node = trans.source

                if next_node and next_node not in visited:
                    if dfs(next_node, visited.copy()):
                        stage_trans.add(trans.name)
                        any_path_found = True
                        # NO early return - continue to find all parallel paths

            return any_path_found

        dfs(power_node, set())
        return stage_trans

    def extract_multi_stage_paths(self, cell, external_inputs: List[str],
                                   delay_type: str,
                                   output_nodes: Optional[List[str]] = None,
                                   max_stages: int = 10) -> MultiStageInfo:
        """
        Extract multi-stage current paths for complex cells (XOR, XNOR, etc.).

        Iteratively traces back from output until all gate nodes are external inputs.
        Alternates between PMOS and NMOS for each stage.

        Args:
            cell: Cell object with transistors
            external_inputs: List of external input port names
            delay_type: 'rise_transition' or 'fall_transition'
            output_nodes: Output node list
            max_stages: Maximum stages to prevent infinite loops

        Returns:
            MultiStageInfo with all stages
        """
        if output_nodes is None:
            output_nodes = ['Y']

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

        stages_reversed = []  # Build from output backwards, then reverse
        all_intermediate_nodes = []
        visited_target_nodes = set()  # Track visited targets to prevent cycles
        visited_transistors = set()   # Track transistors already assigned to a stage

        current_target_nodes = set(output_nodes)
        current_mos_type = final_mos_type
        current_power = final_power
        stage_count = 0

        print(f"\n{'='*60}")
        print(f"Multi-Stage Analysis (ASAP7): {delay_type}")
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
                paths_raw = self.trace_pmos_paths(cell, current_power, current_target_nodes)
            else:
                paths_raw = self.trace_nmos_paths(cell, current_power, current_target_nodes)

            paths, transistor_list = self._paths_to_edges(
                paths_raw, f"Stage-{stage_count} {current_mos_type.upper()}"
            )

            # 3. Find gate nodes of transistors in this stage
            gate_nodes = set()
            for trans in cell.transistors:
                if trans.name in stage_transistors:
                    gate = trans.gate
                    if gate and gate not in power_nets:
                        gate_nodes.add(gate)

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
            multi_stage_info: MultiStageInfo from extract_multi_stage_paths
            all_nodes: List of all node names
            max_attr_dim: Maximum dimension for edge attributes (default 5)

        Returns:
            Tuple of (edge_index, edge_attr)
        """
        edges = []
        edge_attrs = []

        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

        print(f"   Creating multi-stage edges ({multi_stage_info.num_stages} stages):")

        def compress_paths(paths, stage_name):
            """Convert paths with intermediate nets to direct transistor connections."""
            compressed = []
            net_connections = {}
            direct_edges = []

            for src, dst in paths:
                src_in_nodes = src in node_to_idx
                dst_in_nodes = dst in node_to_idx

                if src_in_nodes and dst_in_nodes:
                    direct_edges.append((src, dst))
                elif src_in_nodes and not dst_in_nodes:
                    if dst not in net_connections:
                        net_connections[dst] = ([], [])
                    net_connections[dst][0].append(src)
                elif not src_in_nodes and dst_in_nodes:
                    if src not in net_connections:
                        net_connections[src] = ([], [])
                    net_connections[src][1].append(dst)

            compressed.extend(direct_edges)

            for net, (incoming, outgoing) in net_connections.items():
                for src in incoming:
                    for dst in outgoing:
                        compressed.append((src, dst))
                        print(f"     {stage_name} (via {net}): {src} -> {dst}")

            return compressed

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


# ============================================================================
# Delay-Aware Stage Extractor Classes
# ============================================================================

@dataclass
class StageInfo:
    """Stage 정보를 담는 클래스"""
    stage_type: str  # "one_stage" or "two_stage"
    intermediate_gates: List[str]  # 중간 gate nodes
    stage1_paths: List[Tuple[str, str]]  # (source, destination) for first stage
    stage2_paths: List[Tuple[str, str]]  # (source, destination) for second stage
    stage1_transistors: List[str]  # First stage controlled transistors
    stage2_transistors: List[str]  # Second stage controlled transistors

# class DelayAwareStageExtractor:
#     """
#     Delay-type aware stage extraction system
#     - rise_transition: VDD (pull-up) paths only
#     - fall_transition: VSS (pull-down) paths only
#     """
    
#     def __init__(self, spice_file_path: str):
#         self.spice_file_path = spice_file_path
#         self.complete_extractor = CompleteCurrentPathExtractor(spice_file_path)
        
#     def classify_stage_structure(self, cell, external_inputs: List[str], delay_type: str = "rise_transition", use_complete_extraction: bool = True, output_nodes: Optional[List[str]] = None) -> StageInfo:
#         """
#         Cell을 one-stage 또는 two-stage로 분류하고 delay_type에 따라 path 정보 추출

#         Args:
#             use_complete_extraction: True면 complete current path extraction 사용
#             output_nodes: Output node list (e.g., ['Y'] or ['CON', 'SN']). If None, defaults to ['Y']
#         """

#         # Default output nodes to ['Y'] if not provided
#         if output_nodes is None:
#             output_nodes = ['Y']

#         if use_complete_extraction:
#             return self._classify_with_complete_extraction(cell, external_inputs, delay_type, output_nodes)
#         else:
#             return self._classify_with_legacy_method(cell, external_inputs, delay_type, output_nodes)
    
#     def _classify_with_complete_extraction(self, cell, external_inputs: List[str], delay_type: str, output_nodes: List[str]) -> StageInfo:
#         """Complete current path extraction을 사용한 분류"""

#         print(f"🔍 Using Complete Current Path Extraction")
#         complete_info = self.complete_extractor.extract_complete_current_paths(cell, external_inputs, delay_type, output_nodes)

#         # CompleteStageInfo를 StageInfo로 변환
#         return StageInfo(
#             stage_type=complete_info.stage_type,
#             intermediate_gates=complete_info.all_intermediate_nodes,
#             stage1_paths=complete_info.stage1_current_paths,
#             stage2_paths=complete_info.stage2_current_paths,
#             stage1_transistors=complete_info.stage1_transistors,
#             stage2_transistors=complete_info.stage2_transistors
#         )

#     def _classify_with_legacy_method(self, cell, external_inputs: List[str], delay_type: str, output_nodes: List[str]) -> StageInfo:
#         """
#         Legacy method - 기존 gate-control 방식
#         """
        
#         # 1. 모든 gate nets 수집
#         all_gate_nets = set(t.gate for t in cell.transistors)

#         # 2. Intermediate gate nodes 감지
#         power_nets = {'VDD', 'VSS'}
#         output_nets = set(output_nodes)  # Use dynamically detected output nodes
#         external_nets = set(external_inputs)

#         intermediate_gates = all_gate_nets - power_nets - output_nets - external_nets
#         intermediate_gates = [gate for gate in intermediate_gates if gate]  # 빈 문자열 제거
        
#         print(f"🔍 Legacy Stage Classification (Gate-Control Based):")
#         print(f"   Delay type: {delay_type} ({'pull-up only' if 'rise' in delay_type else 'pull-down only'})")
#         print(f"   Intermediate gates: {intermediate_gates}")
        
#         # CKINV cell 특별 처리: 항상 one-stage로 처리 (MM0, MM1만 사용)
#         is_ckinv = ('CKINV' in cell.name) if hasattr(cell, 'name') else False
        
#         # 3. Stage 분류 (delay_type에 따른 path 필터링)
#         if intermediate_gates and not is_ckinv:
#             return self._analyze_two_stage_structure(cell, external_inputs, intermediate_gates, delay_type, output_nodes)
#         else:
#             return self._analyze_one_stage_structure(cell, external_inputs, delay_type, output_nodes)
    
#     def _analyze_one_stage_structure(self, cell, external_inputs: List[str], delay_type: str, output_nodes: List[str]) -> StageInfo:
#         """One-stage structure 분석: delay_type에 따른 selective path extraction"""

#         print(f"   📊 ONE-STAGE structure detected")

#         stage1_paths = []
#         stage1_transistors = []

#         # delay_type에 따라 target power rail 결정
#         target_power = 'VDD' if 'rise' in delay_type else 'VSS'
#         path_type_name = "pull-up" if target_power == 'VDD' else "pull-down"

#         print(f"   🎯 Target: {target_power} ({path_type_name} paths)")

#         # CKINV cell 예외 처리: MM0, MM1만 고려
#         is_ckinv = ('CKINV' in cell.name) if hasattr(cell, 'name') else False

#         if is_ckinv:
#             print(f"   ⚙️ CKINV cell detected: MM0, MM1 transistors only")
#             allowed_transistors = ['MM0', 'MM1']
#         else:
#             allowed_transistors = None

#         # Target power → output paths만 찾기
#         for trans in cell.transistors:
#             # CKINV cell인 경우 MM0, MM1만 허용
#             if is_ckinv and trans.name not in allowed_transistors:
#                 continue
#             # Target power rail connection만 확인
#             if trans.source == target_power or trans.drain == target_power:
#                 other_terminal = trans.drain if trans.source == target_power else trans.source

#                 # Add power → transistor connection
#                 stage1_paths.append((target_power, trans.name))
#                 stage1_transistors.append(trans.name)

#                 if other_terminal in output_nodes:
#                     # Direct connection: target_power → transistor → output
#                     stage1_paths.append((trans.name, other_terminal))
#                     print(f"     Direct {path_type_name}: {target_power} → {trans.name} → {other_terminal}")
#                 else:
#                     # Series connection: target_power → transistor → intermediate net
#                     stage1_paths.append((trans.name, other_terminal))
#                     print(f"     Series {path_type_name}: {target_power} → {trans.name} → {other_terminal}")

#                     # Find series transistor connections (skip intermediate nets)
#                     self._trace_series_transistors(cell, trans, other_terminal, target_power, stage1_paths, stage1_transistors, path_type_name, output_nodes)
        
#         return StageInfo(
#             stage_type="one_stage",
#             intermediate_gates=[],
#             stage1_paths=stage1_paths,
#             stage2_paths=[],
#             stage1_transistors=stage1_transistors,
#             stage2_transistors=[]
#         )
    
#     def _analyze_two_stage_structure(self, cell, external_inputs: List[str], intermediate_gates: List[str], delay_type: str, output_nodes: List[str]) -> StageInfo:
#         """Two-stage structure 분석: delay_type에 따른 selective path extraction"""
        
#         print(f"   📊 TWO-STAGE structure detected")
#         print(f"   🎯 Intermediate gate(s): {intermediate_gates}")
        
#         stage1_paths = []
#         stage2_paths = []
#         stage1_transistors = []
#         stage2_transistors = []
        
#         # Two-stage에서는 inverting logic 고려
#         # Rise: First stage (pull-down) → Second stage (pull-up)
#         # Fall: First stage (pull-up) → Second stage (pull-down)
#         if 'rise' in delay_type:
#             stage1_power = 'VSS'  # First stage pull-down
#             stage2_power = 'VDD'  # Second stage pull-up
#             stage1_type = "pull-down"
#             stage2_type = "pull-up"
#         else:  # fall_transition
#             stage1_power = 'VDD'  # First stage pull-up
#             stage2_power = 'VSS'  # Second stage pull-down
#             stage1_type = "pull-up"
#             stage2_type = "pull-down"
        
#         print(f"   🎯 Two-stage inverting logic:")
#         print(f"     Stage 1: {stage1_power} ({stage1_type}) → intermediate gates")
#         print(f"     Stage 2: {stage2_power} ({stage2_type}) → output")
        
#         # CKINV cell 예외 처리: MM0, MM1만 고려
#         is_ckinv = ('CKINV' in cell.name) if hasattr(cell, 'name') else False
        
#         if is_ckinv:
#             print(f"   ⚙️ CKINV cell detected: MM0, MM1 transistors only")
#             allowed_transistors = ['MM0', 'MM1']
#         else:
#             allowed_transistors = None
        
#         # Stage 1: Stage1 Power → intermediate gate nodes (including series connections)
#         print(f"   🔸 Stage 1 Analysis ({stage1_power} → intermediate gates):")
#         for intermediate_gate in intermediate_gates:
#             # Find all transistors that could contribute to generating intermediate gate
#             stage1_trans_candidates = []
            
#             for trans in cell.transistors:
#                 # CKINV cell인 경우 MM0, MM1만 허용
#                 if is_ckinv and trans.name not in allowed_transistors:
#                     continue
                
#                 # Check if this transistor connects to intermediate gate
#                 if intermediate_gate in [trans.source, trans.drain]:
#                     # Check transistor type matches stage1 requirement
#                     is_nmos = 'nmos' in trans.type.lower()
#                     is_pmos = 'pmos' in trans.type.lower()
                    
#                     # For rise: stage1 needs pull-down (NMOS), for fall: stage1 needs pull-up (PMOS)
#                     if ('rise' in delay_type and is_nmos) or ('fall' in delay_type and is_pmos):
#                         stage1_trans_candidates.append(trans)
            
#             # Build paths from stage1 power through series transistors
#             for trans in stage1_trans_candidates:
#                 other_terminal = trans.drain if trans.source == intermediate_gate else trans.source
                
#                 # Direct power connection
#                 if other_terminal == stage1_power:
#                     stage1_paths.append((other_terminal, trans.name))
#                     stage1_paths.append((trans.name, intermediate_gate))
#                     stage1_transistors.append(trans.name)
#                     print(f"     Stage 1 {stage1_type}: {other_terminal} → {trans.name} → {intermediate_gate} (controlled by {trans.gate})")
#                 else:
#                     # Series connection - trace back to power
#                     series_path = self._trace_to_power(cell, trans, other_terminal, stage1_power, intermediate_gate, [trans.name])
#                     if series_path:
#                         for path_trans in series_path:
#                             if path_trans not in stage1_transistors:
#                                 stage1_transistors.append(path_trans)
                        
#                         # Build transistor-to-transistor connections
#                         if len(series_path) > 1:
#                             # Power to first transistor
#                             stage1_paths.append((stage1_power, series_path[0]))
#                             # Transistor-to-transistor connections
#                             for i in range(len(series_path) - 1):
#                                 stage1_paths.append((series_path[i], series_path[i+1]))
#                             # Last transistor to intermediate gate
#                             stage1_paths.append((series_path[-1], intermediate_gate))
#                         else:
#                             # Single transistor
#                             stage1_paths.append((stage1_power, trans.name))
#                             stage1_paths.append((trans.name, intermediate_gate))
                        
#                         print(f"     Stage 1 {stage1_type} series: {stage1_power} → {' → '.join(series_path)} → {intermediate_gate}")
        
#         # Stage 2: All transistors that contribute to pull-up/pull-down paths after intermediate gates
#         print(f"   🔸 Stage 2 Analysis (paths after intermediate gates):")
        
#         # Stage 2는 intermediate gate 뒤의 모든 pull-up/pull-down path를 포함
#         # 1) intermediate gate에 의해 직접 제어되는 transistors
#         # 2) intermediate gate가 활성화된 후 전류가 흐르는 경로의 모든 transistors
        
#         for intermediate_gate in intermediate_gates:
#             print(f"   🎯 Analyzing paths after intermediate gate: {intermediate_gate}")
            
#             # 1. intermediate gate에 의해 직접 제어되는 transistors (기존 로직)
#             direct_controlled = []
#             for trans in cell.transistors:
#                 if is_ckinv and trans.name not in allowed_transistors:
#                     continue
                    
#                 if trans.gate == intermediate_gate:
#                     direct_controlled.append(trans)
            
#             # 2. intermediate gate가 활성화된 후 전류 경로 분석
#             # rise_transition: pull-up path (VDD → Y)
#             # fall_transition: pull-down path (VSS → Y)
            
#             if 'rise' in delay_type:
#                 # Pull-up path analysis: VDD → ... → output
#                 stage2_pull_up_paths = self._analyze_stage2_pull_up_paths(cell, intermediate_gate, direct_controlled, output_nodes)
#                 stage2_transistors.extend(stage2_pull_up_paths['transistors'])
#                 stage2_paths.extend(stage2_pull_up_paths['paths'])

#                 for path_info in stage2_pull_up_paths['descriptions']:
#                     print(f"     {path_info}")

#             else:  # fall_transition
#                 # Pull-down path analysis: VSS → ... → output
#                 stage2_pull_down_paths = self._analyze_stage2_pull_down_paths(cell, intermediate_gate, direct_controlled, output_nodes)
#                 stage2_transistors.extend(stage2_pull_down_paths['transistors'])
#                 stage2_paths.extend(stage2_pull_down_paths['paths'])

#                 for path_info in stage2_pull_down_paths['descriptions']:
#                     print(f"     {path_info}")
        
#         return StageInfo(
#             stage_type="two_stage", 
#             intermediate_gates=intermediate_gates,
#             stage1_paths=stage1_paths,
#             stage2_paths=stage2_paths,
#             stage1_transistors=stage1_transistors,
#             stage2_transistors=stage2_transistors
#         )
    
#     def create_full_graph_edges(self, cell, all_nodes: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
#         """
#         전체 cell에 대한 complete adjacency matrix 생성 (baseline용)
#         🔧 FIX: Intermediate node 없이 transistor 간 직접 연결만 표현
#         Gate, Source, Drain 연결을 모두 동일한 edge로 처리

#         Edge attributes:
#         - [1,0,0]: All connections (gate/source/drain 구분 없이 동일)
#         """

#         edges = []
#         edge_attrs = []

#         # Node index mapping
#         node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

#         print(f"🔗 Creating full graph edges (baseline - no intermediate nodes):")

#         # Net을 통한 transistor 간 연결 맵 생성
#         # net -> [connected_transistors/nodes]
#         # 🔧 FIX: intermediate net뿐만 아니라 input/output node도 포함하여 공유하는 transistor 연결
#         net_connections = {}

#         # 모든 transistor의 연결 정보 수집
#         for trans in cell.transistors:
#             trans_name = trans.name

#             # Transistor가 node list에 있는 경우만 처리
#             if trans_name not in node_to_idx:
#                 continue

#             # Source, Drain, Gate 연결 수집
#             for terminal_type, terminal_node in [('source', trans.source), ('drain', trans.drain), ('gate', trans.gate)]:
#                 # Terminal이 node list에 있으면 직접 연결
#                 if terminal_node in node_to_idx:
#                     trans_idx = node_to_idx[trans_name]
#                     terminal_idx = node_to_idx[terminal_node]

#                     # Bidirectional edge (무방향 그래프)
#                     edges.append([trans_idx, terminal_idx])
#                     edge_attrs.append([1.0, 0.0, 0.0])
#                     edges.append([terminal_idx, trans_idx])
#                     edge_attrs.append([1.0, 0.0, 0.0])

#                     # 🔧 FIX: Input/output node를 공유하는 transistor들도 연결하기 위해 추가
#                     # Terminal node를 통해 연결된 transistor 추적
#                     if terminal_node not in net_connections:
#                         net_connections[terminal_node] = []
#                     net_connections[terminal_node].append(trans_name)

#                 else:
#                     # Terminal이 intermediate net이면 나중에 처리
#                     if terminal_node not in net_connections:
#                         net_connections[terminal_node] = []
#                     net_connections[terminal_node].append(trans_name)

#         # Intermediate nets를 통한 transistor 간 직접 연결
#         for net, connected_trans in net_connections.items():
#             print(f"   Via {net}: connecting {len(connected_trans)} transistors")
#             # 같은 net에 연결된 모든 transistor 쌍을 연결
#             for i in range(len(connected_trans)):
#                 for j in range(i + 1, len(connected_trans)):
#                     trans1 = connected_trans[i]
#                     trans2 = connected_trans[j]

#                     if trans1 in node_to_idx and trans2 in node_to_idx:
#                         idx1 = node_to_idx[trans1]
#                         idx2 = node_to_idx[trans2]

#                         # Bidirectional edge
#                         edges.append([idx1, idx2])
#                         edge_attrs.append([1.0, 0.0, 0.0])
#                         edges.append([idx2, idx1])
#                         edge_attrs.append([1.0, 0.0, 0.0])

#                         print(f"   {trans1} ↔ {trans2} (via {net})")

#         print(f"   📊 Total edges: {len(edges)} (all connections treated equally)")

#         return edges, edge_attrs

#     def create_stage_aware_edges(self, stage_info: StageInfo, all_nodes: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
#         """
#         Stage 정보를 바탕으로 edge_index와 edge_attr 생성

#         🔧 FIX: intermediate net (net89, net90 등)을 통한 연결은 transistor 간 직접 edge로 변환
#         예: MM2 → net90 → MM1 을 MM2 → MM1 로 변환

#         Edge attributes:
#         - [1,0,0]: Stage 1 source-drain connection
#         - [0,1,0]: Stage 2 source-drain connection
#         - [0,0,1]: Gate control connection (stage transition)
#         """

#         edges = []
#         edge_attrs = []

#         # Node index mapping
#         node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

#         print(f"🔗 Creating stage-aware edges:")

#         # Helper function: paths를 변환하여 intermediate net 제거
#         def compress_paths(paths, stage_name):
#             """intermediate net을 통한 연결을 transistor 간 직접 연결로 변환"""
#             compressed = []

#             # intermediate net을 key로 하는 연결 맵 생성
#             # net -> (incoming_nodes, outgoing_nodes)
#             net_connections = {}
#             direct_edges = []

#             for src, dst in paths:
#                 src_in_nodes = src in node_to_idx
#                 dst_in_nodes = dst in node_to_idx

#                 if src_in_nodes and dst_in_nodes:
#                     # 둘 다 node list에 있으면 직접 연결
#                     direct_edges.append((src, dst))
#                 elif src_in_nodes and not dst_in_nodes:
#                     # dst가 intermediate net
#                     if dst not in net_connections:
#                         net_connections[dst] = ([], [])
#                     net_connections[dst][0].append(src)  # incoming
#                 elif not src_in_nodes and dst_in_nodes:
#                     # src가 intermediate net
#                     if src not in net_connections:
#                         net_connections[src] = ([], [])
#                     net_connections[src][1].append(dst)  # outgoing

#             # Direct edges 추가
#             compressed.extend(direct_edges)

#             # Intermediate nets를 통한 연결을 transistor 간 직접 연결로 변환
#             for net, (incoming, outgoing) in net_connections.items():
#                 for src in incoming:
#                     for dst in outgoing:
#                         compressed.append((src, dst))
#                         print(f"   {stage_name} (via {net}): {src} → {dst}")

#             return compressed

#         if stage_info.stage_type == "one_stage":
#             # One-stage: 모든 path를 stage 1로 처리
#             compressed_paths = compress_paths(stage_info.stage1_paths, "One-stage")

#             for src, dst in compressed_paths:
#                 if src in node_to_idx and dst in node_to_idx:
#                     src_idx = node_to_idx[src]
#                     dst_idx = node_to_idx[dst]
#                     edges.append([src_idx, dst_idx])
#                     edge_attrs.append([1.0, 0.0, 0.0])  # Stage 1 source-drain
#                     print(f"   One-stage: {src}[{src_idx}] → {dst}[{dst_idx}] [1,0,0]")

#         else:  # two_stage
#             # Stage 1 paths (intermediate nets 제거)
#             compressed_stage1 = compress_paths(stage_info.stage1_paths, "Stage 1")

#             for src, dst in compressed_stage1:
#                 if src in node_to_idx and dst in node_to_idx:
#                     src_idx = node_to_idx[src]
#                     dst_idx = node_to_idx[dst]
#                     edges.append([src_idx, dst_idx])
#                     edge_attrs.append([1.0, 0.0, 0.0])  # Stage 1 source-drain
#                     print(f"   Stage 1: {src}[{src_idx}] → {dst}[{dst_idx}] [1,0,0]")

#             # Gate control connections (stage transition)
#             # 🔧 REMOVED: intermediate gate → transistor edge 제거
#             # Stage-aware mode에서는 intermediate gate node가 있지만
#             # gate control edge는 포함하지 않음 (node to transistor path 제외)
#             # for intermediate_gate in stage_info.intermediate_gates:
#             #     if intermediate_gate in node_to_idx:
#             #         gate_idx = node_to_idx[intermediate_gate]
#             #         for trans_name in stage_info.stage2_transistors:
#             #             if trans_name in node_to_idx:
#             #                 trans_idx = node_to_idx[trans_name]
#             #                 edges.append([gate_idx, trans_idx])
#             #                 edge_attrs.append([0.0, 0.0, 1.0])

#             # Stage 2 paths (intermediate nets 제거)
#             compressed_stage2 = compress_paths(stage_info.stage2_paths, "Stage 2")

#             for src, dst in compressed_stage2:
#                 if src in node_to_idx and dst in node_to_idx:
#                     src_idx = node_to_idx[src]
#                     dst_idx = node_to_idx[dst]
#                     edges.append([src_idx, dst_idx])
#                     edge_attrs.append([0.0, 1.0, 0.0])  # Stage 2 source-drain
#                     print(f"   Stage 2: {src}[{src_idx}] → {dst}[{dst_idx}] [0,1,0]")

#         print(f"   📊 Total edges: {len(edges)} ({len([a for a in edge_attrs if a[0] == 1.0])} stage1, "
#               f"{len([a for a in edge_attrs if a[1] == 1.0])} stage2, "
#               f"{len([a for a in edge_attrs if a[2] == 1.0])} gate-control)")

#         return edges, edge_attrs

#     def _trace_to_power(self, cell, start_trans, start_net, target_power, target_gate, visited):
#         """
#         Series transistor chain을 power rail까지 추적
#         """
#         # Find transistor connected to start_net
#         for trans in cell.transistors:
#             if trans.name in visited:
#                 continue
                
#             if start_net in [trans.source, trans.drain]:
#                 other = trans.drain if trans.source == start_net else trans.source
                
#                 if other == target_power:
#                     # Found power connection
#                     paths = []
#                     paths.append((target_power, trans.name))
#                     paths.append((trans.name, start_net))
#                     paths.append((start_net, start_trans.name))
#                     return [trans.name, start_trans.name]
#                 else:
#                     # Continue tracing
#                     deeper_path = self._trace_to_power(cell, trans, other, target_power, target_gate, visited + [trans.name])
#                     if deeper_path:
#                         return [trans.name] + deeper_path
        
#         return None
    
#     def _trace_stage2_to_output(self, cell, start_trans, intermediate_gate, target_power, visited):
#         """
#         Stage 2에서 intermediate gate 제어 transistor로부터 output까지의 path 추적
#         Stage 2는 intermediate gate에 의해 제어되는 transistor들만 포함
#         """
#         # start_trans가 intermediate gate에 의해 제어되지 않으면 Stage 2가 아님
#         if start_trans.gate != intermediate_gate:
#             return None
            
#         # start_trans의 power connection 확인
#         power_connection = None
#         output_connection = None
        
#         if start_trans.source == target_power:
#             power_connection = start_trans.source
#             output_connection = start_trans.drain
#         elif start_trans.drain == target_power:
#             power_connection = start_trans.drain
#             output_connection = start_trans.source
        
#         if power_connection:
#             # Direct power connection
#             return [start_trans.name]
        
#         # Indirect connection through intermediate nets
#         # start_trans에서 output 방향으로만 추적 (power 방향은 Stage 2가 아님)
#         start_net = None
#         if start_trans.source not in ['VDD', 'VSS'] and start_trans.source != 'Y':
#             start_net = start_trans.source
#         elif start_trans.drain not in ['VDD', 'VSS'] and start_trans.drain != 'Y':
#             start_net = start_trans.drain
            
#         if not start_net or start_net in visited:
#             return None
            
#         # start_net에서 Y로의 path가 있는지 확인 (intermediate gate controlled transistor만)
#         current_path = [start_trans.name]
#         current_net = start_net
#         processed_nets = set(visited)
        
#         while current_net != 'Y' and current_net not in processed_nets:
#             processed_nets.add(current_net)
#             found_next = False
            
#             for trans in cell.transistors:
#                 if (trans.name not in [t for t in current_path] and 
#                     trans.gate == intermediate_gate and  # Must be controlled by same intermediate gate
#                     current_net in [trans.source, trans.drain]):
                    
#                     other_net = trans.drain if trans.source == current_net else trans.source
#                     current_path.append(trans.name)
                    
#                     if other_net == 'Y':
#                         return current_path
#                     elif other_net not in ['VDD', 'VSS']:
#                         current_net = other_net
#                         found_next = True
#                         break
            
#             if not found_next:
#                 break
                
#         return None
    
#     def _analyze_stage2_pull_up_paths(self, cell, intermediate_gate, direct_controlled, output_nodes: List[str]):
#         """
#         Stage 2 pull-up path 분석: intermediate gate 뒤의 VDD → output 경로
#         intermediate gate가 켜진 후 전류가 흐르는 모든 transistor 포함
#         """
#         result = {
#             'transistors': [],
#             'paths': [],
#             'descriptions': []
#         }

#         # 1. intermediate gate에 의해 직접 제어되는 PMOS transistors
#         for trans in direct_controlled:
#             is_pmos = 'pmos' in trans.type.lower() if hasattr(trans, 'type') else False

#             if is_pmos:
#                 result['transistors'].append(trans.name)

#                 # transistor의 연결 분석
#                 if trans.source == 'VDD':
#                     # VDD → transistor → output/net
#                     result['paths'].append(('VDD', trans.name))
#                     result['paths'].append((trans.name, trans.drain))
#                     result['descriptions'].append(f"Stage 2 PMOS (direct): VDD → {trans.name} → {trans.drain} (controlled by {intermediate_gate})")
#                 elif trans.drain == 'VDD':
#                     # VDD → transistor → output/net
#                     result['paths'].append(('VDD', trans.name))
#                     result['paths'].append((trans.name, trans.source))
#                     result['descriptions'].append(f"Stage 2 PMOS (direct): VDD → {trans.name} → {trans.source} (controlled by {intermediate_gate})")
#                 elif trans.source not in ['VSS'] and trans.drain in output_nodes:
#                     # intermediate_net → transistor → output
#                     result['paths'].append((trans.source, trans.name))
#                     result['paths'].append((trans.name, trans.drain))
#                     result['descriptions'].append(f"Stage 2 PMOS (to output): {trans.source} → {trans.name} → {trans.drain} (controlled by {intermediate_gate})")
#                 elif trans.drain not in ['VSS'] and trans.source in output_nodes:
#                     # intermediate_net → transistor → output
#                     result['paths'].append((trans.drain, trans.name))
#                     result['paths'].append((trans.name, trans.source))
#                     result['descriptions'].append(f"Stage 2 PMOS (to output): {trans.drain} → {trans.name} → {trans.source} (controlled by {intermediate_gate})")
        
#         # 2. intermediate gate와 연결된 모든 PMOS transistors (parallel/series)
#         # intermediate gate 자체에 연결된 다른 transistor들도 Stage 2에 포함
#         processed_transistors = set([t.name for t in direct_controlled])
        
#         # intermediate gate 자체와 연결된 transistors 찾기
#         for trans in cell.transistors:
#             if (trans.name not in processed_transistors):
#                 is_pmos = 'pmos' in trans.type.lower() if hasattr(trans, 'type') else False
#                 if not is_pmos:
#                     continue
                
#                 # intermediate gate와 연결되어 있는지 확인 (source, drain, 또는 gate)
#                 connected_to_intermediate = False
#                 connection_type = None
                
#                 if trans.source == intermediate_gate:
#                     connected_to_intermediate = True
#                     connection_type = f"{intermediate_gate} → {trans.name} → {trans.drain}"
#                 elif trans.drain == intermediate_gate:
#                     connected_to_intermediate = True  
#                     connection_type = f"{trans.source} → {trans.name} → {intermediate_gate}"
                
#                 if connected_to_intermediate:
#                     result['transistors'].append(trans.name)
                    
#                     # Add appropriate paths based on connection
#                     if trans.source == 'VDD':
#                         result['paths'].append(('VDD', trans.name))
#                         result['paths'].append((trans.name, trans.drain))
#                         result['descriptions'].append(f"Stage 2 PMOS (feeding intermediate): VDD → {trans.name} → {trans.drain} (feeds {intermediate_gate})")
#                     elif trans.drain == 'VDD':
#                         result['paths'].append((trans.source, trans.name))
#                         result['paths'].append((trans.name, 'VDD'))
#                         result['descriptions'].append(f"Stage 2 PMOS (feeding intermediate): {trans.source} → {trans.name} → VDD (feeds {intermediate_gate})")
#                     else:
#                         # Connected to intermediate gate but not to power directly
#                         if trans.source == intermediate_gate:
#                             result['paths'].append((intermediate_gate, trans.name))
#                             result['paths'].append((trans.name, trans.drain))
#                         else:
#                             result['paths'].append((trans.source, trans.name))
#                             result['paths'].append((trans.name, intermediate_gate))
#                         result['descriptions'].append(f"Stage 2 PMOS (intermediate connected): {connection_type} (part of Stage 2 path)")
                    
#                     processed_transistors.add(trans.name)
        
#         return result
    
#     def _analyze_stage2_pull_down_paths(self, cell, intermediate_gate, direct_controlled, output_nodes: List[str]):
#         """
#         Stage 2 pull-down path 분석: intermediate gate 뒤의 VSS → output 경로
#         """
#         result = {
#             'transistors': [],
#             'paths': [],
#             'descriptions': []
#         }

#         # 1. intermediate gate에 의해 직접 제어되는 NMOS transistors
#         for trans in direct_controlled:
#             is_nmos = 'nmos' in trans.type.lower() if hasattr(trans, 'type') else False

#             if is_nmos:
#                 result['transistors'].append(trans.name)

#                 # transistor의 연결 분석
#                 if trans.source == 'VSS':
#                     # VSS → transistor → output/net
#                     result['paths'].append(('VSS', trans.name))
#                     result['paths'].append((trans.name, trans.drain))
#                     result['descriptions'].append(f"Stage 2 NMOS (direct): VSS → {trans.name} → {trans.drain} (controlled by {intermediate_gate})")
#                 elif trans.drain == 'VSS':
#                     # VSS → transistor → output/net
#                     result['paths'].append(('VSS', trans.name))
#                     result['paths'].append((trans.name, trans.source))
#                     result['descriptions'].append(f"Stage 2 NMOS (direct): VSS → {trans.name} → {trans.source} (controlled by {intermediate_gate})")
#                 elif trans.source not in ['VDD'] and trans.drain in output_nodes:
#                     # intermediate_net → transistor → output
#                     result['paths'].append((trans.source, trans.name))
#                     result['paths'].append((trans.name, trans.drain))
#                     result['descriptions'].append(f"Stage 2 NMOS (to output): {trans.source} → {trans.name} → {trans.drain} (controlled by {intermediate_gate})")
#                 elif trans.drain not in ['VDD'] and trans.source in output_nodes:
#                     # intermediate_net → transistor → output
#                     result['paths'].append((trans.drain, trans.name))
#                     result['paths'].append((trans.name, trans.source))
#                     result['descriptions'].append(f"Stage 2 NMOS (to output): {trans.drain} → {trans.name} → {trans.source} (controlled by {intermediate_gate})")
        
#         # 2. Similar logic for parallel NMOS transistors (omitted for brevity)
        
#         return result
    
#     def _trace_series_transistors(self, cell, first_trans, first_net, target_power, paths, transistors, path_type, output_nodes: List[str]):
#         """
#         Series로 연결된 transistor들을 추적하여 intermediate net 없이 transistor-to-transistor 연결 생성
#         """
#         current_net = first_net
#         processed_nets = set()

#         while current_net not in output_nodes and current_net not in processed_nets:
#             processed_nets.add(current_net)
#             next_trans_found = None

#             # 현재 net에 연결된 다음 transistor 찾기
#             for next_trans in cell.transistors:
#                 if (next_trans != first_trans and
#                     next_trans.name not in transistors and
#                     current_net in [next_trans.source, next_trans.drain]):

#                     next_trans_found = next_trans
#                     break

#             if next_trans_found:
#                 # Transistor → next transistor 직접 연결 (intermediate net 생략)
#                 paths.append((first_trans.name, next_trans_found.name))
#                 transistors.append(next_trans_found.name)

#                 # Next transistor의 다른 terminal 찾기
#                 next_other = (next_trans_found.drain if next_trans_found.source == current_net
#                              else next_trans_found.source)

#                 if next_other in output_nodes:
#                     # 최종 output 연결
#                     paths.append((next_trans_found.name, next_other))
#                     print(f"     Series {path_type}: {first_trans.name} → {next_trans_found.name} → {next_other}")
#                     break
#                 else:
#                     print(f"     Series {path_type}: {first_trans.name} → {next_trans_found.name}")
#                     # 다음 iteration을 위해 업데이트
#                     current_net = next_other
#                     first_trans = next_trans_found
#             else:
#                 break


if __name__ == "__main__":
    print("🚀 Delay-Aware Stage-Aware Path Extractor")
    print("=" * 50)
    print("✅ Rise transition: VDD (pull-up) paths only")
    print("✅ Fall transition: VSS (pull-down) paths only")  
    print("✅ Stage information in edge attributes [stage1, stage2, gate_control]")