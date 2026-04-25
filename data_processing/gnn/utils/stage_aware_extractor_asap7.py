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
# ASAP7 Stage-Aware Extractor Classes
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


class ASAP7StageAwareExtractor:
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


if __name__ == "__main__":
    print("🚀 Delay-Aware Stage-Aware Path Extractor")
    print("=" * 50)
    print("✅ Rise transition: VDD (pull-up) paths only")
    print("✅ Fall transition: VSS (pull-down) paths only")  
    print("✅ Stage information in edge attributes [stage1, stage2, gate_control]")