#!/usr/bin/env python

"""
Stage-Aware Path Extractor
One-stage vs Two-stage 구분하여 path 추출하는 범용적인 시스템
"""

from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
import re

@dataclass
class StageInfo:
    """Stage 정보를 담는 클래스"""
    stage_type: str  # "one_stage" or "two_stage"
    intermediate_gates: List[str]  # 중간 gate nodes
    stage1_paths: List[Tuple[str, str]]  # (source, destination) for first stage
    stage2_paths: List[Tuple[str, str]]  # (source, destination) for second stage
    stage1_transistors: List[str]  # First stage controlled transistors
    stage2_transistors: List[str]  # Second stage controlled transistors

class StageAwarePathExtractor:
    """
    Stage-aware path extraction system
    
    Logic:
    1. Detect intermediate gate nodes (not VDD/VSS/Y/external_inputs)
    2. If intermediate gates exist → Two-stage
    3. If no intermediate gates → One-stage
    
    One-stage: Direct paths VDD/VSS → Y
    Two-stage: First stage (VDD/VSS → intermediate) + Second stage (VDD/VSS → Y via intermediate gate)
    """
    
    def __init__(self, spice_file_path: str):
        self.spice_file_path = spice_file_path
        
    def classify_stage_structure(self, cell, external_inputs: List[str], delay_type: str = "rise_transition") -> StageInfo:
        """
        Cell을 one-stage 또는 two-stage로 분류하고 path 정보 추출
        
        Args:
            cell: SPICE cell object with transistors
            external_inputs: 외부 입력 포트 리스트 (예: ['A', 'B'])
            
        Returns:
            StageInfo: 분류 결과와 path 정보
        """
        
        # 1. 모든 gate nets 수집
        all_gate_nets = set(t.gate for t in cell.transistors)
        
        # 2. Intermediate gate nodes 감지
        power_nets = {'VDD', 'VSS'}
        output_nets = {'Y', 'Z', 'Q', 'OUT', 'O'}  # 일반적인 output names
        external_nets = set(external_inputs)
        
        intermediate_gates = all_gate_nets - power_nets - output_nets - external_nets
        intermediate_gates = [gate for gate in intermediate_gates if gate]  # 빈 문자열 제거
        
        print(f"🔍 Stage Classification:")
        print(f"   All gate nets: {sorted(all_gate_nets)}")
        print(f"   External inputs: {sorted(external_nets)}")
        print(f"   Power/output nets: {sorted(power_nets | output_nets)}")
        print(f"   🎯 Intermediate gates: {intermediate_gates}")
        
        # 3. Stage 분류 (delay_type에 따른 path 필터링)
        if intermediate_gates:
            return self._analyze_two_stage_structure(cell, external_inputs, intermediate_gates, delay_type)
        else:
            return self._analyze_one_stage_structure(cell, external_inputs, delay_type)
    
    def _analyze_one_stage_structure(self, cell, external_inputs: List[str], delay_type: str) -> StageInfo:
        """One-stage structure 분석: Direct VDD/VSS → Y paths"""
        
        print(f"   📊 ONE-STAGE structure detected")
        
        stage1_paths = []
        stage1_transistors = []
        
        # Find all transistors (including series connections)
        for trans in cell.transistors:
            # Check for direct power rail connections
            if trans.source in ['VDD', 'VSS'] or trans.drain in ['VDD', 'VSS']:
                power_rail = 'VDD' if 'VDD' in [trans.source, trans.drain] else 'VSS'
                other_terminal = trans.drain if trans.source == power_rail else trans.source
                
                # Add power → transistor connection
                stage1_paths.append((power_rail, trans.name))
                stage1_transistors.append(trans.name)
                
                path_type = "pull-up" if power_rail == 'VDD' else "pull-down"
                
                if other_terminal == 'Y':
                    # Direct connection: power → transistor → Y
                    stage1_paths.append((trans.name, 'Y'))
                    print(f"     Direct {path_type}: {power_rail} → {trans.name} → Y")
                else:
                    # Series connection: power → transistor → intermediate net
                    stage1_paths.append((trans.name, other_terminal))
                    print(f"     Series {path_type}: {power_rail} → {trans.name} → {other_terminal}")
                    
                    # Find next transistor in series (connected via intermediate net)
                    for next_trans in cell.transistors:
                        if (next_trans != trans and 
                            other_terminal in [next_trans.source, next_trans.drain] and
                            next_trans.name not in stage1_transistors):
                            
                            next_other = next_trans.drain if next_trans.source == other_terminal else next_trans.source
                            stage1_paths.append((other_terminal, next_trans.name))
                            stage1_paths.append((next_trans.name, next_other))
                            stage1_transistors.append(next_trans.name)
                            
                            print(f"     Series continuation: {other_terminal} → {next_trans.name} → {next_other}")
        
        return StageInfo(
            stage_type="one_stage",
            intermediate_gates=[],
            stage1_paths=stage1_paths,
            stage2_paths=[],
            stage1_transistors=stage1_transistors,
            stage2_transistors=[]
        )
    
    def _analyze_two_stage_structure(self, cell, external_inputs: List[str], intermediate_gates: List[str], delay_type: str) -> StageInfo:
        """Two-stage structure 분석"""
        
        print(f"   📊 TWO-STAGE structure detected")
        print(f"   🎯 Intermediate gate(s): {intermediate_gates}")
        
        stage1_paths = []
        stage2_paths = []
        stage1_transistors = []
        stage2_transistors = []
        
        # Stage 1: VDD/VSS → intermediate gate nodes
        print(f"   🔸 Stage 1 Analysis (VDD/VSS → intermediate gates):")
        for intermediate_gate in intermediate_gates:
            # 이 intermediate gate을 생성하는 transistor들 찾기
            for trans in cell.transistors:
                if intermediate_gate in [trans.source, trans.drain]:
                    # 이 transistor가 intermediate gate을 생성
                    other_terminal = trans.drain if trans.source == intermediate_gate else trans.source
                    
                    if other_terminal in ['VDD', 'VSS']:
                        # Power → transistor → intermediate gate path (through transistor)
                        stage1_paths.append((other_terminal, trans.name))  # Power → transistor
                        stage1_paths.append((trans.name, intermediate_gate))  # transistor → intermediate gate
                        stage1_transistors.append(trans.name)
                        
                        path_type = "pull-up" if other_terminal == 'VDD' else "pull-down"
                        control_gate = trans.gate
                        print(f"     Stage 1 {path_type}: {other_terminal} → {trans.name} → {intermediate_gate} (controlled by {control_gate})")
        
        # Stage 2: intermediate gates control transistors → Y  
        print(f"   🔸 Stage 2 Analysis (intermediate gates → Y):")
        for intermediate_gate in intermediate_gates:
            # 이 intermediate gate가 제어하는 transistor들 찾기
            for trans in cell.transistors:
                if trans.gate == intermediate_gate:
                    # 이 transistor는 intermediate gate에 의해 제어됨
                    stage2_transistors.append(trans.name)
                    
                    # 이 transistor의 power connection 찾기
                    if trans.source in ['VDD', 'VSS']:
                        power_source = trans.source
                        output_dest = trans.drain
                    elif trans.drain in ['VDD', 'VSS']:
                        power_source = trans.drain  
                        output_dest = trans.source
                    else:
                        continue  # Power connection이 없으면 skip
                    
                    # Stage 2: Power → transistor → output (controlled by intermediate gate)
                    stage2_paths.append((power_source, trans.name))  # Power → transistor
                    stage2_paths.append((trans.name, output_dest))    # transistor → output
                    
                    path_type = "pull-up" if power_source == 'VDD' else "pull-down"
                    print(f"     Stage 2 {path_type}: {power_source} → {trans.name} → {output_dest} (controlled by {intermediate_gate})")
        
        return StageInfo(
            stage_type="two_stage", 
            intermediate_gates=intermediate_gates,
            stage1_paths=stage1_paths,
            stage2_paths=stage2_paths,
            stage1_transistors=stage1_transistors,
            stage2_transistors=stage2_transistors
        )
    
    def create_stage_aware_edges(self, stage_info: StageInfo, all_nodes: List[str]) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Stage 정보를 바탕으로 edge_index와 edge_attr 생성
        
        Edge attributes:
        - [1,0,0]: Stage 1 source-drain connection
        - [0,1,0]: Stage 2 source-drain connection  
        - [0,0,1]: Gate control connection (stage transition)
        """
        
        edges = []
        edge_attrs = []
        
        # Node index mapping
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        print(f"🔗 Creating stage-aware edges:")
        
        if stage_info.stage_type == "one_stage":
            # One-stage: 모든 path를 stage 1로 처리
            for src, dst in stage_info.stage1_paths:
                if src in node_to_idx and dst in node_to_idx:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edges.append([src_idx, dst_idx])
                    edge_attrs.append([1.0, 0.0, 0.0])  # Stage 1 source-drain
                    print(f"   One-stage: {src}[{src_idx}] → {dst}[{dst_idx}] [1,0,0]")
        
        else:  # two_stage
            # Stage 1 paths
            for src, dst in stage_info.stage1_paths:
                if src in node_to_idx and dst in node_to_idx:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edges.append([src_idx, dst_idx])
                    edge_attrs.append([1.0, 0.0, 0.0])  # Stage 1 source-drain
                    print(f"   Stage 1: {src}[{src_idx}] → {dst}[{dst_idx}] [1,0,0]")
            
            # Gate control connections (stage transition)
            for intermediate_gate in stage_info.intermediate_gates:
                if intermediate_gate in node_to_idx:
                    gate_idx = node_to_idx[intermediate_gate]
                    
                    # intermediate gate → controlled transistors
                    for trans_name in stage_info.stage2_transistors:
                        if trans_name in node_to_idx:
                            trans_idx = node_to_idx[trans_name]
                            edges.append([gate_idx, trans_idx])
                            edge_attrs.append([0.0, 0.0, 1.0])  # Gate control
                            print(f"   Gate control: {intermediate_gate}[{gate_idx}] → {trans_name}[{trans_idx}] [0,0,1]")
            
            # Stage 2 paths  
            for src, dst in stage_info.stage2_paths:
                if src in node_to_idx and dst in node_to_idx:
                    src_idx = node_to_idx[src]
                    dst_idx = node_to_idx[dst]
                    edges.append([src_idx, dst_idx])
                    edge_attrs.append([0.0, 1.0, 0.0])  # Stage 2 source-drain
                    print(f"   Stage 2: {src}[{src_idx}] → {dst}[{dst_idx}] [0,1,0]")
        
        print(f"   📊 Total edges: {len(edges)} ({len([a for a in edge_attrs if a[0] == 1.0])} stage1, "
              f"{len([a for a in edge_attrs if a[1] == 1.0])} stage2, "
              f"{len([a for a in edge_attrs if a[2] == 1.0])} gate-control)")
        
        return edges, edge_attrs


if __name__ == "__main__":
    print("🚀 Stage-Aware Path Extractor")
    print("=" * 40)
    print("✅ One-stage: Direct VDD/VSS → Y paths")  
    print("✅ Two-stage: Stage1 (VDD/VSS → intermediate) + Stage2 (VDD/VSS → Y via gate)")
    print("✅ Stage information in edge attributes [stage1, stage2, gate_control]")