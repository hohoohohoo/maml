#!/usr/bin/env python

"""
Complete Current Path Extractor - Fixed Version
PMOS/NMOS path를 따라가서 complete current path 구성 (power rail에서만 시작)
"""

from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Dict
import re

@dataclass
class CompleteStageInfo:
    """Complete stage 정보"""
    stage_type: str
    all_intermediate_nodes: List[str]  # 모든 intermediate nodes
    stage1_current_paths: List[Tuple[str, str]]  # Complete current paths for stage 1
    stage2_current_paths: List[Tuple[str, str]]  # Complete current paths for stage 2
    stage1_transistors: List[str]
    stage2_transistors: List[str]

class CompleteCurrentPathExtractor:
    """Complete current path extraction using PMOS/NMOS tracing"""
    
    def __init__(self, spice_file_path: str):
        self.spice_file_path = spice_file_path
    
    def find_intermediate_gate_nodes(self, cell, external_inputs: List[str]) -> List[str]:
        """
        Intermediate gate node 찾기: gate로 사용되지만 input pin이 아닌 node
        이것만이 진짜 intermediate node이고 stage 구분의 기준이 됨
        """
        # 모든 gate nets 수집
        all_gate_nets = set(t.gate for t in cell.transistors)
        
        # Power rails, output, external inputs 제외
        power_nets = {'VDD', 'VSS'}
        output_nets = {'Y', 'Z', 'Q', 'OUT', 'O'}
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
    
    def extract_complete_current_paths(self, cell, external_inputs: List[str], delay_type: str) -> CompleteStageInfo:
        """Complete current path extraction based on intermediate gate nodes"""
        
        # 1. Intermediate gate nodes만 찾기 (stage 구분의 기준)
        intermediate_gates = self.find_intermediate_gate_nodes(cell, external_inputs)
        print(f"🔍 Intermediate gate nodes (stage dividers): {intermediate_gates}")
        
        if not intermediate_gates:
            # One-stage logic
            return self._extract_one_stage_paths(cell, delay_type)
        
        # 2. Two-stage logic: current path 구성 (intermediate gate 기준)
        stage1_paths = []
        stage2_paths = []
        stage1_transistors = []
        stage2_transistors = []
        
        print(f"📊 Two-stage logic detected with intermediate gates: {intermediate_gates}")
        
        if 'rise' in delay_type:
            # Rise: Stage1(pull-down to intermediate gate) → Stage2(pull-up to Y)
            print("   Rise timing: Stage1(NMOS → intermediate gate) → Stage2(PMOS → Y)")
            
            # Stage 1: VSS에서 intermediate gates까지 NMOS path (생성하는 경로)
            stage1_nmos_paths = self.trace_nmos_paths(cell, 'VSS', set(intermediate_gates))
            print(f"   Stage 1 NMOS paths (to intermediate): {len(stage1_nmos_paths)} found")
            
            # Stage 2: VDD에서 Y까지 PMOS path만 추적 (power rail에서 시작하는 complete path만)
            stage2_pmos_paths = self.trace_pmos_paths(cell, 'VDD', {'Y'})
            print(f"   Stage 2 PMOS paths (to Y): {len(stage2_pmos_paths)} found")
            
            # Convert paths to edges
            stage1_paths, stage1_transistors = self._paths_to_edges(stage1_nmos_paths, "Stage1 NMOS")
            stage2_paths, stage2_transistors = self._paths_to_edges(stage2_pmos_paths, "Stage2 PMOS")
            
        else:  # fall
            # Fall: Stage1(pull-up to intermediate gate) → Stage2(pull-down to Y)
            print("   Fall timing: Stage1(PMOS → intermediate gate) → Stage2(NMOS → Y)")
            
            # Stage 1: VDD에서 intermediate gates까지 PMOS path
            stage1_pmos_paths = self.trace_pmos_paths(cell, 'VDD', set(intermediate_gates))
            print(f"   Stage 1 PMOS paths (to intermediate): {len(stage1_pmos_paths)} found")
            
            # Stage 2: VSS에서 Y까지 NMOS path만 추적 (power rail에서 시작하는 complete path만)
            stage2_nmos_paths = self.trace_nmos_paths(cell, 'VSS', {'Y'})
            print(f"   Stage 2 NMOS paths (to Y): {len(stage2_nmos_paths)} found")
            
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
    
    def _extract_one_stage_paths(self, cell, delay_type: str) -> CompleteStageInfo:
        """One-stage logic path extraction"""
        
        if 'rise' in delay_type:
            # Direct VDD → Y pull-up paths
            pmos_paths = self.trace_pmos_paths(cell, 'VDD', {'Y'})
            stage1_paths, stage1_transistors = self._paths_to_edges(pmos_paths, "One-stage PMOS")
            stage2_paths, stage2_transistors = [], []
        else:
            # Direct VSS → Y pull-down paths  
            nmos_paths = self.trace_nmos_paths(cell, 'VSS', {'Y'})
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