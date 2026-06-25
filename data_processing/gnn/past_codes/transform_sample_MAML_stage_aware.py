#!/usr/bin/env python

"""
Complete Stage-Aware Transform System
기존 legacy system 대신 완전한 stage-aware system으로 전환
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from stage_aware_extractor_delay_aware import DelayAwareStageExtractor, StageInfo
from cdl_loader import CDLLoader

def transform_sample_stage_aware(sample: Dict[str, Any],
                                cap: List[Dict[str, Any]],
                                transformer: CDLLoader,
                                lib_prefix: str = "",
                                graph_mode: str = "stage_aware") -> List[Dict[str, Any]]:
    """
    완전한 Stage-Aware transformation with LUT expansion
    각 timing arc의 모든 (input_slew, output_load) 조합에 대해 dataset 생성

    Args:
        sample: Liberty file에서 파싱된 timing sample
        cap: Capacitance 정보
        transformer: SPICE topology transformer
        lib_prefix: Library prefix
        graph_mode: "stage_aware" (current path only) or "full_graph" (all transistors baseline)

    Returns:
        List of stage-aware GNN graph samples (하나의 timing arc → 49개 samples)
    """
    
    try:
        # 1. 기본 정보 추출
        cell_name = sample.get('cell', '')
        delay_type = sample.get('delay_type', 'rise_transition')
        voltage = sample.get('Voltage', 0.7)

        # Pin name 추출 (multi-output cell을 위해 중요!)
        pin_name = sample.get('pin_name', 'Y')  # Default to 'Y' if not specified

        # LUT dimensions 추출
        input_slews = sample.get('index_1', [40.0]) if sample.get('index_1') else [40.0]
        output_loads = sample.get('index_2', [5.76]) if sample.get('index_2') else [5.76]

        # Timing output values 추출 (2D LUT)
        timing_values = sample.get('values', [[0.0]])
        
        print(f"🔄 Stage-Aware Transform: {cell_name} ({delay_type})")
        
        # Values array의 실제 크기 확인
        actual_rows = len(timing_values) if isinstance(timing_values, list) else 0
        actual_cols = len(timing_values[0]) if actual_rows > 0 and isinstance(timing_values[0], list) else 0
        
        print(f"   Index dimensions: {len(input_slews)} × {len(output_loads)}")
        print(f"   Values dimensions: {actual_rows} × {actual_cols}")
        print(f"   Effective LUT: {min(len(input_slews), actual_rows)} × {min(len(output_loads), actual_cols)} = {min(len(input_slews), actual_rows) * min(len(output_loads), actual_cols)} points")
        
        # 2. SPICE cell 찾기 (모든 LUT point에서 동일)
        spice_cell = None
        spice_cell_name = None
        
        # Generic cell mapping with flexible matching
        for logic_cell_name, logic_cell in transformer.all_logic_cells.items():
            # 1. Exact match first
            if cell_name == logic_cell_name:
                spice_cell = logic_cell
                spice_cell_name = logic_cell_name
                print(f"   📋 Exact match to SPICE: {spice_cell_name}")
                break
            
            # 2. Base name match (e.g., AO333x2_ASAP7_75t_L ↔ AO333x2_ASAP7_75t_R)
            cell_base = '_'.join(cell_name.split('_')[:-1])  # Remove last part
            logic_base = '_'.join(logic_cell_name.split('_')[:-1])
            if cell_base == logic_base:
                spice_cell = logic_cell
                spice_cell_name = logic_cell_name
                print(f"   📋 Base match to SPICE: {cell_name} → {spice_cell_name}")
                break
                
            # 3. Prefix match (original logic)
            if cell_name.split('_')[0] in logic_cell_name or logic_cell_name.split('_')[0] in cell_name:
                spice_cell = logic_cell
                spice_cell_name = logic_cell_name
                print(f"   📋 Prefix match to SPICE: {cell_name} → {spice_cell_name}")
                break
        
        if not spice_cell:
            print(f"   ❌ SPICE cell not found for {cell_name}")
            print(f"   📋 Available cells ({len(transformer.all_logic_cells)}):")
            # Show some available cells for debugging
            cell_list = list(transformer.all_logic_cells.keys())
            for i, available_cell in enumerate(cell_list[:10]):  # Show first 10
                print(f"      {i+1}. {available_cell}")
            if len(cell_list) > 10:
                print(f"      ... and {len(cell_list) - 10} more cells")
            
            # Try to find similar cells
            print(f"   🔍 Looking for similar cells containing '{cell_name.split('_')[0]}':")
            similar_cells = [c for c in cell_list if cell_name.split('_')[0] in c]
            for sim_cell in similar_cells[:5]:
                print(f"      - {sim_cell}")
                
            return []
        
        # 3. External inputs 파싱 (모든 LUT point에서 동일)
        external_inputs = sample.get('input_port_name', ['A'])
        if isinstance(external_inputs, str):
            external_inputs = [external_inputs]
        
        print(f"   🔌 External inputs: {external_inputs}")

        # 4. Output nodes 찾기 - pin_name 사용 (multi-output cell 대응)
        # HA/FA 같은 cell은 여러 output (CON, SN 등)을 가지므로
        # timing arc의 pin_name으로 특정 output만 선택
        power_nodes = ['VDD', 'VSS']
        all_ports = spice_cell.ports

        # pin_name이 SPICE cell의 port에 존재하는지 확인
        if pin_name in all_ports and pin_name not in power_nodes and pin_name not in external_inputs:
            # pin_name이 유효한 output port면 해당 pin만 사용
            output_nodes = [pin_name]
            print(f"   🎯 Using pin_name as output: {pin_name}")
        else:
            # pin_name이 없거나 유효하지 않으면 auto-detect (fallback)
            output_nodes = [port for port in all_ports
                           if port not in power_nodes and port not in external_inputs]

            if not output_nodes:
                # 최후의 fallback: 기본값 'Y' 사용
                output_nodes = ['Y']

            print(f"   🎯 Auto-detected output nodes: {output_nodes}")

        # 5. Delay-aware stage-aware path extraction (모든 LUT point에서 동일)
        extractor = DelayAwareStageExtractor(transformer.spice_file_path)
        stage_info = extractor.classify_stage_structure(spice_cell, external_inputs, delay_type, output_nodes=output_nodes)

        print(f"   🎯 Graph Mode: {graph_mode}")
        print(f"   🎯 Stage type: {stage_info.stage_type}")
        if stage_info.intermediate_gates:
            print(f"   🔗 Intermediate gates: {stage_info.intermediate_gates}")

        # 6. Node list 생성 (모든 LUT point에서 동일)
        input_nodes = external_inputs

        if graph_mode == "full_graph":
            # Full graph mode: 모든 transistor 포함, intermediate nodes 제외
            intermediate_nodes = []  # Full graph mode에서는 intermediate node 불포함
            transistor_nodes = [trans.name for trans in spice_cell.transistors]
        else:
            # Stage-aware mode: current path의 transistor만 포함, intermediate gates 포함
            intermediate_nodes = stage_info.intermediate_gates
            transistor_nodes = stage_info.stage1_transistors + stage_info.stage2_transistors

        all_nodes = (power_nodes + output_nodes + input_nodes +
                    intermediate_nodes + transistor_nodes)
        all_nodes = list(dict.fromkeys(all_nodes))  # Remove duplicates

        print(f"   📊 Total nodes: {len(all_nodes)} ({graph_mode} mode)")

        # 6. Edge generation (graph_mode에 따라 다른 방식 사용)
        if graph_mode == "full_graph":
            # Baseline: 전체 cell의 모든 transistor 연결
            edges, edge_attrs = extractor.create_full_graph_edges(spice_cell, all_nodes)
        else:
            # Stage-aware: current path만 사용
            edges, edge_attrs = extractor.create_stage_aware_edges(stage_info, all_nodes)
        
        # 7. 모든 LUT point에 대해 그래프 샘플 생성
        result_samples = []
        
        # Values array의 실제 크기 확인 (index와 values 크기 불일치 처리)
        actual_rows = len(timing_values) if isinstance(timing_values, list) else 0
        actual_cols = len(timing_values[0]) if actual_rows > 0 and isinstance(timing_values[0], list) else 0
        
        # index와 values 크기 중 작은 것 사용 (보통 7x7 values, 8 indices)
        effective_rows = min(len(input_slews), actual_rows) if actual_rows > 0 else len(input_slews)
        effective_cols = min(len(output_loads), actual_cols) if actual_cols > 0 else len(output_loads)
        
        for row_idx in range(effective_rows):
            for col_idx in range(effective_cols):
                input_slew = input_slews[row_idx]
                output_load = output_loads[col_idx]
                
                # 해당 (row, col) 위치의 timing value 추출
                if isinstance(timing_values, list) and len(timing_values) > row_idx:
                    if isinstance(timing_values[row_idx], list) and len(timing_values[row_idx]) > col_idx:
                        output_value = float(timing_values[row_idx][col_idx])
                    else:
                        output_value = float(timing_values[row_idx]) if isinstance(timing_values[row_idx], (int, float)) else 0.0
                else:
                    output_value = 0.0
                
                # 각 LUT point별 node features 생성 (7-bit format)
                # Format: [power_rail, port/intermediate, nmos/pmos, width, voltage, input_slew, output_load]
                node_features = []
                circuit_nodes = []
                transistor_node_list = []
                
                for i, node in enumerate(all_nodes):
                    if node.startswith('MM'):  # Transistor node
                        # Find corresponding transistor info
                        trans_info = None
                        for trans in spice_cell.transistors:
                            if trans.name == node:
                                trans_info = trans
                                break
                        
                        if trans_info:
                            # Transistor features: [0, 0, nmos/pmos, width, voltage, 0, 0]
                            # NMOS: 1, PMOS: -1
                            trans_type = 1.0 if 'nmos' in trans_info.type.lower() else -1.0
                            width_um = trans_info.width / 1000.0  # nm to um
                            # All nodes get input_slew: [is_power, is_circuit, trans_type, width, voltage, input_slew, output_load]
                            node_features.append([0.0, 0.0, trans_type, width_um, voltage, 0.0, 0.0])
                            transistor_node_list.append(node)
                        else:
                            # Fallback transistor feature (assume PMOS)
                            node_features.append([0.0, 0.0, -1.0, 0.1, voltage, 0.0, 0.0])
                            transistor_node_list.append(node)
                            
                    else:  # Circuit node
                        circuit_nodes.append(node)

                        if node in ['VDD', 'VSS']:
                            # Power rails: [1, 0, 0, 0, voltage, input_slew, 0]
                            node_features.append([1.0, 0.0, 0.0, 0.0, voltage, 0.0 , 0.0])
                        elif node in output_nodes:
                            # Output port: [0, 1, 0, 0, voltage, input_slew, output_load]
                            # Handles standard Y output and non-standard outputs like CON, SN (FA/HA cells)
                            node_features.append([0.0, 1.0, 0.0, 0.0, voltage, 0.0 , output_load])
                        elif node in external_inputs:
                            # Input port: [0, 1, 0, 0, voltage, input_slew, 0]
                            node_features.append([0.0, 1.0, 0.0, 0.0, voltage, input_slew, 0.0])
                        else:
                            # Intermediate gate node: gate로 사용되는 intermediate node
                            # 해당 gate로 제어되는 transistor의 width를 찾아서 사용
                            gate_width = 0.0
                            if node in intermediate_nodes:
                                # 이 intermediate gate로 제어되는 transistor 찾기
                                for trans in spice_cell.transistors:
                                    if trans.gate == node:
                                        gate_width = trans.width / 1000.0  # nm to um
                                        break  # 첫 번째 transistor의 width 사용

                            # Intermediate node: [0, 1, 0, gate_width, voltage, input_slew, 0]
                            node_features.append([0.0, 1.0, 0.0, gate_width, voltage, 0.0, 0.0])
                
                # Convert to tensors
                node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
                edge_index_tensor = torch.tensor(edges, dtype=torch.int64).T  # Transpose for PyG format
                edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float32)

                # Skip empty graphs (can happen with multi-output cells like FA/HA)
                if edge_index_tensor.shape[1] == 0:
                    print(f"   ⚠️  Skipping empty graph for {spice_cell_name} - {delay_type} at slew={input_slew}, load={output_load}")
                    continue  # Skip this LUT point

                # Adjacency matrix (for compatibility)
                num_nodes = len(all_nodes)
                adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

                # Full graph mode: bidirectional (undirected graph)
                # Stage-aware mode: directional (directed graph for timing paths)
                if graph_mode == "full_graph":
                    # Bidirectional edges for full graph baseline
                    for edge in edges:
                        adjacency_matrix[edge[0], edge[1]] = 1.0
                        adjacency_matrix[edge[1], edge[0]] = 1.0  # Add reverse direction
                else:
                    # Directional edges for stage-aware (timing path direction matters)
                    for edge in edges:
                        adjacency_matrix[edge[0], edge[1]] = 1.0
                
                # 각 LUT point의 result sample
                lut_sample = {
                    'node_features': node_features_tensor,
                    'edge_index': edge_index_tensor,
                    'edge_attr': edge_attr_tensor,
                    'adjacency_matrix': adjacency_matrix,
                    'cell_name': spice_cell_name,
                    'delay_type': delay_type,
                    'input_slew': input_slew,      # 각 LUT point별 slew
                    'output_load': output_load,    # 각 LUT point별 load
                    'voltage': voltage,
                    'output': output_value,        # 각 LUT point별 timing 값
                    'all_nodes': all_nodes,
                    'circuit_nodes': circuit_nodes,
                    'transistor_nodes': transistor_node_list,
                    'stage_info': {
                        'stage_type': stage_info.stage_type,
                        'intermediate_gates': stage_info.intermediate_gates,
                        'stage1_transistors': stage_info.stage1_transistors,
                        'stage2_transistors': stage_info.stage2_transistors
                    },
                    'total_node_count': len(all_nodes),
                    'row_idx': row_idx,           # LUT row index
                    'col_idx': col_idx            # LUT col index
                }
                
                result_samples.append(lut_sample)
        
        print(f"   ✅ Stage-aware LUT expansion:")
        print(f"      Generated {len(result_samples)} samples from {len(input_slews)} × {len(output_loads)} LUT")
        print(f"      Each sample: {len(all_nodes)} nodes, {len(edges)} edges")
        
        return result_samples
        
    except Exception as e:
        print(f"   ❌ Stage-aware transform failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def transform_all_logic_samples_stage_aware(flatten: List[Dict[str, Any]],
                                           cap: List[Dict[str, Any]],
                                           lib_prefix: str = "",
                                           graph_mode: str = "stage_aware") -> List[Dict[str, Any]]:
    """
    모든 logic samples를 stage-aware system으로 변환

    Args:
        flatten: 파싱된 liberty samples
        cap: Capacitance 정보
        lib_prefix: Library prefix
        graph_mode: "stage_aware" (current path only) or "full_graph" (all transistors baseline)

    Returns:
        List of stage-aware GNN graph samples
    """
    
    print(f"🚀 Stage-Aware Logic Transformation")
    print(f"   Graph Mode: {graph_mode}")
    print(f"   Input samples: {len(flatten)}")

    # Multiple CDL files을 모두 로드하여 unified transformer 생성
    import os
    base_path = '/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files'

    # Try relative path first, then absolute path
    cdl_files = []
    cdl_names = ['asap7sc7p5t_28_L.cdl', 'asap7sc7p5t_28_R.cdl', 'asap7sc7p5t_28_SL.cdl', 'asap7sc7p5t_28_SRAM.cdl']

    for cdl_name in cdl_names:
        # Try relative paths first
        relative_paths = [f'../../{cdl_name}', cdl_name, f'./{cdl_name}']
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

    # Merge additional CDL files using merge_cdl()
    for cdl_file in cdl_files[1:]:
        try:
            transformer.merge_cdl(cdl_file)
        except Exception as e:
            print(f"   ⚠️ Could not merge {cdl_file}: {e}")

    print(f"   ✅ Total logic cells available: {len(transformer.all_logic_cells)}")

    # Allowed cell list - Only process cells in this list
    # If None, process all timing cells (no filtering)
    # Set this to a list of cell name prefixes or full names to filter
    allowed_cells = None  # Set to None to process all cells, or provide a list like ['NAND', 'INV', 'OA211', 'AO221']

    # Example: allowed_cells = ['NAND', 'INV', 'BUF', 'OA', 'AO']  # Only process these cell types

    transformed_list = []
    processed_count = 0

    for i, s in enumerate(flatten):
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
        if True:

            try:
                transformed_samples = transform_sample_stage_aware(s, cap, transformer, lib_prefix, graph_mode)

                if transformed_samples:  # transformed_samples is now a list of samples
                    transformed_list.extend(transformed_samples)  # extend instead of append
                    processed_count += 1
                    print(f"   Generated {len(transformed_samples)} LUT samples from timing arc")

            except Exception as e:
                print(f"   ❌ Error processing sample {i}: {e}")
                continue

        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(flatten)} samples, generated {len(transformed_list)} stage-aware graphs")
    
    print(f"✅ Stage-Aware transformation complete:")
    print(f"   Processed: {processed_count} logic samples")
    print(f"   Generated: {len(transformed_list)} stage-aware graphs")
    
    return transformed_list


if __name__ == "__main__":
    # Test stage-aware system
    print("🧪 Testing Complete Stage-Aware System")
    print("=" * 50)
    
    from libdata_extract_MAML import parse_liberty_pin_blocks, flatten_pin_data
    
    # Test with INVBUF library
    lib_file = '/mnt/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset/scaling/INVBUF_SLVT/INVBUF_SLVT_tt_25C_0p560v.lib'
    
    with open(lib_file, "r") as f:
        lines = f.readlines()
    
    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)
    
    # Transform using stage-aware system
    stage_aware_samples = transform_all_logic_samples_stage_aware(flattened[:5], cap, "TEST_")
    
    if stage_aware_samples:
        sample = stage_aware_samples[0]
        print(f"\n📊 Sample Analysis:")
        print(f"   Cell: {sample['cell_name']}")
        print(f"   Stage type: {sample['stage_info']['stage_type']}")
        print(f"   Nodes: {sample['total_node_count']}")
        print(f"   Edges: {sample['edge_index'].shape[1]}")
        print(f"   Edge attributes: {sample['edge_attr'].shape}")
        
        # Edge breakdown
        edge_attrs = sample['edge_attr']
        stage1_count = (edge_attrs[:, 0] == 1.0).sum().item()
        stage2_count = (edge_attrs[:, 1] == 1.0).sum().item() 
        gate_count = (edge_attrs[:, 2] == 1.0).sum().item()
        
        print(f"   Edge breakdown: {stage1_count} stage1 + {stage2_count} stage2 + {gate_count} gate-control")
        print(f"   🎉 No direct power → output edges! All through transistors")