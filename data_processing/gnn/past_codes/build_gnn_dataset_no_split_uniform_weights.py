# build_gnn_dataset_no_split_uniform_weights.py
import torch
from pathlib import Path
from datasets import libdata
from libdata_extract_MAML import parse_liberty_pin_blocks, flatten_pin_data
from transform_sample_MAML_stage_aware import transform_all_logic_samples_stage_aware
import sys
import os

def create_uniform_adjacency_matrix(graph_sample):
    """
    모든 adjacency matrix 값을 1.0으로 설정 (uniform weights)
    Stage 정보는 유지하지만 모든 연결의 weight는 1.0으로 통일
    """
    import torch
    
    # 기본 adjacency matrix
    original_adj = graph_sample.get('adjacency_matrix')
    if original_adj is None:
        num_nodes = graph_sample['node_features'].shape[0]
        original_adj = torch.eye(num_nodes, dtype=torch.float32)
    
    if not isinstance(original_adj, torch.Tensor):
        original_adj = torch.tensor(original_adj, dtype=torch.float32)
    
    # Stage 정보 추출
    all_nodes = graph_sample.get('all_nodes', [])
    stage_info = graph_sample.get('stage_info', {})
    delay_type = graph_sample.get('delay_type', 'rise_transition')
    
    if not all_nodes or not stage_info:
        print(f"   No stage info available, using uniform adjacency matrix")
        # 모든 값을 1.0으로 변경 (0인 부분은 그대로 유지)
        uniform_adj = torch.where(original_adj > 0, torch.ones_like(original_adj), original_adj)
        return uniform_adj
    
    stage_type = stage_info.get('stage_type', 'one_stage')
    intermediate_gates = stage_info.get('intermediate_gates', [])
    
    print(f"   Creating uniform-weight adjacency matrix for {stage_type}")
    
    # Enhanced adjacency matrix 시작 (original 복사)
    enhanced_adj = original_adj.clone()
    
    if stage_type == "two_stage" and intermediate_gates:
        print(f"   Adding intermediate paths for gates: {intermediate_gates}")
        
        # Node name to index mapping
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        # Power rail과 output indices
        vdd_idx = node_to_idx.get('VDD', -1)
        vss_idx = node_to_idx.get('VSS', -1)
        y_idx = node_to_idx.get('Y', -1)
        
        # Intermediate gate indices
        intermediate_indices = []
        for gate in intermediate_gates:
            if gate in node_to_idx:
                intermediate_indices.append(node_to_idx[gate])
        
        print(f"   VDD: {vdd_idx}, VSS: {vss_idx}, Y: {y_idx}")
        print(f"   Intermediate indices: {intermediate_indices}")
        
        # Stage-aware 연결 추가 (모든 weight를 1.0으로)
        for inter_idx in intermediate_indices:
            if 'rise' in delay_type:
                # Rise: Stage1(VSS → intermediate) → Stage2(VDD → Y, intermediate → VDD)
                # Stage1: VSS → intermediate 경로
                if vss_idx >= 0:
                    enhanced_adj[vss_idx, inter_idx] = 1.0  # 원래 2.0 → 1.0
                
                # Stage2: VDD → Y 및 intermediate → VDD 경로  
                if vdd_idx >= 0 and y_idx >= 0:
                    enhanced_adj[vdd_idx, y_idx] = 1.0      # 원래 2.0 → 1.0
                if vdd_idx >= 0:
                    enhanced_adj[inter_idx, vdd_idx] = 1.0  # 원래 2.0 → 1.0
                
                # intermediate → Y 연결
                if y_idx >= 0:
                    enhanced_adj[inter_idx, y_idx] = 1.0   # 원래 1.8 → 1.0
                    
                print(f"   Rise paths: VSS({vss_idx}) → {inter_idx}, {inter_idx} → VDD({vdd_idx}), VDD → Y({y_idx}) (all weights = 1.0)")
                
            else:  # fall_transition
                # Fall: Stage1(VDD → intermediate) → Stage2(VSS → Y, intermediate → VSS)
                # Stage1: VDD → intermediate 경로
                if vdd_idx >= 0:
                    enhanced_adj[vdd_idx, inter_idx] = 1.0  # 원래 2.0 → 1.0
                
                # Stage2: VSS → Y 및 intermediate → VSS 경로
                if vss_idx >= 0 and y_idx >= 0:
                    enhanced_adj[vss_idx, y_idx] = 1.0      # 원래 2.0 → 1.0
                if vss_idx >= 0:
                    enhanced_adj[inter_idx, vss_idx] = 1.0  # 원래 2.0 → 1.0
                
                # intermediate → Y 연결
                if y_idx >= 0:
                    enhanced_adj[inter_idx, y_idx] = 1.0   # 원래 1.8 → 1.0
                    
                print(f"   Fall paths: VDD({vdd_idx}) → {inter_idx}, {inter_idx} → VSS({vss_idx}), VSS → Y({y_idx}) (all weights = 1.0)")
        
        # 모든 intermediate간 연결 (multi-gate의 경우)
        for i, idx1 in enumerate(intermediate_indices):
            for j, idx2 in enumerate(intermediate_indices):
                if i != j:
                    enhanced_adj[idx1, idx2] = 1.0  # 원래 1.2 → 1.0
    
    else:
        print(f"   One-stage logic, using uniform adjacency matrix")
    
    # 최종적으로 모든 0이 아닌 값을 1.0으로 통일
    uniform_adj = torch.where(enhanced_adj > 0, torch.ones_like(enhanced_adj), enhanced_adj)
    
    return uniform_adj

def dataextract_gnn(text, lib_prefix=""):
    """Extract GNN graph data from .lib file"""
    with open(text, "r") as f:
        lines = f.readlines()
    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)
    
    # GNN 그래프 변환 사용 (stage-aware system)
    graph_dataset = transform_all_logic_samples_stage_aware(flattened, cap, lib_prefix)
    return graph_dataset

def data_management_gnn(text, lib_prefix=""):
    """Process GNN graph data from .lib file"""
    from pathlib import Path
    file_path = Path(text)
    graph_dataset = dataextract_gnn(file_path, lib_prefix)
    return graph_dataset
    
def build_all_gnn_data(start=40, end=101, prefix="OA_LVT_2_25_", 
                       save_input="graph_input_data_uniform.pth", save_output="graph_output_data_uniform.pth", 
                       data_dir="OA_LVT"):
    """Build GNN graph dataset from multiple .lib files with uniform weights (all weights = 1.0)"""
    graph_data_per_file = []
    output_data_per_file = []

    for i in range(start, end):
        v_str = f"{i//100}{i%100:02d}"
        filename = f"{data_dir}/{prefix}{v_str}.lib"
        print(f"📥 Processing {filename}")
        
        try:
            graph_dataset = data_management_gnn(filename, prefix)
            
            if graph_dataset:
                print(f"   Generated {len(graph_dataset)} graph samples")
                
                # 현재 lib file의 모든 graph data와 output을 리스트로 수집
                current_file_graphs = []
                current_file_outputs = []
                
                for graph_sample in graph_dataset:
                    # 각 그래프 샘플에는 node_features, edge_index, output 등이 포함
                    
                    # Uniform adjacency matrix 생성 (모든 weight = 1.0)
                    uniform_adj_matrix = create_uniform_adjacency_matrix(graph_sample)
                    
                    graph_data = {
                        'node_features': graph_sample['node_features'],
                        'edge_index': graph_sample['edge_index'],
                        'edge_attr': graph_sample.get('edge_attr', torch.ones(graph_sample['edge_index'].shape[1], dtype=torch.float32)),
                        'adjacency_matrix': uniform_adj_matrix,  # 모든 weight가 1.0인 adjacency matrix
                        'cell_name': graph_sample['cell_name'],
                        'delay_type': graph_sample['delay_type'],
                        'input_slew': graph_sample['input_slew'],
                        'output_load': graph_sample['output_load'],
                        'voltage': graph_sample['voltage'],
                        'row_idx': graph_sample.get('row_idx', 0),
                        'col_idx': graph_sample.get('col_idx', 0),
                        # Stage 정보 보존
                        'all_nodes': graph_sample.get('all_nodes', []),
                        'circuit_nodes': graph_sample.get('circuit_nodes', []),
                        'transistor_nodes': graph_sample.get('transistor_nodes', []),
                        'stage_info': graph_sample.get('stage_info', {})
                    }
                    output_value = graph_sample['output']
                    
                    current_file_graphs.append(graph_data)
                    current_file_outputs.append(output_value)
                
                # 이 lib file의 데이터를 전체에 추가
                graph_data_per_file.append(current_file_graphs)
                output_data_per_file.append(torch.tensor(current_file_outputs, dtype=torch.float32))
                print(f"   File {i}: {len(current_file_graphs)} samples (uniform weights)")
                
            else:
                print(f"   No valid graph data found in {filename}")
                
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

    # Stack output data across lib files: [samples_per_file, num_lib_files]
    if output_data_per_file:
        stacked_outputs = torch.stack(output_data_per_file, dim=1)  # [samples, lib_files]
        print(f"🔄 Finalizing uniform-weight dataset...")
        print(f"   Output tensor shape: {stacked_outputs.shape}")  # [samples_per_file, num_lib_files]
        print(f"   Graph data structure: [samples_per_file][lib_file_idx] (uniform weights)")
        
        # 저장할 데이터 구조
        final_data = {
            'graph_data_per_file': graph_data_per_file,  # [samples_per_file][lib_file_idx][graph_data]
            'stacked_outputs': stacked_outputs,          # [samples_per_file, num_lib_files]
            'num_samples_per_file': len(graph_data_per_file[0]) if graph_data_per_file else 0,
            'num_lib_files': len(graph_data_per_file),
            'adjacency_weights': 'uniform'  # 모든 weight가 1.0임을 표시
        }
        
        torch.save(final_data, save_input)
        print(f"✅ Uniform-weight GNN dataset saved: {save_input}")
        return final_data
    else:
        print("❌ No valid graph data found!")
        return None

def organize_gnn_dataset_no_split(graph_input_path, output_dir):
    """Save all GNN graph data without train/test split (uniform weights)"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("📊 Loading uniform-weight GNN graph dataset...")
    try:
        stacked_data = torch.load(graph_input_path, weights_only=False)
    except Exception as e:
        print(f"❌ Error loading graph data from {graph_input_path}: {e}")
        return None

    graph_data_per_file = stacked_data['graph_data_per_file']  # [samples_per_file][lib_file_idx]
    stacked_outputs = stacked_data['stacked_outputs']          # [samples_per_file, num_lib_files]
    num_samples = stacked_data['num_samples_per_file']
    num_lib_files = stacked_data['num_lib_files']
    
    print(f"   Samples per file: {num_samples}")
    print(f"   Number of lib files: {num_lib_files}")
    print(f"   Output tensor shape: {stacked_outputs.shape}")
    print(f"   Adjacency weights: uniform (all = 1.0)")
    
    # 그래프 데이터를 저장할 디렉토리 생성
    graph_data_dir = Path(output_dir) / "graph_data"
    graph_data_dir.mkdir(exist_ok=True)
    
    # 분할하지 않고 모든 데이터를 그대로 저장
    print("   📁 Saving all uniform-weight data without train/test split...")
    
    # 저장 경로
    all_input_path = graph_data_dir / "all_graph_data_uniform.pth"
    
    # 전체 데이터 구조 (분할하지 않음)
    all_data = {
        'graph_data_per_file': graph_data_per_file,  # [lib_file_idx][all_samples]
        'stacked_outputs': stacked_outputs,          # [all_samples, num_lib_files]
        'num_samples': num_samples,
        'num_lib_files': num_lib_files,
        'adjacency_weights': 'uniform'  # 모든 weight가 1.0임을 표시
    }
    
    torch.save(all_data, all_input_path)
    
    print(f"✅ Complete uniform-weight graph dataset saved:")
    print(f"   All data: {num_samples} samples × {num_lib_files} lib files → {all_input_path}")
    print(f"   All adjacency matrix weights = 1.0")
    
    return all_data

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python build_gnn_dataset_no_split_uniform_weights.py <output_dir> <lib_prefix> <data_dir>")
        print("Example: python build_gnn_dataset_no_split_uniform_weights.py dataset_test5_gnn_uniform OA_LVT_2_25_ OA_LVT")
        sys.exit(1)
        
    output_dir = sys.argv[1]       # e.g., dataset_test5_gnn_uniform
    lib_prefix = sys.argv[2]       # e.g., OA_LVT_2_25_
    data_dir = sys.argv[3]         # e.g., OA_LVT
    
    # 출력 디렉토리 미리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("🚀 Building GNN Graph Dataset (No Split, Uniform Weights)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Library prefix: {lib_prefix}")
    print(f"Data directory: {data_dir}")
    print(f"Adjacency matrix weights: ALL = 1.0 (uniform)")
    
    # Step 1: Build GNN graph dataset from .lib files
    print("\\n📥 Step 1: Processing .lib files and generating uniform-weight graph data...")
    
    # 임시 파일명으로 저장
    temp_input = f"{output_dir}_temp_graph_input_uniform.pth"
    temp_output = f"{output_dir}_temp_graph_output_uniform.pth"
    
    stacked_data = build_all_gnn_data(
        start=40, 
        end=101, 
        prefix=lib_prefix, 
        save_input=temp_input,
        save_output=temp_output,
        data_dir=data_dir
    )
    
    if stacked_data is None:
        print("❌ Failed to generate uniform-weight stacked graph data!")
        sys.exit(1)
    
    # Step 2: Organize dataset (no split)
    print("\\n📊 Step 2: Organizing uniform-weight dataset (no train/test split)...")
    final_data = organize_gnn_dataset_no_split(temp_input, output_dir)
    
    if final_data is None:
        print("❌ Failed to organize uniform-weight dataset!")
        sys.exit(1)
    
    # Step 3: Clean up temporary files
    if os.path.exists(temp_input):
        os.remove(temp_input)
    
    print(f"\\n✅ Uniform-weight GNN dataset generation completed!")
    print(f"   Dataset directory: {output_dir}/graph_data/")
    print(f"   File: all_graph_data_uniform.pth")
    print(f"   All adjacency matrix weights = 1.0")
    print(f"   Ready for uniform-weight training!")