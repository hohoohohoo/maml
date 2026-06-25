# build_gnn_dataset_with_process.py
"""
GNN dataset builder with process condition parameters
This version adds 4 process parameters to node features: param_a, param_b, param_c, temperature
Node features: 7D → 11D
"""

import torch
from pathlib import Path
import sys
import os

# Add utils path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from transform_sample_MAML_with_process import transform_all_logic_samples_with_process

def create_stage_aware_adjacency_matrix(graph_sample):
    """
    Stage 정보를 반영한 enhanced adjacency matrix 생성
    VDD → intermediate → Y 및 VSS → intermediate → Y 경로를 명시적으로 추가
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
        print(f"   No stage info available, using original adjacency matrix")
        return original_adj

    stage_type = stage_info.get('stage_type', 'one_stage')
    intermediate_gates = stage_info.get('intermediate_gates', [])

    print(f"   Creating stage-aware adjacency matrix for {stage_type}")

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

        # Stage-aware 연결 추가
        for inter_idx in intermediate_indices:
            if 'rise' in delay_type:
                # Rise: Stage1(VSS → intermediate) → Stage2(VDD → Y, intermediate → VDD)
                # Stage1: VSS → intermediate 경로 강화
                if vss_idx >= 0:
                    enhanced_adj[vss_idx, inter_idx] = 2.0  # Stage1: VSS → intermediate

                # Stage2: VDD → Y 및 intermediate → VDD 경로 강화
                if vdd_idx >= 0 and y_idx >= 0:
                    enhanced_adj[vdd_idx, y_idx] = 2.0      # Stage2: VDD → Y
                if vdd_idx >= 0:
                    enhanced_adj[inter_idx, vdd_idx] = 2.0  # Stage2: intermediate → VDD

                # intermediate → Y 연결
                if y_idx >= 0:
                    enhanced_adj[inter_idx, y_idx] = 1.8   # intermediate → Y (강한 연결)

                print(f"   Rise paths: VSS({vss_idx}) → {inter_idx}, {inter_idx} → VDD({vdd_idx}), VDD → Y({y_idx})")

            else:  # fall_transition
                # Fall: Stage1(VDD → intermediate) → Stage2(VSS → Y, intermediate → VSS)
                # Stage1: VDD → intermediate 경로 강화
                if vdd_idx >= 0:
                    enhanced_adj[vdd_idx, inter_idx] = 2.0  # Stage1: VDD → intermediate

                # Stage2: VSS → Y 및 intermediate → VSS 경로 강화
                if vss_idx >= 0 and y_idx >= 0:
                    enhanced_adj[vss_idx, y_idx] = 2.0      # Stage2: VSS → Y
                if vss_idx >= 0:
                    enhanced_adj[inter_idx, vss_idx] = 2.0  # Stage2: intermediate → VSS

                # intermediate → Y 연결
                if y_idx >= 0:
                    enhanced_adj[inter_idx, y_idx] = 1.8   # intermediate → Y (강한 연결)

                print(f"   Fall paths: VDD({vdd_idx}) → {inter_idx}, {inter_idx} → VSS({vss_idx}), VSS → Y({y_idx})")

        # 모든 intermediate간 연결 (multi-gate의 경우)
        for i, idx1 in enumerate(intermediate_indices):
            for j, idx2 in enumerate(intermediate_indices):
                if i != j:
                    enhanced_adj[idx1, idx2] = 1.2  # Intermediate gate 간 약한 연결

    else:
        print(f"   One-stage logic, using original adjacency matrix")

    #return enhanced_adj
    return original_adj

def dataextract_gnn_with_process(text, lib_prefix="", data_type="cell", graph_mode="stage_aware", is_test=False):
    """
    Extract GNN graph data from .lib file with process conditions
    Node features: 11D (7 base + 4 process parameters)

    Args:
        text: Path to .lib file
        lib_prefix: Library file prefix (used to parse process conditions)
        data_type: 'cell' or 'transition' to select parser
        graph_mode: 'stage_aware' (current path only) or 'full_graph' (all transistors baseline)
        is_test: Whether this is from test dataset
    """
    # Import appropriate parser based on data_type
    if data_type == "cell":
        from libdata_extract_MAML_cell import parse_liberty_pin_blocks, flatten_pin_data
    elif data_type == "transition":
        from libdata_extract_MAML_transition import parse_liberty_pin_blocks, flatten_pin_data
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'cell' or 'transition'")

    with open(text, "r") as f:
        lines = f.readlines()
    pin_data = parse_liberty_pin_blocks(lines)
    flattened, cap = flatten_pin_data(pin_data)

    # GNN 그래프 변환 with process conditions
    graph_dataset = transform_all_logic_samples_with_process(flattened, cap, lib_prefix, graph_mode, is_test)
    return graph_dataset

def data_management_gnn_with_process(text, lib_prefix="", data_type="cell", graph_mode="stage_aware", is_test=False):
    """
    Process GNN graph data from .lib file with process conditions

    Args:
        text: Path to .lib file
        lib_prefix: Library file prefix (used to parse process conditions)
        data_type: 'cell' or 'transition' to select parser
        graph_mode: 'stage_aware' (current path only) or 'full_graph' (all transistors baseline)
        is_test: Whether this is from test dataset
    """
    from pathlib import Path
    file_path = Path(text)
    graph_dataset = dataextract_gnn_with_process(file_path, lib_prefix, data_type, graph_mode, is_test)
    return graph_dataset

def build_all_gnn_data_with_process(start=40, end=101, prefix="invbuf_0_0_0_",
                       save_input="graph_input_data.pth", save_output="graph_output_data.pth",
                       data_dir="simple", lib_base_path="/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed_simple",
                       data_type="cell", graph_mode="stage_aware", is_test=False):
    """
    Build GNN graph dataset from multiple .lib files with process conditions (Memory-efficient version)
    Node features: 11D (7 base + 4 process parameters)

    Args:
        start, end: Range of voltage indices
        prefix: Library file prefix (e.g., "invbuf_0_0_0_")
        save_input, save_output: Output file paths
        data_dir: Directory containing lib files
        lib_base_path: Base path to lib files
        data_type: 'cell' or 'transition' to select parser
        graph_mode: 'stage_aware' (current path only) or 'full_graph' (all transistors baseline)
        is_test: Whether this is from test dataset
    """
    import tempfile
    import gc

    temp_files = []
    num_samples = None

    for i in range(start, end):
        v_str = f"{i//100}{i%100:02d}"
        # Use absolute path for lib files
        filename = f"{lib_base_path}/{data_dir}/{prefix}{v_str}.lib"
        print(f"📥 Processing {filename} (mode: {graph_mode}, process-aware)")

        try:
            graph_dataset = data_management_gnn_with_process(filename, prefix, data_type, graph_mode, is_test)

            if graph_dataset:
                print(f"   Generated {len(graph_dataset)} graph samples with process conditions")

                # 현재 lib file의 모든 graph data와 output을 리스트로 수집
                current_file_graphs = []
                current_file_outputs = []

                for graph_sample in graph_dataset:
                    # Verify node features dimension
                    expected_dim = 11  # 7 base + 4 process
                    actual_dim = graph_sample['node_features'].shape[1]
                    if actual_dim != expected_dim:
                        print(f"   ⚠️  Warning: Expected {expected_dim}D features, got {actual_dim}D")

                    # Stage-aware adjacency matrix 생성
                    enhanced_adj_matrix = create_stage_aware_adjacency_matrix(graph_sample)

                    # Safe edge_attr handling
                    edge_index = graph_sample['edge_index']
                    edge_attr = graph_sample.get('edge_attr', None)
                    if edge_attr is None or edge_attr.numel() == 0:
                        # Create default edge_attr with correct shape
                        num_edges = edge_index.shape[1]
                        edge_attr = torch.ones(num_edges, dtype=torch.float32)

                    graph_data = {
                        'node_features': graph_sample['node_features'],  # 11D features
                        # 'edge_index': edge_index,
                        # 'edge_attr': edge_attr,
                        'adjacency_matrix': enhanced_adj_matrix,
                        # 'cell_name': graph_sample['cell_name'],
                        # 'delay_type': graph_sample['delay_type'],
                        # 'input_slew': graph_sample['input_slew'],
                        # 'output_load': graph_sample['output_load'],
                        # 'voltage': graph_sample['voltage'],
                        # 'row_idx': graph_sample.get('row_idx', 0),
                        # 'col_idx': graph_sample.get('col_idx', 0),
                        # # Stage 정보 보존
                        # 'all_nodes': graph_sample.get('all_nodes', []),
                        # 'circuit_nodes': graph_sample.get('circuit_nodes', []),
                        # 'transistor_nodes': graph_sample.get('transistor_nodes', []),
                        # 'stage_info': graph_sample.get('stage_info', {}),
                        # # Process parameters 보존
                        # 'process_params': graph_sample.get('process_params', {})
                    }
                    output_value = graph_sample['output']

                    current_file_graphs.append(graph_data)
                    current_file_outputs.append(output_value)

                # Verify consistent sample count across files
                current_sample_count = len(current_file_graphs)
                if num_samples is None:
                    num_samples = current_sample_count
                elif num_samples != current_sample_count:
                    print(f"   ⚠️  Warning: Sample count mismatch! Expected {num_samples}, got {current_sample_count}")
                    # Skip this file to maintain consistency
                    continue

                # Save to temporary file immediately and free memory
                temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth')
                temp_data = {
                    'graphs': current_file_graphs,
                    'outputs': torch.tensor(current_file_outputs, dtype=torch.float32)
                }
                torch.save(temp_data, temp_file.name)
                temp_files.append(temp_file.name)
                temp_file.close()

                print(f"   File {i}: {len(current_file_graphs)} samples (11D) → saved to temp")

                # Free memory
                del current_file_graphs, current_file_outputs, graph_dataset, temp_data
                gc.collect()

            else:
                print(f"   No valid graph data found in {filename}")

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Load from temp files and stack
    if temp_files:
        print(f"🔄 Finalizing dataset from {len(temp_files)} temp files...")
        graph_data_per_file = []
        output_data_per_file = []

        for temp_file in temp_files:
            temp_data = torch.load(temp_file, weights_only=False)
            graph_data_per_file.append(temp_data['graphs'])
            output_data_per_file.append(temp_data['outputs'])
            del temp_data
            gc.collect()

        # Stack output data across lib files: [samples_per_file, num_lib_files]
        stacked_outputs = torch.stack(output_data_per_file, dim=1)  # [samples, lib_files]
        print(f"   Output tensor shape: {stacked_outputs.shape}")  # [samples_per_file, num_lib_files]
        print(f"   Graph data structure: [samples_per_file][lib_file_idx]")
        print(f"   Node feature dimension: 11D (7 base + 4 process)")

        # 저장할 데이터 구조
        final_data = {
            'graph_data_per_file': graph_data_per_file,  # [samples_per_file][lib_file_idx][graph_data]
            'stacked_outputs': stacked_outputs,          # [samples_per_file, num_lib_files]
            # 'num_samples_per_file': len(graph_data_per_file[0]) if graph_data_per_file else 0,
            # 'num_lib_files': len(graph_data_per_file),
            # 'node_feature_dim': 11,  # Track feature dimension
            # 'has_process_params': True  # Flag for process-aware dataset
        }

        torch.save(final_data, save_input)
        print(f"✅ Stacked GNN dataset with process conditions saved: {save_input}")

        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

        return final_data
    else:
        print("❌ No valid graph data found!")
        return None

def organize_gnn_dataset_no_split_with_process(graph_input_path, output_dir, graph_mode="stage_aware"):
    """Save all GNN graph data with process conditions without train/test split"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("📊 Loading stacked GNN graph dataset with process conditions...")
    try:
        stacked_data = torch.load(graph_input_path, weights_only=False)
    except Exception as e:
        print(f"❌ Error loading graph data from {graph_input_path}: {e}")
        return None

    graph_data_per_file = stacked_data['graph_data_per_file']  # [samples_per_file][lib_file_idx]
    stacked_outputs = stacked_data['stacked_outputs']          # [samples_per_file, num_lib_files]
    # num_samples = stacked_data['num_samples_per_file']
    # num_lib_files = stacked_data['num_lib_files']
    # node_feature_dim = stacked_data.get('node_feature_dim', 11)
    # has_process_params = stacked_data.get('has_process_params', True)

    # print(f"   Samples per file: {num_samples}")
    # print(f"   Number of lib files: {num_lib_files}")
    # print(f"   Output tensor shape: {stacked_outputs.shape}")
    # print(f"   Graph mode: {graph_mode}")
    # print(f"   Node feature dimension: {node_feature_dim}D")
    # print(f"   Process parameters: {'Yes' if has_process_params else 'No'}")

    # 그래프 데이터를 저장할 디렉토리 생성
    graph_data_dir = Path(output_dir) / "graph_data"
    graph_data_dir.mkdir(exist_ok=True)

    # 분할하지 않고 모든 데이터를 그대로 저장
    print("   📁 Saving all data without train/test split...")

    # 저장 경로 (graph_mode + process 표시 포함)
    all_input_path = graph_data_dir / f"cell_all_graph_data_{graph_mode}_process.pth"

    # 전체 데이터 구조 (분할하지 않음)
    all_data = {
        'graph_data_per_file': graph_data_per_file,  # [lib_file_idx][all_samples]
        'stacked_outputs': stacked_outputs,          # [all_samples, num_lib_files]
        # 'num_samples': num_samples,
        # 'num_lib_files': num_lib_files,
        # 'graph_mode': graph_mode,  # graph_mode 정보 저장
        # 'node_feature_dim': node_feature_dim,
        # 'has_process_params': has_process_params
    }

    torch.save(all_data, all_input_path)

    print(f"✅ Complete graph dataset with process conditions saved:")
    # print(f"   All data: {num_samples} samples × {num_lib_files} lib files → {all_input_path}")
    # print(f"   Node features: {node_feature_dim}D (7 base + 4 process)")

    return all_data

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python build_gnn_dataset_with_process.py <output_dir> <lib_prefix> <data_dir> [lib_base_path] [data_type] [graph_mode] [is_test]")
        print("Example: python build_gnn_dataset_with_process.py dataset_process_gnn invbuf_0_0_0_ simple")
        print("Example with custom lib path: python build_gnn_dataset_with_process.py dataset_process_gnn invbuf_0_0_0_ simple /custom/path/to/processed_libs")
        print("Example with data_type: python build_gnn_dataset_with_process.py dataset_process_gnn invbuf_0_0_0_ simple /custom/path transition")
        print("Example with graph_mode: python build_gnn_dataset_with_process.py dataset_process_gnn invbuf_0_0_0_ simple /custom/path cell full_graph")
        print("Example with test flag: python build_gnn_dataset_with_process.py dataset_process_gnn invbuf_0_0_0_ simple /custom/path cell stage_aware true")
        print("\nGraph modes:")
        print("  - stage_aware (default): Current path only (stage-aware extraction)")
        print("  - full_graph: All transistors in the cell (baseline)")
        print("\nProcess condition parameters:")
        print("  - Adds 4 parameters to node features: param_a, param_b, param_c, temperature")
        print("  - Node features: 7D → 11D")
        print("  - Parameters parsed from lib_prefix filename pattern")
        sys.exit(1)

    output_dir = sys.argv[1]       # e.g., dataset_process_gnn
    lib_prefix = sys.argv[2]       # e.g., invbuf_0_0_0_
    data_dir = sys.argv[3]         # e.g., simple
    lib_base_path = sys.argv[4] if len(sys.argv) > 4 else "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/ASAP7_lib_files/processed_simple"
    data_type = sys.argv[5] if len(sys.argv) > 5 else "cell"  # 'cell' or 'transition'
    graph_mode = sys.argv[6] if len(sys.argv) > 6 else "stage_aware"  # 'stage_aware' or 'full_graph'
    is_test = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False  # True for test dataset

    # 출력 디렉토리 미리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("🚀 Building GNN Graph Dataset with Process Conditions (No Split)")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Library prefix: {lib_prefix}")
    print(f"Data directory: {data_dir}")
    print(f"Library base path: {lib_base_path}")
    print(f"Data type: {data_type}")
    print(f"Graph mode: {graph_mode}")
    print(f"Test dataset: {is_test}")
    print(f"Node features: 11D (7 base + 4 process parameters)")

    # Step 1: Build GNN graph dataset from .lib files with process conditions
    print("\n📥 Step 1: Processing .lib files and generating graph data with process conditions...")

    # 임시 파일명으로 저장
    temp_input = f"{output_dir}_temp_graph_input.pth"
    temp_output = f"{output_dir}_temp_graph_output.pth"

    stacked_data = build_all_gnn_data_with_process(
        start=40,
        end=101,
        prefix=lib_prefix,
        save_input=temp_input,
        save_output=temp_output,
        data_dir=data_dir,
        lib_base_path=lib_base_path,
        data_type=data_type,
        graph_mode=graph_mode,
        is_test=is_test
    )

    if stacked_data is None:
        print("❌ Failed to generate stacked graph data with process conditions!")
        sys.exit(1)

    # Step 2: Organize dataset (no split)
    print("\n📊 Step 2: Organizing dataset (no train/test split)...")
    final_data = organize_gnn_dataset_no_split_with_process(temp_input, output_dir, graph_mode)

    if final_data is None:
        print("❌ Failed to organize dataset!")
        sys.exit(1)

    # Step 3: Clean up temporary files
    if os.path.exists(temp_input):
        os.remove(temp_input)

    print(f"\n✅ GNN dataset with process conditions generation completed!")
    print(f"   Dataset directory: {output_dir}/graph_data/")
    print(f"   File: cell_all_graph_data_{graph_mode}_process.pth")
    print(f"   Graph mode: {graph_mode}")
    print(f"   Node features: 11D (7 base + 4 process)")
    print(f"   Ready for batch processing!")
