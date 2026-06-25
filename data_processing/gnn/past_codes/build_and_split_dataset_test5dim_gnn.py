# build_and_split_dataset_gnn.py
import torch
from pathlib import Path
from datasets import libdata
from libdata_extract_MAML import parse_liberty_pin_blocks, flatten_pin_data
from transform_sample_MAML_stage_aware import transform_all_logic_samples_stage_aware
import sys
import os

def filetered_data(data_input,data_output):
    mask = data_input[..., -1] == 0 # 맨 뒤의 index가 0인 경우의 data만 남김
    #mask = torch.all(data_input[..., -3:] == 0, dim=-1) # 맨 뒤에서부터 3개의 index가 모두 0인 경우의 data만 남김
    filtered_input = data_input[mask]
    filtered_output = data_output[mask]
    return filtered_input,filtered_output

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
                       save_input="graph_input_data.pth", save_output="graph_output_data.pth", 
                       data_dir="OA_LVT"):
    """Build GNN graph dataset from multiple .lib files with stacking by lib file dimension"""
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
                    graph_data = {
                        'node_features': graph_sample['node_features'],
                        'edge_index': graph_sample['edge_index'],
                        'edge_attr': graph_sample.get('edge_attr', torch.ones(graph_sample['edge_index'].shape[1], dtype=torch.float32)),
                        'adjacency_matrix': graph_sample.get('adjacency_matrix', torch.eye(graph_sample['node_features'].shape[0], dtype=torch.float32)),
                        'cell_name': graph_sample['cell_name'],
                        'delay_type': graph_sample['delay_type'],
                        'input_slew': graph_sample['input_slew'],
                        'output_load': graph_sample['output_load'],
                        'voltage': graph_sample['voltage'],
                        'row_idx': graph_sample.get('row_idx', 0),
                        'col_idx': graph_sample.get('col_idx', 0)
                    }
                    output_value = graph_sample['output']
                    
                    current_file_graphs.append(graph_data)
                    current_file_outputs.append(output_value)
                
                # 이 lib file의 데이터를 전체에 추가
                graph_data_per_file.append(current_file_graphs)
                output_data_per_file.append(torch.tensor(current_file_outputs, dtype=torch.float32))
                print(f"   File {i}: {len(current_file_graphs)} samples")
                
            else:
                print(f"   No valid graph data found in {filename}")
                
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

    # Stack output data across lib files: [samples_per_file, num_lib_files]
    if output_data_per_file:
        stacked_outputs = torch.stack(output_data_per_file, dim=1)  # [samples, lib_files]
        print(f"🔄 Finalizing dataset...")
        print(f"   Output tensor shape: {stacked_outputs.shape}")  # [samples_per_file, num_lib_files]
        print(f"   Graph data structure: [samples_per_file][lib_file_idx]")
        
        # 저장할 데이터 구조
        final_data = {
            'graph_data_per_file': graph_data_per_file,  # [samples_per_file][lib_file_idx][graph_data]
            'stacked_outputs': stacked_outputs,          # [samples_per_file, num_lib_files]
            'num_samples_per_file': len(graph_data_per_file[0]) if graph_data_per_file else 0,
            'num_lib_files': len(graph_data_per_file)
        }
        
        torch.save(final_data, save_input)
        print(f"✅ Stacked GNN dataset saved: {save_input}")
        return final_data
    else:
        print("❌ No valid graph data found!")
        return None

def organize_gnn_dataset(graph_input_path, output_dir):
    """Organize GNN graph dataset (no train/test split - save all data)"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("📊 Loading stacked GNN graph dataset...")
    try:
        stacked_data = torch.load(graph_input_path, weights_only=False)
    except Exception as e:
        print(f"❌ Error loading graph data from {graph_input_path}: {e}")
        return
    
    graph_data_per_file = stacked_data['graph_data_per_file']  # [samples_per_file][lib_file_idx]
    stacked_outputs = stacked_data['stacked_outputs']          # [samples_per_file, num_lib_files]
    num_samples = stacked_data['num_samples_per_file']
    num_lib_files = stacked_data['num_lib_files']
    
    print(f"   Samples per file: {num_samples}")
    print(f"   Number of lib files: {num_lib_files}")
    print(f"   Output tensor shape: {stacked_outputs.shape}")
    
    # 그래프 데이터를 training/test용 디렉토리로 복사
    graph_data_dir = Path(output_dir) / "graph_data"
    graph_data_dir.mkdir(exist_ok=True)
    
    # Train/test split (80/20) - samples 기준으로 split
    train_size = int(0.8 * num_samples)
    
    # Graph data split
    train_graph_data = []
    test_graph_data = []
    
    for lib_file_idx in range(num_lib_files):
        # 각 lib file에서 train/test split
        lib_file_data = graph_data_per_file[lib_file_idx]
        train_graph_data.append(lib_file_data[:train_size])
        test_graph_data.append(lib_file_data[train_size:])
    
    # Output data split
    train_output = stacked_outputs[:train_size]  # [train_samples, num_lib_files]
    test_output = stacked_outputs[train_size:]   # [test_samples, num_lib_files]
    
    # 저장
    train_input_path = graph_data_dir / "transition_graph_input.pth"
    train_output_path = graph_data_dir / "transition_graph_output.pth"
    test_input_path = graph_data_dir / "transition_test_graph_input.pth"
    test_output_path = graph_data_dir / "transition_test_graph_output.pth"
    
    # 훈련용 데이터 구조
    train_data = {
        'graph_data_per_file': train_graph_data,  # [lib_file_idx][train_samples]
        'stacked_outputs': train_output,           # [train_samples, num_lib_files]
        'num_samples': train_size,
        'num_lib_files': num_lib_files
    }
    
    test_data = {
        'graph_data_per_file': test_graph_data,   # [lib_file_idx][test_samples]
        'stacked_outputs': test_output,            # [test_samples, num_lib_files]
        'num_samples': num_samples - train_size,
        'num_lib_files': num_lib_files
    }
    
    torch.save(train_data, train_input_path)
    torch.save(test_data, test_input_path)
    
    print(f"✅ Stacked graph dataset organized:")
    print(f"   Train: {train_size} samples × {num_lib_files} lib files → {train_input_path}")
    print(f"   Test: {num_samples - train_size} samples × {num_lib_files} lib files → {test_input_path}")
    
    return train_data, test_data

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python build_and_split_dataset_test5dim_gnn.py <output_dir> <lib_prefix> <data_dir>")
        print("Example: python build_and_split_dataset_test5dim_gnn.py dataset_test5_gnn OA_LVT_2_25_ OA_LVT")
        sys.exit(1)
        
    output_dir = sys.argv[1]       # e.g., dataset_test5_gnn
    lib_prefix = sys.argv[2]       # e.g., OA_LVT_2_25_
    data_dir = sys.argv[3]         # e.g., OA_LVT
    
    print("🚀 Building GNN Graph Dataset")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Library prefix: {lib_prefix}")
    print(f"Data directory: {data_dir}")
    
    # Step 1: Build GNN graph dataset from .lib files
    print("\n📥 Step 1: Processing .lib files and generating graph data...")
    
    # 임시 파일명으로 저장
    temp_input = f"{output_dir}_temp_graph_input.pth"
    temp_output = f"{output_dir}_temp_graph_output.pth"
    
    stacked_data = build_all_gnn_data(
        start=40, 
        end=101, 
        prefix=lib_prefix, 
        save_input=temp_input,
        save_output=temp_output,
        data_dir=data_dir
    )
    
    if stacked_data is None:
        print("❌ Failed to generate stacked graph data!")
        sys.exit(1)
    
    # Step 2: Organize dataset for training
    print("\n📊 Step 2: Organizing dataset for training/testing...")
    organize_gnn_dataset(temp_input, output_dir)
    
    # Step 3: Clean up temporary files
    if os.path.exists(temp_input):
        os.remove(temp_input)
    
    print(f"\n✅ GNN dataset generation completed!")
    print(f"   Dataset directory: {output_dir}/graph_data/")
    print(f"   Ready for GNN pretraining!")
