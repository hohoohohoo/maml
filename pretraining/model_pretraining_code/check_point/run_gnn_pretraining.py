#!/usr/bin/env python
# coding: utf-8

"""
GNN Pretraining Runner
======================

This script runs GNN-based pretraining for different configurations.
You can easily switch between different GNN architectures and graph structures.
"""

import os
import sys
import json
import subprocess
from datetime import datetime

def run_gnn_pretraining(gnn_type='GCN', graph_type='fully_connected', 
                       node_features=3, gpu_id="2", 
                       iterations=100000, chunk_size=10000):
    """
    Run GNN pretraining with specified configuration
    
    Args:
        gnn_type: 'GCN', 'GraphSAGE', or 'GAT'
        graph_type: 'fully_connected', 'path_bidirectional', or 'path_unidirectional'
        node_features: Number of features per node (1-5)
        gpu_id: GPU device ID
        iterations: Total training iterations
        chunk_size: Iterations per chunk
    """
    
    print(f"🚀 Starting GNN Pretraining")
    print(f"   Architecture: {gnn_type}")
    print(f"   Graph structure: {graph_type}")
    print(f"   Node features: {node_features}")
    print(f"   GPU: {gpu_id}")
    print(f"   Iterations: {iterations:,}")
    
    # Create a temporary script with the specific configuration
    script_content = f'''#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch import optim
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from collections import OrderedDict
import sys
import matplotlib.pyplot as plt
import time

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# GPU 최적화 설정
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# GNN MAML import
sys.path.append('../../')
from gnn_maml_optimized import (
    create_gcn_model, 
    create_graphsage_model, 
    create_gat_model,
    OptimizedGNNMAML,
    SequenceToGraphConverter
)

def main():
    total_iterations = {iterations}
    chunk_size = {chunk_size}
    num_chunks = total_iterations // chunk_size
    
    # Configuration
    gnn_type = '{gnn_type}'
    graph_type = '{graph_type}'
    node_features = {node_features}
    layer_length = 40
    inner_step = 1
    
    start_time = time.time()
    
    print(f"\\n🚀 Processing transition dataset with GNN-MAML")
    print(f"🏗️ GNN Configuration:")
    print(f"   Architecture: {{gnn_type}}")
    print(f"   Graph structure: {{graph_type}}")
    print(f"   Node features: {{node_features}}")
    
    # 데이터 로드 및 전처리 (동일한 로직)
    print("📊 Loading and preprocessing data...")
    test_data_input = torch.load(f"../../dataset_test5(dim5)_SS/taskdivide_rvt/traindatainput/transition_train_input.pth")
    test_data_output_1 = torch.load(f"../../dataset_test5(dim5)_SS/taskdivide_rvt/traindataoutput/transition_train_output.pth")
    
    # 정규화 (동일한 로직)
    voltage_mean = test_data_input[:,:,0].mean()
    voltage_std = test_data_input[:,:,0].std()
    input_features = test_data_input.shape[2]
    
    if voltage_std > 1e-8:
        test_data_input[:,:,0] = ((test_data_input[:,:,0] - voltage_mean) / voltage_std)
    
    if input_features >= 5:
        feature_4_mean = test_data_input[:,:,3].mean()
        feature_4_std = test_data_input[:,:,3].std()
        if feature_4_std > 1e-8:
            test_data_input[:,:,3] = ((test_data_input[:,:,3] - feature_4_mean) / feature_4_std)
            
        feature_5_mean = test_data_input[:,:,4].mean()
        feature_5_std = test_data_input[:,:,4].std()
        if feature_5_std > 1e-8:
            test_data_input[:,:,4] = ((test_data_input[:,:,4] - feature_5_mean) / feature_5_std)
    
    # 필터링 (동일한 로직)
    filtered_input = []
    filtered_output = []
    min_std_threshold = 1e-6
    
    for i in range(len(test_data_output_1)):
        output_mean = test_data_output_1[i,:,:].mean()
        output_std = test_data_output_1[i,:,:].std()
        
        has_nan_inf = (torch.isnan(test_data_output_1[i]).any() or 
                      torch.isinf(test_data_output_1[i]).any() or
                      torch.isnan(test_data_input[i]).any() or 
                      torch.isinf(test_data_input[i]).any())
        
        if output_std > min_std_threshold and not has_nan_inf:
            normalized_output = (test_data_output_1[i,:,:] - output_mean) / output_std
            filtered_input.append(test_data_input[i])
            filtered_output.append(normalized_output)
    
    if filtered_input:
        test_data_input = torch.stack(filtered_input)
        test_data_output_1 = torch.stack(filtered_output)
        print(f"✅ Valid samples: {{len(filtered_input)}}")
    else:
        print("❌ No valid samples found!")
        return
    
    test_data_input = test_data_input.to(device)
    test_data_output_1 = test_data_output_1.to(device)
    
    # 그래프 변환 함수 선택
    graph_converters = {{
        'fully_connected': lambda x: SequenceToGraphConverter.sequence_to_fully_connected_graph(x, node_features),
        'path_bidirectional': lambda x: SequenceToGraphConverter.sequence_to_path_graph(x, node_features, True),
        'path_unidirectional': lambda x: SequenceToGraphConverter.sequence_to_path_graph(x, node_features, False)
    }}
    
    graph_converter_func = graph_converters[graph_type]
    
    # GNN 모델 생성
    print(f"🤖 Creating GNN-MAML model ({{gnn_type}})...")
    
    if gnn_type == 'GCN':
        gnn_model = create_gcn_model(node_features=node_features, hidden_dim=layer_length, num_layers=3, pooling='mean')
    elif gnn_type == 'GraphSAGE':
        gnn_model = create_graphsage_model(node_features=node_features, hidden_dim=layer_length, num_layers=3, pooling='mean')
    elif gnn_type == 'GAT':
        gnn_model = create_gat_model(node_features=node_features, hidden_dim=layer_length, num_layers=3, heads=4, pooling='mean')
    
    total_params = sum(p.numel() for p in gnn_model.parameters())
    print(f"📊 Model parameters: {{total_params:,}}")
    
    # GNN-MAML 생성
    gnn_maml = OptimizedGNNMAML(
        model=gnn_model,
        dataset_in=test_data_input,
        dataset_out=test_data_output_1,
        inner_lr=0.001,
        meta_lr=0.0001,
        inner_steps=inner_step,
        tasks_per_meta_batch=16,
        graph_converter_func=graph_converter_func
    )
    
    # 훈련
    for chunk in range(1, num_chunks + 1):
        print(f"▶️ Chunk {{chunk}}/{{num_chunks}}")
        
        try:
            gnn_maml.main_loop_sequential(num_iterations=chunk_size)
        except Exception as e:
            print(f"⚠️ 훈련 실패: {{e}}")
            gnn_maml.inner_lr *= 0.5
            gnn_maml.meta_lr *= 0.5
            try:
                gnn_maml.main_loop_sequential(num_iterations=chunk_size//2)
            except:
                print("⚠️ 이 청크를 건너뜁니다.")
                continue
        
        # 체크포인트 저장
        checkpoint_path = f"../../pretrained_models/checkpoints/taskdivide_all_checkpoints/transition_full1D{{gnn_type}}MAML_weights_3hidden_({{layer_length}})_{{chunk*chunk_size}}_RVT_SS_test5(dim5)_inner{{inner_step}}_{{graph_type}}.pth"
        torch.save(gnn_maml.model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {{checkpoint_path}}")
    
    # 최종 모델 저장
    final_model_path = f"../../pretrained_models/taskdivide_all/transition_full1D{{gnn_type}}MAML_weights_3hidden_({{layer_length}})_{{total_iterations}}_RVT_SS_test5(dim5)_inner{{inner_step}}_{{graph_type}}.pth"
    torch.save(gnn_maml.model.state_dict(), final_model_path)
    print(f"🏁 Training complete. Model saved to: {{final_model_path}}")
    
    # 설정 저장
    config_info = {{
        'gnn_type': gnn_type,
        'graph_type': graph_type,
        'node_features': node_features,
        'hidden_dim': layer_length,
        'total_parameters': total_params,
        'total_iterations': total_iterations,
        'final_model_path': final_model_path,
        'training_completed_at': datetime.now().isoformat()
    }}
    
    import json
    from datetime import datetime
    config_path = final_model_path.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\\n🎉 GNN-MAML Training completed in {{total_time:.2f}}s")
    
    del gnn_maml, test_data_input, test_data_output_1
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
'''
    
    # Write temporary script
    temp_script = f"temp_gnn_pretraining_{gnn_type}_{graph_type}.py"
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    try:
        # Run the script
        result = subprocess.run(['python', temp_script], 
                              capture_output=True, text=True, timeout=3600*5)  # 5 hour timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Process failed with return code {result.returncode}")
        else:
            print("✅ Process completed successfully")
            
    except subprocess.TimeoutExpired:
        print("⏰ Process timed out after 5 hours")
    except Exception as e:
        print(f"❌ Error running process: {e}")
    finally:
        # Clean up temporary script
        if os.path.exists(temp_script):
            os.remove(temp_script)

def run_multiple_configurations():
    """Run multiple GNN configurations"""
    
    configurations = [
        # GCN experiments
        {'gnn_type': 'GCN', 'graph_type': 'fully_connected', 'node_features': 3},
        {'gnn_type': 'GCN', 'graph_type': 'path_bidirectional', 'node_features': 3},
        
        # GraphSAGE experiments  
        {'gnn_type': 'GraphSAGE', 'graph_type': 'fully_connected', 'node_features': 3},
        
        # GAT experiments
        {'gnn_type': 'GAT', 'graph_type': 'fully_connected', 'node_features': 3},
        
        # Different node features
        {'gnn_type': 'GCN', 'graph_type': 'fully_connected', 'node_features': 5},
    ]
    
    for i, config in enumerate(configurations, 1):
        print(f"\n{'='*60}")
        print(f"Running Configuration {i}/{len(configurations)}")
        print(f"{'='*60}")
        
        run_gnn_pretraining(**config, iterations=50000)  # Shorter for multiple configs
        
        print(f"Configuration {i} completed")

def quick_test():
    """Quick test with minimal iterations"""
    print("🧪 Running quick test...")
    run_gnn_pretraining(
        gnn_type='GCN', 
        graph_type='fully_connected', 
        node_features=3,
        iterations=1000,  # Very short for testing
        chunk_size=500
    )

if __name__ == "__main__":
    print("GNN Pretraining Runner")
    print("=" * 30)
    print("1. Quick test (1000 iterations)")
    print("2. Single configuration (100k iterations)")
    print("3. Multiple configurations (50k each)")
    print("4. Custom configuration")
    
    choice = input("\\nSelect option (1-4): ").strip()
    
    if choice == '1':
        quick_test()
    elif choice == '2':
        # Default configuration
        run_gnn_pretraining()
    elif choice == '3':
        run_multiple_configurations()
    elif choice == '4':
        print("\\nCustom Configuration:")
        gnn_type = input("GNN type (GCN/GraphSAGE/GAT): ").strip() or 'GCN'
        graph_type = input("Graph type (fully_connected/path_bidirectional/path_unidirectional): ").strip() or 'fully_connected'
        node_features = int(input("Node features (1-5): ").strip() or '3')
        iterations = int(input("Total iterations: ").strip() or '100000')
        
        run_gnn_pretraining(
            gnn_type=gnn_type,
            graph_type=graph_type, 
            node_features=node_features,
            iterations=iterations
        )
    else:
        print("Invalid choice. Running default configuration...")
        run_gnn_pretraining()