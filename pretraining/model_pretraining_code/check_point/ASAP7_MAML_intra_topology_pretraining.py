#!/usr/bin/env python
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
import argparse
import glob
import re

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# GPU 최적화 설정
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# MAML import
sys.path.append('../../model_code/')
from maml_optimized import OptimizedMAML, MAMLModel_3hidden

# Import utility functions
from maml_utils import extract_iteration_from_filename, find_pretrained_model, load_pretrained_model

def main():
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='MAML Pretraining - Resume from checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to specific pretrained model file to resume from')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Automatically find and resume from latest pretrained model')
    parser.add_argument('--inner', type=int, default=1,
                        help='Inner loop steps (default: 1)')
    parser.add_argument('--innerdiv', type=int, default=100,
                        help='Inner learning rate divisor: inner_lr = 0.001/innerdiv (default: 100)')
    parser.add_argument('--meta', type=int, default=32,
                        help='Tasks per meta batch (default: 32)')
    parser.add_argument('--data_type', type=str, default='cell',
                        help='Data type: cell/transition (default: cell)')
    args = parser.parse_args()
    # 학습 설정
    additional_iterations = 300000  # 추가로 학습할 iteration 수
    chunk_size = 30000
    num_chunks = additional_iterations // chunk_size

    # GPU utilization 모니터링
    start_time = time.time()

    layer_length = 40
    inner = args.inner
    innerdiv = args.innerdiv
    meta = args.meta
    data_type = args.data_type.lower()

    print(f"\n🚀 Processing separate {data_type} type SHARED train dataset")

    # 설정값 출력
    print(f"⚙️ Training configuration:")
    print(f"   Data type: {data_type}")
    print(f"   Inner loop steps: {inner}")
    print(f"   Inner learning rate divisor: {innerdiv} (inner_lr = 0.001/{innerdiv} = {0.001/innerdiv})")
    print(f"   Tasks per meta batch: {meta}")

    # 모델 디렉토리 경로
    pretrained_models_dir = "../../pretrained_models/taskdivide_all"
    checkpoint_dir = "../../pretrained_models/checkpoints/taskdivide_all_checkpoints"

    # 디렉토리 생성 (없으면)
    os.makedirs(pretrained_models_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 데이터 로드 및 전처리
    print("📊 Loading and preprocessing separate cell type SHARED train data...")
    
    # Separate cell type SHARED train data 경로 - shared train data 사용
    data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded"
    test_data_input = torch.load(f"{data_dir}/{data_type}_intratopology_train_input.pth")      # [total_samples, 61, 9]
    test_data_output_1 = torch.load(f"{data_dir}/{data_type}_intratopology_train_output.pth")  # [total_samples, 61, 1]
    # Note: separate_cell_type_split shared train data already has correct shape [total_samples, 61, 1]
    test_data_input2 = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/unified_invbuf/merged_invbuf_input_{data_type}.pth")      # [total_samples, 61, 9]
    test_data_output_2 = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/unified_invbuf/merged_invbuf_output_{data_type}.pth")  # [total_samples, 61]
    
    test_data_input = torch.cat([test_data_input,test_data_input2],dim=0)
    test_data_output_1= torch.cat([test_data_output_1,test_data_output_2],dim=0)
    test_data_output_1 = test_data_output_1 # [total_samples, 61, 1]
    print(f"📊 Data shapes: Input {test_data_input.shape}, Output {test_data_output_1.shape}")
    
    # 9차원 invbuf 입력 특성:
    # 0: a_param, 1: b_param, 2: c_param, 3: temperature, 4: voltage, 
    # 5: additional_dim, 6: delay_indicator, 7: index_1_val(slew), 8: index_2_val(load_cap)
    
    # 정규화할 특성들: slew, load_cap, temperature, voltage
    normalize_features = [7, 8, 3, 4]  # index_1(slew), index_2(load_cap), temperature, voltage
    feature_names = ['slew', 'load_cap', 'temperature', 'voltage']
    
    print(f"🔧 Normalizing features: {feature_names}")
    print(f"   Feature indices: {normalize_features}")
    
    # 9차원 유지하면서 선택된 특성들만 정규화
    input_features = test_data_input.shape[2]  # 9차원 유지
    
    print(f"📊 Input shape (9 features maintained): {test_data_input.shape}")
    
    # 각 특성별 정규화 (9차원 중 선택된 4개만)
    for feature_idx, feature_name in zip(normalize_features, feature_names):
        feature_mean = test_data_input[:, :, feature_idx].mean()
        feature_std = test_data_input[:, :, feature_idx].std()
        
        print(f"📊 {feature_name} (idx {feature_idx}) stats: mean={feature_mean:.6f}, std={feature_std:.6f}")
        
        # 안전한 정규화 (std가 0이면 정규화 생략)
        if feature_std > 1e-8:
            test_data_input[:, :, feature_idx] = (test_data_input[:, :, feature_idx] - feature_mean) / feature_std
            print(f"   ✅ {feature_name} normalized")
        else:
            print(f"   ⚠️ {feature_name} std too small, skipping normalization")
    
    # 데이터 유효성 검사
    print(f"📊 Input data after normalization:")
    print(f"   Range: min={test_data_input.min():.6f}, max={test_data_input.max():.6f}")
    print(f"   Contains NaN: {torch.isnan(test_data_input).any()}")
    print(f"   Contains Inf: {torch.isinf(test_data_input).any()}")
    
    print(f"📊 Output data stats:")
    print(f"   Range: min={test_data_output_1.min():.6f}, max={test_data_output_1.max():.6f}")
    print(f"   Contains NaN: {torch.isnan(test_data_output_1).any()}")
    print(f"   Contains Inf: {torch.isinf(test_data_output_1).any()}")
    
    # Task별 정규화 및 유효한 샘플 필터링
    valid_indices = []
    filtered_input = []
    filtered_output = []
    
    min_std_threshold = 1e-6  # 최소 std 임계값
    
    print(f"🔍 Filtering tasks with std >= {min_std_threshold}...")
    original_size = len(test_data_output_1)
    
    for i in range(len(test_data_output_1)):
        # Task별 출력 정규화 (61개 point에 대해)
        output_mean = test_data_output_1[i, :, :].mean()
        output_std = test_data_output_1[i, :, :].std()
        
        # 입력 데이터의 변동성 체크 (각 feature별로)
        input_stds = []
        for feature_idx in range(test_data_input.shape[2]):
            feature_std = test_data_input[i, :, feature_idx].std()
            input_stds.append(feature_std.item())
        
        # 적어도 하나의 feature는 충분한 변동성을 가져야 함
        #min_input_std = min(input_stds)
        #has_sufficient_input_variation = min_input_std > min_std_threshold * 0.1
        
        # NaN/Inf 체크
        has_nan_inf = (torch.isnan(test_data_output_1[i]).any() or 
                      torch.isinf(test_data_output_1[i]).any() or
                      torch.isnan(test_data_input[i]).any() or 
                      torch.isinf(test_data_input[i]).any())
        
        # 유효성 검사
        is_valid = (output_std >= min_std_threshold and 
                   not has_nan_inf and
                   not torch.isnan(output_std))
        
        if is_valid:
            valid_indices.append(i)
            
            # Task별 출력 정규화 적용
            normalized_output = (test_data_output_1[i, :, :] - output_mean) / output_std
            
            filtered_input.append(test_data_input[i])
            filtered_output.append(normalized_output)
            
            # 진행상황 출력
            if len(valid_indices) % 1000 == 0:
                print(f"   Processed {i+1}/{original_size} tasks, Valid: {len(valid_indices)}")
        
        # 디버그: 처음 몇 개 invalid 샘플 정보 출력
        elif len(valid_indices) < 10:
            print(f"   Task {i}: Invalid - output_std={output_std:.2e}, "
                  f"has_nan_inf={has_nan_inf}")
    
    if not valid_indices:
        raise ValueError("❌ No valid samples found after filtering!")
    
    # 필터링된 데이터를 tensor로 변환
    test_data_input = torch.stack(filtered_input)      # [valid_samples, 61, 9]
    test_data_output_1 = torch.stack(filtered_output)  # [valid_samples, 61, 1]
    
    filtered_size = len(valid_indices)
    print(f"✅ Filtering completed:")
    print(f"   Original tasks: {original_size}")
    print(f"   Valid tasks: {filtered_size}")
    print(f"   Filtering ratio: {filtered_size/original_size*100:.1f}%")
    print(f"   Final data shapes: Input {test_data_input.shape}, Output {test_data_output_1.shape}")
    
    # MAML 모델 설정
    print(f"🤖 Setting up MAML model...")
    print(f"   Input features: {input_features}")
    print(f"   Hidden layer size: {layer_length}")
    print(f"   Inner loop steps: {inner}")
    
    # GPU로 데이터 이동 (최적화)
    print(f"Input shape: {test_data_input.shape}")
    print(f"Output shape: {test_data_output_1.shape}")
    test_data_input = test_data_input.to(device)
    test_data_output_1 = test_data_output_1.to(device)
    
    # 입력 차원 동적 결정
    input_features = test_data_input.shape[2]
    print(f"Input features: {input_features}")
    
    # 최적화된 MAML 모델 생성 (더 보수적인 학습률)
    print("🤖 Creating optimized MAML model...")
    calculated_inner_lr = 0.001 / innerdiv
    print(f"📊 Learning rates:")
    print(f"   Inner LR: 0.001 / {innerdiv} = {calculated_inner_lr}")
    print(f"   Meta LR: 0.0001")

    maml2 = OptimizedMAML(
        model=MAMLModel_3hidden(in_features=input_features, layer_length=layer_length),
        dataset_in=test_data_input,
        dataset_out=test_data_output_1,
        inner_lr=calculated_inner_lr,  # 계산된 inner learning rate
        meta_lr=0.0001,  # 더 작은 meta learning rate
        inner_steps=inner,  # 기본값 유지
        tasks_per_meta_batch=meta  # 더 작은 배치 크기로 안정성 증대
    )

    # 기존 학습된 모델 로드
    start_iteration = 0
    loaded_model = False

    if args.resume:
        # 사용자가 직접 지정한 모델 로드
        if os.path.exists(args.resume):
            start_iteration = extract_iteration_from_filename(args.resume, layer_length, inner)
            loaded_model = load_pretrained_model(maml2.model, args.resume, device)
        else:
            print(f"❌ Specified model file not found: {args.resume}")
            print("   Starting from scratch...")
    elif args.auto_resume:
        # 자동으로 최신 모델 찾아서 로드
        model_path, iteration = find_pretrained_model(pretrained_models_dir, layer_length, inner, innerdiv, meta, data_type)
        if model_path:
            start_iteration = iteration
            loaded_model = load_pretrained_model(maml2.model, model_path, device)

    if loaded_model:
        print(f"🔄 Resuming training from iteration: {start_iteration}")
        print(f"   Additional iterations to train: {additional_iterations}")
        print(f"   Target final iteration: {start_iteration + additional_iterations}")
    else:
        print(f"🆕 Starting training from scratch")
        print(f"   Total iterations to train: {additional_iterations}")

    # 모델 파라미터 초기화 체크 (새로 시작하는 경우만)
    if not loaded_model:
        for name, param in maml2.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: Found NaN/Inf in model parameter {name}")
                # 파라미터를 다시 초기화
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)
    
    # 훈련 실행
    for chunk in range(1, num_chunks + 1):
        # 현재 누적 iteration 계산
        current_iteration = start_iteration + (chunk * chunk_size)
        chunk_range_start = start_iteration + ((chunk-1) * chunk_size)
        chunk_range_end = current_iteration

        print(f"▶️ Starting iteration chunk {chunk}/{num_chunks}")
        print(f"   Cumulative iterations: [{chunk_range_start} → {chunk_range_end}]")

        # GPU utilization 측정 시작
        torch.cuda.synchronize()
        chunk_start_time = time.time()
        
        # 안정적인 메인 루프 실행 (NaN 감지 포함)
        try:
            # 훈련 전 파라미터 상태 체크
            param_check_passed = True
            for name, param in maml2.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"⚠️ NaN/Inf detected in {name} before training chunk {chunk}")
                    param_check_passed = False
            
            if not param_check_passed:
                print("⚠️ Reinitializing model due to NaN/Inf in parameters")
                reinit_inner_lr = 0.005 / innerdiv
                print(f"   Reinit Inner LR: 0.005 / {innerdiv} = {reinit_inner_lr}")
                maml2 = OptimizedMAML(
                    model=MAMLModel_3hidden(in_features=input_features, layer_length=layer_length),
                    dataset_in=test_data_input,
                    dataset_out=test_data_output_1,
                    inner_lr=0.000005,
                    meta_lr=0.00005,
                    inner_steps=inner,
                    tasks_per_meta_batch=32
                )
            
            maml2.main_loop_optimized(num_iterations=chunk_size)
        except Exception as e:
            print(f"⚠️ 병렬 처리 실패, 순차적 처리로 전환: {e}")
            try:
                maml2.main_loop_sequential(num_iterations=chunk_size)
            except Exception as e2:
                print(f"⚠️ 순차 처리도 실패: {e2}")
                print("⚠️ 학습률을 더 낮춰서 재시도...")
                maml2.inner_lr *= 0.5
                maml2.meta_lr *= 0.5
                maml2.main_loop_sequential(num_iterations=chunk_size//2)
        
        # GPU utilization 측정 종료
        torch.cuda.synchronize()
        chunk_end_time = time.time()
        
        # 성능 통계 출력
        chunk_time = chunk_end_time - chunk_start_time
        print(f"⏱️ Chunk {chunk} completed in {chunk_time:.2f}s")
        print(f"📈 Average time per iteration: {chunk_time/chunk_size:.4f}s")
        
        # GPU 메모리 사용량 출력
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"💾 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        # 체크포인트 저장 (누적 iteration 사용)
        checkpoint_path = f"{checkpoint_dir}/{data_type}_innerdiv{innerdiv}_meta{meta}_intratopology_519traintask_full1DMAML_weights_3hidden_({layer_length})_{current_iteration}_inner{inner}_upgraded.pth"
        torch.save(maml2.model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"   Cumulative iteration: {current_iteration}")

    # 최종 모델 저장 (누적 iteration 사용)
    final_iteration = start_iteration + additional_iterations
    final_model_path = f"{pretrained_models_dir}/{data_type}_innerdiv{innerdiv}_meta{meta}_intratopology_519traintask_full1DMAML_weights_3hidden_({layer_length})_{final_iteration}_inner{inner}_upgraded.pth"
    torch.save(maml2.model.state_dict(), final_model_path)
    print(f"\n🏁 Training complete!")
    print(f"   Final model saved to: {os.path.basename(final_model_path)}")
    print(f"   Final cumulative iteration: {final_iteration}")
    
    # GPU 메모리 정리
    del maml2, test_data_input, test_data_output_1
    torch.cuda.empty_cache()
    
    # 전체 실행 시간 출력
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"   Started from iteration: {start_iteration}")
    print(f"   Trained additional iterations: {additional_iterations}")
    print(f"   Final iteration: {final_iteration}")

if __name__ == "__main__":
    main()