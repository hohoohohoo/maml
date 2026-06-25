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

# MAML import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../model_code'))
from mlp_maml import OptimizedMAML, MAMLModel_3hidden

# Import utility functions (utils is in parent directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.maml_utils import (extract_iteration_from_filename, find_pretrained_model, load_pretrained_model,
                        normalize_input_features, normalize_and_filter_tasks)
from utils.dataset_config import get_dataset_config, print_available_datasets, load_dataset_by_config

def main():
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='MAML Unified Pretraining - Multiple Dataset Configurations')
    parser.add_argument('--dataset_config', type=int, required=True, choices=[0, 1, 2, 3, 4, 5],
                        help='Dataset configuration: 0=ASAP7 intra, 1=ASAP7 agnostic, 2=TSMC intra, '
                             '3=TSMC agnostic, 4=TSMC intra 2-D V×T, 5=TSMC agnostic 2-D V×T')
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
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID (default: 0)')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='Number of training iterations (default: 300000)')
    # Loss logging options
    parser.add_argument('--enable_loss_logging', action='store_true',
                        help='Enable training loss logging at specified intervals')
    parser.add_argument('--loss_log_every', type=int, default=1000,
                        help='Log training loss every N iterations (default: 1000)')
    parser.add_argument('--loss_log_dir', type=str, default=None,
                        help='Directory to save loss logs (default: loss_logs/)')
    args = parser.parse_args()

    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # GPU 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # 학습 설정
    additional_iterations = args.num_iterations  # 추가로 학습할 iteration 수
    chunk_size = 10000
    num_chunks = additional_iterations // chunk_size

    # GPU utilization 모니터링
    start_time = time.time()

    layer_length = 40
    inner = args.inner
    innerdiv = args.innerdiv
    meta = args.meta
    data_type = args.data_type.lower()
    dataset_config_id = args.dataset_config

    # 데이터셋 설정 가져오기
    dataset_config = get_dataset_config(dataset_config_id)
    tech = dataset_config['tech']
    topology_type = dataset_config['topology_type']
    dataset_name = dataset_config['name']

    print(f"\n🚀 Processing {dataset_name} {data_type} dataset")
    print_available_datasets()

    # 설정값 출력
    print(f"\n⚙️ Training configuration:")
    print(f"   Dataset config: {dataset_config_id} ({dataset_name})")
    print(f"   Data type: {data_type}")
    print(f"   Technology: {tech}")
    print(f"   Topology type: {topology_type}")
    print(f"   Inner loop steps: {inner}")
    print(f"   Inner learning rate divisor: {innerdiv} (inner_lr = 0.001/{innerdiv} = {0.001/innerdiv})")
    print(f"   Tasks per meta batch: {meta}")
    print(f"   Iterations: {args.num_iterations}")
    print(f"   GPU: {args.gpu}")

    # 모델 디렉토리 경로
    pretrained_models_dir = "../../../pretrained_models/training_loss_taskdivide_all"
    checkpoint_dir = "../../../pretrained_models/checkpoints/training_loss_taskdivide_all_checkpoints"

    # 디렉토리 생성 (없으면)
    os.makedirs(pretrained_models_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 데이터 로드 및 전처리
    print("📊 Loading and preprocessing data...")

    # 데이터셋 설정에 따라 데이터 로드
    test_data_input, test_data_output_1 = load_dataset_by_config(dataset_config_id, data_type)

    # 2-D V×T datasets arrive as [N, V, T, 9] / [N, V, T, 1]. The downstream MAML
    # code expects [N, samples_per_task, F] / [N, samples_per_task, 1]. Flatten
    # V×T → V*T (= 61 * 6 = 366 typical for train) so OptimizedMAML can sample K
    # support points across the full V×T plane (identical methodology to 1-D).
    if test_data_input.dim() == 4:
        n_tasks, n_v, n_t, n_feat = test_data_input.shape
        test_data_input = test_data_input.reshape(n_tasks, n_v * n_t, n_feat)
        print(f"   Flattened 2-D V×T input  to: {test_data_input.shape}")
    if test_data_output_1.dim() == 4:
        n_tasks, n_v, n_t, n_out = test_data_output_1.shape
        test_data_output_1 = test_data_output_1.reshape(n_tasks, n_v * n_t, n_out)
        print(f"   Flattened 2-D V×T output to: {test_data_output_1.shape}")

    print(f"📊 Data shapes (before normalize): Input {test_data_input.shape}, Output {test_data_output_1.shape}")

    # 9차원 입력 특성:
    # 0: a_param, 1: b_param, 2: c_param, 3: temperature, 4: voltage,
    # 5: additional_dim, 6: delay_indicator, 7: index_1_val(slew), 8: index_2_val(load_cap)

    # 정규화할 특성들: slew, load_cap, temperature, voltage
    normalize_indices = [7, 8, 3, 4]  # index_1(slew), index_2(load_cap), temperature, voltage
    feature_names = ['slew', 'load_cap', 'temperature', 'voltage']

    # Normalize input features using utility function
    test_data_input = normalize_input_features(test_data_input, normalize_indices, feature_names)

    # Normalize and filter tasks using utility function — keep output as 3-D [N, S, 1]
    # because normalize_and_filter_tasks does data_output[i, :, :] indexing.
    test_data_input, test_data_output_1 = normalize_and_filter_tasks(test_data_input, test_data_output_1, min_std_threshold=1e-6)

    # Now squeeze trailing 1 to match 1-D pipeline shape [N, S].
    if test_data_output_1.dim() == 3 and test_data_output_1.shape[-1] == 1:
        test_data_output_1 = test_data_output_1.squeeze(-1)

    # MAML 모델 설정
    print(f"🤖 Setting up MAML model...")
    input_features = test_data_input.shape[2]
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

    # Build loss logging configuration
    loss_logging_config = {
        'enabled': args.enable_loss_logging,
        'log_every': args.loss_log_every,
        'save_dir': args.loss_log_dir
    }

    # 최적화된 MAML 모델 생성 (더 보수적인 학습률)
    print("🤖 Creating optimized MAML model...")
    calculated_inner_lr = 0.001 / innerdiv
    print(f"📊 Learning rates:")
    print(f"   Inner LR: 0.001 / {innerdiv} = {calculated_inner_lr}")
    print(f"   Meta LR: 0.0001")
    if args.enable_loss_logging:
        print(f"   Loss logging: enabled (every {args.loss_log_every} iterations)")

    maml2 = OptimizedMAML(
        model=MAMLModel_3hidden(in_features=input_features, layer_length=layer_length),
        dataset_in=test_data_input,
        dataset_out=test_data_output_1,
        inner_lr=calculated_inner_lr,  # 계산된 inner learning rate
        meta_lr=0.0001,  # 더 작은 meta learning rate
        inner_steps=inner,  # 기본값 유지
        tasks_per_meta_batch=meta,  # 더 작은 배치 크기로 안정성 증대
        loss_logging_config=loss_logging_config
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
        model_path, iteration = find_pretrained_model(pretrained_models_dir, layer_length, inner, innerdiv, meta, data_type, tech=tech)
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

    # 모델 파일명 결정 (topology_type에 따라)
    if topology_type == 'intra':
        model_suffix = 'intratopology' if tech == 'asap7' else 'intra_topology'
    else:  # agnostic
        model_suffix = 'topology_agnostic'

    # tech suffix 추가
    tech_suffix = '' if tech == 'asap7' else '_tsmc'

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

        # Calculate starting iteration for this chunk (for loss logging)
        chunk_start_iteration = start_iteration + ((chunk-1) * chunk_size)

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
                    inner_lr=reinit_inner_lr,
                    meta_lr=0.00005,
                    inner_steps=inner,
                    tasks_per_meta_batch=meta,
                    loss_logging_config=loss_logging_config
                )

            maml2.main_loop_optimized(num_iterations=chunk_size, start_iteration=chunk_start_iteration)
        except Exception as e:
            print(f"⚠️ 병렬 처리 실패, 순차적 처리로 전환: {e}")
            try:
                maml2.main_loop_sequential(num_iterations=chunk_size, start_iteration=chunk_start_iteration)
            except Exception as e2:
                print(f"⚠️ 순차 처리도 실패: {e2}")
                print("⚠️ 학습률을 더 낮춰서 재시도...")
                maml2.inner_lr *= 0.5
                maml2.meta_lr *= 0.5
                maml2.main_loop_sequential(num_iterations=chunk_size//2, start_iteration=chunk_start_iteration)

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
        checkpoint_path = f"{checkpoint_dir}/{data_type}_innerdiv{innerdiv}_meta{meta}_{model_suffix}_519traintask_full1DMAML_weights_3hidden_({layer_length})_{current_iteration}_inner{inner}_upgraded{tech_suffix}_2d.pth"
        torch.save(maml2.model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"   Cumulative iteration: {current_iteration}")

    # 최종 모델 저장 (누적 iteration 사용)
    final_iteration = start_iteration + additional_iterations
    final_model_path = f"{pretrained_models_dir}/{data_type}_innerdiv{innerdiv}_meta{meta}_{model_suffix}_519traintask_full1DMAML_weights_3hidden_({layer_length})_{final_iteration}_inner{inner}_upgraded{tech_suffix}_2d.pth"
    torch.save(maml2.model.state_dict(), final_model_path)
    print(f"\n🏁 Training complete!")
    print(f"   Final model saved to: {os.path.basename(final_model_path)}")
    print(f"   Final cumulative iteration: {final_iteration}")

    # Save loss log if enabled
    if args.enable_loss_logging and maml2.iteration_loss_log:
        loss_log_dir = args.loss_log_dir or "../../../pretrained_models/loss_logs_maml"
        os.makedirs(loss_log_dir, exist_ok=True)
        loss_log_filename = f"loss_log_maml_{tech}_{model_suffix}_{data_type}_innerdiv{innerdiv}_meta{meta}_iter{final_iteration}_inner{inner}_2d.json"
        loss_log_path = os.path.join(loss_log_dir, loss_log_filename)
        maml2.save_loss_log(loss_log_path)

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
