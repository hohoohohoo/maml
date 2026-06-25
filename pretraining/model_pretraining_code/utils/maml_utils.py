#!/usr/bin/env python
# coding: utf-8

"""
MAML Pretraining Utility Functions

This module contains common utility functions used across MAML pretraining scripts:
- extract_iteration_from_filename: Extract iteration number from model filename
- find_pretrained_model: Find existing pretrained models in directory
- load_pretrained_model: Load pretrained model with validation
- normalize_input_features: Normalize selected input features with detailed logging
- normalize_and_filter_tasks: Normalize outputs and filter valid tasks with validation
- get_dataset_config: Get dataset configuration for unified pretraining
- load_dataset_by_config: Load dataset based on configuration
"""

import os
import torch
import glob
import re


def extract_iteration_from_filename(filepath, layer_length, inner):
    """
    파일명에서 iteration 번호를 추출합니다.
    ASAP7과 TSMC 두 가지 파일명 패턴을 모두 지원합니다.

    Args:
        filepath: 모델 파일 경로
        layer_length: hidden layer 크기
        inner: inner loop steps

    Returns:
        int: iteration 번호 (없으면 0)
    """
    filename = os.path.basename(filepath)

    # ASAP7 패턴: ..._3hidden_(40)_100000_inner1_upgraded.pth
    # TSMC 패턴: ..._3hidden_(40)_100000_inner1_upgraded_tsmc.pth 또는 ..._3hidden_40_100000_inner1_upgraded_tsmc.pth
    patterns = [
        rf'_\({layer_length}\)_(\d+)_inner{inner}_upgraded\.pth',           # ASAP7 with parentheses
        rf'_\({layer_length}\)_(\d+)_inner{inner}_upgraded_tsmc\.pth',      # TSMC with parentheses
        rf'_{layer_length}_(\d+)_inner{inner}_upgraded_tsmc\.pth',          # TSMC without parentheses
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            iteration = int(match.group(1))
            print(f"📊 Extracted iteration from filename: {iteration}")
            return iteration

    print(f"⚠️ Could not extract iteration from filename: {filename}")
    return 0


def find_pretrained_model(model_dir, layer_length, inner, innerdiv=100, meta=32, data_type='cell', tech='asap7'):
    """
    지정된 디렉토리에서 기존 학습된 모델을 찾습니다.
    ASAP7과 TSMC 두 가지 기술을 지원합니다.

    Args:
        model_dir: 모델 디렉토리 경로
        layer_length: hidden layer 크기
        inner: inner loop steps
        innerdiv: inner learning rate divisor
        meta: tasks per meta batch
        data_type: data type - cell/transition (default: 'cell')
        tech: technology node - 'asap7' or 'tsmc' (default: 'asap7')

    Returns:
        (filepath, iteration): 모델 파일 경로와 iteration 번호 튜플
    """
    models = []

    if tech.lower() == 'tsmc':
        # TSMC 패턴 (combined ckpt; 괄호 있는 / 없는 변형 모두 검색)
        patterns = [
            f"{data_type}_innerdiv{innerdiv}_meta{meta}_combined_519traintask_full1DMAML_weights_3hidden_({layer_length})_*_inner{inner}_upgraded_tsmc.pth",
            f"{data_type}_innerdiv{innerdiv}_meta{meta}_combined_519traintask_full1DMAML_weights_3hidden_{layer_length}_*_inner{inner}_upgraded_tsmc.pth",
        ]
    else:  # asap7
        # ASAP7 패턴: intra-topology와 topology-agnostic 모두 지원
        patterns = [
            f"{data_type}_innerdiv{innerdiv}_meta{meta}_intratopology_519traintask_full1DMAML_weights_3hidden_({layer_length})_*_inner{inner}_upgraded.pth",
            f"{data_type}_innerdiv{innerdiv}_meta{meta}_topology_agnostic_519traintask_full1DMAML_weights_3hidden_({layer_length})_*_inner{inner}_upgraded.pth",
        ]

    print(f"🔍 Searching for pretrained models ({tech.upper()})...")
    for pattern in patterns:
        search_pattern = os.path.join(model_dir, pattern)
        found = glob.glob(search_pattern)
        models.extend(found)
        if found:
            print(f"   Found {len(found)} model(s) with pattern: {pattern}")

    if not models:
        print(f"⚠️ No pretrained model found in {model_dir}")
        return None, 0

    # 모델들의 iteration 추출
    model_iterations = []
    for model_path in models:
        iteration = extract_iteration_from_filename(model_path, layer_length, inner)
        if iteration > 0:
            model_iterations.append((model_path, iteration))

    if not model_iterations:
        print(f"⚠️ No valid pretrained model found")
        return None, 0

    # 가장 큰 iteration을 가진 모델 선택
    best_model = max(model_iterations, key=lambda x: x[1])
    print(f"✅ Found pretrained model: {os.path.basename(best_model[0])}")
    print(f"   Iteration: {best_model[1]}")

    return best_model


def load_pretrained_model(model, filepath, device):
    """
    학습된 모델을 로드합니다.

    Args:
        model: MAML 모델 객체
        filepath: 모델 파일 경로
        device: 사용할 디바이스

    Returns:
        bool: 로드 성공 여부
    """
    try:
        print(f"📂 Loading pretrained model from: {filepath}")
        state_dict = torch.load(filepath, map_location=device)
        model.load_state_dict(state_dict)

        # 로드된 파라미터 검증
        has_nan_inf = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️ Warning: Found NaN/Inf in parameter {name}")
                has_nan_inf = True

        if has_nan_inf:
            print("❌ Pretrained model contains NaN/Inf values")
            return False

        print(f"✅ Successfully loaded pretrained model")
        return True

    except Exception as e:
        print(f"❌ Failed to load pretrained model: {e}")
        return False


def normalize_input_features(data_input, normalize_indices=[7, 8, 3, 4],
                             feature_names=['slew', 'load_cap', 'temperature', 'voltage']):
    """
    Normalize selected input features with detailed logging for MAML training.

    Args:
        data_input: Input tensor of shape [tasks, samples, features]
        normalize_indices: List of feature indices to normalize (default: [7,8,3,4])
        feature_names: List of feature names for logging (default: ['slew', 'load_cap', 'temperature', 'voltage'])

    Returns:
        normalized data_input (modified in-place)
    """
    print(f"🔧 Normalizing features: {feature_names}")
    print(f"   Feature indices: {normalize_indices}")

    input_features = data_input.shape[2]
    print(f"📊 Input shape ({input_features} features maintained): {data_input.shape}")

    # 각 특성별 정규화
    for feature_idx, feature_name in zip(normalize_indices, feature_names):
        feature_mean = data_input[:, :, feature_idx].mean()
        feature_std = data_input[:, :, feature_idx].std()

        print(f"📊 {feature_name} (idx {feature_idx}) stats: mean={feature_mean:.6f}, std={feature_std:.6f}")

        # 안전한 정규화 (std가 0이면 정규화 생략)
        if feature_std > 1e-8:
            data_input[:, :, feature_idx] = (data_input[:, :, feature_idx] - feature_mean) / feature_std
            print(f"   ✅ {feature_name} normalized")
        else:
            print(f"   ⚠️ {feature_name} std too small, skipping normalization")

    # 데이터 유효성 검사
    print(f"📊 Input data after normalization:")
    print(f"   Range: min={data_input.min():.6f}, max={data_input.max():.6f}")
    print(f"   Contains NaN: {torch.isnan(data_input).any()}")
    print(f"   Contains Inf: {torch.isinf(data_input).any()}")

    return data_input


def normalize_and_filter_tasks(data_input, data_output, min_std_threshold=1e-6):
    """
    Normalize outputs per task and filter out invalid tasks with detailed validation.

    Args:
        data_input: Input tensor of shape [tasks, samples, features]
        data_output: Output tensor of shape [tasks, samples, output_dim]
        min_std_threshold: Minimum standard deviation threshold (default: 1e-6)

    Returns:
        tuple: (filtered_input, filtered_output) as stacked tensors
    """
    print(f"📊 Output data stats before filtering:")
    print(f"   Range: min={data_output.min():.6f}, max={data_output.max():.6f}")
    print(f"   Contains NaN: {torch.isnan(data_output).any()}")
    print(f"   Contains Inf: {torch.isinf(data_output).any()}")

    valid_indices = []
    filtered_input = []
    filtered_output = []

    print(f"🔍 Filtering tasks with std >= {min_std_threshold}...")
    original_size = len(data_output)

    for i in range(len(data_output)):
        # Task별 출력 정규화
        output_mean = data_output[i, :, :].mean()
        output_std = data_output[i, :, :].std()

        # 입력 데이터의 변동성 체크 (각 feature별로)
        input_stds = []
        for feature_idx in range(data_input.shape[2]):
            feature_std = data_input[i, :, feature_idx].std()
            input_stds.append(feature_std.item())

        # NaN/Inf 체크
        has_nan_inf = (torch.isnan(data_output[i]).any() or
                      torch.isinf(data_output[i]).any() or
                      torch.isnan(data_input[i]).any() or
                      torch.isinf(data_input[i]).any())

        # 유효성 검사
        is_valid = (output_std >= min_std_threshold and
                   not has_nan_inf and
                   not torch.isnan(output_std))

        if is_valid:
            valid_indices.append(i)

            # Task별 출력 정규화 적용
            normalized_output = (data_output[i, :, :] - output_mean) / output_std

            filtered_input.append(data_input[i])
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
    filtered_input_tensor = torch.stack(filtered_input)
    filtered_output_tensor = torch.stack(filtered_output)

    filtered_size = len(valid_indices)
    print(f"✅ Filtering completed:")
    print(f"   Original tasks: {original_size}")
    print(f"   Valid tasks: {filtered_size}")
    print(f"   Filtering ratio: {filtered_size/original_size*100:.1f}%")
    print(f"   Final data shapes: Input {filtered_input_tensor.shape}, Output {filtered_output_tensor.shape}")

    return filtered_input_tensor, filtered_output_tensor
