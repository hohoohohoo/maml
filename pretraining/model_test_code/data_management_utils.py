"""
Common utility functions for extrapolation testing
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


def check_strict_continuity(data, threshold_ratio=0.2, min_points=5):
    """
    매우 엄격한 연속성을 체크합니다. 이웃한 점들 사이에서 하나라도 임계값을 넘는 점프가 있으면 불연속으로 판단합니다.
    """
    if len(data.shape) > 1:
        data = data.flatten()

    data = np.array(data)

    # NaN이나 inf 값 제거
    valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        return False, 0.0, [], 0.0, 0.0

    valid_data = data[valid_mask]

    if len(valid_data) < min_points:
        return False, 0.0, [], 0.0, 0.0

    # 데이터 범위 계산
    data_range = np.max(valid_data) - np.min(valid_data)
    if data_range == 0:
        return True, 1.0, [], 0.0, 0.0  # 상수 데이터는 연속으로 간주

    # 차분 계산 (이웃한 점들 간의 차이)
    diff = np.diff(valid_data)
    threshold = data_range * threshold_ratio

    # 절댓값으로 큰 점프 찾기
    abs_diff = np.abs(diff)
    max_jump = np.max(abs_diff)
    max_jump_ratio = max_jump / threshold if threshold > 0 else 0

    # 큰 점프 찾기
    large_jumps = abs_diff > threshold
    gap_indices = np.where(large_jumps)[0]

    # 엄격한 연속성 판단: 하나라도 큰 점프가 있으면 불연속
    is_continuous = len(gap_indices) == 0

    # 연속성 점수 계산
    continuity_score = 1.0 - (len(gap_indices) / len(diff)) if len(diff) > 0 else 1.0

    # 불연속 구간 정보
    gaps = []
    for idx in gap_indices:
        gaps.append({
            'position': idx,
            'jump_size': abs_diff[idx],
            'jump_ratio': abs_diff[idx] / threshold,
            'from_value': valid_data[idx],
            'to_value': valid_data[idx + 1],
            'threshold': threshold
        })

    return is_continuous, continuity_score, gaps, max_jump, max_jump_ratio


def check_output_continuity(output_data, threshold_ratio=0.1):
    """
    출력 데이터의 연속성을 체크합니다.
    """
    return check_strict_continuity(output_data, threshold_ratio)


def check_input_continuity(input_data, feature_idx=0, threshold_ratio=0.1):
    """
    입력 데이터의 특정 feature에 대해 연속성을 체크합니다.
    """
    if len(input_data.shape) == 2:  # [seq_len, features]
        data = input_data[:, feature_idx]
    elif len(input_data.shape) == 1:  # [seq_len]
        data = input_data
    else:
        raise ValueError(f"Unexpected input shape: {input_data.shape}")

    return check_strict_continuity(data, threshold_ratio)


def add_onehot_to_input(input_tensor, onehot_vec):
    """
    Add one-hot vector to input tensor for x_axis plotting
    input_tensor: [num_samples, 1] -> [num_samples, features]
    """
    num_samples = input_tensor.shape[0]

    # Create feature matrix for all samples
    feature_matrix = onehot_vec.unsqueeze(0).repeat(num_samples, 1)  # [features] -> [1, features] -> [num_samples, features]

    # Concatenate original input with feature matrix
    return torch.cat([input_tensor, feature_matrix], dim=1)


def analyze_continuity(test_data_input, test_data_output, threshold_ratio=0.18, max_check_samples=10000000):
    """
    테스트 데이터의 연속성을 분석합니다.

    Args:
        test_data_input: 입력 데이터
        test_data_output: 출력 데이터
        threshold_ratio: 연속성 임계값 비율
        max_check_samples: 최대 체크 샘플 수

    Returns:
        continuous_task_ids: 연속적인 태스크 ID 리스트
        discontinuous_task_ids: 불연속적인 태스크 ID 리스트
        continuity_analysis: 연속성 분석 결과
    """
    continuous_task_ids = []
    discontinuous_task_ids = []
    continuity_analysis = []

    num_check_samples = min(max_check_samples, len(test_data_output))
    check_indices = list(range(num_check_samples))

    print(f"처음 {num_check_samples}개 태스크에 대해 연속성 분석...")

    for i, task_id in enumerate(check_indices):
        if i % 5000 == 0:
            print(f"진행 상황: {i+1}/{num_check_samples}")

        try:
            # 입력 데이터 연속성 체크 (voltage feature - index 4)
            input_continuous, input_score, input_gaps, input_max_jump, input_max_ratio = check_input_continuity(
                test_data_input[task_id].cpu().numpy(), feature_idx=4, threshold_ratio=threshold_ratio
            )

            # 출력 데이터 연속성 체크
            output_continuous, output_score, output_gaps, output_max_jump, output_max_ratio = check_output_continuity(
                test_data_output[task_id].cpu().numpy(), threshold_ratio=threshold_ratio
            )

            # 전체적으로 연속적인지 판단 (입력과 출력 모두 연속적이어야 함)
            is_overall_continuous = input_continuous and output_continuous

            analysis_result = {
                'task_id': task_id,
                'input_continuous': input_continuous,
                'input_score': input_score,
                'input_gaps': len(input_gaps),
                'output_continuous': output_continuous,
                'output_score': output_score,
                'output_gaps': len(output_gaps),
                'overall_continuous': is_overall_continuous
            }

            continuity_analysis.append(analysis_result)

            if is_overall_continuous:
                continuous_task_ids.append(task_id)
            else:
                discontinuous_task_ids.append(task_id)

        except Exception as e:
            if i < 10:  # Print first few errors
                print(f"Error processing task {task_id}: {e}")
            continue

    print(f"\n📊 연속성 분석 완료!")
    print(f"   • 분석된 태스크: {len(continuity_analysis)}")
    print(f"   • 연속적인 태스크: {len(continuous_task_ids)} ({len(continuous_task_ids)/len(continuity_analysis)*100:.1f}%)")
    print(f"   • 불연속적인 태스크: {len(discontinuous_task_ids)} ({len(discontinuous_task_ids)/len(continuity_analysis)*100:.1f}%)")

    if len(continuous_task_ids) > 0:
        print(f"   • 연속적인 태스크 범위: {min(continuous_task_ids)} ~ {max(continuous_task_ids)}")
        print(f"   • 처음 10개 연속적인 태스크: {continuous_task_ids[:10]}")

    return continuous_task_ids, discontinuous_task_ids, continuity_analysis


def load_and_normalize_data(data_paths, normalize_features=[7, 8, 3, 4]):
    """
    훈련 데이터를 로드하고 정규화 통계를 계산합니다.

    Args:
        data_paths: 데이터 경로들의 리스트 [(input_path1, output_path1), (input_path2, output_path2), ...]
        normalize_features: 정규화할 feature 인덱스 리스트

    Returns:
        norm_stats: 정규화 통계
    """
    print("📊 Loading TRAINING dataset for normalization statistics...")

    # Load and concatenate all data
    all_inputs = []
    all_outputs = []

    for input_path, output_path in data_paths:
        train_input = torch.load(input_path)
        train_output = torch.load(output_path)

        # Add dimension to output if needed
        if len(train_output.shape) == 2:
            train_output = train_output.unsqueeze(-1)

        all_inputs.append(train_input)
        all_outputs.append(train_output)

    train_data_input = torch.cat(all_inputs, dim=0)
    train_data_output = torch.cat(all_outputs, dim=0)

    print(f"Train input shape: {train_data_input.shape}")
    print(f"Train output shape: {train_data_output.shape}")

    # Calculate normalization statistics
    feature_names = ['slew', 'load_cap', 'temperature', 'voltage']
    norm_stats = {}

    print("\n📊 Calculating normalization statistics from TRAINING data:")
    print("="*60)

    for feature_idx, feature_name in zip(normalize_features, feature_names):
        feature_mean = train_data_input[:, :, feature_idx].mean().item()
        feature_std = train_data_input[:, :, feature_idx].std().item()

        norm_stats[feature_idx] = {
            'name': feature_name,
            'mean': feature_mean,
            'std': feature_std
        }

        print(f"{feature_name} (idx {feature_idx}):")
        print(f"  Mean: {feature_mean:.6f}")
        print(f"  Std:  {feature_std:.6f}")

    # Clean up
    del train_data_input, train_data_output
    print("\n✅ Training data statistics calculated and training data removed from memory")

    return norm_stats


def apply_normalization(test_data_input, norm_stats, normalize_features=[7, 8, 3, 4]):
    """
    테스트 데이터에 정규화를 적용합니다.

    Args:
        test_data_input: 테스트 입력 데이터
        norm_stats: 정규화 통계
        normalize_features: 정규화할 feature 인덱스 리스트
    """
    print("\n🔧 Applying normalization to TEST data using TRAINING statistics...")
    print("="*60)

    for feature_idx in normalize_features:
        stats = norm_stats[feature_idx]
        feature_mean = stats['mean']
        feature_std = stats['std']
        feature_name = stats['name']

        # Before normalization
        test_mean_before = test_data_input[:, :, feature_idx].mean().item()
        test_std_before = test_data_input[:, :, feature_idx].std().item()

        # Apply normalization
        if feature_std > 1e-8:
            test_data_input[:, :, feature_idx] = (test_data_input[:, :, feature_idx] - feature_mean) / feature_std

            # After normalization
            test_mean_after = test_data_input[:, :, feature_idx].mean().item()
            test_std_after = test_data_input[:, :, feature_idx].std().item()

            print(f"{feature_name} (idx {feature_idx}):")
            print(f"  Training stats - Mean: {feature_mean:.6f}, Std: {feature_std:.6f}")
            print(f"  Test before norm - Mean: {test_mean_before:.6f}, Std: {test_std_before:.6f}")
            print(f"  Test after norm - Mean: {test_mean_after:.6f}, Std: {test_std_after:.6f}")
            print()
        else:
            print(f"⚠️ {feature_name} std too small ({feature_std:.8f}), skipping normalization")

    print("✅ Normalization complete using training statistics")
