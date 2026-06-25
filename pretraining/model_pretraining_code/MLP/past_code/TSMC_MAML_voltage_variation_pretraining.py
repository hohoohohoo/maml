#!/usr/bin/env python
# coding: utf-8

import os
import torch
import sys
import time
import argparse

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

def main():
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='MAML Multi-Temperature Training')
    parser.add_argument('--temperatures', type=int, nargs='+', default=[100,75,50,25,0],
                        help='List of temperatures to train (default: [100,75,50,25,0])')
    parser.add_argument('--layer_length', type=int, default=40,
                        help='Hidden layer size (default: 40)')
    parser.add_argument('--inner_step', type=int, default=1,
                        help='Inner loop steps (default: 1)')
    parser.add_argument('--innerdiv', type=int, default=10,
                        help='Inner learning rate divisor: inner_lr = 0.001/innerdiv (default: 10)')
    parser.add_argument('--meta', type=int, default=32,
                        help='Tasks per meta batch (default: 32)')
    parser.add_argument('--data_type', type=str, default='transition',
                        help='Data type: cell/transition (default: transition)')
    args = parser.parse_args()

    # 변수 초기화
    temperatures = args.temperatures
    layer_length = args.layer_length
    inner_step = args.inner_step
    innerdiv = args.innerdiv
    meta = args.meta
    data_type = args.data_type.lower()
    condition_type = 'ff'  # FF condition

    total_iterations = 30000
    chunk_size = 5000
    num_chunks = total_iterations // chunk_size

    print("="*60)
    print("🚀 MAML Multi-Temperature Training")
    print(f"📋 Condition: {condition_type.upper()}")
    print(f"🌡️ Temperatures: {temperatures}")
    print(f"🔄 Iterations per temperature: {total_iterations}")
    print("="*60)

    # 설정값 출력
    print(f"\n⚙️ Training configuration:")
    print(f"   Data type: {data_type}")
    print(f"   Layer length: {layer_length}")
    print(f"   Inner loop steps: {inner_step}")
    print(f"   Inner learning rate divisor: {innerdiv} (inner_lr = 0.001/{innerdiv} = {0.001/innerdiv})")
    print(f"   Tasks per meta batch: {meta}")
    
    # 각 온도에 대해 훈련 실행
    successful_temps = []
    failed_temps = []
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"🚀 Processing {data_type} dataset for {condition_type.upper()} at {temp}°C")
        print(f"{'='*60}")

        # GPU utilization 모니터링
        start_time = time.time()

        # 데이터 경로 설정
        base_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_test5(dim5)_TSMC/taskdivide_{condition_type}_{temp}"

        # 경로 확인
        if not os.path.exists(base_path):
            print(f"❌ Path not found: {base_path}")
            print(f"   Skipping temperature {temp}°C")
            failed_temps.append(temp)
            continue

        # 데이터 로드 및 전처리
        from voltage_variation_pretraining_utils import (
            load_tsmc_voltage_data, preprocess_voltage_data
        )

        try:
            test_data_input, test_data_output_1 = load_tsmc_voltage_data(condition_type, temp, data_type)

            # Preprocess (unified with MLP)
            test_data_input, test_data_output_1, valid_indices = preprocess_voltage_data(
                test_data_input, test_data_output_1, device=device, return_feature_stats=False
            )

            if test_data_input is None:
                print(f"❌ Preprocessing failed for {temp}°C")
                failed_temps.append(temp)
                continue

        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print(f"   Skipping temperature {temp}°C")
            failed_temps.append(temp)
            continue
        
        # 입력 차원 동적 결정 (5차원으로 변경)
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
            inner_steps=inner_step,  # 기본값 유지
            tasks_per_meta_batch=meta  # 더 작은 배치 크기로 안정성 증대
        )
        
        # 모델 파라미터 초기화 체크
        from voltage_variation_pretraining_utils import check_model_parameters, reinitialize_invalid_parameters

        if not check_model_parameters(maml2.model, f"MAML Model ({temp}°C)"):
            reinitialize_invalid_parameters(maml2.model)
        
        # 체크포인트 디렉토리 생성
        checkpoint_dir = f"../../pretrained_models/checkpoints/taskdivide_all_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 최종 모델 디렉토리 생성
        final_model_dir = f"../../pretrained_models/taskdivide_all"
        os.makedirs(final_model_dir, exist_ok=True)
        
        # 훈련 실행
        for chunk in range(1, num_chunks + 1):
            print(f"▶️ Starting iteration chunk {chunk}: [{(chunk-1)*chunk_size} → {chunk*chunk_size}]")
            
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
                        inner_lr=reinit_inner_lr,
                        meta_lr=0.00005,
                        inner_steps=inner_step,
                        tasks_per_meta_batch=meta
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
            
            # 체크포인트 저장
            checkpoint_path = f"{checkpoint_dir}/{data_type}_innerdiv{innerdiv}_meta{meta}_full1DMAML_weights_3hidden_({layer_length})_{chunk*chunk_size}_TSMC_{condition_type.upper()}_{temp}_test5(dim5)_inner{inner_step}.pth"
            torch.save(maml2.model.state_dict(), checkpoint_path)
            print(f"✅ Saved checkpoint: {checkpoint_path}")

        # 최종 모델 저장
        final_model_path = f"{final_model_dir}/{data_type}_innerdiv{innerdiv}_meta{meta}_full1DMAML_weights_3hidden_({layer_length})_{total_iterations}_TSMC_{condition_type.upper()}_{temp}_test5(dim5)_inner{inner_step}.pth"
        torch.save(maml2.model.state_dict(), final_model_path)
        print(f"🏁 Training complete. Model saved to: {final_model_path}")
        
        # GPU 메모리 정리
        del maml2, test_data_input, test_data_output_1
        torch.cuda.empty_cache()
        
        # 전체 실행 시간 출력
        total_time = time.time() - start_time
        print(f"\n🎉 Training for {temp}°C completed in {total_time:.2f}s")
        
        successful_temps.append(temp)
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        print("-"*60)
    
    # 최종 요약
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"✅ Successfully trained temperatures: {successful_temps}")
    if failed_temps:
        print(f"❌ Failed temperatures: {failed_temps}")
    print(f"📈 Total: {len(successful_temps)}/{len(temperatures)} temperatures completed")
    print("="*60)

if __name__ == "__main__":
    main()