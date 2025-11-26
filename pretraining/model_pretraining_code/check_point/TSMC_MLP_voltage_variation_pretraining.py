#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import pandas as pd
import random
import sys
import time
import argparse

# Import paths
sys.path.append('../../model_code/')

# Import MLP utilities
from mlp_utils import MLP_pretraining

def load_and_preprocess_data(temp, data_type, corner='ff', device='cuda'):
    """Load and preprocess data for MLP1 training"""
    from voltage_variation_pretraining_utils import load_tsmc_voltage_data, preprocess_voltage_data

    # Load data
    test_data_input, test_data_output = load_tsmc_voltage_data(corner, temp, data_type)

    # Preprocess (unified with MAML, but returns feature stats for MLP checkpoint saving)
    test_data_input, test_data_output, feature_means, feature_stds = preprocess_voltage_data(
        test_data_input, test_data_output, device=device, return_feature_stats=True
    )

    return test_data_input, test_data_output, feature_means, feature_stds

def validate_model(model, val_input, val_output, num_samples=100):
    """Validate model to check if it's producing varied predictions"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(val_input))):
            pred = model(val_input[i])
            predictions.append(pred.cpu())
    
    predictions = torch.cat(predictions)
    pred_std = predictions.std().item()
    pred_mean = predictions.mean().item()
    pred_min = predictions.min().item()
    pred_max = predictions.max().item()
    
    print(f"📊 Validation Statistics:")
    print(f"   Mean: {pred_mean:.6f}")
    print(f"   Std:  {pred_std:.6f}")
    print(f"   Min:  {pred_min:.6f}")
    print(f"   Max:  {pred_max:.6f}")
    
    if pred_std < 1e-6:
        print("⚠️ WARNING: Model is producing nearly identical predictions!")
    
    return pred_std > 1e-6  # Return True if predictions are varied

def train_mlp1_model(num_iterations=100000, temp=25, data_type='transition', corner='ff', model_type='aadam', device='cuda'):
    """Train MLP1 model on merged data"""
    print(f"\nStarting MLP1 training with {num_iterations} iterations")

    # Set hidden size based on model type
    hidden_size = 256 if model_type == 'aadam' else 40

    # Load and preprocess data
    test_data_input, test_data_output, feature_means, feature_stds = load_and_preprocess_data(temp, data_type, corner, device)

    # Split data for validation
    num_tasks = len(test_data_input)
    val_size = int(0.1 * num_tasks)
    val_indices = random.sample(range(num_tasks), val_size)
    train_indices = [i for i in range(num_tasks) if i not in val_indices]

    train_input = test_data_input[train_indices]
    train_output = test_data_output[train_indices]
    val_input = test_data_input[val_indices]
    val_output = test_data_output[val_indices]

    # Train MLP1 model using utility class (input_size=5 for voltage variation)
    print("Training MLP1 model...")
    mlp1 = MLP_pretraining(
        lr=1e-4,
        wd=0,
        dataset_in=train_input,
        dataset_out=train_output,
        iteration=num_iterations,
        hidden_size=hidden_size,
        input_size=5
    )

    # Note: voltage_variation doesn't use checkpoint directory
    mlp1_train_loss = mlp1.loop(checkpoint_dir='checkpoints')

    # Validate model
    print("\nValidating model...")
    is_valid = validate_model(mlp1.model, val_input, val_output)

    if not is_valid:
        print("Model validation failed - predictions are not varied!")
        print("   Consider checking:")
        print("   - Learning rate (try increasing)")
        print("   - Weight initialization")
        print("   - Data normalization")

    # Save model with additional metadata
    model_save_path = f'MLP_pretrained_model/pretrained_tsmc_{corner.upper()}_{temp}_test5_{data_type}_{model_type}_{num_iterations}.pth'
    torch.save({
        'model_state_dict': mlp1.model.state_dict(),
        'optimizer_state_dict': mlp1.optimizer.state_dict(),
        'train_loss': mlp1_train_loss,
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'output_normalized': False,
    }, model_save_path)

    print(f"MLP1 - Train Loss: {mlp1_train_loss:.6f}")
    print(f"Model saved to: {model_save_path}")

    results = {
        'mlp1_train_loss': mlp1_train_loss,
        'model_path': model_save_path,
        'validation_passed': is_valid
    }

    return results

def main():
    """Main function for MLP1 training"""
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='MLP TSMC Multi-Temperature Training')
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='GPU device ID (default: 1)')
    parser.add_argument('--temperatures', type=int, nargs='+', default=[0,25,50,75,100],
                        help='List of temperatures to train (default: [0,25,50,75,100])')
    parser.add_argument('--data_type', type=str, default='cell',
                        help='Data type: cell/transition (default: cell)')
    parser.add_argument('--corner', type=str, default='ff',
                        help='Corner condition: ss/ff/tt (default: ff)')
    parser.add_argument('--num_iterations', type=int, default=30000,
                        help='Number of training iterations (default: 30000)')
    parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                        help='Model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')
    args = parser.parse_args()

    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    temp_list = args.temperatures
    data_type = args.data_type.lower()
    corner = args.corner.lower()
    num_iterations = args.num_iterations
    model_type = args.model_type.lower()

    print("\n🚀 Starting MLP1 training on TSMC voltage variation dataset")
    print(f"\n⚙️ Training configuration:")
    print(f"   Data type: {data_type}")
    print(f"   Model type: {model_type} (hidden_size={'256' if model_type=='aadam' else '40'})")
    print(f"   Corner: {corner.upper()}")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Temperatures: {temp_list}")
    print(f"   Iterations per temperature: {num_iterations}")
    print("\n📝 Fixed version with:")
    print("   - No output normalization")
    print("   - Xavier weight initialization")
    print("   - Dropout enabled")
    print("   - Gradient clipping")
    print("   - Learning rate scheduling")
    print("   - Early stopping")

    all_results = {}

    for temp in temp_list:
        print(f"\n{'='*60}")
        print(f"🌡️ Training model for temperature: {temp}°C")
        print(f"{'='*60}")

        try:
            results = train_mlp1_model(num_iterations=num_iterations, temp=temp, data_type=data_type, corner=corner, model_type=model_type, device=device)

            print(f"\n📊 Training Results for temp={temp}°C:")
            print(f"   MLP1 Train Loss: {results['mlp1_train_loss']:.6f}")
            print(f"   Model saved to: {results['model_path']}")
            print(f"   Validation passed: {results['validation_passed']}")

            all_results[temp] = results

        except Exception as e:
            print(f"❌ Error during training for temp={temp}: {e}")
            import traceback
            traceback.print_exc()
            all_results[temp] = None

    # Summary of all results
    print(f"\n{'='*60}")
    print("📊 Summary of All Temperature Training Results:")
    print(f"{'='*60}")
    for temp, results in all_results.items():
        if results:
            print(f"Temperature {temp}°C: Loss={results['mlp1_train_loss']:.6f}, Validation={results['validation_passed']}")
        else:
            print(f"Temperature {temp}°C: Training failed")

    return all_results

if __name__ == "__main__":
    main()