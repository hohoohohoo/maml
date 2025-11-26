#!/usr/bin/env python
# coding: utf-8

"""
ASAP7 Voltage Variation Pretraining (Unified MLP/MAML)

Unified script for ASAP7 voltage variation pretraining supporting both MLP and MAML.
Select model framework with --model flag.
"""

import os
import torch
import random
import sys
import time
import argparse

# Import MLP utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../model_code'))
from networks import MLP_pretraining

def validate_model(model, val_input, num_samples=100):
    """Validate MLP model"""
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

    return pred_std > 1e-6

def train_mlp(args, device):
    """Train MLP model"""
    print("\n" + "="*80)
    print("🚀 Starting MLP Training")
    print("="*80)

    from utils.voltage_variation_pretraining_utils import load_asap7_voltage_data, preprocess_voltage_data

    # Load and preprocess data
    test_data_input, test_data_output = load_asap7_voltage_data(args.corner, args.cell_type, args.data_type)
    test_data_input, test_data_output, feature_means, feature_stds = preprocess_voltage_data(
        test_data_input, test_data_output, device=device, return_feature_stats=True
    )

    # Set hidden size based on model type
    hidden_size = 256 if args.model_type == 'aadam' else 40

    # Split data for validation
    num_tasks = len(test_data_input)
    val_size = int(0.1 * num_tasks)
    val_indices = random.sample(range(num_tasks), val_size)
    train_indices = [i for i in range(num_tasks) if i not in val_indices]

    train_input = test_data_input[train_indices]
    train_output = test_data_output[train_indices]
    val_input = test_data_input[val_indices]

    # Train MLP model
    print("🚀 Training MLP model...")
    mlp = MLP_pretraining(
        lr=1e-4,
        wd=0,
        dataset_in=train_input,
        dataset_out=train_output,
        iteration=args.num_iterations,
        hidden_size=hidden_size,
        input_size=5
    )

    mlp_train_loss = mlp.loop()

    # Validate model
    print("\n🔍 Validating model...")
    is_valid = validate_model(mlp.model, val_input)

    if not is_valid:
        print("❌ Model validation failed - predictions are not varied!")

    # Save model
    model_save_path = f'../../pretrained_models/MLP_pretrained_model/pretrained_asap7_{args.cell_type}_{args.data_type}_{args.corner}_test5_{args.model_type}_{args.num_iterations}.pth'
    os.makedirs('MLP_pretrained_model', exist_ok=True)

    torch.save({
        'model_state_dict': mlp.model.state_dict(),
        'optimizer_state_dict': mlp.optimizer.state_dict(),
        'train_loss': mlp_train_loss,
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'output_normalized': False,
    }, model_save_path)

    print(f"✅ MLP - Train Loss: {mlp_train_loss:.6f}")
    print(f"💾 Model saved to: {model_save_path}")
    print(f"🎯 Validation passed: {is_valid}")

def train_maml(args, device):
    """Train MAML model"""
    print("\n" + "="*80)
    print("🚀 Starting MAML Training")
    print("="*80)

    # Import MAML
    sys.path.append('../../model_code/')
    from maml_optimized import OptimizedMAML, MAMLModel_3hidden
    from utils.voltage_variation_pretraining_utils import (
        load_asap7_voltage_data, preprocess_voltage_data,
        check_model_parameters, reinitialize_invalid_parameters
    )

    # Load and preprocess data
    test_data_input, test_data_output = load_asap7_voltage_data(args.corner, args.cell_type, args.data_type)
    test_data_input, test_data_output, _ = preprocess_voltage_data(
        test_data_input, test_data_output, device=device, return_feature_stats=False
    )

    if test_data_input is None:
        print("❌ Preprocessing failed, exiting...")
        return

    # Model configuration
    input_features = test_data_input.shape[2]
    layer_length = args.layer_length
    calculated_inner_lr = 0.001 / args.innerdiv

    print(f"Input features: {input_features}")
    print(f"📊 Learning rates:")
    print(f"   Inner LR: 0.001 / {args.innerdiv} = {calculated_inner_lr}")
    print(f"   Meta LR: 0.0001")

    # Create MAML model
    print("🤖 Creating optimized MAML model...")
    maml = OptimizedMAML(
        model=MAMLModel_3hidden(in_features=input_features, layer_length=layer_length),
        dataset_in=test_data_input,
        dataset_out=test_data_output,
        inner_lr=calculated_inner_lr,
        meta_lr=0.0001,
        inner_steps=args.inner_step,
        tasks_per_meta_batch=args.meta
    )

    # Check model parameters
    if not check_model_parameters(maml.model, "MAML Model"):
        reinitialize_invalid_parameters(maml.model)

    # Training loop with checkpoints
    total_iterations = args.num_iterations
    chunk_size = 10000
    num_chunks = total_iterations // chunk_size

    # Create checkpoint directory
    checkpoint_dir = "../../pretrained_models/checkpoints/taskdivide_all_checkpoints"
    final_model_dir = "../../pretrained_models/taskdivide_all"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)

    start_time = time.time()

    for chunk in range(1, num_chunks + 1):
        print(f"▶️ Starting iteration chunk {chunk}: [{(chunk-1)*chunk_size} → {chunk*chunk_size}]")

        torch.cuda.synchronize()
        chunk_start_time = time.time()

        # Check parameters before training
        param_check_passed = True
        for name, param in maml.model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️ NaN/Inf detected in {name} before training chunk {chunk}")
                param_check_passed = False

        if not param_check_passed:
            print("⚠️ Reinitializing model due to NaN/Inf in parameters")
            reinit_inner_lr = 0.005 / args.innerdiv
            maml = OptimizedMAML(
                model=MAMLModel_3hidden(in_features=input_features, layer_length=layer_length),
                dataset_in=test_data_input,
                dataset_out=test_data_output,
                inner_lr=reinit_inner_lr,
                meta_lr=0.00005,
                inner_steps=args.inner_step,
                tasks_per_meta_batch=16
            )

        # Train
        try:
            maml.main_loop_optimized(num_iterations=chunk_size)
        except Exception as e:
            print(f"⚠️ Optimized loop failed, switching to sequential: {e}")
            try:
                maml.main_loop_sequential(num_iterations=chunk_size)
            except Exception as e2:
                print(f"⚠️ Sequential also failed: {e2}")
                print("⚠️ Reducing learning rate and retrying...")
                maml.inner_lr *= 0.5
                maml.meta_lr *= 0.5
                maml.main_loop_sequential(num_iterations=chunk_size//2)

        torch.cuda.synchronize()
        chunk_end_time = time.time()

        # Print statistics
        chunk_time = chunk_end_time - chunk_start_time
        print(f"⏱️ Chunk {chunk} completed in {chunk_time:.2f}s")
        print(f"📈 Average time per iteration: {chunk_time/chunk_size:.4f}s")

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"💾 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        # Save checkpoint
        checkpoint_path = f"{checkpoint_dir}/{args.data_type}_innerdiv{args.innerdiv}_meta{args.meta}_full1DMAML_weights_3hidden_({layer_length})_{chunk*chunk_size}_{args.cell_type.upper()}_{args.corner}_test5(dim5)_inner{args.inner_step}_fixed.pth"
        torch.save(maml.model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = f"{final_model_dir}/{args.data_type}_innerdiv{args.innerdiv}_meta{args.meta}_full1DMAML_weights_3hidden_({layer_length})_{total_iterations}_{args.cell_type.upper()}_{args.corner}_test5(dim5)_inner{args.inner_step}_fixed.pth"
    torch.save(maml.model.state_dict(), final_model_path)
    print(f"🏁 Training complete. Model saved to: {final_model_path}")

    # Cleanup
    del maml, test_data_input, test_data_output
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description='ASAP7 Voltage Variation Pretraining (MLP/MAML)')

    # Model selection
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'maml'],
                        help='Model framework: mlp or maml')

    # Common parameters
    parser.add_argument('--corner', type=str, default='FF', choices=['SS', 'FF', 'TT'],
                        help='Corner condition (default: FF)')
    parser.add_argument('--cell_type', type=str, default='lvt', choices=['lvt', 'rvt', 'slvt', 'sram'],
                        help='Cell type (default: lvt)')
    parser.add_argument('--data_type', type=str, default='cell', choices=['cell', 'transition'],
                        help='Data type (default: cell)')
    parser.add_argument('--gpu_id', type=str, default='4',
                        help='GPU device ID (default: 4)')
    parser.add_argument('--num_iterations', type=int, default=100000,
                        help='Number of training iterations (default: 100000)')

    # MLP-specific parameters
    parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                        help='[MLP] Model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')

    # MAML-specific parameters
    parser.add_argument('--layer_length', type=int, default=40,
                        help='[MAML] Hidden layer size (default: 40)')
    parser.add_argument('--inner_step', type=int, default=3,
                        help='[MAML] Inner loop steps (default: 3)')
    parser.add_argument('--innerdiv', type=int, default=10,
                        help='[MAML] Inner learning rate divisor (default: 10)')
    parser.add_argument('--meta', type=int, default=16,
                        help='[MAML] Meta batch size (default: 16)')

    args = parser.parse_args()

    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args.model == 'maml':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # Normalize parameters
    args.corner = args.corner.upper()
    args.cell_type = args.cell_type.lower()
    args.data_type = args.data_type.lower()
    if hasattr(args, 'model_type'):
        args.model_type = args.model_type.lower()

    # Print configuration
    print("\n⚙️ Training configuration:")
    print(f"   PDK: ASAP7")
    print(f"   Model: {args.model.upper()}")
    print(f"   Corner: {args.corner}")
    print(f"   Cell type: {args.cell_type.upper()}")
    print(f"   Data type: {args.data_type}")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Iterations: {args.num_iterations}")

    if args.model == 'mlp':
        print(f"   Model type: {args.model_type} (hidden={'256' if args.model_type=='aadam' else '40'})")
    else:  # maml
        print(f"   Layer length: {args.layer_length}")
        print(f"   Inner steps: {args.inner_step}")
        print(f"   Inner div: {args.innerdiv} (inner_lr = {0.001/args.innerdiv})")
        print(f"   Meta batch: {args.meta}")

    # Train based on model selection
    if args.model == 'mlp':
        train_mlp(args, device)
    else:  # maml
        train_maml(args, device)

if __name__ == "__main__":
    main()
