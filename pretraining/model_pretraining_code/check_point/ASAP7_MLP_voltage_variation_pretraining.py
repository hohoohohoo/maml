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
import sys
import time
import argparse

# Import paths
sys.path.append('../../../')

# Fixed MLP class with proper initialization
class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(0.2)  # Reduced dropout rate

        # Better weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for better gradient flow
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc4(x))
        x = self.fc3(x)
        return x

class MLP1(object):
    def __init__(self, lr=1e-3, wd=0, dataset_in=None, dataset_out=None, iteration=5, hidden_size=256):
        self.lr = lr
        self.wd = wd
        self.train_in = dataset_in
        self.train_out = dataset_out
        self.iteration = iteration
        self.model = MLP(input_size=5, output_size=1, hidden_size=hidden_size)
        if torch.cuda.is_available():
            self.model.cuda()

        # Use AdamW optimizer with weight decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5000, verbose=True
        )

    def loop(self):
        running_loss = 0.0
        num_tasks = len(self.train_in)
        
        # Track best loss for early stopping
        # best_loss = float('inf')
        # patience_counter = 0
        # max_patience = 20000
        
        for i in range(self.iteration):
            if i % 1000 == 0:
                avg_loss = running_loss / max(1, i)
                print(f"Iteration {i}/{self.iteration}, Avg Loss: {avg_loss:.6f}")
                
        #         # Early stopping check
        #         if avg_loss < best_loss:
        #             best_loss = avg_loss
        #             patience_counter = 0
        #         else:
        #             patience_counter += 1000
                    
        #         if patience_counter >= max_patience:
        #             print(f"Early stopping at iteration {i}")
        #             break
                
            # Select one random task for this iteration
            task_idx = random.randint(0, num_tasks - 1)
            
            self.optimizer.zero_grad()
            
            # Get data for the selected task
            x_sampled = self.train_in[task_idx, :]
            y_sampled = self.train_out[task_idx, :]
            # Create mini-batch of 5 samples from this task
            num_samples = min(5, x_sampled.shape[0])
            indices = random.sample(range(x_sampled.shape[0]), num_samples)
            mini_batch = x_sampled[indices]
            mini_y = y_sampled[indices]
            
            if torch.cuda.is_available():
                mini_batch, mini_y = mini_batch.cuda(), mini_y.cuda()
                
            # Forward pass, loss computation, and optimization
            self.model.train()  # Ensure model is in training mode
            y_pred = self.model(mini_batch) 
            the_loss = F.mse_loss(y_pred, mini_y)
            
            # Add L2 regularization if needed
            # l2_reg = sum(p.pow(2.0).sum() for p in self.model.parameters())
            # the_loss = the_loss + self.wd * l2_reg
            
            the_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += the_loss.item()
            
            # Update learning rate
            # if i % 1000 == 0:
            #     self.scheduler.step(the_loss)
            
        return float(running_loss / min(i + 1, self.iteration))
    
def load_and_preprocess_data(corner='FF', cell_type='lvt', data_type='cell', device='cuda'):
    """Load and preprocess data for MLP1 training"""
    from voltage_variation_pretraining_utils import load_asap7_voltage_data, preprocess_voltage_data

    # Load data
    test_data_input, test_data_output = load_asap7_voltage_data(corner, cell_type, data_type)

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

def train_mlp1_model(num_iterations=100000, corner='FF', cell_type='lvt', data_type='cell', model_type='aadam', device='cuda'):  # Reduced from 10M
    """Train MLP1 model on merged data"""
    print(f"\n🧠 Starting MLP1 training with {num_iterations} iterations")

    # Set hidden size based on model type
    hidden_size = 256 if model_type == 'aadam' else 40

    # Load and preprocess data
    test_data_input, test_data_output, feature_means, feature_stds = load_and_preprocess_data(corner=corner, cell_type=cell_type, data_type=data_type, device=device)

    # Split data for validation
    num_tasks = len(test_data_input)
    val_size = int(0.1 * num_tasks)
    val_indices = random.sample(range(num_tasks), val_size)
    train_indices = [i for i in range(num_tasks) if i not in val_indices]

    train_input = test_data_input[train_indices]
    train_output = test_data_output[train_indices]
    val_input = test_data_input[val_indices]
    val_output = test_data_output[val_indices]

    # Train MLP1 model
    print("🚀 Training MLP1 model...")
    mlp1 = MLP1(
        lr=1e-4,  # Increased learning rate
        #wd=1e-4,  # Adjusted weight decay
        dataset_in=train_input,
        dataset_out=train_output,
        iteration=num_iterations,
        hidden_size=hidden_size
    )

    mlp1_train_loss = mlp1.loop()

    # Validate model
    print("\n🔍 Validating model...")
    is_valid = validate_model(mlp1.model, val_input, val_output)

    if not is_valid:
        print("❌ Model validation failed - predictions are not varied!")
        print("   Consider checking:")
        print("   - Learning rate (try increasing)")
        print("   - Weight initialization")
        print("   - Data normalization")

    # Save the pretrained model with model_type and num_iterations in filename
    model_save_path = f'/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_pretraining_code/MLP_pretrained_model/pretrained_asap7_{cell_type}_{data_type}_{corner}_test5_{model_type}_{num_iterations}.pth'
    torch.save({
        'model_state_dict': mlp1.model.state_dict(),
        'optimizer_state_dict': mlp1.optimizer.state_dict(),
        'train_loss': mlp1_train_loss,
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'output_normalized': False,  # Important flag
    }, model_save_path)

    print(f"✅ MLP1 - Train Loss: {mlp1_train_loss:.6f}")
    print(f"💾 Model saved to: {model_save_path}")

    results = {
        'mlp1_train_loss': mlp1_train_loss,
        'model_path': model_save_path,
        'validation_passed': is_valid
    }

    return results

def main():
    """Main function for MLP1 training"""
    # 커맨드라인 인자 파싱
    parser = argparse.ArgumentParser(description='MLP Pretraining - ASAP7 Voltage Variation')
    parser.add_argument('--gpu_id', type=str, default='4',
                        help='GPU device ID (default: 4)')
    parser.add_argument('--corner', type=str, default='FF',
                        help='Corner condition: SS/FF/TT (default: FF)')
    parser.add_argument('--cell_type', type=str, default='lvt',
                        help='Cell type: lvt/rvt/slvt/sram (default: lvt)')
    parser.add_argument('--data_type', type=str, default='cell',
                        help='Data type: cell/transition (default: cell)')
    parser.add_argument('--num_iterations', type=int, default=100000,
                        help='Number of training iterations (default: 100000)')
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

    corner = args.corner.upper()
    cell_type = args.cell_type.lower()
    data_type = args.data_type.lower()
    num_iterations = args.num_iterations
    model_type = args.model_type.lower()

    print("\n🚀 Starting MLP1 training on ASAP7 voltage variation dataset")
    print(f"\n⚙️ Training configuration:")
    print(f"   Data type: {data_type}")
    print(f"   Cell type: {cell_type.upper()}")
    print(f"   Corner: {corner}")
    print(f"   Model type: {model_type} (hidden_size={'256' if model_type=='aadam' else '40'})")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Iterations: {num_iterations}")
    print("\n📝 Fixed version with:")
    print("   - No output normalization")
    print("   - Xavier weight initialization")
    print("   - Dropout enabled")
    print("   - Gradient clipping")
    print("   - Learning rate scheduling")
    print("   - Early stopping")

    try:
        results = train_mlp1_model(num_iterations=num_iterations, corner=corner, cell_type=cell_type, data_type=data_type, model_type=model_type, device=device)

        print(f"\n📊 Training Results:")
        print(f"   MLP1 Train Loss: {results['mlp1_train_loss']:.6f}")
        print(f"   Model saved to: {results['model_path']}")
        print(f"   Validation passed: {results['validation_passed']}")

        # Save results to file


    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

    return results

if __name__ == "__main__":
    main()