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

# Import MLP utilities
from mlp_utils import MLP_pretraining, normalize_features, normalize_outputs, save_model
    

def load_and_preprocess_data(data_type='cell', device='cuda'):
    """Load and preprocess data for MLP1 training"""
    print(f"Loading and preprocessing merged data...")
    data_dir = "/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_ex2/intra_topology_data_upgraded"
    test_data_input = torch.load(f"{data_dir}/{data_type}_intratopology_train_input.pth")
    test_data_output_1 = torch.load(f"{data_dir}/{data_type}_intratopology_train_output.pth")
    test_data_input2 = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_ex2/unified_invbuf/merged_invbuf_input_{data_type}.pth")
    test_data_output_2 = torch.load(f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_ex2/unified_invbuf/merged_invbuf_output_{data_type}.pth")
    test_data_input = torch.cat([test_data_input, test_data_input2], dim=0)
    test_data_output_1 = torch.cat([test_data_output_1, test_data_output_2], dim=0)

    # Normalize features using utility function
    test_data_input, feature_means, feature_stds = normalize_features(
        test_data_input, normalize_indices=[7, 8, 3, 4], num_features=9
    )

    # Normalize outputs using utility function
    test_data_output_1, output_means, output_stds, valid_indices = normalize_outputs(
        test_data_output_1, min_std_threshold=1e-6
    )

    # Keep only valid tasks
    if valid_indices:
        test_data_input = test_data_input[valid_indices]
        test_data_output_1 = test_data_output_1[valid_indices]

    # Move to GPU
    test_data_input = test_data_input.to(device)
    test_data_output_1 = test_data_output_1.to(device)

    return test_data_input, test_data_output_1, feature_means, feature_stds, output_means, output_stds


def train_mlp1_model(num_iterations=100000, data_type='cell', learning_rate=1e-4, model_type='aadam', device='cuda'):
    """Train MLP1 model on merged data"""
    print(f"\nStarting MLP1 training with {num_iterations} iterations")

    # Set hidden size based on model type
    hidden_size = 256 if model_type == 'aadam' else 40

    # Load and preprocess data
    test_data_input, test_data_output_1, feature_means, feature_stds, output_means, output_stds = load_and_preprocess_data(
        data_type=data_type, device=device
    )

    # Train MLP1 model using utility class
    print("Training MLP1 model...")
    mlp1 = MLP_pretraining(
        lr=learning_rate,
        wd=0,
        dataset_in=test_data_input,
        dataset_out=test_data_output_1,
        iteration=num_iterations,
        hidden_size=hidden_size,
        input_size=9
    )

    # Checkpoint directory
    checkpoint_dir = f'/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_pretraining_code/MLP_pretrained_model/checkpoints_intratopology_{data_type}_{model_type}_{num_iterations}'
    mlp1_train_loss = mlp1.loop(checkpoint_dir=checkpoint_dir)

    # Save model using utility function
    model_save_path = f'/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_pretraining_code/MLP_pretrained_model/pretrained_mlp1_intratopology_{data_type}_{model_type}_{num_iterations}.pth'
    save_model(mlp1.model, model_save_path, train_loss=mlp1_train_loss)

    print(f"MLP1 - Train Loss: {mlp1_train_loss:.6f}")

    results = {
        'mlp1_train_loss': mlp1_train_loss,
        'model_path': model_save_path
    }

    return results

def main():
    """Main function for MLP1 training"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MLP Pretraining - ASAP7 Intra Topology')
    parser.add_argument('--gpu_id', type=str, default='3',
                        help='GPU device ID (default: 3)')
    parser.add_argument('--data_type', type=str, default='cell',
                        help='Data type: cell/transition (default: cell)')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='Number of training iterations (default: 300000)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for Adam optimizer (default: 1e-4)')
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

    # Import paths
    sys.path.append('../../../')

    data_type = args.data_type.lower()
    model_type = args.model_type.lower()

    print(f"\n🚀 Starting MLP1 training on ASAP7 intra topology {data_type} dataset")
    print(f"⚙️ Training configuration:")
    print(f"   Data type: {data_type}")
    print(f"   Model type: {model_type} (hidden_size={'256' if model_type=='aadam' else '40'})")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Number of iterations: {args.num_iterations}")
    print(f"   Learning rate: {args.learning_rate}")

    try:
        results = train_mlp1_model(
            num_iterations=args.num_iterations,
            data_type=data_type,
            learning_rate=args.learning_rate,
            model_type=model_type,
            device=device
        )

        print(f"\n📊 Training Results:")
        print(f"   MLP1 Train Loss: {results['mlp1_train_loss']:.6f}")
        print(f"   Model saved to: {results['model_path']}")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

    return results

if __name__ == "__main__":
    main()