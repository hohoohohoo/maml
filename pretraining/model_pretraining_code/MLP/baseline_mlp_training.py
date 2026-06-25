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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../model_code'))
from baseline_mlp import MLP_pretraining

# Import utility functions (utils is in parent directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mlp_utils import normalize_features, normalize_outputs, save_model
from utils.dataset_config import get_dataset_config, print_available_datasets, load_dataset_by_config


def load_and_preprocess_data(config_id, data_type='cell', device='cuda'):
    """
    Load and preprocess data for MLP training based on dataset configuration

    Args:
        config_id (int): Dataset configuration ID (0-3)
        data_type (str): Data type - 'cell' or 'transition'
        device (str): Device to move data to

    Returns:
        tuple: (input_tensor, output_tensor, feature_means, feature_stds, output_means, output_stds)
    """
    print(f"Loading and preprocessing data...")

    # Load dataset using dataset_config module
    test_data_input, test_data_output_1 = load_dataset_by_config(config_id, data_type)

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


def train_mlp_model(config_id, num_iterations=100000, data_type='cell',
                    learning_rate=1e-4, model_type='aadam', device='cuda',
                    loss_logging_config=None):
    """
    Train MLP model on specified dataset configuration

    Args:
        config_id (int): Dataset configuration ID (0-3)
        num_iterations (int): Number of training iterations
        data_type (str): Data type - 'cell' or 'transition'
        learning_rate (float): Learning rate for Adam optimizer
        model_type (str): Model type - 'aadam' (hidden=256) or 'mlp' (hidden=40)
        device (str): Device to train on
        loss_logging_config (dict): Loss logging configuration

    Returns:
        dict: Training results including loss and model path
    """
    print(f"\nStarting MLP training with {num_iterations} iterations")

    # Set hidden size based on model type
    hidden_size = 256 if model_type == 'aadam' else 40

    # Get dataset configuration
    dataset_config = get_dataset_config(config_id)
    tech = dataset_config['tech']
    topology_type = dataset_config['topology_type']

    # Load and preprocess data
    test_data_input, test_data_output_1, feature_means, feature_stds, output_means, output_stds = load_and_preprocess_data(
        config_id=config_id, data_type=data_type, device=device
    )

    # Train MLP model using utility class
    print("Training MLP model...")
    mlp = MLP_pretraining(
        lr=learning_rate,
        wd=0,
        dataset_in=test_data_input,
        dataset_out=test_data_output_1,
        iteration=num_iterations,
        hidden_size=hidden_size,
        input_size=9,
        loss_logging_config=loss_logging_config
    )

    # Determine naming convention to match validation script expectations
    # Format: pretrained_{tech}{model_path_suffix}_{topology_name}_{data_type}_{model_type}_{num_iterations}.pth
    #
    # Configs 6/7 carry 'model_path_suffix' = '_combined'. That suffix
    # is appended right after the tech token so the resulting filename is
    # physically distinct from the legacy config-2/3 checkpoints, which lets
    # us run before/after comparisons without overwriting anything. Configs
    # 0..3 (no suffix in the dataset_config) keep their original filenames.
    if topology_type == 'intra':
        topology_name = 'intra_topology'
    else:  # agnostic
        topology_name = 'topology_agnostic'

    model_path_suffix = dataset_config.get('model_path_suffix', '')

    # Checkpoint directory
    checkpoint_dir = f'../../../pretrained_models/MLP_pretrained_model/training_loss_checkpoints_FAIRINIT_{tech}{model_path_suffix}_{topology_name}_{data_type}_{model_type}_{num_iterations}'
    mlp_train_loss = mlp.loop(checkpoint_dir=checkpoint_dir)

    # Save model using utility function
    # Naming convention matches test_dataset_config.py mlp_model_path patterns
    model_save_path = f'../../../pretrained_models/MLP_pretrained_model/training_loss_pretrained_FAIRINIT_{tech}{model_path_suffix}_{topology_name}_{data_type}_{model_type}_{num_iterations}.pth'
    save_model(mlp.model, model_save_path, train_loss=mlp_train_loss)

    print(f"MLP - Train Loss: {mlp_train_loss:.6f}")

    # Save loss log if enabled
    if loss_logging_config and loss_logging_config.get('enabled', False) and mlp.iteration_loss_log:
        loss_log_dir = loss_logging_config.get('save_dir') or f'../../../pretrained_models/loss_logs_mlp'
        import os
        os.makedirs(loss_log_dir, exist_ok=True)
        loss_log_filename = f"loss_log_mlp_{tech}{model_path_suffix}_{topology_name}_{data_type}_{model_type}_{num_iterations}.json"
        loss_log_path = os.path.join(loss_log_dir, loss_log_filename)
        mlp.save_loss_log(loss_log_path)

    results = {
        'mlp_train_loss': mlp_train_loss,
        'model_path': model_save_path
    }

    return results


def main():
    """Main function for unified MLP training"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MLP Unified Pretraining - Multiple Dataset Configurations')
    parser.add_argument('--dataset_config', type=int, required=True, choices=[0, 1, 2, 3, 6, 7],
                        help='Dataset configuration: 0=ASAP7 intra, 1=ASAP7 agnostic, 2=TSMC intra, 3=TSMC agnostic')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU device ID (default: 0)')
    parser.add_argument('--data_type', type=str, default='cell',
                        help='Data type: cell/transition (default: cell)')
    parser.add_argument('--num_iterations', type=int, default=300000,
                        help='Number of training iterations (default: 300000)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for Adam optimizer (default: 1e-4)')
    parser.add_argument('--model_type', type=str, default='aadam', choices=['aadam', 'mlp'],
                        help='Model type: aadam (hidden=256) or mlp (hidden=40) (default: aadam)')
    # Loss logging options
    parser.add_argument('--enable_loss_logging', action='store_true',
                        help='Enable training loss logging at specified intervals')
    parser.add_argument('--loss_log_every', type=int, default=1000,
                        help='Log training loss every N iterations (default: 1000)')
    parser.add_argument('--loss_log_dir', type=str, default=None,
                        help='Directory to save loss logs (default: loss_logs/)')
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
    dataset_config_id = args.dataset_config

    # Get dataset configuration info
    dataset_config = get_dataset_config(dataset_config_id)
    tech = dataset_config['tech']
    topology_type = dataset_config['topology_type']
    dataset_name = dataset_config['name']

    print(f"\n🚀 Starting MLP training on {dataset_name} {data_type} dataset")
    print_available_datasets()

    print(f"\n⚙️ Training configuration:")
    print(f"   Dataset config: {dataset_config_id} ({dataset_name})")
    print(f"   Data type: {data_type}")
    print(f"   Technology: {tech.upper()}")
    print(f"   Topology type: {topology_type}")
    print(f"   Model type: {model_type} (hidden_size={'256' if model_type=='aadam' else '40'})")
    print(f"   GPU ID: {args.gpu_id}")
    print(f"   Number of iterations: {args.num_iterations}")
    print(f"   Learning rate: {args.learning_rate}")

    # Build loss logging configuration
    loss_logging_config = {
        'enabled': args.enable_loss_logging,
        'log_every': args.loss_log_every,
        'save_dir': args.loss_log_dir
    }

    try:
        results = train_mlp_model(
            config_id=dataset_config_id,
            num_iterations=args.num_iterations,
            data_type=data_type,
            learning_rate=args.learning_rate,
            model_type=model_type,
            device=device,
            loss_logging_config=loss_logging_config
        )

        print(f"\n📊 Training Results:")
        print(f"   MLP Train Loss: {results['mlp_train_loss']:.6f}")
        print(f"   Model saved to: {results['model_path']}")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None

    return results


if __name__ == "__main__":
    main()
