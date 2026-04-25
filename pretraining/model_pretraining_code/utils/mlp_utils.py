#!/usr/bin/env python
# coding: utf-8

"""
MLP Pretraining Utility Functions

This module contains common utility functions and classes used across MLP pretraining scripts:
- MLP: Neural network model class
- MLP1: Training wrapper class
- normalize_features: Normalize input features
- normalize_outputs: Normalize output values with task filtering
- train_mlp1_with_checkpoints: Training loop with checkpoint support
"""

import os
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random


def normalize_features(data_input, normalize_indices=[7, 8, 3, 4], num_features=9,
                      feature_names=['slew', 'load_cap', 'temperature', 'voltage']):
    """
    Normalize specified input features using z-score normalization with validation.

    Args:
        data_input: Input tensor of shape [tasks, samples, features] (3D) or [tasks, features] (2D)
        normalize_indices: List of feature indices to normalize (default: [7,8,3,4])
        num_features: Total number of features (default: 9)
        feature_names: List of feature names for logging (default: ['slew', 'load_cap', 'temperature', 'voltage'])

    Returns:
        tuple: (normalized_input, feature_means, feature_stds)
    """
    print(f"🔧 Normalizing features: {feature_names}")
    print(f"   Feature indices: {normalize_indices}")
    print(f"📊 Input shape: {data_input.shape}")

    feature_means = [None] * num_features
    feature_stds = [None] * num_features

    # Handle both 2D and 3D tensors
    is_3d = data_input.dim() == 3

    for feature_idx, feature_name in zip(normalize_indices, feature_names):
        if is_3d:
            feature_mean = data_input[:, :, feature_idx].mean()
            feature_std = data_input[:, :, feature_idx].std()
        else:
            feature_mean = data_input[:, feature_idx].mean()
            feature_std = data_input[:, feature_idx].std()

        print(f"📊 {feature_name} (idx {feature_idx}) stats: mean={feature_mean:.6f}, std={feature_std:.6f}")

        # Safe normalization (skip if std is too small)
        if feature_std > 1e-8:
            if is_3d:
                data_input[:, :, feature_idx] = (
                    (data_input[:, :, feature_idx] - feature_mean) / feature_std
                )
            else:
                data_input[:, feature_idx] = (
                    (data_input[:, feature_idx] - feature_mean) / feature_std
                )
            feature_means[feature_idx] = feature_mean
            feature_stds[feature_idx] = feature_std
            print(f"   ✅ {feature_name} normalized")
        else:
            print(f"   ⚠️ {feature_name} std too small ({feature_std:.2e}), skipping normalization")
            feature_means[feature_idx] = feature_mean
            feature_stds[feature_idx] = 1.0

    # Data validity check
    print(f"📊 Input data after normalization:")
    print(f"   Range: min={data_input.min():.6f}, max={data_input.max():.6f}")
    print(f"   Contains NaN: {torch.isnan(data_input).any()}")
    print(f"   Contains Inf: {torch.isinf(data_input).any()}")

    return data_input, feature_means, feature_stds


def normalize_outputs(data_output, min_std_threshold=1e-6):
    """
    Normalize outputs per task and filter out invalid tasks with detailed validation.

    Args:
        data_output: Output tensor of shape [tasks, samples, output_dim]
        min_std_threshold: Minimum standard deviation threshold (default: 1e-6)

    Returns:
        tuple: (normalized_output, output_means, output_stds, valid_indices)
    """
    print(f"📊 Output data stats before filtering:")
    print(f"   Range: min={data_output.min():.6f}, max={data_output.max():.6f}")
    print(f"   Contains NaN: {torch.isnan(data_output).any()}")
    print(f"   Contains Inf: {torch.isinf(data_output).any()}")

    output_means = []
    output_stds = []
    valid_indices = []

    print(f"🔍 Filtering tasks with std >= {min_std_threshold}...")
    original_size = len(data_output)

    # Handle both 2D and 3D tensors
    is_3d = data_output.dim() == 3

    for i in range(len(data_output)):
        if is_3d:
            output_mean = data_output[i, :, :].mean()
            output_std = data_output[i, :, :].std()
        else:
            output_mean = data_output[i, :].mean()
            output_std = data_output[i, :].std()

        # NaN/Inf check
        has_nan_inf = (torch.isnan(data_output[i]).any() or
                      torch.isinf(data_output[i]).any())

        # Validity check
        is_valid = (output_std >= min_std_threshold and
                   not has_nan_inf and
                   not torch.isnan(output_std))

        if is_valid:
            # Normalize output
            if is_3d:
                data_output[i, :, :] = (data_output[i, :, :] - output_mean) / (output_std + 1e-8)
            else:
                data_output[i, :] = (data_output[i, :] - output_mean) / (output_std + 1e-8)
            output_means.append(output_mean)
            output_stds.append(output_std)
            valid_indices.append(i)

            # Progress output
            if len(valid_indices) % 1000 == 0:
                print(f"   Processed {i+1}/{original_size} tasks, Valid: {len(valid_indices)}")

        # Debug: print first few invalid samples
        elif len(valid_indices) < 10:
            print(f"   Task {i}: Invalid - output_std={output_std:.2e}, "
                  f"has_nan_inf={has_nan_inf}")

    if not valid_indices:
        raise ValueError("❌ No valid samples found after filtering!")

    filtered_size = len(valid_indices)
    print(f"✅ Filtering completed:")
    print(f"   Original tasks: {original_size}")
    print(f"   Valid tasks: {filtered_size}")
    print(f"   Filtering ratio: {filtered_size/original_size*100:.1f}%")

    return data_output, output_means, output_stds, valid_indices


def save_model(model, save_path, train_loss=None):
    """
    Save model checkpoint.

    Args:
        model: Model to save (MLP instance)
        save_path: Path to save the model
        train_loss: Optional training loss to save
    """
    save_dict = {'model_state_dict': model.state_dict()}
    if train_loss is not None:
        save_dict['train_loss'] = train_loss

    torch.save(save_dict, save_path)
    print(f"Model saved to: {save_path}")
