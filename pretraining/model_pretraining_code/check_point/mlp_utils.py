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


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with Xavier initialization.

    Architecture: input -> hidden -> hidden -> hidden -> output
    Uses ReLU activation and optional dropout.

    Args:
        input_size: Number of input features
        output_size: Number of output values (default: 1)
        hidden_size: Number of hidden units (default: 256)
    """
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(0.2)

        # Xavier/Glorot initialization for better gradient flow
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = self.fc3(x)
        return x


class MLP_pretraining(object):
    """
    MLP training wrapper with Adam optimizer.

    Args:
        lr: Learning rate (default: 2e-3)
        wd: Weight decay (default: 5e-3)
        dataset_in: Input dataset tensor
        dataset_out: Output dataset tensor
        iteration: Number of training iterations (default: 5)
        hidden_size: Hidden layer size (default: 256)
        input_size: Input feature size (default: 9)
    """
    def __init__(self, lr=2e-3, wd=5e-3, dataset_in=None, dataset_out=None,
                 iteration=5, hidden_size=256, input_size=9):
        self.lr = lr
        self.wd = wd
        self.train_in = dataset_in
        self.train_out = dataset_out
        self.train_in_test = dataset_in
        self.train_out_test = dataset_out
        self.iteration = iteration
        self.model = MLP(input_size=input_size, output_size=1, hidden_size=hidden_size)
        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    def loop(self, checkpoint_dir='checkpoints'):
        """
        Training loop with checkpoint saving.

        Args:
            checkpoint_dir: Directory to save checkpoints (default: 'checkpoints')

        Returns:
            Average training loss
        """
        running_loss = 0.0
        num_tasks = len(self.train_in)

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        for i in range(self.iteration):
            if i % 1000 == 0:
                avg_loss = running_loss / max(1, i)
                print(f"Iteration {i}/{self.iteration}, Avg Loss: {avg_loss:.6f}")

            # Save checkpoint every 1000000 iterations
            if i > 0 and i % 1000000 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{i}.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict()
                }, checkpoint_path)
                print(f"Checkpoint saved at iteration {i}: {checkpoint_path}")

            # Select one random task for this iteration
            task_idx = random.randint(0, num_tasks - 1)

            self.optimizer.zero_grad()

            # Get data for the selected task
            x_sampled = self.train_in[task_idx, :]
            y_sampled = self.train_out[task_idx, :]

            # Create mini-batch of 5 samples from this task
            indices = random.sample(range(x_sampled.shape[0]), min(5, x_sampled.shape[0]))
            mini_batch = x_sampled[indices]
            mini_y = y_sampled[indices]

            if torch.cuda.is_available():
                mini_batch, mini_y = mini_batch.cuda(), mini_y.cuda()

            # Forward pass, loss computation, and optimization
            self.model.train()
            y_pred = self.model(mini_batch)
            the_loss = F.mse_loss(y_pred, mini_y)

            the_loss.backward()
            self.optimizer.step()

            running_loss += the_loss.item()

        return float(running_loss / self.iteration)


def normalize_features(data_input, normalize_indices=[7, 8, 3, 4], num_features=9):
    """
    Normalize specified input features using z-score normalization.

    Args:
        data_input: Input tensor of shape [tasks, samples, features]
        normalize_indices: List of feature indices to normalize (default: [7,8,3,4])
        num_features: Total number of features (default: 9)

    Returns:
        tuple: (normalized_input, feature_means, feature_stds)
    """
    feature_means = [None] * num_features
    feature_stds = [None] * num_features

    for feature_idx in normalize_indices:
        feature_mean = data_input[:, :, feature_idx].mean()
        feature_std = data_input[:, :, feature_idx].std()
        if feature_std < 1e-8:
            feature_std = 1.0
            print(f"Feature {feature_idx} has very low std: {feature_std:.2e}")
        data_input[:, :, feature_idx] = (
            (data_input[:, :, feature_idx] - feature_mean) / feature_std
        )
        feature_means[feature_idx] = feature_mean
        feature_stds[feature_idx] = feature_std

    return data_input, feature_means, feature_stds


def normalize_outputs(data_output, min_std_threshold=1e-6):
    """
    Normalize outputs per task and filter out tasks with low variation.

    Args:
        data_output: Output tensor of shape [tasks, samples, output_dim]
        min_std_threshold: Minimum standard deviation threshold (default: 1e-6)

    Returns:
        tuple: (normalized_output, output_means, output_stds, valid_indices)
    """
    output_means = []
    output_stds = []
    valid_indices = []

    for i in range(len(data_output)):
        output_mean = data_output[i, :, :].mean()
        output_std = data_output[i, :, :].std()

        # Skip tasks with very low variation
        if output_std < min_std_threshold:
            print(f"Skipping task {i} with very low output std: {output_std:.2e}")
            continue

        data_output[i, :, :] = (data_output[i, :, :] - output_mean) / (output_std + 1e-8)
        output_means.append(output_mean)
        output_stds.append(output_std)
        valid_indices.append(i)

    print(f"Kept {len(valid_indices)} valid tasks after filtering")

    return data_output, output_means, output_stds, valid_indices


def create_mlp1_trainer(data_input, data_output, learning_rate=1e-4, hidden_size=256,
                        num_iterations=100000, input_size=9):
    """
    Create and configure MLP1 trainer.

    Args:
        data_input: Input dataset
        data_output: Output dataset
        learning_rate: Learning rate for optimizer (default: 1e-4)
        hidden_size: Hidden layer size (default: 256)
        num_iterations: Number of training iterations (default: 100000)
        input_size: Input feature size (default: 9)

    Returns:
        MLP1 trainer instance
    """
    mlp1 = MLP_pretraining(
        lr=learning_rate,
        wd=0,
        dataset_in=data_input,
        dataset_out=data_output,
        iteration=num_iterations,
        hidden_size=hidden_size,
        input_size=input_size
    )

    return mlp1


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
