"""
Voltage Variation Pretraining Utility Functions

Common utility functions for voltage variation pretraining across ASAP7 and TSMC PDKs.
Handles normalization, data loading, and filtering.
"""

import torch


def normalize_input_features(test_data_input, normalize_indices=[0, 3, 4], min_std_threshold=1e-8):
    """
    Normalize input features (voltage, process, temperature).

    Args:
        test_data_input: Input tensor [tasks, points, features]
        normalize_indices: List of feature indices to normalize (default: [0,3,4])
        min_std_threshold: Minimum std threshold to perform normalization

    Returns:
        test_data_input: Normalized input (modified in-place)
        feature_means: List of means for each feature
        feature_stds: List of stds for each feature
    """
    input_features = test_data_input.shape[2]
    feature_means = [None] * input_features
    feature_stds = [None] * input_features

    print(f"Input features detected: {input_features}")
    print(f"Input data range: min={test_data_input.min():.6f}, max={test_data_input.max():.6f}")
    print(f"Input contains NaN: {torch.isnan(test_data_input).any()}")
    print(f"Input contains Inf: {torch.isinf(test_data_input).any()}")

    # Normalize specified features
    for feature_idx in normalize_indices:
        if feature_idx < input_features:
            feature_mean = test_data_input[:,:,feature_idx].mean()
            feature_std = test_data_input[:,:,feature_idx].std()

            feature_name = {0: "voltage", 3: "process_value", 4: "temperature_value"}.get(feature_idx, f"feature_{feature_idx}")
            print(f"Feature {feature_idx} ({feature_name}) stats: mean={feature_mean:.6f}, std={feature_std:.6f}")

            if feature_std > min_std_threshold:
                test_data_input[:,:,feature_idx] = ((test_data_input[:,:,feature_idx] - feature_mean) / feature_std)
                feature_means[feature_idx] = feature_mean
                feature_stds[feature_idx] = feature_std
            else:
                print(f"⚠️ Warning: feature {feature_idx} std is too small ({feature_std:.8f}), skipping normalization")
                feature_means[feature_idx] = feature_mean
                feature_stds[feature_idx] = 1.0  # Use 1.0 to avoid division by zero

    print(f"Input data after normalization - range: min={test_data_input.min():.6f}, max={test_data_input.max():.6f}")

    return test_data_input, feature_means, feature_stds


def filter_and_normalize_outputs(test_data_input, test_data_output, min_std_threshold=1e-6, verbose=True):
    """
    Filter samples based on output std and normalize outputs.

    Args:
        test_data_input: Input tensor [tasks, points, features]
        test_data_output: Output tensor [tasks, points, 1]
        min_std_threshold: Minimum std threshold for output
        verbose: Whether to print detailed filtering info

    Returns:
        filtered_input: Filtered and stacked input tensor
        filtered_output: Filtered and normalized output tensor
        valid_indices: List of valid task indices
    """
    print(f"\nOutput data range: min={test_data_output.min():.6f}, max={test_data_output.max():.6f}")
    print(f"Output contains NaN: {torch.isnan(test_data_output).any()}")
    print(f"Output contains Inf: {torch.isinf(test_data_output).any()}")

    valid_indices = []
    filtered_input = []
    filtered_output = []

    original_size = len(test_data_output)
    print(f"\nFiltering samples with output std < {min_std_threshold}...")

    for i in range(original_size):
        output_mean = test_data_output[i,:,:].mean()
        output_std = test_data_output[i,:,:].std()

        # Check for NaN/Inf
        has_nan_inf = (torch.isnan(test_data_output[i]).any() or
                      torch.isinf(test_data_output[i]).any() or
                      torch.isnan(test_data_input[i]).any() or
                      torch.isinf(test_data_input[i]).any())

        # Valid sample check
        if output_std > min_std_threshold and not has_nan_inf:
            # Normalize output
            normalized_output = (test_data_output[i,:,:] - output_mean) / output_std

            filtered_input.append(test_data_input[i])
            filtered_output.append(normalized_output)
            valid_indices.append(i)
        else:
            if verbose and i < 5:  # Print first few filtered samples
                if output_std <= min_std_threshold:
                    print(f"  Filtered out sample {i}: output std too small ({output_std:.8f})")
                if has_nan_inf:
                    print(f"  Filtered out sample {i}: contains NaN/Inf")

    # Check if we have valid samples
    if not filtered_input:
        print(f"❌ No valid samples found after filtering!")
        print(f"   Try lowering min_std_threshold from {min_std_threshold}")
        return None, None, []

    # Stack filtered data
    filtered_input_tensor = torch.stack(filtered_input)
    filtered_output_tensor = torch.stack(filtered_output)

    filter_ratio = len(filtered_input) / original_size * 100
    print(f"✅ Dataset filtering completed:")
    print(f"   Original samples: {original_size}")
    print(f"   Valid samples: {len(filtered_input)} ({filter_ratio:.1f}%)")
    print(f"   Filtered out: {original_size - len(filtered_input)} samples")

    if len(filtered_input) < 100:
        print(f"⚠️ Warning: Only {len(filtered_input)} samples remaining. Consider lowering the std threshold.")

    return filtered_input_tensor, filtered_output_tensor, valid_indices


def load_asap7_voltage_data(corner, cell_type, data_type):
    """
    Load ASAP7 voltage variation dataset.

    Args:
        corner: Corner condition (SS/FF/TT)
        cell_type: Cell type (lvt/rvt/slvt/sram)
        data_type: Data type (cell/transition)

    Returns:
        test_data_input: Input tensor
        test_data_output: Output tensor
    """
    print(f"📊 Loading ASAP7 data: {corner} corner, {cell_type} cell type, {data_type} data")

    base_path = f"../../../dataset_all/dataset_ASAP7_dim5/taskdivide_{cell_type}_{corner}"
    test_data_input = torch.load(f"{base_path}/traindatainput/{data_type}_train_input.pth")
    test_data_output = torch.load(f"{base_path}/traindataoutput/{data_type}_train_output.pth")

    print(f"   Input shape: {test_data_input.shape}")
    print(f"   Output shape: {test_data_output.shape}")

    return test_data_input, test_data_output


def load_tsmc_voltage_data(corner, temp, data_type):
    """
    Load TSMC voltage variation dataset.

    Args:
        corner: Corner condition (ff/ss/tt)
        temp: Temperature (0/25/50/75/100)
        data_type: Data type (cell/transition)

    Returns:
        test_data_input: Input tensor
        test_data_output: Output tensor
    """
    print(f"📊 Loading TSMC data: {corner} corner, {temp}°C, {data_type} data")

    base_path = f"/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_TSMC_dim5/taskdivide_{corner}_{temp}"
    test_data_input = torch.load(f"{base_path}/traindatainput/{data_type}_train_input.pth")
    test_data_output = torch.load(f"{base_path}/traindataoutput/{data_type}_train_output.pth")

    print(f"   Input shape: {test_data_input.shape}")
    print(f"   Output shape: {test_data_output.shape}")

    return test_data_input, test_data_output


def preprocess_voltage_data(test_data_input, test_data_output, device='cuda',
                           normalize_indices=[0, 3, 4], min_std_threshold=1e-6,
                           return_feature_stats=False):
    """
    Unified preprocessing for voltage variation data (MLP and MAML).
    Applies input normalization, output filtering, and output normalization.

    Args:
        test_data_input: Raw input tensor
        test_data_output: Raw output tensor
        device: torch device
        normalize_indices: Feature indices to normalize (default: [0,3,4])
        min_std_threshold: Minimum std for output filtering (default: 1e-6)
        return_feature_stats: If True, returns feature_means and feature_stds (for MLP saving)

    Returns:
        If return_feature_stats=False (MAML):
            test_data_input: Preprocessed input on device
            test_data_output: Preprocessed output on device
            valid_indices: List of valid task indices

        If return_feature_stats=True (MLP):
            test_data_input: Preprocessed input on device
            test_data_output: Preprocessed output on device
            feature_means: Feature means for checkpoint saving
            feature_stds: Feature stds for checkpoint saving
    """
    # Normalize input features
    test_data_input, feature_means, feature_stds = normalize_input_features(
        test_data_input, normalize_indices=normalize_indices
    )

    # Filter and normalize outputs
    test_data_input, test_data_output, valid_indices = filter_and_normalize_outputs(
        test_data_input, test_data_output, min_std_threshold=min_std_threshold
    )

    if test_data_input is None:
        if return_feature_stats:
            return None, None, feature_means, feature_stds
        else:
            return None, None, []

    # Move to device
    test_data_input = test_data_input.to(device)
    test_data_output = test_data_output.to(device)

    print(f"Final input shape: {test_data_input.shape}")
    print(f"Final output shape: {test_data_output.shape}")

    # Return based on mode
    if return_feature_stats:
        return test_data_input, test_data_output, feature_means, feature_stds
    else:
        return test_data_input, test_data_output, valid_indices


def check_model_parameters(model, model_name="Model"):
    """
    Check model parameters for NaN/Inf values.

    Args:
        model: PyTorch model
        model_name: Name for logging

    Returns:
        bool: True if parameters are valid, False if NaN/Inf detected
    """
    param_check_passed = True

    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"⚠️ {model_name}: NaN/Inf detected in parameter {name}")
            param_check_passed = False

    return param_check_passed


def reinitialize_invalid_parameters(model):
    """
    Reinitialize model parameters that contain NaN/Inf.

    Args:
        model: PyTorch model to fix
    """
    print("🔧 Reinitializing invalid parameters...")

    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"   Reinitializing {name}")
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)
