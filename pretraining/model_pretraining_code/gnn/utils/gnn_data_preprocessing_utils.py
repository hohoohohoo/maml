"""
GNN Data Preprocessing Utility Functions

Common utility functions for GNN pretraining data preprocessing.
Based on voltage_variation_pretraining_utils.py structure.
Handles normalization, filtering, and NaN/Inf detection for node features and outputs.
"""

import torch
import numpy as np


def normalize_node_features(node_features, normalize_indices=[4, 5, 6], min_std_threshold=1e-8):
    """
    Normalize node features (voltage, input_slew, output_load).
    Based on normalize_input_features from voltage_variation_pretraining_utils.py

    Args:
        node_features: Node feature tensor [num_nodes, num_features]
        normalize_indices: List of feature indices to normalize (default: [4,5,6])
                          4: voltage, 5: input_slew, 6: output_load
        min_std_threshold: Minimum std threshold to perform normalization

    Returns:
        node_features: Normalized features (modified in-place)
        feature_means: List of means for each feature
        feature_stds: List of stds for each feature
    """
    num_features = node_features.shape[1]
    feature_means = [None] * num_features
    feature_stds = [None] * num_features

    # Feature name mapping
    feature_names = {4: "voltage", 5: "input_slew", 6: "output_load"}

    # Check for NaN/Inf in input
    if torch.isnan(node_features).any() or torch.isinf(node_features).any():
        print(f"⚠️ Warning: Input node_features contains NaN/Inf")
        # Replace NaN/Inf with zeros
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize specified features (only non-zero values)
    for feature_idx in normalize_indices:
        if feature_idx < num_features:
            # Get non-zero mask
            feature_values = node_features[:, feature_idx]
            non_zero_mask = feature_values != 0

            if non_zero_mask.any():
                feature_mean = feature_values[non_zero_mask].mean()
                feature_std = feature_values[non_zero_mask].std()

                feature_name = feature_names.get(feature_idx, f"feature_{feature_idx}")

                if feature_std > min_std_threshold:
                    node_features[non_zero_mask, feature_idx] = (
                        (feature_values[non_zero_mask] - feature_mean) / feature_std
                    )
                    feature_means[feature_idx] = feature_mean.item()
                    feature_stds[feature_idx] = feature_std.item()
                else:
                    print(f"⚠️ Warning: {feature_name} std is too small ({feature_std:.8f}), skipping normalization")
                    feature_means[feature_idx] = feature_mean.item()
                    feature_stds[feature_idx] = 1.0  # Use 1.0 to avoid division by zero
            else:
                print(f"⚠️ Warning: No non-zero values for feature {feature_idx}")
                feature_means[feature_idx] = 0.0
                feature_stds[feature_idx] = 1.0

    # Final NaN/Inf check after normalization
    if torch.isnan(node_features).any() or torch.isinf(node_features).any():
        print(f"⚠️ Warning: Normalized features contain NaN/Inf, replacing with zeros")
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

    return node_features, feature_means, feature_stds


def filter_and_normalize_task_outputs(minimal_data_per_file, min_std_threshold=1e-6, verbose=True):
    """
    Filter tasks based on output std and normalize outputs per task.
    Based on filter_and_normalize_outputs from voltage_variation_pretraining_utils.py

    Args:
        minimal_data_per_file: List of lists [num_libs][num_samples]
        min_std_threshold: Minimum std threshold for output
        verbose: Whether to print detailed filtering info

    Returns:
        filtered_data_per_file: Filtered data with valid tasks only
        normalized_outputs: Normalized output tensor [num_valid_tasks, num_libs]
        task_norm_stats: Dict mapping task_idx to {'mean': float, 'std': float}
        valid_task_indices: List of original task indices that passed filtering
    """
    num_libs = len(minimal_data_per_file)
    num_samples = len(minimal_data_per_file[0])

    print(f"\n🔍 Filtering and normalizing task outputs...")
    print(f"   Total libs: {num_libs}")
    print(f"   Total samples per lib: {num_samples}")
    print(f"   Min std threshold: {min_std_threshold}")

    valid_task_indices = []
    filtered_data_per_lib = [[] for _ in range(num_libs)]

    nan_inf_count = 0
    low_std_count = 0

    # First pass: filter tasks
    for task_idx in range(num_samples):
        # Collect outputs for this task across all libs
        task_outputs = []
        has_nan_inf = False

        for lib_idx in range(num_libs):
            sample = minimal_data_per_file[lib_idx][task_idx]
            output_val = sample['output']

            # Check for NaN/Inf in output
            if isinstance(output_val, torch.Tensor):
                if torch.isnan(output_val).any() or torch.isinf(output_val).any():
                    has_nan_inf = True
                    break
                task_outputs.append(output_val.item())
            else:
                if np.isnan(output_val) or np.isinf(output_val):
                    has_nan_inf = True
                    break
                task_outputs.append(output_val)

            # Check for NaN/Inf in node features
            if 'node_features' in sample:
                node_features = sample['node_features']
                if torch.isnan(node_features).any() or torch.isinf(node_features).any():
                    has_nan_inf = True
                    break

        if has_nan_inf:
            nan_inf_count += 1
            if verbose and nan_inf_count <= 5:
                print(f"  Task {task_idx}: Filtered (NaN/Inf detected)")
            continue

        # Calculate output std across libs
        task_outputs_array = np.array(task_outputs)
        output_std = task_outputs_array.std()

        # Check if std is sufficient
        if output_std > min_std_threshold:
            valid_task_indices.append(task_idx)
            # Add to filtered data
            for lib_idx in range(num_libs):
                filtered_data_per_lib[lib_idx].append(minimal_data_per_file[lib_idx][task_idx])
        else:
            low_std_count += 1
            if verbose and low_std_count <= 5:
                print(f"  Task {task_idx}: Filtered (std={output_std:.8f} too small)")

    num_valid_tasks = len(valid_task_indices)
    filter_ratio = num_valid_tasks / num_samples * 100

    print(f"\n✅ Task filtering completed:")
    print(f"   Original tasks: {num_samples}")
    print(f"   Valid tasks: {num_valid_tasks} ({filter_ratio:.1f}%)")
    print(f"   Filtered out: {num_samples - num_valid_tasks} tasks")
    print(f"     - NaN/Inf: {nan_inf_count}")
    print(f"     - Low std: {low_std_count}")

    if num_valid_tasks < 100:
        print(f"⚠️ Warning: Only {num_valid_tasks} tasks remaining. Consider lowering threshold.")

    if num_valid_tasks == 0:
        print(f"❌ No valid tasks found after filtering!")
        return None, None, {}, []

    # Second pass: normalize outputs per task
    print(f"\n📊 Normalizing outputs for {num_valid_tasks} valid tasks (per-task normalization)...")

    normalized_outputs = torch.zeros(num_valid_tasks, num_libs, dtype=torch.float32)
    task_norm_stats = {}

    for new_task_idx, original_task_idx in enumerate(valid_task_indices):
        # Collect outputs for this task
        task_outputs = []
        for lib_idx in range(num_libs):
            sample = filtered_data_per_lib[lib_idx][new_task_idx]
            output_val = sample['output']
            if isinstance(output_val, torch.Tensor):
                task_outputs.append(output_val.item())
            else:
                task_outputs.append(output_val)

        # Calculate per-task statistics
        task_outputs_tensor = torch.tensor(task_outputs, dtype=torch.float32)
        task_mean = task_outputs_tensor.mean().item()
        task_std = task_outputs_tensor.std().item()

        # Store task normalization stats (using original task index)
        task_norm_stats[new_task_idx] = {
            'mean': task_mean,
            'std': task_std,
            'original_idx': original_task_idx
        }

        # Normalize this task's outputs
        if task_std > min_std_threshold:
            normalized_outputs[new_task_idx] = (task_outputs_tensor - task_mean) / task_std
        else:
            # If std is too small, just center the data
            normalized_outputs[new_task_idx] = task_outputs_tensor - task_mean
            if verbose and new_task_idx < 5:
                print(f"   Task {new_task_idx}: std too small, using centering only")

    print(f"   ✅ Per-task output normalization complete")
    print(f"   Normalized range: min={normalized_outputs.min():.6f}, max={normalized_outputs.max():.6f}")

    return filtered_data_per_lib, normalized_outputs, task_norm_stats, valid_task_indices


def normalize_node_features_safe(node_features, norm_stats=None, min_std_threshold=1e-8, temp_mode=None):
    """
    Safely normalize node features with NaN/Inf detection using pre-computed stats.

    Args:
        node_features: Node feature tensor [num_nodes, num_features]
        norm_stats: Pre-computed normalization statistics
                   Dict with structure: {'voltage': {'mean': float, 'std': float}, ...}
        min_std_threshold: Minimum std to perform normalization
        temp_mode: Temperature normalization mode. If provided, overrides data detection.
                   'temp_all' - normalize all nodes, 'typical'/'mos_only' - normalize MOS nodes only
                   None - auto-detect from data (fallback)

    Returns:
        normalized_features: Normalized node features
        norm_stats: Normalization statistics (if not provided)
    """
    normalized = node_features.clone()

    # Check for NaN/Inf in input
    if torch.isnan(normalized).any() or torch.isinf(normalized).any():
        print(f"⚠️ Warning: Input node_features contains NaN/Inf")
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # If norm_stats not provided, compute from data
    if norm_stats is None:
        norm_stats = {}

        # Normalize voltage (column 4)
        voltage_values = normalized[:, 4]
        voltage_mask = voltage_values != 0
        if voltage_mask.any():
            voltage_mean = voltage_values[voltage_mask].mean().item()
            voltage_std = voltage_values[voltage_mask].std().item()

            if voltage_std < min_std_threshold:
                voltage_std = 1.0
                print(f"⚠️ Voltage std too small, using 1.0")

            normalized[voltage_mask, 4] = (voltage_values[voltage_mask] - voltage_mean) / voltage_std
            norm_stats['voltage'] = {'mean': voltage_mean, 'std': voltage_std}
        else:
            norm_stats['voltage'] = {'mean': 0.0, 'std': 1.0}

        # Normalize input_slew (column 5)
        slew_values = normalized[:, 5]
        slew_mask = slew_values != 0
        if slew_mask.any():
            slew_mean = slew_values[slew_mask].mean().item()
            slew_std = slew_values[slew_mask].std().item()

            if slew_std < min_std_threshold:
                slew_std = 1.0
                print(f"⚠️ Input slew std too small, using 1.0")

            normalized[slew_mask, 5] = (slew_values[slew_mask] - slew_mean) / slew_std
            norm_stats['input_slew'] = {'mean': slew_mean, 'std': slew_std}
        else:
            norm_stats['input_slew'] = {'mean': 0.0, 'std': 1.0}

        # Normalize output_load (column 6)
        load_values = normalized[:, 6]
        load_mask = load_values != 0
        if load_mask.any():
            load_mean = load_values[load_mask].mean().item()
            load_std = load_values[load_mask].std().item()

            if load_std < min_std_threshold:
                load_std = 1.0
                print(f"⚠️ Output load std too small, using 1.0")

            normalized[load_mask, 6] = (load_values[load_mask] - load_mean) / load_std
            norm_stats['output_load'] = {'mean': load_mean, 'std': load_std}
        else:
            norm_stats['output_load'] = {'mean': 0.0, 'std': 1.0}

        # Normalize temperature (column 10) - for 11D features with process params
        if normalized.shape[1] > 10:  # Check if this is 11D features
            temp_values = normalized[:, 10]
            mosfet_mask = normalized[:, 2] != 0  # MOSFET nodes (PMOS=+1, NMOS=-1)
            non_mosfet_mask = normalized[:, 2] == 0  # Non-MOSFET nodes

            # Determine temp_all mode: use temp_mode if provided, otherwise detect from data
            if temp_mode is not None:
                is_temp_all = (temp_mode == 'temp_all')
            else:
                # Fallback: detect from data (check if non-MOS nodes have non-zero temperature)
                non_mos_temps = temp_values[non_mosfet_mask]
                is_temp_all = non_mos_temps.abs().max() > 1e-6 if non_mosfet_mask.any() else False

            if is_temp_all:
                # temp_all mode: normalize all nodes (use temp != 0 mask for 0°C handling)
                # For temp_all, even at 0°C, all nodes should have the same temp value
                # So we use all nodes for stats, but only normalize non-zero for 0°C safety
                all_temps = temp_values[temp_values != 0] if (temp_values == 0).all() else temp_values
                if len(all_temps) > 0:
                    temp_mean = all_temps.mean().item()
                    temp_std = all_temps.std().item()
                    if temp_std < min_std_threshold:
                        temp_std = 1.0
                    # Normalize all nodes
                    normalized[:, 10] = (temp_values - temp_mean) / temp_std
                    norm_stats['temperature'] = {'mean': temp_mean, 'std': temp_std, 'mode': 'temp_all'}
                else:
                    norm_stats['temperature'] = {'mean': 0.0, 'std': 1.0, 'mode': 'temp_all'}
            else:
                # mos_only mode: only normalize MOS nodes
                if mosfet_mask.any():
                    temp_mean = temp_values[mosfet_mask].mean().item()
                    temp_std = temp_values[mosfet_mask].std().item()

                    if temp_std < min_std_threshold:
                        temp_std = 1.0
                        print(f"⚠️ Temperature std too small, using 1.0")

                    normalized[mosfet_mask, 10] = (temp_values[mosfet_mask] - temp_mean) / temp_std
                    norm_stats['temperature'] = {'mean': temp_mean, 'std': temp_std, 'mode': 'mos_only'}
                else:
                    norm_stats['temperature'] = {'mean': 0.0, 'std': 1.0, 'mode': 'mos_only'}

    else:
        # Use provided norm_stats
        # Detect normalization method: zscore (mean/std) or minmax (min/max/epsilon)
        def apply_norm(values, stats):
            """Apply normalization based on stats structure"""
            if 'method' in stats and stats['method'] == 'minmax_positive':
                # minmax: normalized = epsilon + (x - min) / (max - min) * (1 - epsilon)
                epsilon = stats.get('epsilon', 0.01)
                feat_min, feat_max = stats['min'], stats['max']
                if feat_max > feat_min:
                    return epsilon + (values - feat_min) / (feat_max - feat_min) * (1 - epsilon)
                else:
                    return torch.ones_like(values) * epsilon
            else:
                # zscore: normalized = (x - mean) / std
                return (values - stats['mean']) / stats['std']

        voltage_mask = normalized[:, 4] != 0
        if voltage_mask.any():
            normalized[voltage_mask, 4] = apply_norm(
                normalized[voltage_mask, 4], norm_stats['voltage']
            )

        slew_mask = normalized[:, 5] != 0
        if slew_mask.any():
            normalized[slew_mask, 5] = apply_norm(
                normalized[slew_mask, 5], norm_stats['input_slew']
            )

        load_mask = normalized[:, 6] != 0
        if load_mask.any():
            normalized[load_mask, 6] = apply_norm(
                normalized[load_mask, 6], norm_stats['output_load']
            )

        # Normalize temperature (column 10) if present
        if normalized.shape[1] > 10 and 'temperature' in norm_stats:
            temp_values = normalized[:, 10]
            temp_stats = norm_stats['temperature']

            # Determine temp_all mode: use temp_mode param > norm_stats mode > data detection
            if temp_mode is not None:
                is_temp_all = (temp_mode == 'temp_all')
            elif 'mode' in temp_stats:
                is_temp_all = temp_stats['mode'] == 'temp_all'
            else:
                # Fallback: detect from data (check if non-MOS nodes have temp values)
                mosfet_mask = normalized[:, 2] != 0
                non_mosfet_mask = normalized[:, 2] == 0
                non_mos_temps = temp_values[non_mosfet_mask]
                is_temp_all = non_mos_temps.abs().max() > 1e-6 if non_mosfet_mask.any() else False

            if is_temp_all:
                # temp_all mode: normalize all nodes
                normalized[:, 10] = apply_norm(temp_values, temp_stats)
            else:
                # mos_only mode: only normalize MOS nodes
                mosfet_mask = normalized[:, 2] != 0  # MOSFET nodes (PMOS=+1, NMOS=-1)
                if mosfet_mask.any():
                    normalized[mosfet_mask, 10] = apply_norm(
                        normalized[mosfet_mask, 10], temp_stats
                    )

    # Final NaN/Inf check after normalization
    if torch.isnan(normalized).any() or torch.isinf(normalized).any():
        print(f"⚠️ Warning: Normalized features contain NaN/Inf, replacing with zeros")
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return normalized, norm_stats


def normalize_task_outputs(stacked_outputs, min_std_threshold=1e-8):
    """
    Normalize outputs for each task (per-task normalization).
    This version assumes stacked_outputs is already filtered.

    Similar to normalize_all_task_outputs in voltage_variation but for already filtered data.

    Args:
        stacked_outputs: Tensor of shape [num_tasks, num_libs] (already filtered)
        min_std_threshold: Minimum std to perform normalization

    Returns:
        normalized_outputs: Normalized tensor
        task_norm_stats: Dict mapping task_idx to {'mean': float, 'std': float}
    """
    num_tasks = stacked_outputs.shape[0]
    normalized = torch.zeros_like(stacked_outputs)
    task_norm_stats = {}

    print(f"\n📊 Normalizing outputs for {num_tasks} tasks (per-task normalization)...")

    low_std_count = 0
    for task_idx in range(num_tasks):
        task_outputs = stacked_outputs[task_idx]

        task_mean = task_outputs.mean().item()
        task_std = task_outputs.std().item()

        task_norm_stats[task_idx] = {
            'mean': task_mean,
            'std': task_std
        }

        if task_std > min_std_threshold:
            normalized[task_idx] = (task_outputs - task_mean) / task_std
        else:
            # If std is too small, just center the data
            normalized[task_idx] = task_outputs - task_mean
            low_std_count += 1

    if low_std_count > 0:
        print(f"   ⚠️ {low_std_count} tasks had std < {min_std_threshold}, only centered")

    print(f"   ✅ Output normalization complete")
    print(f"   Normalized range: min={normalized.min():.6f}, max={normalized.max():.6f}")

    return normalized, task_norm_stats


def calculate_norm_stats_from_minimal_data_safe(minimal_data_per_file, sample_rate=10):
    """
    Calculate normalization statistics from minimal dataset with NaN/Inf detection.

    Args:
        minimal_data_per_file: List of lists [num_libs][num_samples]
        sample_rate: Sample every Nth lib and task to speed up computation

    Returns:
        norm_stats: Dict with normalization statistics for node features and outputs
    """
    print(f"\n🔧 Calculating normalization statistics (safe mode)...")
    print(f"   Total lib files: {len(minimal_data_per_file)}")
    print(f"   Sample rate: every {sample_rate}th lib and task")

    all_voltages = []
    all_input_slews = []
    all_output_loads = []
    all_output_values = []

    sample_count = 0
    skipped_count = 0

    # Sample from multiple lib files and tasks
    for lib_idx, lib_samples in enumerate(minimal_data_per_file):
        if lib_idx % sample_rate == 0:
            for task_idx in range(0, min(len(lib_samples), 1000), sample_rate):
                sample = lib_samples[task_idx]

                # Check for NaN/Inf in output
                output_val = sample['output']
                if isinstance(output_val, torch.Tensor):
                    if torch.isnan(output_val).any() or torch.isinf(output_val).any():
                        skipped_count += 1
                        continue
                    output_val = output_val.item()
                else:
                    if np.isnan(output_val) or np.isinf(output_val):
                        skipped_count += 1
                        continue

                all_output_values.append(output_val)

                if 'node_features' in sample:
                    features = sample['node_features']

                    # Check for NaN/Inf in features
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        skipped_count += 1
                        continue

                    # Extract non-zero values
                    voltage_values = features[:, 4]
                    voltage_values = voltage_values[voltage_values != 0]
                    if len(voltage_values) > 0:
                        all_voltages.extend(voltage_values.tolist())

                    slew_values = features[:, 5]
                    slew_values = slew_values[slew_values != 0]
                    if len(slew_values) > 0:
                        all_input_slews.extend(slew_values.tolist())

                    load_values = features[:, 6]
                    load_values = load_values[load_values != 0]
                    if len(load_values) > 0:
                        all_output_loads.extend(load_values.tolist())

                sample_count += 1

    print(f"   📊 Sampled {sample_count} samples")
    if skipped_count > 0:
        print(f"   ⚠️ Skipped {skipped_count} samples due to NaN/Inf")

    # Calculate statistics with safety checks
    def safe_stats(values, name):
        if len(values) == 0:
            print(f"     ⚠️ No {name} values found, using defaults")
            return {'mean': 1.0, 'std': 0.1}

        values_array = np.array(values)

        # Remove any remaining NaN/Inf
        values_array = values_array[~np.isnan(values_array)]
        values_array = values_array[~np.isinf(values_array)]

        if len(values_array) == 0:
            print(f"     ⚠️ All {name} values were NaN/Inf, using defaults")
            return {'mean': 1.0, 'std': 0.1}

        mean_val = values_array.mean()
        std_val = values_array.std()

        if std_val < 1e-8 or np.isnan(std_val) or np.isinf(std_val):
            print(f"     ⚠️ {name} std invalid ({std_val:.2e}), using 0.1")
            std_val = 0.1

        if np.isnan(mean_val) or np.isinf(mean_val):
            print(f"     ⚠️ {name} mean invalid, using default")
            mean_val = 1.0

        print(f"     {name}: mean={mean_val:.6f}, std={std_val:.6f} (n={len(values_array)})")
        return {'mean': float(mean_val), 'std': float(std_val)}

    norm_stats = {
        'node_features': {
            'voltage': safe_stats(all_voltages, 'Voltage'),
            'input_slew': safe_stats(all_input_slews, 'Input Slew'),
            'output_load': safe_stats(all_output_loads, 'Output Load')
        },
        'output': safe_stats(all_output_values, 'Output')
    }

    return norm_stats


def validate_gnn_data(minimal_data_per_file, topology_cache=None):
    """
    Validate GNN data for NaN/Inf and consistency issues.

    Args:
        minimal_data_per_file: List of lists [num_libs][num_samples]
        topology_cache: Optional topology cache to validate

    Returns:
        validation_results: Dict with validation statistics
    """
    print(f"\n🔍 Validating GNN data...")

    num_libs = len(minimal_data_per_file)
    num_samples = len(minimal_data_per_file[0])

    issues = {
        'nan_in_output': 0,
        'inf_in_output': 0,
        'nan_in_features': 0,
        'inf_in_features': 0,
        'missing_cell_in_cache': 0,
        'shape_mismatch': 0
    }

    for lib_idx in range(num_libs):
        for task_idx in range(num_samples):
            sample = minimal_data_per_file[lib_idx][task_idx]

            # Check output
            output_val = sample['output']
            if isinstance(output_val, torch.Tensor):
                if torch.isnan(output_val).any():
                    issues['nan_in_output'] += 1
                if torch.isinf(output_val).any():
                    issues['inf_in_output'] += 1
            else:
                if np.isnan(output_val):
                    issues['nan_in_output'] += 1
                if np.isinf(output_val):
                    issues['inf_in_output'] += 1

            # Check node features
            if 'node_features' in sample:
                features = sample['node_features']
                if torch.isnan(features).any():
                    issues['nan_in_features'] += 1
                if torch.isinf(features).any():
                    issues['inf_in_features'] += 1

            # Check topology cache
            if topology_cache is not None and 'cell_name' in sample:
                cell_name = sample['cell_name']
                if cell_name not in topology_cache:
                    issues['missing_cell_in_cache'] += 1

    total_samples = num_libs * num_samples

    print(f"   Total samples checked: {total_samples}")
    print(f"   Issues found:")
    for issue_type, count in issues.items():
        if count > 0:
            print(f"     - {issue_type}: {count} ({count/total_samples*100:.2f}%)")

    has_issues = any(count > 0 for count in issues.values())

    if not has_issues:
        print(f"   ✅ No data quality issues found!")
    else:
        print(f"   ⚠️ Data quality issues detected!")

    validation_results = {
        'total_samples': total_samples,
        'issues': issues,
        'has_issues': has_issues
    }

    return validation_results


def preprocess_gnn_minimal_data(minimal_data_per_file,
                                min_std_threshold=1e-6,
                                enable_filtering=True,
                                verbose=True):
    """
    Complete preprocessing pipeline for GNN minimal data.
    Based on preprocess_voltage_data from voltage_variation_pretraining_utils.py

    Combines filtering, validation, and normalization in the correct order:
    1. Validate input data
    2. Filter tasks by output std and NaN/Inf
    3. Calculate normalization statistics from filtered data
    4. Return filtered data with norm stats

    Args:
        minimal_data_per_file: List of lists [num_libs][num_samples]
        min_std_threshold: Minimum std threshold for output filtering
        enable_filtering: Whether to filter low-variance samples
        verbose: Whether to print detailed info

    Returns:
        preprocessed_data: Preprocessed minimal_data_per_file (filtered)
        norm_stats: Normalization statistics
        preprocessing_stats: Statistics about preprocessing
    """
    print(f"\n{'='*80}")
    print(f"🔧 GNN Data Preprocessing Pipeline")
    print(f"{'='*80}")

    # Step 1: Validate input data
    validation_results = validate_gnn_data(minimal_data_per_file)

    # Step 2: Filter and normalize outputs per task
    if enable_filtering:
        filtered_data, normalized_outputs, task_norm_stats, valid_indices = filter_and_normalize_task_outputs(
            minimal_data_per_file,
            min_std_threshold=min_std_threshold,
            verbose=verbose
        )

        if filtered_data is None:
            print(f"\n❌ Filtering failed - no valid tasks!")
            return None, None, None

        filter_stats = {
            'original_tasks': len(minimal_data_per_file[0]),
            'valid_tasks': len(valid_indices),
            'filter_ratio': len(valid_indices) / len(minimal_data_per_file[0]) * 100
        }
    else:
        print(f"\n⚠️ Filtering disabled, using all samples")
        filtered_data = minimal_data_per_file
        filter_stats = {
            'original_tasks': len(minimal_data_per_file[0]),
            'valid_tasks': len(minimal_data_per_file[0]),
            'filter_ratio': 100.0
        }

    # Step 3: Calculate normalization statistics from filtered data
    norm_stats = calculate_norm_stats_from_minimal_data_safe(filtered_data)

    preprocessing_stats = {
        'validation': validation_results,
        'filtering': filter_stats,
        'normalization': norm_stats
    }

    print(f"\n{'='*80}")
    print(f"✅ Preprocessing Complete")
    print(f"{'='*80}")

    return filtered_data, norm_stats, preprocessing_stats
