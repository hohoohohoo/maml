# GNN Data Preprocessing Utilities

This utility module provides robust data preprocessing functions for GNN pretraining, based on the structure of `voltage_variation_pretraining_utils.py`.

## Problem

The original GNN training code was experiencing `loss = nan` during training due to:
- NaN/Inf values in node features or outputs
- Samples with very low standard deviation (near-constant values)
- Missing normalization safety checks
- Filtered tasks not being excluded from normalization

## Solution

This module implements a preprocessing pipeline similar to `voltage_variation_pretraining_utils.py`:

1. **NaN/Inf Detection**: Automatically detects and handles invalid values
2. **Output Filtering**: Removes tasks with insufficient variance (low std) or containing NaN/Inf
3. **Safe Normalization**: Normalizes features with safety checks for edge cases
4. **Per-task Output Normalization**: Normalizes outputs individually for each task
5. **Filtered Task Exclusion**: Tasks filtered out in step 2 are completely excluded from normalization

## Key Differences from voltage_variation_pretraining_utils.py

### Voltage Variation Utils (MLP/MAML)
- **Data format**: `[num_tasks, num_points, num_features]`
- **Input normalization**: Normalizes specific feature columns (voltage, process, temperature)
- **Output filtering**: Returns filtered tensors directly
- **Task organization**: Tasks are pre-defined by input conditions

### GNN Utils
- **Data format**: `minimal_data_per_file` - List of lists `[num_libs][num_samples]`
- **Input normalization**: Normalizes node features (voltage, input_slew, output_load) only for non-zero values
- **Output filtering**: Returns filtered data structure maintaining lib organization
- **Task organization**: Tasks defined by same input condition across different lib files

## Core Functions

### 1. `normalize_node_features(node_features, normalize_indices=[4,5,6], min_std_threshold=1e-8)`

Normalize node features based on `normalize_input_features` from voltage_variation utils.

**Features:**
- Normalizes only non-zero values (preserves sparsity)
- Handles NaN/Inf detection
- Handles low standard deviation cases
- Returns feature means and stds

**Args:**
- `node_features`: Tensor `[num_nodes, num_features]`
- `normalize_indices`: Feature indices to normalize (default: [4,5,6])
- `min_std_threshold`: Minimum std to perform normalization

**Returns:**
- `node_features`: Normalized features (modified in-place)
- `feature_means`: List of means
- `feature_stds`: List of stds

### 2. `filter_and_normalize_task_outputs(minimal_data_per_file, min_std_threshold=1e-6, verbose=True)`

**THE KEY FUNCTION** - Filter tasks AND normalize outputs in one step.
Based on `filter_and_normalize_outputs` from voltage_variation utils.

**Two-Pass Process:**

**Pass 1 - Filtering:**
- Checks each task across all lib files
- Removes tasks with output std below threshold
- Removes tasks containing NaN/Inf in outputs or features
- Builds filtered data structure

**Pass 2 - Normalization:**
- Normalizes ONLY the valid tasks (filtered tasks are excluded)
- Per-task normalization using task's own mean/std
- Stores original task indices for reference

**Args:**
- `minimal_data_per_file`: List of lists `[num_libs][num_samples]`
- `min_std_threshold`: Minimum std for valid tasks (default: 1e-6)
- `verbose`: Print detailed filtering info

**Returns:**
- `filtered_data_per_file`: Filtered data `[num_libs][num_valid_samples]`
- `normalized_outputs`: Normalized tensor `[num_valid_tasks, num_libs]`
- `task_norm_stats`: Dict with normalization stats for each valid task
  ```python
  {
      task_idx: {
          'mean': float,
          'std': float,
          'original_idx': int  # Original task index before filtering
      }
  }
  ```
- `valid_task_indices`: List of original task indices that passed filtering

**Example:**
```python
filtered_data, normalized_outputs, task_stats, valid_indices = filter_and_normalize_task_outputs(
    minimal_data_per_file,
    min_std_threshold=1e-6,
    verbose=True
)

# Output:
# 🔍 Filtering and normalizing task outputs...
#    Total libs: 61
#    Total samples per lib: 5000
#    Min std threshold: 1e-06
#
# ✅ Task filtering completed:
#    Original tasks: 5000
#    Valid tasks: 4850 (97.0%)
#    Filtered out: 150 tasks
#      - NaN/Inf: 10
#      - Low std: 140
#
# 📊 Normalizing outputs for 4850 valid tasks...
#    ✅ Per-task output normalization complete
```

### 3. `normalize_task_outputs(stacked_outputs, min_std_threshold=1e-8)`

Normalize outputs for already-filtered data. Use this when you have already filtered tasks and just need normalization.

**Args:**
- `stacked_outputs`: Tensor `[num_tasks, num_libs]` (already filtered)
- `min_std_threshold`: Minimum std for normalization

**Returns:**
- `normalized_outputs`: Normalized tensor
- `task_norm_stats`: Dict mapping task_idx to stats

### 4. `normalize_node_features_safe(node_features, norm_stats=None, min_std_threshold=1e-8)`

Apply normalization using pre-computed statistics (for inference/training).

**Args:**
- `node_features`: Tensor `[num_nodes, num_features]`
- `norm_stats`: Pre-computed stats dict
- `min_std_threshold`: Minimum std

**Returns:**
- `normalized_features`: Normalized tensor
- `norm_stats`: Statistics (if not provided)

### 5. `calculate_norm_stats_from_minimal_data_safe(minimal_data_per_file, sample_rate=10)`

Calculate normalization statistics with NaN/Inf safety.

**Args:**
- `minimal_data_per_file`: List of lists `[num_libs][num_samples]`
- `sample_rate`: Sample every Nth lib and task

**Returns:**
- `norm_stats`: Dict with statistics

### 6. `validate_gnn_data(minimal_data_per_file, topology_cache=None)`

Validate data quality before training.

**Returns:**
- `validation_results`: Dict with validation statistics

### 7. `preprocess_gnn_minimal_data(minimal_data_per_file, min_std_threshold=1e-6, enable_filtering=True, verbose=True)`

**Complete preprocessing pipeline** - use this for most cases.

**Pipeline Order:**
1. Validate input data (NaN/Inf detection)
2. Filter tasks by output std and NaN/Inf (using `filter_and_normalize_task_outputs`)
3. Calculate normalization statistics from **filtered data only**

**Args:**
- `minimal_data_per_file`: Raw minimal data
- `min_std_threshold`: Minimum std for filtering
- `enable_filtering`: Whether to filter samples
- `verbose`: Print detailed info

**Returns:**
- `preprocessed_data`: Filtered data `[num_libs][num_valid_samples]`
- `norm_stats`: Normalization statistics
- `preprocessing_stats`: Detailed statistics

## Usage in Training Scripts

### Step 1: Import utilities

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from gnn_data_preprocessing_utils import (
    preprocess_gnn_minimal_data,
    normalize_node_features_safe,
    normalize_task_outputs
)
```

### Step 2: Apply preprocessing in data loading

```python
def load_cached_gnn_data_for_maml(process_type, corner_type, data_type='cell', graph_mode='stage_aware'):
    # ... load raw data ...

    # Apply preprocessing pipeline
    preprocessed_data, norm_stats, preprocessing_stats = preprocess_gnn_minimal_data(
        all_minimal_data_per_file,
        min_std_threshold=1e-6,
        enable_filtering=True,
        verbose=True
    )

    print(f"\n📊 Preprocessing Summary:")
    print(f"   Valid tasks after filtering: {preprocessing_stats['filtering']['valid_tasks']}")
    print(f"   Filter ratio: {preprocessing_stats['filtering']['filter_ratio']:.1f}%")

    return preprocessed_data, topology_cache, norm_stats
```

### Step 3: Use safe normalization in model

```python
class GNNCachedMAML:
    def normalize_node_features(self, node_features):
        """Normalize node features (with NaN/Inf protection)"""
        if self.norm_stats is None:
            return node_features

        normalized, _ = normalize_node_features_safe(
            node_features,
            norm_stats=self.norm_stats['node_features']
        )
        return normalized
```

## Comparison with voltage_variation_pretraining_utils.py

| Feature | Voltage Variation Utils | GNN Utils |
|---------|------------------------|-----------|
| **Structure** | Single module | Single module (matching) |
| **Data Format** | `[tasks, points, features]` | `[libs][samples]` (minimal data) |
| **Filtering** | `filter_and_normalize_outputs` | `filter_and_normalize_task_outputs` |
| **Input Norm** | `normalize_input_features` | `normalize_node_features` |
| **Output Norm** | Per-task in filtering step | Per-task in filtering step |
| **Safety Checks** | NaN/Inf, low std | NaN/Inf, low std (same) |
| **Pipeline** | `preprocess_voltage_data` | `preprocess_gnn_minimal_data` |
| **Task Definition** | Input conditions | Same input across libs |

## Example Output

When running preprocessing, you'll see:

```
================================================================================
🔧 GNN Data Preprocessing Pipeline
================================================================================

🔍 Validating GNN data...
   Total samples checked: 305000
   ✅ No data quality issues found!

🔍 Filtering and normalizing task outputs...
   Total libs: 61
   Total samples per lib: 5000
   Min std threshold: 1e-06

✅ Task filtering completed:
   Original tasks: 5000
   Valid tasks: 4850 (97.0%)
   Filtered out: 150 tasks
     - NaN/Inf: 10
     - Low std: 140

📊 Normalizing outputs for 4850 valid tasks (per-task normalization)...
   ✅ Per-task output normalization complete
   Normalized range: min=-3.124567, max=3.456789

🔧 Calculating normalization statistics (safe mode)...
   Total lib files: 61
   Sample rate: every 10th lib and task
   📊 Sampled 2425 samples
     Voltage: mean=0.700000, std=0.050000 (n=48500)
     Input Slew: mean=0.000015, std=0.000005 (n=48500)
     Output Load: mean=0.000010, std=0.000003 (n=48500)
     Output: mean=1.234567, std=0.567890 (n=2425)

================================================================================
✅ Preprocessing Complete
================================================================================

📊 Preprocessing Summary:
   Valid tasks after filtering: 4850
   Filter ratio: 97.0%
```

## Benefits

1. **Prevents NaN loss**: Filters out problematic samples before training
2. **Better numerical stability**: Safe normalization with edge case handling
3. **Data quality validation**: Reports statistics on filtered samples
4. **Consistent with voltage variation code**: Same structure and approach
5. **Filtered tasks excluded**: Tasks removed during filtering don't affect normalization

## Files Updated

- ✅ `gnn/utils/gnn_data_preprocessing_utils.py` - Main utilities (635 lines)
- ✅ `gnn/utils/__init__.py` - Package initialization
- ✅ `gnn/utils/README.md` - Documentation
- ✅ `maml_gnn_training_cached.py` - Using new utils
- ✅ `baseline_gnn_training_cached.py` - Using new utils
- ✅ `baseline_gnn_training_cached_global_norm.py` - Using new utils
- ✅ `maml_gnn_training_cached_global_norm.py` - Using new utils
