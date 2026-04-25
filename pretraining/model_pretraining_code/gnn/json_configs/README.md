# GNN Architecture Sweep Configuration

This directory contains JSON configuration files for running GNN architecture hyperparameter sweeps.

## Two Sweep Systems

### 1. **Simple Sweep** (Single script generation)
- Uses `parse_gnn_sweep_config.py`
- Generates static shell script with all commands
- Good for: Pre-defined architecture comparisons

### 2. **MAML-style Sweep** (Dynamic multi-parameter sweep)
- Uses `parse_gnn_maml_sweep_config.py` + `run_gnn_maml_sweep.sh`
- Cartesian product of 4 parameters
- Good for: Comprehensive hyperparameter search

---

## System 1: Simple Sweep

### Quick Start

#### 1. Generate sweep commands

```bash
python utils/parse_gnn_sweep_config.py \
  --config json_configs/gnn_architecture_sweep.json \
  --script maml_gnn_training_cached.py \
  --output run_gnn_architecture_sweep.sh
```

#### 2. Run the sweep

```bash
bash run_gnn_architecture_sweep.sh
```

Or run in background with nohup:

```bash
nohup bash run_gnn_architecture_sweep.sh > sweep_log.txt 2>&1 &
```

---

## System 2: MAML-style Sweep (Recommended)

**Two Execution Modes:**

- **V1 (Legacy)**: Separate Python process per architecture - slower, loads data N times
- **V2 (Recommended)**: Single Python process - **much faster**, loads data once

### Quick Start - MAML Training (V2 - Efficient)

#### 1. Dry-run (preview command)

```bash
bash run_gnn_maml_sweep_v2.sh json_configs/gnn_maml_sweep_config.json --dry-run
```

#### 2. Run the sweep

```bash
bash run_gnn_maml_sweep_v2.sh json_configs/gnn_maml_sweep_config.json
```

Or in background:

```bash
nohup bash run_gnn_maml_sweep_v2.sh json_configs/gnn_maml_sweep_config.json > sweep_log.txt 2>&1 &
```

#### 3. Skip git commit

```bash
bash run_gnn_maml_sweep_v2.sh json_configs/gnn_maml_sweep_config.json --no-commit
```

#### 4. Direct Python call (alternative)

```bash
python maml_gnn_training_cached.py \
    --process LVT --corner FF \
    --conv_hidden_dim 64 128 256 \
    --num_conv_layers 2 3 4 \
    --fc_hidden_dim 40 64 128 \
    --num_fc_layers 2 3 4
```

### Quick Start - Baseline Training (V2 - Efficient)

#### 1. Dry-run (preview command)

```bash
bash run_baseline_gnn_sweep_v2.sh json_configs/baseline_gnn_sweep_config.json --dry-run
```

#### 2. Run the sweep

```bash
bash run_baseline_gnn_sweep_v2.sh json_configs/baseline_gnn_sweep_config.json
```

Or in background:

```bash
nohup bash run_baseline_gnn_sweep_v2.sh json_configs/baseline_gnn_sweep_config.json > sweep_log.txt 2>&1 &
```

#### 3. Skip git commit

```bash
bash run_baseline_gnn_sweep_v2.sh json_configs/baseline_gnn_sweep_config.json --no-commit
```

#### 4. Direct Python call (alternative)

```bash
python baseline_gnn_training_cached.py \
    --process LVT --corner FF \
    --conv_hidden_dim 64 128 256 \
    --num_conv_layers 2 3 4 \
    --fc_hidden_dim 40 64 128 \
    --num_fc_layers 1 2 3
```

---

## Configuration File Structure

### System 1: Simple Sweep Format

**Example:** `gnn_architecture_sweep.json`

```json
{
  "sweep_description": "GNN Architecture Hyperparameter Sweep",
  "sweep_type": "architecture",
  "vary_parameter": "conv_hidden_dim",

  "common_parameters": {
    "data_dir": "/path/to/dataset",
    "process": "LVT",
    "corner": "FF",
    "num_iterations": 100000,
    "meta_lr": 0.001,
    ...
  },

  "sweep_configs": [
    {
      "id": 1,
      "description": "Conv128_Conv3_FC40_FC3",
      "parameters": {
        "conv_hidden_dim": 128,
        "num_conv_layers": 3,
        "fc_hidden_dim": 40,
        "num_fc_layers": 3
      }
    },
    ...
  ]
}
```

**Fields:**
- **sweep_description**: Human-readable description of the sweep
- **sweep_type**: Type of sweep (e.g., "architecture", "learning_rate")
- **vary_parameter**: Main parameter being swept (for analysis)
- **common_parameters**: Parameters shared across all configs
- **sweep_configs**: List of individual configurations
  - **id**: Unique identifier for this config
  - **description**: Short description (used in filenames)
  - **parameters**: Config-specific parameters (override common_parameters)

### System 2: MAML-style Sweep Format

**Example:** `gnn_maml_sweep_config.json`

```json
{
  "experiment_name": "gnn_maml_architecture_sweep",
  "description": "Sweep over 4 architecture parameters",
  "base_config": {
    "data_dir": "/path/to/dataset",
    "process": "LVT",
    "corner": "FF",
    "num_iterations": 100000,
    "meta_lr": 0.001,
    "inner_lr": 0.01,
    "K": 5,
    "auto_resume": true,
    ...
  },
  "sweep_params": {
    "conv_hidden_dim": [64, 128, 256],
    "num_conv_layers": [2, 3, 4],
    "fc_hidden_dim": [40, 64, 128],
    "num_fc_layers": [2, 3, 4]
  }
}
```

**Fields:**
- **experiment_name**: Experiment identifier
- **description**: Human-readable description
- **base_config**: Base parameters (all configs share these)
- **sweep_params**: Parameters to sweep (cartesian product)
  - Each key maps to a list of values to try
  - Total experiments = product of all list lengths
  - Example: `[3, 3, 3, 3]` → 81 experiments

**Cartesian Product Example:**
```json
"sweep_params": {
  "conv_hidden_dim": [64, 128],      // 2 values
  "num_conv_layers": [2, 3],         // 2 values
  "fc_hidden_dim": [40, 64],         // 2 values
  "num_fc_layers": [2, 3]            // 2 values
}
// Total: 2 × 2 × 2 × 2 = 16 experiments
```

---

## Available Sweep Configurations

### System 1: Simple Sweep

#### `gnn_architecture_sweep.json`
- **Type**: Manual architecture comparison
- **Total configs**: 9
- **Varies**: Different architectures (one at a time)

### System 2: MAML-style Sweep

#### MAML Training

**`gnn_maml_sweep_config.json`**
- **Type**: Full 4-parameter sweep
- **Total configs**: 81 (3×3×3×3)
- **Parameters**:
  - conv_hidden_dim: [64, 128, 256]
  - num_conv_layers: [2, 3, 4]
  - fc_hidden_dim: [40, 64, 128]
  - num_fc_layers: [2, 3, 4]

**`gnn_maml_sweep_example.json`**
- **Type**: Smaller test sweep
- **Total configs**: 9 (1×3×3×1)
- **Parameters**:
  - conv_hidden_dim: [128] (fixed)
  - num_conv_layers: [2, 3, 4]
  - fc_hidden_dim: [40, 64, 128]
  - num_fc_layers: [3] (fixed)

#### Baseline Training

**`baseline_gnn_sweep_config.json`**
- **Type**: Full 4-parameter sweep (Baseline)
- **Total configs**: 81 (3×3×3×3)
- **Parameters**:
  - conv_hidden_dim: [64, 128, 256]
  - num_conv_layers: [2, 3, 4]
  - fc_hidden_dim: [40, 64, 128]
  - num_fc_layers: [1, 2, 3]

**`baseline_gnn_sweep_example.json`**
- **Type**: Smaller test sweep (Baseline)
- **Total configs**: 9 (1×3×3×1)
- **Parameters**:
  - conv_hidden_dim: [128] (fixed)
  - num_conv_layers: [2, 3, 4]
  - fc_hidden_dim: [40, 64, 128]
  - num_fc_layers: [2] (fixed)

---

## Creating Custom Sweeps

### System 2 Examples (MAML-style)

#### Example 1: Learning Rate Sweep

Create `gnn_lr_sweep.json`:

```json
{
  "experiment_name": "gnn_lr_sweep",
  "description": "Meta learning rate sweep",
  "base_config": {
    "data_dir": "/path/to/dataset",
    "process": "LVT",
    "corner": "FF",
    "conv_hidden_dim": 128,
    "num_conv_layers": 3,
    "fc_hidden_dim": 64,
    "num_fc_layers": 3,
    "num_iterations": 100000,
    "inner_lr": 0.01,
    "K": 5,
    "auto_resume": true
  },
  "sweep_params": {
    "meta_lr": [0.0001, 0.0005, 0.001, 0.005, 0.01]
  }
}
```

**Total experiments:** 5 (just varying meta_lr)

Run:
```bash
bash run_gnn_maml_sweep.sh json_configs/gnn_lr_sweep.json
```

#### Example 2: Process & Corner Sweep

Create `gnn_process_corner_sweep.json`:

```json
{
  "experiment_name": "gnn_process_corner",
  "description": "Sweep across process corners",
  "base_config": {
    "data_dir": "/path/to/dataset",
    "data_type": "cell",
    "graph_mode": "stage_aware",
    "conv_hidden_dim": 128,
    "num_conv_layers": 3,
    "fc_hidden_dim": 64,
    "num_fc_layers": 3,
    "num_iterations": 100000,
    "meta_lr": 0.001,
    "inner_lr": 0.01,
    "K": 5,
    "auto_resume": true
  },
  "sweep_params": {
    "process": ["RVT", "LVT", "SLVT"],
    "corner": ["TT", "FF", "SS"]
  }
}
```

**Total experiments:** 9 (3×3 = 9 process-corner combinations)

#### Example 3: Mixed Sweep

```json
{
  "experiment_name": "gnn_mixed_sweep",
  "description": "Architecture + learning rate sweep",
  "base_config": {
    "data_dir": "/path/to/dataset",
    "process": "LVT",
    "corner": "FF",
    "num_iterations": 100000,
    "inner_lr": 0.01,
    "K": 5,
    "auto_resume": true
  },
  "sweep_params": {
    "conv_hidden_dim": [64, 128, 256],
    "fc_hidden_dim": [40, 64],
    "meta_lr": [0.0005, 0.001, 0.002]
  }
}
```

**Total experiments:** 18 (3×2×3 = 18)

## Analysis

After running sweeps, use the result management tools:

```bash
cd /home/tkdgn2907/Deepsets_test/MAML/Projects/result_management

# Analyze sweep results
python analyze_sweep_results.py \
  --data_dir ../pretraining/model_test_code/gnn/data_result_npy_directory \
  --output_dir ./gnn_sweep_results \
  --vary conv_hidden_dim

# With aggregation across cells
python analyze_sweep_results.py \
  --data_dir ../pretraining/model_test_code/gnn/data_result_npy_directory \
  --output_dir ./gnn_sweep_results \
  --vary conv_hidden_dim \
  --aggregate
```

## Performance Comparison: V1 vs V2

### V1 (Legacy): Separate Processes
- Each architecture = new Python process
- Data loaded 81 times for 81 architectures
- Total time ≈ 81 × (data_load_time + training_time)
- **Example**: 5min load × 81 + 30min train × 81 = **47.25 hours**

### V2 (Efficient): Single Process
- One Python process for all architectures
- Data loaded once, shared across architectures
- Total time ≈ data_load_time + (81 × training_time)
- **Example**: 5min load + 30min train × 81 = **40.58 hours**
- **Savings: ~6.7 hours (14% faster)** for this example

### When to Use Each

**Use V2 (Recommended) when:**
- Sweeping over architecture parameters only
- Same process/corner for all experiments
- Want maximum efficiency

**Use V1 when:**
- Need complete isolation between experiments
- Different process/corner combinations
- Using the old JSON format

## Tips

1. **Start with a small sweep**: Test 2-3 configs first to verify everything works
2. **Use descriptive IDs**: Make descriptions match the varied parameters
3. **Monitor resources**: Check GPU memory usage, especially for large models
4. **Save logs**: Use `nohup` or redirect output to log files
5. **Checkpoint frequently**: Set reasonable checkpoint intervals in training

## Supported Training Scripts

- `maml_gnn_training_cached.py`: MAML training with cached topology
- `maml_gnn_training_cached_global_norm.py`: MAML with global normalization
- `baseline_gnn_training_cached.py`: Baseline (non-MAML) training
- `baseline_gnn_training_cached_global_norm.py`: Baseline with global normalization

## Command-Line Options

### Parser Script

```bash
python utils/parse_gnn_sweep_config.py --help

Options:
  --config PATH       Path to sweep configuration JSON file (required)
  --script NAME       Training script name (default: maml_gnn_training_cached.py)
  --output PATH       Output shell script file (default: stdout)
  --dry-run          Print commands without executing
```

### Examples

Preview commands without creating script:
```bash
python utils/parse_gnn_sweep_config.py \
  --config json_configs/gnn_architecture_sweep.json
```

Use different training script:
```bash
python utils/parse_gnn_sweep_config.py \
  --config json_configs/gnn_architecture_sweep.json \
  --script baseline_gnn_training_cached.py \
  --output run_baseline_sweep.sh
```
