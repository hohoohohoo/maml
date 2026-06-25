# MLP Unified Pretraining

This document explains how to use the unified MLP pretraining script that consolidates all 4 dataset configurations into a single file.

## Overview

The `MLP_unified_pretraining.py` script replaces the following 4 separate files:
- `ASAP7_MLP_intra_topology_pretraining.py`
- `ASAP7_MLP_topology_agnostic_pretraining.py`
- `TSMC_MLP_intra_topology_pretraining.py`
- `TSMC_MLP_topology_agnostic_pretraining.py`

## Dataset Configurations

The script supports 4 dataset configurations via the `--dataset_config` argument:

| Config ID | Name | Technology | Topology Type | Dataset Source |
|-----------|------|------------|---------------|----------------|
| 0 | ASAP7_intra_topology | ASAP7 | intra | intra_topology_data_upgraded + unified_invbuf |
| 1 | ASAP7_topology_agnostic | ASAP7 | agnostic | unified_invbuf + topology_agnostic_data_upgraded |
| 2 | TSMC_intra_topology | TSMC | intra | dataset_tsmc_processed/intra_topology_data |
| 3 | TSMC_topology_agnostic | TSMC | agnostic | dataset_tsmc_processed/topology_agnostic_data |

## Usage

### Basic Usage

```bash
# Train ASAP7 intra-topology configuration
python MLP_unified_pretraining.py --dataset_config 0

# Train ASAP7 topology-agnostic configuration
python MLP_unified_pretraining.py --dataset_config 1

# Train TSMC intra-topology configuration
python MLP_unified_pretraining.py --dataset_config 2

# Train TSMC topology-agnostic configuration
python MLP_unified_pretraining.py --dataset_config 3
```

### Advanced Options

```bash
python MLP_unified_pretraining.py \
  --dataset_config 0 \
  --data_type cell \
  --num_iterations 300000 \
  --model_type aadam \
  --learning_rate 1e-4 \
  --gpu_id 0
```

## Command-Line Arguments

### Required Arguments
- `--dataset_config` (int, choices: 0-3): Dataset configuration to use

### Optional Arguments
- `--data_type` (str, default: 'cell'): Data type - 'cell' or 'transition'
- `--num_iterations` (int, default: 300000): Number of training iterations
- `--learning_rate` (float, default: 1e-4): Learning rate for Adam optimizer
- `--model_type` (str, default: 'aadam'): Model type - 'aadam' (hidden=256) or 'mlp' (hidden=40)
- `--gpu_id` (str, default: '0'): GPU device ID to use

## Examples

### Train on different GPUs in parallel

```bash
# Train 4 configurations in parallel on different GPUs
python MLP_unified_pretraining.py --dataset_config 0 --gpu_id 0 &
python MLP_unified_pretraining.py --dataset_config 1 --gpu_id 1 &
python MLP_unified_pretraining.py --dataset_config 2 --gpu_id 2 &
python MLP_unified_pretraining.py --dataset_config 3 --gpu_id 3 &
```

### Train with different model types

```bash
# Train with aadam model (hidden_size=256)
python MLP_unified_pretraining.py \
  --dataset_config 0 \
  --model_type aadam \
  --data_type cell

# Train with standard MLP model (hidden_size=40)
python MLP_unified_pretraining.py \
  --dataset_config 1 \
  --model_type mlp \
  --data_type transition
```

### Train with custom hyperparameters

```bash
# Train with smaller learning rate and more iterations
python MLP_unified_pretraining.py \
  --dataset_config 2 \
  --num_iterations 500000 \
  --learning_rate 5e-5 \
  --model_type aadam \
  --data_type cell
```

## Output Files

### Checkpoint Files
Saved periodically during training to:
```
MLP_pretrained_model/checkpoints_{topology_suffix}_{data_type}{tech_suffix}_{model_type}_{num_iterations}/
```

### Final Model Files
Saved at the end of training to:
```
MLP_pretrained_model/pretrained_mlp1_{topology_suffix}_{data_type}{tech_suffix}_{model_type}_{num_iterations}.pth
```

Where:
- `{topology_suffix}` = 'intratopology' (intra) or 'topology_agnostic' (agnostic)
- `{tech_suffix}` = '' (ASAP7) or '_tsmc' (TSMC)
- `{model_type}` = 'aadam' or 'mlp'
- `{data_type}` = 'cell' or 'transition'

## Model Naming Examples

- **Config 0 (ASAP7 intra, aadam)**: `pretrained_mlp1_intratopology_cell_aadam_300000.pth`
- **Config 1 (ASAP7 agnostic, mlp)**: `pretrained_mlp1_topology_agnostic_transition_mlp_300000.pth`
- **Config 2 (TSMC intra, aadam)**: `pretrained_mlp1_intratopology_cell_tsmc_aadam_300000.pth`
- **Config 3 (TSMC agnostic, mlp)**: `pretrained_mlp1_topology_agnostic_transition_tsmc_mlp_300000.pth`

## Training Configuration

- **Default iterations**: 300,000
- **Default learning rate**: 1e-4
- **Optimizer**: Adam with weight decay = 0
- **Hidden layer sizes**:
  - aadam model: 256
  - mlp model: 40
- **Input features**: 9 (automatically determined from data)
- **Checkpoint frequency**: Every 10,000 iterations (configurable in `mlp_utils.py`)

## Data Normalization

The script automatically normalizes:
1. **Input features** (indices 7, 8, 3, 4):
   - slew (index 7)
   - load_cap (index 8)
   - temperature (index 3)
   - voltage (index 4)

2. **Output values**: z-score normalization per task

3. **Task filtering**: Removes tasks with std < 1e-6

## Utilities Used

The script uses functions from multiple utility modules:

### `mlp_utils.py`
- `MLP_pretraining`: MLP training class with Adam optimizer
- `normalize_features()`: Normalizes input features
- `normalize_outputs()`: Normalizes outputs and filters invalid tasks
- `save_model()`: Saves trained model with metadata

### `dataset_config.py`
- `DATASET_CONFIGS`: Dictionary mapping configuration IDs (0-3) to dataset information
- `get_dataset_config()`: Retrieves dataset configuration by ID
- `print_available_datasets()`: Displays all available dataset configurations
- `load_dataset_by_config()`: Loads appropriate dataset based on configuration ID

## Migration from Old Scripts

If you were using the old separate scripts, simply replace:

```bash
# Old
python ASAP7_MLP_intra_topology_pretraining.py --data_type cell --num_iterations 300000

# New
python MLP_unified_pretraining.py --dataset_config 0 --data_type cell --num_iterations 300000
```

The model files generated will have the same naming convention and are fully compatible.

## Comparison with MAML

Both MLP and MAML unified pretraining scripts use the same dataset configuration system:

| Feature | MLP Unified | MAML Unified |
|---------|-------------|--------------|
| Dataset configs | 0-3 | 0-3 |
| Data types | cell/transition | cell/transition |
| Model types | aadam/mlp | N/A (layer_length parameter) |
| Special params | learning_rate | inner, innerdiv, meta |
| Resume support | No | Yes (--resume, --auto_resume) |

## Notes

1. **Model Type Selection**: Use 'aadam' for larger capacity (256 hidden units) and 'mlp' for smaller models (40 hidden units)
2. **GPU Memory**: The aadam model requires more GPU memory than the mlp model
3. **Training Time**: Training time varies based on dataset size and model type
4. **Checkpoint Saving**: Checkpoints are automatically saved during training for recovery
5. **Data Consistency**: All configurations use the same normalization and filtering procedures
