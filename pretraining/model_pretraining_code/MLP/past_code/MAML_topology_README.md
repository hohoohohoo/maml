# MAML Unified Pretraining

This document explains how to use the unified MAML pretraining script that consolidates all 4 dataset configurations into a single file.

## Overview

The `MAML_unified_pretraining.py` script replaces the following 4 separate files:
- `ASAP7_MAML_intra_topology_pretraining.py`
- `ASAP7_MAML_topology_agnostic_pretraining.py`
- `TSMC_MAML_intra_topology_pretraining.py`
- `TSMC_MAML_topology_agnostic_pretraining.py`

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
python MAML_unified_pretraining.py --dataset_config 0

# Train ASAP7 topology-agnostic configuration
python MAML_unified_pretraining.py --dataset_config 1

# Train TSMC intra-topology configuration
python MAML_unified_pretraining.py --dataset_config 2

# Train TSMC topology-agnostic configuration
python MAML_unified_pretraining.py --dataset_config 3
```

### Advanced Options

```bash
python MAML_unified_pretraining.py \
  --dataset_config 0 \
  --data_type cell \
  --inner 1 \
  --innerdiv 100 \
  --meta 32 \
  --gpu 0
```

### Resume Training

```bash
# Resume from specific checkpoint
python MAML_unified_pretraining.py \
  --dataset_config 0 \
  --resume /path/to/checkpoint.pth

# Auto-resume from latest checkpoint
python MAML_unified_pretraining.py \
  --dataset_config 0 \
  --auto_resume
```

## Command-Line Arguments

### Required Arguments
- `--dataset_config` (int, choices: 0-3): Dataset configuration to use

### Optional Arguments
- `--data_type` (str, default: 'cell'): Data type - 'cell' or 'transition'
- `--inner` (int, default: 1): Number of inner loop steps
- `--innerdiv` (int, default: 100): Inner learning rate divisor (inner_lr = 0.001/innerdiv)
- `--meta` (int, default: 32): Number of tasks per meta batch
- `--gpu` (str, default: '0'): GPU device ID to use
- `--resume` (str): Path to specific pretrained model file to resume from
- `--auto_resume` (flag): Automatically find and resume from latest pretrained model

## Examples

### Train on different GPUs

```bash
# Train 4 configurations in parallel on different GPUs
python MAML_unified_pretraining.py --dataset_config 0 --gpu 0 &
python MAML_unified_pretraining.py --dataset_config 1 --gpu 1 &
python MAML_unified_pretraining.py --dataset_config 2 --gpu 2 &
python MAML_unified_pretraining.py --dataset_config 3 --gpu 3 &
```

### Train with custom hyperparameters

```bash
# Train with smaller inner learning rate
python MAML_unified_pretraining.py \
  --dataset_config 0 \
  --innerdiv 200 \
  --data_type cell

# Train with more meta batch tasks
python MAML_unified_pretraining.py \
  --dataset_config 1 \
  --meta 64 \
  --data_type transition
```

## Output Files

### Checkpoint Files
Saved every 30,000 iterations to:
```
../../pretrained_models/checkpoints/taskdivide_all_checkpoints/
{data_type}_innerdiv{innerdiv}_meta{meta}_{model_suffix}_519traintask_full1DMAML_weights_3hidden_(40)_{iteration}_inner{inner}_upgraded{tech_suffix}.pth
```

### Final Model Files
Saved at the end of training to:
```
../../pretrained_models/taskdivide_all/
{data_type}_innerdiv{innerdiv}_meta{meta}_{model_suffix}_519traintask_full1DMAML_weights_3hidden_(40)_{final_iteration}_inner{inner}_upgraded{tech_suffix}.pth
```

Where:
- `{model_suffix}` = 'intratopology' (ASAP7 intra) or 'intra_topology' (TSMC intra) or 'topology_agnostic' (both agnostic)
- `{tech_suffix}` = '' (ASAP7) or '_tsmc' (TSMC)

## Model Naming Examples

- **Config 0 (ASAP7 intra)**: `cell_innerdiv100_meta32_intratopology_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded.pth`
- **Config 1 (ASAP7 agnostic)**: `cell_innerdiv100_meta32_topology_agnostic_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded.pth`
- **Config 2 (TSMC intra)**: `cell_innerdiv100_meta32_intra_topology_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth`
- **Config 3 (TSMC agnostic)**: `cell_innerdiv100_meta32_topology_agnostic_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth`

## Training Configuration

- **Total iterations**: 300,000 (default)
- **Chunk size**: 30,000 iterations per chunk
- **Inner learning rate**: 0.001 / innerdiv (default: 0.00001)
- **Meta learning rate**: 0.0001
- **Layer length**: 40 (hidden layer size)
- **Input features**: 9 (automatically determined from data)

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

### `maml_utils.py`
- `normalize_input_features()`: Normalizes input features (slew, load_cap, temperature, voltage)
- `normalize_and_filter_tasks()`: Normalizes outputs and filters invalid tasks
- `extract_iteration_from_filename()`: Extracts iteration number from model filename
- `find_pretrained_model()`: Finds latest checkpoint for auto-resume
- `load_pretrained_model()`: Loads model weights from checkpoint

### `dataset_config.py`
- `DATASET_CONFIGS`: Dictionary mapping configuration IDs (0-3) to dataset information
- `get_dataset_config()`: Retrieves dataset configuration by ID
- `print_available_datasets()`: Displays all available dataset configurations
- `load_dataset_by_config()`: Loads appropriate dataset based on configuration ID

## Migration from Old Scripts

If you were using the old separate scripts, simply replace:

```bash
# Old
python ASAP7_MAML_intra_topology_pretraining.py --inner 1 --innerdiv 100

# New
python MAML_unified_pretraining.py --dataset_config 0 --inner 1 --innerdiv 100
```

The model files generated will have the same naming convention and are fully compatible.
