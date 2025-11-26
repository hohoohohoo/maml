# Model Test Code

This directory contains scripts for validating pretrained MAML and MLP models on various test datasets.

## Overview

The validation framework supports two types of testing:
1. **Topology Validation**: Testing across different cell topologies (ASAP7/TSMC, Intra/Agnostic)
2. **Voltage Variation Validation**: Testing across voltage variations (ASAP7 with cell types, TSMC with temperatures)

**Note**: This code uses models pretrained by scripts in `/model_pretraining_code/`. The pretrained models are automatically loaded from:
- MAML models: `../../pretrained_models/taskdivide_all/`
- MLP models: `../model_pretraining_code/MLP_pretrained_model/`

## File Types

### 1. Wrapper Scripts (run_*.py)
**User-friendly interface scripts for easy validation**

These scripts provide:
- Interactive mode with step-by-step prompts
- Command-line mode with argument parsing
- Parameter validation and confirmation
- Easy selection of model types, configurations, and test parameters

**Available Wrapper Scripts:**

- **`run_topology_validation.py`** ⭐
  - Wrapper for topology validation testing
  - Supports both MAML and MLP models
  - Dataset configurations: Intra-topology and Technology-agnostic
  - Automatically routes to appropriate validation scripts

- **`run_voltage_variation_validation.py`** ⭐
  - Wrapper for voltage variation testing
  - Supports both ASAP7 and TSMC PDKs
  - Configurable corner conditions and cell types/temperatures

### 2. Validation Implementation Scripts (*_validation.py)
**Actual validation scripts that load pretrained models and run tests**

These scripts contain:
- Pretrained model loading from `model_pretraining_code`
- Test dataset loading
- Model evaluation and metrics calculation
- Results aggregation and reporting

**Available Validation Scripts:**

#### Topology Validation
- **`MAML_topology_validation.py`**
  - MAML model validation across topologies
  - Loads models from: `../../pretrained_models/taskdivide_all/`
  - Supports extrapolation and interpolation modes

- **`MLP_topology_validation.py`**
  - MLP model validation across topologies
  - Loads models from: `../model_pretraining_code/MLP_pretrained_model/`
  - Supports both aadam (hidden=256) and mlp (hidden=40) architectures

#### Voltage Variation Validation
- **`ASAP7_voltage_variation_validation.py`**
  - ASAP7 PDK voltage variation validation
  - Supports multiple cell types (LVT, RVT, SLVT, SRAM)
  - Multiple corner conditions (SS, FF, TT)

- **`TSMC_voltage_variation_validation.py`**
  - TSMC PDK voltage variation validation
  - Multi-temperature support (0°C, 25°C, 50°C, 75°C, 100°C)
  - Corner condition support (ff, ss, tt)

### 3. Utility Modules (utils/)
**Helper modules providing shared functionality**

- **`test_dataset_config.py`**
  - Centralized configuration for test datasets and model paths
  - Defines 4 configurations (ASAP7/TSMC × Intra/Agnostic)
  - Provides model path mappings to pretrained models
  - Default parameters (cells, GPU ID, meta batch size, data type)

- **`data_management_utils.py`**
  - Data preprocessing and validation utilities
  - Continuity analysis for time-series data
  - Filtering functions for discontinuous tasks

- **`maml_functions.py`**
  - MAML-specific evaluation functions
  - Inner loop adaptation logic
  - Performance metrics calculation (NRMSE, MAPE, MAE)

- **`mlp_functions.py`**
  - MLP-specific evaluation functions
  - Direct inference without adaptation
  - Performance metrics calculation (NRMSE, MAPE)

## Model Loading Path Mapping

### MAML Models
```
Pretraining saves to: model_pretraining_code/../../pretrained_models/taskdivide_all/
                   → /pretrained_models/taskdivide_all/

Validation loads from: model_test_code/../../pretrained_models/taskdivide_all/
                    → /pretrained_models/taskdivide_all/
✅ Paths match correctly
```

### MLP Models
```
Pretraining saves to: model_pretraining_code/MLP_pretrained_model/
Validation loads from: model_test_code/../model_pretraining_code/MLP_pretrained_model/
✅ Paths match correctly
```

## Quick Start

### For Beginners: Use Wrapper Scripts

**Interactive Mode (Recommended):**
```bash
# Topology validation
python run_topology_validation.py

# Voltage variation validation
python run_voltage_variation_validation.py
```

The interactive mode will guide you through:
1. Model selection (MLP/MAML)
2. Configuration/PDK selection
3. Test parameters input
4. Confirmation before execution

**Command-Line Mode:**
```bash
# Topology validation - MAML on ASAP7 Intra
python run_topology_validation.py --model maml --config 0 --cells NAND3x2 OR2x6

# Topology validation - MLP on TSMC Agnostic
python run_topology_validation.py --model mlp --config 3 --mode interpolation

# Voltage variation - ASAP7 with MLP
python run_voltage_variation_validation.py --pdk asap7 --model mlp \
    --corner FF --cell_type lvt --mode extrapolation

# Voltage variation - TSMC with MAML
python run_voltage_variation_validation.py --pdk tsmc --model maml \
    --corner ff --temperatures 0 25 50 --inner_step 1
```

### For Advanced Users: Direct Execution

1. **Ensure pretrained models exist**
2. **Run validation script directly:**
   ```bash
   python MAML_topology_validation.py \
       --config 0 \
       --mode extrapolation \
       --cells NAND3x2 OR2x6 \
       --data_type cell \
       --gpu_id 0 \
       --inner 1 \
       --innerdiv 100 \
       --meta 32
   ```

## Key Parameters

### Common Parameters
- `--mode`: Testing mode (`extrapolation` or `interpolation`)
- `--data_type`: Data type (`cell` or `transition`)
- `--gpu_id`: GPU device ID
- `--indices`: Support set sampling indices
- `--save_results`: Save predictions/actuals to .npy files
- `--num_test_samples`: Number of test samples to process

### MAML Parameters
- `--inner`: Inner loop steps (default: 1)
- `--innerdiv`: Inner learning rate divisor (default: 100)
- `--meta`: Meta batch size / tasks per batch (default: config-dependent)
- `--layer_length`: Hidden layer size (default: 40)

### MLP Parameters
- `--model_type`: Architecture type (`aadam` with 256 hidden units or `mlp` with 40)
- `--num_iterations`: Number of pretraining iterations used

### Topology Validation Parameters
- `--config`: Dataset configuration (0-3)
  - 0: ASAP7 Intra Topology
  - 1: ASAP7 Technology Agnostic
  - 2: TSMC Intra Topology
  - 3: TSMC Technology Agnostic
- `--cells`: List of cell types to test

### Voltage Variation Parameters (PDK-Specific)

**ASAP7:**
- `--corner`: Process corner (SS/FF/TT)
- `--cell_type`: Cell threshold type (lvt/rvt/slvt/sram)

**TSMC:**
- `--corner`: Process corner (ff/ss/tt)
- `--temperatures`: Test temperatures (0/25/50/75/100)

## Output

All validation scripts provide:
- Real-time progress updates during testing
- Comprehensive metrics:
  - **NRMSE** (Normalized Root Mean Square Error)
  - **MAPE** (Mean Absolute Percentage Error)
  - **MAE** (Mean Absolute Error, MAML only)
  - **R² Score** (global fit quality)
- Region-specific metrics (left extrapolation, interpolation, right extrapolation)
- Optional .npy file output for predictions and actuals

### Output Files (when --save_results is used)
- Saved to: `data_result_npy_directory/`
- Format: `{PDK}_{topology_type}_{cell}_{data_type}_{mode}_{MODEL}_pred.npy`
- Format: `{PDK}_{topology_type}_{cell}_{data_type}_{mode}_{MODEL}_act.npy`

## File Organization

```
model_test_code/
├── run_*.py                                # 🚀 User-friendly wrappers
├── *_validation.py                         # 📊 Validation implementations
├── utils/                                  # 🛠️ Utility modules
│   ├── test_dataset_config.py             # Configuration & model paths
│   ├── data_management_utils.py           # Data utilities
│   ├── maml_functions.py                  # MAML evaluation
│   └── mlp_functions.py                   # MLP evaluation
├── data_result_npy_directory/             # 📁 Saved predictions/actuals
└── README.md                              # 📖 This file
```

## Prerequisites

Before running validation:
1. **Pretrained models must exist** in the correct locations
   - Run pretraining scripts in `/model_pretraining_code/` first
   - Or ensure models are available at the expected paths
2. **Test datasets must be available** at paths specified in `test_dataset_config.py`

## Notes

- **Always use wrapper scripts** (`run_*.py`) for best user experience
- Interactive mode guides you through all parameters
- Command-line mode allows automation and scripting
- Default parameters are optimized for typical use cases
- All scripts support `--help` for detailed parameter information
- Model paths are automatically resolved from configuration
