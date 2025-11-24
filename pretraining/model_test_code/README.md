# Model Test Code

This directory contains scripts for validating pretrained MAML and MLP models on various test datasets.

## Overview

The validation framework supports two types of testing:
1. **Topology Validation**: Testing across different cell topologies (ASAP7/TSMC, Intra/Agnostic)
2. **Voltage Variation Validation**: Testing across voltage variations (ASAP7 with cell types, TSMC with temperatures)

## Main Scripts

### Entry Point Scripts (Recommended)

- **`run_topology_validation.py`** ⭐
  - Unified wrapper for topology validation testing
  - Supports both interactive and command-line modes
  - Automatically routes to MAML or MLP topology validation scripts
  - Handles parameter validation and configuration selection

- **`run_voltage_variation_validation.py`** ⭐
  - Unified wrapper for voltage variation testing
  - Supports both ASAP7 and TSMC PDKs
  - Automatically routes to appropriate validation scripts based on PDK selection
  - Provides interactive parameter input with sensible defaults

### Core Validation Scripts

#### Topology Validation
- **`MAML_topology_validation.py`**
  - MAML model validation across different cell topologies
  - Supports extrapolation and interpolation modes
  - Configurable through `test_dataset_config.py`

- **`MLP_topology_validation.py`**
  - MLP model validation across different cell topologies
  - Supports both MLP and MLP_Aadam architectures
  - Parallel testing of multiple cell types

#### Voltage Variation Validation
- **`ASAP7_voltage_variation_validation.py`**
  - ASAP7 PDK voltage variation validation
  - Supports multiple cell types (LVT, RVT, SLVT, SRAM)
  - Multiple corner conditions (SS, FF, TT)

- **`TSMC_voltage_variation_validation.py`**
  - TSMC PDK voltage variation validation
  - Multi-temperature support (0°C, 25°C, 50°C, 75°C, 100°C)
  - Corner condition support (ff, ss, tt)

## Utility Modules

- **`test_dataset_config.py`**
  - Centralized configuration for test datasets
  - Defines 4 configurations:
    - [0] ASAP7 Intra Topology
    - [1] ASAP7 Technology Agnostic
    - [2] TSMC Intra Topology
    - [3] TSMC Technology Agnostic
  - Provides default parameters (cells, GPU ID, meta batch size, data type)

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

## Quick Start

### Interactive Mode (Easiest)

**For Topology Validation:**
```bash
python run_topology_validation.py
```

**For Voltage Variation Validation:**
```bash
python run_voltage_variation_validation.py
```

### Command-Line Mode

**Topology Validation Examples:**
```bash
# MAML on ASAP7 Intra Topology
python run_topology_validation.py --model maml --config 0 --cells NAND3x2 OR2x6

# MLP on TSMC Agnostic
python run_topology_validation.py --model mlp --config 3 --mode interpolation
```

**Voltage Variation Validation Examples:**
```bash
# ASAP7 with MLP
python run_voltage_variation_validation.py --pdk asap7 --model mlp \
    --corner FF --cell_type lvt --mode extrapolation

# TSMC with MAML (specific temperatures)
python run_voltage_variation_validation.py --pdk tsmc --model maml \
    --corner ff --temperatures 0 25 50 --inner_step 1
```

## Key Parameters

### Common Parameters
- `--mode`: Testing mode (`extrapolation` or `interpolation`)
- `--data_type`: Data type (`cell` or `transition`)
- `--gpu_id`: GPU device ID
- `--indices`: Support set sampling indices
- `--save_results`: Save predictions/actuals to .npy files

### MAML Parameters
- `--inner`: Inner loop steps
- `--innerdiv`: Inner learning rate divisor
- `--meta`: Meta batch size (tasks per batch)
- `--layer_length`: Hidden layer size

### MLP Parameters
- `--model_type`: Architecture type (`aadam` with 256 hidden units or `mlp` with 40)
- `--num_iterations`: Number of pretraining iterations

### PDK-Specific Parameters -> only case in voltage vartion validation code

**ASAP7:**
- `--corner`: Process corner (SS/FF/TT)
- `--cell_type`: Cell threshold type (lvt/rvt/slvt/sram)

**TSMC:**
- `--corner`: Process corner (ff/ss/tt)
- `--temperatures`: Test temperatures (0/25/50/75/100)

## Output

All scripts provide:
- Real-time progress updates during testing
- Comprehensive metrics:
  - NRMSE (Normalized Root Mean Square Error)
  - MAPE (Mean Absolute Percentage Error)
  - MAE (Mean Absolute Error, MAML only)
  - R² Score (global)
- Optional .npy file output for predictions and actuals

## File Organization

```
model_test_code/
├── run_topology_validation.py              # 🚀 Topology wrapper
├── run_voltage_variation_validation.py     # 🚀 Voltage variation wrapper
├── MAML_topology_validation.py             # MAML topology testing
├── MLP_topology_validation.py              # MLP topology testing
├── ASAP7_voltage_variation_validation.py   # ASAP7 voltage_variation testing
├── TSMC_voltage_variation_validation.py    # TSMC voltage_variation testing
├── test_dataset_config.py                  # Configuration manager
├── data_management_utils.py                # Data utilities
├── maml_functions.py                       # MAML evaluation
└── mlp_functions.py                        # MLP evaluation
```

## Notes

- Use wrapper scripts (`run_*.py`) for best user experience
- Interactive mode guides you through all parameters
- Command-line mode allows automation and scripting
- Default parameters are optimized for typical use cases
- All scripts support `--help` for detailed parameter information
