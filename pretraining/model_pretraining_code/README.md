# Model Pretraining Code

This directory contains scripts for pretraining deep learning models (MLP and MAML) for delay prediction tasks.

## Directory Structure

```
model_pretraining_code/
├── run_*.py                    # User-friendly wrapper scripts
├── *_pretraining.py            # Actual training implementation scripts
├── utils/                      # Utility modules
└── README.md                   # This file
```

## File Types

### 1. Wrapper Scripts (run_*.py)
**User-friendly interface scripts that simplify model training**

These scripts provide:
- Interactive mode with step-by-step prompts
- Command-line mode with argument parsing
- Parameter validation and confirmation
- Easy selection of model types, configurations, and hyperparameters

**Available Wrapper Scripts:**

- **`run_topology_pretraining.py`**
  - Wrapper for topology-based pretraining (ASAP7/TSMC)
  - Supports both MLP and MAML models
  - Dataset configurations: Intra-topology and Technology-agnostic

- **`run_voltage_variation_pretraining.py`**
  - Wrapper for voltage variation pretraining
  - Supports both ASAP7 and TSMC PDKs
  - Configurable corner conditions and cell types

**Usage Examples:**
```bash
# Interactive mode (recommended for beginners)
python run_topology_pretraining.py

# Command-line mode with confirmation
python run_topology_pretraining.py maml --config 0 --data_type cell

# Skip confirmation prompt
python run_topology_pretraining.py maml --config 0 --data_type cell --yes
```

### 2. Training Implementation Scripts (*_pretraining.py)
**Actual training scripts that perform the model pretraining**

These scripts contain:
- Model architecture implementation
- Training loop logic
- Data loading and preprocessing
- Checkpoint management
- Loss calculation and optimization

**Available Training Scripts:**

- **`MAML_topology_pretraining.py`**
  - MAML model training for topology datasets
  - Meta-learning optimization
  - Supports resume from checkpoints

- **`MLP_topology_pretraining.py`**
  - MLP model training for topology datasets
  - Standard supervised learning

- **`ASAP7_voltage_variation_pretraining.py`**
  - ASAP7 PDK voltage variation training
  - Cell type and corner-specific training

- **`TSMC_voltage_variation_pretraining.py`**
  - TSMC PDK voltage variation training
  - Temperature and corner-specific training

**Direct Usage (Advanced):**
```bash
# Direct execution with all parameters specified
python MAML_topology_pretraining.py --dataset_config 0 --data_type cell --gpu 0 --inner 1 --innerdiv 100 --meta 32
```

### 3. Utility Modules (utils/)
**Helper modules providing shared functionality**

- **`dataset_config.py`**
  - Dataset configuration management
  - Path definitions for training data
  - Configuration mappings (ASAP7/TSMC, Intra/Agnostic)

- **`maml_utils.py`**
  - MAML-specific utility functions
  - Model loading and saving
  - Feature normalization
  - Task filtering

- **`mlp_utils.py`**
  - MLP-specific utility functions
  - Training class implementation
  - Feature and output normalization

- **`voltage_variation_pretraining_utils.py`**
  - Voltage variation specific utilities
  - PDK-specific data loading
  - Corner and temperature handling

## Quick Start

### For Beginners: Use Wrapper Scripts

1. **Start interactive mode:**
   ```bash
   python run_topology_pretraining.py
   ```

2. **Follow the prompts:**
   - Select model framework (MLP/MAML)
   - Choose dataset configuration
   - Enter training parameters
   - Review and confirm settings

3. **Training begins automatically**

### For Advanced Users: Direct Execution

1. **Prepare your configuration**
2. **Run training script directly:**
   ```bash
   python MAML_topology_pretraining.py \
       --dataset_config 0 \
       --data_type cell \
       --gpu 0 \
       --inner 1 \
       --innerdiv 100 \
       --meta 32
   ```

## Model Types

### MLP (Multi-Layer Perceptron)
- Standard feedforward neural network
- Two variants: `aadam` (hidden=256) and `mlp` (hidden=40)
- Suitable for single-task learning

### MAML (Model-Agnostic Meta-Learning)
- Meta-learning approach for fast adaptation
- Configurable inner loop steps and learning rates
- Ideal for few-shot learning scenarios

## Dataset Configurations

### Topology-based:
- **Config 0**: ASAP7 Intra Topology
- **Config 1**: ASAP7 Technology Agnostic
- **Config 2**: TSMC Intra Topology
- **Config 3**: TSMC Technology Agnostic

### Voltage Variation:
- **ASAP7**: Corner conditions (SS/FF/TT), Cell types (lvt/rvt/slvt/sram)
- **TSMC**: Temperatures (0/25/50/75/100°C), Corner conditions (ff/ss/tt)

## Output

Trained models are saved in:
- `../../pretrained_models/taskdivide_all/` (Topology models)
- `MLP_pretrained_model/` (MLP specific models)
- With automatic checkpoint management in respective checkpoint directories

## Notes

- **Always use wrapper scripts** (`run_*.py`) for user-friendly experience
- Direct script execution is available for automation and advanced use cases
- All scripts support GPU acceleration (specify GPU ID)
- Training can be resumed from checkpoints (MAML only)
