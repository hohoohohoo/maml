# Dataset Configuration Module

This document explains the `dataset_config.py` module used by both MAML and MLP unified pretraining scripts.

## Overview

The `dataset_config.py` module provides a centralized system for managing different dataset configurations. It eliminates code duplication by consolidating dataset loading logic that was previously scattered across 8 separate files (4 MAML + 4 MLP).

## Dataset Configurations

The module defines 4 dataset configurations that cover all combinations of technology (ASAP7/TSMC) and topology type (intra/agnostic):

| Config ID | Name | Technology | Topology Type | Description |
|-----------|------|------------|---------------|-------------|
| 0 | ASAP7_intra_topology | ASAP7 | intra | ASAP7 intra-topology dataset |
| 1 | ASAP7_topology_agnostic | ASAP7 | agnostic | ASAP7 topology-agnostic dataset |
| 2 | TSMC_intra_topology | TSMC | intra | TSMC intra-topology dataset |
| 3 | TSMC_topology_agnostic | TSMC | agnostic | TSMC topology-agnostic dataset |

## Dataset Sources

Each configuration loads data from specific paths:

### Config 0: ASAP7 Intra-Topology
```
/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/intra_topology_data_upgraded/
  - {data_type}_intratopology_train_input.pth
  - {data_type}_intratopology_train_output.pth

+ /home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/unified_invbuf/
  - merged_invbuf_input_{data_type}.pth
  - merged_invbuf_output_{data_type}.pth
```

### Config 1: ASAP7 Topology-Agnostic
```
/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/unified_invbuf/
  - merged_invbuf_input_{data_type}.pth
  - merged_invbuf_output_{data_type}.pth

+ /home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_ASAP7/topology_agnostic_data_upgraded/
  - {data_type}_topology_agnostic_train_input.pth
  - {data_type}_topology_agnostic_train_output.pth
```

### Config 2: TSMC Intra-Topology
```
/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_tsmc_processed/intra_topology_data/
  - tsmc_intra_topology_train_input_{data_type}.pth
  - tsmc_intra_topology_train_output_{data_type}.pth
```

### Config 3: TSMC Topology-Agnostic
```
/home/tkdgn2907/Deepsets_test/MAML/Projects/dataset_all/dataset_tsmc_processed/topology_agnostic_data/
  - tsmc_topology_agnostic_train_input_{data_type}.pth
  - tsmc_topology_agnostic_train_output_{data_type}.pth
```

## Module Functions

### `get_dataset_config(config_id)`

Retrieves configuration information for a given ID.

**Parameters:**
- `config_id` (int): Dataset configuration ID (0-3)

**Returns:**
- `dict`: Configuration dictionary containing:
  - `name` (str): Dataset name
  - `tech` (str): Technology ('asap7' or 'tsmc')
  - `topology_type` (str): Topology type ('intra' or 'agnostic')
  - `description` (str): Human-readable description

**Raises:**
- `ValueError`: If config_id is not in valid range (0-3)

**Example:**
```python
from dataset_config import get_dataset_config

config = get_dataset_config(0)
print(config)
# Output: {
#   'name': 'ASAP7_intra_topology',
#   'tech': 'asap7',
#   'topology_type': 'intra',
#   'description': 'ASAP7 intra-topology dataset ...'
# }
```

### `print_available_datasets()`

Prints all available dataset configurations to stdout.

**Parameters:** None

**Returns:** None

**Example:**
```python
from dataset_config import print_available_datasets

print_available_datasets()
# Output:
# 📋 Available dataset configurations:
#    [0] ASAP7_intra_topology
#        Tech: asap7, Topology: intra
#        ASAP7 intra-topology dataset ...
#    [1] ASAP7_topology_agnostic
#        ...
```

### `load_dataset_by_config(config_id, data_type='cell')`

Loads dataset based on configuration ID and data type.

**Parameters:**
- `config_id` (int): Dataset configuration ID (0-3)
- `data_type` (str, optional): Data type - 'cell' or 'transition'. Default: 'cell'

**Returns:**
- `tuple`: (input_tensor, output_tensor)
  - `input_tensor` (torch.Tensor): Shape [total_samples, 61, 9]
  - `output_tensor` (torch.Tensor): Shape [total_samples, 61, 1]

**Raises:**
- `ValueError`: If config_id is invalid
- `FileNotFoundError`: If dataset files cannot be found

**Example:**
```python
from dataset_config import load_dataset_by_config

# Load ASAP7 intra-topology cell data
input_data, output_data = load_dataset_by_config(0, data_type='cell')
print(f"Input shape: {input_data.shape}")   # [N, 61, 9]
print(f"Output shape: {output_data.shape}") # [N, 61, 1]

# Load TSMC topology-agnostic transition data
input_data, output_data = load_dataset_by_config(3, data_type='transition')
```

## Data Format

All datasets follow a consistent format:

### Input Tensor Shape: `[num_tasks, 61, 9]`

**9 input features:**
- Index 0: a_param
- Index 1: b_param
- Index 2: c_param
- Index 3: temperature
- Index 4: voltage
- Index 5: additional_dim
- Index 6: delay_indicator
- Index 7: index_1_val (slew)
- Index 8: index_2_val (load_cap)

### Output Tensor Shape: `[num_tasks, 61, 1]`

**1 output value:**
- Delay value for each data point

## Usage in Unified Scripts

Both `MAML_unified_pretraining.py` and `MLP_unified_pretraining.py` use this module:

```python
from dataset_config import get_dataset_config, print_available_datasets, load_dataset_by_config

# Get configuration
config = get_dataset_config(args.dataset_config)
tech = config['tech']
topology_type = config['topology_type']

# Display available options
print_available_datasets()

# Load dataset
input_data, output_data = load_dataset_by_config(args.dataset_config, args.data_type)
```

## Adding New Configurations

To add a new dataset configuration:

1. Add entry to `DATASET_CONFIGS` dictionary:
```python
DATASET_CONFIGS = {
    # ... existing configs ...
    4: {
        'name': 'NEW_CONFIG_NAME',
        'tech': 'new_tech',
        'topology_type': 'new_type',
        'description': 'Description of new config'
    }
}
```

2. Add loading logic in `load_dataset_by_config()`:
```python
elif config_id == 4:
    # Load new dataset
    data_dir = "/path/to/new/dataset"
    test_data_input = torch.load(f"{data_dir}/input_file.pth")
    test_data_output_1 = torch.load(f"{data_dir}/output_file.pth")
```

3. Update unified scripts to accept new config ID in argparse choices.

## Benefits

1. **Eliminates Code Duplication**: Single source of truth for dataset loading
2. **Consistency**: All scripts use identical loading procedures
3. **Maintainability**: Changes to dataset paths only need to be made in one place
4. **Scalability**: Easy to add new dataset configurations
5. **Type Safety**: Proper error handling and validation
6. **Documentation**: Centralized documentation of all datasets

## Integration with Other Modules

### With `maml_utils.py`
```python
from dataset_config import load_dataset_by_config
from maml_utils import normalize_input_features, normalize_and_filter_tasks

# Load dataset
input_data, output_data = load_dataset_by_config(config_id, data_type)

# Normalize using maml_utils
input_data = normalize_input_features(input_data, [7, 8, 3, 4], ['slew', 'load_cap', 'temperature', 'voltage'])
input_data, output_data = normalize_and_filter_tasks(input_data, output_data, min_std_threshold=1e-6)
```

### With `mlp_utils.py`
```python
from dataset_config import load_dataset_by_config
from mlp_utils import normalize_features, normalize_outputs

# Load dataset
input_data, output_data = load_dataset_by_config(config_id, data_type)

# Normalize using mlp_utils
input_data, means, stds = normalize_features(input_data, [7, 8, 3, 4], num_features=9)
output_data, out_means, out_stds, valid_idx = normalize_outputs(output_data, min_std_threshold=1e-6)
```

## Error Handling

The module provides clear error messages:

```python
# Invalid config ID
>>> get_dataset_config(5)
ValueError: Invalid dataset config ID: 5. Must be 0-3.

# File not found
>>> load_dataset_by_config(0, 'invalid_type')
FileNotFoundError: [Errno 2] No such file or directory: '.../merged_invbuf_input_invalid_type.pth'
```

## Module Dependencies

- `torch`: For loading .pth files

## File Location

```
/home/tkdgn2907/Deepsets_test/MAML/Projects/pretraining/model_pretraining_code/dataset_config.py
```
