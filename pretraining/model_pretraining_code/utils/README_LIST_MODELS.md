# Pretrained Models Listing Utility

This utility scans and analyzes pretrained model weights in the `taskdivide_all` directory, extracting parameters from filenames and providing structured overview.

## Features

- 🔍 Parse multiple filename formats (legacy and new 519traintask format)
- 📊 Statistical analysis of available models
- 🎯 Advanced filtering by multiple parameters
- 💾 CSV export for further analysis
- 📋 Formatted table output

## Quick Start

```bash
# List all models with analysis
./list_models.sh

# List with limit (show only first 20)
./list_models.sh --limit 20

# Filter by specific criteria
./list_models.sh --data_type cell --iterations 300000 --upgraded
```

## Supported Filename Formats

The utility automatically parses multiple filename formats:

### New Format (519traintask)
```
{data_type}_innerdiv{innerdiv}_meta{meta}_{model_suffix}_519traintask_full1DMAML_weights_3hidden_({layer_length})_{iterations}_inner{inner}_upgraded{tech_suffix}.pth
```
Example:
```
cell_innerdiv100_meta32_topology_agnostic_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth
```

### Legacy Formats
```
{data_type}_innerdiv{innerdiv}_meta{meta}_full1DMAML_weights_3hidden_({layer_length})_{iterations}_{tech}_{corner}_{voltage}_test5(dim5)_inner{inner}.pth
{data_type}_innerdiv{innerdiv}_full1DMAML_weights_3hidden_({layer_length})_{iterations}_{tech}_{corner}_test5(dim5)_inner{inner}_fixed.pth
{data_type}_full1DMAML_weights_3hidden_({layer_length})_{iterations}_{tech}_{corner}_test5(dim5)_inner{inner}.pth
```

## Usage

### Basic Usage

```bash
# List all models
python utils/list_pretrained_models.py

# Or use wrapper script
./list_models.sh
```

### Filtering Options

```bash
# Filter by data type
./list_models.sh --data_type cell
./list_models.sh --data_type transition

# Filter by iterations
./list_models.sh --iterations 300000

# Filter by inner steps
./list_models.sh --inner 1

# Filter by inner divisor
./list_models.sh --innerdiv 100

# Filter by meta batch size
./list_models.sh --meta 32

# Filter by model type
./list_models.sh --model_suffix topology_agnostic
./list_models.sh --model_suffix intratopology

# Filter only upgraded models
./list_models.sh --upgraded

# Combine multiple filters
./list_models.sh --data_type cell --iterations 300000 --upgraded --innerdiv 100
```

### Output Control

```bash
# Limit number of displayed models
./list_models.sh --limit 20

# Skip analysis section
./list_models.sh --no-analysis

# Verbose mode (show parsing details)
./list_models.sh --verbose
```

### CSV Export

```bash
# Export to CSV
./list_models.sh --export models.csv

# Export filtered results
./list_models.sh --data_type cell --upgraded --export cell_upgraded_models.csv
```

## Extracted Parameters

The utility extracts the following parameters from filenames:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data_type` | Type of data (cell/transition) | `cell`, `transition` |
| `innerdiv` | Inner learning rate divisor | `10`, `100` |
| `meta` | Meta batch size (tasks per batch) | `32`, `64` |
| `model_suffix` | Model type/topology | `topology_agnostic`, `intratopology` |
| `layer_length` | Hidden layer size | `40` |
| `iterations` | Number of training iterations | `300000` |
| `inner` | Inner loop steps | `1`, `5` |
| `upgraded` | Whether model is upgraded | `True`, `False` |
| `tech` | Technology node | `TSMC`, `ASAP7` |
| `corner` | Process corner (legacy) | `FF`, `SS`, `TT` |
| `voltage` | Voltage level (legacy) | `0`, `25`, `50`, `75`, `100` |
| `format` | Filename format type | `new_519traintask`, `legacy` |

## Examples

### Example 1: Find All Upgraded Cell Models

```bash
./list_models.sh --data_type cell --upgraded
```

Output:
```
📂 Scanning directory: ../../../pretrained_models/taskdivide_all
📊 Found 605 .pth files

✅ Successfully parsed: 559 models

🔍 Filtered to 21 models

📊 Model Analysis
==================
📦 By Data Type:
  cell: 21 models
...
```

### Example 2: Find 300K Iteration Topology Agnostic Models

```bash
./list_models.sh --iterations 300000 --model_suffix topology_agnostic
```

### Example 3: Export All Models to CSV

```bash
./list_models.sh --export all_models.csv --no-analysis
```

Then analyze in Python/Excel:
```python
import pandas as pd
df = pd.read_csv('all_models.csv')
print(df.groupby(['data_type', 'iterations']).size())
```

### Example 4: Find Models with Specific Configuration

```bash
# Find all cell models with innerdiv=100, meta=32, upgraded
./list_models.sh \
    --data_type cell \
    --innerdiv 100 \
    --meta 32 \
    --upgraded \
    --limit 10
```

### Example 5: Quick Overview of New Format Models

```bash
./list_models.sh | grep "new_519traintask"
```

## Analysis Output

The utility provides statistical analysis:

- **By Data Type**: Distribution of cell vs transition models
- **By Format**: Legacy vs new 519traintask format
- **By Iterations**: Number of models per iteration count
- **By Inner Steps**: Distribution of inner loop configurations
- **By Inner Divisor**: Distribution of learning rate divisors
- **By Meta Batch Size**: Distribution of meta batch sizes
- **By Model Type**: Topology agnostic vs intra-topology
- **Upgraded Count**: Number of upgraded models
- **By Technology**: Distribution across technology nodes

## Table Output Format

```
============================================================================================================================================
Filename                                                                         Type   Iter    Inner  InDiv  Meta  Upg
============================================================================================================================================
cell_innerdiv100_meta32_topology_agnostic_519traintask_full1DMAML_weight...      cell   300000  1      100    32    Yes
transition_innerdiv10_meta64_topology_agnostic_519traintask_full1DMAML_w...      transition 300000  1      10     64    Yes
...
```

## CSV Output Format

The CSV file contains all extracted parameters in structured format:

```csv
filename,data_type,iterations,inner,innerdiv,meta,model_suffix,layer_length,upgraded,tech,corner,voltage,format
cell_innerdiv100_meta32_topology_agnostic_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded.pth,cell,300000,1,100.0,32.0,topology_agnostic,40,True,,,,new_519traintask
...
```

## Advanced Usage

### Find Models for Specific Configuration

To find the exact model file for a specific configuration:

```bash
# Find MAML model for config 0 (ASAP7 intra-topology), 300k iterations, innerdiv=100
./list_models.sh \
    --data_type cell \
    --iterations 300000 \
    --innerdiv 100 \
    --model_suffix intratopology \
    --upgraded \
    --no-analysis
```

### Generate Training Configuration Report

```bash
# Export all 300k iteration models
./list_models.sh --iterations 300000 --export training_300k.csv

# Then use pandas/Excel to analyze:
# - Which configurations have been trained
# - Which configurations are missing
# - Distribution of hyperparameters
```

### Find Compatible Models for Validation

When running validation, you need models with specific parameters. Use this utility to find them:

```bash
# For validation with inner=1, innerdiv=100
./list_models.sh \
    --inner 1 \
    --innerdiv 100 \
    --iterations 300000 \
    --no-analysis
```

## Command-Line Reference

```
usage: list_pretrained_models.py [-h] [--models_dir MODELS_DIR]
                                  [--data_type {cell,transition}]
                                  [--iterations ITERATIONS] [--inner INNER]
                                  [--innerdiv INNERDIV] [--meta META]
                                  [--model_suffix {topology_agnostic,intratopology}]
                                  [--upgraded] [--export FILE] [--limit LIMIT]
                                  [--verbose] [--no-analysis]

Options:
  -h, --help            show this help message and exit
  --models_dir MODELS_DIR
                        Path to pretrained models directory
  --data_type {cell,transition}
                        Filter by data type
  --iterations ITERATIONS
                        Filter by iteration count
  --inner INNER         Filter by inner steps
  --innerdiv INNERDIV   Filter by inner divisor
  --meta META           Filter by meta batch size
  --model_suffix {topology_agnostic,intratopology}
                        Filter by model type
  --upgraded            Filter only upgraded models
  --export FILE         Export to CSV file
  --limit LIMIT         Limit number of models to display
  --verbose             Print detailed parsing information
  --no-analysis         Skip analysis section
```

## Troubleshooting

### No models found

```bash
# Check if directory exists
ls ../../../pretrained_models/taskdivide_all/

# Specify custom directory
./list_models.sh --models_dir /path/to/pretrained_models
```

### Some files not parsed

The utility will show a warning like:
```
⚠️ Could not parse: 46 files
```

Use `--verbose` to see which files failed:
```bash
./list_models.sh --verbose
```

### CSV export fails

Make sure you have write permissions:
```bash
./list_models.sh --export ~/models.csv
```

## Integration with Other Tools

### Use with grep

```bash
./list_models.sh --no-analysis | grep "300000"
```

### Pipe to file

```bash
./list_models.sh > models_report.txt
```

### Use in scripts

```bash
#!/bin/bash
MODEL_COUNT=$(./list_models.sh --data_type cell --no-analysis 2>&1 | grep "Successfully parsed" | grep -oP '\d+')
echo "Found $MODEL_COUNT cell models"
```

## Related Files

- `../MAML_topology_pretraining.py` - Creates MAML pretrained models
- `../MLP_topology_pretraining.py` - Creates MLP pretrained models
- `../../model_test_code/utils/test_dataset_config.py` - Model path configuration for validation
- `parse_sweep_config.py` - JSON sweep configuration parser
