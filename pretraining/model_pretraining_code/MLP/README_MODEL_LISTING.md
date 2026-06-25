# Pretrained Model Listing Tool

A comprehensive tool for finding and filtering pretrained MAML model files based on their filename parameters.

## Features

- 🔍 **Parameter-based filtering**: Filter models by any combination of parameters
- 📊 **Statistical analysis**: View model distribution by various parameters
- 📝 **Multiple output formats**: Paths only, table view, or CSV export
- 🎯 **New format support**: Full support for the new 519traintask model format
- 🔄 **Automatic parsing**: Extracts parameters from multiple filename formats

## Model Filename Format

The new format:
```
{data_type}_innerdiv{innerdiv}_meta{meta}_{model_suffix}_519traintask_full1DMAML_weights_3hidden_({layer_length})_{iterations}_inner{inner}_upgraded{tech_suffix}.pth
```

Parameters:
- **data_type**: `cell` or `transition`
- **innerdiv**: Inner divisor (e.g., 1, 10, 50, 100)
- **meta**: Meta batch size (e.g., 8, 16, 32, 64)
- **model_suffix**: `topology_agnostic` or `intratopology`
- **layer_length**: Hidden layer size (e.g., 40, 300)
- **iterations**: Training iterations (e.g., 300000)
- **inner**: Inner loop steps (e.g., 1, 5)
- **tech_suffix**: Technology (e.g., `ASAP7`, `TSMC`)

## Quick Start

### Using Shell Script (Recommended)

```bash
# Show help
./list_models.sh --help

# Find all cell models
./list_models.sh --data_type cell --paths-only

# Find specific model configuration
./list_models.sh --data_type cell --innerdiv 100 --meta 32 \
    --model_suffix topology_agnostic --iterations 300000 \
    --inner 1 --tech_suffix TSMC --upgraded --paths-only
```

### Using Python Directly

```bash
# Find models with filters
python utils/list_pretrained_models.py --data_type cell \
    --model_suffix topology_agnostic --upgraded --paths-only

# Show detailed analysis
python utils/list_pretrained_models.py --data_type cell

# Export to CSV
python utils/list_pretrained_models.py --data_type cell \
    --export cell_models.csv
```

## Available Parameters

### Model Parameters (Filters)

| Parameter | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `--data_type` | choice | Data type | `cell`, `transition` |
| `--innerdiv` | int | Inner divisor | `1`, `10`, `50`, `100` |
| `--meta` | int | Meta batch size | `8`, `16`, `32`, `64` |
| `--model_suffix` | choice | Model type | `topology_agnostic`, `intratopology` |
| `--layer_length` | int | Hidden layer size | `40`, `300` |
| `--iterations` | int | Training iterations | `300000`, `100000` |
| `--inner` | int | Inner loop steps | `1`, `5` |
| `--tech_suffix` | str | Technology suffix | `ASAP7`, `TSMC` |
| `--upgraded` | flag | Filter only upgraded models | - |

### Output Options

| Parameter | Description |
|-----------|-------------|
| `--paths-only` | Print only model paths (no analysis) |
| `--limit NUM` | Limit number of models to display |
| `--export FILE` | Export results to CSV file |
| `--verbose` | Print detailed parsing information |
| `--no-analysis` | Skip statistical analysis section |

## Usage Examples

### Example 1: Find Cell Models with Specific Parameters

```bash
./list_models.sh --data_type cell --innerdiv 1 --meta 8 --paths-only
```

Output:
```
📂 Scanning directory: /path/to/pretrained_models/taskdivide_all
📊 Found 606 .pth files

✅ Successfully parsed: 560 models
⚠️ Could not parse: 46 files

🔍 Filtered to 12 models

📋 Found 12 model(s) matching conditions:

/path/to/pretrained_models/taskdivide_all/cell_innerdiv1_meta8_topology_agnostic_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded.pth
...
```

### Example 2: Find Topology Agnostic Models with TSMC Tech

```bash
./list_models.sh --model_suffix topology_agnostic --tech_suffix TSMC --upgraded --paths-only
```

### Example 3: Full Parameter Search

```bash
./list_models.sh --data_type cell --innerdiv 100 --meta 32 \
    --model_suffix topology_agnostic --layer_length 40 \
    --iterations 300000 --inner 1 --tech_suffix TSMC \
    --upgraded --paths-only
```

Output:
```
🔍 Filtered to 1 models

📋 Found 1 model(s) matching conditions:

/path/to/cell_innerdiv100_meta32_topology_agnostic_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth
```

### Example 4: Statistical Analysis

```bash
./list_models.sh --data_type cell --model_suffix topology_agnostic --upgraded
```

Output includes:
```
================================================================================
📊 Model Analysis
================================================================================

📦 By Data Type:
  cell: 15 models

🔧 By Format:
  new_519traintask: 15 models

🔄 By Iterations:
  300000: 15 models

🔁 By Inner Steps:
  inner=1: 15 models

📐 By Inner Divisor:
  innerdiv=50: 3 models
  innerdiv=100: 12 models

🎯 By Meta Batch Size:
  meta=16: 3 models
  meta=32: 9 models
  meta=64: 3 models

⬆️ Upgraded Models: 15
```

### Example 5: Export to CSV

```bash
./list_models.sh --data_type cell --export cell_models.csv
```

CSV columns:
- `filename`
- `data_type`
- `iterations`
- `inner`
- `innerdiv`
- `meta`
- `model_suffix`
- `layer_length`
- `upgraded`
- `tech`
- `format`

### Example 6: Find Models by Iteration Count

```bash
./list_models.sh --iterations 300000 --data_type cell --paths-only --limit 10
```

## Common Use Cases

### Use Case 1: Find the Right Model for Validation

```bash
# You need: cell data, topology_agnostic, TSMC tech, 300k iterations
./list_models.sh --data_type cell \
    --model_suffix topology_agnostic \
    --tech_suffix TSMC \
    --iterations 300000 \
    --upgraded --paths-only
```

### Use Case 2: Compare Different Meta Batch Sizes

```bash
# Find all models with innerdiv=100, varying meta
./list_models.sh --data_type cell --innerdiv 100 \
    --model_suffix topology_agnostic --upgraded

# Then check the "By Meta Batch Size" section in the analysis
```

### Use Case 3: Export Model Inventory

```bash
# Export all upgraded topology_agnostic models
./list_models.sh --model_suffix topology_agnostic \
    --upgraded --export topology_agnostic_models.csv

# Import in pandas for analysis
python -c "import pandas as pd; df = pd.read_csv('topology_agnostic_models.csv'); print(df.groupby('meta').size())"
```

## Supported Filename Formats

The tool supports multiple filename formats:

1. **New format (519traintask)**:
   ```
   cell_innerdiv100_meta32_topology_agnostic_519traintask_full1DMAML_weights_3hidden_(40)_300000_inner1_upgraded_tsmc.pth
   ```

2. **Legacy format with meta**:
   ```
   cell_innerdiv10_meta64_full1DMAML_weights_3hidden_(40)_30000_TSMC_FF_0_test5(dim5)_inner1.pth
   ```

3. **Simple format**:
   ```
   cell_innerdiv10_full1DMAML_weights_3hidden_(40)_100000_SLVT_TT_test5(dim5)_inner1_fixed.pth
   ```

4. **Legacy format**:
   ```
   cell_full1DMAML_weights_3hidden_(40)_50000_ASAP7_TT_test5(dim5)_inner1_fixed.pth
   ```

## Tips

1. **Use `--paths-only` for scripting**: Get clean output for use in other scripts
   ```bash
   MODEL_PATH=$(./list_models.sh --data_type cell --innerdiv 1 --meta 8 --paths-only | tail -1)
   ```

2. **Combine with grep for further filtering**:
   ```bash
   ./list_models.sh --data_type cell --paths-only | grep "merged"
   ```

3. **Check available models before validation**:
   ```bash
   ./list_models.sh --data_type cell --model_suffix topology_agnostic --upgraded
   ```

4. **Export and analyze in spreadsheet**:
   ```bash
   ./list_models.sh --export all_models.csv
   # Open all_models.csv in Excel/Google Sheets
   ```

## Troubleshooting

### No models found

```
📋 Found 0 model(s) matching conditions:
No models found matching the given conditions
```

**Solutions:**
- Check if the filters are too restrictive
- Try removing some filters one by one
- Use `--verbose` to see which files couldn't be parsed
- Verify the `pretrained_models/taskdivide_all` directory exists

### Could not parse files

```
⚠️ Could not parse: 46 files
```

This is normal - some files may not follow the expected naming convention. The tool will skip these files and process the ones that match.

### Models directory not found

```
❌ Error: Directory not found: /path/to/pretrained_models/taskdivide_all
```

**Solution:**
Specify custom directory:
```bash
./list_models.sh --models_dir /custom/path/to/models --data_type cell
```

## Integration with Validation Scripts

Use the tool to find model paths for validation:

```bash
# Find the model path
MODEL_PATH=$(./list_models.sh --data_type cell --innerdiv 1 --meta 8 \
    --model_suffix topology_agnostic --iterations 300000 --inner 1 \
    --tech_suffix ASAP7 --upgraded --paths-only | grep -v "Found" | grep "\.pth$")

# Use in validation
echo "Using model: $MODEL_PATH"
python validation_script.py --model_path "$MODEL_PATH"
```

## Related Files

- `utils/list_pretrained_models.py` - Main Python implementation
- `list_models.sh` - Convenient shell wrapper
- `../pretrained_models/taskdivide_all/` - Default models directory

## Summary

This tool provides a powerful way to:
- ✅ Find models by any combination of parameters
- ✅ Analyze model distribution and availability
- ✅ Export model inventory to CSV
- ✅ Integrate with validation and testing scripts
- ✅ Support both new and legacy filename formats

Use the shell script for quick searches and the Python script for advanced analysis.
