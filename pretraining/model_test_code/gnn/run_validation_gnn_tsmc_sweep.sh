#!/bin/bash
# GNN Validation Sweep for TSMC Dataset
# Runs validation across corner-temperature combinations and architectures

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/validation_gnn_tsmc_sweep_config.json"
    echo "  $0 json_configs/validation_gnn_tsmc_sweep_config.json --dry-run"
    echo ""
    echo "Note: Requires unified test dataset to be pre-generated."
    echo "      Run split_gnn_dataset_tsmc.py --corner <C> --temperature <T> first."
    exit 1
fi

CONFIG_FILE="$1"
DRY_RUN=false

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "GNN Validation - TSMC Dataset Sweep"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo ""

# Parse JSON config using Python
EXPERIMENT_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['experiment_name'])")
MODEL_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['model_type'])")
DATA_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['data_type'])")
GRAPH_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['graph_mode'])")
MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['mode'])")
TOTAL_POINTS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['total_points'])")
NUM_TEST_SAMPLES=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['num_test_samples'])")
NUM_ITERATIONS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['num_iterations'])")
SAVE_RESULTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--save_results' if c.get('save_results', False) else '')")
FILTER_CONTINUOUS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--filter_continuous' if c.get('filter_continuous', False) else '')")
CONTINUITY_THRESHOLD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('continuity_threshold', 0.18))")
GPU=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['gpu'])")

# MAML-specific params (only used if model_type is maml)
INNERDIV=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('innerdiv', 10))")
TASKS_PER_META_BATCH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('tasks_per_meta_batch', 16))")
INNER_STEPS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('inner_steps', 1))")

# Get sweep parameters (TSMC: corner and temperature instead of process and corner)
CORNERS=$(python3 -c "import json; print(' '.join(json.load(open('$CONFIG_FILE'))['sweep_params']['corner']))")
TEMPERATURES=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$CONFIG_FILE'))['sweep_params']['temperature'])))")
CONV_HIDDEN_DIMS=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$CONFIG_FILE'))['sweep_params']['conv_hidden_dim'])))")
NUM_CONV_LAYERS=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$CONFIG_FILE'))['sweep_params']['num_conv_layers'])))")
FC_HIDDEN_DIMS=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$CONFIG_FILE'))['sweep_params']['fc_hidden_dim'])))")
NUM_FC_LAYERS=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$CONFIG_FILE'))['sweep_params']['num_fc_layers'])))")

# Calculate total combinations
TOTAL_COMBINATIONS=$(python3 -c "
import json
import itertools
c = json.load(open('$CONFIG_FILE'))
s = c['sweep_params']
combos = list(itertools.product(
    s['corner'], s['temperature'],
    s['conv_hidden_dim'], s['num_conv_layers'],
    s['fc_hidden_dim'], s['num_fc_layers']
))
print(len(combos))
")

echo "Experiment: $EXPERIMENT_NAME"
echo "Model type: $MODEL_TYPE"
echo "Data type: $DATA_TYPE"
echo "Graph mode: $GRAPH_MODE"
echo "Mode: $MODE"
echo "GPU: $GPU"
echo ""
echo "Sweep parameters:"
echo "  Corners: $CORNERS"
echo "  Temperatures: $TEMPERATURES"
echo "  Conv hidden dims: $CONV_HIDDEN_DIMS"
echo "  Num conv layers: $NUM_CONV_LAYERS"
echo "  FC hidden dims: $FC_HIDDEN_DIMS"
echo "  Num FC layers: $NUM_FC_LAYERS"
echo ""
echo "Total combinations: $TOTAL_COMBINATIONS"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE"
    echo ""
    echo "Would run the following combinations:"

    python3 -c "
import json
import itertools

c = json.load(open('$CONFIG_FILE'))
s = c['sweep_params']

combos = list(itertools.product(
    s['corner'], s['temperature'],
    s['conv_hidden_dim'], s['num_conv_layers'],
    s['fc_hidden_dim'], s['num_fc_layers']
))

for i, (corner, temp, conv_dim, conv_layers, fc_dim, fc_layers) in enumerate(combos, 1):
    print(f'  {i}. TSMC_{corner}_{temp} - conv{conv_dim}x{conv_layers}_fc{fc_dim}x{fc_layers}')
"
    echo ""
    echo "Dry run complete. Use without --dry-run to execute."
    exit 0
fi

# Run validation sweep
echo "Starting validation sweep..."
echo "=========================================="
echo ""

START_TIME=$(date +%s)

# Run each combination
python3 -c "
import json
import itertools

c = json.load(open('$CONFIG_FILE'))
s = c['sweep_params']

combos = list(itertools.product(
    s['corner'], s['temperature'],
    s['conv_hidden_dim'], s['num_conv_layers'],
    s['fc_hidden_dim'], s['num_fc_layers']
))

for corner, temp, conv_dim, conv_layers, fc_dim, fc_layers in combos:
    print(f'{corner} {temp} {conv_dim} {conv_layers} {fc_dim} {fc_layers}')
" | while read CORNER TEMP CONV_DIM CONV_LAYERS FC_DIM FC_LAYERS; do

    echo ""
    echo "=========================================="
    echo "Running: TSMC_${CORNER}_${TEMP} - conv${CONV_DIM}x${CONV_LAYERS}_fc${FC_DIM}x${FC_LAYERS}"
    echo "=========================================="

    # Build command
    CMD="python -u TSMC_GCN_voltage_validation.py"
    CMD="$CMD --corner $CORNER --temperature $TEMP"
    CMD="$CMD --model_type $MODEL_TYPE"
    CMD="$CMD --data_type $DATA_TYPE"
    CMD="$CMD --graph_mode $GRAPH_MODE"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --total_points $TOTAL_POINTS"
    CMD="$CMD --num_test_samples $NUM_TEST_SAMPLES"
    CMD="$CMD --num_iterations $NUM_ITERATIONS"
    CMD="$CMD --conv_hidden_dim $CONV_DIM"
    CMD="$CMD --num_conv_layers $CONV_LAYERS"
    CMD="$CMD --fc_hidden_dim $FC_DIM"
    CMD="$CMD --num_fc_layers $FC_LAYERS"
    CMD="$CMD --gpu $GPU"

    # Add MAML-specific params if needed
    if [ "$MODEL_TYPE" = "maml" ]; then
        CMD="$CMD --innerdiv $INNERDIV"
        CMD="$CMD --tasks_per_meta_batch $TASKS_PER_META_BATCH"
        CMD="$CMD --inner_steps $INNER_STEPS"
    fi

    # Add save_results flag if set
    if [ -n "$SAVE_RESULTS" ]; then
        CMD="$CMD $SAVE_RESULTS"
    fi

    # Add filter_continuous flag if set
    if [ -n "$FILTER_CONTINUOUS" ]; then
        CMD="$CMD $FILTER_CONTINUOUS"
        CMD="$CMD --continuity_threshold $CONTINUITY_THRESHOLD"
    fi

    echo "Command: $CMD"
    echo ""

    # Execute
    eval $CMD

done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "Validation Sweep Complete"
echo "=========================================="
echo "Total time: ${DURATION}s ($((DURATION / 60))m)"
echo ""

# Clean up PyTorch Geometric JIT cache files
echo "Cleaning up JIT cache files..."
rm -f "$SCRIPT_DIR"/torch_geometric.*.py 2>/dev/null
echo "Done."

exit 0
