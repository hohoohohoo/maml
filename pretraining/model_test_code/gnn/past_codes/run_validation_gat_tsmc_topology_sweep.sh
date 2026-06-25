#!/bin/bash
# GAT Topology Validation Sweep for TSMC Dataset
# Runs validation across experiment types (intra_topology/topology_agnostic) and architectures
# Uses Graph Attention Network with multi-head attention

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/validation_gat_topology_sweep_config.json"
    echo "  $0 json_configs/validation_gat_topology_sweep_config.json --dry-run"
    echo ""
    echo "Note: Requires TSMC GNN unified dataset to be pre-generated."
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
echo "GAT Topology Validation - TSMC Dataset Sweep"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo ""

# Parse JSON config using Python
EXPERIMENT_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['experiment_name'])")
MODEL_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['model_type'])")
DATA_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['data_type'])")
GRAPH_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['graph_mode'])")
VOLTAGE_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('voltage_mode', 'all_nodes'))")
NORMALIZATION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('normalization', 'zscore'))")
CACHE_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('cache_path', ''))")
MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['mode'])")
TOTAL_POINTS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['total_points'])")
NUM_TEST_SAMPLES=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['num_test_samples'])")
NUM_ITERATIONS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['num_iterations'])")
SAVE_RESULTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--save_results' if c.get('save_results', False) else '')")
FILTER_CONTINUOUS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--filter_continuous' if c.get('filter_continuous', False) else '')")
CONTINUITY_THRESHOLD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('continuity_threshold', 0.18))")
POOLING=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('pooling', 'mean'))")
HEADS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('heads', 4))")
ADAPTATION_METHOD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('adaptation_method', 'selective_adam'))")
DATASET_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('dataset_dir', ''))")
OUTPUT_PREFIX=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('output_prefix', 'TSMC_GAT'))")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('output_dir', 'final'))")
GPU=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['gpu'])")
INPUTPORT=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--inputport' if c.get('inputport', False) else '')")
RELATED_PIN_ONLY=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--related_pin_only' if c.get('related_pin_only', False) else '')")

# MAML-specific params (only used if model_type is maml)
INNERDIV=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('innerdiv', 10))")
TASKS_PER_META_BATCH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('tasks_per_meta_batch', 16))")
INNER_STEPS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('inner_steps', 1))")

# Get sweep parameters
EXPERIMENTS=$(python3 -c "import json; print(' '.join(json.load(open('$CONFIG_FILE'))['sweep_params']['experiment']))")
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
    s['experiment'],
    s['conv_hidden_dim'], s['num_conv_layers'],
    s['fc_hidden_dim'], s['num_fc_layers']
))
print(len(combos))
")

echo "Experiment: $EXPERIMENT_NAME"
echo "Model type: $MODEL_TYPE"
echo "Model: GAT (Graph Attention Network)"
echo "Data type: $DATA_TYPE"
echo "Graph mode: $GRAPH_MODE"
echo "Voltage mode: $VOLTAGE_MODE"
echo "Normalization: $NORMALIZATION"
if [ -n "$CACHE_PATH" ]; then
    echo "Cache path: $CACHE_PATH"
fi
echo "Mode: $MODE"
echo "Pooling: $POOLING"
echo "Attention heads: $HEADS"
echo "Adaptation method: $ADAPTATION_METHOD"
if [ -n "$DATASET_DIR" ]; then
    echo "Dataset dir: $DATASET_DIR"
fi
echo "Output prefix: $OUTPUT_PREFIX"
echo "Output dir: $OUTPUT_DIR"
echo "GPU: $GPU"
echo ""
echo "Sweep parameters:"
echo "  Experiments: $EXPERIMENTS"
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
heads = c['base_config'].get('heads', 4)

combos = list(itertools.product(
    s['experiment'],
    s['conv_hidden_dim'], s['num_conv_layers'],
    s['fc_hidden_dim'], s['num_fc_layers']
))

for i, (exp, conv_dim, conv_layers, fc_dim, fc_layers) in enumerate(combos, 1):
    print(f'  {i}. {exp} - conv{conv_dim}x{conv_layers}_fc{fc_dim}x{fc_layers}_heads{heads}')
"
    echo ""
    echo "GAT-specific options:"
    echo "  - heads: Number of attention heads (default: 4)"
    echo "  - num_fc_layers: GAT supports only 2 or 3 FC layers"
    echo ""
    echo "Supported pooling options:"
    echo "  - mean: Global mean pooling (default)"
    echo "  - max: Global max pooling"
    echo "  - add: Global sum pooling"
    echo "  - output: Output-node-only pooling"
    echo ""
    echo "Supported adaptation methods:"
    echo "  - selective_adam: Grad/Move scaling + conditional Adam (default)"
    echo "  - adam: Direct Adam optimization (no grad/move)"
    echo ""
    echo "Dry run complete. Use without --dry-run to execute."
    exit 0
fi

# Run validation sweep
echo "Starting GAT topology validation sweep..."
echo "=========================================="
echo ""

START_TIME=$(date +%s)

# Run each combination
COMBO_NUM=0
python3 -c "
import json
import itertools

c = json.load(open('$CONFIG_FILE'))
s = c['sweep_params']

combos = list(itertools.product(
    s['experiment'],
    s['conv_hidden_dim'], s['num_conv_layers'],
    s['fc_hidden_dim'], s['num_fc_layers']
))

for exp, conv_dim, conv_layers, fc_dim, fc_layers in combos:
    print(f'{exp} {conv_dim} {conv_layers} {fc_dim} {fc_layers}')
" | while read EXP CONV_DIM CONV_LAYERS FC_DIM FC_LAYERS; do

    COMBO_NUM=$((COMBO_NUM + 1))

    echo ""
    echo "=========================================="
    echo "[$COMBO_NUM/$TOTAL_COMBINATIONS] ${EXP} - conv${CONV_DIM}x${CONV_LAYERS}_fc${FC_DIM}x${FC_LAYERS}_heads${HEADS}"
    echo "=========================================="

    # Build command - use GAT validation script
    CMD="python -u TSMC_GAT_topology_validation.py"
    CMD="$CMD --experiment $EXP"
    CMD="$CMD --model_type $MODEL_TYPE"
    CMD="$CMD --data_type $DATA_TYPE"
    CMD="$CMD --graph_mode $GRAPH_MODE"
    CMD="$CMD --voltage_mode $VOLTAGE_MODE"
    CMD="$CMD --normalization $NORMALIZATION"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --total_points $TOTAL_POINTS"
    CMD="$CMD --num_test_samples $NUM_TEST_SAMPLES"
    CMD="$CMD --num_iterations $NUM_ITERATIONS"
    CMD="$CMD --conv_hidden_dim $CONV_DIM"
    CMD="$CMD --num_conv_layers $CONV_LAYERS"
    CMD="$CMD --fc_hidden_dim $FC_DIM"
    CMD="$CMD --num_fc_layers $FC_LAYERS"
    CMD="$CMD --heads $HEADS"
    CMD="$CMD --pooling $POOLING"
    CMD="$CMD --adaptation_method $ADAPTATION_METHOD"
    CMD="$CMD --output_prefix $OUTPUT_PREFIX"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --gpu $GPU"

    # Add cache_path if specified
    if [ -n "$CACHE_PATH" ]; then
        CMD="$CMD --cache_path $CACHE_PATH"
    fi

    # Add dataset_dir if specified
    if [ -n "$DATASET_DIR" ]; then
        CMD="$CMD --dataset_dir $DATASET_DIR"
    fi

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

    # Add inputport flag if set
    if [ -n "$INPUTPORT" ]; then
        CMD="$CMD $INPUTPORT"
    fi

    # Add related_pin_only flag if set
    if [ -n "$RELATED_PIN_ONLY" ]; then
        CMD="$CMD $RELATED_PIN_ONLY"
    fi

    echo "Command: $CMD"
    echo ""

    # Execute with real-time output using tee
    COMBO_START=$(date +%s)
    TEMP_OUTPUT=$(mktemp)
    eval $CMD 2>&1 | tee "$TEMP_OUTPUT"
    EXIT_CODE=${PIPESTATUS[0]}
    COMBO_END=$(date +%s)
    COMBO_DURATION=$((COMBO_END - COMBO_START))

    echo ""
    echo "Duration: ${COMBO_DURATION}s"

    rm -f "$TEMP_OUTPUT"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: Combination failed with exit code $EXIT_CODE"
    fi

done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
echo "GAT Topology Validation Sweep Complete"
echo "=========================================="
echo "Total time: ${DURATION}s ($((DURATION / 60))m)"
echo ""

# Clean up PyTorch Geometric JIT cache files
echo "Cleaning up JIT cache files..."
rm -f "$SCRIPT_DIR"/torch_geometric.*.py 2>/dev/null
echo "Done."

exit 0
