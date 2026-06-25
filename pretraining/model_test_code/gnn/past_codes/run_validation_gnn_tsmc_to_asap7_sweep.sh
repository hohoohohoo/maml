#!/bin/bash
# Cross-PDK Validation: TSMC Model -> ASAP7 Test Data
# Runs validation using TSMC-trained models on ASAP7 test cells
# Tests cross-PDK generalization capability

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/validation_gnn_tsmc_to_asap7_config.json"
    echo "  $0 json_configs/validation_gnn_tsmc_to_asap7_config.json --dry-run"
    echo ""
    echo "Note: Requires both TSMC model and ASAP7 test dataset."
    echo "      TSMC Model: pretrained_models/gnn_maml_tsmc_process_checkpoints/"
    echo "      ASAP7 Data: dataset_all/dataset_temp_process/"
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
echo "Cross-PDK Validation: TSMC Model -> ASAP7 Test"
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
TEMP_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('temp_mode', 'typical'))")
TSMC_CACHE_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('tsmc_cache_path', ''))")
ASAP7_CACHE_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('asap7_cache_path', ''))")
INPUTPORT=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--inputport' if c.get('inputport', False) else '')")
RELATED_PIN_ONLY=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--related_pin_only' if c.get('related_pin_only', False) else '')")
USE_TARGET_NORM=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--use_target_norm' if c.get('use_target_norm', False) else '')")
PDK_SCALE_FACTOR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('pdk_scale_factor', 1.0))")
VOLTAGE_SHIFT=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('voltage_shift', 0.0))")
MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['mode'])")
TOTAL_POINTS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['total_points'])")
NUM_TEST_SAMPLES=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['num_test_samples'])")
NUM_ITERATIONS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['num_iterations'])")
SAVE_RESULTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--save_results' if c.get('save_results', False) else '')")
FILTER_CONTINUOUS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--filter_continuous' if c.get('filter_continuous', False) else '')")
CONTINUITY_THRESHOLD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('continuity_threshold', 0.18))")
POOLING=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('pooling', 'mean'))")
ADAPTATION_METHOD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('adaptation_method', 'selective_adam'))")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('output_dir', 'final'))")
GPU=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['gpu'])")

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
echo ""
echo "Cross-PDK Configuration:"
echo "  Source: TSMC model"
echo "  Target: ASAP7 test cells"
echo ""
echo "Model type: $MODEL_TYPE"
echo "Data type: $DATA_TYPE"
echo "Graph mode: $GRAPH_MODE"
echo "Voltage mode: $VOLTAGE_MODE"
echo "Normalization: $NORMALIZATION"
echo "Temp mode: $TEMP_MODE"
if [ -n "$TSMC_CACHE_PATH" ]; then
    echo "TSMC cache path: $TSMC_CACHE_PATH"
fi
if [ -n "$ASAP7_CACHE_PATH" ]; then
    echo "ASAP7 cache path: $ASAP7_CACHE_PATH"
fi
if [ -n "$INPUTPORT" ]; then
    echo "Inputport: enabled"
fi
if [ -n "$RELATED_PIN_ONLY" ]; then
    echo "Related pin only: enabled"
fi
if [ -n "$USE_TARGET_NORM" ]; then
    echo "Use target norm: enabled (ASAP7 norm_stats)"
fi
if [ "$PDK_SCALE_FACTOR" != "1.0" ]; then
    echo "PDK scale factor: $PDK_SCALE_FACTOR (ASAP7 ps/ff -> TSMC ns/pf)"
fi
if [ "$VOLTAGE_SHIFT" != "0.0" ] && [ "$VOLTAGE_SHIFT" != "0" ]; then
    echo "Voltage shift: $VOLTAGE_SHIFT V (ASAP7 0.7V -> TSMC 0.9V)"
fi
echo "Mode: $MODE"
echo "Pooling: $POOLING"
echo "Adaptation method: $ADAPTATION_METHOD"
echo "Output dir: $OUTPUT_DIR"
echo "GPU: $GPU"
echo "Node features: 11D (7 base + 4 process params)"
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

combos = list(itertools.product(
    s['experiment'],
    s['conv_hidden_dim'], s['num_conv_layers'],
    s['fc_hidden_dim'], s['num_fc_layers']
))

for i, (exp, conv_dim, conv_layers, fc_dim, fc_layers) in enumerate(combos, 1):
    print(f'  {i}. {exp} - conv{conv_dim}x{conv_layers}_fc{fc_dim}x{fc_layers}')
"
    echo ""
    echo "Cross-PDK validation:"
    echo "  - TSMC-trained model will be loaded"
    echo "  - Normalization stats from TSMC model checkpoint will be used"
    echo "  - ASAP7 test cells will be evaluated"
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
echo "Starting TSMC to ASAP7 cross-PDK validation sweep..."
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
    echo "[$COMBO_NUM/$TOTAL_COMBINATIONS] ${EXP} - conv${CONV_DIM}x${CONV_LAYERS}_fc${FC_DIM}x${FC_LAYERS}"
    echo "=========================================="

    # Build command - use TSMC to ASAP7 cross-PDK version
    CMD="python -u TSMC_to_ASAP7_GCN_validation.py"
    CMD="$CMD --experiment $EXP"
    CMD="$CMD --model_type $MODEL_TYPE"
    CMD="$CMD --data_type $DATA_TYPE"
    CMD="$CMD --graph_mode $GRAPH_MODE"
    CMD="$CMD --voltage_mode $VOLTAGE_MODE"
    CMD="$CMD --normalization $NORMALIZATION"
    CMD="$CMD --temp_mode $TEMP_MODE"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --total_points $TOTAL_POINTS"
    CMD="$CMD --num_test_samples $NUM_TEST_SAMPLES"
    CMD="$CMD --num_iterations $NUM_ITERATIONS"
    CMD="$CMD --conv_hidden_dim $CONV_DIM"
    CMD="$CMD --num_conv_layers $CONV_LAYERS"
    CMD="$CMD --fc_hidden_dim $FC_DIM"
    CMD="$CMD --num_fc_layers $FC_LAYERS"
    CMD="$CMD --pooling $POOLING"
    CMD="$CMD --adaptation_method $ADAPTATION_METHOD"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --gpu $GPU"

    # Add TSMC cache_path if specified
    if [ -n "$TSMC_CACHE_PATH" ]; then
        CMD="$CMD --tsmc_cache_path $TSMC_CACHE_PATH"
    fi

    # Add ASAP7 cache_path if specified
    if [ -n "$ASAP7_CACHE_PATH" ]; then
        CMD="$CMD --asap7_cache_path $ASAP7_CACHE_PATH"
    fi

    # Add inputport flag if set
    if [ -n "$INPUTPORT" ]; then
        CMD="$CMD $INPUTPORT"
    fi

    # Add related_pin_only flag if set
    if [ -n "$RELATED_PIN_ONLY" ]; then
        CMD="$CMD $RELATED_PIN_ONLY"
    fi

    # Add use_target_norm flag if set
    if [ -n "$USE_TARGET_NORM" ]; then
        CMD="$CMD $USE_TARGET_NORM"
    fi

    # Add PDK scale factor if set (for cross-PDK unit conversion)
    if [ "$PDK_SCALE_FACTOR" != "1.0" ]; then
        CMD="$CMD --pdk_scale_factor $PDK_SCALE_FACTOR"
    fi

    # Add voltage shift if set (for cross-PDK voltage alignment)
    if [ "$VOLTAGE_SHIFT" != "0.0" ] && [ "$VOLTAGE_SHIFT" != "0" ]; then
        CMD="$CMD --voltage_shift $VOLTAGE_SHIFT"
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
echo "TSMC to ASAP7 Cross-PDK Validation Complete"
echo "=========================================="
echo "Total time: ${DURATION}s ($((DURATION / 60))m)"
echo ""

# Clean up PyTorch Geometric JIT cache files
echo "Cleaning up JIT cache files..."
rm -f "$SCRIPT_DIR"/torch_geometric.*.py 2>/dev/null
echo "Done."

exit 0
