#!/bin/bash

###########################################
# AADAM Iteration Sweep Runner
###########################################
#
# Usage:
#   ./run_aadam_iteration_sweep.sh <config.json> [--dry-run]
#
# Examples:
#   ./run_aadam_iteration_sweep.sh configs/aadam_iteration_sweep.json
#   ./run_aadam_iteration_sweep.sh configs/aadam_iteration_sweep.json --dry-run
#
###########################################

set -e  # Exit on error

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run]"
    echo ""
    echo "Example:"
    echo "  $0 configs/aadam_iteration_sweep.json"
    echo "  $0 configs/aadam_iteration_sweep.json --dry-run"
    exit 1
fi

CONFIG_FILE=$1
DRY_RUN=false

# Check for dry-run flag
if [ $# -eq 2 ] && [ "$2" == "--dry-run" ]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - Commands will be printed but not executed"
    echo ""
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "AADAM Iteration Sweep Runner"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo ""

# Generate commands from JSON config
COMMANDS=$(python3 generate_aadam_sweep_commands.py "$CONFIG_FILE")

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate commands from config"
    exit 1
fi

# Parse the output
EXPERIMENT_NAME=$(echo "$COMMANDS" | grep "^EXPERIMENT_NAME=" | cut -d'=' -f2)
MODE=$(echo "$COMMANDS" | grep "^MODE=" | cut -d'=' -f2)
BASE_CONFIG=$(echo "$COMMANDS" | grep "^BASE_CONFIG=" | cut -d'=' -f2)
BASE_NUM_TEST_SAMPLES=$(echo "$COMMANDS" | grep "^BASE_NUM_TEST_SAMPLES=" | cut -d'=' -f2)
BASE_SAVE_RESULTS=$(echo "$COMMANDS" | grep "^BASE_SAVE_RESULTS=" | cut -d'=' -f2)
BASE_TOTAL_POINTS=$(echo "$COMMANDS" | grep "^BASE_TOTAL_POINTS=" | cut -d'=' -f2)
BASE_GPU_ID=$(echo "$COMMANDS" | grep "^BASE_GPU_ID=" | cut -d'=' -f2)
BASE_MODEL_TYPE=$(echo "$COMMANDS" | grep "^BASE_MODEL_TYPE=" | cut -d'=' -f2)

echo "Experiment: $EXPERIMENT_NAME"
echo "Mode: $MODE"
echo ""

# Build base command parts
BASE_CMD="python -u ../MLP_topology_validation.py"
BASE_ARGS=""

[ -n "$BASE_CONFIG" ] && BASE_ARGS="$BASE_ARGS --config $BASE_CONFIG"
[ -n "$BASE_NUM_TEST_SAMPLES" ] && BASE_ARGS="$BASE_ARGS --num_test_samples $BASE_NUM_TEST_SAMPLES"
[ -n "$BASE_SAVE_RESULTS" ] && [ "$BASE_SAVE_RESULTS" = "True" -o "$BASE_SAVE_RESULTS" = "true" ] && BASE_ARGS="$BASE_ARGS --save_results"
[ -n "$BASE_TOTAL_POINTS" ] && BASE_ARGS="$BASE_ARGS --total_points $BASE_TOTAL_POINTS"
[ -n "$BASE_GPU_ID" ] && BASE_ARGS="$BASE_ARGS --gpu_id $BASE_GPU_ID"
[ -n "$BASE_MODEL_TYPE" ] && BASE_ARGS="$BASE_ARGS --model_type $BASE_MODEL_TYPE"

# Extract experiments
EXPERIMENTS=()
while IFS= read -r line; do
    if [[ $line == EXP_* ]]; then
        EXPERIMENTS+=("$line")
    fi
done <<< "$COMMANDS"

TOTAL_EXPS=${#EXPERIMENTS[@]}
echo "📋 Total experiments to run: $TOTAL_EXPS"
echo ""

# Run experiments
for i in "${!EXPERIMENTS[@]}"; do
    EXP_NUM=$((i + 1))
    EXP_LINE="${EXPERIMENTS[$i]}"

    # Parse experiment line
    EXP_ARGS=$(echo "$EXP_LINE" | cut -d'=' -f2- | sed 's/#.*//' | xargs)
    EXP_LABEL=$(echo "$EXP_LINE" | grep -o '#.*' | sed 's/# *//')

    echo "=========================================="
    echo "Experiment $EXP_NUM/$TOTAL_EXPS: $EXP_LABEL"
    echo "=========================================="

    # Build full command
    CMD="$BASE_CMD $BASE_ARGS $EXP_ARGS"

    echo "Command: $CMD"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "✓ Dry run - skipping execution"
    else
        eval $CMD
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Experiment $EXP_NUM completed successfully"
        else
            echo "✗ Experiment $EXP_NUM failed (exit code: $EXIT_CODE)"
        fi

        # GPU 메모리 정리
        echo ""
        echo "🧹 Cleaning up GPU memory..."
        python3 -c "import torch; torch.cuda.empty_cache(); print('✅ GPU cache cleared')" 2>/dev/null || echo "⚠️ GPU cleanup skipped (CUDA not available)"

        # 잠시 대기
        sleep 2
    fi
    echo ""
done

echo "=========================================="
if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed! ($TOTAL_EXPS experiments would be run in parallel)"
else
    if [ $FAILED -eq 0 ]; then
        echo "All experiments completed successfully! ($TOTAL_EXPS experiments run in parallel)"
    else
        echo "Experiments completed with $FAILED failures! ($TOTAL_EXPS experiments run in parallel)"
    fi
    echo ""
    echo "📝 Log files:"
    for LOG_FILE in "${LOG_FILES[@]}"; do
        echo "  - $LOG_FILE"
    done
fi
echo "=========================================="
