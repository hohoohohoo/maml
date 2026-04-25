#!/bin/bash
# MLP Validation Sweep Script
# Runs MLP validation with parameter sweep from JSON config

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--no-commit]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/mlp_topology_validation_sweep.json"
    echo "  $0 json_configs/mlp_topology_validation_sweep.json --dry-run"
    echo "  $0 json_configs/mlp_topology_validation_sweep.json --no-commit"
    exit 1
fi

CONFIG_FILE="$1"
DRY_RUN=false
NO_COMMIT=false

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-commit)
            NO_COMMIT=true
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
echo "MLP Validation Sweep from JSON Config"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo ""

# Git commit for experiment tracking
if [ "$NO_COMMIT" = false ] && [ "$DRY_RUN" = false ]; then
    # Check if we're in a git repository
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "📝 Creating git commit for experiment tracking..."

        # Add config file to git
        git add "$CONFIG_FILE" 2>/dev/null

        # Check if there are changes to commit
        if git diff --cached --quiet; then
            echo "ℹ️  No changes to commit (config file unchanged)"
        else
            # Get current timestamp
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

            # Create commit message with experiment info
            COMMIT_MSG="Start MLP validation sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Script: run_mlp_topology_validation_sweep.sh

🤖 Generated with Claude Code
"

            # Commit the changes
            git commit -m "$COMMIT_MSG" --no-verify

            if [ $? -eq 0 ]; then
                echo "✅ Experiment config committed to git"
                COMMIT_HASH=$(git rev-parse --short HEAD)
                echo "   Commit: $COMMIT_HASH"
            else
                echo "⚠️  Warning: Git commit failed, continuing anyway..."
            fi
        fi
    else
        echo "ℹ️  Not in a git repository, skipping commit"
    fi
    echo ""
elif [ "$NO_COMMIT" = true ]; then
    echo "ℹ️  Git commit disabled (--no-commit flag)"
    echo ""
fi

# Use Python utility to parse JSON and generate commands
COMMANDS=$(python3 utils/parse_sweep_config.py "$CONFIG_FILE")

# Parse the output
EXPERIMENT_NAME=$(echo "$COMMANDS" | grep "^EXPERIMENT_NAME=" | cut -d'=' -f2)
MODE=$(echo "$COMMANDS" | grep "^MODE=" | cut -d'=' -f2)
BASE_CONFIG=$(echo "$COMMANDS" | grep "^BASE_CONFIG=" | cut -d'=' -f2)
BASE_MODE=$(echo "$COMMANDS" | grep "^BASE_MODE=" | cut -d'=' -f2)
BASE_NUM_ITERATIONS=$(echo "$COMMANDS" | grep "^BASE_NUM_ITERATIONS=" | cut -d'=' -f2)
BASE_NUM_TEST_SAMPLES=$(echo "$COMMANDS" | grep "^BASE_NUM_TEST_SAMPLES=" | cut -d'=' -f2)
BASE_SAVE_RESULTS=$(echo "$COMMANDS" | grep "^BASE_SAVE_RESULTS=" | cut -d'=' -f2)
BASE_TOTAL_POINTS=$(echo "$COMMANDS" | grep "^BASE_TOTAL_POINTS=" | cut -d'=' -f2)
BASE_GPU_ID=$(echo "$COMMANDS" | grep "^BASE_GPU_ID=" | cut -d'=' -f2)
BASE_DATA_TYPE=$(echo "$COMMANDS" | grep "^BASE_DATA_TYPE=" | cut -d'=' -f2)
BASE_MODEL_TYPE=$(echo "$COMMANDS" | grep "^BASE_MODEL_TYPE=" | cut -d'=' -f2)
BASE_ADAPTATION_METHOD=$(echo "$COMMANDS" | grep "^BASE_ADAPTATION_METHOD=" | cut -d'=' -f2)

echo "Experiment: $EXPERIMENT_NAME"
echo "Adaptation method: $BASE_ADAPTATION_METHOD"
echo "Mode: $MODE"
echo ""

# Build base command parts
BASE_CMD="python -u MLP_topology_validation.py"
BASE_ARGS=""

[ -n "$BASE_CONFIG" ] && BASE_ARGS="$BASE_ARGS --config $BASE_CONFIG"
[ -n "$BASE_MODE" ] && BASE_ARGS="$BASE_ARGS --mode $BASE_MODE"
[ -n "$BASE_NUM_ITERATIONS" ] && BASE_ARGS="$BASE_ARGS --num_iterations $BASE_NUM_ITERATIONS"
[ -n "$BASE_NUM_TEST_SAMPLES" ] && BASE_ARGS="$BASE_ARGS --num_test_samples $BASE_NUM_TEST_SAMPLES"
[ -n "$BASE_SAVE_RESULTS" ] && [ "$BASE_SAVE_RESULTS" = "True" -o "$BASE_SAVE_RESULTS" = "true" ] && BASE_ARGS="$BASE_ARGS --save_results"
[ -n "$BASE_TOTAL_POINTS" ] && BASE_ARGS="$BASE_ARGS --total_points $BASE_TOTAL_POINTS"
[ -n "$BASE_GPU_ID" ] && BASE_ARGS="$BASE_ARGS --gpu_id $BASE_GPU_ID"
[ -n "$BASE_DATA_TYPE" ] && BASE_ARGS="$BASE_ARGS --data_type $BASE_DATA_TYPE"
[ -n "$BASE_MODEL_TYPE" ] && BASE_ARGS="$BASE_ARGS --model_type $BASE_MODEL_TYPE"
[ -n "$BASE_ADAPTATION_METHOD" ] && BASE_ARGS="$BASE_ARGS --adaptation_method $BASE_ADAPTATION_METHOD"

# Extract experiments
EXPERIMENTS=()
while IFS= read -r line; do
    if [[ $line == EXP_* ]]; then
        EXPERIMENTS+=("$line")
    fi
done <<< "$COMMANDS"

TOTAL_EXPS=${#EXPERIMENTS[@]}
echo "Total experiments: $TOTAL_EXPS"
echo ""

# Run each experiment
for ((idx=0; idx<${#EXPERIMENTS[@]}; idx++)); do
    EXP_LINE="${EXPERIMENTS[$idx]}"
    EXP_NUM=$((idx + 1))

    # Parse experiment parameters
    EXP_PARAMS="${EXP_LINE#*:}"

    echo "=========================================="
    echo "Experiment $EXP_NUM / $TOTAL_EXPS"
    echo "=========================================="

    # Build command
    CMD="$BASE_CMD $BASE_ARGS"

    # Parse and add experiment-specific parameters
    IFS=' ' read -ra PARAMS <<< "$EXP_PARAMS"
    for param in "${PARAMS[@]}"; do
        param_name="${param%=*}"
        param_value="${param#*=}"

        echo "  $param_name: $param_value"

        # Handle special case for cells (list parameters)
        if [ "$param_name" = "cells" ]; then
            # Replace commas with spaces for cell list
            cell_list=$(echo "$param_value" | tr ',' ' ')
            CMD="$CMD --$param_name $cell_list"
        else
            # Add to command normally
            CMD="$CMD --$param_name $param_value"
        fi
    done

    echo ""
    echo "Command: $CMD"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN - not executing]"
    else
        eval $CMD
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Experiment $EXP_NUM completed successfully"
        else
            echo "✗ Experiment $EXP_NUM failed (exit code: $EXIT_CODE)"
        fi
    fi
    echo ""
done

echo "=========================================="
if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed! ($TOTAL_EXPS experiments would be run)"
else
    echo "All experiments completed! ($TOTAL_EXPS experiments run)"
fi
echo "=========================================="
