#!/bin/bash
# Direct shell script to run MLP training from JSON config
# This script parses JSON using Python and directly executes baseline_mlp_training.py

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--no-commit]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/mlp_sweep_config.json"
    echo "  $0 json_configs/mlp_sweep_config.json --dry-run"
    echo "  $0 json_configs/mlp_sweep_config.json --no-commit"
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
echo "MLP Training from JSON Config"
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
            COMMIT_MSG="Start MLP training sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Script: run_baseline_mlp_sweep.sh

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

# Parse loss logging config (optional)
LOSS_LOGGING_CONFIG=$(python3 -c "
import json
config = json.load(open('$CONFIG_FILE'))
loss_logging = config.get('loss_logging', {})
args = []
if loss_logging.get('enabled', False):
    args.append('--enable_loss_logging')
    if loss_logging.get('log_every'):
        args.append(f'--loss_log_every {loss_logging[\"log_every\"]}')
    if loss_logging.get('save_dir'):
        args.append(f'--loss_log_dir {loss_logging[\"save_dir\"]}')
print(' '.join(args))
")

# Use Python utility to parse JSON and generate commands (utils is in parent directory)
COMMANDS=$(python3 ../utils/parse_sweep_config.py "$CONFIG_FILE")

# Parse the output
EXPERIMENT_NAME=$(echo "$COMMANDS" | grep "^EXPERIMENT_NAME=" | cut -d'=' -f2)
MODE=$(echo "$COMMANDS" | grep "^MODE=" | cut -d'=' -f2)
BASE_DATASET_CONFIG=$(echo "$COMMANDS" | grep "^BASE_DATASET_CONFIG=" | cut -d'=' -f2)
BASE_DATA_TYPE=$(echo "$COMMANDS" | grep "^BASE_DATA_TYPE=" | cut -d'=' -f2)
BASE_GPU_ID=$(echo "$COMMANDS" | grep "^BASE_GPU_ID=" | cut -d'=' -f2)
BASE_NUM_ITERATIONS=$(echo "$COMMANDS" | grep "^BASE_NUM_ITERATIONS=" | cut -d'=' -f2)

echo "Experiment: $EXPERIMENT_NAME"
echo "Mode: $MODE"
if [ -n "$LOSS_LOGGING_CONFIG" ]; then
    echo "Loss logging: $LOSS_LOGGING_CONFIG"
fi
echo ""

# Build base command parts
# Use python -u for unbuffered real-time output
# For background execution, use tmux/screen or run with: nohup ./script.sh &
BASE_CMD="python -u baseline_mlp_training.py"
BASE_ARGS=""

[ -n "$BASE_DATASET_CONFIG" ] && BASE_ARGS="$BASE_ARGS --dataset_config $BASE_DATASET_CONFIG"
[ -n "$BASE_DATA_TYPE" ] && BASE_ARGS="$BASE_ARGS --data_type $BASE_DATA_TYPE"
[ -n "$BASE_GPU_ID" ] && BASE_ARGS="$BASE_ARGS --gpu_id $BASE_GPU_ID"
[ -n "$BASE_NUM_ITERATIONS" ] && BASE_ARGS="$BASE_ARGS --num_iterations $BASE_NUM_ITERATIONS"
[ -n "$LOSS_LOGGING_CONFIG" ] && BASE_ARGS="$BASE_ARGS $LOSS_LOGGING_CONFIG"

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

        # Add to command
        CMD="$CMD --$param_name $param_value"
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
