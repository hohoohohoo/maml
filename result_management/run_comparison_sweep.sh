#!/bin/bash
# Adam vs MAML Comparison Sweep Script
# Runs comparison across multiple configurations from JSON config

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--no-commit]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/comparison_sweep.json"
    echo "  $0 json_configs/comparison_sweep.json --dry-run"
    echo "  $0 json_configs/comparison_sweep.json --no-commit"
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
echo "Adam vs MAML Comparison Sweep"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo ""

# Git commit for experiment tracking
if [ "$NO_COMMIT" = false ] && [ "$DRY_RUN" = false ]; then
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "📝 Creating git commit for experiment tracking..."
        git add "$CONFIG_FILE" 2>/dev/null
        if git diff --cached --quiet; then
            echo "ℹ️  No changes to commit (config file unchanged)"
        else
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            COMMIT_MSG="Start comparison sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Script: run_comparison_sweep.sh

🤖 Generated with Claude Code
"
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
COMMANDS=$(python3 utils/parse_comparison_sweep.py "$CONFIG_FILE")

# Parse the output
EXPERIMENT_NAME=$(echo "$COMMANDS" | grep "^EXPERIMENT_NAME=" | cut -d'=' -f2)
MODE=$(echo "$COMMANDS" | grep "^MODE=" | cut -d'=' -f2)
BASE_NUM_ITERATIONS=$(echo "$COMMANDS" | grep "^BASE_NUM_ITERATIONS=" | cut -d'=' -f2-)
BASE_GROUP_SIZE=$(echo "$COMMANDS" | grep "^BASE_GROUP_SIZE=" | cut -d'=' -f2)
BASE_SAVE=$(echo "$COMMANDS" | grep "^BASE_SAVE=" | cut -d'=' -f2)
BASE_METRIC=$(echo "$COMMANDS" | grep "^BASE_METRIC=" | cut -d'=' -f2)
BASE_INNERDIV=$(echo "$COMMANDS" | grep "^BASE_INNERDIV=" | cut -d'=' -f2-)
BASE_META=$(echo "$COMMANDS" | grep "^BASE_META=" | cut -d'=' -f2-)

echo "Experiment: $EXPERIMENT_NAME"
echo "Mode: $MODE"
echo ""

# Build base command parts
BASE_CMD="python compare_adam_maml.py"
BASE_ARGS=""

[ -n "$BASE_NUM_ITERATIONS" ] && BASE_ARGS="$BASE_ARGS --num-iterations $BASE_NUM_ITERATIONS"
[ -n "$BASE_GROUP_SIZE" ] && BASE_ARGS="$BASE_ARGS --group-size $BASE_GROUP_SIZE"
[ -n "$BASE_SAVE" ] && [ "$BASE_SAVE" = "true" ] && BASE_ARGS="$BASE_ARGS --save"
[ -n "$BASE_METRIC" ] && BASE_ARGS="$BASE_ARGS --metric $BASE_METRIC"
[ -n "$BASE_INNERDIV" ] && BASE_ARGS="$BASE_ARGS --innerdiv $BASE_INNERDIV"
[ -n "$BASE_META" ] && BASE_ARGS="$BASE_ARGS --meta $BASE_META"

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

        # Add to command (convert underscores to hyphens for CLI)
        cli_param_name=$(echo "$param_name" | tr '_' '-')
        CMD="$CMD --$cli_param_name $param_value"
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
