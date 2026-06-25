#!/bin/bash
# MAML GNN Training Sweep for ASAP7 Process Parasitic Cap Dataset (Unified 3D Format)
# Uses pre-processed ASAP7 unified dataset with 12D node features (including parasitic cap)
#
# Node Features (12D):
#   - 7 base electrical features
#   - param_a, param_b, param_c, temperature
#   - parasitic_cap (capacitance values from SPI extraction)
#
# Pooling Options (set in JSON config):
#   - mean: Global mean pooling (default, baseline)
#   - max: Global max pooling
#   - add: Global sum pooling
#   - output: Output-node-only pooling (extracts only output node embedding)
#
# Output node index is dynamically determined from topology cache.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--no-commit]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/gnn_maml_asap7_process_parasitic_sweep_config.json"
    echo "  $0 json_configs/gnn_maml_asap7_process_parasitic_sweep_config.json --dry-run"
    echo ""
    echo "Note: Requires ASAP7 unified dataset with parasitic cap to be pre-generated."
    echo "      Run build_gnn_dataset_cached_with_process.py with --include_parasitic_cap first."
    echo ""
    echo "Pooling Options (set 'pooling' in JSON config):"
    echo "  - mean: Global mean pooling (default)"
    echo "  - max: Global max pooling"
    echo "  - add: Global sum pooling"
    echo "  - output: Output-node-only pooling"
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
echo "MAML GNN Training - ASAP7 Process Parasitic Cap (12D)"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo ""

# Git commit for experiment tracking
if [ "$NO_COMMIT" = false ] && [ "$DRY_RUN" = false ]; then
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "Creating git commit for experiment tracking..."
        git add "$CONFIG_FILE" 2>/dev/null
        if git diff --cached --quiet; then
            echo "No changes to commit (config file unchanged)"
        else
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            COMMIT_MSG="Start MAML GNN ASAP7 Process Parasitic sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Dataset: ASAP7 Process Parasitic Cap (12D node features: 7 base + 4 process params + parasitic_cap)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
"
            git commit -m "$COMMIT_MSG" --no-verify
            if [ $? -eq 0 ]; then
                echo "Experiment config committed to git"
                COMMIT_HASH=$(git rev-parse --short HEAD)
                echo "Commit: $COMMIT_HASH"
            else
                echo "Warning: Git commit failed, continuing anyway..."
            fi
        fi
    else
        echo "Not in a git repository, skipping commit"
    fi
    echo ""
elif [ "$NO_COMMIT" = true ]; then
    echo "Git commit disabled (--no-commit flag)"
    echo ""
fi

# Parse JSON config
EXPERIMENT_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['experiment_name'])")
BASE_CONFIG=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print(' '.join([f'--{k} {v}' if not isinstance(v, bool) else (f'--{k}' if v else '') for k,v in c.items()]))")
SWEEP_PARAMS=$(python3 -c "import json; s=json.load(open('$CONFIG_FILE'))['sweep_params']; print(' '.join([f'--{k} {\" \".join(map(str, v))}' for k,v in s.items()]))")

# Calculate total combinations
TOTAL_COMBINATIONS=$(python3 -c "import json, itertools; s=json.load(open('$CONFIG_FILE'))['sweep_params']; print(len(list(itertools.product(*s.values()))))")

echo "Experiment: $EXPERIMENT_NAME"
echo "Total combinations: $TOTAL_COMBINATIONS"
echo ""

# Build command - use ASAP7 Process Parasitic version
FULL_CMD="python -u maml_gnn_training_asap7_process_parasitic.py $BASE_CONFIG $SWEEP_PARAMS"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE"
    echo ""
    echo "Command to be executed:"
    echo "$FULL_CMD"
    echo ""
    echo "This will:"
    echo "  1. Load ASAP7 Process Parasitic unified dataset (12D node features with parasitic cap)"
    echo "  2. Train $TOTAL_COMBINATIONS architectures sequentially"
    echo "  3. Save all models to pretrained_models/gnn_maml_asap7_process_parasitic_final/"
    echo ""
    echo "Pooling: (configured in JSON config)"
    echo "  - mean (default): Global mean pooling"
    echo "  - output: Output-node-only pooling"
    echo ""
    echo "Dry run complete. Use without --dry-run to execute."
    exit 0
fi

# Execute
echo "Starting ASAP7 Process Parasitic sweep..."
echo ""
echo "Command:"
echo "$FULL_CMD"
echo ""
echo "=========================================="
echo ""

START_TIME=$(date +%s)
eval $FULL_CMD
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Sweep completed successfully"
else
    echo "Sweep failed with exit code $EXIT_CODE"
fi
echo "Total time: ${DURATION}s ($((DURATION / 60))m)"
echo "=========================================="

exit $EXIT_CODE
