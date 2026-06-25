#!/bin/bash
# Baseline GNN Training Sweep for TSMC Dataset (mmap Loading)
# Uses pre-processed TSMC dataset - no preprocessing needed

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--no-commit]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/baseline_gnn_tsmc_sweep_config.json"
    echo "  $0 json_configs/baseline_gnn_tsmc_sweep_config.json --dry-run"
    echo ""
    echo "Note: Requires TSMC dataset to be pre-generated."
    echo "      Run split_gnn_dataset_tsmc.py --corner <C> --temperature <T> first."
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
echo "Baseline GNN Training - TSMC Dataset"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Mode: mmap loading (memory efficient)"
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
            COMMIT_MSG="Start Baseline GNN TSMC sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Mode: TSMC dataset (mmap loading)

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

# Parse base_config - handle integer and string values properly
CORNER=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['corner'])")
TEMPERATURE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['temperature'])")
DATA_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['data_type'])")
GRAPH_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['graph_mode'])")
TOTAL_ITERATIONS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['total_iterations'])")
CHUNK_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['chunk_size'])")
LR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['lr'])")
WD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['wd'])")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('batch_size', 5))")
GPU=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['gpu'])")

# Parse sweep_params
SWEEP_PARAMS=$(python3 -c "import json; s=json.load(open('$CONFIG_FILE'))['sweep_params']; print(' '.join([f'--{k} {\" \".join(map(str, v))}' for k,v in s.items()]))")

# Calculate total combinations
TOTAL_COMBINATIONS=$(python3 -c "import json, itertools; s=json.load(open('$CONFIG_FILE'))['sweep_params']; print(len(list(itertools.product(*s.values()))))")

echo "Experiment: $EXPERIMENT_NAME"
echo ""
echo "Base config:"
echo "  Corner: $CORNER"
echo "  Temperature: $TEMPERATURE"
echo "  Data type: $DATA_TYPE"
echo "  Graph mode: $GRAPH_MODE"
echo "  Total iterations: $TOTAL_ITERATIONS"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Learning rate: $LR"
echo "  Weight decay: $WD"
echo "  Batch size: $BATCH_SIZE"
echo "  GPU: $GPU"
echo ""
echo "Total combinations: $TOTAL_COMBINATIONS"
echo ""

# Build command for TSMC baseline training
FULL_CMD="python -u baseline_gnn_training_tsmc.py"
FULL_CMD="$FULL_CMD --corner $CORNER --temperature $TEMPERATURE"
FULL_CMD="$FULL_CMD --data_type $DATA_TYPE --graph_mode $GRAPH_MODE"
FULL_CMD="$FULL_CMD --total_iterations $TOTAL_ITERATIONS --chunk_size $CHUNK_SIZE"
FULL_CMD="$FULL_CMD --lr $LR --wd $WD --batch_size $BATCH_SIZE"
FULL_CMD="$FULL_CMD --gpu $GPU"
FULL_CMD="$FULL_CMD $SWEEP_PARAMS"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE"
    echo ""
    echo "Command to be executed:"
    echo "$FULL_CMD"
    echo ""
    echo "This will:"
    echo "  1. Load TSMC dataset with mmap (memory efficient)"
    echo "  2. Train $TOTAL_COMBINATIONS architectures sequentially"
    echo "  3. Save all models"
    echo ""
    echo "Dry run complete. Use without --dry-run to execute."
    exit 0
fi

# Execute
echo "Starting TSMC baseline sweep..."
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
