#!/bin/bash
# Baseline GNN Training Sweep for TSMC Process Dataset (Unified 3D Format)
# Uses pre-processed TSMC unified dataset with 11D node features
# Standard mini-batch training (NOT MAML)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--no-commit]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/gnn_baseline_tsmc_process_sweep_config.json"
    echo "  $0 json_configs/gnn_baseline_tsmc_process_sweep_config.json --dry-run"
    echo ""
    echo "Note: Requires TSMC unified dataset to be pre-generated."
    echo "      Run build_gnn_dataset_tsmc_unified.py first."
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
echo "Baseline GNN Training - TSMC Process (Unified)"
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
            COMMIT_MSG="Start Baseline GNN TSMC Process sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Dataset: TSMC Process (11D node features: 7 base + 4 process params)
- Training type: Standard mini-batch (NOT MAML)

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

# Parse base_config with special handling for boolean flags
BASE_CONFIG=$(python3 -c "
import json
c = json.load(open('$CONFIG_FILE'))['base_config']
args = []
for k, v in c.items():
    if isinstance(v, bool):
        if v:
            args.append(f'--{k}')
    else:
        args.append(f'--{k} {v}')
print(' '.join(args))
")

SWEEP_PARAMS=$(python3 -c "import json; s=json.load(open('$CONFIG_FILE'))['sweep_params']; print(' '.join([f'--{k} {\" \".join(map(str, v))}' for k,v in s.items()]))")

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

# Calculate total combinations
TOTAL_COMBINATIONS=$(python3 -c "import json, itertools; s=json.load(open('$CONFIG_FILE'))['sweep_params']; print(len(list(itertools.product(*s.values()))))")

echo "Experiment: $EXPERIMENT_NAME"
echo "Total combinations: $TOTAL_COMBINATIONS"
if [ -n "$LOSS_LOGGING_CONFIG" ]; then
    echo "Loss logging: $LOSS_LOGGING_CONFIG"
fi
echo ""

# Build command - use Baseline version
FULL_CMD="python -u baseline_gnn_training_tsmc_process.py $BASE_CONFIG $SWEEP_PARAMS $LOSS_LOGGING_CONFIG"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE"
    echo ""
    echo "Command to be executed:"
    echo "$FULL_CMD"
    echo ""
    echo "This will:"
    echo "  1. Load TSMC Process unified dataset (11D node features)"
    echo "  2. Train $TOTAL_COMBINATIONS architectures sequentially"
    echo "  3. Save all models to pretrained_models/gnn_baseline_tsmc_process_final{suffix}/"
    echo "     (suffix: _vddonly if voltage_mode=vdd_only, _relpin if related_pin_only=true)"
    echo ""
    echo "Training type: Standard mini-batch (NOT MAML)"
    echo "  - Each iteration: randomly select one task"
    echo "  - Sample batch_size samples from that task"
    echo "  - Direct forward pass with Adam optimizer"
    echo "  - Weight decay (L2 regularization)"
    echo ""
    echo "Voltage mode options (voltage_mode):"
    echo "  - all_nodes: Voltage feature on all nodes (default)"
    echo "  - vdd_only: Voltage only on VDD nodes, 0 elsewhere"
    echo ""
    echo "Related pin only option (related_pin_only):"
    echo "  - false: Use all slew assignments (default)"
    echo "  - true: Use related_pin_only slew assignment (adds _relpin suffix)"
    echo ""
    echo "Loss logging options (in loss_logging):"
    echo "  - enabled: true/false - Enable/disable loss logging"
    echo "  - log_every: N - Log loss every N iterations (default: 1000)"
    echo "  - save_dir: path - Directory to save loss logs (default: loss_logs/)"
    echo ""
    echo "Dry run complete. Use without --dry-run to execute."
    exit 0
fi

# Execute
echo "Starting TSMC Process Baseline sweep..."
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
