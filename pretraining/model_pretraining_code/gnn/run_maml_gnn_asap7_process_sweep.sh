#!/bin/bash
# MAML GNN Training Sweep for ASAP7 Process Dataset (Unified 3D Format)
# Uses pre-processed ASAP7 unified dataset with 11D node features
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
    echo "  $0 json_configs/gnn_maml_asap7_process_sweep_config.json"
    echo "  $0 json_configs/gnn_maml_asap7_process_sweep_config.json --dry-run"
    echo ""
    echo "Note: Requires ASAP7 unified dataset to be pre-generated."
    echo "      Run asap7/build_gnn_dataset_process_cached_asap7.py first."
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
echo "MAML GNN Training - ASAP7 Process (Unified)"
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
            COMMIT_MSG="Start MAML GNN ASAP7 Process sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Dataset: ASAP7 Process (11D node features: 7 base + 4 process params)

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
# Parse base_config, handling inputport and related_pin_only specially (boolean flags)
BASE_CONFIG=$(python3 -c "
import json
c = json.load(open('$CONFIG_FILE'))['base_config']
args = []
for k, v in c.items():
    if k == 'inputport':
        if v:
            args.append('--inputport')
    elif k == 'related_pin_only':
        if v:
            args.append('--related_pin_only')
    elif isinstance(v, bool):
        if v:
            args.append(f'--{k}')
    else:
        args.append(f'--{k} {v}')
print(' '.join(args))
")
SWEEP_PARAMS=$(python3 -c "import json; s=json.load(open('$CONFIG_FILE'))['sweep_params']; print(' '.join([f'--{k} {\" \".join(map(str, v))}' for k,v in s.items()]))")

# Parse topology options for display
INPUTPORT=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('enabled' if c.get('inputport', False) else 'disabled')")
RELATED_PIN_ONLY=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('enabled' if c.get('related_pin_only', False) else 'disabled')")
SAMPLING=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('sampling', '10pct'))")
VOLTAGE_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('voltage_mode', 'all_nodes'))")
CACHE_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('cache_path', ''))")

# Parse resume config (optional)
RESUME_CONFIG=$(python3 -c "
import json
config = json.load(open('$CONFIG_FILE'))
resume_config = config.get('resume_config', {})
args = []
if resume_config.get('auto_resume', False):
    args.append('--auto_resume')
if resume_config.get('resume'):
    args.append(f'--resume {resume_config[\"resume\"]}')
if resume_config.get('additional_iterations'):
    args.append(f'--additional_iterations {resume_config[\"additional_iterations\"]}')
print(' '.join(args))
")

# Calculate total combinations
TOTAL_COMBINATIONS=$(python3 -c "import json, itertools; s=json.load(open('$CONFIG_FILE'))['sweep_params']; print(len(list(itertools.product(*s.values()))))")

echo "Experiment: $EXPERIMENT_NAME"
echo "Total combinations: $TOTAL_COMBINATIONS"
echo ""
echo "Topology settings:"
echo "  Inputport: $INPUTPORT"
echo "  Related pin only: $RELATED_PIN_ONLY"
echo "  Sampling: $SAMPLING"
echo "  Voltage mode: $VOLTAGE_MODE"
if [ -n "$CACHE_PATH" ]; then
    echo "  Cache path: $CACHE_PATH"
    # Show auto-detected topology options from cache_path
    if [[ "$CACHE_PATH" == *"_gatectrl"* ]]; then
        echo "  Gate control: auto-detected (enabled)"
    else
        echo "  Gate control: auto-detected (disabled)"
    fi
    if [[ "$CACHE_PATH" == *"_directmos"* ]]; then
        echo "  Direct MOS: auto-detected (enabled - skip intermediate nodes)"
    fi
    if [[ "$CACHE_PATH" == *"_bidir"* ]]; then
        echo "  Bidirectional: auto-detected (enabled)"
    fi
fi
if [ -n "$RESUME_CONFIG" ]; then
    echo "Resume config: $RESUME_CONFIG"
fi
echo ""

# Build command - use ASAP7 Process version
FULL_CMD="python -u maml_gnn_training_asap7_process.py $BASE_CONFIG $SWEEP_PARAMS $RESUME_CONFIG"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE"
    echo ""
    echo "Command to be executed:"
    echo "$FULL_CMD"
    echo ""
    echo "This will:"
    echo "  1. Load ASAP7 Process unified dataset (11D node features)"
    echo "  2. Train $TOTAL_COMBINATIONS architectures sequentially"
    echo "  3. Save all models to pretrained_models/gnn_maml_asap7_process_final/"
    echo ""
    echo "Pooling: (configured in JSON config)"
    echo "  - mean (default): Global mean pooling"
    echo "  - output: Output-node-only pooling"
    echo ""
    echo "Supported voltage_mode options:"
    echo "  - all_nodes: Voltage applied to all nodes (default)"
    echo "  - vdd_only: Voltage only on VDD node, 0 elsewhere"
    echo "  - vdd_mos: Voltage on VDD and MOS transistor nodes only"
    echo ""
    echo "Topology options:"
    echo "  - inputport: Adds input port nodes to graph (affects train file)"
    echo "  - related_pin_only: Input slew assigned only to related pin's MOS/port"
    echo "  - sampling: 'full' or 'Xpct' (e.g., '10pct', '50pct') - always adds _{sampling} suffix"
    echo ""
    echo "Resume training options (in resume_config):"
    echo "  - auto_resume: true/false - Auto-find latest checkpoint"
    echo "  - resume: path - Specific checkpoint file to resume from"
    echo "  - additional_iterations: N - Train N more iterations when resuming"
    echo ""
    echo "Dry run complete. Use without --dry-run to execute."
    exit 0
fi

# Execute
echo "Starting ASAP7 Process sweep..."
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
