#!/bin/bash
# MAML Heterogeneous GNN Training Sweep for TSMC Process Dataset (Unified 3D Format)
# Uses pre-processed TSMC unified dataset with 11D node features
# Heterogeneous GNN uses type-specific transformations for Power, Port, NMOS, PMOS nodes

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--no-commit]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/hetero_gnn_maml_tsmc_process_sweep_config.json"
    echo "  $0 json_configs/hetero_gnn_maml_tsmc_process_sweep_config.json --dry-run"
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
echo "MAML Heterogeneous GNN Training - TSMC Process (Unified)"
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
            COMMIT_MSG="Start MAML Heterogeneous GNN TSMC Process sweep: $(basename $CONFIG_FILE)

Experiment Configuration:
- Config file: $CONFIG_FILE
- Timestamp: $TIMESTAMP
- Dataset: TSMC Process (11D node features: 7 base + 4 process params)
- Model: Heterogeneous GNN (type-specific transformations for Power, Port, NMOS, PMOS)

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
if [ -n "$RESUME_CONFIG" ]; then
    echo "Resume config: $RESUME_CONFIG"
fi
echo ""

# Build command - use Heterogeneous GNN version
FULL_CMD="python -u maml_hetero_gnn_training_tsmc_process.py $BASE_CONFIG $SWEEP_PARAMS $RESUME_CONFIG"

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE"
    echo ""
    echo "Command to be executed:"
    echo "$FULL_CMD"
    echo ""
    echo "This will:"
    echo "  1. Load TSMC Process unified dataset (11D node features)"
    echo "  2. Train $TOTAL_COMBINATIONS Heterogeneous GNN architectures sequentially"
    echo "  3. Save all models to pretrained_models/hetero_gnn_maml_tsmc_process_final[_suffixes]/"
    echo ""
    echo "Heterogeneous GNN Features:"
    echo "  - Type-specific transformations for 4 node types:"
    echo "    * Power (VDD/VSS)"
    echo "    * Port (Output/Input/Intermediate)"
    echo "    * NMOS transistors"
    echo "    * PMOS transistors"
    echo "  - Supports both GCN and GAT convolution types"
    echo ""
    echo "Supported conv_type options:"
    echo "  - gcn: GCN-based heterogeneous convolution (default)"
    echo "  - gat: GAT-based with attention mechanism"
    echo ""
    echo "Supported voltage_mode options:"
    echo "  - all_nodes: Voltage applied to all nodes (default)"
    echo "  - vdd_only: Voltage only on VDD node, 0 elsewhere"
    echo "  - vdd_mos: Voltage on VDD and MOS transistor nodes only"
    echo ""
    echo "Supported temp_mode options:"
    echo "  - typical: Temperature on MOS nodes only (default)"
    echo "  - temp_all: Temperature on all nodes"
    echo ""
    echo "Supported topology_suffix options:"
    echo "  - (empty): Default topology"
    echo "  - _gatectrl: With gate control edges (cache only)"
    echo "  - _inputport: With input port nodes"
    echo "  - _relpin: Input slew only on related pin's MOS/port"
    echo "  - Combinations: _inputport_relpin, _gatectrl_inputport, etc."
    echo ""
    echo "Supported pooling options:"
    echo "  - mean: Global mean pooling (default)"
    echo "  - max: Global max pooling"
    echo "  - add: Global sum pooling"
    echo "  - output: Output-node-only pooling"
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
echo "Starting TSMC Process Heterogeneous GNN sweep..."
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
