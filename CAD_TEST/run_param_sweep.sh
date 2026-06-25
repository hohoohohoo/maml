#!/bin/bash
# Simple parameter sweep script using GNU Parallel or xargs
# Usage: ./run_param_sweep.sh [--dry-run] [--max-parallel N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/validate_lut_table_sweep_full_cells.py"
LOG_DIR="$SCRIPT_DIR/sweep_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Default settings
MAX_PARALLEL=4
DRY_RUN=false
GPU_ID=4
REF_MODE=middle
MODEL_BASE_DIR="/home/tkdgn2907/Deepsets_test/MAML/Projects/CAD_TEST/trained_models_lib_sweep"
ITERATION=""  # Leave empty to use ITERATIONS sweep in Python code below
CORNER_FILTER=""
TEMP_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-parallel|-p) MAX_PARALLEL="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --gpu|-g) GPU_ID="$2"; shift 2 ;;
        --ref-mode|-r) REF_MODE="$2"; shift 2 ;;
        --model-base-dir|-m) MODEL_BASE_DIR="$2"; shift 2 ;;
        --iteration|-i) ITERATION="$2"; shift 2 ;;
        --corner-filter|-c) CORNER_FILTER="$2"; shift 2 ;;
        --temp-filter|-t) TEMP_FILTER="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --max-parallel, -p N      Max parallel jobs (default: 4)"
            echo "  --gpu, -g ID              GPU ID (default: 4)"
            echo "  --ref-mode, -r MODE       Reference voltage mode: corner, middle, both (default: middle)"
            echo "  --model-base-dir, -m DIR  Base directory for condition-specific models"
            echo "  --iteration, -i NUM       Specific iteration for model checkpoint"
            echo "  --corner-filter, -c LIST  Filter corners (e.g., SS,FF)"
            echo "  --temp-filter, -t LIST    Filter temperatures (e.g., 125,-40)"
            echo "  --dry-run                 Print commands without executing"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build fixed parameters
FIXED_PARAMS="--local_norm --local_temp_norm --local_volt_norm --num_tables 50 --gpu $GPU_ID --ref_mode $REF_MODE"

# Add optional parameters if specified
if [ -n "$MODEL_BASE_DIR" ]; then
    FIXED_PARAMS="$FIXED_PARAMS --model_base_dir $MODEL_BASE_DIR"
fi
if [ -n "$ITERATION" ]; then
    FIXED_PARAMS="$FIXED_PARAMS --iteration $ITERATION"
    echo "NOTE: Using fixed iteration=$ITERATION (set ITERATIONS=[] in Python code)"
fi
if [ -n "$CORNER_FILTER" ]; then
    FIXED_PARAMS="$FIXED_PARAMS --corner_filter $CORNER_FILTER"
fi
if [ -n "$TEMP_FILTER" ]; then
    FIXED_PARAMS="$FIXED_PARAMS --temp_filter $TEMP_FILTER"
fi

echo "======================================================"
echo "  Parameter Sweep Script"
echo "======================================================"
echo "GPU: $GPU_ID"
echo "Max parallel: $MAX_PARALLEL"
echo "Ref mode: $REF_MODE"
echo "Model base dir: ${MODEL_BASE_DIR:-'(not set)'}"
echo "Corner filter: ${CORNER_FILTER:-'all'}"
echo "Temp filter: ${TEMP_FILTER:-'all'}"
echo "Log dir: $LOG_DIR"
echo ""

# Generate commands file
CMD_FILE="$LOG_DIR/commands.txt"

echo "Generating parameter combinations..."

# Use Python to generate combinations (much faster than bash loops)
python3 << 'PYTHON_SCRIPT' > "$CMD_FILE"
import itertools

# Sweep parameters - modify these lists as needed
ITERATIONS = [2000, 4000, 6000, 8000, 10000]  # Empty list [] to skip iteration sweep
CENTER_STEPS = [100]
CENTER_LR = ['3e-4']
CENTER_LOSS_THRESHOLD = ['1e-4', '1e-3', '1e-2', '0']  # 0 means always run Adam
MAPE_THRESHOLD = [1.0]
PERIPHERAL_ADAM_STEPS = [20]
PERIPHERAL_ADAM_LR = ['3e-4']

# If ITERATIONS is empty, use None as placeholder
if not ITERATIONS:
    ITERATIONS = [None]

for it, cs, cl, clt, mt, pas, pal in itertools.product(
    ITERATIONS, CENTER_STEPS, CENTER_LR, CENTER_LOSS_THRESHOLD, MAPE_THRESHOLD, PERIPHERAL_ADAM_STEPS, PERIPHERAL_ADAM_LR
):
    params = f"--center_steps {cs} --center_lr {cl} --center_loss_threshold {clt} --mape_threshold {mt} --peripheral_adam_steps {pas} --peripheral_adam_lr {pal}"
    log_name = f"cs{cs}_cl{cl}_clt{clt}_mt{mt}_pas{pas}_pal{pal}"

    if it is not None:
        params = f"--iteration {it} {params}"
        log_name = f"iter{it}_{log_name}"

    print(f"{params}|{log_name}")
PYTHON_SCRIPT

TOTAL=$(wc -l < "$CMD_FILE")
echo "Generated $TOTAL combinations"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - First 10 commands:"
    head -10 "$CMD_FILE" | while IFS='|' read -r params log_name; do
        echo "python $PYTHON_SCRIPT $params $FIXED_PARAMS"
        echo "  -> $LOG_DIR/${log_name}.log"
        echo ""
    done
    echo "... and $((TOTAL - 10)) more"
    exit 0
fi

echo "Starting $TOTAL experiments (max $MAX_PARALLEL parallel)..."
echo ""

# Run using xargs for parallel execution
run_one() {
    params="$1"
    log_name="$2"
    log_file="$LOG_DIR/${log_name}.log"

    echo "[$(date +%H:%M:%S)] Starting: $log_name"
    python "$PYTHON_SCRIPT" $params $FIXED_PARAMS > "$log_file" 2>&1
    echo "[$(date +%H:%M:%S)] Done: $log_name"
}
export -f run_one
export PYTHON_SCRIPT LOG_DIR FIXED_PARAMS

# Use GNU parallel if available, otherwise xargs
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel"
    cat "$CMD_FILE" | parallel -j $MAX_PARALLEL --colsep '\|' run_one {1} {2}
else
    echo "Using xargs (install GNU parallel for better progress tracking)"
    cat "$CMD_FILE" | xargs -P $MAX_PARALLEL -I {} bash -c '
        IFS="|" read -r params log_name <<< "{}"
        run_one "$params" "$log_name"
    '
fi

echo ""
echo "======================================================"
echo "  All experiments completed!"
echo "======================================================"
echo "Logs saved to: $LOG_DIR"
