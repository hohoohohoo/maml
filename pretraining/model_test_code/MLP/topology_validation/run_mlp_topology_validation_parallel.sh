#!/bin/bash
# MLP Topology Validation - Parallel Execution by Cell
# Runs each cell as independent background process on different GPUs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--gpus 0,1,2,3] [--jobs-per-gpu 2]"
    echo ""
    echo "Example:"
    echo "  $0 configs/mlp_topology_validation_parallel.json --gpus 0,1,2,3"
    echo "  $0 configs/mlp_topology_validation_parallel.json --gpus 0,1 --jobs-per-gpu 4"
    echo "  $0 configs/mlp_topology_validation_parallel.json --dry-run"
    exit 1
fi

CONFIG_FILE="$1"
DRY_RUN=false
GPU_LIST="0,1,2"
JOBS_PER_GPU=1

# Parse optional flags
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gpus)
            GPU_LIST="$2"
            shift 2
            ;;
        --jobs-per-gpu)
            JOBS_PER_GPU="$2"
            shift 2
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

# Convert GPU list to array
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

# Calculate max parallel jobs
MAX_PARALLEL=$((NUM_GPUS * JOBS_PER_GPU))

echo "=========================================="
echo "MLP Topology Validation - PARALLEL MODE"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Available GPUs: ${GPUS[*]} (total: $NUM_GPUS)"
echo "Jobs per GPU: $JOBS_PER_GPU"
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""

# Parse base config from JSON
CONFIG_ID=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['config'])")
MODEL_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('model_type', 'aadam'))")
NUM_ITERATIONS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('num_iterations', 300000))")
NUM_TEST_SAMPLES=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('num_test_samples', 1000000))")
TOTAL_POINTS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('total_points', 61))")
ADAPTATION_METHOD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('adaptation_method', 'adam'))")
SAVE_RESULTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--save_results' if c.get('save_results', False) else '')")
OUTPUT_PREFIX=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('output_prefix', ''))")

# Get sweep parameters
DATA_TYPES=$(python3 -c "import json; print(' '.join(json.load(open('$CONFIG_FILE'))['sweep_params'].get('data_type', ['cell'])))")
MODES=$(python3 -c "import json; print(' '.join(json.load(open('$CONFIG_FILE'))['sweep_params'].get('mode', ['extrapolation'])))")

# Get cells from config or from test_dataset_config.py defaults
CELLS=$(python3 -c "
import json
import sys
sys.path.insert(0, 'utils')
from test_dataset_config import get_test_config

config = json.load(open('$CONFIG_FILE'))
cells = config.get('cells', [])

if not cells:
    # Get default cells from test_dataset_config
    config_id = config['base_config']['config']
    test_config = get_test_config(config_id)
    cells = test_config.get('default_cells', [])

print(' '.join(cells))
")

if [ -z "$CELLS" ]; then
    echo "Error: No cells found in config or test_dataset_config.py"
    exit 1
fi

# Create log directory
LOG_DIR="$SCRIPT_DIR/parallel_logs/mlp_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Config ID: $CONFIG_ID"
echo "Model type: $MODEL_TYPE"
echo "Adaptation method: $ADAPTATION_METHOD"
echo "Num iterations: $NUM_ITERATIONS"
echo ""
echo "Sweep parameters:"
echo "  Data types: $DATA_TYPES"
echo "  Modes: $MODES"
echo ""
echo "Cells: $CELLS"
echo ""
echo "Log directory: $LOG_DIR"
echo ""

# Function to build command
build_cmd() {
    local DATA_TYPE=$1
    local MODE=$2
    local CELL=$3
    local GPU=$4

    CMD="python -u MLP_topology_validation.py"
    CMD="$CMD --config $CONFIG_ID"
    CMD="$CMD --data_type $DATA_TYPE"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --cells $CELL"
    CMD="$CMD --model_type $MODEL_TYPE"
    CMD="$CMD --num_iterations $NUM_ITERATIONS"
    CMD="$CMD --num_test_samples $NUM_TEST_SAMPLES"
    CMD="$CMD --total_points $TOTAL_POINTS"
    CMD="$CMD --adaptation_method $ADAPTATION_METHOD"
    CMD="$CMD --gpu_id $GPU"

    if [ -n "$SAVE_RESULTS" ]; then
        CMD="$CMD $SAVE_RESULTS"
    fi
    if [ -n "$OUTPUT_PREFIX" ]; then
        CMD="$CMD --output_prefix $OUTPUT_PREFIX"
    fi

    echo "$CMD"
}

# Collect all jobs
declare -a JOBS
JOB_IDX=0

for DATA_TYPE in $DATA_TYPES; do
    for MODE in $MODES; do
        for CELL in $CELLS; do
            JOBS[$JOB_IDX]="$DATA_TYPE|$MODE|$CELL"
            JOB_IDX=$((JOB_IDX + 1))
        done
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Total jobs: $TOTAL_JOBS"
echo "Jobs will be distributed across $NUM_GPUS GPUs ($JOBS_PER_GPU jobs per GPU)"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands to be executed:"
    echo ""
    for i in "${!JOBS[@]}"; do
        IFS='|' read -r DATA_TYPE MODE CELL <<< "${JOBS[$i]}"
        GPU_IDX=$((i % NUM_GPUS))
        GPU=${GPUS[$GPU_IDX]}
        echo "[$((i+1))/$TOTAL_JOBS] GPU $GPU: $DATA_TYPE - $MODE - $CELL"
        CMD=$(build_cmd "$DATA_TYPE" "$MODE" "$CELL" "$GPU")
        echo "  $CMD"
        echo ""
    done
    exit 0
fi

# Run jobs in parallel, limited by MAX_PARALLEL
echo "Starting parallel execution (max $MAX_PARALLEL concurrent jobs)..."
echo ""

PIDS=()
RUNNING=0

for i in "${!JOBS[@]}"; do
    IFS='|' read -r DATA_TYPE MODE CELL <<< "${JOBS[$i]}"
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_IDX]}

    # Wait if max parallel jobs reached
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        for pid_idx in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$pid_idx]}" 2>/dev/null; then
                wait "${PIDS[$pid_idx]}"
                unset 'PIDS[pid_idx]'
                RUNNING=$((RUNNING - 1))
            fi
        done
        sleep 1
    done

    LOG_FILE="$LOG_DIR/${DATA_TYPE}_${MODE}_${CELL}_gpu${GPU}.log"
    CMD=$(build_cmd "$DATA_TYPE" "$MODE" "$CELL" "$GPU")

    echo "[$((i+1))/$TOTAL_JOBS] Starting: $DATA_TYPE - $MODE - $CELL on GPU $GPU"
    echo "  Log: $LOG_FILE"

    # Run in background
    eval "$CMD" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
    RUNNING=$((RUNNING + 1))
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs to complete..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "=========================================="
echo "All jobs completed!"
echo "=========================================="
echo "Logs saved in: $LOG_DIR"
echo ""

# Summary
echo "Results summary:"
PASS_COUNT=0
FAIL_COUNT=0
for log in "$LOG_DIR"/*.log; do
    if [ -f "$log" ]; then
        BASENAME=$(basename "$log" .log)
        if grep -q "Error\|Exception\|Traceback" "$log"; then
            echo "  FAILED: $BASENAME"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            echo "  OK: $BASENAME"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
    fi
done

echo ""
echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed"

exit 0
