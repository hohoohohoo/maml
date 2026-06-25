#!/bin/bash
# GNN Iteration Sweep Validation - Parallel Execution by Iteration
# Runs each iteration checkpoint as independent background process on different GPUs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.json> [--dry-run] [--gpus 0,1,2,3] [--jobs-per-gpu 2] [--cells CELL1,CELL2]"
    echo ""
    echo "Example:"
    echo "  $0 json_configs/validation_gnn_asap7_iteration_sweep_config.json --gpus 4,5,6,7"
    echo "  $0 json_configs/validation_gnn_asap7_iteration_sweep_config.json --gpus 0,1,2,3 --jobs-per-gpu 2"
    echo "  $0 json_configs/validation_gnn_asap7_iteration_sweep_config.json --gpus 4,5 --cells AND2x6,NAND3x2"
    echo "  $0 json_configs/validation_gnn_asap7_iteration_sweep_config.json --dry-run"
    exit 1
fi

CONFIG_FILE="$1"
DRY_RUN=false
GPU_LIST="0,1"
JOBS_PER_GPU=2
SPECIFIC_CELLS=""

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
        --cells)
            SPECIFIC_CELLS="$2"
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
echo "GNN ITERATION SWEEP - PARALLEL MODE"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Available GPUs: ${GPUS[*]} (total: $NUM_GPUS)"
echo "Jobs per GPU: $JOBS_PER_GPU"
echo "Max parallel jobs: $MAX_PARALLEL"
echo ""

# Parse base config from JSON
MODEL_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['model_type'])")
DATA_TYPE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['data_type'])")
GRAPH_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['graph_mode'])")
VOLTAGE_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('voltage_mode', 'all_nodes'))")
NORMALIZATION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('normalization', 'zscore'))")
TEMP_MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('temp_mode', 'typical'))")
CACHE_PATH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('cache_path', ''))")
INPUTPORT=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--inputport' if c.get('inputport', False) else '')")
RELATED_PIN_ONLY=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--related_pin_only' if c.get('related_pin_only', False) else '')")
SAMPLE_SUFFIX=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('sample_suffix', '_10pct'))")
MODE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['mode'])")
TOTAL_POINTS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config']['total_points'])")
NUM_TEST_SAMPLES=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('num_test_samples', 2000))")
SAVE_RESULTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--save_results' if c.get('save_results', False) else '')")
FILTER_CONTINUOUS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE'))['base_config']; print('--filter_continuous' if c.get('filter_continuous', False) else '')")
CONTINUITY_THRESHOLD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('continuity_threshold', 0.18))")
POOLING=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('pooling', 'mean'))")
ADAPTATION_METHOD=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('adaptation_method', 'selective_adam'))")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['base_config'].get('output_dir', 'iteration_sweep'))")

# MAML-specific params
INNERDIV=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('innerdiv', 1))")
TASKS_PER_META_BATCH=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('tasks_per_meta_batch', 16))")
INNER_STEPS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('maml_config', {}).get('inner_steps', 1))")

# Get sweep parameters
CONV_DIM=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['sweep_params']['conv_hidden_dim'][0])")
CONV_LAYERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['sweep_params']['num_conv_layers'][0])")
FC_DIM=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['sweep_params']['fc_hidden_dim'][0])")
FC_LAYERS=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['sweep_params']['num_fc_layers'][0])")
EXPERIMENTS=$(python3 -c "import json; print(' '.join(json.load(open('$CONFIG_FILE'))['sweep_params']['experiment']))")
ITERATIONS=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$CONFIG_FILE'))['sweep_params']['iterations'])))")

echo "Sweep parameters:"
echo "  Experiments: $EXPERIMENTS"
echo "  Iterations: $ITERATIONS"
echo ""

# Define cell lists
INTRA_TOPOLOGY_CELLS="AND2x6 NAND3x2 NOR2xp67 OR2x6"
TOPOLOGY_AGNOSTIC_CELLS="A2O1A1O1Ixp25 AO21x1 AO32x1 AOI332xp5 O2A1O1Ixp5 OAI22x1 HAxp5 MAJIxp5 MAJx2 MAJx3 XNOR2x1 XNOR2x2 XNOR2xp5 XOR2x1 XOR2x2 XOR2xp5"

# Create log directory
LOG_DIR="$SCRIPT_DIR/parallel_logs/iteration_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Log directory: $LOG_DIR"
echo ""

# Function to build base command
build_base_cmd() {
    local EXP=$1
    local CELL=$2
    local ITER=$3
    local GPU=$4

    CMD="python -u ASAP7_GCN_topology_validation.py"
    CMD="$CMD --experiment $EXP"
    CMD="$CMD --cells $CELL"
    CMD="$CMD --model_type $MODEL_TYPE"
    CMD="$CMD --data_type $DATA_TYPE"
    CMD="$CMD --graph_mode $GRAPH_MODE"
    CMD="$CMD --voltage_mode $VOLTAGE_MODE"
    CMD="$CMD --normalization $NORMALIZATION"
    CMD="$CMD --temp_mode $TEMP_MODE"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --total_points $TOTAL_POINTS"
    CMD="$CMD --num_test_samples $NUM_TEST_SAMPLES"
    CMD="$CMD --num_iterations $ITER"
    CMD="$CMD --conv_hidden_dim $CONV_DIM"
    CMD="$CMD --num_conv_layers $CONV_LAYERS"
    CMD="$CMD --fc_hidden_dim $FC_DIM"
    CMD="$CMD --num_fc_layers $FC_LAYERS"
    CMD="$CMD --pooling $POOLING"
    CMD="$CMD --adaptation_method $ADAPTATION_METHOD"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --gpu $GPU"

    if [ -n "$CACHE_PATH" ]; then
        CMD="$CMD --cache_path $CACHE_PATH"
    fi
    if [ -n "$INPUTPORT" ]; then
        CMD="$CMD $INPUTPORT"
    fi
    if [ -n "$RELATED_PIN_ONLY" ]; then
        CMD="$CMD $RELATED_PIN_ONLY"
    fi
    CMD="$CMD --sample_suffix $SAMPLE_SUFFIX"

    if [ "$MODEL_TYPE" = "maml" ]; then
        CMD="$CMD --innerdiv $INNERDIV"
        CMD="$CMD --tasks_per_meta_batch $TASKS_PER_META_BATCH"
        CMD="$CMD --inner_steps $INNER_STEPS"
    fi
    if [ -n "$SAVE_RESULTS" ]; then
        CMD="$CMD $SAVE_RESULTS"
    fi
    if [ -n "$FILTER_CONTINUOUS" ]; then
        CMD="$CMD $FILTER_CONTINUOUS"
        CMD="$CMD --continuity_threshold $CONTINUITY_THRESHOLD"
    fi

    echo "$CMD"
}

# Collect all jobs
declare -a JOBS
JOB_IDX=0

for EXP in $EXPERIMENTS; do
    if [ "$EXP" = "intra_topology" ]; then
        CELLS="$INTRA_TOPOLOGY_CELLS"
    else
        CELLS="$TOPOLOGY_AGNOSTIC_CELLS"
    fi

    # Use specific cells if provided
    if [ -n "$SPECIFIC_CELLS" ]; then
        CELLS=$(echo "$SPECIFIC_CELLS" | tr ',' ' ')
    fi

    for CELL in $CELLS; do
        for ITER in $ITERATIONS; do
            JOBS[$JOB_IDX]="$EXP|$CELL|$ITER"
            JOB_IDX=$((JOB_IDX + 1))
        done
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Total jobs: $TOTAL_JOBS"
echo "  = ${#GPUS[@]} experiments x cells x iterations"
echo "Jobs will be distributed across $NUM_GPUS GPUs ($JOBS_PER_GPU jobs per GPU)"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Commands to be executed:"
    echo ""
    for i in "${!JOBS[@]}"; do
        IFS='|' read -r EXP CELL ITER <<< "${JOBS[$i]}"
        GPU_IDX=$((i % NUM_GPUS))
        GPU=${GPUS[$GPU_IDX]}
        echo "[$((i+1))/$TOTAL_JOBS] GPU $GPU: $EXP - $CELL - iter$ITER"
        CMD=$(build_base_cmd "$EXP" "$CELL" "$ITER" "$GPU")
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
    IFS='|' read -r EXP CELL ITER <<< "${JOBS[$i]}"
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

    LOG_FILE="$LOG_DIR/${EXP}_${CELL}_iter${ITER}_gpu${GPU}.log"
    CMD=$(build_base_cmd "$EXP" "$CELL" "$ITER" "$GPU")

    echo "[$((i+1))/$TOTAL_JOBS] Starting: $EXP - $CELL - iter$ITER on GPU $GPU"
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
SUCCESS=0
FAILED=0
for log in "$LOG_DIR"/*.log; do
    if [ -f "$log" ]; then
        BASENAME=$(basename "$log" .log)
        if grep -q "Error\|Exception\|Traceback" "$log"; then
            echo "  FAILED: $BASENAME"
            FAILED=$((FAILED + 1))
        else
            echo "  OK: $BASENAME"
            SUCCESS=$((SUCCESS + 1))
        fi
    fi
done

echo ""
echo "Total: $SUCCESS succeeded, $FAILED failed"
echo ""

# Generate summary CSV
echo "Generating iteration sweep summary..."
SUMMARY_FILE="$LOG_DIR/iteration_sweep_summary.csv"
echo "experiment,cell,iteration,mape_total,mape_inter,mape_leftex,mape_rightex" > "$SUMMARY_FILE"

for log in "$LOG_DIR"/*.log; do
    if [ -f "$log" ]; then
        BASENAME=$(basename "$log" .log)
        # Extract experiment, cell, iteration from filename
        EXP=$(echo "$BASENAME" | cut -d'_' -f1-2)
        CELL=$(echo "$BASENAME" | cut -d'_' -f3)
        ITER=$(echo "$BASENAME" | grep -oP 'iter\K[0-9]+')

        # Extract MAPE values from log
        MAPE_TOTAL=$(grep "Total MAPE:" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        MAPE_INTER=$(grep "Interpolation MAPE:" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        MAPE_LEFT=$(grep "Left Extrapolation MAPE:" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
        MAPE_RIGHT=$(grep "Right Extrapolation MAPE:" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")

        echo "$EXP,$CELL,$ITER,$MAPE_TOTAL,$MAPE_INTER,$MAPE_LEFT,$MAPE_RIGHT" >> "$SUMMARY_FILE"
    fi
done

echo "Summary saved to: $SUMMARY_FILE"
