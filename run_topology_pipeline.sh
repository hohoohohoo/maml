#!/bin/bash

################################################################################
# Topology Processing Pipeline
#
# Complete workflow for topology data processing:
# 1. Data Preprocessing (ASAP7 or TSMC)
# 2. Model Pretraining
# 3. Model Validation
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PROCESSING_DIR="${PROJECT_ROOT}/data_processing"
PRETRAINING_DIR="${PROJECT_ROOT}/pretraining/model_pretraining_code"
VALIDATION_DIR="${PROJECT_ROOT}/pretraining/model_test_code"

# Log file
LOG_FILE="${PROJECT_ROOT}/topology_pipeline.log"

################################################################################
# Helper Functions
################################################################################

print_banner() {
    echo -e "${CYAN}"
    echo "================================================================================"
    echo "                  Topology Processing Pipeline"
    echo "================================================================================"
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}[STEP $1/$2]${NC} $3"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

run_command() {
    local step_name=$1
    shift

    echo ""
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    log_message "Starting: $step_name"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"

    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        print_success "Completed: $step_name"
        log_message "SUCCESS: $step_name"
        return 0
    else
        print_error "Failed: $step_name"
        log_message "FAILED: $step_name"
        return 1
    fi
}

################################################################################
# Pipeline Steps
################################################################################

step_preprocessing() {
    print_step 1 3 "Data Preprocessing"

    if [ "$INTERACTIVE" = true ]; then
        if [ "$PDK" = "asap7" ]; then
            run_command "Data Preprocessing (ASAP7)" \
                python "${DATA_PROCESSING_DIR}/run_asap7_topology_preprocessing.py"
        else
            run_command "Data Preprocessing (TSMC)" \
                python "${DATA_PROCESSING_DIR}/run_tsmc_topology_preprocessing.py"
        fi
    else
        # Command-line mode with provided arguments
        if [ "$PDK" = "asap7" ]; then
            local cmd="python ${DATA_PROCESSING_DIR}/run_asap7_topology_preprocessing.py"

            if [ -n "$DATA_DIR" ]; then
                cmd="$cmd --data_dir $DATA_DIR"
            fi

            if [ -n "$DELAY_TYPE" ]; then
                cmd="$cmd --delay_type $DELAY_TYPE"
            fi

            if [ -n "$TOPOLOGY_TYPE" ]; then
                cmd="$cmd --topology_type $TOPOLOGY_TYPE"
            fi

            cmd="$cmd --yes"

            run_command "Data Preprocessing (ASAP7)" bash -c "$cmd"
        else
            local cmd="python ${DATA_PROCESSING_DIR}/run_tsmc_topology_preprocessing.py"

            if [ -n "$DATASET_TYPE" ]; then
                cmd="$cmd --dataset_type $DATASET_TYPE"
            fi

            if [ -n "$DELAY_TYPE" ]; then
                cmd="$cmd --delay_type $DELAY_TYPE"
            fi

            cmd="$cmd --yes"

            run_command "Data Preprocessing (TSMC)" bash -c "$cmd"
        fi
    fi
}

step_pretraining() {
    print_step 2 3 "Model Pretraining"

    if [ "$INTERACTIVE" = true ]; then
        run_command "Model Pretraining" \
            python "${PRETRAINING_DIR}/run_topology_pretraining.py"
    else
        local cmd="python ${PRETRAINING_DIR}/run_topology_pretraining.py"
        cmd="$cmd $MODEL --config $CONFIG"

        if [ -n "$DATA_TYPE" ]; then
            cmd="$cmd --data_type $DATA_TYPE"
        fi

        if [ -n "$GPU_ID" ]; then
            if [ "$MODEL" = "mlp" ]; then
                cmd="$cmd --gpu_id $GPU_ID"
            else
                cmd="$cmd --gpu $GPU_ID"
            fi
        fi

        # Model-specific parameters
        if [ "$MODEL" = "mlp" ]; then
            if [ -n "$MODEL_TYPE" ]; then
                cmd="$cmd --model_type $MODEL_TYPE"
            fi
            if [ -n "$NUM_ITERATIONS" ]; then
                cmd="$cmd --num_iterations $NUM_ITERATIONS"
            fi
        else  # maml
            if [ -n "$INNER" ]; then
                cmd="$cmd --inner $INNER"
            fi
            if [ -n "$INNERDIV" ]; then
                cmd="$cmd --innerdiv $INNERDIV"
            fi
            if [ -n "$META" ]; then
                cmd="$cmd --meta $META"
            fi
            if [ -n "$NUM_ITERATIONS" ]; then
                cmd="$cmd --num_iterations $NUM_ITERATIONS"
            fi
        fi

        cmd="$cmd --yes"

        run_command "Model Pretraining" bash -c "$cmd"
    fi
}

step_validation() {
    print_step 3 3 "Model Validation"

    if [ "$INTERACTIVE" = true ]; then
        run_command "Model Validation" \
            python "${VALIDATION_DIR}/run_topology_validation.py"
    else
        local cmd="python ${VALIDATION_DIR}/run_topology_validation.py"
        cmd="$cmd --model $MODEL --config $CONFIG"

        if [ -n "$TEST_MODE" ]; then
            cmd="$cmd --mode $TEST_MODE"
        fi

        if [ -n "$DATA_TYPE" ]; then
            cmd="$cmd --data_type $DATA_TYPE"
        fi

        if [ -n "$GPU_ID" ]; then
            cmd="$cmd --gpu_id $GPU_ID"
        fi

        cmd="$cmd --yes"

        run_command "Model Validation" bash -c "$cmd"
    fi
}

################################################################################
# Interactive Mode
################################################################################

select_start_step() {
    echo ""
    echo "Select starting step:"
    echo "────────────────────────────────────────────────────────────────"
    echo "  [1] Data Preprocessing (Start from beginning)"
    echo "  [2] Model Pretraining (Skip preprocessing)"
    echo "  [3] Model Validation (Only validation)"
    echo "  [4] Full Pipeline (All steps)"
    echo ""

    while true; do
        read -p "Select step [1-4] (default: 4): " choice
        choice=${choice:-4}

        case $choice in
            1) START_STEP=1; break ;;
            2) START_STEP=2; break ;;
            3) START_STEP=3; break ;;
            4) START_STEP=1; END_STEP=3; break ;;
            *) echo "Invalid choice. Please select 1-4." ;;
        esac
    done

    if [ "$choice" != "4" ]; then
        echo ""
        echo "Select ending step:"
        echo "────────────────────────────────────────────────────────────────"

        # Define step names array
        local step_names=("Data Preprocessing" "Model Pretraining" "Model Validation")

        # Show only steps from START_STEP to 3
        for ((i=START_STEP; i<=3; i++)); do
            local step_index=$((i-1))
            echo "  [$i] ${step_names[$step_index]}"
        done
        echo ""

        while true; do
            read -p "Select end step [$START_STEP-3] (default: 3): " choice
            choice=${choice:-3}

            if [ "$choice" -ge "$START_STEP" ] && [ "$choice" -le 3 ]; then
                END_STEP=$choice
                break
            else
                echo "Invalid choice. Please select $START_STEP-3."
            fi
        done
    fi
}

select_pdk() {
    echo ""
    echo "Select PDK:"
    echo "────────────────────────────────────────────────────────────────"
    echo "  [0] ASAP7"
    echo "  [1] TSMC (Default)"
    echo ""

    while true; do
        read -p "Select PDK [0/1] (default: 1): " choice
        choice=${choice:-1}

        case $choice in
            0) PDK="asap7"; break ;;
            1) PDK="tsmc"; break ;;
            *) echo "Invalid choice. Please select 0 or 1." ;;
        esac
    done
}

interactive_mode() {
    print_banner

    echo "Running in Interactive Mode"
    echo ""

    select_start_step
    select_pdk

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Pipeline Configuration:"
    echo "  PDK: ${PDK^^}"
    echo "  Start Step: $START_STEP"
    echo "  End Step: $END_STEP"
    echo "  Log File: $LOG_FILE"
    echo "────────────────────────────────────────────────────────────────"
    echo ""

    read -p "Proceed with pipeline execution? [Y/n]: " confirm
    confirm=${confirm:-Y}

    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Pipeline cancelled."
        exit 0
    fi

    # Execute selected steps
    STEP_NAMES=("preprocessing" "pretraining" "validation")

    for ((i=START_STEP; i<=END_STEP; i++)); do
        step_name=${STEP_NAMES[$((i-1))]}

        if ! "step_$step_name"; then
            print_error "Pipeline failed at step $i: $step_name"
            exit 1
        fi
    done
}

################################################################################
# Command-line Mode
################################################################################

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Topology Processing Pipeline

Options:
  -h, --help                 Show this help message
  -i, --interactive          Run in interactive mode (default if no args)

  Pipeline Control:
  --start-step STEP          Starting step (1-3, default: 1)
  --end-step STEP            Ending step (1-3, default: 3)

  PDK Options:
  --pdk PDK                  PDK selection: asap7 or tsmc (required for command-line)

  ASAP7 Preprocessing Options:
  --data-dir DIR             Data directory index: 0-3 (default: 1)
  --topology-type TYPE       Topology type: intra or agnostic

  TSMC Preprocessing Options:
  --dataset-type TYPE        Dataset type: original_agnostic, original_intra, nor_nand, seq

  Common Preprocessing Options:
  --delay-type TYPE          Delay type: cell or transition (default: cell)

  Pretraining Options:
  --model MODEL              Model type: mlp or maml (required for step 2+)
  --config CONFIG            Dataset config: 0-3 (required for step 2+)
                             0=ASAP7 intra, 1=ASAP7 agnostic, 2=TSMC intra, 3=TSMC agnostic
  --data-type TYPE           Data type: cell or transition
  --gpu-id ID                GPU device ID (default: 0)

  MLP Options:
  --model-type TYPE          MLP model type: aadam or mlp
  --num-iterations N         Number of iterations

  MAML Options:
  --inner N                  Inner loop steps
  --innerdiv N               Inner learning rate divisor
  --meta N                   Meta batch size
  --num-iterations N         Number of iterations

  Validation Options:
  --test-mode MODE           Test mode: extrapolation or interpolation

Pipeline Steps:
  1. Data Preprocessing
  2. Model Pretraining
  3. Model Validation

Examples:
  # Interactive mode
  $0 -i

  # Full ASAP7 pipeline
  $0 --pdk asap7 --model maml --config 0 --data-dir 1 --topology-type intra

  # Full TSMC pipeline
  $0 --pdk tsmc --model mlp --config 2 --dataset-type original_intra

  # Only preprocessing and pretraining
  $0 --pdk tsmc --model maml --config 3 --start-step 1 --end-step 2

  # Only validation
  $0 --pdk asap7 --model mlp --config 1 --start-step 3 --end-step 3

EOF
}

commandline_mode() {
    print_banner

    echo "Running in Command-line Mode"
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Pipeline Configuration:"
    echo "  PDK: ${PDK^^}"
    echo "  Start Step: $START_STEP"
    echo "  End Step: $END_STEP"

    if [ "$START_STEP" -le 1 ]; then
        echo ""
        echo "Preprocessing:"
        if [ "$PDK" = "asap7" ]; then
            echo "  Data directory: ${DATA_DIR:-1}"
            echo "  Topology type: ${TOPOLOGY_TYPE:-agnostic}"
        else
            echo "  Dataset type: ${DATASET_TYPE:-original_agnostic}"
        fi
        echo "  Delay type: ${DELAY_TYPE:-cell}"
    fi

    if [ "$START_STEP" -le 2 ] && [ "$END_STEP" -ge 2 ]; then
        echo ""
        echo "Pretraining:"
        echo "  Model: ${MODEL^^}"
        echo "  Config: $CONFIG"
        echo "  Data type: ${DATA_TYPE:-cell}"
        echo "  GPU ID: ${GPU_ID:-0}"
    fi

    if [ "$END_STEP" -ge 3 ]; then
        echo ""
        echo "Validation:"
        echo "  Test mode: ${TEST_MODE:-extrapolation}"
    fi

    echo "  Log File: $LOG_FILE"
    echo "────────────────────────────────────────────────────────────────"
    echo ""

    # Execute selected steps
    STEP_NAMES=("preprocessing" "pretraining" "validation")

    for ((i=START_STEP; i<=END_STEP; i++)); do
        step_name=${STEP_NAMES[$((i-1))]}

        if ! "step_$step_name"; then
            print_error "Pipeline failed at step $i: $step_name"
            exit 1
        fi
    done
}

################################################################################
# Main
################################################################################

main() {
    # Default values
    INTERACTIVE=false
    START_STEP=1
    END_STEP=3
    PDK=""
    DATA_DIR=""
    DATASET_TYPE=""
    DELAY_TYPE=""
    TOPOLOGY_TYPE=""
    MODEL=""
    CONFIG=""
    DATA_TYPE=""
    GPU_ID=""
    MODEL_TYPE=""
    NUM_ITERATIONS=""
    INNER=""
    INNERDIV=""
    META=""
    TEST_MODE=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                print_usage
                exit 0
                ;;
            -i|--interactive)
                INTERACTIVE=true
                shift
                ;;
            --start-step)
                START_STEP="$2"
                shift 2
                ;;
            --end-step)
                END_STEP="$2"
                shift 2
                ;;
            --pdk)
                PDK="$2"
                shift 2
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --dataset-type)
                DATASET_TYPE="$2"
                shift 2
                ;;
            --delay-type)
                DELAY_TYPE="$2"
                shift 2
                ;;
            --topology-type)
                TOPOLOGY_TYPE="$2"
                shift 2
                ;;
            --model)
                MODEL="$2"
                shift 2
                ;;
            --config)
                CONFIG="$2"
                shift 2
                ;;
            --data-type)
                DATA_TYPE="$2"
                shift 2
                ;;
            --gpu-id)
                GPU_ID="$2"
                shift 2
                ;;
            --model-type)
                MODEL_TYPE="$2"
                shift 2
                ;;
            --num-iterations)
                NUM_ITERATIONS="$2"
                shift 2
                ;;
            --inner)
                INNER="$2"
                shift 2
                ;;
            --innerdiv)
                INNERDIV="$2"
                shift 2
                ;;
            --meta)
                META="$2"
                shift 2
                ;;
            --test-mode)
                TEST_MODE="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Initialize log file
    echo "==============================================================================" > "$LOG_FILE"
    echo "Topology Pipeline Log - $(date)" >> "$LOG_FILE"
    echo "==============================================================================" >> "$LOG_FILE"

    # Determine mode
    if [ "$INTERACTIVE" = true ] || [ -z "$PDK" ]; then
        INTERACTIVE=true
        interactive_mode
    else
        commandline_mode
    fi

    # Success summary
    echo ""
    echo -e "${GREEN}================================================================================"
    echo "                    PIPELINE COMPLETED SUCCESSFULLY"
    echo "================================================================================${NC}"
    echo "Log file: $LOG_FILE"
    echo ""
}

# Run main
main "$@"
