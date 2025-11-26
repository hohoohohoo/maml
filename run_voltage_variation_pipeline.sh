#!/bin/bash

################################################################################
# Voltage Variation Processing Pipeline
#
# Complete workflow for voltage variation data processing:
# 1. Data Preprocessing
# 2. Data Merge & Train/Test Split
# 3. Model Pretraining
# 4. Model Validation
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
LOG_FILE="${PROJECT_ROOT}/voltage_variation_pipeline.log"

################################################################################
# Helper Functions
################################################################################

print_banner() {
    echo -e "${CYAN}"
    echo "================================================================================"
    echo "              Voltage Variation Processing Pipeline"
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
    print_step 1 4 "Data Preprocessing"

    if [ "$INTERACTIVE" = true ]; then
        run_command "Data Preprocessing" \
            python "${DATA_PROCESSING_DIR}/run_voltage_variation_preprocessing.py"
    else
        # Command-line mode with provided arguments
        local cmd="python ${DATA_PROCESSING_DIR}/run_voltage_variation_preprocessing.py"
        cmd="$cmd -t $DATASET_TYPE"

        if [ -n "$VT_TYPE" ] && [ -n "$CORNER" ]; then
            cmd="$cmd --vt-type $VT_TYPE --corner $CORNER"
        elif [ -n "$FOLDER" ]; then
            cmd="$cmd --folder $FOLDER"
        fi

        if [ -n "$DELAY_TYPE" ]; then
            cmd="$cmd --delay-type $DELAY_TYPE"
        fi

        cmd="$cmd --yes"

        run_command "Data Preprocessing" bash -c "$cmd"
    fi
}

step_merge_split() {
    print_step 2 4 "Data Merge & Train/Test Split"

    if [ "$INTERACTIVE" = true ]; then
        run_command "Data Merge & Split" \
            python "${DATA_PROCESSING_DIR}/run_voltage_variation_data_merge_split.py"
    else
        run_command "Data Merge & Split" \
            python "${DATA_PROCESSING_DIR}/run_voltage_variation_data_merge_split.py" \
            -t "$DATASET_TYPE" --yes
    fi
}

step_pretraining() {
    print_step 3 4 "Model Pretraining"

    if [ "$INTERACTIVE" = true ]; then
        run_command "Model Pretraining" \
            python "${PRETRAINING_DIR}/run_voltage_variation_pretraining.py"
    else
        local cmd="python ${PRETRAINING_DIR}/run_voltage_variation_pretraining.py"
        cmd="$cmd -t $DATASET_TYPE"

        if [ -n "$VT_TYPE" ] && [ -n "$CORNER" ]; then
            cmd="$cmd --vt-type $VT_TYPE --corner $CORNER"
        fi

        if [ -n "$DELAY_TYPE" ]; then
            cmd="$cmd --delay-type $DELAY_TYPE"
        fi

        run_command "Model Pretraining" bash -c "$cmd"
    fi
}

step_validation() {
    print_step 4 4 "Model Validation"

    if [ "$INTERACTIVE" = true ]; then
        run_command "Model Validation" \
            python "${VALIDATION_DIR}/run_voltage_variation_validation.py"
    else
        local cmd="python ${VALIDATION_DIR}/run_voltage_variation_validation.py"
        cmd="$cmd -t $DATASET_TYPE"

        if [ -n "$VT_TYPE" ] && [ -n "$CORNER" ]; then
            cmd="$cmd --vt-type $VT_TYPE --corner $CORNER"
        fi

        if [ -n "$DELAY_TYPE" ]; then
            cmd="$cmd --delay-type $DELAY_TYPE"
        fi

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
    echo "  [2] Data Merge & Split (Skip preprocessing)"
    echo "  [3] Model Pretraining (Skip to training)"
    echo "  [4] Model Validation (Only validation)"
    echo "  [5] Full Pipeline (All steps)"
    echo ""

    while true; do
        read -p "Select step [1-5] (default: 5): " choice
        choice=${choice:-5}

        case $choice in
            1) START_STEP=1; break ;;
            2) START_STEP=2; break ;;
            3) START_STEP=3; break ;;
            4) START_STEP=4; break ;;
            5) START_STEP=1; END_STEP=4; break ;;
            *) echo "Invalid choice. Please select 1-5." ;;
        esac
    done

    if [ "$choice" != "5" ]; then
        echo ""
        echo "Select ending step:"
        echo "────────────────────────────────────────────────────────────────"

        # Define step names array
        local step_names=("Data Preprocessing" "Data Merge & Split" "Model Pretraining" "Model Validation")

        # Show only steps from START_STEP to 4
        for ((i=START_STEP; i<=4; i++)); do
            local step_index=$((i-1))
            echo "  [$i] ${step_names[$step_index]}"
        done
        echo ""

        while true; do
            read -p "Select end step [$START_STEP-4] (default: 4): " choice
            choice=${choice:-4}

            if [ "$choice" -ge "$START_STEP" ] && [ "$choice" -le 4 ]; then
                END_STEP=$choice
                break
            else
                echo "Invalid choice. Please select $START_STEP-4."
            fi
        done
    fi
}

interactive_mode() {
    print_banner

    echo "Running in Interactive Mode"
    echo ""

    select_start_step

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Pipeline Configuration:"
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
    STEP_NAMES=("preprocessing" "merge_split" "pretraining" "validation")

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

Voltage Variation Processing Pipeline

Options:
  -h, --help                 Show this help message
  -i, --interactive          Run in interactive mode (default if no args)

  Pipeline Control:
  --start-step STEP          Starting step (1-4, default: 1)
  --end-step STEP            Ending step (1-4, default: 4)

  Dataset Options:
  -t, --dataset-type TYPE    Dataset type: tsmc or asap7 (required)
  --delay-type TYPE          Delay type: cell or transition (default: cell)

  ASAP7 Options:
  --vt-type VT               VT type: LVT, RVT, SLVT, SRAM
  --corner CORNER            Corner: FF, TT, SS
  --folder FOLDER            Specific folder to process

  TSMC Options:
  --folder FOLDER            Specific folder to process (e.g., TSMC_FF_0)

Pipeline Steps:
  1. Data Preprocessing
  2. Data Merge & Train/Test Split
  3. Model Pretraining
  4. Model Validation

Examples:
  # Interactive mode
  $0 -i

  # Full pipeline for ASAP7 LVT-FF
  $0 -t asap7 --vt-type LVT --corner FF

  # Only preprocessing and merge/split
  $0 -t tsmc --folder TSMC_FF_0 --start-step 1 --end-step 2

  # Only training and validation
  $0 -t asap7 --vt-type RVT --corner TT --start-step 3 --end-step 4

EOF
}

commandline_mode() {
    print_banner

    echo "Running in Command-line Mode"
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "Pipeline Configuration:"
    echo "  Dataset Type: $DATASET_TYPE"
    echo "  Delay Type: ${DELAY_TYPE:-cell}"
    echo "  Start Step: $START_STEP"
    echo "  End Step: $END_STEP"

    if [ -n "$VT_TYPE" ] && [ -n "$CORNER" ]; then
        echo "  VT Type: $VT_TYPE"
        echo "  Corner: $CORNER"
    elif [ -n "$FOLDER" ]; then
        echo "  Folder: $FOLDER"
    fi

    echo "  Log File: $LOG_FILE"
    echo "────────────────────────────────────────────────────────────────"
    echo ""

    # Execute selected steps
    STEP_NAMES=("preprocessing" "merge_split" "pretraining" "validation")

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
    END_STEP=4
    DATASET_TYPE=""
    DELAY_TYPE=""
    VT_TYPE=""
    CORNER=""
    FOLDER=""

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
            -t|--dataset-type)
                DATASET_TYPE="$2"
                shift 2
                ;;
            --delay-type)
                DELAY_TYPE="$2"
                shift 2
                ;;
            --vt-type)
                VT_TYPE="$2"
                shift 2
                ;;
            --corner)
                CORNER="$2"
                shift 2
                ;;
            --folder)
                FOLDER="$2"
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
    echo "Voltage Variation Pipeline Log - $(date)" >> "$LOG_FILE"
    echo "==============================================================================" >> "$LOG_FILE"

    # Determine mode
    if [ "$INTERACTIVE" = true ] || [ -z "$DATASET_TYPE" ]; then
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
