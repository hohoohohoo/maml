#!/usr/bin/env bash
set -euo pipefail

# Sequential sweep runner for train_maml_from_lib.py
# Sweeps only the valid (corner, temp) pairs defined in VOLTAGE_SWEEP_GROUPS.
#
# Example:
#   chmod +x run_corner_temp_sweep.sh
#   ./run_corner_temp_sweep.sh
#
# Override settings with environment variables:
#   GPU=0 DATA_TYPE=cell NUM_ITERATIONS=2000 ./run_corner_temp_sweep.sh
#   GPU=1 DATA_TYPE=transition AUTO_RESUME=1 ./run_corner_temp_sweep.sh --enable_loss_logging
#
# Extra CLI arguments passed to this script are forwarded to train_maml_from_lib.py.

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./train_maml_from_lib.py}"
GPU="${GPU:-0}"
DATA_TYPE="${DATA_TYPE:-transition}"            # cell | transition
NUM_ITERATIONS="${NUM_ITERATIONS:-10000}"
INNER="${INNER:-1}"
INNERDIV="${INNERDIV:-100}"
META="${META:-32}"
K="${K:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./trained_models_lib_sweep}"
AUTO_RESUME="${AUTO_RESUME:-0}"            # 1 => add --auto_resume
UNIT_CONVERT="${UNIT_CONVERT:-0}"          # 1 => add --unit_convert
FREEZE_HIDDEN="${FREEZE_HIDDEN:-0}"        # 1 => add --freeze_hidden
ENABLE_LOSS_LOGGING="${ENABLE_LOSS_LOGGING:-0}"  # 1 => add --enable_loss_logging
LOSS_LOG_EVERY="${LOSS_LOG_EVERY:-1000}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"      # 1 => continue even if one run fails

EXTRA_ARGS=("$@")

VALID_DATA_TYPES=("cell" "transition")
if [[ ! " ${VALID_DATA_TYPES[*]} " =~ " ${DATA_TYPE} " ]]; then
  echo "[ERROR] DATA_TYPE must be 'cell' or 'transition', got: ${DATA_TYPE}" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[ERROR] Training script not found: $TRAIN_SCRIPT" >&2
  echo "        Set TRAIN_SCRIPT=/path/to/train_maml_from_lib.py" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# Valid sweep pairs from VOLTAGE_SWEEP_GROUPS in train_maml_from_lib.py
SWEEP_PAIRS=(
  "SS -40"
  "SS 125"
  "FF -40"
  "FF 125"
  "TT 25"
)

run_count=0
fail_count=0

for pair in "${SWEEP_PAIRS[@]}"; do
  read -r CORNER TEMP <<< "$pair"
  run_count=$((run_count + 1))

  temp_tag="${TEMP}C"
  temp_tag="${temp_tag/-/m}"

  RUN_NAME="${DATA_TYPE}_${CORNER}_${temp_tag}"
  RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
  LOG_DIR="${RUN_DIR}/logs"
  LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p "$LOG_DIR"

  CMD=(
    "$PYTHON_BIN" "$TRAIN_SCRIPT"
    --gpu "$GPU"
    --data_type "$DATA_TYPE"
    --num_iterations "$NUM_ITERATIONS"
    --inner "$INNER"
    --innerdiv "$INNERDIV"
    --meta "$META"
    --K "$K"
    --output_dir "$RUN_DIR"
    --corner "$CORNER"
    --temperature "$TEMP"
    --loss_log_every "$LOSS_LOG_EVERY"
  )

  if [[ "$AUTO_RESUME" == "1" ]]; then
    CMD+=(--auto_resume)
  fi
  if [[ "$UNIT_CONVERT" == "1" ]]; then
    CMD+=(--unit_convert)
  fi
  if [[ "$FREEZE_HIDDEN" == "1" ]]; then
    CMD+=(--freeze_hidden)
  fi
  if [[ "$ENABLE_LOSS_LOGGING" == "1" ]]; then
    CMD+=(--enable_loss_logging)
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  echo "============================================================"
  echo "[RUN ${run_count}/${#SWEEP_PAIRS[@]}] corner=${CORNER}, temp=${TEMP}C"
  echo "[OUT] ${RUN_DIR}"
  echo "[LOG] ${LOG_FILE}"
  echo "[CMD] ${CMD[*]}"
  echo "============================================================"

  set +e
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  status=${PIPESTATUS[0]}
  set -e

  if [[ $status -ne 0 ]]; then
    fail_count=$((fail_count + 1))
    echo "[FAIL] corner=${CORNER}, temp=${TEMP}C, exit_code=${status}" | tee -a "$LOG_FILE"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "[STOP] CONTINUE_ON_ERROR=0 이므로 여기서 종료합니다."
      exit $status
    fi
  else
    echo "[DONE] corner=${CORNER}, temp=${TEMP}C" | tee -a "$LOG_FILE"
  fi

done

echo "============================================================"
echo "Sweep finished"
echo "  total runs : ${run_count}"
echo "  failed runs: ${fail_count}"
echo "  output root: ${OUTPUT_ROOT}"
echo "============================================================"

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
