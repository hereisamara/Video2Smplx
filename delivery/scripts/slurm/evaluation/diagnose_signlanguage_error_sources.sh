#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J sl_err_src
#SBATCH --output=sl_err_src.%j.out
#SBATCH --error=sl_err_src.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"

# Default: diagnose the latest global+hand corrected predictions.
# Override these at sbatch time to diagnose raw/fused/global-only outputs.
PRED_ROOT="${PRED_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs}"
PARAMS_SUBDIR="${PARAMS_SUBDIR:-fused_params_model_global_hand_corrected}"
MODEL_PATH="${MODEL_PATH:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_FEMALE.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PRED_ROOT}/evaluation/error_sources_${PARAMS_SUBDIR}}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
GROUPS="${GROUPS:-all}"

echo "[check] working directory: $(pwd)"
echo "[check] pred root: ${PRED_ROOT}"
echo "[check] params subdir: ${PARAMS_SUBDIR}"
echo "[check] model path: ${MODEL_PATH}"
echo "[check] output dir: ${OUTPUT_DIR}"
echo "[check] frame stride: ${FRAME_STRIDE}"
echo "[check] max frames: ${MAX_FRAMES}"
echo "[check] groups: ${GROUPS}"
nvidia-smi || true

time ${PYTHON} -m tools.diagnostics.diagnose_signlanguage_error_sources \
  --pred-root "${PRED_ROOT}" \
  --params-subdir "${PARAMS_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --frame-stride "${FRAME_STRIDE}" \
  --max-frames "${MAX_FRAMES}" \
  --groups ${GROUPS} \
  --output-dir "${OUTPUT_DIR}"

echo "[top counterfactuals]"
${PYTHON} - <<PY
import csv
p = "${OUTPUT_DIR}/counterfactual_group_summary.csv"
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
keys = [
    "group",
    "visible_upper_mpvpe_mm_before_mean",
    "visible_upper_mpvpe_mm_after_mean",
    "visible_upper_mpvpe_mm_improvement_mean",
    "mpvpe_mm_improvement_mean",
    "hands_wrist_mpvpe_mm_improvement_mean",
]
print(",".join(keys))
for r in rows[:20]:
    print(",".join(r[k] for k in keys))
PY

echo "[top joints]"
head -20 "${OUTPUT_DIR}/joint_error_summary.csv"

echo "[top parameters]"
head -25 "${OUTPUT_DIR}/parameter_error_summary.csv"

echo "[done] Diagnostic written to ${OUTPUT_DIR}"
