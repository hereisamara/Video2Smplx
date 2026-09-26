#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 12:00:00
#SBATCH -A lt200246
#SBATCH -J signify_eval
#SBATCH --output=signify_eval.%j.out
#SBATCH --error=signify_eval.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"

FRAMES_ROOT="${FRAMES_ROOT:-/project/lt200246-mmacma/khtun/signify/frames}"
GT_ROOT="${GT_ROOT:-/project/lt200246-mmacma/khtun/signify/smplx_gt}"
PRED_ROOT="${PRED_ROOT:-/project/lt200246-mmacma/khtun/signify_outputs_smplestx_only}"
PARAMS_SUBDIR="${PARAMS_SUBDIR:-smplestx_params}"
MODEL_PATH="${MODEL_PATH:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PRED_ROOT}/evaluation/signify_smplx_female_${PARAMS_SUBDIR}}"
SEQUENCES="${SEQUENCES:-all}"
GT_FRAME_MULTIPLIER="${GT_FRAME_MULTIPLIER:-2.0}"
GT_FRAME_OFFSET="${GT_FRAME_OFFSET:-0.0}"

echo "[check] frames root: ${FRAMES_ROOT}"
echo "[check] gt root: ${GT_ROOT}"
echo "[check] pred root: ${PRED_ROOT}"
echo "[check] params subdir: ${PARAMS_SUBDIR}"
echo "[check] model path: ${MODEL_PATH}"
echo "[check] output dir: ${OUTPUT_DIR}"
echo "[check] sequence selector: ${SEQUENCES}"

time ${PYTHON} -m tools.evaluation.evaluate_signify_geometry \
  --frames-root "${FRAMES_ROOT}" \
  --gt-root "${GT_ROOT}" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir "${PARAMS_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --sequences ${SEQUENCES} \
  --gt-frame-multiplier "${GT_FRAME_MULTIPLIER}" \
  --gt-frame-offset "${GT_FRAME_OFFSET}" \
  --batch-size 16

echo "[all-row]"
${PYTHON} - <<PY
import csv
p = "${OUTPUT_DIR}/geometry_summary.csv"
row = [r for r in csv.DictReader(open(p)) if r["sequence"] == "ALL"][0]
keys = [
    "sequence",
    "frames",
    "mpjpe_mm_mean",
    "pa_mpjpe_mm_mean",
    "mpvpe_mm_mean",
    "pa_mpvpe_mm_mean",
    "body_mpjpe_mm_mean",
    "body_pa_mpjpe_mm_mean",
    "hand_mpjpe_mm_mean",
    "hand_pa_mpjpe_mm_mean",
    "visible_upper_mpvpe_mm_mean",
    "hands_wrist_mpvpe_mm_mean",
    "hands_pa_mpvpe_mm_mean",
    "face_head_mpvpe_mm_mean",
    "face_pa_mpvpe_mm_mean",
]
print(",".join(row.get(k, "") for k in keys))
PY

echo "[done] ${OUTPUT_DIR}"
