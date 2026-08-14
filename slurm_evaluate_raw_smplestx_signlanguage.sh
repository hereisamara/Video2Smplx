#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J sl_raw_sx_eval
#SBATCH --output=sl_raw_sx_eval.%j.out
#SBATCH --error=sl_raw_sx_eval.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
PRED_ROOT="${PRED_ROOT:-outputs_fusion_signlanguage_no_stablized}"
PARAMS_SUBDIR="${PARAMS_SUBDIR:-smplestx_params}"
PKG_DIR="${PKG_DIR:-signlanguage_global_correction_server}"
MODEL_PATH="${MODEL_PATH:-${PKG_DIR}/models/SMPLX_FEMALE.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${PRED_ROOT}/evaluation/smplx_female_raw_smplestx}"

echo "[check] working directory: $(pwd)"
echo "[check] prediction root: ${PRED_ROOT}"
echo "[check] params subdir: ${PARAMS_SUBDIR}"
echo "[check] model path: ${MODEL_PATH}"
echo "[check] output dir: ${OUTPUT_DIR}"

FOUND_COUNT=$(find "${PRED_ROOT}" -path "*/${PARAMS_SUBDIR}/*_params.pkl" | head -100 | wc -l)
echo "[check] first-100 raw SMPLest-X PKL count: ${FOUND_COUNT}"
if [ "${FOUND_COUNT}" -eq 0 ]; then
  echo "[error] No ${PARAMS_SUBDIR} PKLs found. Rerun integrated pipeline with --save_raw_smplestx first." >&2
  exit 2
fi

time ${PYTHON} "${PKG_DIR}/evaluate_signlanguage_geometry.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir "${PARAMS_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --batch-size 16 \
  --output-dir "${OUTPUT_DIR}"

echo "[all-row]"
${PYTHON} -c "import csv; p='${OUTPUT_DIR}/geometry_summary.csv'; r=[x for x in csv.DictReader(open(p)) if x['sequence']=='ALL'][0]; keys=['mpjpe_mm_mean','pa_mpjpe_mm_mean','mpvpe_mm_mean','pa_mpvpe_mm_mean','body_mpjpe_mm_mean','body_pa_mpjpe_mm_mean','hand_mpjpe_mm_mean','hand_pa_mpjpe_mm_mean','visible_upper_mpvpe_mm_mean','hands_wrist_mpvpe_mm_mean','hands_pa_mpvpe_mm_mean','face_head_mpvpe_mm_mean','face_pa_mpvpe_mm_mean']; print(','.join([r['sequence'], r['frames']] + [str(r.get(k,'')) for k in keys]))"

echo "[done] evaluation: ${OUTPUT_DIR}"
