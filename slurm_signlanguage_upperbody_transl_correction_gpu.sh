#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J sl_uptransl
#SBATCH --output=sl_uptransl.%j.out
#SBATCH --error=sl_uptransl.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
PRED_ROOT="${PRED_ROOT:-outputs_fusion_signlanguage_no_stablized}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_upperbody_transl_correction_outputs}"
PKG_DIR="${PKG_DIR:-signlanguage_global_correction_server}"
MODEL_PATH="${MODEL_PATH:-${PKG_DIR}/models/SMPLX_FEMALE.npz}"

# Train on already globally corrected predictions.
INPUT_SUBDIR="${INPUT_SUBDIR:-fused_params_model_global_corrected}"
TARGET_MODE="${TARGET_MODE:-torso_arms}"
ROT_SCALES="${ROT_SCALES:-0.25 0.50 0.75 1.00}"
TRANSL_SCALE="${TRANSL_SCALE:-1.0}"
MAX_ROT_DELTA_DEGREES="${MAX_ROT_DELTA_DEGREES:-60.0}"
MAX_TRANSL_DELTA_M="${MAX_TRANSL_DELTA_M:-1.0}"
TRANSL_WEIGHT="${TRANSL_WEIGHT:-4.0}"

DATASET_PATH="${OUTPUT_ROOT}/evaluation/upperbody_transl_dataset_${TARGET_MODE}_r1.npz"
TRAIN_DIR="${OUTPUT_ROOT}/evaluation/upperbody_transl_model_${TARGET_MODE}_r1"

echo "[check] working directory: $(pwd)"
echo "[check] prediction root: ${PRED_ROOT}"
echo "[check] output root: ${OUTPUT_ROOT}"
echo "[check] input subdir: ${INPUT_SUBDIR}"
echo "[check] target mode: ${TARGET_MODE}"
echo "[check] rot scales: ${ROT_SCALES}"
echo "[check] transl scale: ${TRANSL_SCALE}"
echo "[check] transl weight: ${TRANSL_WEIGHT}"
echo "[check] max rot delta degrees: ${MAX_ROT_DELTA_DEGREES}"
echo "[check] max transl delta m: ${MAX_TRANSL_DELTA_M}"
nvidia-smi || true

echo "[1/3] Build upper-body+translation correction dataset"
time ${PYTHON} "${PKG_DIR}/build_signlanguage_upperbody_transl_correction_dataset.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir "${INPUT_SUBDIR}" \
  --output "${DATASET_PATH}" \
  --feature-preset upper_global \
  --temporal-radius 1 \
  --target "${TARGET_MODE}" \
  --transl-weight "${TRANSL_WEIGHT}"

echo "[2/3] Train upper-body+translation correction model"
time ${PYTHON} "${PKG_DIR}/train_signlanguage_upperbody_transl_correction.py" \
  --dataset "${DATASET_PATH}" \
  --output-dir "${TRAIN_DIR}" \
  --epochs 160 \
  --batch-size 512 \
  --device cuda

echo "[3/3] Apply/evaluate scale sweep"
for ROT_SCALE in ${ROT_SCALES}; do
  SCALE_TAG="${ROT_SCALE/./}"
  OUTPUT_SUBDIR="fused_params_model_global_${TARGET_MODE}_uppertransl_r${SCALE_TAG}_t${TRANSL_SCALE/./}"
  EVAL_DIR="${OUTPUT_ROOT}/evaluation/smplx_female_${OUTPUT_SUBDIR}"

  echo "[apply] target=${TARGET_MODE} rot_scale=${ROT_SCALE} transl_scale=${TRANSL_SCALE} -> ${OUTPUT_SUBDIR}"
  time ${PYTHON} "${PKG_DIR}/apply_signlanguage_upperbody_transl_correction.py" \
    --pred-root "${PRED_ROOT}" \
    --params-subdir "${INPUT_SUBDIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --output-subdir "${OUTPUT_SUBDIR}" \
    --checkpoint "${TRAIN_DIR}/best_model.pt" \
    --rot-scale "${ROT_SCALE}" \
    --transl-scale "${TRANSL_SCALE}" \
    --max-delta-degrees "${MAX_ROT_DELTA_DEGREES}" \
    --max-transl-delta-m "${MAX_TRANSL_DELTA_M}" \
    --device cuda

  echo "[eval] ${OUTPUT_SUBDIR}"
  time ${PYTHON} "${PKG_DIR}/evaluate_signlanguage_geometry.py" \
    --pred-root "${OUTPUT_ROOT}" \
    --params-subdir "${OUTPUT_SUBDIR}" \
    --model-path "${MODEL_PATH}" \
    --batch-size 16 \
    --output-dir "${EVAL_DIR}"

  echo "[all-row] ${OUTPUT_SUBDIR}"
  ${PYTHON} -c "import csv; p='${EVAL_DIR}/geometry_summary.csv'; r=[x for x in csv.DictReader(open(p)) if x['sequence']=='ALL'][0]; keys=['mpjpe_mm_mean','pa_mpjpe_mm_mean','mpvpe_mm_mean','pa_mpvpe_mm_mean','body_mpjpe_mm_mean','body_pa_mpjpe_mm_mean','hand_mpjpe_mm_mean','hand_pa_mpjpe_mm_mean','visible_upper_mpvpe_mm_mean','hands_wrist_mpvpe_mm_mean','hands_pa_mpvpe_mm_mean']; print('${OUTPUT_SUBDIR},' + ','.join(str(r.get(k,'')) for k in keys))"
done

echo "[done] Results:"
echo "  checkpoint: ${TRAIN_DIR}/best_model.pt"
echo "  outputs: ${OUTPUT_ROOT}/SignLanguage_S*/fused_params_model_global_${TARGET_MODE}_uppertransl_*"
echo "  evaluations: ${OUTPUT_ROOT}/evaluation/smplx_female_fused_params_model_global_${TARGET_MODE}_uppertransl_*"
