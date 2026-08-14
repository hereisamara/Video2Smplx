#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J sl_upper_corr
#SBATCH --output=sl_upper_corr.%j.out
#SBATCH --error=sl_upper_corr.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
PRED_ROOT="${PRED_ROOT:-outputs_fusion_signlanguage_no_stablized}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_upperbody_correction_outputs}"
PKG_DIR="${PKG_DIR:-signlanguage_global_correction_server}"
MODEL_PATH="${MODEL_PATH:-${PKG_DIR}/models/SMPLX_FEMALE.npz}"

# This stage expects the already globally corrected outputs.
INPUT_SUBDIR="${INPUT_SUBDIR:-fused_params_model_global_corrected}"
UPPER_SUBDIR="${UPPER_SUBDIR:-fused_params_model_global_upper_corrected}"
FINAL_SUBDIR="${FINAL_SUBDIR:-fused_params_model_global_upper_hand_corrected}"
UPPER_SCALE="${UPPER_SCALE:-1.0}"
UPPER_MAX_DELTA_DEGREES="${UPPER_MAX_DELTA_DEGREES:-60.0}"
HAND_SCALE="${HAND_SCALE:-1.0}"

DATASET_PATH="${OUTPUT_ROOT}/evaluation/upperbody_correction_dataset_upper_body_r1.npz"
TRAIN_DIR="${OUTPUT_ROOT}/evaluation/upperbody_correction_model_upper_body_r1"
HAND_CHECKPOINT="${HAND_CHECKPOINT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt}"

echo "[check] working directory: $(pwd)"
echo "[check] prediction root: ${PRED_ROOT}"
echo "[check] output root: ${OUTPUT_ROOT}"
echo "[check] input subdir: ${INPUT_SUBDIR}"
echo "[check] upper output subdir: ${UPPER_SUBDIR}"
echo "[check] final output subdir: ${FINAL_SUBDIR}"
echo "[check] upper scale: ${UPPER_SCALE}"
echo "[check] upper max delta degrees: ${UPPER_MAX_DELTA_DEGREES}"
echo "[check] hand scale: ${HAND_SCALE}"
echo "[check] hand checkpoint: ${HAND_CHECKPOINT}"
nvidia-smi || true

echo "[1/6] Build upper-body correction dataset"
time ${PYTHON} "${PKG_DIR}/build_signlanguage_upperbody_correction_dataset.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir "${INPUT_SUBDIR}" \
  --output "${DATASET_PATH}" \
  --feature-preset upper_global \
  --temporal-radius 1 \
  --target upper_body

echo "[2/6] Train upper-body correction model"
time ${PYTHON} "${PKG_DIR}/train_signlanguage_upperbody_correction.py" \
  --dataset "${DATASET_PATH}" \
  --output-dir "${TRAIN_DIR}" \
  --epochs 160 \
  --batch-size 512 \
  --device cuda

echo "[3/6] Apply upper-body correction model"
time ${PYTHON} "${PKG_DIR}/apply_signlanguage_upperbody_correction.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir "${INPUT_SUBDIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-subdir "${UPPER_SUBDIR}" \
  --checkpoint "${TRAIN_DIR}/best_model.pt" \
  --scale "${UPPER_SCALE}" \
  --max-delta-degrees "${UPPER_MAX_DELTA_DEGREES}" \
  --device cuda

echo "[4/6] Evaluate global+upper corrected predictions"
time ${PYTHON} "${PKG_DIR}/evaluate_signlanguage_geometry.py" \
  --pred-root "${OUTPUT_ROOT}" \
  --params-subdir "${UPPER_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --batch-size 16 \
  --output-dir "${OUTPUT_ROOT}/evaluation/smplx_female_model_global_upper_corrected"

echo "[5/6] Apply existing hand correction on top of upper-body correction"
time ${PYTHON} "${PKG_DIR}/apply_signlanguage_hand_correction.py" \
  --pred-root "${OUTPUT_ROOT}" \
  --params-subdir "${UPPER_SUBDIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-subdir "${FINAL_SUBDIR}" \
  --checkpoint "${HAND_CHECKPOINT}" \
  --scale "${HAND_SCALE}" \
  --device cuda

echo "[6/6] Evaluate global+upper+hand corrected predictions"
time ${PYTHON} "${PKG_DIR}/evaluate_signlanguage_geometry.py" \
  --pred-root "${OUTPUT_ROOT}" \
  --params-subdir "${FINAL_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --batch-size 16 \
  --output-dir "${OUTPUT_ROOT}/evaluation/smplx_female_model_global_upper_hand_corrected"

echo "[done] Results:"
echo "  upper checkpoint: ${TRAIN_DIR}/best_model.pt"
echo "  upper corrected params: ${OUTPUT_ROOT}/SignLanguage_S*/${UPPER_SUBDIR}"
echo "  final corrected params: ${OUTPUT_ROOT}/SignLanguage_S*/${FINAL_SUBDIR}"
echo "  upper evaluation: ${OUTPUT_ROOT}/evaluation/smplx_female_model_global_upper_corrected"
echo "  final evaluation: ${OUTPUT_ROOT}/evaluation/smplx_female_model_global_upper_hand_corrected"
