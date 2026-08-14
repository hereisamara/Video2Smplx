#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J sl_hand_corr
#SBATCH --output=sl_hand_corr.%j.out
#SBATCH --error=sl_hand_corr.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
PRED_ROOT="outputs_fusion_signlanguage_no_stablized"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs}"
PKG_DIR="signlanguage_global_correction_server"
MODEL_PATH="${PKG_DIR}/models/SMPLX_FEMALE.npz"

# Use globally corrected params as the hand-correction input.
INPUT_SUBDIR="fused_params_model_global_corrected"
HAND_SUBDIR="fused_params_model_global_hand_corrected"
DATASET_PATH="${OUTPUT_ROOT}/evaluation/hand_correction_dataset_wrist_fingers_r1.npz"
TRAIN_DIR="${OUTPUT_ROOT}/evaluation/hand_correction_model_wrist_fingers_r1"

echo "[check] working directory: $(pwd)"
echo "[check] prediction root: ${PRED_ROOT}"
echo "[check] output root: ${OUTPUT_ROOT}"
echo "[check] input subdir: ${INPUT_SUBDIR}"
echo "[check] hand output subdir: ${HAND_SUBDIR}"
nvidia-smi || true

echo "[1/4] Build hand correction dataset"
${PYTHON} "${PKG_DIR}/build_signlanguage_hand_correction_dataset.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir "${INPUT_SUBDIR}" \
  --output "${DATASET_PATH}" \
  --feature-preset upper_global \
  --temporal-radius 1 \
  --target wrist_fingers

echo "[2/4] Train hand correction model"
${PYTHON} "${PKG_DIR}/train_signlanguage_hand_correction.py" \
  --dataset "${DATASET_PATH}" \
  --output-dir "${TRAIN_DIR}" \
  --epochs 160 \
  --batch-size 512 \
  --device cuda

echo "[3/4] Apply hand correction model"
${PYTHON} "${PKG_DIR}/apply_signlanguage_hand_correction.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir "${INPUT_SUBDIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-subdir "${HAND_SUBDIR}" \
  --checkpoint "${TRAIN_DIR}/best_model.pt" \
  --device cuda

echo "[4/4] Evaluate global+hand corrected predictions"
${PYTHON} "${PKG_DIR}/evaluate_signlanguage_geometry.py" \
  --pred-root "${OUTPUT_ROOT}" \
  --params-subdir "${HAND_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --batch-size 16 \
  --output-dir "${OUTPUT_ROOT}/evaluation/smplx_female_model_global_hand_corrected"

echo "[done] Results:"
echo "  hand checkpoint: ${TRAIN_DIR}/best_model.pt"
echo "  corrected params: ${OUTPUT_ROOT}/SignLanguage_S*/${HAND_SUBDIR}"
echo "  evaluation: ${OUTPUT_ROOT}/evaluation/smplx_female_model_global_hand_corrected"
