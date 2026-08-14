#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 120:00:00
#SBATCH -A lt200246
#SBATCH -J sl_global_corr
#SBATCH --output=sl_global_corr.%j.out
#SBATCH --error=sl_global_corr.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
PRED_ROOT="outputs_fusion_signlanguage_no_stablized"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PRED_ROOT}}"
PKG_DIR="signlanguage_global_correction_server"
MODEL_PATH="${PKG_DIR}/models/SMPLX_FEMALE.npz"

ORACLE_SUBDIR="fused_params_oracle_upper_body_only_frame"
MODEL_SUBDIR="fused_params_model_global_corrected"
DATASET_PATH="${PRED_ROOT}/evaluation/global_correction_dataset_upper_r1.npz"
TRAIN_DIR="${PRED_ROOT}/evaluation/global_correction_model_upper_r1"

echo "[check] working directory: $(pwd)"
echo "[check] prediction root: ${PRED_ROOT}"
echo "[check] output root: ${OUTPUT_ROOT}"
echo "[check] model path: ${MODEL_PATH}"
nvidia-smi || true

echo "[1/5] Create oracle upper-body correction labels"
${PYTHON} "${PKG_DIR}/oracle_optimize_signlanguage_global.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir fused_params \
  --output-subdir "${ORACLE_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --fit-scope frame \
  --joint-set upper_body \
  --optimize both \
  --translation-mode upper-centroid \
  --fast-labels-only \
  --batch-size 16 \
  --output-dir "${PRED_ROOT}/evaluation/oracle_upper_body_only_frame_female"

echo "[2/5] Build correction training dataset"
${PYTHON} "${PKG_DIR}/build_signlanguage_global_correction_dataset.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir fused_params \
  --oracle-subdir "${ORACLE_SUBDIR}" \
  --output "${DATASET_PATH}" \
  --feature-preset upper_global \
  --temporal-radius 1

echo "[3/5] Train correction model"
${PYTHON} "${PKG_DIR}/train_signlanguage_global_correction.py" \
  --dataset "${DATASET_PATH}" \
  --output-dir "${TRAIN_DIR}" \
  --epochs 120 \
  --batch-size 512 \
  --device cuda

echo "[4/5] Apply trained correction model"
${PYTHON} "${PKG_DIR}/apply_signlanguage_global_correction.py" \
  --pred-root "${PRED_ROOT}" \
  --params-subdir fused_params \
  --output-root "${OUTPUT_ROOT}" \
  --output-subdir "${MODEL_SUBDIR}" \
  --checkpoint "${TRAIN_DIR}/best_model.pt" \
  --device cuda

echo "[5/5] Evaluate corrected predictions"
${PYTHON} "${PKG_DIR}/evaluate_signlanguage_geometry.py" \
  --pred-root "${OUTPUT_ROOT}" \
  --params-subdir "${MODEL_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --batch-size 16 \
  --output-dir "${OUTPUT_ROOT}/evaluation/smplx_female_model_global_corrected"

echo "[done] Results:"
echo "  model checkpoint: ${TRAIN_DIR}/best_model.pt"
echo "  corrected params: ${OUTPUT_ROOT}/SignLanguage_S*/${MODEL_SUBDIR}"
echo "  evaluation: ${OUTPUT_ROOT}/evaluation/smplx_female_model_global_corrected"
