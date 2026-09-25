#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J sl_no2d_upper
#SBATCH --output=sl_no2d_upper.%j.out
#SBATCH --error=sl_no2d_upper.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"

INPUT_ROOT="${INPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs}"
INPUT_SUBDIR="${INPUT_SUBDIR:-fused_params_model_global_hand_corrected}"

GUIDED_OUTPUT_ROOT="${GUIDED_OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_no2d_upperbody_ablation_outputs}"
FULL_DATASET="${FULL_DATASET:-${GUIDED_OUTPUT_ROOT}/evaluation/2d_guided_upperbody_dataset_r1.npz}"
NO2D_DATASET="${NO2D_DATASET:-${OUTPUT_ROOT}/evaluation/upperbody_smplx_only_dataset_r1.npz}"
TRAIN_DIR="${TRAIN_DIR:-${OUTPUT_ROOT}/evaluation/upperbody_smplx_only_model_r1}"
MODEL_PATH="${MODEL_PATH:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"
APPLY_SCALES="${APPLY_SCALES:-0.50 0.75 1.00}"
SEQUENCES="${SEQUENCES:-all}"
SPLITS="${SPLITS:-test}"

echo "[check] input root: ${INPUT_ROOT}"
echo "[check] input subdir: ${INPUT_SUBDIR}"
echo "[check] guided output root: ${GUIDED_OUTPUT_ROOT}"
echo "[check] no-2d output root: ${OUTPUT_ROOT}"
echo "[check] full dataset: ${FULL_DATASET}"
echo "[check] no-2d dataset: ${NO2D_DATASET}"
echo "[check] scales: ${APPLY_SCALES}"
nvidia-smi || true

mkdir -p "${OUTPUT_ROOT}/evaluation"

echo "[1/5] Build SMPL-X-only feature dataset from existing 2D-guided dataset"
time ${PYTHON} transform_signlanguage_feature_ablation_dataset.py \
  --input "${FULL_DATASET}" \
  --output "${NO2D_DATASET}" \
  --mode smplx_only

echo "[2/5] Train SMPL-X-only upper-body corrector"
time ${PYTHON} train_signlanguage_2d_guided_upperbody_corrector.py \
  --dataset "${NO2D_DATASET}" \
  --output-dir "${TRAIN_DIR}" \
  --epochs 180 \
  --batch-size 512 \
  --device cuda

echo "[3/5] Apply SMPL-X-only correction scales"
for SCALE in ${APPLY_SCALES}; do
  SCALE_TAG=$(echo "${SCALE}" | tr '.' 'p')
  OUTPUT_SUBDIR="fused_params_no2d_upper_corrected_s${SCALE_TAG}"
  echo "[scale ${SCALE}] apply -> ${OUTPUT_SUBDIR}"
  time ${PYTHON} apply_signlanguage_2d_guided_upperbody_corrector.py \
    --pred-root "${INPUT_ROOT}" \
    --params-subdir "${INPUT_SUBDIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --output-subdir "${OUTPUT_SUBDIR}" \
    --checkpoint "${TRAIN_DIR}/best_model.pt" \
    --model-path "${MODEL_PATH}" \
    --sequences ${SEQUENCES} \
    --device cuda \
    --scale "${SCALE}"
done

echo "[4/5] Evaluate no-2D vs 2D-guided on the same held-out split"
time ${PYTHON} evaluate_signlanguage_ablation.py \
  --dataset "${FULL_DATASET}" \
  --model-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}/evaluation/no2d_vs_2d_guided_ablation" \
  --splits ${SPLITS} \
  --baseline "Global + hand corrector" \
  --skip-missing \
  --method "Global + hand corrector|${INPUT_ROOT}|${INPUT_SUBDIR}" \
  --method "No 2D upper scale 0.50|${OUTPUT_ROOT}|fused_params_no2d_upper_corrected_s0p50" \
  --method "No 2D upper scale 0.75|${OUTPUT_ROOT}|fused_params_no2d_upper_corrected_s0p75" \
  --method "No 2D upper scale 1.00|${OUTPUT_ROOT}|fused_params_no2d_upper_corrected_s1p00" \
  --method "2D upper scale 0.50|${GUIDED_OUTPUT_ROOT}|fused_params_2d_upper_corrected_s0p50" \
  --method "2D upper scale 0.75|${GUIDED_OUTPUT_ROOT}|fused_params_2d_upper_corrected_s0p75" \
  --method "2D upper scale 1.00|${GUIDED_OUTPUT_ROOT}|fused_params_2d_upper_corrected_s1p00" \
  --batch-size 16

echo "[5/5] Done"
echo "  no-2d dataset: ${NO2D_DATASET}"
echo "  no-2d model: ${TRAIN_DIR}/best_model.pt"
echo "  corrected params: ${OUTPUT_ROOT}/SignLanguage_S*/fused_params_no2d_upper_corrected_s*"
echo "  comparison: ${OUTPUT_ROOT}/evaluation/no2d_vs_2d_guided_ablation/ablation_summary.md"
