#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J sl_ablation
#SBATCH --output=sl_ablation.%j.out
#SBATCH --error=sl_ablation.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"

DATASET_PATH="${DATASET_PATH:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_dataset_r1.npz}"
MODEL_PATH="${MODEL_PATH:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/ablation_study}"
SPLITS="${SPLITS:-test}"
BASELINE="${BASELINE:-Raw SMPLest-X}"

RAW_SMPLESTX_ROOT="${RAW_SMPLESTX_ROOT:-/project/lt200246-mmacma/khtun/outputs_smplestx_only_signlanguage}"
FUSION_ROOT="${FUSION_ROOT:-outputs_fusion_signlanguage_no_stablized}"
GLOBAL_ROOT="${GLOBAL_ROOT:-outputs_fusion_signlanguage_no_stablized}"
HAND_ROOT="${HAND_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs}"
UPPER2D_ROOT="${UPPER2D_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs}"

echo "[check] dataset: ${DATASET_PATH}"
echo "[check] model: ${MODEL_PATH}"
echo "[check] output: ${OUTPUT_ROOT}"
echo "[check] splits: ${SPLITS}"
echo "[check] baseline: ${BASELINE}"
nvidia-smi || true

time ${PYTHON} -m tools.evaluation.evaluate_signlanguage_ablation \
  --dataset "${DATASET_PATH}" \
  --model-path "${MODEL_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --splits ${SPLITS} \
  --baseline "${BASELINE}" \
  --skip-missing \
  --method "Raw SMPLest-X|${RAW_SMPLESTX_ROOT}|smplestx_params" \
  --method "Fusion no stabilization|${FUSION_ROOT}|fused_params" \
  --method "Global corrector|${GLOBAL_ROOT}|fused_params_model_global_corrected" \
  --method "Global + hand corrector|${HAND_ROOT}|fused_params_model_global_hand_corrected" \
  --method "2D upper scale 0.50|${UPPER2D_ROOT}|fused_params_2d_upper_corrected_s0p50" \
  --method "2D upper scale 0.75|${UPPER2D_ROOT}|fused_params_2d_upper_corrected_s0p75" \
  --method "2D upper scale 1.00|${UPPER2D_ROOT}|fused_params_2d_upper_corrected_s1p00" \
  --batch-size 16

echo "[done] ablation outputs:"
echo "  ${OUTPUT_ROOT}/ablation_summary.csv"
echo "  ${OUTPUT_ROOT}/ablation_summary.md"
echo "  ${OUTPUT_ROOT}/skipped.json"
