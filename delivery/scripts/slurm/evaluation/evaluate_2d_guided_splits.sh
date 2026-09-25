#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J sl_2d_split
#SBATCH --output=sl_2d_split.%j.out
#SBATCH --error=sl_2d_split.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs}"
DATASET_PATH="${DATASET_PATH:-${OUTPUT_ROOT}/evaluation/2d_guided_upperbody_dataset_r1.npz}"
MODEL_PATH="${MODEL_PATH:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"
SPLITS="${SPLITS:-test}"
PARAMS_SUBDIRS="${PARAMS_SUBDIRS:-fused_params_2d_upper_corrected_s0p75}"
EVAL_OUT="${EVAL_OUT:-${OUTPUT_ROOT}/evaluation/split_eval}"

echo "[check] output root: ${OUTPUT_ROOT}"
echo "[check] dataset: ${DATASET_PATH}"
echo "[check] params subdirs: ${PARAMS_SUBDIRS}"
echo "[check] splits: ${SPLITS}"
echo "[check] eval out: ${EVAL_OUT}"
nvidia-smi || true

${PYTHON} evaluate_signlanguage_2d_guided_splits.py \
  --dataset "${DATASET_PATH}" \
  --pred-root "${OUTPUT_ROOT}" \
  --params-subdirs ${PARAMS_SUBDIRS} \
  --model-path "${MODEL_PATH}" \
  --output-root "${EVAL_OUT}" \
  --splits ${SPLITS} \
  --batch-size 16

echo "[done] ${EVAL_OUT}/split_geometry_summary.csv"
