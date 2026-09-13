#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_deliv_test
#SBATCH --output=v2sx_deliv_test.%j.out
#SBATCH --error=v2sx_deliv_test.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
VIDEO="${VIDEO:-datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_delivery_test}"
RUN_NAME="${RUN_NAME:-SignLanguage_S2}"
RUN="${RUN_ROOT}/${RUN_NAME}"
SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "[check] cwd: $(pwd)"
echo "[check] video: ${VIDEO}"
echo "[check] run output: ${RUN}"
echo "[check] smplx model: ${SMPLX_MODEL}"
nvidia-smi || true

mkdir -p "${RUN_ROOT}"

echo "[1/4] Body estimation"
time ${PYTHON} delivery/source/body_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda \
  --detector-stride 5 \
  --batch-size 8 \
  --inference-mode \
  --single-gpu-model

echo "[2/4] Hand estimation"
time ${PYTHON} delivery/source/hand_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda

echo "[3/4] Face estimation"
time ${PYTHON} delivery/source/face_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda

echo "[4/4] Combine, smooth, render, export NPZ"
time ${PYTHON} delivery/source/combine_smooth_render_cli.py \
  --body-dir "${RUN}/body_params" \
  --hand-dir "${RUN}/hand_params" \
  --face-dir "${RUN}/face_params" \
  --output "${RUN}" \
  --input-video "${VIDEO}" \
  --smplx-model "${SMPLX_MODEL}" \
  --fps 30 \
  --gender neutral

echo "[verify] required outputs"
test -s "${RUN}/smplx_params.npz"
test -s "${RUN}/rendered/smplx_render.mp4"
test -s "${RUN}/side_by_side_input_render.mp4"
test -s "${RUN}/combine_render_report.json"

echo "[done] integrated delivery pipeline test passed"
echo "  smplx params : ${RUN}/smplx_params.npz"
echo "  render       : ${RUN}/rendered/smplx_render.mp4"
echo "  side-by-side : ${RUN}/side_by_side_input_render.mp4"
echo "  report       : ${RUN}/combine_render_report.json"
