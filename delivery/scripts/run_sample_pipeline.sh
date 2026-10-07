#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

VIDEO="${1:-datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4}"
RUN="${2:-runs_delivery/SignLanguage_S2}"
SMPLX_MODEL="${3:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

python delivery/source/body_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda \
  --detector-stride 5 \
  --batch-size 8 \
  --inference-mode \
  --single-gpu-model

python delivery/source/hand_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda

python delivery/source/face_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda

python delivery/source/combine_smooth_render_cli.py \
  --body-dir "${RUN}/body_params" \
  --hand-dir "${RUN}/hand_params" \
  --face-dir "${RUN}/face_params" \
  --output "${RUN}" \
  --input-video "${VIDEO}" \
  --smplx-model "${SMPLX_MODEL}" \
  --fps 30 \
  --gender neutral

echo "[done] smplx params: ${RUN}/smplx_params.npz"
echo "[done] render: ${RUN}/rendered/smplx_render.mp4"
echo "[done] side-by-side: ${RUN}/side_by_side_input_render.mp4"
