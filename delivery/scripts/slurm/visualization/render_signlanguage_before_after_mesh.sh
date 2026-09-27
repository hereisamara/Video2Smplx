#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -A lt200246
#SBATCH -J sl_mesh_vis
#SBATCH --output=sl_mesh_vis.%j.out
#SBATCH --error=sl_mesh_vis.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"

VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
ANNOTATION_DIR="${ANNOTATION_DIR:-datasets/annotations/SignLanguage}"
MODEL_PATH="${MODEL_PATH:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_FEMALE.npz}"

BEFORE_ROOT="${BEFORE_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs}"
BEFORE_SUBDIR="${BEFORE_SUBDIR:-fused_params_model_global_hand_corrected}"
AFTER_ROOT="${AFTER_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs}"
AFTER_SUBDIR="${AFTER_SUBDIR:-fused_params_2d_upper_corrected_s0p75}"

OUTPUT_DIR="${OUTPUT_DIR:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/visualizations/before_after_mesh_random3}"
SEQUENCES="${SEQUENCES:-random}"
SAMPLE_COUNT="${SAMPLE_COUNT:-3}"
RANDOM_SEED="${RANDOM_SEED:-7}"
MAX_FRAMES="${MAX_FRAMES:-120}"
STRIDE="${STRIDE:-2}"
PANEL_SIZE="${PANEL_SIZE:-260}"

echo "[check] cwd: $(pwd)"
echo "[check] before: ${BEFORE_ROOT}/SignLanguage_S*/${BEFORE_SUBDIR}"
echo "[check] after: ${AFTER_ROOT}/SignLanguage_S*/${AFTER_SUBDIR}"
echo "[check] output: ${OUTPUT_DIR}"
echo "[check] sequences: ${SEQUENCES}"
echo "[check] sample count: ${SAMPLE_COUNT}"
echo "[check] seed: ${RANDOM_SEED}"

time ${PYTHON} -m tools.visualization.render_signlanguage_before_after_mesh_gif \
  --video-dir "${VIDEO_DIR}" \
  --annotation-dir "${ANNOTATION_DIR}" \
  --before-root "${BEFORE_ROOT}" \
  --before-subdir "${BEFORE_SUBDIR}" \
  --after-root "${AFTER_ROOT}" \
  --after-subdir "${AFTER_SUBDIR}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --sequences ${SEQUENCES} \
  --sample-count "${SAMPLE_COUNT}" \
  --random-seed "${RANDOM_SEED}" \
  --max-frames "${MAX_FRAMES}" \
  --stride "${STRIDE}" \
  --panel-size "${PANEL_SIZE}" \
  --view front \
  --alignment root \
  --before-label "${BEFORE_SUBDIR}" \
  --after-label "${AFTER_SUBDIR}" \
  --tag before_after_mesh

echo "[done] outputs:"
find "${OUTPUT_DIR}" -maxdepth 2 \( -name '*.gif' -o -name '*samples.png' -o -name '*report.json' \) -print
