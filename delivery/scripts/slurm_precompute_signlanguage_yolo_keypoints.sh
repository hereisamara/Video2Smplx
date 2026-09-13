#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J sl_yolo2d
#SBATCH --output=sl_yolo2d.%j.out
#SBATCH --error=sl_yolo2d.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCES="${SEQUENCES:-SignLanguage_S2 SignLanguage_S3 SignLanguage_S4}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark}"
YOLO_MODEL="${YOLO_MODEL:-yolov8x-pose.pt}"
YOLO_OUTPUT="${YOLO_OUTPUT:-${RUN_ROOT}/evaluation/precomputed_yolo_pose_keypoints.json}"
YOLO_IMGSZ="${YOLO_IMGSZ:-960}"
YOLO_CONF="${YOLO_CONF:-0.25}"

echo "[check] cwd: $(pwd)"
echo "[check] sequences: ${SEQUENCES}"
echo "[check] video dir: ${VIDEO_DIR}"
echo "[check] yolo model: ${YOLO_MODEL}"
echo "[check] output: ${YOLO_OUTPUT}"
nvidia-smi || true

if [ ! -s "${YOLO_MODEL}" ]; then
  echo "[error] missing YOLO model: ${YOLO_MODEL}" >&2
  exit 2
fi

mkdir -p "$(dirname "${YOLO_OUTPUT}")"

time "${PYTHON[@]}" extract_signlanguage_yolo_pose_keypoints.py \
  --video-dir "${VIDEO_DIR}" \
  --output "${YOLO_OUTPUT}" \
  --model "${YOLO_MODEL}" \
  --sequences ${SEQUENCES} \
  --device cuda \
  --imgsz "${YOLO_IMGSZ}" \
  --conf "${YOLO_CONF}"

echo "[done] precomputed keypoints: ${YOLO_OUTPUT}"
