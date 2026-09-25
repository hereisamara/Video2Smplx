#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --array=0-5
#SBATCH -t 48:00:00
#SBATCH -A lt200246
#SBATCH -J signify_sx
#SBATCH --output=signify_sx.%A_%a.out
#SBATCH --error=signify_sx.%A_%a.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
FRAMES_ROOT="${FRAMES_ROOT:-/project/lt200246-mmacma/khtun/signify/frames}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signify_outputs_smplestx_only}"
LOG_ROOT="${LOG_ROOT:-/project/lt200246-mmacma/khtun/signify_logs_smplestx_only}"
BATCHES="${BATCHES:-6}"
START_FPS="${START_FPS:-30}"

echo "[check] cwd: $(pwd)"
echo "[check] frames root: ${FRAMES_ROOT}"
echo "[check] output root: ${OUTPUT_ROOT}"
echo "[check] log root: ${LOG_ROOT}"
echo "[check] array task: ${SLURM_ARRAY_TASK_ID:-0}/${BATCHES}"
nvidia-smi || true

if [ ! -d "${FRAMES_ROOT}" ]; then
  echo "[error] Missing frames root: ${FRAMES_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
mapfile -t SEQUENCES < <(find "${FRAMES_ROOT}" -mindepth 1 -maxdepth 1 -type d -print | xargs -n 1 basename | sort)

if [ "${#SEQUENCES[@]}" -eq 0 ]; then
  echo "[error] No Signify sequence folders found under ${FRAMES_ROOT}" >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAILURES=0
for INDEX in "${!SEQUENCES[@]}"; do
  if [ $((INDEX % BATCHES)) -ne "${TASK_ID}" ]; then
    continue
  fi
  SEQ="${SEQUENCES[$INDEX]}"
  FRAME_DIR="${FRAMES_ROOT}/${SEQ}"
  OUT_DIR="${OUTPUT_ROOT}/${SEQ}"
  LOG_FILE="${LOG_ROOT}/${SEQ}.log"
  DONE_FILE="${OUT_DIR}/.smplestx_only.done"
  FAILED_FILE="${OUT_DIR}/.smplestx_only.failed"

  if [ -f "${DONE_FILE}" ]; then
    echo "[skip] ${SEQ}: already done"
    continue
  fi

  mkdir -p "${OUT_DIR}"
  rm -f "${FAILED_FILE}"
  echo "======================================"
  echo "Running ${SEQ}"
  echo "Index: ${INDEX}"
  echo "Frames: ${FRAME_DIR}"
  echo "Output: ${OUT_DIR}"
  echo "Log: ${LOG_FILE}"
  echo "======================================"

  set +e
  ${PYTHON} -m video2smplx.integrated_pipeline \
    --video "${SEQ}.mp4" \
    --frames_dir "${FRAME_DIR}" \
    --reuse_frames \
    --output "${OUT_DIR}" \
    --name "${SEQ}" \
    --fps "${START_FPS}" \
    --device cuda \
    --smplestx_only \
    --smplestx_detector_stride 5 \
    --smplestx_batch_size 8 \
    --smplestx_inference_mode \
    --smplestx_single_gpu_model \
    > "${LOG_FILE}" 2>&1
  EXIT_CODE=$?
  set -e

  if [ "${EXIT_CODE}" -eq 0 ]; then
    touch "${DONE_FILE}"
    echo "Finished ${SEQ}"
  else
    echo "Exit code ${EXIT_CODE}" > "${FAILED_FILE}"
    echo "[failed] ${SEQ}: see ${LOG_FILE}" >&2
    FAILURES=$((FAILURES + 1))
  fi
done

if [ "${FAILURES}" -gt 0 ]; then
  echo "Batch ${TASK_ID} finished with ${FAILURES} failures." >&2
  exit 1
fi

echo "Batch ${TASK_ID} done."
