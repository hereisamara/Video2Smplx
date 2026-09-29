#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_config
#SBATCH --output=v2sx_config.%j.out
#SBATCH --error=v2sx_config.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCE="${SEQUENCE:-SignLanguage_S2}"
VIDEO="${VIDEO:-datasets/videos/SignLanguage/${SEQUENCE}/${SEQUENCE}.mp4}"
OUTPUT="${OUTPUT:-/project/lt200246-mmacma/khtun/video2smplx_configurable/${SEQUENCE}}"
EXECUTION_MODE="${EXECUTION_MODE:-realtime}"
MAX_INFLIGHT_FRAMES="${MAX_INFLIGHT_FRAMES:-2}"
SMPLESTX_ONLY="${SMPLESTX_ONLY:-0}"
ENABLE_WILOR="${ENABLE_WILOR:-1}"
ENABLE_EMOCA="${ENABLE_EMOCA:-1}"
POSTPROCESS_MODE="${POSTPROCESS_MODE:-accurate}"
ENABLE_SMOOTHING="${ENABLE_SMOOTHING:-1}"
ZERO_TRANSLATION="${ZERO_TRANSLATION:-1}"
ENABLE_RENDER="${ENABLE_RENDER:-0}"
SAVE_FINAL_PKLS="${SAVE_FINAL_PKLS:-0}"
SMPLESTX_BATCH_SIZE="${SMPLESTX_BATCH_SIZE:-8}"
WILOR_BATCH_SIZE="${WILOR_BATCH_SIZE:-16}"
EMOCA_BATCH_SIZE="${EMOCA_BATCH_SIZE:-16}"
WILOR_PARAMS_ONLY="${WILOR_PARAMS_ONLY:-0}"
EMOCA_EXPRESSION_ONLY="${EMOCA_EXPRESSION_ONLY:-0}"
WARMUP_EXCLUDE_FRAMES="${WARMUP_EXCLUDE_FRAMES:-2}"
WARMUP_EXCLUDE_BATCHES="${WARMUP_EXCLUDE_BATCHES:-1}"
YOLO_KEYPOINTS="${YOLO_KEYPOINTS:-/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json}"

if [ ! -s "${VIDEO}" ]; then
  echo "[error] missing video: ${VIDEO}" >&2
  exit 2
fi

ARGS=(
  --video "${VIDEO}"
  --output "${OUTPUT}"
  --name "${SEQUENCE}"
  --sequence "${SEQUENCE}"
  --device cuda
  --execution-mode "${EXECUTION_MODE}"
  --max-inflight-frames "${MAX_INFLIGHT_FRAMES}"
  --smplestx-detector-stride 5
  --smplestx-batch-size "${SMPLESTX_BATCH_SIZE}"
  --wilor-batch-size "${WILOR_BATCH_SIZE}"
  --emoca-batch-size "${EMOCA_BATCH_SIZE}"
  --warmup-exclude-frames "${WARMUP_EXCLUDE_FRAMES}"
  --warmup-exclude-batches "${WARMUP_EXCLUDE_BATCHES}"
  --smplestx-inference-mode
  --smplestx-single-gpu-model
  --fast-io
)

if [ "${SMPLESTX_ONLY}" = "1" ]; then
  ARGS+=(--smplestx-only)
else
  if [ "${ENABLE_WILOR}" != "1" ]; then
    ARGS+=(--disable-wilor)
  fi
  if [ "${ENABLE_EMOCA}" != "1" ]; then
    ARGS+=(--disable-emoca)
  fi
  if [ "${WILOR_PARAMS_ONLY}" = "1" ]; then
    ARGS+=(--wilor-params-only)
  fi
  if [ "${EMOCA_EXPRESSION_ONLY}" = "1" ]; then
    ARGS+=(--emoca-expression-only)
  fi
  if [ "${POSTPROCESS_MODE}" = "none" ]; then
    ARGS+=(--disable-postprocessing)
  else
    ARGS+=(--postprocess-mode "${POSTPROCESS_MODE}")
    if [ "${POSTPROCESS_MODE}" = "accurate" ]; then
      if [ ! -s "${YOLO_KEYPOINTS}" ]; then
        echo "[error] accurate mode needs keypoints: ${YOLO_KEYPOINTS}" >&2
        exit 2
      fi
      ARGS+=(--precomputed-yolo-keypoints "${YOLO_KEYPOINTS}")
    fi
  fi
  if [ "${ENABLE_SMOOTHING}" != "1" ]; then
    ARGS+=(--disable-smoothing)
  fi
  if [ "${ZERO_TRANSLATION}" != "1" ]; then
    ARGS+=(--disable-zero-translation)
  fi
fi

if [ "${ENABLE_RENDER}" != "1" ]; then
  ARGS+=(--skip-render)
fi
if [ "${SAVE_FINAL_PKLS}" = "1" ]; then
  ARGS+=(--save-final-pkls)
fi

echo "[config] sequence=${SEQUENCE}"
echo "[config] video=${VIDEO}"
echo "[config] output=${OUTPUT}"
echo "[config] execution=${EXECUTION_MODE} inflight=${MAX_INFLIGHT_FRAMES}"
echo "[config] smplestx_only=${SMPLESTX_ONLY} wilor=${ENABLE_WILOR} emoca=${ENABLE_EMOCA}"
echo "[config] wilor_params_only=${WILOR_PARAMS_ONLY} emoca_expression_only=${EMOCA_EXPRESSION_ONLY}"
echo "[config] postprocess=${POSTPROCESS_MODE} smoothing=${ENABLE_SMOOTHING} zero_translation=${ZERO_TRANSLATION}"
echo "[config] render=${ENABLE_RENDER} save_final_pkls=${SAVE_FINAL_PKLS}"
echo "[config] warmup_exclude_frames=${WARMUP_EXCLUDE_FRAMES} warmup_exclude_batches=${WARMUP_EXCLUDE_BATCHES}"
nvidia-smi || true

"${PYTHON[@]}" delivery/source/integrated_postprocessed_pipeline_cli.py "${ARGS[@]}"

echo "[done] runtime report: ${OUTPUT}/runtime/integrated_postprocessed_runtime_report.json"
find "${OUTPUT}" -name smplx_params.npz -print
