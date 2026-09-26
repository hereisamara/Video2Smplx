#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 120:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_ablate
#SBATCH --output=v2sx_ablate.%j.out
#SBATCH --error=v2sx_ablate.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCES="${SEQUENCES:-SignLanguage_S2}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_ablation_samples}"
VARIANTS="${VARIANTS:-base_no_stab base_stab global_only fast_global_hand accurate_2d accurate_2d_stab}"

YOLO_MODEL="${YOLO_MODEL:-yolov8x-pose.pt}"
YOLO_KEYPOINTS="${YOLO_KEYPOINTS:-${RUN_ROOT}/evaluation/precomputed_yolo_pose_keypoints.json}"
SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"
EVAL_MODEL="${EVAL_MODEL:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"
GLOBAL_CKPT="${GLOBAL_CKPT:-outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt}"
HAND_CKPT="${HAND_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt}"
UPPER2D_CKPT="${UPPER2D_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt}"
UPPER_SCALE="${UPPER_SCALE:-0.75}"
SKIP_RENDER="${SKIP_RENDER:-0}"
EVALUATE="${EVALUATE:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "[check] cwd: $(pwd)"
echo "[check] sequences: ${SEQUENCES}"
echo "[check] variants: ${VARIANTS}"
echo "[check] run root: ${RUN_ROOT}"
echo "[check] yolo keypoints: ${YOLO_KEYPOINTS}"
nvidia-smi || true

for REQUIRED in "${SMPLX_MODEL}" "${EVAL_MODEL}" "${GLOBAL_CKPT}" "${HAND_CKPT}" "${UPPER2D_CKPT}"; do
  if [ ! -s "${REQUIRED}" ]; then
    echo "[error] missing required file: ${REQUIRED}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/evaluation"
MANIFEST="${RUN_ROOT}/evaluation/ablation_sample_manifest.csv"
echo "sequence,variant,postprocess_mode,stabilize_global_orient,stabilize_shape,stabilize_lower_body,stabilize_torso,upper_scale,output_dir,final_dir,smplx_npz,rendered_video,side_by_side_video,runtime_report,evaluation_dir" > "${MANIFEST}"

needs_accurate=0
for VARIANT in ${VARIANTS}; do
  case "${VARIANT}" in
    accurate_2d|accurate_2d_stab)
      needs_accurate=1
      ;;
  esac
done

if [ "${needs_accurate}" -eq 1 ] && [ ! -s "${YOLO_KEYPOINTS}" ]; then
  if [ ! -s "${YOLO_MODEL}" ]; then
    echo "[error] missing YOLO model for precompute: ${YOLO_MODEL}" >&2
    exit 2
  fi
  echo "[precompute] YOLO keypoints -> ${YOLO_KEYPOINTS}"
  mkdir -p "$(dirname "${YOLO_KEYPOINTS}")"
  "${PYTHON[@]}" -m tools.preprocessing.extract_signlanguage_yolo_pose_keypoints \
    --video-dir "${VIDEO_DIR}" \
    --output "${YOLO_KEYPOINTS}" \
    --model "${YOLO_MODEL}" \
    --sequences ${SEQUENCES} \
    --device cuda \
    --imgsz 960 \
    --conf 0.25
fi

variant_args() {
  local variant="$1"
  POST_MODE="none"
  FINAL_NAME="final_base"
  STAB_GLOBAL=0
  STAB_SHAPE=0
  STAB_LOWER=0
  STAB_TORSO=0
  EXTRA_ARGS=()

  case "${variant}" in
    base_no_stab)
      POST_MODE="none"
      FINAL_NAME="final_base"
      ;;
    base_stab)
      POST_MODE="none"
      FINAL_NAME="final_base"
      STAB_GLOBAL=1
      STAB_SHAPE=1
      EXTRA_ARGS+=(--stabilize-global-orient --stabilize-shape)
      ;;
    global_only)
      POST_MODE="global"
      FINAL_NAME="final_global"
      ;;
    fast_global_hand)
      POST_MODE="fast"
      FINAL_NAME="final_fast"
      ;;
    accurate_2d)
      POST_MODE="accurate"
      FINAL_NAME="final_postprocessed"
      EXTRA_ARGS+=(--precomputed-yolo-keypoints "${YOLO_KEYPOINTS}")
      ;;
    accurate_2d_stab)
      POST_MODE="accurate"
      FINAL_NAME="final_postprocessed"
      STAB_GLOBAL=1
      STAB_SHAPE=1
      EXTRA_ARGS+=(--stabilize-global-orient --stabilize-shape --precomputed-yolo-keypoints "${YOLO_KEYPOINTS}")
      ;;
    *)
      echo "[error] unknown variant: ${variant}" >&2
      exit 2
      ;;
  esac
}

for SEQUENCE in ${SEQUENCES}; do
  VIDEO="${VIDEO_DIR}/${SEQUENCE}/${SEQUENCE}.mp4"
  if [ ! -s "${VIDEO}" ]; then
    echo "[error] missing video: ${VIDEO}" >&2
    exit 2
  fi

  for VARIANT in ${VARIANTS}; do
    variant_args "${VARIANT}"
    OUT="${RUN_ROOT}/${VARIANT}/${SEQUENCE}"
    RENDER_ARGS=()
    if [ "${SKIP_RENDER}" = "1" ]; then
      RENDER_ARGS+=(--skip-render)
    fi

    echo "[run] sequence=${SEQUENCE} variant=${VARIANT} postprocess=${POST_MODE}"
    "${PYTHON[@]}" delivery/source/integrated_postprocessed_pipeline_cli.py \
      --video "${VIDEO}" \
      --output "${OUT}" \
      --name "${SEQUENCE}" \
      --sequence "${SEQUENCE}" \
      --device cuda \
      --smplx-model "${SMPLX_MODEL}" \
      --eval-model "${EVAL_MODEL}" \
      --global-ckpt "${GLOBAL_CKPT}" \
      --hand-ckpt "${HAND_CKPT}" \
      --upper2d-ckpt "${UPPER2D_CKPT}" \
      --yolo-model "${YOLO_MODEL}" \
      --upper-scale "${UPPER_SCALE}" \
      --postprocess-mode "${POST_MODE}" \
      --smplestx-detector-stride 5 \
      --smplestx-batch-size 8 \
      --smplestx-inference-mode \
      --smplestx-single-gpu-model \
      --parallel-models \
      "${EXTRA_ARGS[@]}" \
      "${RENDER_ARGS[@]}"

    FINAL_DIR="${OUT}/${FINAL_NAME}"
    RUNTIME_REPORT="${OUT}/runtime/integrated_postprocessed_runtime_report.json"
    EVAL_DIR=""

    if [ "${EVALUATE}" = "1" ]; then
      EVAL_DIR="${RUN_ROOT}/evaluation/${VARIANT}/${SEQUENCE}"
      "${PYTHON[@]}" -m tools.evaluation.evaluate_signlanguage_geometry \
        --pred-root "${RUN_ROOT}/${VARIANT}" \
        --params-subdir "${FINAL_NAME}/rendered/params" \
        --model-path "${EVAL_MODEL}" \
        --batch-size "${BATCH_SIZE}" \
        --output-dir "${EVAL_DIR}" \
        --sequences "${SEQUENCE}"
    fi

    echo "${SEQUENCE},${VARIANT},${POST_MODE},${STAB_GLOBAL},${STAB_SHAPE},${STAB_LOWER},${STAB_TORSO},${UPPER_SCALE},${OUT},${FINAL_DIR},${FINAL_DIR}/smplx_params.npz,${FINAL_DIR}/rendered/smplx_render.mp4,${FINAL_DIR}/side_by_side_input_render.mp4,${RUNTIME_REPORT},${EVAL_DIR}" >> "${MANIFEST}"
  done
done

"${PYTHON[@]}" delivery/source/summarize_ablation_samples.py \
  --manifest "${MANIFEST}" \
  --output-dir "${RUN_ROOT}/evaluation"

echo "[done] manifest: ${MANIFEST}"
echo "[done] summary: ${RUN_ROOT}/evaluation/ablation_sample_summary.md"
