#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_final_test
#SBATCH --output=v2sx_final_test.%j.out
#SBATCH --error=v2sx_final_test.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"

SEQUENCE="${SEQUENCE:-SignLanguage_S2}"
VIDEO="${VIDEO:-datasets/videos/SignLanguage/${SEQUENCE}/${SEQUENCE}.mp4}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_delivery_final_test}"
RUN="${RUN_ROOT}/${SEQUENCE}"
FINAL_RUN="${RUN}/final_postprocessed"

SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"
EVAL_MODEL="${EVAL_MODEL:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"

GLOBAL_CKPT="${GLOBAL_CKPT:-outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt}"
HAND_CKPT="${HAND_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt}"
UPPER2D_CKPT="${UPPER2D_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt}"

YOLO_MODEL="${YOLO_MODEL:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/models/yolov8x-pose.pt}"
YOLO_JSON="${YOLO_JSON:-${RUN_ROOT}/evaluation/yolo_pose_${SEQUENCE}.json}"
UPPER_SCALE="${UPPER_SCALE:-0.75}"

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "[check] cwd: $(pwd)"
echo "[check] sequence: ${SEQUENCE}"
echo "[check] video: ${VIDEO}"
echo "[check] run root: ${RUN_ROOT}"
echo "[check] global checkpoint: ${GLOBAL_CKPT}"
echo "[check] hand checkpoint: ${HAND_CKPT}"
echo "[check] 2D upper checkpoint: ${UPPER2D_CKPT}"
echo "[check] YOLO pose model: ${YOLO_MODEL}"
echo "[check] YOLO pose json: ${YOLO_JSON}"
echo "[check] upper scale: ${UPPER_SCALE}"
nvidia-smi || true

for REQUIRED in "${VIDEO}" "${SMPLX_MODEL}" "${EVAL_MODEL}" "${GLOBAL_CKPT}" "${HAND_CKPT}" "${UPPER2D_CKPT}" "${YOLO_MODEL}"; do
  if [ ! -s "${REQUIRED}" ]; then
    echo "[error] required file missing or empty: ${REQUIRED}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/evaluation"

echo "[1/8] Body estimation"
time ${PYTHON} delivery/source/body_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda \
  --detector-stride 5 \
  --batch-size 8 \
  --inference-mode \
  --single-gpu-model

echo "[2/8] Hand estimation"
time ${PYTHON} delivery/source/hand_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda

echo "[3/8] Face estimation"
time ${PYTHON} delivery/source/face_estimation_cli.py \
  --video "${VIDEO}" \
  --output "${RUN}" \
  --device cuda

echo "[4/8] Base combine, smooth, render, export NPZ"
time ${PYTHON} delivery/source/combine_smooth_render_cli.py \
  --body-dir "${RUN}/body_params" \
  --hand-dir "${RUN}/hand_params" \
  --face-dir "${RUN}/face_params" \
  --output "${RUN}/base_fusion" \
  --input-video "${VIDEO}" \
  --smplx-model "${SMPLX_MODEL}" \
  --fps 30 \
  --gender neutral

if [ ! -d "${RUN_ROOT}/${SEQUENCE}" ]; then
  echo "[error] expected sequence directory missing: ${RUN_ROOT}/${SEQUENCE}" >&2
  exit 2
fi

echo "[5/8] Apply global_orient + transl corrector"
time ${PYTHON} apply_signlanguage_global_correction.py \
  --pred-root "${RUN_ROOT}" \
  --params-subdir base_fusion/fused_params \
  --output-root "${RUN_ROOT}" \
  --output-subdir fused_params_model_global_corrected \
  --checkpoint "${GLOBAL_CKPT}" \
  --sequences "${SEQUENCE}" \
  --device cuda

echo "[6/8] Apply hand wrist/finger corrector"
time ${PYTHON} apply_signlanguage_hand_correction.py \
  --pred-root "${RUN_ROOT}" \
  --params-subdir fused_params_model_global_corrected \
  --output-root "${RUN_ROOT}" \
  --output-subdir fused_params_model_global_hand_corrected \
  --checkpoint "${HAND_CKPT}" \
  --sequences "${SEQUENCE}" \
  --device cuda

if [ ! -s "${YOLO_JSON}" ]; then
  echo "[7/8] Extract YOLO pose keypoints"
  time ${PYTHON} extract_signlanguage_yolo_pose_keypoints.py \
    --video-dir "${VIDEO_DIR}" \
    --output "${YOLO_JSON}" \
    --model "${YOLO_MODEL}" \
    --sequences "${SEQUENCE}" \
    --device cuda \
    --imgsz 960 \
    --conf 0.25
else
  echo "[7/8] Reuse YOLO pose keypoints: ${YOLO_JSON}"
fi

echo "[8/8] Apply 2D-guided upper-body corrector and render final output"
time ${PYTHON} apply_signlanguage_2d_guided_upperbody_corrector.py \
  --pred-root "${RUN_ROOT}" \
  --params-subdir fused_params_model_global_hand_corrected \
  --output-root "${RUN_ROOT}" \
  --output-subdir fused_params_2d_upper_corrected_s0p75 \
  --checkpoint "${UPPER2D_CKPT}" \
  --yolo-keypoints "${YOLO_JSON}" \
  --model-path "${EVAL_MODEL}" \
  --sequences "${SEQUENCE}" \
  --device cuda \
  --scale "${UPPER_SCALE}"

time ${PYTHON} delivery/source/combine_smooth_render_cli.py \
  --body-dir "${RUN_ROOT}/${SEQUENCE}/fused_params_2d_upper_corrected_s0p75" \
  --output "${FINAL_RUN}" \
  --input-video "${VIDEO}" \
  --smplx-model "${SMPLX_MODEL}" \
  --fps 30 \
  --gender neutral

echo "[verify] required final outputs"
test -s "${FINAL_RUN}/smplx_params.npz"
test -s "${FINAL_RUN}/rendered/smplx_render.mp4"
test -s "${FINAL_RUN}/side_by_side_input_render.mp4"
test -s "${FINAL_RUN}/combine_render_report.json"

echo "[done] final postprocessed delivery pipeline test passed"
echo "  base output       : ${RUN}/base_fusion"
echo "  corrected params  : ${RUN_ROOT}/${SEQUENCE}/fused_params_2d_upper_corrected_s0p75"
echo "  final smplx npz   : ${FINAL_RUN}/smplx_params.npz"
echo "  final render      : ${FINAL_RUN}/rendered/smplx_render.mp4"
echo "  final side-by-side: ${FINAL_RUN}/side_by_side_input_render.mp4"
