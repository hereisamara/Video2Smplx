#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_final_bench
#SBATCH --output=v2sx_final_bench.%j.out
#SBATCH --error=v2sx_final_bench.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCES="${SEQUENCES:-SignLanguage_S2}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_delivery_final_benchmark}"

SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"
EVAL_MODEL="${EVAL_MODEL:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"

GLOBAL_CKPT="${GLOBAL_CKPT:-outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt}"
HAND_CKPT="${HAND_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt}"
UPPER2D_CKPT="${UPPER2D_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt}"

YOLO_MODEL="${YOLO_MODEL:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/models/yolov8x-pose.pt}"
UPPER_SCALE="${UPPER_SCALE:-0.75}"

export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "[check] cwd: $(pwd)"
echo "[check] sequences: ${SEQUENCES}"
echo "[check] run root: ${RUN_ROOT}"
echo "[check] global checkpoint: ${GLOBAL_CKPT}"
echo "[check] hand checkpoint: ${HAND_CKPT}"
echo "[check] 2D upper checkpoint: ${UPPER2D_CKPT}"
echo "[check] YOLO pose model: ${YOLO_MODEL}"
echo "[check] upper scale: ${UPPER_SCALE}"
nvidia-smi || true

for REQUIRED in "${SMPLX_MODEL}" "${EVAL_MODEL}" "${GLOBAL_CKPT}" "${HAND_CKPT}" "${UPPER2D_CKPT}" "${YOLO_MODEL}"; do
  if [ ! -s "${REQUIRED}" ]; then
    echo "[error] required file missing or empty: ${REQUIRED}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/evaluation"
ALL_TIMINGS="${RUN_ROOT}/evaluation/all_sequences_stage_timings.csv"
echo "sequence,stage,seconds,status" > "${ALL_TIMINGS}"

run_step() {
  local sequence="$1"
  local stage="$2"
  shift 2
  local timing_csv="${RUN_ROOT}/${sequence}/runtime/stage_timings.csv"
  mkdir -p "$(dirname "${timing_csv}")"
  if [ ! -s "${timing_csv}" ]; then
    echo "stage,seconds,status" > "${timing_csv}"
  fi
  echo "[stage] ${sequence} :: ${stage}"
  local start
  local end
  local seconds
  start=$(date +%s)
  "$@"
  end=$(date +%s)
  seconds=$((end - start))
  echo "${stage},${seconds},ok" >> "${timing_csv}"
  echo "${sequence},${stage},${seconds},ok" >> "${ALL_TIMINGS}"
  echo "[stage-done] ${sequence} :: ${stage} seconds=${seconds}"
}

for SEQUENCE in ${SEQUENCES}; do
  VIDEO="${VIDEO_DIR}/${SEQUENCE}/${SEQUENCE}.mp4"
  RUN="${RUN_ROOT}/${SEQUENCE}"
  FINAL_RUN="${RUN}/final_postprocessed"
  YOLO_JSON="${RUN}/runtime/yolo_pose_${SEQUENCE}.json"
  TIMING_CSV="${RUN}/runtime/stage_timings.csv"
  SUMMARY_DIR="${RUN}/runtime"

  if [ ! -s "${VIDEO}" ]; then
    echo "[error] video missing: ${VIDEO}" >&2
    exit 2
  fi

  rm -f "${TIMING_CSV}"
  mkdir -p "${SUMMARY_DIR}"
  echo "stage,seconds,status" > "${TIMING_CSV}"

  run_step "${SEQUENCE}" "body_smplestx" \
    "${PYTHON[@]}" delivery/source/body_estimation_cli.py \
      --video "${VIDEO}" \
      --output "${RUN}" \
      --device cuda \
      --detector-stride 5 \
      --batch-size 8 \
      --inference-mode \
      --single-gpu-model

  run_step "${SEQUENCE}" "hand_wilor" \
    "${PYTHON[@]}" delivery/source/hand_estimation_cli.py \
      --video "${VIDEO}" \
      --output "${RUN}" \
      --device cuda

  run_step "${SEQUENCE}" "face_emoca" \
    "${PYTHON[@]}" delivery/source/face_estimation_cli.py \
      --video "${VIDEO}" \
      --output "${RUN}" \
      --device cuda

  run_step "${SEQUENCE}" "base_combine_smooth_render_npz" \
    "${PYTHON[@]}" delivery/source/combine_smooth_render_cli.py \
      --body-dir "${RUN}/body_params" \
      --hand-dir "${RUN}/hand_params" \
      --face-dir "${RUN}/face_params" \
      --output "${RUN}/base_fusion" \
      --input-video "${VIDEO}" \
      --smplx-model "${SMPLX_MODEL}" \
      --fps 30 \
      --gender neutral

  run_step "${SEQUENCE}" "correct_global_orient_transl" \
    "${PYTHON[@]}" apply_signlanguage_global_correction.py \
      --pred-root "${RUN_ROOT}" \
      --params-subdir base_fusion/fused_params \
      --output-root "${RUN_ROOT}" \
      --output-subdir fused_params_model_global_corrected \
      --checkpoint "${GLOBAL_CKPT}" \
      --sequences "${SEQUENCE}" \
      --device cuda

  run_step "${SEQUENCE}" "correct_hand_wrist_fingers" \
    "${PYTHON[@]}" apply_signlanguage_hand_correction.py \
      --pred-root "${RUN_ROOT}" \
      --params-subdir fused_params_model_global_corrected \
      --output-root "${RUN_ROOT}" \
      --output-subdir fused_params_model_global_hand_corrected \
      --checkpoint "${HAND_CKPT}" \
      --sequences "${SEQUENCE}" \
      --device cuda

  run_step "${SEQUENCE}" "extract_yolo_pose_2d" \
    "${PYTHON[@]}" extract_signlanguage_yolo_pose_keypoints.py \
      --video-dir "${VIDEO_DIR}" \
      --output "${YOLO_JSON}" \
      --model "${YOLO_MODEL}" \
      --sequences "${SEQUENCE}" \
      --device cuda \
      --imgsz 960 \
      --conf 0.25

  run_step "${SEQUENCE}" "correct_2d_guided_upper_body" \
    "${PYTHON[@]}" apply_signlanguage_2d_guided_upperbody_corrector.py \
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

  run_step "${SEQUENCE}" "final_smooth_render_npz_side_by_side" \
    "${PYTHON[@]}" delivery/source/combine_smooth_render_cli.py \
      --body-dir "${RUN_ROOT}/${SEQUENCE}/fused_params_2d_upper_corrected_s0p75" \
      --output "${FINAL_RUN}" \
      --input-video "${VIDEO}" \
      --smplx-model "${SMPLX_MODEL}" \
      --fps 30 \
      --gender neutral

  echo "[verify] ${SEQUENCE}"
  test -s "${FINAL_RUN}/smplx_params.npz"
  test -s "${FINAL_RUN}/rendered/smplx_render.mp4"
  test -s "${FINAL_RUN}/side_by_side_input_render.mp4"
  test -s "${FINAL_RUN}/combine_render_report.json"

  run_step "${SEQUENCE}" "summarize_runtime_complexity" \
    "${PYTHON[@]}" delivery/source/summarize_runtime_complexity.py \
      --run-root "${RUN_ROOT}" \
      --sequence "${SEQUENCE}" \
      --timing-csv "${TIMING_CSV}" \
      --output-dir "${SUMMARY_DIR}" \
      --global-ckpt "${GLOBAL_CKPT}" \
      --hand-ckpt "${HAND_CKPT}" \
      --upper2d-ckpt "${UPPER2D_CKPT}" \
      --yolo-model "${YOLO_MODEL}"

  echo "[done] ${SEQUENCE}"
  echo "  summary md : ${SUMMARY_DIR}/runtime_complexity_summary.md"
  echo "  summary csv: ${SUMMARY_DIR}/runtime_stage_summary.csv"
  echo "  final npz  : ${FINAL_RUN}/smplx_params.npz"
  echo "  final video: ${FINAL_RUN}/rendered/smplx_render.mp4"
done

echo "[done] all sequence timings: ${ALL_TIMINGS}"
