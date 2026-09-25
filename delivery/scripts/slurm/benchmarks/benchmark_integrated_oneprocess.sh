#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_oneproc
#SBATCH --output=v2sx_oneproc.%j.out
#SBATCH --error=v2sx_oneproc.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCES="${SEQUENCES:-SignLanguage_S2}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark}"
YOLO_MODEL="${YOLO_MODEL:-yolov8x-pose.pt}"
YOLO_KEYPOINTS="${YOLO_KEYPOINTS:-}"
SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"
EVAL_MODEL="${EVAL_MODEL:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"
GLOBAL_CKPT="${GLOBAL_CKPT:-outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt}"
HAND_CKPT="${HAND_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt}"
UPPER2D_CKPT="${UPPER2D_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt}"

echo "[check] cwd: $(pwd)"
echo "[check] sequences: ${SEQUENCES}"
echo "[check] run root: ${RUN_ROOT}"
echo "[check] yolo model: ${YOLO_MODEL}"
echo "[check] precomputed yolo keypoints: ${YOLO_KEYPOINTS:-none}"
nvidia-smi || true

REQUIRED_FILES=("${SMPLX_MODEL}" "${EVAL_MODEL}" "${GLOBAL_CKPT}" "${HAND_CKPT}" "${UPPER2D_CKPT}")
if [ -n "${YOLO_KEYPOINTS}" ]; then
  REQUIRED_FILES+=("${YOLO_KEYPOINTS}")
else
  REQUIRED_FILES+=("${YOLO_MODEL}")
fi
for REQUIRED in "${REQUIRED_FILES[@]}"; do
  if [ ! -s "${REQUIRED}" ]; then
    echo "[error] missing required file: ${REQUIRED}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/evaluation"
SUMMARY_CSV="${RUN_ROOT}/evaluation/oneprocess_runtime_summary.csv"
echo "sequence,mode,frames,total_seconds,total_fps,no_render_seconds,no_render_fps,initial_load_seconds,no_initial_load_total_seconds,no_initial_load_total_fps,no_initial_load_no_render_seconds,no_initial_load_no_render_fps,steady_state_model_fps,steady_state_model_sec_per_frame,warmup_excluded_frames,report" > "${SUMMARY_CSV}"

for SEQUENCE in ${SEQUENCES}; do
  VIDEO="${VIDEO_DIR}/${SEQUENCE}/${SEQUENCE}.mp4"
  if [ ! -s "${VIDEO}" ]; then
    echo "[error] missing video: ${VIDEO}" >&2
    exit 2
  fi

  for MODE in base post; do
    OUT="${RUN_ROOT}/${SEQUENCE}_${MODE}"
    EXTRA=()
    if [ "${MODE}" = "base" ]; then
      EXTRA+=(--disable-postprocessing)
    elif [ -n "${YOLO_KEYPOINTS}" ]; then
      EXTRA+=(--precomputed-yolo-keypoints "${YOLO_KEYPOINTS}")
    fi
    echo "[run] ${SEQUENCE} mode=${MODE}"
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
      --smplestx-detector-stride 5 \
      --smplestx-batch-size 8 \
      --smplestx-inference-mode \
      --smplestx-single-gpu-model \
      --parallel-models \
      "${EXTRA[@]}"

    REPORT="${OUT}/runtime/integrated_postprocessed_runtime_report.json"
    "${PYTHON[@]}" - <<PY
import csv, json
p = "${REPORT}"
d = json.load(open(p))
s = d["summary"]
steady = s.get("steady_state") or {}
initial_load = s.get("initial_load") or {}
with open("${SUMMARY_CSV}", "a", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "${SEQUENCE}",
        "${MODE}",
        s["frames"],
        s["total_seconds"],
        s["total_fps"],
        s["inference_post_no_render_seconds"],
        s["inference_post_no_render_fps"],
        initial_load.get("initial_load_seconds", ""),
        s.get("no_initial_load_total_seconds", ""),
        s.get("no_initial_load_total_fps", ""),
        s.get("no_initial_load_no_render_seconds", ""),
        s.get("no_initial_load_no_render_fps", ""),
        steady.get("steady_state_model_fps", ""),
        steady.get("steady_state_model_seconds_per_frame", ""),
        steady.get("warmup_excluded_frames", ""),
        p,
    ])
print("${SEQUENCE},${MODE}", json.dumps(s, indent=2))
PY
  done
done

echo "[done] one-process runtime summary: ${SUMMARY_CSV}"
