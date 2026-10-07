#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_10fps
#SBATCH --output=v2sx_10fps.%j.out
#SBATCH --error=v2sx_10fps.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCES="${SEQUENCES:-SignLanguage_S2}"
VARIANTS="${VARIANTS:-baseline face5 hand3_face5 hand5_face5}"
REPEATS="${REPEATS:-1}"
TARGET_FPS="${TARGET_FPS:-10.0}"
SAVE_FINAL_PKLS="${SAVE_FINAL_PKLS:-0}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_10fps_detector_stride}"
YOLO_KEYPOINTS="${YOLO_KEYPOINTS:-/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json}"
SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"
EVAL_MODEL="${EVAL_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_FEMALE.npz}"
GLOBAL_CKPT="${GLOBAL_CKPT:-delivery/models/correctors/global/best_model.pt}"
HAND_CKPT="${HAND_CKPT:-delivery/models/correctors/hand/best_model.pt}"
UPPER2D_CKPT="${UPPER2D_CKPT:-delivery/models/correctors/upper2d/best_model.pt}"

echo "[check] cwd: $(pwd)"
echo "[check] sequences: ${SEQUENCES}"
echo "[check] variants: ${VARIANTS}"
echo "[check] target FPS: ${TARGET_FPS}"
echo "[check] save final PKLs: ${SAVE_FINAL_PKLS}"
echo "[check] run root: ${RUN_ROOT}"
nvidia-smi || true

for REQUIRED in \
  "${YOLO_KEYPOINTS}" \
  "${SMPLX_MODEL}" \
  "${EVAL_MODEL}" \
  "${GLOBAL_CKPT}" \
  "${HAND_CKPT}" \
  "${UPPER2D_CKPT}"; do
  if [ ! -s "${REQUIRED}" ]; then
    echo "[error] missing required file: ${REQUIRED}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/evaluation"
SUMMARY_CSV="${RUN_ROOT}/evaluation/target_10fps_summary.csv"
echo "sequence,variant,repeat,wilor_detector_stride,emoca_detector_stride,frames,total_fps,no_initial_load_total_fps,steady_model_fps,warm_pipeline_fps,warm_pipeline_ms_per_frame,smplestx_ms_per_frame,wilor_ms_per_frame,emoca_ms_per_frame,postprocess_export_ms_per_frame,target_fps,reaches_target,report" > "${SUMMARY_CSV}"

for SEQUENCE in ${SEQUENCES}; do
  VIDEO="${VIDEO_DIR}/${SEQUENCE}/${SEQUENCE}.mp4"
  if [ ! -s "${VIDEO}" ]; then
    echo "[error] missing video: ${VIDEO}" >&2
    exit 2
  fi

  for REPEAT in $(seq 1 "${REPEATS}"); do
    for VARIANT in ${VARIANTS}; do
      case "${VARIANT}" in
        baseline)
          WILOR_STRIDE=1
          EMOCA_STRIDE=1
          ;;
        face5)
          WILOR_STRIDE=1
          EMOCA_STRIDE=5
          ;;
        hand3_face5)
          WILOR_STRIDE=3
          EMOCA_STRIDE=5
          ;;
        hand5_face5)
          WILOR_STRIDE=5
          EMOCA_STRIDE=5
          ;;
        *)
          echo "[error] unknown variant: ${VARIANT}" >&2
          exit 2
          ;;
      esac

      OUT="${RUN_ROOT}/${SEQUENCE}/${VARIANT}_r${REPEAT}"
      FINAL_PKL_ARGS=()
      if [ "${SAVE_FINAL_PKLS}" = "1" ]; then
        FINAL_PKL_ARGS+=(--save-final-pkls)
      fi

      echo "[run] sequence=${SEQUENCE} variant=${VARIANT} hand_stride=${WILOR_STRIDE} face_stride=${EMOCA_STRIDE}"
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
        --precomputed-yolo-keypoints "${YOLO_KEYPOINTS}" \
        --postprocess-mode accurate \
        --upper-scale 0.75 \
        --smplestx-detector-stride 5 \
        --wilor-detector-stride "${WILOR_STRIDE}" \
        --emoca-detector-stride "${EMOCA_STRIDE}" \
        --smplestx-batch-size 8 \
        --smplestx-inference-mode \
        --smplestx-single-gpu-model \
        --post-batch-size 512 \
        --timing-log-interval 0 \
        --warmup-exclude-frames 2 \
        --parallel-models \
        --fast-io \
        --skip-render \
        "${FINAL_PKL_ARGS[@]}"

      REPORT="${OUT}/runtime/integrated_postprocessed_runtime_report.json"
      "${PYTHON[@]}" - "${REPORT}" "${SUMMARY_CSV}" "${SEQUENCE}" "${VARIANT}" "${REPEAT}" "${WILOR_STRIDE}" "${EMOCA_STRIDE}" "${TARGET_FPS}" <<'PY'
import csv
import json
import sys

(
    report_path,
    summary_path,
    sequence,
    variant,
    repeat,
    wilor_stride,
    emoca_stride,
    target_fps,
) = sys.argv[1:]
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
summary = report["summary"]
steady = summary.get("steady_state") or {}
per_model = steady.get("per_model") or {}
warm = summary.get("warm_steady_pipeline_estimate") or {}
warm_fps = warm.get("fps")
warm_spf = warm.get("seconds_per_frame")

def model_ms(name):
    value = (per_model.get(name) or {}).get("seconds_per_frame")
    return "" if value is None else value * 1000.0

with open(summary_path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow([
        sequence,
        variant,
        repeat,
        wilor_stride,
        emoca_stride,
        summary["frames"],
        summary["total_fps"],
        summary.get("no_initial_load_total_fps", ""),
        steady.get("steady_state_model_fps", ""),
        "" if warm_fps is None else warm_fps,
        "" if warm_spf is None else warm_spf * 1000.0,
        model_ms("smplestx"),
        model_ms("wilor"),
        model_ms("emoca"),
        (warm.get("downstream_seconds_per_frame") or 0.0) * 1000.0,
        target_fps,
        bool(warm_fps is not None and warm_fps >= float(target_fps)),
        report_path,
    ])
print(json.dumps({"sequence": sequence, "variant": variant, "summary": summary}, indent=2))
PY
    done
  done
done

echo "[done] summary: ${SUMMARY_CSV}"
cat "${SUMMARY_CSV}"
