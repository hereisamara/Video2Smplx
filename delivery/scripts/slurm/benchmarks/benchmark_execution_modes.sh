#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_modes
#SBATCH --output=v2sx_modes.%j.out
#SBATCH --error=v2sx_modes.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCES="${SEQUENCES:-SignLanguage_S2}"
VARIANTS="${VARIANTS:-frame_barrier stream2 stream4 batch_all}"
REPEATS="${REPEATS:-1}"
TARGET_FPS="${TARGET_FPS:-10.0}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_execution_mode_benchmark}"
YOLO_KEYPOINTS="${YOLO_KEYPOINTS:-/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json}"
SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"
EVAL_MODEL="${EVAL_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_FEMALE.npz}"
GLOBAL_CKPT="${GLOBAL_CKPT:-delivery/models/correctors/global/best_model.pt}"
HAND_CKPT="${HAND_CKPT:-delivery/models/correctors/hand/best_model.pt}"
UPPER2D_CKPT="${UPPER2D_CKPT:-delivery/models/correctors/upper2d/best_model.pt}"
WARMUP_EXCLUDE="${WARMUP_EXCLUDE:-8}"
SMPLESTX_BATCH_SIZE="${SMPLESTX_BATCH_SIZE:-8}"
WILOR_BATCH_SIZE="${WILOR_BATCH_SIZE:-16}"
EMOCA_BATCH_SIZE="${EMOCA_BATCH_SIZE:-16}"

echo "[check] cwd: $(pwd)"
echo "[check] sequences: ${SEQUENCES}"
echo "[check] variants: ${VARIANTS}"
echo "[check] repeats: ${REPEATS}"
echo "[check] run root: ${RUN_ROOT}"
echo "[check] body/hand/face batch: ${SMPLESTX_BATCH_SIZE}/${WILOR_BATCH_SIZE}/${EMOCA_BATCH_SIZE}"
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
SUMMARY_CSV="${RUN_ROOT}/evaluation/execution_mode_summary.csv"
echo "sequence,variant,repeat,execution_mode,max_inflight_frames,frames,total_fps,no_initial_load_fps,base_inference_fusion_fps,warm_pipeline_fps,warm_pipeline_ms_per_frame,fusion_combine_ms_per_frame,global_corrector_ms_per_frame,hand_corrector_ms_per_frame,yolo_keypoint_load_ms_per_frame,upper2d_corrector_ms_per_frame,final_export_ms_per_frame,all_downstream_ms_per_frame,target_fps,reaches_target,report" > "${SUMMARY_CSV}"

for SEQUENCE in ${SEQUENCES}; do
  VIDEO="${VIDEO_DIR}/${SEQUENCE}/${SEQUENCE}.mp4"
  if [ ! -s "${VIDEO}" ]; then
    echo "[error] missing video: ${VIDEO}" >&2
    exit 2
  fi

  for REPEAT in $(seq 1 "${REPEATS}"); do
    for VARIANT in ${VARIANTS}; do
      case "${VARIANT}" in
        frame_barrier)
          EXECUTION_MODE=frame_parallel
          MAX_INFLIGHT=1
          ;;
        stream2)
          EXECUTION_MODE=streaming
          MAX_INFLIGHT=2
          ;;
        stream4)
          EXECUTION_MODE=streaming
          MAX_INFLIGHT=4
          ;;
        batch_all)
          EXECUTION_MODE=batch_then_combine
          MAX_INFLIGHT=1
          ;;
        *)
          echo "[error] unknown variant: ${VARIANT}" >&2
          exit 2
          ;;
      esac

      OUT="${RUN_ROOT}/${SEQUENCE}/${VARIANT}_r${REPEAT}"
      echo "[run] sequence=${SEQUENCE} variant=${VARIANT} mode=${EXECUTION_MODE} inflight=${MAX_INFLIGHT}"
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
        --wilor-detector-stride 1 \
        --emoca-detector-stride 1 \
        --smplestx-batch-size "${SMPLESTX_BATCH_SIZE}" \
        --wilor-batch-size "${WILOR_BATCH_SIZE}" \
        --emoca-batch-size "${EMOCA_BATCH_SIZE}" \
        --smplestx-inference-mode \
        --smplestx-single-gpu-model \
        --model-execution-mode "${EXECUTION_MODE}" \
        --max-inflight-frames "${MAX_INFLIGHT}" \
        --post-batch-size 512 \
        --timing-log-interval 0 \
        --warmup-exclude-frames "${WARMUP_EXCLUDE}" \
        --fast-io \
        --skip-render

      REPORT="${OUT}/runtime/integrated_postprocessed_runtime_report.json"
      "${PYTHON[@]}" - "${REPORT}" "${SUMMARY_CSV}" "${SEQUENCE}" "${VARIANT}" "${REPEAT}" "${EXECUTION_MODE}" "${MAX_INFLIGHT}" "${TARGET_FPS}" <<'PY'
import csv
import json
import sys

(
    report_path,
    summary_path,
    sequence,
    variant,
    repeat,
    execution_mode,
    max_inflight,
    target_fps,
) = sys.argv[1:]
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
summary = report["summary"]
base_timing = report["base_summary"]["timings"]
stages = {row["stage"]: row for row in report.get("timings", [])}
frames = int(summary["frames"])

def stage_ms(name):
    return float((stages.get(name) or {}).get("seconds", 0.0)) * 1000.0 / frames

inference_seconds = float(base_timing["inference_total_sec"])
warm = summary.get("warm_steady_pipeline_estimate") or {}
warm_fps = warm.get("fps")
warm_seconds_per_frame = warm.get("seconds_per_frame")

with open(summary_path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow([
        sequence,
        variant,
        repeat,
        execution_mode,
        max_inflight,
        frames,
        summary["total_fps"],
        summary.get("no_initial_load_total_fps", ""),
        frames / inference_seconds if inference_seconds > 0 else "",
        "" if warm_fps is None else warm_fps,
        "" if warm_seconds_per_frame is None else warm_seconds_per_frame * 1000.0,
        float(base_timing.get("fusion_total_sec") or 0.0) * 1000.0 / frames,
        stage_ms("post_global_orient_transl"),
        stage_ms("post_hand_wrist_fingers"),
        stage_ms("post_load_precomputed_yolo_2d"),
        stage_ms("post_2d_guided_upper_body"),
        stage_ms("final_npz_smooth_render"),
        float(warm.get("downstream_seconds_per_frame") or 0.0) * 1000.0,
        target_fps,
        bool(warm_fps is not None and warm_fps >= float(target_fps)),
        report_path,
    ])
print(json.dumps({"variant": variant, "summary": summary}, indent=2))
PY
    done
  done
done

echo "[done] summary: ${SUMMARY_CSV}"
cat "${SUMMARY_CSV}"
