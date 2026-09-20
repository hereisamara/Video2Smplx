#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_timeopt
#SBATCH --output=v2sx_timeopt.%j.out
#SBATCH --error=v2sx_timeopt.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCES="${SEQUENCES:-SignLanguage_S2}"
VARIANTS="${VARIANTS:-legacy_parallel fastio_parallel fastio_batched}"
REPEATS="${REPEATS:-1}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_time_optimization_benchmark}"
YOLO_KEYPOINTS="${YOLO_KEYPOINTS:-/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json}"
SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"
EVAL_MODEL="${EVAL_MODEL:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"
GLOBAL_CKPT="${GLOBAL_CKPT:-outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt}"
HAND_CKPT="${HAND_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt}"
UPPER2D_CKPT="${UPPER2D_CKPT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt}"

echo "[check] cwd: $(pwd)"
echo "[check] sequences: ${SEQUENCES}"
echo "[check] variants: ${VARIANTS}"
echo "[check] repeats: ${REPEATS}"
echo "[check] run root: ${RUN_ROOT}"
echo "[check] precomputed YOLO: ${YOLO_KEYPOINTS}"
nvidia-smi || true

if [[ " ${VARIANTS} " != *" legacy_parallel "* ]]; then
  echo "[error] VARIANTS must include legacy_parallel as the NPZ equivalence reference" >&2
  exit 2
fi

REQUIRED_FILES=(
  "${YOLO_KEYPOINTS}"
  "${SMPLX_MODEL}"
  "${EVAL_MODEL}"
  "${GLOBAL_CKPT}"
  "${HAND_CKPT}"
  "${UPPER2D_CKPT}"
)
for REQUIRED in "${REQUIRED_FILES[@]}"; do
  if [ ! -s "${REQUIRED}" ]; then
    echo "[error] missing required file: ${REQUIRED}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/evaluation"
SUMMARY_CSV="${RUN_ROOT}/evaluation/time_optimization_summary.csv"
EQUIVALENCE_CSV="${RUN_ROOT}/evaluation/time_optimization_npz_equivalence.csv"
echo "sequence,variant,repeat,frames,total_seconds,total_fps,no_initial_load_total_seconds,no_initial_load_total_fps,no_initial_load_no_render_seconds,no_initial_load_no_render_fps,steady_state_model_fps,steady_state_model_ms_per_frame,base_seconds,global_seconds,hand_seconds,yolo_load_seconds,upper2d_seconds,final_export_seconds,report" > "${SUMMARY_CSV}"
echo "sequence,repeat,variant,frame_ids_equal,valid_mismatch_count,max_abs_param_diff,mean_abs_param_diff,reference_npz,candidate_npz" > "${EQUIVALENCE_CSV}"

for SEQUENCE in ${SEQUENCES}; do
  VIDEO="${VIDEO_DIR}/${SEQUENCE}/${SEQUENCE}.mp4"
  if [ ! -s "${VIDEO}" ]; then
    echo "[error] missing video: ${VIDEO}" >&2
    exit 2
  fi

  for REPEAT in $(seq 1 "${REPEATS}"); do
    for VARIANT in ${VARIANTS}; do
      EXTRA=(--parallel-models)
      case "${VARIANT}" in
        legacy_parallel)
          ;;
        fastio_parallel)
          EXTRA+=(--fast-io)
          ;;
        fastio_batched)
          EXTRA=(--no-parallel-models --fast-io)
          ;;
        *)
          echo "[error] unknown variant: ${VARIANT}" >&2
          exit 2
          ;;
      esac

      OUT="${RUN_ROOT}/${SEQUENCE}/${VARIANT}_r${REPEAT}"
      echo "[run] sequence=${SEQUENCE} variant=${VARIANT} repeat=${REPEAT}"
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
        --smplestx-batch-size 8 \
        --smplestx-inference-mode \
        --smplestx-single-gpu-model \
        --post-batch-size 512 \
        --timing-log-interval 0 \
        --warmup-exclude-frames 2 \
        --skip-render \
        "${EXTRA[@]}"

      REPORT="${OUT}/runtime/integrated_postprocessed_runtime_report.json"
      "${PYTHON[@]}" - "${REPORT}" "${SUMMARY_CSV}" "${SEQUENCE}" "${VARIANT}" "${REPEAT}" <<'PY'
import csv
import json
import sys

report_path, summary_path, sequence, variant, repeat = sys.argv[1:]
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
summary = report["summary"]
steady = summary.get("steady_state") or {}
stages = {row["stage"]: row for row in report.get("timings", [])}

def seconds(stage):
    return (stages.get(stage) or {}).get("seconds", "")

steady_spf = steady.get("steady_state_model_seconds_per_frame")
with open(summary_path, "a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        sequence,
        variant,
        repeat,
        summary["frames"],
        summary["total_seconds"],
        summary["total_fps"],
        summary.get("no_initial_load_total_seconds", ""),
        summary.get("no_initial_load_total_fps", ""),
        summary.get("no_initial_load_no_render_seconds", ""),
        summary.get("no_initial_load_no_render_fps", ""),
        steady.get("steady_state_model_fps", ""),
        "" if steady_spf is None else steady_spf * 1000.0,
        seconds("base_integrated_inference_fusion_no_render"),
        seconds("post_global_orient_transl"),
        seconds("post_hand_wrist_fingers"),
        seconds("post_load_precomputed_yolo_2d"),
        seconds("post_2d_guided_upper_body"),
        seconds("final_npz_smooth_render"),
        report_path,
    ])
print(json.dumps({"sequence": sequence, "variant": variant, "summary": summary}, indent=2))
PY
    done

    "${PYTHON[@]}" - "${RUN_ROOT}" "${EQUIVALENCE_CSV}" "${SEQUENCE}" "${REPEAT}" ${VARIANTS} <<'PY'
import csv
import sys
from pathlib import Path

import numpy as np

run_root, output_csv, sequence, repeat, *variants = sys.argv[1:]
root = Path(run_root) / sequence
reference = root / f"legacy_parallel_r{repeat}" / "final_postprocessed" / "smplx_params.npz"
if not reference.is_file():
    raise FileNotFoundError(f"Missing equivalence reference: {reference}")

with np.load(reference) as ref:
    ref_data = {key: ref[key] for key in ref.files}

rows = []
for variant in variants:
    if variant == "legacy_parallel":
        continue
    candidate = root / f"{variant}_r{repeat}" / "final_postprocessed" / "smplx_params.npz"
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing equivalence candidate: {candidate}")
    with np.load(candidate) as current:
        frame_equal = bool(np.array_equal(ref_data["frame_id"], current["frame_id"]))
        valid_mismatch = int(np.count_nonzero(ref_data["valid"] != current["valid"]))
        max_diff = 0.0
        diff_sum = 0.0
        diff_count = 0
        for key, ref_value in ref_data.items():
            if key in {"frame_id", "valid"}:
                continue
            value = current[key]
            if ref_value.shape != value.shape:
                max_diff = float("inf")
                continue
            finite = np.isfinite(ref_value) & np.isfinite(value)
            if not np.any(finite):
                continue
            diff = np.abs(ref_value[finite].astype(np.float64) - value[finite].astype(np.float64))
            max_diff = max(max_diff, float(diff.max()))
            diff_sum += float(diff.sum())
            diff_count += int(diff.size)
        rows.append([
            sequence,
            repeat,
            variant,
            frame_equal,
            valid_mismatch,
            max_diff,
            diff_sum / diff_count if diff_count else 0.0,
            reference,
            candidate,
        ])

with open(output_csv, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerows(rows)
for row in rows:
    print("[equivalence]", row)
PY
  done
done

echo "[done] summary: ${SUMMARY_CSV}"
cat "${SUMMARY_CSV}"
echo "[done] NPZ equivalence: ${EQUIVALENCE_CSV}"
cat "${EQUIVALENCE_CSV}"
