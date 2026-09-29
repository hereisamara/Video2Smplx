#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 48:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_specialist
#SBATCH --output=v2sx_specialist.%j.out
#SBATCH --error=v2sx_specialist.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

SEQUENCES="${SEQUENCES:-SignLanguage_S2}"
VARIANTS="${VARIANTS:-baseline emoca_expression_only wilor_params_only both_optimized}"
REPEATS="${REPEATS:-2}"
TARGET_MS="${TARGET_MS:-90.0}"
EQUIVALENCE_TOLERANCE="${EQUIVALENCE_TOLERANCE:-0.0001}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
RUN_ROOT="${RUN_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_specialist_output_optimization}"
YOLO_KEYPOINTS="${YOLO_KEYPOINTS:-/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json}"
SMPLX_MODEL="${SMPLX_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz}"
EVAL_MODEL="${EVAL_MODEL:-SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_FEMALE.npz}"
GLOBAL_CKPT="${GLOBAL_CKPT:-delivery/models/correctors/global/best_model.pt}"
HAND_CKPT="${HAND_CKPT:-delivery/models/correctors/hand/best_model.pt}"
UPPER2D_CKPT="${UPPER2D_CKPT:-delivery/models/correctors/upper2d/best_model.pt}"
SMPLESTX_BATCH_SIZE="${SMPLESTX_BATCH_SIZE:-8}"
WILOR_BATCH_SIZE="${WILOR_BATCH_SIZE:-16}"
EMOCA_BATCH_SIZE="${EMOCA_BATCH_SIZE:-16}"

echo "[check] sequences: ${SEQUENCES}"
echo "[check] variants: ${VARIANTS}"
echo "[check] repeats: ${REPEATS}"
echo "[check] target per specialist: <= ${TARGET_MS} ms/frame"
echo "[check] equivalence tolerance: ${EQUIVALENCE_TOLERANCE}"
echo "[check] body/hand/face batch: ${SMPLESTX_BATCH_SIZE}/${WILOR_BATCH_SIZE}/${EMOCA_BATCH_SIZE}"
echo "[check] output: ${RUN_ROOT}"
nvidia-smi || true

if [[ " ${VARIANTS} " != *" baseline "* ]]; then
  echo "[error] VARIANTS must contain baseline for equivalence comparison" >&2
  exit 2
fi

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
SUMMARY_CSV="${RUN_ROOT}/evaluation/specialist_output_optimization_summary.csv"
EQUIVALENCE_CSV="${RUN_ROOT}/evaluation/specialist_output_optimization_equivalence.csv"
echo "sequence,variant,repeat,frames,wilor_params_only,emoca_expression_only,smplestx_ms_per_frame,wilor_ms_per_frame,emoca_ms_per_frame,steady_model_fps,warm_pipeline_fps,no_initial_load_no_render_fps,wilor_meets_target,emoca_meets_target,report" > "${SUMMARY_CSV}"
echo "sequence,variant,repeat,frame_ids_equal,valid_mismatch_count,max_abs_param_diff,mean_abs_param_diff,tolerance,equivalent,reference_npz,candidate_npz" > "${EQUIVALENCE_CSV}"

for SEQUENCE in ${SEQUENCES}; do
  VIDEO="${VIDEO_DIR}/${SEQUENCE}/${SEQUENCE}.mp4"
  if [ ! -s "${VIDEO}" ]; then
    echo "[error] missing video: ${VIDEO}" >&2
    exit 2
  fi

  for REPEAT in $(seq 1 "${REPEATS}"); do
    for VARIANT in ${VARIANTS}; do
      EXTRA=()
      WILOR_FAST=0
      EMOCA_FAST=0
      case "${VARIANT}" in
        baseline)
          ;;
        emoca_expression_only)
          EXTRA+=(--emoca-expression-only)
          EMOCA_FAST=1
          ;;
        wilor_params_only)
          EXTRA+=(--wilor-params-only)
          WILOR_FAST=1
          ;;
        both_optimized)
          EXTRA+=(--wilor-params-only --emoca-expression-only)
          WILOR_FAST=1
          EMOCA_FAST=1
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
        --execution-mode batch \
        --smplx-model "${SMPLX_MODEL}" \
        --eval-model "${EVAL_MODEL}" \
        --global-ckpt "${GLOBAL_CKPT}" \
        --hand-ckpt "${HAND_CKPT}" \
        --upper2d-ckpt "${UPPER2D_CKPT}" \
        --precomputed-yolo-keypoints "${YOLO_KEYPOINTS}" \
        --postprocess-mode accurate \
        --upper-scale 0.75 \
        --smplestx-detector-stride 5 \
        --smplestx-batch-size "${SMPLESTX_BATCH_SIZE}" \
        --wilor-batch-size "${WILOR_BATCH_SIZE}" \
        --emoca-batch-size "${EMOCA_BATCH_SIZE}" \
        --smplestx-inference-mode \
        --smplestx-single-gpu-model \
        --post-batch-size 512 \
        --timing-log-interval 0 \
        --warmup-exclude-batches 1 \
        --fast-io \
        --skip-render \
        "${EXTRA[@]}"

      REPORT="${OUT}/runtime/integrated_postprocessed_runtime_report.json"
      "${PYTHON[@]}" - \
        "${REPORT}" "${SUMMARY_CSV}" "${SEQUENCE}" "${VARIANT}" "${REPEAT}" \
        "${WILOR_FAST}" "${EMOCA_FAST}" "${TARGET_MS}" <<'PY'
import csv
import json
import sys

(
    report_path,
    summary_path,
    sequence,
    variant,
    repeat,
    wilor_fast,
    emoca_fast,
    target_ms,
) = sys.argv[1:]
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
summary = report["summary"]
steady = summary.get("steady_state") or {}
per_model = steady.get("per_model") or {}
warm = summary.get("warm_steady_pipeline_estimate") or {}
target = float(target_ms)

def model_ms(name):
    seconds = (per_model.get(name) or {}).get("seconds_per_frame")
    return None if seconds is None else float(seconds) * 1000.0

sx_ms = model_ms("smplestx")
wilor_ms = model_ms("wilor")
emoca_ms = model_ms("emoca")
with open(summary_path, "a", newline="", encoding="utf-8") as handle:
    csv.writer(handle).writerow([
        sequence,
        variant,
        repeat,
        summary["frames"],
        bool(int(wilor_fast)),
        bool(int(emoca_fast)),
        sx_ms,
        wilor_ms,
        emoca_ms,
        steady.get("normal_working_fps", ""),
        warm.get("fps", ""),
        summary.get("no_initial_load_no_render_fps", ""),
        wilor_ms is not None and wilor_ms <= target,
        emoca_ms is not None and emoca_ms <= target,
        report_path,
    ])
print(json.dumps({
    "variant": variant,
    "smplestx_ms": sx_ms,
    "wilor_ms": wilor_ms,
    "emoca_ms": emoca_ms,
    "normal_working_fps": steady.get("normal_working_fps"),
}, indent=2))
PY
    done

    "${PYTHON[@]}" - \
      "${RUN_ROOT}" "${EQUIVALENCE_CSV}" "${SEQUENCE}" "${REPEAT}" \
      "${EQUIVALENCE_TOLERANCE}" ${VARIANTS} <<'PY'
import csv
import sys
from pathlib import Path

import numpy as np

run_root, output_csv, sequence, repeat, tolerance, *variants = sys.argv[1:]
tolerance = float(tolerance)
root = Path(run_root) / sequence
reference = root / f"baseline_r{repeat}" / "final_postprocessed" / "smplx_params.npz"
with np.load(reference) as loaded:
    reference_data = {key: loaded[key] for key in loaded.files}

rows = []
for variant in variants:
    if variant == "baseline":
        continue
    candidate = root / f"{variant}_r{repeat}" / "final_postprocessed" / "smplx_params.npz"
    with np.load(candidate) as loaded:
        current = {key: loaded[key] for key in loaded.files}
    frame_equal = bool(np.array_equal(reference_data["frame_id"], current["frame_id"]))
    valid_mismatch = int(np.count_nonzero(reference_data["valid"] != current["valid"]))
    max_diff = 0.0
    diff_sum = 0.0
    diff_count = 0
    for key, expected in reference_data.items():
        if key in {"frame_id", "valid"}:
            continue
        actual = current[key]
        if expected.shape != actual.shape:
            max_diff = float("inf")
            continue
        expected_finite = np.isfinite(expected)
        actual_finite = np.isfinite(actual)
        if not np.array_equal(expected_finite, actual_finite):
            max_diff = float("inf")
            continue
        finite = expected_finite
        if not np.any(finite):
            continue
        difference = np.abs(
            expected[finite].astype(np.float64) - actual[finite].astype(np.float64)
        )
        max_diff = max(max_diff, float(difference.max()))
        diff_sum += float(difference.sum())
        diff_count += int(difference.size)
    mean_diff = diff_sum / diff_count if diff_count else 0.0
    equivalent = frame_equal and valid_mismatch == 0 and max_diff <= tolerance
    rows.append([
        sequence,
        variant,
        repeat,
        frame_equal,
        valid_mismatch,
        max_diff,
        mean_diff,
        tolerance,
        equivalent,
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

echo "[done] timing summary: ${SUMMARY_CSV}"
cat "${SUMMARY_CSV}"
echo "[done] equivalence summary: ${EQUIVALENCE_CSV}"
cat "${EQUIVALENCE_CSV}"
