#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 120:00:00
#SBATCH -A lt200246
#SBATCH -J sl_2d_upper
#SBATCH --output=sl_2d_upper.%j.out
#SBATCH --error=sl_2d_upper.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"

# Current best predictions are the input by default.
INPUT_ROOT="${INPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs}"
INPUT_SUBDIR="${INPUT_SUBDIR:-fused_params_model_global_hand_corrected}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs}"
MODEL_PATH="${MODEL_PATH:-signlanguage_global_correction_server/models/SMPLX_FEMALE.npz}"
VIDEO_DIR="${VIDEO_DIR:-datasets/videos/SignLanguage}"
YOLO_MODEL="${YOLO_MODEL:-${OUTPUT_ROOT}/models/yolo26l-pose.pt}"
YOLO_JSON="${YOLO_JSON:-${OUTPUT_ROOT}/evaluation/yolo26l_pose_keypoints.json}"
DATASET_PATH="${DATASET_PATH:-${OUTPUT_ROOT}/evaluation/2d_guided_upperbody_dataset_r1.npz}"
TRAIN_DIR="${TRAIN_DIR:-${OUTPUT_ROOT}/evaluation/2d_guided_upperbody_model_r1}"
OUTPUT_SUBDIR_BASE="${OUTPUT_SUBDIR_BASE:-fused_params_2d_upper_corrected}"
APPLY_SCALES="${APPLY_SCALES:-0.25 0.50 0.75 1.00}"
SEQUENCES="${SEQUENCES:-all}"

echo "[check] cwd: $(pwd)"
echo "[check] input root: ${INPUT_ROOT}"
echo "[check] input subdir: ${INPUT_SUBDIR}"
echo "[check] output root: ${OUTPUT_ROOT}"
echo "[check] model path: ${MODEL_PATH}"
echo "[check] yolo model: ${YOLO_MODEL}"
echo "[check] yolo json: ${YOLO_JSON}"
echo "[check] apply scales: ${APPLY_SCALES}"
echo "[check] sequences: ${SEQUENCES}"
nvidia-smi || true

mkdir -p "${OUTPUT_ROOT}/evaluation"
if [ ! -s "${YOLO_MODEL}" ]; then
  echo "[error] YOLO model file not found: ${YOLO_MODEL}" >&2
  echo "[error] Put yolo26l-pose.pt there, or submit with YOLO_MODEL=/path/to/yolo26l-pose.pt" >&2
  echo "[error] The cluster compute node has no internet, so Ultralytics auto-download cannot work." >&2
  exit 2
fi

if [ ! -s "${YOLO_JSON}" ]; then
  echo "[1/5] Extract YOLO pose keypoints"
  time ${PYTHON} -m tools.preprocessing.extract_signlanguage_yolo_pose_keypoints \
    --video-dir "${VIDEO_DIR}" \
    --output "${YOLO_JSON}" \
    --model "${YOLO_MODEL}" \
    --sequences ${SEQUENCES} \
    --device cuda \
    --imgsz 960 \
    --conf 0.25
else
  echo "[1/5] Reuse existing YOLO keypoints: ${YOLO_JSON}"
fi

echo "[2/5] Build 2D-guided correction dataset"
time ${PYTHON} research/signlanguage_training/build_signlanguage_2d_guided_upperbody_dataset.py \
  --pred-root "${INPUT_ROOT}" \
  --params-subdir "${INPUT_SUBDIR}" \
  --yolo-keypoints "${YOLO_JSON}" \
  --model-path "${MODEL_PATH}" \
  --output "${DATASET_PATH}" \
  --sequences ${SEQUENCES} \
  --feature-preset upper_global \
  --temporal-radius 1 \
  --min-keypoint-conf 0.15

echo "[3/5] Train 2D-guided upper-body corrector"
time ${PYTHON} research/signlanguage_training/train_signlanguage_2d_guided_upperbody_corrector.py \
  --dataset "${DATASET_PATH}" \
  --output-dir "${TRAIN_DIR}" \
  --epochs 180 \
  --batch-size 512 \
  --device cuda

echo "[4/5] Apply and evaluate correction scales"
for SCALE in ${APPLY_SCALES}; do
  SCALE_TAG=$(echo "${SCALE}" | tr '.' 'p')
  OUTPUT_SUBDIR="${OUTPUT_SUBDIR_BASE}_s${SCALE_TAG}"
  EVAL_DIR="${OUTPUT_ROOT}/evaluation/smplx_female_${OUTPUT_SUBDIR}"
  echo "[scale ${SCALE}] apply -> ${OUTPUT_SUBDIR}"
  time ${PYTHON} -m tools.postprocessing.apply_signlanguage_2d_guided_upperbody_corrector \
    --pred-root "${INPUT_ROOT}" \
    --params-subdir "${INPUT_SUBDIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --output-subdir "${OUTPUT_SUBDIR}" \
    --checkpoint "${TRAIN_DIR}/best_model.pt" \
    --yolo-keypoints "${YOLO_JSON}" \
    --model-path "${MODEL_PATH}" \
    --sequences ${SEQUENCES} \
    --device cuda \
    --scale "${SCALE}"

  echo "[scale ${SCALE}] evaluate"
  time ${PYTHON} -m tools.evaluation.evaluate_signlanguage_geometry \
    --pred-root "${OUTPUT_ROOT}" \
    --params-subdir "${OUTPUT_SUBDIR}" \
    --model-path "${MODEL_PATH}" \
    --sequences ${SEQUENCES} \
    --batch-size 16 \
    --output-dir "${EVAL_DIR}"

  echo "[scale ${SCALE}] ALL row"
  ${PYTHON} - <<PY
import csv
p = "${EVAL_DIR}/geometry_summary.csv"
row = [r for r in csv.DictReader(open(p)) if r["sequence"] == "ALL"][0]
keys = [
    "mpjpe_mm_mean",
    "pa_mpjpe_mm_mean",
    "mpvpe_mm_mean",
    "pa_mpvpe_mm_mean",
    "body_mpjpe_mm_mean",
    "body_pa_mpjpe_mm_mean",
    "hand_mpjpe_mm_mean",
    "hand_pa_mpjpe_mm_mean",
    "visible_upper_mpvpe_mm_mean",
    "hands_wrist_mpvpe_mm_mean",
    "hands_pa_mpvpe_mm_mean",
    "face_head_mpvpe_mm_mean",
    "face_pa_mpvpe_mm_mean",
]
print(",".join([row["sequence"], row["frames"]] + [row.get(k, "") for k in keys]))
PY
done

echo "[5/5] Done"
echo "  YOLO keypoints: ${YOLO_JSON}"
echo "  dataset: ${DATASET_PATH}"
echo "  checkpoint: ${TRAIN_DIR}/best_model.pt"
echo "  outputs: ${OUTPUT_ROOT}/SignLanguage_S*/${OUTPUT_SUBDIR_BASE}_s*"
