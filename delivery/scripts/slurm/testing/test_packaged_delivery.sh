#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -A lt200246
#SBATCH -J v2sx_pkg_test
#SBATCH --output=v2sx_pkg_test.%j.out
#SBATCH --error=v2sx_pkg_test.%j.err

set -euo pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

SOURCE_REPO="${SOURCE_REPO:-/home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx}"
PACKAGE_ARCHIVE="${PACKAGE_ARCHIVE:?Set PACKAGE_ARCHIVE to the final Video2Smplx_document_set_*.tar.gz}"
TEST_ROOT="${TEST_ROOT:-/project/lt200246-mmacma/khtun/video2smplx_clean_package_test_${SLURM_JOB_ID:-manual}}"
RUN_FULL_PIPELINE="${RUN_FULL_PIPELINE:-1}"
EXECUTION_MODE="${EXECUTION_MODE:-batch}"
MAX_INFLIGHT_FRAMES="${MAX_INFLIGHT_FRAMES:-2}"
YOLO_MODEL="${YOLO_MODEL:-${SOURCE_REPO}/yolov8x-pose.pt}"
EVAL_MODEL="${EVAL_MODEL:-${SOURCE_REPO}/SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_FEMALE.npz}"

rm -rf "${TEST_ROOT}"
mkdir -p "${TEST_ROOT}"
tar -xzf "${PACKAGE_ARCHIVE}" -C "${TEST_ROOT}"

PACKAGE_ROOT="$(find "${TEST_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'Video2Smplx_document_set_*' -print -quit)"
if [ -z "${PACKAGE_ROOT}" ]; then
  echo "[error] package root not found after extraction" >&2
  exit 2
fi

for name in SMPLest-X-Inference WiLoR-Inference EMOCA-Inference; do
  if [ ! -d "${SOURCE_REPO}/${name}" ]; then
    echo "[error] external model repository missing: ${SOURCE_REPO}/${name}" >&2
    exit 2
  fi
  ln -s "${SOURCE_REPO}/${name}" "${PACKAGE_ROOT}/${name}"
done

cd "${PACKAGE_ROOT}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
PYTHON=(conda run --no-capture-output -n video2smplx_shared310 python)

echo "[check] package root: ${PACKAGE_ROOT}"
bash delivery/scripts/verify_delivery_readiness.sh

for script in \
  delivery/source/body_estimation_cli.py \
  delivery/source/hand_estimation_cli.py \
  delivery/source/face_estimation_cli.py \
  delivery/source/combine_smooth_render_cli.py \
  delivery/source/integrated_postprocessed_pipeline_cli.py; do
  echo "[import] ${script}"
  "${PYTHON[@]}" "${script}" --help >/dev/null
done

if [ "${RUN_FULL_PIPELINE}" = "1" ]; then
  for asset in "${YOLO_MODEL}" "${EVAL_MODEL}"; do
    if [ ! -s "${asset}" ]; then
      echo "[error] external licensed model missing: ${asset}" >&2
      exit 2
    fi
  done

  echo "[run] full accurate pipeline on bundled sample"
  echo "[run] execution mode: ${EXECUTION_MODE}"
  "${PYTHON[@]}" delivery/source/integrated_postprocessed_pipeline_cli.py \
    --video delivery/sample/input-sample.mp4 \
    --output "${TEST_ROOT}/sample_run" \
    --name SignLanguage_S2 \
    --sequence SignLanguage_S2 \
    --device cuda \
    --execution-mode "${EXECUTION_MODE}" \
    --max-inflight-frames "${MAX_INFLIGHT_FRAMES}" \
    --postprocess-mode accurate \
    --global-ckpt delivery/models/correctors/global/best_model.pt \
    --hand-ckpt delivery/models/correctors/hand/best_model.pt \
    --upper2d-ckpt delivery/models/correctors/upper2d/best_model.pt \
    --eval-model "${EVAL_MODEL}" \
    --yolo-model "${YOLO_MODEL}" \
    --smplestx-detector-stride 5 \
    --smplestx-batch-size 8 \
    --wilor-batch-size 16 \
    --emoca-batch-size 16 \
    --smplestx-inference-mode \
    --smplestx-single-gpu-model \
    --fast-io

  FINAL_DIR="${TEST_ROOT}/sample_run/final_postprocessed"
  test -s "${FINAL_DIR}/smplx_params.npz"
  test -s "${FINAL_DIR}/rendered/smplx_render.mp4"
  test -s "${FINAL_DIR}/side_by_side_input_render.mp4"
  test -s "${TEST_ROOT}/sample_run/runtime/integrated_postprocessed_runtime_report.json"
fi

echo "[done] clean delivery package test passed: ${PACKAGE_ROOT}"
