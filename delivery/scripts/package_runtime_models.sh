#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_SOURCE_ROOT="${MODEL_SOURCE_ROOT:-${ROOT_DIR}}"
CORRECTOR_SOURCE_ROOT="${CORRECTOR_SOURCE_ROOT:-${ROOT_DIR}}"
OUT_DIR="${1:-${ROOT_DIR}/artifacts/delivery_packages}"
YOLO_POSE_MODEL="${YOLO_POSE_MODEL:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/models/yolov8x-pose.pt}"
TORCH_HUB_DIR="${TORCH_HUB_DIR:-${TORCH_HOME:-${HOME}/.cache/torch}/hub}"
COMPRESSION="${COMPRESSION:-none}"
SPLIT_SIZE="${SPLIT_SIZE:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ "${ACKNOWLEDGE_RESTRICTED_MODEL_LICENSES:-0}" != "1" ]; then
  echo "[error] model bundle contains restricted third-party assets" >&2
  echo "Review delivery/docs/MODEL_BUNDLE.md and rerun with:" >&2
  echo "  ACKNOWLEDGE_RESTRICTED_MODEL_LICENSES=1" >&2
  exit 2
fi

case "${COMPRESSION}" in
  none|gzip) ;;
  *) echo "[error] COMPRESSION must be none or gzip" >&2; exit 2 ;;
esac

YOLO_POSE_MODEL="${YOLO_POSE_MODEL}" TORCH_HUB_DIR="${TORCH_HUB_DIR}" \
  CORRECTOR_ROOT="${CORRECTOR_SOURCE_ROOT}" \
  bash "${ROOT_DIR}/delivery/scripts/verify_runtime_models.sh" "${MODEL_SOURCE_ROOT}"

mkdir -p "${OUT_DIR}"
BUNDLE_NAME="Video2Smplx_runtime_models_${STAMP}"
BUNDLE_DIR="${OUT_DIR}/${BUNDLE_NAME}"
mkdir -p \
  "${BUNDLE_DIR}/SMPLest-X-Inference/pretrained_models" \
  "${BUNDLE_DIR}/SMPLest-X-Inference/human_models/human_model_files" \
  "${BUNDLE_DIR}/WiLoR-Inference" \
  "${BUNDLE_DIR}/EMOCA-Inference" \
  "${BUNDLE_DIR}/delivery/models" \
  "${BUNDLE_DIR}/runtime_model_cache/torch/hub"

cp -a "${MODEL_SOURCE_ROOT}/SMPLest-X-Inference/pretrained_models/smplest_x_h" \
  "${BUNDLE_DIR}/SMPLest-X-Inference/pretrained_models/"
cp -a "${MODEL_SOURCE_ROOT}/SMPLest-X-Inference/pretrained_models/yolov8x.pt" \
  "${BUNDLE_DIR}/SMPLest-X-Inference/pretrained_models/yolov8x.pt"
cp -a "${MODEL_SOURCE_ROOT}/SMPLest-X-Inference/human_models/human_model_files/smplx" \
  "${BUNDLE_DIR}/SMPLest-X-Inference/human_models/human_model_files/"

cp -a "${MODEL_SOURCE_ROOT}/WiLoR-Inference/pretrained_models" \
  "${BUNDLE_DIR}/WiLoR-Inference/pretrained_models"
cp -a "${MODEL_SOURCE_ROOT}/WiLoR-Inference/mano_data" \
  "${BUNDLE_DIR}/WiLoR-Inference/mano_data"

cp -a "${MODEL_SOURCE_ROOT}/EMOCA-Inference/assets" \
  "${BUNDLE_DIR}/EMOCA-Inference/assets"
cp -a "${TORCH_HUB_DIR}/checkpoints" \
  "${BUNDLE_DIR}/runtime_model_cache/torch/hub/checkpoints"

cp -a "${CORRECTOR_SOURCE_ROOT}/delivery/models/correctors" \
  "${BUNDLE_DIR}/delivery/models/correctors"
cp -a "${YOLO_POSE_MODEL}" "${BUNDLE_DIR}/yolov8x-pose.pt"

cp "${ROOT_DIR}/delivery/docs/MODEL_SETUP.md" "${BUNDLE_DIR}/MODEL_SETUP.md"
cp "${ROOT_DIR}/delivery/docs/MODEL_BUNDLE.md" "${BUNDLE_DIR}/MODEL_BUNDLE.md"
cp "${ROOT_DIR}/delivery/scripts/verify_runtime_models.sh" "${BUNDLE_DIR}/verify_runtime_models.sh"

YOLO_POSE_MODEL="${BUNDLE_DIR}/yolov8x-pose.pt" \
  TORCH_HUB_DIR="${BUNDLE_DIR}/runtime_model_cache/torch/hub" \
  bash "${BUNDLE_DIR}/verify_runtime_models.sh" "${BUNDLE_DIR}"

du -sh "${BUNDLE_DIR}" | tee "${BUNDLE_DIR}/MODEL_BUNDLE_SIZE.txt"

(
  cd "${BUNDLE_DIR}"
  find . -type f ! -name MODEL_SHA256SUMS.txt -print0 \
    | sort -z | xargs -0 sha256sum > MODEL_SHA256SUMS.txt
)

if [ "${COMPRESSION}" = "gzip" ]; then
  ARCHIVE="${OUT_DIR}/${BUNDLE_NAME}.tar.gz"
  tar -czf "${ARCHIVE}" -C "${OUT_DIR}" "${BUNDLE_NAME}"
else
  ARCHIVE="${OUT_DIR}/${BUNDLE_NAME}.tar"
  tar -cf "${ARCHIVE}" -C "${OUT_DIR}" "${BUNDLE_NAME}"
fi

if [ -n "${SPLIT_SIZE}" ]; then
  split -b "${SPLIT_SIZE}" "${ARCHIVE}" "${ARCHIVE}.part-"
  echo "[done] split parts: ${ARCHIVE}.part-*"
fi

echo "[done] runtime model bundle: ${BUNDLE_DIR}"
echo "[done] runtime model archive: ${ARCHIVE}"
