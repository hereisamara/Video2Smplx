#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/flash-drive-folder" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DELIVERY_DIR="${ROOT_DIR}/delivery"
TARGET="$1/Video2Smplx_delivery"
MODEL_BUNDLE_PATH="${MODEL_BUNDLE_PATH:-}"

if [ "${ALLOW_INCOMPLETE_DELIVERY:-0}" != "1" ]; then
  bash "${ROOT_DIR}/delivery/scripts/verify_delivery_readiness.sh"
fi

mkdir -p "${TARGET}"
cp -R "${DELIVERY_DIR}" "${TARGET}/delivery"
cp -R "${ROOT_DIR}/tools" "${TARGET}/tools"
cp -R "${ROOT_DIR}/video2smplx" "${TARGET}/video2smplx"
cp -R "${ROOT_DIR}/requirements" "${TARGET}/requirements"
cp -R "${ROOT_DIR}/tests" "${TARGET}/tests"
cp "${ROOT_DIR}/zero_filter_render.py" "${TARGET}/zero_filter_render.py"
cp "${ROOT_DIR}/smplestx_wilor_emoca_fuse.py" "${TARGET}/smplestx_wilor_emoca_fuse.py"
cp "${ROOT_DIR}/README.md" "${TARGET}/README.md"

mkdir -p "${TARGET}/docs/media"
cp -R "${ROOT_DIR}/docs/media/." "${TARGET}/docs/media/"

rsync -a \
  --exclude='.git/' --exclude='README.md' --exclude='__pycache__/' --exclude='outputs/' \
  --exclude='pretrained_models/' \
  --exclude='human_models/human_model_files/' \
  "${ROOT_DIR}/SMPLest-X-Inference/" "${TARGET}/SMPLest-X-Inference/"
rsync -a \
  --exclude='.git/' --exclude='README.md' --exclude='__pycache__/' \
  --exclude='pretrained_models/' --exclude='mano_data/' \
  "${ROOT_DIR}/WiLoR-Inference/" "${TARGET}/WiLoR-Inference/"
rsync -a \
  --exclude='.git/' --exclude='README.md' --exclude='__pycache__/' --exclude='assets/' \
  "${ROOT_DIR}/EMOCA-Inference/" "${TARGET}/EMOCA-Inference/"

if [ -n "${MODEL_BUNDLE_PATH}" ]; then
  if [ ! -e "${MODEL_BUNDLE_PATH}" ]; then
    echo "[error] MODEL_BUNDLE_PATH does not exist: ${MODEL_BUNDLE_PATH}" >&2
    exit 2
  fi
  mkdir -p "${TARGET}/runtime_models"
  cp -R "${MODEL_BUNDLE_PATH}" "${TARGET}/runtime_models/"
fi

echo "[done] flash-drive set copied to: ${TARGET}"
