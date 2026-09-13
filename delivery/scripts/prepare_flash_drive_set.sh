#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /path/to/flash-drive-folder" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DELIVERY_DIR="${ROOT_DIR}/delivery"
TARGET="$1/Video2Smplx_delivery"

mkdir -p "${TARGET}"
cp -R "${DELIVERY_DIR}/docs" "${TARGET}/docs"
cp -R "${DELIVERY_DIR}/source" "${TARGET}/source"
cp -R "${DELIVERY_DIR}/scripts" "${TARGET}/scripts"
cp -R "${DELIVERY_DIR}/sample" "${TARGET}/sample"
cp -R "${DELIVERY_DIR}/results" "${TARGET}/results"
cp "${ROOT_DIR}/README_DELIVERY.md" "${TARGET}/README_DELIVERY.md"
cp "${ROOT_DIR}/README_TOR.md" "${TARGET}/README_TOR.md"
cp "${ROOT_DIR}/README_VIDEO_OUTPUT_COMPARISON.md" "${TARGET}/README_VIDEO_OUTPUT_COMPARISON.md"

if [ -d "${DELIVERY_DIR}/final_outputs" ]; then
  cp -R "${DELIVERY_DIR}/final_outputs" "${TARGET}/final_outputs"
fi

echo "[done] flash-drive set copied to: ${TARGET}"
