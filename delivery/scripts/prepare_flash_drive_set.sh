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
cp -R "${DELIVERY_DIR}" "${TARGET}/delivery"
cp -R "${ROOT_DIR}/tools" "${TARGET}/tools"
cp -R "${ROOT_DIR}/video2smplx" "${TARGET}/video2smplx"
cp -R "${ROOT_DIR}/requirements" "${TARGET}/requirements"
cp "${ROOT_DIR}/zero_filter_render.py" "${TARGET}/zero_filter_render.py"
cp "${ROOT_DIR}/smplestx_wilor_emoca_fuse.py" "${TARGET}/smplestx_wilor_emoca_fuse.py"
cp "${ROOT_DIR}/README_DELIVERY.md" "${TARGET}/README_DELIVERY.md"
cp "${ROOT_DIR}/README_TOR.md" "${TARGET}/README_TOR.md"
cp "${ROOT_DIR}/README_VIDEO_OUTPUT_COMPARISON.md" "${TARGET}/README_VIDEO_OUTPUT_COMPARISON.md"

if [ -d "${ROOT_DIR}/docs/media/ablation_s2" ]; then
  mkdir -p "${TARGET}/docs/media"
  cp -R "${ROOT_DIR}/docs/media/ablation_s2" "${TARGET}/docs/media/ablation_s2"
fi

echo "[done] flash-drive set copied to: ${TARGET}"
