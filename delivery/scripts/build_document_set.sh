#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DELIVERY_DIR="${ROOT_DIR}/delivery"
OUT_DIR="${1:-${ROOT_DIR}/artifacts/delivery_packages}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ "${ALLOW_INCOMPLETE_DELIVERY:-0}" != "1" ]; then
  bash "${ROOT_DIR}/delivery/scripts/verify_delivery_readiness.sh"
fi

mkdir -p "${OUT_DIR}"
DOC_SET="${OUT_DIR}/Video2Smplx_document_set_${STAMP}"
mkdir -p "${DOC_SET}"

cp -R "${DELIVERY_DIR}" "${DOC_SET}/delivery"
cp -R "${ROOT_DIR}/tools" "${DOC_SET}/tools"
cp -R "${ROOT_DIR}/video2smplx" "${DOC_SET}/video2smplx"
cp -R "${ROOT_DIR}/requirements" "${DOC_SET}/requirements"
cp "${ROOT_DIR}/zero_filter_render.py" "${DOC_SET}/zero_filter_render.py"
cp "${ROOT_DIR}/smplestx_wilor_emoca_fuse.py" "${DOC_SET}/smplestx_wilor_emoca_fuse.py"
cp "${ROOT_DIR}/README_DELIVERY.md" "${DOC_SET}/README_DELIVERY.md"
cp "${ROOT_DIR}/README_TOR.md" "${DOC_SET}/README_TOR.md"
cp "${ROOT_DIR}/README_VIDEO_OUTPUT_COMPARISON.md" "${DOC_SET}/README_VIDEO_OUTPUT_COMPARISON.md"
cp "${ROOT_DIR}/README.md" "${DOC_SET}/README.md"

if [ -d "${ROOT_DIR}/docs/media/ablation_s2" ]; then
  mkdir -p "${DOC_SET}/docs/media"
  cp -R "${ROOT_DIR}/docs/media/ablation_s2" "${DOC_SET}/docs/media/ablation_s2"
fi

COPYFILE_DISABLE=1 tar --no-xattrs -czf "${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz" -C "${OUT_DIR}" "$(basename "${DOC_SET}")"

echo "[done] document set folder: ${DOC_SET}"
echo "[done] archive: ${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz"
