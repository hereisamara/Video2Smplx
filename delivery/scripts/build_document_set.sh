#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DELIVERY_DIR="${ROOT_DIR}/delivery"
OUT_DIR="${1:-${ROOT_DIR}/delivery_package}"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}"
DOC_SET="${OUT_DIR}/Video2Smplx_document_set_${STAMP}"
mkdir -p "${DOC_SET}"

cp -R "${DELIVERY_DIR}/docs" "${DOC_SET}/docs"
cp -R "${DELIVERY_DIR}/source" "${DOC_SET}/source"
cp -R "${DELIVERY_DIR}/scripts" "${DOC_SET}/scripts"
cp -R "${DELIVERY_DIR}/sample" "${DOC_SET}/sample"
cp -R "${DELIVERY_DIR}/results" "${DOC_SET}/results"
cp "${ROOT_DIR}/README_DELIVERY.md" "${DOC_SET}/README_DELIVERY.md"
cp "${ROOT_DIR}/README_TOR.md" "${DOC_SET}/README_TOR.md"
cp "${ROOT_DIR}/README_VIDEO_OUTPUT_COMPARISON.md" "${DOC_SET}/README_VIDEO_OUTPUT_COMPARISON.md"

if [ -d "${DELIVERY_DIR}/final_outputs" ]; then
  cp -R "${DELIVERY_DIR}/final_outputs" "${DOC_SET}/final_outputs"
fi

COPYFILE_DISABLE=1 tar --no-xattrs -czf "${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz" -C "${OUT_DIR}" "$(basename "${DOC_SET}")"

echo "[done] document set folder: ${DOC_SET}"
echo "[done] archive: ${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz"
