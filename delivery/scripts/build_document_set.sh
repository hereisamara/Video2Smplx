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
cp -R "${ROOT_DIR}/tests" "${DOC_SET}/tests"
cp "${ROOT_DIR}/zero_filter_render.py" "${DOC_SET}/zero_filter_render.py"
cp "${ROOT_DIR}/smplestx_wilor_emoca_fuse.py" "${DOC_SET}/smplestx_wilor_emoca_fuse.py"
cp "${ROOT_DIR}/README.md" "${DOC_SET}/README.md"

mkdir -p "${DOC_SET}/docs/media"
cp -R "${ROOT_DIR}/docs/media/." "${DOC_SET}/docs/media/"

# Copy upstream runtime source without restricted/large model assets. Overlay
# the separately generated runtime-model archive after extraction.
rsync -a \
  --exclude='.git/' --exclude='README.md' --exclude='__pycache__/' --exclude='outputs/' \
  --exclude='pretrained_models/' \
  --exclude='human_models/human_model_files/' \
  "${ROOT_DIR}/SMPLest-X-Inference/" "${DOC_SET}/SMPLest-X-Inference/"
rsync -a \
  --exclude='.git/' --exclude='README.md' --exclude='__pycache__/' \
  --exclude='pretrained_models/' --exclude='mano_data/' \
  "${ROOT_DIR}/WiLoR-Inference/" "${DOC_SET}/WiLoR-Inference/"
rsync -a \
  --exclude='.git/' --exclude='README.md' --exclude='__pycache__/' --exclude='assets/' \
  "${ROOT_DIR}/EMOCA-Inference/" "${DOC_SET}/EMOCA-Inference/"

COPYFILE_DISABLE=1 tar --no-xattrs -czf "${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz" -C "${OUT_DIR}" "$(basename "${DOC_SET}")"

echo "[done] document set folder: ${DOC_SET}"
echo "[done] archive: ${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz"
