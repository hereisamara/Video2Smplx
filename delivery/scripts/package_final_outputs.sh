#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/artifacts/delivery_packages}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ "${ALLOW_INCOMPLETE_DELIVERY:-0}" != "1" ]; then
  bash "${ROOT_DIR}/delivery/scripts/verify_delivery_readiness.sh"
fi

mkdir -p "${OUT_DIR}"
PACKAGE_NAME="Video2Smplx_final_outputs_${STAMP}"
PACKAGE_DIR="${OUT_DIR}/${PACKAGE_NAME}"
mkdir -p "${PACKAGE_DIR}/delivery"

cp "${ROOT_DIR}/README.md" "${PACKAGE_DIR}/README.md"
cp -R "${ROOT_DIR}/delivery/final_outputs" "${PACKAGE_DIR}/delivery/final_outputs"
cp -R "${ROOT_DIR}/delivery/sample" "${PACKAGE_DIR}/delivery/sample"
cp -R "${ROOT_DIR}/delivery/results" "${PACKAGE_DIR}/delivery/results"

(
  cd "${PACKAGE_DIR}"
  find . -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS.txt
)

ARCHIVE="${OUT_DIR}/${PACKAGE_NAME}.tar.gz"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "${ARCHIVE}" \
  -C "${OUT_DIR}" "${PACKAGE_NAME}"

echo "[done] final-output package: ${PACKAGE_DIR}"
echo "[done] final-output archive: ${ARCHIVE}"
echo "[done] checksums: ${PACKAGE_DIR}/SHA256SUMS.txt"
