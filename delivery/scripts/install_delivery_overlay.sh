#!/bin/bash
set -euo pipefail

TARGET_ROOT="${1:-.}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DELIVERY="$(cd "${TARGET_ROOT}" && pwd)/delivery"

mkdir -p "${TARGET_DELIVERY}"
cp -R "${SOURCE_ROOT}/docs" "${TARGET_DELIVERY}/docs"
cp -R "${SOURCE_ROOT}/source" "${TARGET_DELIVERY}/source"
cp -R "${SOURCE_ROOT}/scripts" "${TARGET_DELIVERY}/scripts"
cp -R "${SOURCE_ROOT}/sample" "${TARGET_DELIVERY}/sample"
cp -R "${SOURCE_ROOT}/results" "${TARGET_DELIVERY}/results"

if [ -d "${SOURCE_ROOT}/final_outputs" ]; then
  cp -R "${SOURCE_ROOT}/final_outputs" "${TARGET_DELIVERY}/final_outputs"
fi

chmod +x "${TARGET_DELIVERY}"/scripts/*.sh

echo "[done] installed delivery overlay to: ${TARGET_DELIVERY}"
echo "[next] sbatch ${TARGET_DELIVERY}/scripts/slurm_benchmark_delivery_final_postprocessed_pipeline.sh"
