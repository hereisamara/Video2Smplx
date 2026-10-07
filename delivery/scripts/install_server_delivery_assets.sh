#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SEQUENCE="${SEQUENCE:-SignLanguage_S2}"
FINAL_SOURCE="${FINAL_SOURCE:-/project/lt200246-mmacma/khtun/video2smplx_ablation_samples/accurate_2d/${SEQUENCE}/final_postprocessed}"
RUNTIME_SOURCE="${RUNTIME_SOURCE:-/project/lt200246-mmacma/khtun/video2smplx_ablation_samples/accurate_2d/${SEQUENCE}/runtime/integrated_postprocessed_runtime_report.json}"
GEOMETRY_SOURCE="${GEOMETRY_SOURCE:-/project/lt200246-mmacma/khtun/video2smplx_ablation_samples/evaluation/accurate_2d/${SEQUENCE}/geometry_summary.csv}"

GLOBAL_CKPT_SOURCE="${GLOBAL_CKPT_SOURCE:-outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt}"
HAND_CKPT_SOURCE="${HAND_CKPT_SOURCE:-/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt}"
UPPER2D_CKPT_SOURCE="${UPPER2D_CKPT_SOURCE:-/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt}"

FINAL_DEST="delivery/final_outputs/${SEQUENCE}"
GLOBAL_DEST="delivery/models/correctors/global/best_model.pt"
HAND_DEST="delivery/models/correctors/hand/best_model.pt"
UPPER2D_DEST="delivery/models/correctors/upper2d/best_model.pt"

require_file() {
  if [ ! -s "$1" ]; then
    echo "[error] required source is missing or empty: $1" >&2
    exit 2
  fi
}

require_file "${FINAL_SOURCE}/smplx_params.npz"
require_file "${FINAL_SOURCE}/rendered/smplx_render.mp4"
require_file "${FINAL_SOURCE}/side_by_side_input_render.mp4"
require_file "${RUNTIME_SOURCE}"
require_file "${GLOBAL_CKPT_SOURCE}"
require_file "${HAND_CKPT_SOURCE}"
require_file "${UPPER2D_CKPT_SOURCE}"

mkdir -p "${FINAL_DEST}/rendered" \
  "$(dirname "${GLOBAL_DEST}")" \
  "$(dirname "${HAND_DEST}")" \
  "$(dirname "${UPPER2D_DEST}")"

cp "${FINAL_SOURCE}/smplx_params.npz" "${FINAL_DEST}/smplx_params.npz"
cp "${FINAL_SOURCE}/rendered/smplx_render.mp4" "${FINAL_DEST}/rendered/smplx_render.mp4"
cp "${FINAL_SOURCE}/side_by_side_input_render.mp4" "${FINAL_DEST}/side_by_side_input_render.mp4"
cp "${RUNTIME_SOURCE}" "${FINAL_DEST}/runtime_report.json"

if [ -s "${FINAL_SOURCE}/combine_render_report.json" ]; then
  cp "${FINAL_SOURCE}/combine_render_report.json" "${FINAL_DEST}/combine_render_report.json"
else
  echo "[info] integrated run has no separate combine report; runtime_report.json contains settings"
fi

if [ -s "${GEOMETRY_SOURCE}" ]; then
  cp "${GEOMETRY_SOURCE}" "${FINAL_DEST}/geometry_summary.csv"
else
  echo "[warn] geometry summary not found: ${GEOMETRY_SOURCE}" >&2
fi

cp "${GLOBAL_CKPT_SOURCE}" "${GLOBAL_DEST}"
cp "${HAND_CKPT_SOURCE}" "${HAND_DEST}"
cp "${UPPER2D_CKPT_SOURCE}" "${UPPER2D_DEST}"

CHECKSUM_FILE="delivery/ASSET_SHA256SUMS.txt"
if command -v sha256sum >/dev/null 2>&1; then
  find "${FINAL_DEST}" delivery/models/correctors -type f \
    ! -name .gitkeep ! -name README.md -print0 \
    | sort -z | xargs -0 sha256sum > "${CHECKSUM_FILE}"
else
  find "${FINAL_DEST}" delivery/models/correctors -type f \
    ! -name .gitkeep ! -name README.md -print0 \
    | sort -z | xargs -0 shasum -a 256 > "${CHECKSUM_FILE}"
fi

echo "[done] final output: ${FINAL_DEST}"
echo "[done] correctors: delivery/models/correctors"
echo "[done] checksums: ${CHECKSUM_FILE}"
bash delivery/scripts/verify_delivery_readiness.sh
