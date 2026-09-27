#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SEQUENCE="${SEQUENCE:-SignLanguage_S2}"
FINAL_DIR="delivery/final_outputs/${SEQUENCE}"
FAILED=0

check_file() {
  local label="$1"
  local path="$2"
  if [ -s "${path}" ]; then
    echo "[ok] ${label}: ${path}"
  else
    echo "[missing] ${label}: ${path}" >&2
    FAILED=1
  fi
}

check_file "global corrector" "delivery/models/correctors/global/best_model.pt"
check_file "hand corrector" "delivery/models/correctors/hand/best_model.pt"
check_file "2D upper corrector" "delivery/models/correctors/upper2d/best_model.pt"

check_file "final SMPL-X parameters" "${FINAL_DIR}/smplx_params.npz"
check_file "final rendered video" "${FINAL_DIR}/rendered/smplx_render.mp4"
check_file "final side-by-side video" "${FINAL_DIR}/side_by_side_input_render.mp4"
check_file "final combine report" "${FINAL_DIR}/combine_render_report.json"
check_file "final runtime report" "${FINAL_DIR}/runtime_report.json"

for path in \
  delivery/source/body_estimation_cli.py \
  delivery/source/hand_estimation_cli.py \
  delivery/source/face_estimation_cli.py \
  delivery/source/combine_smooth_render_cli.py \
  delivery/source/integrated_postprocessed_pipeline_cli.py \
  delivery/docs/INSTALLATION.md \
  delivery/docs/MODEL_SETUP.md \
  delivery/docs/USER_DOCUMENTATION.md \
  delivery/docs/TECHNICAL_STUDY_REPORT.md \
  delivery/sample/input-sample.mp4 \
  delivery/sample/output-sample.mp4; do
  check_file "delivery file" "${path}"
done

if [ "${FAILED}" -ne 0 ]; then
  echo "[error] delivery is incomplete" >&2
  exit 2
fi

echo "[done] delivery assets and required documents are complete"
