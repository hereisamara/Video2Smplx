#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/artifacts/delivery_packages}"
STAMP="$(date +%Y%m%d_%H%M%S)"

required_source_files=(
  "README.md"
  "delivery/source/body_estimation_cli.py"
  "delivery/source/hand_estimation_cli.py"
  "delivery/source/face_estimation_cli.py"
  "delivery/source/combine_smooth_render_cli.py"
  "delivery/source/integrated_postprocessed_pipeline_cli.py"
  "delivery/docs/FINAL_DELIVERY_REPORT.md"
  "delivery/docs/MODEL_SETUP.md"
  "delivery/docs/TECHNICAL_STUDY_REPORT.md"
  "delivery/sample/input-sample.mp4"
  "delivery/sample/output-sample.mp4"
)

for relative_path in "${required_source_files[@]}"; do
  if [ ! -s "${ROOT_DIR}/${relative_path}" ]; then
    echo "[error] required source file is missing or empty: ${relative_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUT_DIR}"
DOC_SET="${OUT_DIR}/Video2Smplx_document_set_${STAMP}"
mkdir -p "${DOC_SET}"

# Package only Git-tracked source and documentation. Runtime weights and final
# generated outputs are delivered in separate archives and overlaid later.
TRACKED_LIST="$(mktemp)"
trap 'rm -f "${TRACKED_LIST}"' EXIT
while IFS= read -r -d '' relative_path; do
  case "${relative_path}" in
    delivery/models/correctors/*|delivery/final_outputs/*) continue ;;
    SMPLest-X-Inference/README.md|SMPLest-X-Inference/SHARED_ENV_CHANGELOG.md) continue ;;
    SMPLest-X-Inference/pretrained_models/*) continue ;;
    SMPLest-X-Inference/human_models/human_model_files/*) continue ;;
    WiLoR-Inference/README.md|WiLoR-Inference/SHARED_ENV_CHANGELOG.md) continue ;;
    WiLoR-Inference/pretrained_models/*|WiLoR-Inference/mano_data/*) continue ;;
    EMOCA-Inference/README.md|EMOCA-Inference/SHARED_ENV_CHANGELOG.md) continue ;;
    EMOCA-Inference/assets/*) continue ;;
  esac
  if [ -e "${ROOT_DIR}/${relative_path}" ]; then
    printf '%s\0' "${relative_path}"
  fi
done < <(git -C "${ROOT_DIR}" ls-files -z) > "${TRACKED_LIST}"

rsync -a --from0 --files-from="${TRACKED_LIST}" "${ROOT_DIR}/" "${DOC_SET}/"

GIT_REVISION="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
if git -C "${ROOT_DIR}" diff --quiet && git -C "${ROOT_DIR}" diff --cached --quiet; then
  SOURCE_STATE="clean"
else
  SOURCE_STATE="modified tracked files included"
fi
cat > "${DOC_SET}/SOURCE_MANIFEST.txt" <<EOF
Video2Smplx source and documentation package
Git revision: ${GIT_REVISION}
Source state: ${SOURCE_STATE}
Created UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)

Runtime model weights are intentionally excluded. Extract the separate
Video2Smplx_runtime_models_*.tar archive and overlay its contents on this
directory before running inference. See README.md section 6.
EOF

COPYFILE_DISABLE=1 tar --no-xattrs -czf "${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz" -C "${OUT_DIR}" "$(basename "${DOC_SET}")"
shasum -a 256 "${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz" \
  > "${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz.sha256"

echo "[done] document set folder: ${DOC_SET}"
echo "[done] archive: ${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz"
echo "[done] checksum: ${OUT_DIR}/Video2Smplx_document_set_${STAMP}.tar.gz.sha256"
