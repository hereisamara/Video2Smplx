#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 user@server:/absolute/path/to/final_postprocessed [local_final_outputs_dir] [remote_runtime_report] [remote_geometry_summary]" >&2
  exit 2
fi

REMOTE_RUN="$1"
LOCAL_OUT="${2:-delivery/final_outputs/server_run}"
REMOTE_RUNTIME="${3:-}"
REMOTE_GEOMETRY="${4:-}"

mkdir -p "${LOCAL_OUT}/rendered"
scp "${REMOTE_RUN}/smplx_params.npz" "${LOCAL_OUT}/"
scp "${REMOTE_RUN}/rendered/smplx_render.mp4" "${LOCAL_OUT}/rendered/"
scp "${REMOTE_RUN}/side_by_side_input_render.mp4" "${LOCAL_OUT}/"
scp "${REMOTE_RUN}/combine_render_report.json" "${LOCAL_OUT}/"

if [ -n "${REMOTE_RUNTIME}" ]; then
  scp "${REMOTE_RUNTIME}" "${LOCAL_OUT}/runtime_report.json"
fi

if [ -n "${REMOTE_GEOMETRY}" ]; then
  scp "${REMOTE_GEOMETRY}" "${LOCAL_OUT}/geometry_summary.csv"
fi

echo "[done] copied server outputs to: ${LOCAL_OUT}"
