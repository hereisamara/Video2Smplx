#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 user@server:/absolute/path/to/run_output [local_final_outputs_dir]" >&2
  exit 2
fi

REMOTE_RUN="$1"
LOCAL_OUT="${2:-delivery/final_outputs/server_run}"

mkdir -p "${LOCAL_OUT}"
scp "${REMOTE_RUN}/smplx_params.npz" "${LOCAL_OUT}/"
scp "${REMOTE_RUN}/rendered/smplx_render.mp4" "${LOCAL_OUT}/"
scp "${REMOTE_RUN}/side_by_side_input_render.mp4" "${LOCAL_OUT}/"
scp "${REMOTE_RUN}/combine_render_report.json" "${LOCAL_OUT}/"

echo "[done] copied server outputs to: ${LOCAL_OUT}"
