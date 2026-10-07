#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
YOLO_POSE_MODEL="${YOLO_POSE_MODEL:-${ROOT_DIR}/yolov8x-pose.pt}"
TORCH_HUB_DIR="${TORCH_HUB_DIR:-${ROOT_DIR}/runtime_model_cache/torch/hub}"
CORRECTOR_ROOT="${CORRECTOR_ROOT:-${ROOT_DIR}}"
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

check_file "SMPLest-X config" "${ROOT_DIR}/SMPLest-X-Inference/pretrained_models/smplest_x_h/config_base.py"
check_file "SMPLest-X checkpoint" "${ROOT_DIR}/SMPLest-X-Inference/pretrained_models/smplest_x_h/smplest_x_h.pth.tar"
check_file "SMPLest-X YOLO detector" "${ROOT_DIR}/SMPLest-X-Inference/pretrained_models/yolov8x.pt"

for name in \
  SMPLX_NEUTRAL.npz \
  SMPLX_MALE.npz \
  SMPLX_FEMALE.npz \
  SMPLX_to_J14.pkl \
  MANO_SMPLX_vertex_ids.pkl \
  SMPL-X__FLAME_vertex_ids.npy; do
  check_file "SMPL-X asset" "${ROOT_DIR}/SMPLest-X-Inference/human_models/human_model_files/smplx/${name}"
done

check_file "WiLoR checkpoint" "${ROOT_DIR}/WiLoR-Inference/pretrained_models/wilor_final.ckpt"
check_file "WiLoR detector" "${ROOT_DIR}/WiLoR-Inference/pretrained_models/detector.pt"
check_file "WiLoR configuration" "${ROOT_DIR}/WiLoR-Inference/pretrained_models/model_config.yaml"
check_file "MANO model" "${ROOT_DIR}/WiLoR-Inference/mano_data/MANO_RIGHT.pkl"
check_file "MANO mean parameters" "${ROOT_DIR}/WiLoR-Inference/mano_data/mano_mean_params.npz"

EMOCA_MODEL="${ROOT_DIR}/EMOCA-Inference/assets/EMOCA/models/EMOCA_v2_lr_mse_20"
check_file "EMOCA configuration" "${EMOCA_MODEL}/cfg.yaml"
EMOCA_CHECKPOINT="$(find "${EMOCA_MODEL}/detail/checkpoints" -type f \( -name '*.ckpt' -o ! -name '*.*' \) -print -quit 2>/dev/null || true)"
if [ -n "${EMOCA_CHECKPOINT}" ] && [ -s "${EMOCA_CHECKPOINT}" ]; then
  echo "[ok] EMOCA checkpoint: ${EMOCA_CHECKPOINT}"
else
  echo "[missing] EMOCA checkpoint under ${EMOCA_MODEL}/detail/checkpoints" >&2
  FAILED=1
fi

check_file "DECA model" "${ROOT_DIR}/EMOCA-Inference/assets/DECA/data/deca_model.tar"
check_file "FLAME model" "${ROOT_DIR}/EMOCA-Inference/assets/FLAME/geometry/generic_model.pkl"
check_file "FLAME landmarks" "${ROOT_DIR}/EMOCA-Inference/assets/FLAME/geometry/landmark_embedding.npy"
check_file "FLAME MediaPipe landmarks" "${ROOT_DIR}/EMOCA-Inference/assets/FLAME/geometry/mediapipe_landmark_embedding.npz"
check_file "FLAME head template" "${ROOT_DIR}/EMOCA-Inference/assets/FLAME/geometry/head_template.obj"
check_file "FLAME displacement" "${ROOT_DIR}/EMOCA-Inference/assets/FLAME/geometry/fixed_uv_displacements/fixed_displacement_256.npy"
check_file "FLAME face mask" "${ROOT_DIR}/EMOCA-Inference/assets/FLAME/mask/uv_face_mask.png"
check_file "FLAME eye mask" "${ROOT_DIR}/EMOCA-Inference/assets/FLAME/mask/uv_face_eye_mask.png"
check_file "face-recognition weights" "${ROOT_DIR}/EMOCA-Inference/assets/FaceRecognition/resnet50_ft_weight.pkl"

FAN_CHECKPOINT="$(find "${TORCH_HUB_DIR}/checkpoints" -type f -iname '*fan*' -print -quit 2>/dev/null || true)"
SFD_CHECKPOINT="$(find "${TORCH_HUB_DIR}/checkpoints" -type f -iname 's3fd*' -print -quit 2>/dev/null || true)"
if [ -n "${FAN_CHECKPOINT}" ] && [ -s "${FAN_CHECKPOINT}" ]; then
  echo "[ok] FAN landmark checkpoint: ${FAN_CHECKPOINT}"
else
  echo "[missing] FAN landmark checkpoint under ${TORCH_HUB_DIR}/checkpoints" >&2
  FAILED=1
fi
if [ -n "${SFD_CHECKPOINT}" ] && [ -s "${SFD_CHECKPOINT}" ]; then
  echo "[ok] SFD face-detector checkpoint: ${SFD_CHECKPOINT}"
else
  echo "[missing] SFD face-detector checkpoint under ${TORCH_HUB_DIR}/checkpoints" >&2
  FAILED=1
fi

check_file "YOLO pose model" "${YOLO_POSE_MODEL}"
check_file "global corrector" "${CORRECTOR_ROOT}/delivery/models/correctors/global/best_model.pt"
check_file "hand corrector" "${CORRECTOR_ROOT}/delivery/models/correctors/hand/best_model.pt"
check_file "2D upper corrector" "${CORRECTOR_ROOT}/delivery/models/correctors/upper2d/best_model.pt"

if [ "${FAILED}" -ne 0 ]; then
  echo "[error] runtime model set is incomplete" >&2
  exit 2
fi

echo "[done] all runtime models are present"
