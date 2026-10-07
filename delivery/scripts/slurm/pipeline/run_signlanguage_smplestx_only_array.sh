#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -t 120:00:00
#SBATCH -A lt200246
#SBATCH -J sl_smplestx_only
#SBATCH --array=0-5
#SBATCH --output=sl_smplestx_only.%A_%a.out
#SBATCH --error=sl_smplestx_only.%A_%a.err

set -u
set -o pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

BASE_DIR="${BASE_DIR:-/home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx}"
DATASET_NAME="${DATASET_NAME:-SignLanguage}"
DATA_DIR="${DATA_DIR:-${BASE_DIR}/datasets/videos/${DATASET_NAME}}"
OUT_DIR="${OUT_DIR:-/project/lt200246-mmacma/khtun/outputs_smplestx_only_signlanguage}"
LOG_DIR="${LOG_DIR:-/project/lt200246-mmacma/khtun/logs_smplestx_only_signlanguage}"
BATCH_SIZE="${BATCH_SIZE:-50}"
BATCH_ID="${SLURM_ARRAY_TASK_ID:-${BATCH_ID:-0}}"
FORCE="${FORCE:-0}"
STABILIZE="${STABILIZE:-0}"

cd "${BASE_DIR}" || exit 1

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

mapfile -t VIDEOS < <(
    find "${DATA_DIR}" -mindepth 2 -maxdepth 2 -type f -name "${DATASET_NAME}_S*.mp4" |
    sort -V
)

TOTAL=${#VIDEOS[@]}
START=$((BATCH_ID * BATCH_SIZE))
END=$((START + BATCH_SIZE))
if (( END > TOTAL )); then
    END=${TOTAL}
fi

echo "[check] base dir: ${BASE_DIR}"
echo "[check] data dir: ${DATA_DIR}"
echo "[check] output dir: ${OUT_DIR}"
echo "[check] log dir: ${LOG_DIR}"
echo "[check] total videos: ${TOTAL}"
echo "[check] batch id: ${BATCH_ID}"
echo "[check] batch size: ${BATCH_SIZE}"
echo "[check] processing indexes: ${START} to $((END - 1))"
echo "[check] stabilize raw SMPLest-X params: ${STABILIZE}"
nvidia-smi || true

if (( START >= TOTAL )); then
    echo "Nothing to do for batch ${BATCH_ID}."
    exit 0
fi

failed=0

for ((idx = START; idx < END; idx++)); do
    VIDEO="${VIDEOS[$idx]}"
    NAME="$(basename "${VIDEO%.mp4}")"
    OUTPUT="${OUT_DIR}/${NAME}"
    LOG="${LOG_DIR}/${NAME}.log"
    DONE_MARKER="${OUTPUT}/.smplestx_only.done"
    FAILED_MARKER="${OUTPUT}/.smplestx_only.failed"

    if [[ "${FORCE}" != "1" && -f "${DONE_MARKER}" ]]; then
        echo "Skipping ${NAME} because ${DONE_MARKER} exists. Set FORCE=1 to rerun."
        continue
    fi

    mkdir -p "${OUTPUT}"
    rm -f "${FAILED_MARKER}"

    EXTRA_ARGS=()
    if [[ "${STABILIZE}" == "1" ]]; then
        EXTRA_ARGS+=(
            --stabilize_lower_body
            --stabilize_global_orient
            --stabilize_torso
            --stabilize_shape
        )
    fi

    echo "======================================"
    echo "Running ${NAME}"
    echo "Index: ${idx}"
    echo "Video: ${VIDEO}"
    echo "Output: ${OUTPUT}"
    echo "Log: ${LOG}"
    echo "======================================"

    conda run --no-capture-output -n video2smplx_shared310 python -m video2smplx.integrated_pipeline \
        --video "${VIDEO}" \
        --output "${OUTPUT}" \
        --device cuda \
        --smplestx_only \
        --smplestx_detector_stride 5 \
        --smplestx_batch_size 8 \
        --smplestx_inference_mode \
        --smplestx_single_gpu_model \
        "${EXTRA_ARGS[@]}" > "${LOG}" 2>&1

    rc=$?
    if (( rc == 0 )); then
        date > "${DONE_MARKER}"
        echo "Finished ${NAME}"
    else
        echo "Exit code ${rc}" > "${FAILED_MARKER}"
        echo "Failed ${NAME} with exit code ${rc}. See ${LOG}"
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "Batch ${BATCH_ID} finished with failures."
    exit 1
fi

echo "Batch ${BATCH_ID} done."
