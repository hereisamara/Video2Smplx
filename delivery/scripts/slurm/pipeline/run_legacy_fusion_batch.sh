#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 16
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 120:00:00
#SBATCH -A lt200246
#SBATCH -J video2smplx
#SBATCH --array=0-5
#SBATCH --output=fusion_output.%A_%a.out
#SBATCH --error=fusion_output.%A_%a.err

set -u
set -o pipefail

module load Mamba/23.11.0-0
conda activate video2smplx_shared310

BASE_DIR="${BASE_DIR:-/lustrefs/disk/home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx}"
DATASET_NAME="${DATASET_NAME:-SignLanguage}"
DATA_DIR="${DATA_DIR:-$BASE_DIR/datasets/videos/$DATASET_NAME}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/outputs_fusion_signlanguage}"
LOG_DIR="${LOG_DIR:-$BASE_DIR/logs_signlanguage}"
BATCH_SIZE="${BATCH_SIZE:-50}"
BATCH_ID="${SLURM_ARRAY_TASK_ID:-${BATCH_ID:-0}}"
FORCE="${FORCE:-0}"

cd "$BASE_DIR" || exit 1

mkdir -p "$OUT_DIR" "$LOG_DIR"

mapfile -t VIDEOS < <(
    find "$DATA_DIR" -mindepth 2 -maxdepth 2 -type f -name "${DATASET_NAME}_S*.mp4" |
    sort -V
)

TOTAL=${#VIDEOS[@]}
START=$((BATCH_ID * BATCH_SIZE))
END=$((START + BATCH_SIZE))
if (( END > TOTAL )); then
    END=$TOTAL
fi

echo "Dataset: $DATASET_NAME"
echo "Total videos: $TOTAL"
echo "Batch id: $BATCH_ID"
echo "Batch size: $BATCH_SIZE"
echo "Processing indexes: $START to $((END - 1))"

if (( START >= TOTAL )); then
    echo "Nothing to do for batch $BATCH_ID."
    exit 0
fi

failed=0

for ((idx = START; idx < END; idx++)); do
    VIDEO="${VIDEOS[$idx]}"
    NAME="$(basename "${VIDEO%.mp4}")"
    OUTPUT="$OUT_DIR/$NAME"
    LOG="$LOG_DIR/$NAME.log"
    DONE_MARKER="$OUTPUT/.done"
    FAILED_MARKER="$OUTPUT/.failed"

    if [[ "$FORCE" != "1" && -f "$DONE_MARKER" ]]; then
        echo "Skipping $NAME because $DONE_MARKER exists. Set FORCE=1 to rerun."
        continue
    fi

    mkdir -p "$OUTPUT"
    rm -f "$FAILED_MARKER"

    echo "======================================"
    echo "Running $NAME"
    echo "Index: $idx"
    echo "Video: $VIDEO"
    echo "Output: $OUTPUT"
    echo "Log: $LOG"
    echo "======================================"

    conda run --no-capture-output -n video2smplx_shared310 python -m video2smplx.integrated_pipeline \
        --video "$VIDEO" \
        --output "$OUTPUT" \
        --device cuda \
        --stabilize_lower_body \
        --stabilize_global_orient \
        --stabilize_torso \
        --stabilize_shape \
        --smplestx_detector_stride 5 \
        --smplestx_batch_size 8 \
        --smplestx_inference_mode \
        --smplestx_single_gpu_model \
        --parallel_models > "$LOG" 2>&1

    rc=$?
    if (( rc == 0 )); then
        date > "$DONE_MARKER"
        echo "Finished $NAME"
    else
        echo "Exit code $rc" > "$FAILED_MARKER"
        echo "Failed $NAME with exit code $rc. See $LOG"
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "Batch $BATCH_ID finished with failures."
    exit 1
fi

echo "Batch $BATCH_ID done."
