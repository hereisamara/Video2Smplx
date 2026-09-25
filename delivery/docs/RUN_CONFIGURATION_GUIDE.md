# Pipeline Run Configuration

Use `delivery/source/integrated_postprocessed_pipeline_cli.py` as the unified
program. It always writes a final `smplx_params.npz`; rendering is optional.

## Component controls

| Requirement | CLI option |
| --- | --- |
| Raw SMPLest-X alone | `--smplestx-only` |
| Disable WiLoR/MANO hands | `--disable-wilor` or `--disable-mano` |
| Disable EMOCA face | `--disable-emoca` |
| Realtime frame barrier | `--execution-mode realtime` |
| Bounded streaming pipeline | `--execution-mode streaming --max-inflight-frames 2` |
| Offline batched inference | `--execution-mode batch` |
| Disable all learned correctors | `--disable-postprocessing` |
| Global corrector only | `--postprocess-mode global` |
| Global and hand correctors | `--postprocess-mode global_hand` |
| Global, hand, and 2D upper correctors | `--postprocess-mode accurate` |
| Disable temporal smoothing | `--disable-smoothing` |
| Preserve predicted translation | `--disable-zero-translation` |
| Skip mesh video rendering | `--skip-render` |
| Save final per-frame PKLs | `--save-final-pkls` |

WiLoR estimates MANO hand pose. Consequently, `--disable-wilor` and
`--disable-mano` are aliases for the same pipeline component.

`--smplestx-only` is a strict convenience preset. It disables WiLoR/MANO,
EMOCA, learned correctors, zero-translation, and smoothing, while retaining
the final NPZ export.

## Common arguments

Set these paths once on the server:

```bash
VIDEO=datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4
OUT=/project/lt200246-mmacma/khtun/config_test/SignLanguage_S2
YOLO_JSON=/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json
```

All examples assume:

```bash
module load Mamba/23.11.0-0
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON="conda run --no-capture-output -n video2smplx_shared310 python"
```

## SMPLest-X alone

```bash
${PYTHON} delivery/source/integrated_postprocessed_pipeline_cli.py \
  --video "${VIDEO}" \
  --output "${OUT}_smplestx_only" \
  --smplestx-only \
  --execution-mode batch \
  --smplestx-batch-size 8 \
  --fast-io \
  --skip-render
```

## Full realtime-style pipeline

```bash
${PYTHON} delivery/source/integrated_postprocessed_pipeline_cli.py \
  --video "${VIDEO}" \
  --output "${OUT}_realtime" \
  --sequence SignLanguage_S2 \
  --execution-mode realtime \
  --postprocess-mode accurate \
  --precomputed-yolo-keypoints "${YOLO_JSON}" \
  --fast-io \
  --skip-render
```

This mode waits for body, hand, and face results for the current frame before
advancing the output stream.

## Bounded streaming pipeline

```bash
${PYTHON} delivery/source/integrated_postprocessed_pipeline_cli.py \
  --video "${VIDEO}" \
  --output "${OUT}_streaming" \
  --sequence SignLanguage_S2 \
  --execution-mode streaming \
  --max-inflight-frames 2 \
  --postprocess-mode accurate \
  --precomputed-yolo-keypoints "${YOLO_JSON}" \
  --fast-io \
  --skip-render
```

Model workers continue processing later frames while completed body, hand,
and face parameters wait in memory for ordered fusion.

## Offline batch pipeline

```bash
${PYTHON} delivery/source/integrated_postprocessed_pipeline_cli.py \
  --video "${VIDEO}" \
  --output "${OUT}_batch" \
  --sequence SignLanguage_S2 \
  --execution-mode batch \
  --smplestx-batch-size 8 \
  --wilor-batch-size 16 \
  --emoca-batch-size 16 \
  --postprocess-mode accurate \
  --precomputed-yolo-keypoints "${YOLO_JSON}" \
  --fast-io \
  --skip-render
```

The offline schedule completes the body, hand, and face passes first and then
combines their in-memory outputs. Fusion and every enabled postprocessor remain
included in the runtime report.

## Disable selected components

Body and face only, without learned optimization or smoothing:

```bash
${PYTHON} delivery/source/integrated_postprocessed_pipeline_cli.py \
  --video "${VIDEO}" \
  --output "${OUT}_body_face" \
  --execution-mode realtime \
  --disable-wilor \
  --disable-postprocessing \
  --disable-smoothing \
  --disable-zero-translation \
  --fast-io \
  --skip-render
```

Body and hands only:

```bash
${PYTHON} delivery/source/integrated_postprocessed_pipeline_cli.py \
  --video "${VIDEO}" \
  --output "${OUT}_body_hands" \
  --execution-mode realtime \
  --disable-emoca \
  --postprocess-mode global_hand \
  --fast-io \
  --skip-render
```

## Outputs and timing

Each run writes:

```text
<output>/final_*/smplx_params.npz
<output>/runtime/integrated_postprocessed_runtime_report.json
<output>/runtime/integrated_postprocessed_runtime_report.csv
<output>/runtime/runtime_phase_summary.csv
```

The JSON report records every enabled component and all effective settings.
The runtime includes fusion, enabled correctors, smoothing, and final NPZ
export. Initial model loading and optional rendering are reported separately.

`runtime_phase_summary.csv` separates delivery timing into:

- `startup_model_load`: imports, checkpoint loading, and model construction.
- `first_forward_warmup`: the first complete batch in batch mode, or the
  configured number of initial frames in realtime/streaming modes.
- `normal_working`: steady model throughput after warm-up. This is the FPS to
  report as normal continuous operation.
- `cold_end_to_end`: one-shot execution including startup and final output.

Batch mode excludes one complete first batch by default. Override this with
`WARMUP_EXCLUDE_BATCHES`; realtime and streaming modes use
`WARMUP_EXCLUDE_FRAMES`.

## Configurable Slurm launcher

The same controls are available as environment variables:

```bash
SEQUENCE=SignLanguage_S2 \
EXECUTION_MODE=batch \
ENABLE_WILOR=1 \
ENABLE_EMOCA=1 \
POSTPROCESS_MODE=accurate \
ENABLE_SMOOTHING=1 \
ZERO_TRANSLATION=1 \
ENABLE_RENDER=0 \
sbatch delivery/scripts/slurm/pipeline/run_configurable_pipeline.sh
```

SMPLest-X-only example:

```bash
SEQUENCE=SignLanguage_S2 \
SMPLESTX_ONLY=1 \
EXECUTION_MODE=batch \
sbatch delivery/scripts/slurm/pipeline/run_configurable_pipeline.sh
```
