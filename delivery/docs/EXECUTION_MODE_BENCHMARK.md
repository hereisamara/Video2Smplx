# Model Execution Mode Benchmark

This benchmark compares three execution schedules while keeping the same
SMPLest-X, WiLoR, EMOCA, fusion, and learned post-processing stack.

## Modes

| Variant | Behavior | Intended use |
| --- | --- | --- |
| `frame_barrier` | Run body, hand, and face concurrently for one frame, fuse it, then start the next frame. | Lowest buffering and current baseline. |
| `stream2` / `stream4` | Give each model a persistent worker and keep 2 or 4 frames in flight. Results remain in memory and are fused in frame order. | Streaming video with bounded latency and memory. |
| `batch_all` | Decode the video, batch each model over the full frame set in micro-batches, then fuse after all three result streams finish. | Maximum offline throughput. |

`batch_all` uses independent micro-batches for body, detected hand crops, and
face crops. It does not place every video frame in one GPU tensor. Batch sizes
remain bounded to avoid exhausting GPU memory.

## Run on the server

```bash
git pull origin delivery-readme-video-comparison

bash -n delivery/scripts/slurm_benchmark_execution_modes.sh

SEQUENCES="SignLanguage_S2" \
REPEATS=1 \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_execution_mode_benchmark \
sbatch delivery/scripts/slurm_benchmark_execution_modes.sh
```

Use three repeats for reportable numbers:

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
REPEATS=3 \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_execution_mode_benchmark_r3 \
sbatch delivery/scripts/slurm_benchmark_execution_modes.sh
```

## Results

```bash
cat /project/lt200246-mmacma/khtun/video2smplx_execution_mode_benchmark/evaluation/execution_mode_summary.csv
```

The CSV reports:

- Full FPS including model loading.
- FPS after subtracting initial model loading.
- Base inference plus fusion FPS.
- Warm pipeline FPS.
- Fusion/combination milliseconds per frame.
- Global, hand, and 2D upper-body correction time.
- Final smoothing and NPZ export time.
- All downstream post-processing and export time.

Rendering is disabled because it is an optional consumer of the final
parameters. Fusion, all correction models, temporal smoothing, and final NPZ
export remain included.

## Memory tradeoff

Streaming keeps only a bounded number of source frames and pending results.
`batch_all` retains all decoded frames and all three parameter streams until
fusion completes, so its memory use grows with video length. Use chunked videos
for long recordings if full-video buffering is too large.
