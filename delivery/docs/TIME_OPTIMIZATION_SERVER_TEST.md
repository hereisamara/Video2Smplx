# Time-Optimization Server Test

This test compares the same accurate SignLanguage pipeline under three execution modes. All modes use the same model checkpoints, precomputed YOLO keypoints, correction scale, smoothing, and final `smplx_params.npz` export. Rendering is disabled so GPU inference and parameter-processing time can be compared directly.

| Variant | Intermediate PKLs | Model scheduling |
| --- | --- | --- |
| `legacy_parallel` | Fused, global, hand, upper-body, and render PKLs | SMPLest-X, WiLoR, and EMOCA run concurrently per frame |
| `fastio_parallel` | Final NPZ only | Same concurrent per-frame model scheduling |
| `fastio_batched` | Final NPZ only | SMPLest-X batch size 8, followed by WiLoR and EMOCA |

## Run One Sequence

From the server repository root:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx

SEQUENCES="SignLanguage_S2" \
REPEATS=1 \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_time_optimization_benchmark \
YOLO_KEYPOINTS=/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json \
sbatch delivery/scripts/slurm_benchmark_time_optimized_pipeline.sh
```

Monitor it with:

```bash
myqueue
tail -f v2sx_timeopt.<JOB_ID>.out
tail -f v2sx_timeopt.<JOB_ID>.err
```

## Run Three Sequences

After the S2 smoke test succeeds:

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
REPEATS=2 \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_time_optimization_benchmark_s2_s3_s4 \
YOLO_KEYPOINTS=/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json \
sbatch delivery/scripts/slurm_benchmark_time_optimized_pipeline.sh
```

## Results

The script writes:

```text
<RUN_ROOT>/evaluation/time_optimization_summary.csv
```

Inspect the main fields with:

```bash
column -s, -t < "$RUN_ROOT/evaluation/time_optimization_summary.csv" | less -S
```

Use `no_initial_load_total_fps` for the production comparison. It excludes the SMPLest-X, WiLoR, and EMOCA initialization time but includes correction, smoothing, and final NPZ export. Use `steady_state_model_fps` only to compare foundation-model scheduling; it does not include the correction stack or final export.

The optimized mode intentionally does not create intermediate corrected PKL directories. Its deliverable parameter file is:

```text
<RUN_ROOT>/<SEQUENCE>/<VARIANT>_r<REPEAT>/final_postprocessed/smplx_params.npz
```

For a debugging run that also saves final smoothed PKLs, add `--save-final-pkls` to the Python command. Do not use that option for the speed comparison.

## Expected Interpretation

- `fastio_parallel` versus `legacy_parallel` measures the benefit of keeping correction data in memory and suppressing per-frame timing lines.
- `fastio_batched` versus `fastio_parallel` measures whether SMPLest-X batch size 8 is more valuable than concurrent same-frame execution on the server GPU.
- A difference smaller than about 3% should be treated as run-to-run noise. Use two or more repeats before selecting a production mode.
- This test uses precomputed YOLO keypoints. YOLO extraction time must be reported separately for fully online processing of an unseen video.

## 10 FPS Detector-Stride Test

The foundation models remain the limiting stage after fast I/O. The next test
reuses slowly changing WiLoR hand-detection boxes and EMOCA face crops while
still estimating MANO hand parameters and EMOCA face parameters on every frame.

```bash
SEQUENCES="SignLanguage_S2" \
REPEATS=1 \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_10fps_detector_stride \
YOLO_KEYPOINTS=/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json \
sbatch delivery/scripts/slurm_benchmark_10fps_detector_stride.sh
```

The variants are `baseline`, `face5`, `hand3_face5`, and `hand5_face5`. The
result is written to:

```text
<RUN_ROOT>/evaluation/target_10fps_summary.csv
```

Use `warm_pipeline_fps` as the target metric. It combines warmup-excluded model
time with measured correction, smoothing, and NPZ export time. A configuration
must also pass geometry evaluation before it replaces the baseline.
