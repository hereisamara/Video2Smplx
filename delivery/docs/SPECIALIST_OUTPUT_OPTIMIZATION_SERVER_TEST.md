# WiLoR And EMOCA Output Optimization Test

This test targets approximately `80-90 ms/frame` or less for both WiLoR and
EMOCA while preserving the SMPL-X parameters used by fusion.

The optimized WiLoR path keeps the backbone, temporary MANO mesh, and
refinement network, then skips final vertices, joints, and 2D projection. The
optimized EMOCA path keeps the coarse FLAME encoder outputs and skips the
detail encoder because the fusion pipeline uses expression and jaw pose only.

Run on one sequence first:

```bash
SEQUENCES="SignLanguage_S2" \
REPEATS=2 \
TARGET_MS=90 \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_specialist_output_optimization \
YOLO_KEYPOINTS=/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark_prekeypoints/evaluation/precomputed_yolo_pose_keypoints.json \
sbatch delivery/scripts/slurm/benchmarks/benchmark_specialist_output_optimization.sh
```

Monitor:

```bash
myqueue
tail -f v2sx_specialist.<JOB_ID>.out
tail -f v2sx_specialist.<JOB_ID>.err
```

Results:

```text
<RUN_ROOT>/evaluation/specialist_output_optimization_summary.csv
<RUN_ROOT>/evaluation/specialist_output_optimization_equivalence.csv
```

The four variants are baseline, EMOCA expression-only, WiLoR parameters-only,
and both optimizations together. Runs use the same batch execution mode,
checkpoints, precomputed YOLO keypoints, correction stack, smoothing, and final
NPZ export.

Adopt an optimization only when:

1. its repeated timing is lower than baseline by more than normal run-to-run
   variation;
2. its `max_abs_param_diff` is within the configured tolerance;
3. frame IDs and valid-frame masks match; and
4. the final geometry evaluation does not regress.
