# Reproducible Examples

These commands assume the server project root:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
module load Mamba/23.11.0-0
conda activate video2smplx_shared310
export PYOPENGL_PLATFORM=egl
```

## Single SignLanguage Video

```bash
RUN=runs_delivery/SignLanguage_S2
VIDEO=datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4

python delivery/source/body_estimation_cli.py \
  --video "$VIDEO" \
  --output "$RUN" \
  --device cuda \
  --detector-stride 5 \
  --batch-size 8 \
  --inference-mode \
  --single-gpu-model

python delivery/source/hand_estimation_cli.py \
  --video "$VIDEO" \
  --output "$RUN" \
  --device cuda

python delivery/source/face_estimation_cli.py \
  --video "$VIDEO" \
  --output "$RUN" \
  --device cuda

python delivery/source/combine_smooth_render_cli.py \
  --body-dir "$RUN/body_params" \
  --hand-dir "$RUN/hand_params" \
  --face-dir "$RUN/face_params" \
  --output "$RUN" \
  --input-video "$VIDEO" \
  --smplx-model SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz \
  --fps 30 \
  --gender neutral
```

## Expected Outputs

```bash
ls -lh "$RUN/smplx_params.npz"
ls -lh "$RUN/rendered/smplx_render.mp4"
ls -lh "$RUN/side_by_side_input_render.mp4"
cat "$RUN/combine_render_report.json"
```

## Optional Corrected Benchmark Pipeline

The trained SignLanguage post-processors are used after base fusion. The tested
research path was:

```text
raw body/hand/face fusion
  -> global_orient + transl corrector
  -> hand wrist/finger corrector
  -> 2D-guided upper-body corrector at scale 0.75
  -> evaluation/rendering
```

Use the matching deployment tools:

```text
tools/postprocessing/apply_signlanguage_global_correction.py
tools/postprocessing/apply_signlanguage_hand_correction.py
tools/postprocessing/apply_signlanguage_2d_guided_upperbody_corrector.py
tools/evaluation/evaluate_signlanguage_geometry.py
```

The corrector path is for benchmark reproduction and research evaluation. The
four base delivery CLIs remain independent and GT-free.

## Server Test With Latest Post-Processing

```bash
sbatch delivery/scripts/slurm/testing/test_final_postprocessed_pipeline.sh
```

Optional overrides:

```bash
SEQUENCE=SignLanguage_S3 \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_delivery_final_test \
YOLO_MODEL=/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/models/yolov8x-pose.pt \
GLOBAL_CKPT=delivery/models/correctors/global/best_model.pt \
HAND_CKPT=delivery/models/correctors/hand/best_model.pt \
UPPER2D_CKPT=delivery/models/correctors/upper2d/best_model.pt \
sbatch delivery/scripts/slurm/testing/test_final_postprocessed_pipeline.sh
```

Expected final outputs:

```text
<RUN_ROOT>/<SEQUENCE>/final_postprocessed/smplx_params.npz
<RUN_ROOT>/<SEQUENCE>/final_postprocessed/rendered/smplx_render.mp4
<RUN_ROOT>/<SEQUENCE>/final_postprocessed/side_by_side_input_render.mp4
```

## Runtime/Complexity Benchmark

Run one or more sequences through the full final pipeline and collect timing:

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_delivery_final_benchmark \
sbatch delivery/scripts/slurm/benchmarks/benchmark_final_postprocessed_pipeline.sh
```

Outputs per sequence:

```text
<RUN_ROOT>/<SEQUENCE>/runtime/stage_timings.csv
<RUN_ROOT>/<SEQUENCE>/runtime/runtime_stage_summary.csv
<RUN_ROOT>/<SEQUENCE>/runtime/model_complexity_summary.csv
<RUN_ROOT>/<SEQUENCE>/runtime/runtime_complexity_summary.json
<RUN_ROOT>/<SEQUENCE>/runtime/runtime_complexity_summary.md
```

Combined stage timings:

```text
<RUN_ROOT>/evaluation/all_sequences_stage_timings.csv
```

## Fair FPS: With vs Without Post-Processing

This benchmark avoids double-counting base rendering. It measures the base
pipeline once, then adds the global, hand, YOLO-2D, and upper-body correctors.

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_delivery_fair_fps \
YOLO_MODEL=yolov8x-pose.pt \
sbatch delivery/scripts/slurm/benchmarks/benchmark_fair_fps.sh
```

Main result:

```text
<RUN_ROOT>/evaluation/runtime_with_without_postprocessing/runtime_with_without_postprocessing.md
<RUN_ROOT>/evaluation/runtime_with_without_postprocessing/runtime_with_without_postprocessing.csv
```

Report `without_post_no_render` and `with_post_no_render` for real prediction
throughput. Report `without_post_with_render` and `with_post_with_render` when
rendering is part of the delivered artifact.

## One-Process Integrated FPS Benchmark

This is the fair comparison to the earlier `script_fusion.sh` style run because
it uses `video2smplx.integrated_pipeline` in one Python process with the same
optimized flags, then applies post-processors inside that process.

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark \
YOLO_MODEL=yolov8x-pose.pt \
sbatch delivery/scripts/slurm/benchmarks/benchmark_integrated_oneprocess.sh
```

Main result:

```text
<RUN_ROOT>/evaluation/oneprocess_runtime_summary.csv
```

## Ablation Sample Outputs

Use this when the deliverable needs visible examples for each requirement
setting: stabilization off/on, no postprocessing, global-only correction,
fast global+hand correction, and accurate 2D-guided correction.

```bash
SEQUENCES="SignLanguage_S2" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_ablation_samples \
YOLO_MODEL=yolov8x-pose.pt \
sbatch delivery/scripts/slurm/samples/generate_ablation_samples.sh
```

Default variants:

```text
base_no_stab
base_stab
global_only
fast_global_hand
accurate_2d
accurate_2d_stab
```

Each variant writes:

```text
<RUN_ROOT>/<variant>/<sequence>/final_*/smplx_params.npz
<RUN_ROOT>/<variant>/<sequence>/final_*/rendered/smplx_render.mp4
<RUN_ROOT>/<variant>/<sequence>/final_*/side_by_side_input_render.mp4
<RUN_ROOT>/<variant>/<sequence>/runtime/integrated_postprocessed_runtime_report.json
```

Summary files:

```text
<RUN_ROOT>/evaluation/ablation_sample_manifest.csv
<RUN_ROOT>/evaluation/ablation_sample_summary.csv
<RUN_ROOT>/evaluation/ablation_sample_summary.md
```

To run only selected variants:

```bash
VARIANTS="base_no_stab fast_global_hand accurate_2d" \
SEQUENCES="SignLanguage_S2 SignLanguage_S3" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_ablation_samples \
YOLO_KEYPOINTS=/project/lt200246-mmacma/khtun/video2smplx_ablation_samples/evaluation/precomputed_yolo_pose_keypoints.json \
sbatch delivery/scripts/slurm/samples/generate_ablation_samples.sh
```
