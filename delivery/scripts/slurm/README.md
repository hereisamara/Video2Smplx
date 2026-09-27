# Slurm Job Guide

All maintained Slurm jobs are stored below this directory. Submit jobs from the
repository root because the scripts use repository-relative Python paths.

Generated delivery archives and historical handover folders may contain frozen
copies of older jobs. Those copies are not maintained and should not be used.

## Common Server Setup

The jobs currently target the LANTA configuration used by this project:

```text
partition: gpu
account: lt200246
environment: video2smplx_shared310
repository: /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
```

Before submitting a job:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
module load Mamba/23.11.0-0
bash -n delivery/scripts/slurm/<category>/<job>.sh
sbatch delivery/scripts/slurm/<category>/<job>.sh
```

Override a job setting by placing environment variables before `sbatch`:

```bash
SEQUENCE=SignLanguage_S2 \
OUTPUT=/project/lt200246-mmacma/khtun/example_run \
sbatch delivery/scripts/slurm/pipeline/run_configurable_pipeline.sh
```

## Pipeline Jobs

| Job | Purpose | Main controls |
| --- | --- | --- |
| `pipeline/run_configurable_pipeline.sh` | Recommended configurable one-process pipeline. | `SEQUENCE`, `OUTPUT`, `EXECUTION_MODE`, `SMPLESTX_ONLY`, `ENABLE_WILOR`, `ENABLE_EMOCA`, `POSTPROCESS_MODE`, `ENABLE_SMOOTHING`, `ENABLE_RENDER` |
| `pipeline/run_signlanguage_smplestx_only_array.sh` | Run raw SMPLest-X over the SignLanguage collection as a six-task job array. | `DATA_DIR`, `OUT_DIR`, `LOG_DIR`, `BATCH_SIZE`, `FORCE`, `STABILIZE` |
| `pipeline/run_signify_smplestx_only.sh` | Run raw SMPLest-X over extracted Signify frame sequences. | `FRAMES_ROOT`, `OUTPUT_ROOT`, `LOG_ROOT`, `BATCHES` |
| `pipeline/run_legacy_fusion_batch.sh` | Historical SignLanguage full-fusion job array retained for reproducibility. | `DATA_DIR`, `OUT_DIR`, `LOG_DIR`, `BATCH_SIZE`, `FORCE` |

Recommended final-pipeline example:

```bash
SEQUENCE=SignLanguage_S2 \
EXECUTION_MODE=batch \
POSTPROCESS_MODE=accurate \
ENABLE_WILOR=1 \
ENABLE_EMOCA=1 \
ENABLE_SMOOTHING=1 \
ENABLE_RENDER=1 \
sbatch delivery/scripts/slurm/pipeline/run_configurable_pipeline.sh
```

SMPLest-X-only example:

```bash
SEQUENCE=SignLanguage_S2 \
SMPLESTX_ONLY=1 \
EXECUTION_MODE=batch \
sbatch delivery/scripts/slurm/pipeline/run_configurable_pipeline.sh
```

## Preprocessing Jobs

| Job | Purpose | Main controls |
| --- | --- | --- |
| `preprocessing/precompute_signlanguage_yolo_keypoints.sh` | Extract reusable YOLO pose keypoints for the 2D-guided corrector. | `SEQUENCES`, `VIDEO_DIR`, `RUN_ROOT`, `YOLO_MODEL` |

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_keypoints \
YOLO_MODEL=yolov8x-pose.pt \
sbatch delivery/scripts/slurm/preprocessing/precompute_signlanguage_yolo_keypoints.sh
```

## Evaluation Jobs

| Job | Purpose | Main controls |
| --- | --- | --- |
| `evaluation/evaluate_raw_smplestx_signlanguage.sh` | Evaluate saved raw SMPLest-X parameter PKLs on SignLanguage. | `PRED_ROOT`, `PARAMS_SUBDIR`, `MODEL_PATH`, `OUTPUT_DIR` |
| `evaluation/evaluate_2d_guided_splits.sh` | Evaluate trained 2D-guided outputs on explicit dataset splits. | `OUTPUT_ROOT`, `DATASET_PATH`, `SPLITS`, `PARAMS_SUBDIRS` |
| `evaluation/evaluate_signify_geometry.sh` | Evaluate saved predictions against Signify SMPL-X ground truth. | `FRAMES_ROOT`, `GT_ROOT`, `PRED_ROOT`, `SEQUENCES` |
| `evaluation/diagnose_signlanguage_error_sources.sh` | Produce body-region and parameter-error diagnostics. | `PRED_ROOT`, `PARAMS_SUBDIR`, `GROUPS`, `FRAME_STRIDE` |

```bash
PRED_ROOT=/project/lt200246-mmacma/khtun/outputs_smplestx_only_signlanguage \
PARAMS_SUBDIR=smplestx_params \
sbatch delivery/scripts/slurm/evaluation/evaluate_raw_smplestx_signlanguage.sh
```

The Signify and error-source jobs are research utilities. They require their
matching evaluator modules and downloaded datasets in the repository checkout.

## Ablation Jobs

| Job | Purpose |
| --- | --- |
| `ablations/evaluate_signlanguage_ablation.sh` | Compare raw, fusion, global, hand, and 2D-guided outputs on the held-out split. |

Training and training-ablation jobs are not delivery operations. Their archived
copies are clearly marked under `research/signlanguage_training/slurm/`.

## Benchmark Jobs

| Job | Comparison |
| --- | --- |
| `benchmarks/benchmark_final_postprocessed_pipeline.sh` | Stage-by-stage final corrected pipeline runtime |
| `benchmarks/benchmark_fair_fps.sh` | Fair base-versus-postprocessed FPS comparison |
| `benchmarks/benchmark_integrated_oneprocess.sh` | One-process integrated pipeline throughput |
| `benchmarks/benchmark_time_optimized_pipeline.sh` | Legacy I/O, fast in-memory I/O, and batched execution |
| `benchmarks/benchmark_execution_modes.sh` | Frame barrier, bounded streaming, and full-video batch modes |
| `benchmarks/benchmark_10fps_detector_stride.sh` | Detector-stride variants against a target FPS |

Use at least three repeats for reportable runtime results. Rendering should be
disabled for model-throughput comparisons and measured separately when the
complete delivery latency is required.

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
REPEATS=3 \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_execution_modes_r3 \
sbatch delivery/scripts/slurm/benchmarks/benchmark_execution_modes.sh
```

## Sample Generation

| Job | Purpose | Main controls |
| --- | --- | --- |
| `samples/generate_ablation_samples.sh` | Generate parameter files, renders, side-by-side videos, metrics, and a manifest for delivery ablations. | `SEQUENCES`, `RUN_ROOT`, `YOLO_MODEL` |

```bash
SEQUENCES="SignLanguage_S2" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_ablation_samples \
YOLO_MODEL=yolov8x-pose.pt \
sbatch delivery/scripts/slurm/samples/generate_ablation_samples.sh
```

## Visualization Jobs

| Job | Purpose | Main controls |
| --- | --- | --- |
| `visualization/render_signlanguage_before_after_mesh.sh` | Render random GT/before/after mesh GIFs and sample sheets. | `BEFORE_ROOT`, `AFTER_ROOT`, `SEQUENCES`, `SAMPLE_COUNT`, `RANDOM_SEED` |

## Test Jobs

| Job | Purpose |
| --- | --- |
| `testing/test_integrated_pipeline.sh` | Smoke-test the base integrated pipeline and expected output structure. |
| `testing/test_final_postprocessed_pipeline.sh` | Smoke-test the complete corrected pipeline and final artifacts. |
| `testing/test_packaged_delivery.sh` | Extract the final archive in a clean directory, verify assets/imports, and optionally run the accurate sample pipeline. |

Run the tests before benchmarks after changing a model, environment, or server
checkout.

Run the clean-package test after building the final archive:

```bash
PACKAGE_ARCHIVE=/absolute/path/to/Video2Smplx_document_set_<STAMP>.tar.gz \
YOLO_MODEL=/absolute/path/to/yolov8x-pose.pt \
EVAL_MODEL=/absolute/path/to/SMPLX_FEMALE.npz \
RUN_FULL_PIPELINE=1 \
sbatch delivery/scripts/slurm/testing/test_packaged_delivery.sh
```

Licensed base-model repositories are linked from `SOURCE_REPO`; they are not
copied into the delivery archive.

## Runtime Outputs

Slurm stdout and stderr are written in the repository working directory using
the names declared by each job's `#SBATCH --output` and `#SBATCH --error`
directives. Model artifacts are normally written under `/project`, not `/home`,
to avoid home-directory quota pressure.

For configurable pipeline jobs, inspect:

```text
<output>/runtime/integrated_postprocessed_runtime_report.json
<output>/runtime/integrated_postprocessed_runtime_report.csv
<output>/runtime/runtime_phase_summary.csv
```

`runtime_phase_summary.csv` separates model loading, first-forward warm-up,
normal working throughput, and cold end-to-end throughput.

## Maintenance Rule

New maintained Slurm jobs must be placed in the appropriate subfolder here and
added to this README. Do not add new Slurm jobs at the repository root or
directly under `delivery/scripts/`.
