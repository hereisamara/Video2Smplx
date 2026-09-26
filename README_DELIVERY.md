# Video2Smplx Sign-Language SMPL-X Delivery README

This README is the reviewer-facing guide for the delivered Video2Smplx
sign-language system. It connects the source code, model setup, objective
metrics, qualitative sample videos, ablation outputs, and UBody literature
comparison in one place.

## 1. Delivery Scope

The project estimates expressive SMPL-X parameters from a monocular
sign-language video. The delivered system contains independent body, hand, and
face estimation programs, a parameter combination and rendering program, and an
integrated final pipeline with optional post-processing correctors.

| Requirement | Delivered file or output |
| --- | --- |
| Body estimation CLI | `delivery/source/body_estimation_cli.py` |
| Hand estimation CLI | `delivery/source/hand_estimation_cli.py` |
| Face estimation CLI | `delivery/source/face_estimation_cli.py` |
| Parameter combination, temporal smoothing, and rendering CLI | `delivery/source/combine_smooth_render_cli.py` |
| Integrated final pipeline | `delivery/source/integrated_postprocessed_pipeline_cli.py` |
| Combined SMPL-X parameter output | `<run>/final_*/smplx_params.npz` |
| Rendered SMPL-X video | `<run>/final_*/rendered/smplx_render.mp4` |
| Side-by-side input/render video | `<run>/final_*/side_by_side_input_render.mp4` |
| Installation manual | `delivery/docs/INSTALLATION.md` |
| Model setup instructions | `delivery/docs/MODEL_SETUP.md` |
| User documentation | `delivery/docs/USER_DOCUMENTATION.md` |
| Literature and technical study report | `delivery/docs/TECHNICAL_STUDY_REPORT.md` |
| Reproducible examples | `delivery/docs/REPRODUCIBLE_EXAMPLES.md` |
| Video output comparison README | `README_VIDEO_OUTPUT_COMPARISON.md` and `delivery/docs/VIDEO_OUTPUT_COMPARISON.md` |
| Sample input and result videos | `delivery/sample/` |
| Objective result tables | `delivery/results/` |
| Supporting post-processing and evaluation scripts | `tools/postprocessing/`, `tools/evaluation/`, `tools/preprocessing/`, and `video2smplx/postprocessing/` |

## 2. Pipeline Architecture

```text
Input sign-language video
  -> SMPLest-X body estimation
  -> WiLoR hand estimation
  -> EMOCA face estimation
  -> SMPL-X parameter fusion
  -> optional temporal stabilization
  -> optional global_orient + transl corrector
  -> optional hand wrist/finger corrector
  -> optional YOLO-pose 2D-guided upper-body corrector
  -> temporal smoothing
  -> smplx_params.npz export
  -> rendered SMPL-X video
  -> side-by-side input/render comparison video
```

The large base models estimate the initial SMPL-X, MANO, and face parameters.
The post-processors are lightweight learned residual correctors trained on the
SignLanguage section and used without ground truth at inference time.

## 3. Source Code Entry Points

Run the four independent programs when a reviewer needs to inspect each
estimation stage separately.

```bash
python delivery/source/body_estimation_cli.py \
  --video datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4 \
  --output runs_delivery/SignLanguage_S2 \
  --device cuda \
  --detector-stride 5 \
  --batch-size 8 \
  --inference-mode \
  --single-gpu-model

python delivery/source/hand_estimation_cli.py \
  --video datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4 \
  --output runs_delivery/SignLanguage_S2 \
  --device cuda

python delivery/source/face_estimation_cli.py \
  --video datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4 \
  --output runs_delivery/SignLanguage_S2 \
  --device cuda

python delivery/source/combine_smooth_render_cli.py \
  --body-dir runs_delivery/SignLanguage_S2/body_params \
  --hand-dir runs_delivery/SignLanguage_S2/hand_params \
  --face-dir runs_delivery/SignLanguage_S2/face_params \
  --output runs_delivery/SignLanguage_S2/final_base \
  --input-video datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4 \
  --smplx-model SMPLest-X-Inference/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz \
  --fps 30 \
  --gender neutral
```

For the final integrated system, use one command:

```bash
python delivery/source/integrated_postprocessed_pipeline_cli.py \
  --video datasets/videos/SignLanguage/SignLanguage_S2/SignLanguage_S2.mp4 \
  --output runs_delivery/SignLanguage_S2_accurate \
  --name SignLanguage_S2 \
  --sequence SignLanguage_S2 \
  --device cuda \
  --postprocess-mode accurate \
  --precomputed-yolo-keypoints runs_delivery/precomputed_yolo_pose_keypoints.json \
  --global-ckpt outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt \
  --hand-ckpt /project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt \
  --upper2d-ckpt /project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt \
  --smplestx-detector-stride 5 \
  --smplestx-batch-size 8 \
  --smplestx-inference-mode \
  --smplestx-single-gpu-model \
  --parallel-models
```

The delivery keeps only inference, evaluation, and visualization programs in
the operational source tree:

| Purpose | Files |
| --- | --- |
| Geometry evaluation | `tools/evaluation/evaluate_signlanguage_geometry.py`, `tools/evaluation/evaluate_signlanguage_ablation.py`, `tools/evaluation/evaluate_signlanguage_2d_guided_splits.py` |
| YOLO 2D keypoint extraction | `tools/preprocessing/extract_signlanguage_yolo_pose_keypoints.py` |
| Global corrector | `video2smplx/postprocessing/signlanguage.py`, `tools/postprocessing/apply_signlanguage_global_correction.py` |
| Hand corrector | `tools/postprocessing/apply_signlanguage_hand_correction.py` |
| 2D-guided upper-body corrector | `tools/postprocessing/apply_signlanguage_2d_guided_upperbody_corrector.py` |
| Shared GT-free correction runtime | `video2smplx/postprocessing/` |
| Qualitative rendering helpers | `tools/visualization/` |

Dataset builders, model trainers, GT oracle tools, training Slurm jobs, and
rejected upper-body experiments are archived under
`research/signlanguage_training/`. They are marked research-only and are not
copied into the document set or flash-drive delivery.

## 4. Configurations And Ablations

The integrated CLI exposes stabilization and post-processing explicitly, so the
same sample video can be run under controlled settings.

| Mode | Meaning | YOLO 2D keypoints |
| --- | --- | --- |
| `none` | Base fusion only | Not required |
| `global` | Base fusion plus global orientation and translation correction | Not required |
| `global_hand` | Global correction plus hand wrist/finger correction | Not required |
| `fast` | Alias for global plus hand correction | Not required |
| `accurate` | Global plus hand plus 2D-guided upper-body correction | Required |

Stabilization flags:

```text
--stabilize-global-orient
--stabilize-shape
--stabilize-lower-body
--stabilize-torso
```

Generate subjective sample outputs for each step:

```bash
SEQUENCES="SignLanguage_S2" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_ablation_samples \
YOLO_MODEL=yolov8x-pose.pt \
sbatch delivery/scripts/slurm/samples/generate_ablation_samples.sh
```

Default sample variants:

| Variant | Stabilization | Post-processing |
| --- | --- | --- |
| `base_no_stab` | Off | None |
| `base_stab` | Global orientation plus shape | None |
| `global_only` | Off | Global corrector |
| `fast_global_hand` | Off | Global plus hand correctors |
| `accurate_2d` | Off | Global plus hand plus 2D-guided upper-body correctors |
| `accurate_2d_stab` | Global orientation plus shape | Global plus hand plus 2D-guided upper-body correctors |

Each variant writes these reviewer artifacts:

```text
<RUN_ROOT>/<variant>/<sequence>/final_*/smplx_params.npz
<RUN_ROOT>/<variant>/<sequence>/final_*/rendered/smplx_render.mp4
<RUN_ROOT>/<variant>/<sequence>/final_*/side_by_side_input_render.mp4
<RUN_ROOT>/<variant>/<sequence>/runtime/integrated_postprocessed_runtime_report.json
<RUN_ROOT>/evaluation/ablation_sample_summary.md
```

## 5. Subjective Result Files

Packaged sample previews are kept in `delivery/sample/`:

| File | Purpose |
| --- | --- |
| `delivery/sample/input-sample.mp4` | Sample sign-language input clip |
| `delivery/sample/output-sample.mp4` | Rendered SMPL-X output clip |
| `delivery/sample/input-preview.gif` | Lightweight input preview |
| `delivery/sample/output-preview.gif` | Lightweight output preview |
| `delivery/sample/mesh-comparison.gif` | Mesh comparison preview |

For ablation videos generated on the server, review:

```text
/project/lt200246-mmacma/khtun/video2smplx_ablation_samples/<variant>/<sequence>/final_*/rendered/smplx_render.mp4
/project/lt200246-mmacma/khtun/video2smplx_ablation_samples/<variant>/<sequence>/final_*/side_by_side_input_render.mp4
```

The S2 ablation sample set is included under `docs/media/ablation_s2/`. It
contains all six ablation variants, with preview images, side-by-side MP4s,
rendered-only MP4s, and `smplx_params.npz` for each variant.
The compact result table is saved as
`delivery/results/signlanguage_s2_ablation_sample_summary.csv`.

For a reviewer-facing video table, open:

```text
README_VIDEO_OUTPUT_COMPARISON.md
```

That README embeds GitHub-visible preview images and links to the matching
repo-contained side-by-side MP4 files.

These qualitative outputs should be used together with the quantitative tables,
because a lower MPVPE does not always reveal temporal smoothness or visible hand
quality.

## 6. Objective Results On SignLanguage

The current held-out SignLanguage-section ablation is stored in
`delivery/results/signlanguage_heldout_ablation.csv`.

| Method | MPJPE | PA-MPJPE | MPVPE | PA-MPVPE | Visible Upper MPVPE | Hand Wrist MPVPE | Hand PA-MPVPE | Face MPVPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw SMPLest-X | 74.89 | 41.54 | 84.60 | 30.50 | 83.53 | 45.70 | 10.92 | 14.63 |
| Fusion no stabilization | 73.48 | 41.90 | 83.39 | 30.92 | 82.33 | 45.50 | 20.68 | 14.77 |
| Global corrector | 66.47 | 41.90 | 68.58 | 30.92 | 65.50 | 46.16 | 20.68 | 12.71 |
| Global plus hand corrector | 63.94 | 38.24 | 67.70 | 29.62 | 64.49 | 32.21 | 5.17 | 12.70 |
| Non-2D upper scale 0.75 |  |  | 56.52 |  | 54.42 | 30.89 |  |  |
| 2D upper scale 0.50 | 53.20 | 33.99 | 56.67 | 28.15 | 54.26 | 29.75 | 5.16 | 12.16 |
| 2D upper scale 0.75 | 51.10 | 34.47 | 54.11 | 28.55 | 51.69 | 29.52 | 5.15 | 12.63 |
| 2D upper scale 1.00 | 51.71 | 36.57 | 54.33 | 29.67 | 51.73 | 30.01 | 5.15 | 13.56 |

All values are millimeters. The best current held-out setting is
`2D upper scale 0.75`.

The included S2 sample ablation confirms the same selection for subjective
review. On S2, `accurate_2d` without legacy stabilization gives the lowest MPVPE
and visible-upper MPVPE:

| Variant | MPJPE | MPVPE | PA-MPVPE | Visible Upper MPVPE | Hand Wrist MPVPE | Hand PA-MPVPE | No-load No-render FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base no stabilization | 100.86 | 115.57 | 28.42 | 119.61 | 35.07 | 17.49 | 2.30 |
| Base stabilization | 144.19 | 168.89 | 28.40 | 171.25 | 38.04 | 17.49 | 3.39 |
| Global only | 84.48 | 83.20 | 28.42 | 87.26 | 32.42 | 17.49 | 3.13 |
| Fast global plus hand | 87.69 | 84.23 | 28.92 | 88.45 | 19.49 | 2.02 | 3.12 |
| Accurate 2D | 40.56 | 33.84 | 21.00 | 28.85 | 12.46 | 2.00 | 3.12 |
| Accurate 2D plus stabilization | 55.33 | 38.92 | 26.29 | 36.28 | 23.46 | 2.27 | 3.11 |

The earlier full usable SignLanguage-set summary is stored in
`delivery/results/signlanguage_full_dataset_summary.csv`.

| Method | Frames | MPJPE | PA-MPJPE | MPVPE | PA-MPVPE | Body MPJPE | Body PA-MPJPE | Hand MPJPE | Hand PA-MPJPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw SMPLest-X only | 48,561 | 105.09 | 48.79 | 112.51 | 31.23 | 95.64 | 26.49 | 109.73 | 26.96 |
| Fusion no stabilization | 47,470 | 103.13 | 47.82 | 111.09 | 30.40 | 94.93 | 26.51 | 106.75 | 29.79 |
| Final corrected, global plus hand | 48,443 | 77.02 | 40.00 | 78.49 | 28.26 | 70.22 | 26.51 | 81.23 | 14.62 |

## 7. UBody Literature Comparison

Published UBody numbers are whole-UBody benchmark results. Our table row is
measured only on the SignLanguage section, so it should be reported as an
in-domain comparison, not as an official UBody leaderboard entry.

| Method | Evaluation set | All MPVPE/PVE | All PA-MPVPE/PA-PVE | Hand MPVPE/PVE | Hand PA-MPVPE/PA-PVE | Face MPVPE/PVE | Face PA-MPVPE/PA-PVE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hand4Whole | UBody | 104.1 | 44.8 | 45.7 | 8.9 | 27.0 | 2.8 |
| OSX finetuned | UBody | 81.9 | 42.2 | 41.5 | 8.6 | 21.2 | 2.0 |
| AiOS | UBody | 58.6 | 32.5 | 39.0 | 7.3 | 19.6 | 2.8 |
| SMPLer-X-L20 finetuned | UBody | 57.4 | 31.9 | 40.2 | 10.3 | 21.6 | 2.8 |
| Multi-HMR ViT-L/14 | UBody | 51.2 | 21.0 | 25.0 | 7.2 | 16.2 | 1.8 |
| SMPLest-X-H40 | UBody | 51.1 | 27.8 | 32.9 | 7.9 | 21.4 | 2.5 |
| SMPLest-X-H40 finetuned | UBody | 50.2 | 26.9 | 31.8 | 8.4 | 18.9 | 2.4 |
| CoEvoer | UBody | 51.6 | 26.5 | 27.8 | 7.1 | 16.2 | 1.8 |
| Ours, 2D upper scale 0.75 | UBody SignLanguage section | 54.11 | 28.55 | 29.52 | 5.15 | 12.63 | not reported in this run |

Sources:

- SMPLer-X and earlier UBody baselines: https://ar5iv.labs.arxiv.org/html/2309.17448
- SMPLest-X UBody table: https://arxiv.org/html/2501.09782v1
- Multi-HMR UBody table: https://ar5iv.labs.arxiv.org/html/2402.14654
- CoEvoer UBody table: https://arxiv.org/html/2604.17959v1

## 8. Runtime And FPS

Runtime results with precomputed YOLO keypoints are stored in
`delivery/results/runtime_precomputed_yolo_summary.csv`.

The server A/B test for the optimized in-memory execution path is documented in
`delivery/docs/TIME_OPTIMIZATION_SERVER_TEST.md`. It compares the legacy
per-frame PKL path, in-memory parallel execution, and in-memory batched
SMPLest-X execution using identical checkpoints and final NPZ export.

Measured post-processing overhead on S2/S3/S4:

| Stage | Total seconds | ms/frame |
| --- | ---: | ---: |
| Global corrector | 1.41 | 3.13 |
| Hand corrector | 2.25 | 5.00 |
| Load precomputed YOLO keypoints | 0.045 | 0.10 |
| 2D upper-body corrector | 2.19 | 4.86 |
| Total post-processing | 5.90 | 13.10 |

One-process integrated benchmark with precomputed YOLO keypoints:

| Variant | Total FPS | No-render FPS | No-load plus no-render FPS | Steady model FPS |
| --- | ---: | ---: | ---: | ---: |
| Base pipeline | 1.12 | 1.20 | 2.68 | 6.70 |
| Accurate postprocessed pipeline | 1.98 | 2.22 | 3.57 | 6.76 |

The steady-state model FPS excludes the first warmup frames. The measured
post-processing overhead is small compared with base SMPLest-X, WiLoR, EMOCA,
and rendering time.

## 9. Server Setup

Server project root:

```bash
cd /home/khtun/video2simplx/Video2SmplxPy10/Video2Smplx
module load Mamba/23.11.0-0
conda activate video2smplx_shared310
export PYOPENGL_PLATFORM=egl
```

Expected model directories:

```text
SMPLest-X-Inference/
WiLoR-Inference/
EMOCA-Inference/
signlanguage_global_correction_server/models/SMPLX_FEMALE.npz
```

Post-processor checkpoints used by the current final pipeline:

```text
outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt
/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt
/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt
```

Precompute YOLO keypoints once for repeatable accurate-mode timing:

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_keypoints \
YOLO_MODEL=yolov8x-pose.pt \
sbatch delivery/scripts/slurm/preprocessing/precompute_signlanguage_yolo_keypoints.sh
```

Run the final benchmark:

```bash
SEQUENCES="SignLanguage_S2 SignLanguage_S3 SignLanguage_S4" \
RUN_ROOT=/project/lt200246-mmacma/khtun/video2smplx_integrated_oneprocess_benchmark \
YOLO_KEYPOINTS=/project/lt200246-mmacma/khtun/video2smplx_keypoints/evaluation/precomputed_yolo_pose_keypoints.json \
sbatch delivery/scripts/slurm/benchmarks/benchmark_integrated_oneprocess.sh
```

## 10. Clean Delivery Set

For the final handover, include:

```text
README_DELIVERY.md
README_TOR.md
delivery/source/
delivery/scripts/
delivery/docs/
delivery/results/
delivery/sample/
tools/postprocessing/
tools/evaluation/
tools/preprocessing/
tools/visualization/
video2smplx/postprocessing/
```

Do not include `research/signlanguage_training/`, temporary server logs,
temporary archives, partial experiment folders, `.DS_Store`, local caches, or
large model checkpoints unless the contract explicitly requires physical media
to include model weights.
