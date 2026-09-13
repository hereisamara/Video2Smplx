# Video2Smplx Delivery Manifest

This delivery contains the source code, documentation, model setup instructions,
and sample results needed to run and verify the sign-language SMPL-X pipeline.

## Required Deliverables

| Item | Location |
| --- | --- |
| Body estimation CLI | `delivery/source/body_estimation_cli.py` |
| Hand estimation CLI | `delivery/source/hand_estimation_cli.py` |
| Face estimation CLI | `delivery/source/face_estimation_cli.py` |
| Parameter combination, temporal smoothing, and rendering CLI | `delivery/source/combine_smooth_render_cli.py` |
| Combined SMPL-X parameter output | generated as `<run_output>/smplx_params.npz` |
| Rendered SMPL-X video | generated as `<run_output>/rendered/smplx_render.mp4` |
| Side-by-side comparison video | generated as `<run_output>/side_by_side_input_render.mp4` |
| Literature review and technical study report | `delivery/docs/TECHNICAL_STUDY_REPORT.md` |
| Reviewer delivery README | `README_DELIVERY.md` and `delivery/docs/README_DELIVERY.md` |
| TOR delivery README/checklist | `README_TOR.md` and `delivery/docs/README_TOR.md` |
| Installation manual | `delivery/docs/INSTALLATION.md` |
| Complete user documentation | `delivery/docs/USER_DOCUMENTATION.md` |
| Model setup/download instructions | `delivery/docs/MODEL_SETUP.md` |
| Reproducible examples | `delivery/docs/REPRODUCIBLE_EXAMPLES.md` |
| Video output comparison README | `README_VIDEO_OUTPUT_COMPARISON.md` and `delivery/docs/VIDEO_OUTPUT_COMPARISON.md` |
| Sample sign-language video and results | `delivery/sample/` |
| Objective result tables | `delivery/results/` |
| Document set build script | `delivery/scripts/build_document_set.sh` |
| Flash-drive copy script | `delivery/scripts/prepare_flash_drive_set.sh` |
| Server delivery overlay installer | `delivery/scripts/install_delivery_overlay.sh` |
| Base integrated server test | `delivery/scripts/slurm_test_delivery_integrated_pipeline.sh` |
| Final postprocessed server test | `delivery/scripts/slurm_test_delivery_final_postprocessed_pipeline.sh` |
| Final postprocessed runtime benchmark | `delivery/scripts/slurm_benchmark_delivery_final_postprocessed_pipeline.sh` |
| Fair FPS with/without post-processing benchmark | `delivery/scripts/slurm_benchmark_delivery_fair_fps.sh` |
| One-process integrated postprocessed benchmark | `delivery/scripts/slurm_benchmark_integrated_postprocessed_oneprocess.sh` |
| Ablation sample output generator | `delivery/scripts/slurm_generate_ablation_samples.sh` |
| Supporting SignLanguage post-processing scripts | root-level `apply_signlanguage_*.py`, `build_signlanguage_*.py`, `train_signlanguage_*.py` |
| Supporting SignLanguage evaluation scripts | root-level `evaluate_signlanguage_*.py` and `extract_signlanguage_yolo_pose_keypoints.py` |

## Final Output Folder Contract

A completed processing run should contain:

```text
<run_output>/
  body_params/
  hand_params/
  face_params/
  fused_params/
  rendered/
    params/
    smplx_render.mp4
  smplx_params.npz
  side_by_side_input_render.mp4
  body_report.json
  hand_report.json
  face_report.json
  combine_render_report.json
```

The `smplx_params.npz` archive contains stacked per-frame arrays:

```text
frame_id
valid
global_orient
body_pose
left_hand_pose
right_hand_pose
jaw_pose
betas
expression
transl
smplx_param_vector
```

## Current Best SignLanguage Held-Out Test Result

The best held-out result from the current experiments is the 2D-guided upper-body
corrected model at correction scale `0.75`.

| Method | MPJPE | PA-MPJPE | MPVPE | PA-MPVPE | Visible Upper MPVPE | Hand Wrist MPVPE | Hand PA-MPVPE | Face MPVPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw SMPLest-X | 74.89 | 41.54 | 84.60 | 30.50 | 83.53 | 45.70 | 10.92 | 14.63 |
| Fusion no stabilization | 73.48 | 41.90 | 83.39 | 30.92 | 82.33 | 45.50 | 20.68 | 14.77 |
| Global corrector | 66.47 | 41.90 | 68.58 | 30.92 | 65.50 | 46.16 | 20.68 | 12.71 |
| Global + hand corrector | 63.94 | 38.24 | 67.70 | 29.62 | 64.49 | 32.21 | 5.17 | 12.70 |
| Non-2D upper scale 0.75 | - | - | 56.52 | - | 54.42 | 30.89 | - | - |
| 2D upper scale 0.75 | 51.10 | 34.47 | 54.11 | 28.55 | 51.69 | 29.52 | 5.15 | 12.63 |

All values are millimeters. Report these numbers as SignLanguage-section results,
not as a general UBody-test claim.
