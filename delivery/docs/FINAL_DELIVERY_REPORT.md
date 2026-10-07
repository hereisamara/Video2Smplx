# Video2Smplx Final Delivery Report

## 1. Project Outcome

The project delivers a monocular-video pipeline that estimates expressive
SMPL-X body, hand, and face parameters for sign-language video. The operational
pipeline combines SMPLest-X, WiLoR/MANO, and EMOCA/FLAME, then optionally
applies learned global, hand, and 2D-guided upper-body corrections without
using ground truth during inference.

## 2. Delivered Items

- Independent body, hand, face, and combine/smooth/render Python CLIs.
- One-process integrated inference CLI with realtime, streaming, and batch
  execution modes.
- Final `smplx_params.npz`, rendered mesh video, and input/render comparison
  video outputs.
- Global, hand, and 2D-guided upper-body correction checkpoints.
- Separate verified runtime-model archive, approximately 14 GB.
- Installation, model import, inference, configuration, evaluation, and user
  documentation.
- SignLanguage sample media, ablation outputs, objective result tables, and
  evaluation utilities.

The single project and operational guide is the repository-root `README.md`.

## 3. Final Accuracy Result

The selected held-out SignLanguage configuration is global correction, hand
correction, and 2D-guided upper-body correction with scale `0.75`.

| Metric | Result (mm) |
| --- | ---: |
| MPJPE | 51.10 |
| PA-MPJPE | 34.47 |
| MPVPE | 54.11 |
| PA-MPVPE | 28.55 |
| Visible upper-body MPVPE | 51.69 |
| Hand wrist MPVPE | 29.52 |
| Hand PA-MPVPE | 5.15 |
| Face MPVPE | 12.63 |

Compared with fusion without stabilization, held-out MPVPE decreased from
`83.39 mm` to `54.11 mm`, a reduction of `29.28 mm` or approximately `35.1%`.
Visible upper-body MPVPE decreased from `82.33 mm` to `51.69 mm`.

These are in-domain SignLanguage-section results. Published whole-UBody values
use a different evaluation population and must not be presented as a direct
leaderboard comparison.

## 4. Runtime Result

Runtime values have different scopes and are reported separately.

| Runtime scope | Result |
| --- | ---: |
| SMPLest-X only, batched, warm-up excluded | 19.59 FPS |
| Full foundation models, warm steady state | 10.76 FPS |
| Accurate final pipeline, no initial load or rendering, precomputed 2D keypoints | 3.57 FPS |
| Accurate final pipeline with rendering, precomputed 2D keypoints | 1.98 FPS |
| Latest clean-package foundation-model run | 7.68 FPS |
| Latest clean-package warm full-pipeline estimate with online YOLO and rendering | 4.00 FPS |

The `19.59 FPS` result is SMPLest-X alone in offline batch mode. It is not the
throughput of the complete body, hand, face, correction, export, and rendering
pipeline.

## 5. Verification Status

- Runtime-model packaging completed and all required model assets passed the
  model verifier.
- The model archive was created at approximately 14 GB with checksums and
  preserved repository-relative paths.
- Delivery CLIs imported successfully from an extracted document package.
- Full accurate inference completed on the bundled sample for 321 frames.
- The clean-package run generated final SMPL-X parameters, mesh video,
  side-by-side video, and runtime reports.
- Shell syntax and delivery-document references passed local checks.

The clean-package wrapper used obsolete root-level output assertions after the
successful inference run. Those assertions were corrected to verify files
under `final_postprocessed/`, and the test launcher was changed to use batch
execution with in-memory fast I/O.

## 6. Remaining Acceptance Actions

1. Rerun the batch-mode clean-package test and retain its final `[done]` log
   and runtime report.
2. Build the source/document archive, licensed runtime-model archive, and final
   sample-output archive on the server.
3. Upload the verified archives and checksums to the delivery Drive folder.
4. Add the final Drive links to the package table in `README.md`.
5. Assemble the flash-drive set and verify checksums on the destination media.

## 7. Limitations

- Learned correctors are trained for the SignLanguage domain; generalization to
  unrelated datasets has not been established.
- Complete inference requires NVIDIA CUDA and the supplied upstream estimator
  source trees.
- Model files remain governed by their original SMPL-X, MANO, FLAME, EMOCA,
  SMPLest-X, WiLoR, and Ultralytics licenses.
- Online YOLO pose and synchronous rendering materially reduce end-to-end FPS.
  Precomputed keypoints and asynchronous rendering are recommended when the
  deployment workflow permits them.

## 8. Conclusion

The required body, hand, face, fusion, correction, SMPL-X export, rendering,
documentation, sample, and evaluation capabilities have been implemented and
demonstrated on the server. The selected post-processing configuration provides
a substantial held-out reduction in whole-mesh and visible upper-body error,
with particularly low aligned hand and face errors.

The technical deliverables are complete. Final acceptance is pending the clean
package rerun, server package upload, Drive-link update, and final media
checksum verification. Performance claims must retain their stated scopes:
`19.59 FPS` is the body-only batched rate, while complete-pipeline throughput
depends on enabled detectors, correction, export, and rendering.
