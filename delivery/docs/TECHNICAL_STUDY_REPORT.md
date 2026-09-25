# Literature Review And Technical Study Report

## Objective

The project estimates expressive SMPL-X parameters from sign-language videos:
body pose and shape, both hands, face expression, temporal smoothing, rendering,
and benchmark evaluation.

## Base System

The delivered base system is a modular fusion pipeline:

| Module | Model | Output |
| --- | --- | --- |
| Body | SMPLest-X | SMPL-X root pose, body pose, shape, camera translation |
| Hands | WiLoR | MANO/SMPL-X compatible left and right hand pose |
| Face | EMOCA | Expression coefficients and jaw pose |
| Combination/render | Fusion + smoothing + SMPL-X renderer | `smplx_params.npz`, rendered MP4, side-by-side MP4 |

The four delivered CLI programs are independent entrypoints, but they share the
same parameter contract so their outputs can be combined deterministically.

## Literature Context

| Method | Relevance | Notes |
| --- | --- | --- |
| SMPLest-X | Whole-body expressive SMPL-X estimation | Strong current baseline and the body model used in this pipeline. Official repository: https://github.com/SMPLCap/SMPLest-X |
| SMPLer-X | Generalist expressive human pose/shape model | Predecessor family to SMPLest-X, with UBody benchmark results. Official repository: https://github.com/MotrixLab/SMPLer-X |
| OSX | One-stage whole-body mesh recovery | Introduced UBody and component-aware whole-body recovery. Project page: https://osx-ubody.github.io/ |
| WiLoR | In-the-wild hand localization/reconstruction | Used here as the specialist hand estimator. Official repository: https://github.com/rolpotamias/WiLoR |
| EMOCA | Monocular face capture and animation | Used here for expression and jaw replacement. Project page: https://emoca.is.tue.mpg.de/ |

## Benchmark Position

On the SignLanguage section of UBody, the base and corrected systems were
evaluated using the same local SMPL-X geometry evaluator. The current best held-
out result is:

| Method | MPVPE | PA-MPVPE | Visible Upper MPVPE | Hand Wrist MPVPE | Hand PA-MPVPE | Face MPVPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw SMPLest-X | 84.60 | 30.50 | 83.53 | 45.70 | 10.92 | 14.63 |
| Fusion no stabilization | 83.39 | 30.92 | 82.33 | 45.50 | 20.68 | 14.77 |
| Global + hand corrector | 67.70 | 29.62 | 64.49 | 32.21 | 5.17 | 12.70 |
| 2D-guided upper scale 0.75 | 54.11 | 28.55 | 51.69 | 29.52 | 5.15 | 12.63 |

The gap between MPVPE and PA-MPVPE shows that a large part of the remaining
whole-body error is absolute alignment and upper-body articulation, not local
hand/face shape alone.

### UBody Literature Comparison

Published UBody tables usually report PVE/MPVPE and PA-PVE/PA-MPVPE for all
vertices, hands, and face. The values below are whole-UBody benchmark values
from the cited papers. Our row is measured only on the UBody SignLanguage
section, so it should be described as an in-domain comparison instead of an
official whole-UBody leaderboard result.

| Method | Evaluation set | All MPVPE/PVE | All PA | Hand MPVPE/PVE | Hand PA | Face MPVPE/PVE | Face PA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hand4Whole | UBody | 104.1 | 44.8 | 45.7 | 8.9 | 27.0 | 2.8 |
| OSX finetuned | UBody | 81.9 | 42.2 | 41.5 | 8.6 | 21.2 | 2.0 |
| AiOS | UBody | 58.6 | 32.5 | 39.0 | 7.3 | 19.6 | 2.8 |
| SMPLer-X-L20 finetuned | UBody | 57.4 | 31.9 | 40.2 | 10.3 | 21.6 | 2.8 |
| Multi-HMR ViT-L/14 | UBody | 51.2 | 21.0 | 25.0 | 7.2 | 16.2 | 1.8 |
| SMPLest-X-H40 | UBody | 51.1 | 27.8 | 32.9 | 7.9 | 21.4 | 2.5 |
| SMPLest-X-H40 finetuned | UBody | 50.2 | 26.9 | 31.8 | 8.4 | 18.9 | 2.4 |
| CoEvoer | UBody | 51.6 | 26.5 | 27.8 | 7.1 | 16.2 | 1.8 |
| Ours, 2D upper scale 0.75 | UBody SignLanguage section | 54.11 | 28.55 | 29.52 | 5.15 | 12.63 | not reported |

Sources:

- SMPLer-X UBody Table 6: https://ar5iv.labs.arxiv.org/html/2309.17448
- SMPLest-X UBody Table IX: https://arxiv.org/html/2501.09782v1
- Multi-HMR Table 5: https://ar5iv.labs.arxiv.org/html/2402.14654
- CoEvoer UBody Table 1: https://arxiv.org/html/2604.17959v1

## Error Analysis Summary

The early high MPJPE/MPVPE came from:

- root/global orientation and translation drift;
- upper-body chain errors around torso, shoulders, elbows, and wrists;
- hand global placement being affected by arm/wrist alignment even when local
  finger PA metrics are low;
- lower-body vertices contributing to whole-body MPVPE even though sign-language
  videos mainly show upper body.

The strongest tested improvements came from global correction and upper-body
residual correction. Hand correction strongly improved hand-local metrics, but
could not fully fix arm placement because wrist location depends on upstream arm
and torso pose.

## Corrector Study

| Experiment | Input | Predicted correction | Adopted | Result |
| --- | --- | --- | --- | --- |
| Global corrector | fused params | `global_orient` + `transl` | yes | MPVPE improved from 83.39 to 68.58 |
| Hand corrector | global-corrected params | wrist/finger residuals | yes | Hand wrist MPVPE improved to 32.21 and hand PA-MPVPE to 5.17 |
| Upper-body rotation corrector | global-corrected params | upper-body body-pose deltas | no | Improved some upper-body metrics but was weaker than the final 2D-guided model |
| Upper-body + translation corrector | global-corrected params | torso/arms + `transl` | no | Tested to handle absolute MPVPE; did not beat the final 2D-guided model |
| Non-2D upper corrector | SMPL-X parameter features | upper-body residuals | partial | Best MPVPE around 56.52 |
| 2D-guided upper corrector | SMPL-X + detected 2D keypoint residuals | upper-body residuals | yes | Best MPVPE 54.11 at scale 0.75 |

## Runtime Complexity Measurement

Runtime is measured empirically on the allocated server GPU because the final
pipeline combines several external networks, Python preprocessing, file I/O,
SMPL-X forward passes, and offscreen rendering.

Use:

```bash
sbatch delivery/scripts/slurm/benchmarks/benchmark_final_postprocessed_pipeline.sh
```

The benchmark reports:

- wall-clock seconds per stage;
- seconds per video;
- seconds per frame;
- effective FPS per stage and for the full pipeline;
- percent runtime share per stage;
- post-processor checkpoint size and parameter count;
- final output file sizes.

The expected major runtime cost is still base inference, especially SMPLest-X,
WiLoR, EMOCA, and rendering. The learned post-processors are small MLPs, so their
runtime should be much lower than the foundation estimators. YOLO-pose extraction
adds extra cost for the 2D-guided upper-body corrector.

For a fair FPS comparison, the base render must not be counted twice. Use:

```bash
sbatch delivery/scripts/slurm/benchmarks/benchmark_fair_fps.sh
```

This produces separate rows for:

- without post-processing, no render;
- without post-processing, with render;
- post-processing overhead only;
- with post-processing, no render;
- with post-processing, with final render.

For deployment-style throughput, use the one-process benchmark:

```bash
sbatch delivery/scripts/slurm/benchmarks/benchmark_integrated_oneprocess.sh
```

This is the correct comparison to the earlier optimized `script_fusion.sh`
because SMPLest-X, WiLoR, and EMOCA are loaded by one integrated pipeline run and
the correctors are applied without launching separate Python jobs per stage.

## Contribution

The contribution is not only combining three existing models. The research
contribution is a sign-language-specific SMPL-X post-refinement framework:

- separates body, hand, and face estimation into inspectable modules;
- diagnoses which SMPL-X parameter groups dominate SignLanguage error;
- learns lightweight GT-free-at-inference residual correctors from SMPL-X
  parameters and optional 2D detector evidence;
- improves upper-body and hand geometry without retraining the large base models;
- provides reproducible ablations for raw body, fusion, global correction, hand
  correction, non-2D upper correction, and 2D-guided upper correction.

## Limitations

- The corrected model is trained on SignLanguage-style data and should be tested
  before claiming generalization to other signing datasets.
- The 2D-guided corrector depends on detector quality and camera framing.
- Whole-body MPVPE still includes lower-body vertices that may be invisible or
  weakly constrained in sign-language videos.
- Direct comparison to published UBody leaderboard numbers is only fair when the
  same split, gender model, region definitions, and alignment protocol are used.
