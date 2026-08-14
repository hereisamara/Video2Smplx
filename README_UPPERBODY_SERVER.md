# SignLanguage Upper-Body Correction Server Patch

This patch adds a third post-processor after global correction and before hand correction.

Pipeline:

```text
fused_params
  -> fused_params_model_global_corrected
  -> fused_params_model_global_upper_corrected
  -> fused_params_model_global_upper_hand_corrected
```

The upper-body model predicts SMPL-X `body_pose` rotation deltas for:

```text
spine/chest/neck/head anchor/collars/shoulders/elbows/wrists
```

It uses visibility-weighted training, with shoulders, elbows, and wrists weighted more heavily than torso/head rotations.

## Install On Server

From the server repo root:

```bash
cd ~/video2simplx/Video2SmplxPy10/Video2Smplx
tar -xzf signlanguage_upperbody_correction_server_patch.tar.gz
```

The files extract into:

```text
signlanguage_global_correction_server/
```

## Required Existing Inputs

Global-corrected predictions must exist:

```text
outputs_fusion_signlanguage_no_stablized/SignLanguage_S*/fused_params_model_global_corrected
```

The hand model checkpoint should exist:

```text
/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt
```

## Run

```bash
sbatch signlanguage_global_correction_server/slurm_signlanguage_upperbody_correction_gpu.sh
```

Outputs are written to project storage by default:

```text
/project/lt200246-mmacma/khtun/signlanguage_upperbody_correction_outputs
```

## Results To Check

After the job finishes:

```bash
tail -n 40 sl_upper_corr.<JOBID>.out
```

Upper-body-only evaluation:

```text
/project/lt200246-mmacma/khtun/signlanguage_upperbody_correction_outputs/evaluation/smplx_female_model_global_upper_corrected
```

Final global+upper+hand evaluation:

```text
/project/lt200246-mmacma/khtun/signlanguage_upperbody_correction_outputs/evaluation/smplx_female_model_global_upper_hand_corrected
```

Use these columns from `geometry_summary.csv`:

```text
mpvpe_mm_mean
pa_mpvpe_mm_mean
visible_upper_mpvpe_mm_mean
visible_upper_pa_mpvpe_mm_mean
hand_mpjpe_mm_mean
hands_wrist_mpvpe_mm_mean
hands_pa_mpvpe_mm_mean
face_head_mpvpe_mm_mean
face_pa_mpvpe_mm_mean
```
