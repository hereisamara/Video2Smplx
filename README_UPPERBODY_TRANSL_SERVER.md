# SignLanguage Upper-Body + Translation Correction Server Patch

This patch tests the next bottleneck after global and hand correction: upper-body pose changes need a matching translation update, otherwise PA metrics improve while MPVPE can get worse.

Pipeline tested by this patch:

```text
fused_params_model_global_corrected
  -> upper-body rotation + transl correction
  -> evaluate
```

The model predicts:

```text
selected body_pose rotation deltas + transl delta
```

Default target mode:

```text
torso_arms
```

This is safer than full `upper_body` because the previous full upper-body rotation-only model improved PA metrics but hurt absolute MPVPE.

## Install On Server

```bash
cd ~/video2simplx/Video2SmplxPy10/Video2Smplx
tar -xzf signlanguage_upperbody_transl_correction_server_patch.tar.gz
```

Files extract into:

```text
signlanguage_global_correction_server/
```

## Required Input

Global-corrected predictions must exist:

```text
outputs_fusion_signlanguage_no_stablized/SignLanguage_S*/fused_params_model_global_corrected
```

## Run Default

```bash
sbatch signlanguage_global_correction_server/slurm_signlanguage_upperbody_transl_correction_gpu.sh
```

This trains `torso_arms + transl`, then evaluates rotation scales:

```text
0.25, 0.50, 0.75, 1.00
```

with translation scale fixed at:

```text
1.0
```

Outputs go to:

```text
/project/lt200246-mmacma/khtun/signlanguage_upperbody_transl_correction_outputs
```

## Run Other Ablations

Arms only:

```bash
sbatch --export=ALL,TARGET_MODE=arms signlanguage_global_correction_server/slurm_signlanguage_upperbody_transl_correction_gpu.sh
```

Full upper body:

```bash
sbatch --export=ALL,TARGET_MODE=upper_body signlanguage_global_correction_server/slurm_signlanguage_upperbody_transl_correction_gpu.sh
```

Use fewer scale points:

```bash
sbatch --export=ALL,TARGET_MODE=torso_arms,ROT_SCALES="0.25 0.50 0.75" signlanguage_global_correction_server/slurm_signlanguage_upperbody_transl_correction_gpu.sh
```

Increase/decrease translation emphasis during training:

```bash
sbatch --export=ALL,TRANSL_WEIGHT=2.0 signlanguage_global_correction_server/slurm_signlanguage_upperbody_transl_correction_gpu.sh
```

## What To Compare

The Slurm output prints compact `[all-row]` lines with:

```text
mpjpe
pa_mpjpe
mpvpe
pa_mpvpe
body_mpjpe
body_pa_mpjpe
hand_mpjpe
hand_pa_mpjpe
visible_upper_mpvpe
hands_wrist_mpvpe
hands_pa_mpvpe
```

Compare against the previous best:

```text
global + hand:
MPJPE 77.02
PA-MPJPE 40.00
MPVPE 78.49
PA-MPVPE 28.26
Hand MPJPE 81.23
Hand PA-MPJPE 14.62
```

Good result target:

```text
MPVPE <= 78.49 while PA-MPVPE improves below 28.26
```

If one setting is clearly best, train the hand model again on that corrected output. Do not reuse the old hand checkpoint as final evidence.
