# Raw SMPLest-X SignLanguage Evaluation

This patch adds a `--save_raw_smplestx` flag to `video2smplx.integrated_pipeline`.

When enabled, each run writes:

```text
<output>/smplestx_params/*_params.pkl
```

These are the SMPLest-X-only predictions before WiLoR/EMOCA fusion.

## Important

Existing fused outputs usually do not contain raw SMPLest-X params. Check first:

```bash
find outputs_fusion_signlanguage_no_stablized -path '*/smplestx_params/*_params.pkl' | head
```

If nothing prints, rerun SignLanguage inference with:

```bash
--save_raw_smplestx
```

For a pure raw SMPLest-X baseline, do not use stabilization flags.

For an apples-to-apples fusion-ablation baseline, keep the same SMPLest-X settings/stabilization flags as your fusion run and add only:

```bash
--save_raw_smplestx
```

## Evaluate

After raw params exist:

```bash
sbatch slurm_evaluate_raw_smplestx_signlanguage.sh
```

Default paths:

```text
PRED_ROOT=outputs_fusion_signlanguage_no_stablized
PARAMS_SUBDIR=smplestx_params
OUTPUT_DIR=outputs_fusion_signlanguage_no_stablized/evaluation/smplx_female_raw_smplestx
```

Override example:

```bash
sbatch --export=ALL,PRED_ROOT=outputs_fusion_signlanguage_raw_smplestx slurm_evaluate_raw_smplestx_signlanguage.sh
```

The Slurm log prints a compact `[all-row]` with:

```text
MPJPE, PA-MPJPE, MPVPE, PA-MPVPE,
Body MPJPE, Body PA-MPJPE,
Hand MPJPE, Hand PA-MPJPE,
Visible upper MPVPE,
hands wrist MPVPE, hands PA-MPVPE,
face head MPVPE, face PA-MPVPE
```
