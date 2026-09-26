# SignLanguage Training Archive

This directory contains research and retraining tools. It is intentionally
excluded from the customer delivery document set and flash-drive set.

These programs may read SignLanguage ground truth, generate oracle labels,
build `.npz` datasets, train correction checkpoints, or reproduce rejected
upper-body ablations. They are not required to run the delivered inference
pipeline with existing checkpoints.

## Contents

- `build_signlanguage_*.py`: create supervised training datasets.
- `train_signlanguage_*.py`: train the correction MLP checkpoints.
- `oracle_optimize_signlanguage_*.py`: GT-dependent upper-bound analysis and
  label generation.
- `transform_signlanguage_feature_ablation_dataset.py`: create controlled
  feature-ablation datasets.
- `apply_signlanguage_upperbody*.py`: older non-adopted upper-body experiments.
- `slurm/`: server jobs for retraining and training ablations.
- `docs/`: historical experiment instructions.

The maintained deployment correctors remain at the repository root:

```text
tools/postprocessing/apply_signlanguage_global_correction.py
tools/postprocessing/apply_signlanguage_hand_correction.py
tools/postprocessing/apply_signlanguage_2d_guided_upperbody_corrector.py
```

Their GT-free shared implementation is in `video2smplx/postprocessing/`.
