# Delivery Result Tables

This folder stores compact result tables used by `README_DELIVERY.md` and the
technical report.

| File | Contents |
| --- | --- |
| `signlanguage_heldout_ablation.csv` | Held-out SignLanguage ablation with MPJPE, MPVPE, regional hand, visible-upper, and face metrics |
| `signlanguage_s2_ablation_sample_summary.csv` | Downloaded S2 subjective-sample ablation run with matching runtime and geometry metrics |
| `signlanguage_full_dataset_summary.csv` | Earlier full usable SignLanguage-set summary for raw SMPLest-X, no-stabilization fusion, and global-plus-hand corrected pipeline |
| `ubody_sota_comparison.csv` | Published UBody benchmark values and our SignLanguage-section row |
| `runtime_precomputed_yolo_summary.csv` | Runtime phases for SMPLest-X, standalone YOLO extraction, learned post-processing, and the integrated pipeline |

All geometry errors are in millimeters.

Runtime rows have different scopes. In particular, `19.59 FPS` is the
warm-up-excluded SMPLest-X-only working rate, while `6.76 FPS` is the full
foundation-model schedule and `3.57 FPS` includes the final learned correction
stack and NPZ export but excludes initial model loading and rendering.
