#!/usr/bin/env python3
"""Apply a trained SignLanguage global-correction model to prediction PKLs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from evaluate_signlanguage_geometry import sequence_sort_key
from signlanguage_global_correction import (
    apply_predicted_correction,
    base_feature,
    copy_sidecar_files,
    discover_sequences,
    frame_number,
    list_prediction_files,
    load_one_person_or_none,
    normalize_sequence_name,
    write_person,
)
from train_signlanguage_global_correction import build_model, import_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred-root", type=Path, default=Path("outputs_fusion_signlanguage_no_stablized"))
    parser.add_argument("--params-subdir", default="fused_params")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: write corrected PKLs under --pred-root. Use a scratch/project path if home quota is tight.",
    )
    parser.add_argument("--output-subdir", default="fused_params_model_global_corrected")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--feature-preset", choices=("auto", "minimal", "upper_global", "full"), default="auto")
    parser.add_argument("--temporal-radius", type=int, default=-1, help="-1 means use checkpoint metadata.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--orient-scale", type=float, default=1.0)
    parser.add_argument("--transl-scale", type=float, default=1.0)
    parser.add_argument("--max-delta-degrees", type=float, default=30.0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def temporal_stack(sequence_rows: list[dict], radius: int) -> list[np.ndarray]:
    if radius <= 0:
        return [row["base_feature"] for row in sequence_rows]
    features = [row["base_feature"] for row in sequence_rows]
    output = []
    last_index = len(features) - 1
    for index in range(len(features)):
        stacked = []
        for offset in range(-radius, radius + 1):
            stacked.append(features[min(max(index + offset, 0), last_index)])
        output.append(np.concatenate(stacked, axis=0))
    return output


def main() -> None:
    args = parse_args()
    torch, nn, _, _ = import_torch()
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    feature_preset = checkpoint.get("feature_preset", "upper_global") if args.feature_preset == "auto" else args.feature_preset
    temporal_radius = int(checkpoint.get("temporal_radius", 1)) if args.temporal_radius < 0 else args.temporal_radius
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    feature_std[feature_std < 1e-6] = 1.0

    model = build_model(
        nn,
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["layers"]),
        float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pred_root = args.pred_root.resolve()
    output_root = (args.output_root or args.pred_root).resolve()
    if args.sequences == ["all"]:
        sequences = discover_sequences(pred_root, args.params_subdir)
    else:
        sequences = sorted([normalize_sequence_name(x) for x in args.sequences], key=sequence_sort_key)
    if not sequences:
        raise ValueError(f"No sequences found under {pred_root}/{args.params_subdir}")

    rows_written = []
    skipped = []
    for sequence in sequences:
        source_dir = pred_root / sequence / args.params_subdir
        target_dir = output_root / sequence / args.output_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        copy_sidecar_files(source_dir, target_dir)

        valid_rows = []
        for source_path in list_prediction_files(pred_root, sequence, args.params_subdir):
            person = load_one_person_or_none(source_path)
            if person is None:
                shutil.copy2(source_path, target_dir / source_path.name)
                skipped.append({"sequence": sequence, "file": str(source_path), "reason": "bad_or_empty_prediction_copied"})
                continue
            try:
                feature = base_feature(person, feature_preset)
            except Exception as exc:
                shutil.copy2(source_path, target_dir / source_path.name)
                skipped.append({"sequence": sequence, "file": str(source_path), "reason": f"feature_error: {exc}"})
                continue
            valid_rows.append(
                {
                    "sequence": sequence,
                    "frame": frame_number(source_path),
                    "source_path": source_path,
                    "target_path": target_dir / source_path.name,
                    "person": person,
                    "base_feature": feature,
                }
            )
        valid_rows.sort(key=lambda row: row["frame"])
        if not valid_rows:
            continue

        features = np.stack(temporal_stack(valid_rows, temporal_radius)).astype(np.float32)
        if features.shape[1] != feature_mean.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch for {sequence}: got {features.shape[1]}, "
                f"checkpoint expects {feature_mean.shape[0]}. Check feature preset/temporal radius."
            )
        features = (features - feature_mean) / feature_std
        predictions = []
        with torch.no_grad():
            for start in range(0, len(features), args.batch_size):
                batch = torch.from_numpy(features[start : start + args.batch_size]).to(device)
                predictions.append(model(batch).detach().cpu().numpy())
        deltas = np.concatenate(predictions, axis=0)

        for row, delta in zip(valid_rows, deltas):
            corrected = apply_predicted_correction(
                row["person"],
                delta[:3],
                delta[3:],
                orient_scale=args.orient_scale,
                transl_scale=args.transl_scale,
                max_delta_degrees=args.max_delta_degrees,
            )
            write_person(row["target_path"], corrected)
            rows_written.append(
                {
                    "sequence": row["sequence"],
                    "frame": row["frame"],
                    "file": str(row["target_path"]),
                    "delta_rotvec_x": float(delta[0]),
                    "delta_rotvec_y": float(delta[1]),
                    "delta_rotvec_z": float(delta[2]),
                    "delta_transl_x": float(delta[3]),
                    "delta_transl_y": float(delta[4]),
                    "delta_transl_z": float(delta[5]),
                }
            )

    report_path = args.report or output_root / "evaluation" / f"apply_{args.output_subdir}" / "apply_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = report_path.with_name("applied_corrections.csv")
    if rows_written:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_written[0]))
            writer.writeheader()
            writer.writerows(rows_written)
    report = {
        "pred_root": str(pred_root),
        "output_root": str(output_root),
        "params_subdir": args.params_subdir,
        "output_subdir": args.output_subdir,
        "checkpoint": str(args.checkpoint.resolve()),
        "feature_preset": feature_preset,
        "temporal_radius": temporal_radius,
        "written_frames": len(rows_written),
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:20],
        "corrections_csv": str(csv_path),
        "settings": {
            "orient_scale": args.orient_scale,
            "transl_scale": args.transl_scale,
            "max_delta_degrees": args.max_delta_degrees,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
