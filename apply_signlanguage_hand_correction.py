#!/usr/bin/env python3
"""Apply a trained SignLanguage hand correction model to prediction PKLs."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from evaluate_signlanguage_geometry import sequence_sort_key
from oracle_optimize_signlanguage_global import matrix_to_rotvec, rebuild_param_vector, rotvec_to_matrix
from signlanguage_global_correction import (
    base_feature,
    copy_sidecar_files,
    discover_sequences,
    frame_number,
    list_prediction_files,
    load_one_person_or_none,
    normalize_sequence_name,
)
from train_signlanguage_global_correction import build_model, import_torch


LEFT_WRIST_BODY_INDEX = 20 - 1
RIGHT_WRIST_BODY_INDEX = 21 - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--params-subdir", default="fused_params")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-subdir", default="fused_params_hand_corrected")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--feature-preset", choices=("auto", "minimal", "upper_global", "full"), default="auto")
    parser.add_argument("--temporal-radius", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-delta-degrees", type=float, default=90.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def write_person(path: Path, person: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([person], handle)


def temporal_stack(rows: list[dict], radius: int) -> list[np.ndarray]:
    if radius <= 0:
        return [row["base_feature"] for row in rows]
    features = [row["base_feature"] for row in rows]
    output = []
    last = len(features) - 1
    for idx in range(len(features)):
        output.append(np.concatenate([features[min(max(idx + off, 0), last)] for off in range(-radius, radius + 1)]))
    return output


def compose_rotvec(current_rotvec: np.ndarray, delta_rotvec: np.ndarray, max_degrees: float, scale: float) -> np.ndarray:
    delta = np.asarray(delta_rotvec, dtype=np.float64).reshape(3) * float(scale)
    if max_degrees > 0:
        max_norm = np.deg2rad(max_degrees)
        norm = float(np.linalg.norm(delta))
        if norm > max_norm:
            delta = delta * (max_norm / norm)
    current = rotvec_to_matrix(np.asarray(current_rotvec, dtype=np.float64).reshape(3))
    return matrix_to_rotvec(rotvec_to_matrix(delta) @ current).astype(np.float32)


def apply_hand_delta(person: dict, delta: np.ndarray, target_mode: str, max_degrees: float, scale: float) -> dict:
    output = {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in person.items()
    }
    cursor = 0
    body = np.asarray(output["body_pose"], dtype=np.float32).reshape(21, 3).copy()
    if target_mode in ("wrist", "wrist_fingers"):
        body[LEFT_WRIST_BODY_INDEX] = compose_rotvec(body[LEFT_WRIST_BODY_INDEX], delta[cursor : cursor + 3], max_degrees, scale)
        cursor += 3
        body[RIGHT_WRIST_BODY_INDEX] = compose_rotvec(body[RIGHT_WRIST_BODY_INDEX], delta[cursor : cursor + 3], max_degrees, scale)
        cursor += 3
        output["body_pose"] = body.reshape(-1)
    if target_mode in ("fingers", "wrist_fingers"):
        left = np.asarray(output["left_hand_pose"], dtype=np.float32).reshape(15, 3).copy()
        right = np.asarray(output["right_hand_pose"], dtype=np.float32).reshape(15, 3).copy()
        for index in range(15):
            left[index] = compose_rotvec(left[index], delta[cursor : cursor + 3], max_degrees, scale)
            cursor += 3
        for index in range(15):
            right[index] = compose_rotvec(right[index], delta[cursor : cursor + 3], max_degrees, scale)
            cursor += 3
        output["left_hand_pose"] = left.reshape(-1)
        output["right_hand_pose"] = right.reshape(-1)
    rebuild_param_vector(output)
    return output


def main() -> None:
    args = parse_args()
    torch, nn, _, _ = import_torch()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    feature_preset = checkpoint.get("feature_preset", "upper_global") if args.feature_preset == "auto" else args.feature_preset
    temporal_radius = int(checkpoint.get("temporal_radius", 1)) if args.temporal_radius < 0 else args.temporal_radius
    target_mode = checkpoint.get("target_mode", "wrist_fingers")
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    std[std < 1e-6] = 1.0

    model = build_model(nn, int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), int(checkpoint["layers"]), float(checkpoint["dropout"])).to(device)
    if hasattr(model[-1], "in_features") and int(checkpoint["output_dim"]) != model[-1].out_features:
        model[-1] = nn.Linear(model[-1].in_features, int(checkpoint["output_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    pred_root = args.pred_root.resolve()
    output_root = (args.output_root or args.pred_root).resolve()
    if args.sequences == ["all"]:
        sequences = discover_sequences(pred_root, args.params_subdir)
    else:
        sequences = sorted([normalize_sequence_name(x) for x in args.sequences], key=sequence_sort_key)

    rows_written = []
    skipped = []
    for sequence in sequences:
        source_dir = pred_root / sequence / args.params_subdir
        target_dir = output_root / sequence / args.output_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        copy_sidecar_files(source_dir, target_dir)
        rows = []
        for source_path in list_prediction_files(pred_root, sequence, args.params_subdir):
            person = load_one_person_or_none(source_path)
            if person is None:
                shutil.copy2(source_path, target_dir / source_path.name)
                skipped.append({"sequence": sequence, "file": str(source_path), "reason": "bad_or_empty_prediction_copied"})
                continue
            try:
                feature = base_feature(person, feature_preset)
            except Exception as exc:
                skipped.append({"sequence": sequence, "file": str(source_path), "reason": f"feature_error: {exc}"})
                shutil.copy2(source_path, target_dir / source_path.name)
                continue
            rows.append({"sequence": sequence, "frame": frame_number(source_path), "source_path": source_path, "target_path": target_dir / source_path.name, "person": person, "base_feature": feature})
        rows.sort(key=lambda row: row["frame"])
        if not rows:
            continue
        features = np.stack(temporal_stack(rows, temporal_radius)).astype(np.float32)
        if features.shape[1] != mean.shape[0]:
            raise ValueError(f"Feature dimension mismatch: got {features.shape[1]}, expected {mean.shape[0]}")
        features = (features - mean) / std
        preds = []
        with torch.no_grad():
            for start in range(0, len(features), args.batch_size):
                batch = torch.from_numpy(features[start : start + args.batch_size]).to(device)
                preds.append(model(batch).detach().cpu().numpy())
        deltas = np.concatenate(preds, axis=0)
        for row, delta in zip(rows, deltas):
            corrected = apply_hand_delta(row["person"], delta, target_mode, args.max_delta_degrees, args.scale)
            write_person(row["target_path"], corrected)
            rows_written.append({"sequence": row["sequence"], "frame": row["frame"], "file": str(row["target_path"]), "delta_l2": float(np.linalg.norm(delta))})

    report_path = args.report or output_root / "evaluation" / f"apply_{args.output_subdir}" / "apply_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = report_path.with_name("applied_hand_corrections.csv")
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
        "target_mode": target_mode,
        "written_frames": len(rows_written),
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:20],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
