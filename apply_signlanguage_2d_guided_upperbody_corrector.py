#!/usr/bin/env python3
"""Apply a trained 2D-guided upper-body corrector to SignLanguage PKLs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from build_signlanguage_2d_guided_upperbody_dataset import guided_feature, get_yolo_frame, load_yolo
from evaluate_signlanguage_geometry import NumpySMPLX, sequence_sort_key
from oracle_optimize_signlanguage_global import matrix_to_rotvec, rebuild_param_vector, rotvec_to_matrix
from signlanguage_global_correction import (
    base_feature,
    copy_person,
    copy_sidecar_files,
    discover_sequences,
    frame_number,
    list_prediction_files,
    load_one_person_or_none,
    normalize_sequence_name,
    write_person,
)
from train_signlanguage_global_correction import build_model, import_torch
from transform_signlanguage_feature_ablation_dataset import transform_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--params-subdir", default="fused_params_model_global_hand_corrected")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-subdir", default="fused_params_2d_upper_corrected")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-keypoints", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--feature-preset", choices=("auto", "minimal", "upper_global", "full"), default="auto")
    parser.add_argument("--temporal-radius", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--arms-scale", type=float, default=1.0)
    parser.add_argument("--max-global-delta-degrees", type=float, default=35.0)
    parser.add_argument("--max-arm-delta-degrees", type=float, default=45.0)
    parser.add_argument("--min-keypoint-conf", type=float, default=0.15)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


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


def normalize_indices(values) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(item) for item in values]


def apply_delta(person: dict, delta: np.ndarray, body_indices: list[int], args: argparse.Namespace) -> dict:
    output = copy_person(person)
    delta = np.asarray(delta, dtype=np.float32).reshape(-1)
    expected = 3 + len(body_indices) * 3
    if delta.shape[0] != expected:
        raise ValueError(f"Expected {expected} values, got {delta.shape[0]}")
    output["global_orient"] = compose_rotvec(
        output["global_orient"],
        delta[:3],
        args.max_global_delta_degrees,
        args.scale * args.global_scale,
    )
    body = np.asarray(output["body_pose"], dtype=np.float32).reshape(21, 3).copy()
    for index, delta_rotvec in zip(body_indices, delta[3:].reshape(len(body_indices), 3)):
        body[index] = compose_rotvec(
            body[index],
            delta_rotvec,
            args.max_arm_delta_degrees,
            args.scale * args.arms_scale,
        )
    output["body_pose"] = body.reshape(-1)
    rebuild_param_vector(output)
    return output


def main() -> None:
    args = parse_args()
    torch, nn, _, _ = import_torch()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    feature_preset = checkpoint.get("feature_preset", "upper_global") if args.feature_preset == "auto" else args.feature_preset
    temporal_radius = int(checkpoint.get("temporal_radius", 1)) if args.temporal_radius < 0 else args.temporal_radius
    feature_ablation = str(checkpoint.get("feature_ablation", "full"))
    base_feature_dim = int(checkpoint.get("base_feature_dim", 168))
    body_indices = normalize_indices(checkpoint.get("target_body_indices"))
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    std[std < 1e-6] = 1.0

    model = build_model(nn, int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), int(checkpoint["layers"]), float(checkpoint["dropout"])).to(device)
    if hasattr(model[-1], "in_features") and int(checkpoint["output_dim"]) != model[-1].out_features:
        model[-1] = nn.Linear(model[-1].in_features, int(checkpoint["output_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    needs_detector = feature_ablation != "smplx_only"
    if needs_detector and args.yolo_keypoints is None:
        raise ValueError("--yolo-keypoints is required for detector-guided checkpoints")
    smplx = NumpySMPLX(args.model_path.resolve()) if needs_detector else None
    yolo = load_yolo(args.yolo_keypoints) if needs_detector else {}
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
        files = list_prediction_files(pred_root, sequence, args.params_subdir)
        rows = []
        for start in range(0, len(files), 64):
            batch_files = files[start : start + 64]
            persons = []
            kept_files = []
            for source_path in batch_files:
                person = load_one_person_or_none(source_path)
                if person is None:
                    shutil.copy2(source_path, target_dir / source_path.name)
                    skipped.append({"sequence": sequence, "file": str(source_path), "reason": "bad_or_empty_prediction_copied"})
                    continue
                persons.append(person)
                kept_files.append(source_path)
            if not kept_files:
                continue
            pred_joints = smplx.forward_joints(persons, predicted=True) if needs_detector else None
            for idx, source_path in enumerate(kept_files):
                frame = frame_number(source_path)
                try:
                    if needs_detector:
                        yolo_frame = get_yolo_frame(yolo, sequence, frame)
                        if yolo_frame is None:
                            shutil.copy2(source_path, target_dir / source_path.name)
                            skipped.append({"sequence": sequence, "file": str(source_path), "reason": "missing_yolo_copied"})
                            continue
                        feature = guided_feature(
                            persons[idx],
                            pred_joints[idx],
                            yolo_frame,
                            feature_preset,
                            args.min_keypoint_conf,
                        )
                    else:
                        feature = base_feature(persons[idx], feature_preset)
                except Exception as exc:
                    shutil.copy2(source_path, target_dir / source_path.name)
                    skipped.append({"sequence": sequence, "file": str(source_path), "reason": f"feature_error_copied: {exc}"})
                    continue
                rows.append({"sequence": sequence, "frame": frame, "source_path": source_path, "target_path": target_dir / source_path.name, "person": persons[idx], "base_feature": feature})
        rows.sort(key=lambda row: row["frame"])
        if not rows:
            continue
        features = np.stack(temporal_stack(rows, temporal_radius)).astype(np.float32)
        if needs_detector and feature_ablation != "full":
            features, _ = transform_features(
                features,
                mode=feature_ablation,
                temporal_radius=temporal_radius,
                base_feature_dim=base_feature_dim,
            )
        if features.shape[1] != mean.shape[0]:
            raise ValueError(f"Feature dimension mismatch: got {features.shape[1]}, expected {mean.shape[0]}")
        features = (features - mean) / std
        preds = []
        with torch.no_grad():
            for start in range(0, len(features), args.batch_size):
                preds.append(model(torch.from_numpy(features[start : start + args.batch_size]).to(device)).detach().cpu().numpy())
        deltas = np.concatenate(preds, axis=0)
        for row, delta in zip(rows, deltas):
            corrected = apply_delta(row["person"], delta, body_indices, args)
            write_person(row["target_path"], corrected)
            rows_written.append(
                {
                    "sequence": row["sequence"],
                    "frame": row["frame"],
                    "file": str(row["target_path"]),
                    "global_delta_l2_deg": float(np.degrees(np.linalg.norm(delta[:3]))),
                    "arms_delta_l2_deg_mean": float(np.degrees(np.linalg.norm(delta[3:].reshape(-1, 3), axis=1)).mean()),
                }
            )
        print(f"{sequence}: wrote {len(rows)}", flush=True)

    report_path = args.report or output_root / "evaluation" / f"apply_{args.output_subdir}" / "apply_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if rows_written:
        with report_path.with_name("applied_2d_guided_upperbody_corrections.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_written[0]))
            writer.writeheader()
            writer.writerows(rows_written)
    report = {
        "pred_root": str(pred_root),
        "output_root": str(output_root),
        "params_subdir": args.params_subdir,
        "output_subdir": args.output_subdir,
        "checkpoint": str(args.checkpoint.resolve()),
        "yolo_keypoints": str(args.yolo_keypoints.resolve()) if args.yolo_keypoints is not None else None,
        "feature_preset": feature_preset,
        "feature_ablation": feature_ablation,
        "base_feature_dim": base_feature_dim,
        "temporal_radius": temporal_radius,
        "target_body_indices": body_indices,
        "scale": args.scale,
        "global_scale": args.global_scale,
        "arms_scale": args.arms_scale,
        "written_frames": len(rows_written),
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:20],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
