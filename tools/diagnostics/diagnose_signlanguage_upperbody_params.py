#!/usr/bin/env python3
"""Diagnose which upper-body SMPL-X parameters differ most from SignLanguage GT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.evaluation.evaluate_signlanguage_geometry import (
    build_sequence_items,
    discover_sequences,
    find_ffprobe,
    load_ground_truth_records,
    load_image_index,
    normalize_sequence_name,
    one_person,
    sequence_sort_key,
)


BODY_JOINT_NAMES = {
    3: "spine1",
    6: "spine2",
    9: "spine3",
    12: "neck",
    13: "left_collar",
    14: "right_collar",
    15: "head",
    16: "left_shoulder",
    17: "right_shoulder",
    18: "left_elbow",
    19: "right_elbow",
    20: "left_wrist",
    21: "right_wrist",
}

LEFT_HAND_NAMES = [f"left_hand_{i:02d}" for i in range(15)]
RIGHT_HAND_NAMES = [f"right_hand_{i:02d}" for i in range(15)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--params-subdir", default="fused_params")
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def rodrigues(axis_angles: np.ndarray) -> np.ndarray:
    vectors = np.asarray(axis_angles, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(vectors, axis=1)
    result = np.repeat(np.eye(3, dtype=np.float64)[None], len(vectors), axis=0)
    active = theta > 1e-12
    axes = vectors[active] / theta[active, None]
    skew = np.zeros((len(axes), 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -axes[:, 2]
    skew[:, 0, 2] = axes[:, 1]
    skew[:, 1, 0] = axes[:, 2]
    skew[:, 1, 2] = -axes[:, 0]
    skew[:, 2, 0] = -axes[:, 1]
    skew[:, 2, 1] = axes[:, 0]
    angles = theta[active]
    result[active] = (
        np.eye(3, dtype=np.float64)[None]
        + np.sin(angles)[:, None, None] * skew
        + (1.0 - np.cos(angles))[:, None, None] * (skew @ skew)
    )
    return result


def geodesic_degrees(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_rotation = rodrigues(predicted)
    target_rotation = rodrigues(target)
    relative = pred_rotation @ np.swapaxes(target_rotation, 1, 2)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_items(args: argparse.Namespace) -> tuple[dict[str, list[dict]], list[str]]:
    root = args.root.resolve()
    video_dir = (args.video_dir or root / "datasets" / "videos" / "SignLanguage").resolve()
    annotation_dir = (args.annotation_dir or root / "datasets" / "annotations" / "SignLanguage").resolve()
    pred_root = (args.pred_root or root / "outputs_fusion_signlanguage").resolve()
    image_index = load_image_index(annotation_dir / "keypoint_annotation.json")
    ffprobe = find_ffprobe()
    warnings: list[str] = []

    if args.sequences == ["all"]:
        sequences = discover_sequences(pred_root, args.params_subdir)
    else:
        sequences = sorted([normalize_sequence_name(x) for x in args.sequences], key=sequence_sort_key)

    sequence_data = {}
    for sequence in sequences:
        try:
            items, _ = build_sequence_items(sequence, args, video_dir, pred_root, image_index, ffprobe, warnings)
        except Exception as exc:
            warnings.append(f"{sequence}: skipped setup error: {exc}")
            continue
        sequence_data[sequence] = items
    return sequence_data, warnings


def add_metric(metrics: dict[str, list[float]], name: str, value: float) -> None:
    metrics.setdefault(name, []).append(float(value))


def main() -> None:
    args = parse_args()
    args.strict_frame_count = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir = (args.annotation_dir or args.root / "datasets" / "annotations" / "SignLanguage").resolve()
    sequence_data, warnings = collect_items(args)
    wanted_ids = {item["ground_truth_id"] for items in sequence_data.values() for item in items}
    gt_records, missing_gt = load_ground_truth_records(
        annotation_dir / "smplx_annotation.json",
        wanted_ids,
        allow_missing=True,
    )
    if missing_gt:
        warnings.append(f"missing GT records skipped: {len(missing_gt)}")

    metrics: dict[str, list[float]] = {}
    frame_rows = []
    skipped = []
    total = 0
    for sequence, items in sequence_data.items():
        for item in items:
            if item["ground_truth_id"] not in gt_records:
                skipped.append({**item, "reason": "missing_gt"})
                continue
            try:
                pred = one_person(item["file"])
            except Exception as exc:
                skipped.append({**item, "reason": str(exc)})
                continue
            total += 1
            gt = gt_records[item["ground_truth_id"]]["smplx_param"]

            global_deg = geodesic_degrees(pred["global_orient"], gt["root_pose"])[0]
            add_metric(metrics, "global_orient", global_deg)

            pred_transl = np.asarray(pred["transl"], dtype=np.float64).reshape(3)
            gt_transl = np.asarray(gt["trans"], dtype=np.float64).reshape(3)
            delta_t = pred_transl - gt_transl
            add_metric(metrics, "transl_l2_m", np.linalg.norm(delta_t))
            add_metric(metrics, "transl_x_abs_m", abs(delta_t[0]))
            add_metric(metrics, "transl_y_abs_m", abs(delta_t[1]))
            add_metric(metrics, "transl_z_abs_m", abs(delta_t[2]))

            body_pred = np.asarray(pred["body_pose"], dtype=np.float64).reshape(21, 3)
            body_gt = np.asarray(gt["body_pose"], dtype=np.float64).reshape(21, 3)
            for smplx_joint, name in BODY_JOINT_NAMES.items():
                body_index = smplx_joint - 1
                value = geodesic_degrees(body_pred[body_index], body_gt[body_index])[0]
                add_metric(metrics, f"body_pose.{name}", value)
            upper_body_degrees = [
                metrics[f"body_pose.{name}"][-1]
                for name in BODY_JOINT_NAMES.values()
            ]

            jaw_deg = geodesic_degrees(pred["jaw_pose"], gt["jaw_pose"])[0]
            add_metric(metrics, "jaw_pose", jaw_deg)

            left_pred = np.asarray(pred["left_hand_pose"], dtype=np.float64).reshape(15, 3)
            left_gt = np.asarray(gt["lhand_pose"], dtype=np.float64).reshape(15, 3)
            right_pred = np.asarray(pred["right_hand_pose"], dtype=np.float64).reshape(15, 3)
            right_gt = np.asarray(gt["rhand_pose"], dtype=np.float64).reshape(15, 3)
            left_degrees = geodesic_degrees(left_pred, left_gt)
            right_degrees = geodesic_degrees(right_pred, right_gt)
            for name, value in zip(LEFT_HAND_NAMES, left_degrees):
                add_metric(metrics, f"left_hand_pose.{name}", value)
            for name, value in zip(RIGHT_HAND_NAMES, right_degrees):
                add_metric(metrics, f"right_hand_pose.{name}", value)
            add_metric(metrics, "left_hand_pose.mean", float(np.mean(left_degrees)))
            add_metric(metrics, "right_hand_pose.mean", float(np.mean(right_degrees)))
            add_metric(metrics, "hands_pose.mean", float(np.mean(np.concatenate([left_degrees, right_degrees]))))

            frame_rows.append(
                {
                    "sequence": sequence,
                    "prediction_frame": item["prediction_frame"],
                    "ground_truth_id": item["ground_truth_id"],
                    "global_orient_deg": global_deg,
                    "transl_l2_m": float(np.linalg.norm(delta_t)),
                    "transl_x_abs_m": abs(delta_t[0]),
                    "transl_y_abs_m": abs(delta_t[1]),
                    "transl_z_abs_m": abs(delta_t[2]),
                    "upper_body_pose_deg_mean": mean(upper_body_degrees),
                    "hands_pose_deg_mean": float(np.mean(np.concatenate([left_degrees, right_degrees]))),
                    "jaw_pose_deg": jaw_deg,
                }
            )

    summary_rows = []
    for name, values in metrics.items():
        unit = "m" if name.startswith("transl") else "deg"
        summary_rows.append(
            {
                "parameter": name,
                "unit": unit,
                "mean": mean(values),
                "median": percentile(values, 50),
                "p95": percentile(values, 95),
                "frames": len(values),
            }
        )
    summary_rows.sort(key=lambda row: row["mean"], reverse=True)
    write_csv(args.output_dir / "upperbody_parameter_summary.csv", summary_rows)
    write_csv(args.output_dir / "upperbody_parameter_frame_metrics.csv", frame_rows)
    write_csv(args.output_dir / "skipped_frames.csv", skipped)

    report = {
        "usable_frames": total,
        "skipped_frames": len(skipped),
        "warnings": warnings,
        "top_parameters": summary_rows[: args.top_k],
        "notes": {
            "rotation_metrics": "SO(3) geodesic degrees between predicted parameter and GT parameter",
            "translation_metrics": "absolute/l2 parameter difference in SMPL-X translation units, normally metres",
            "upper_body_body_pose_joints": BODY_JOINT_NAMES,
        },
    }
    (args.output_dir / "upperbody_parameter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("parameter,unit,mean,median,p95,frames")
    for row in summary_rows[: args.top_k]:
        print(
            f"{row['parameter']},{row['unit']},{row['mean']:.4f},"
            f"{row['median']:.4f},{row['p95']:.4f},{row['frames']}"
        )
    print(f"\nWrote upper-body parameter diagnostic to {args.output_dir}")


if __name__ == "__main__":
    main()
