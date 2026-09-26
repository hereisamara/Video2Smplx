#!/usr/bin/env python3
"""Rank SignLanguage SMPL-X error sources with parameter counterfactuals.

This is a diagnostic script, not a deployable post-processor.  It uses GT
SMPL-X parameters only to answer: "If this predicted parameter group were
correct, how much would MPJPE/MPVPE improve?"  That makes the main error source
measurable instead of inferred from high-level metrics.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.evaluation.evaluate_signlanguage_geometry import (
    METRICS,
    NumpySMPLX,
    build_sequence_items,
    describe_prediction_root,
    discover_sequences,
    find_ffprobe,
    geometry_metrics,
    load_ground_truth_records,
    load_image_index,
    load_prediction_batch,
    normalize_sequence_name,
    sequence_sort_key,
    summarize,
)


BODY_NAMES = {
    1: "left_hip",
    2: "right_hip",
    3: "spine1",
    4: "left_knee",
    5: "right_knee",
    6: "spine2",
    7: "left_ankle",
    8: "right_ankle",
    9: "spine3",
    10: "left_foot",
    11: "right_foot",
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

GROUPS = {
    "transl": {"transl"},
    "transl_z": {"transl_z"},
    "global_orient": {"global_orient"},
    "root_global": {"transl", "global_orient"},
    "shape": {"betas"},
    "expression_jaw": {"expression", "jaw_pose"},
    "spine": {"body:3", "body:6", "body:9"},
    "neck_head": {"body:12", "body:15"},
    "collars": {"body:13", "body:14"},
    "shoulders": {"body:16", "body:17"},
    "elbows": {"body:18", "body:19"},
    "wrists": {"body:20", "body:21"},
    "arms_no_hands": {"body:13", "body:14", "body:16", "body:17", "body:18", "body:19", "body:20", "body:21"},
    "upper_body_no_hands": {
        "body:3",
        "body:6",
        "body:9",
        "body:12",
        "body:13",
        "body:14",
        "body:15",
        "body:16",
        "body:17",
        "body:18",
        "body:19",
        "body:20",
        "body:21",
    },
    "hands": {"left_hand_pose", "right_hand_pose"},
    "left_hand": {"left_hand_pose"},
    "right_hand": {"right_hand_pose"},
    "upper_body_with_hands": {
        "body:3",
        "body:6",
        "body:9",
        "body:12",
        "body:13",
        "body:14",
        "body:15",
        "body:16",
        "body:17",
        "body:18",
        "body:19",
        "body:20",
        "body:21",
        "left_hand_pose",
        "right_hand_pose",
    },
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--params-subdir", default="fused_params")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        help="Counterfactual groups to test. Use 'all' for the default full set.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for quick diagnostics.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth paired frame.")
    parser.add_argument("--strict-pred", action="store_true")
    parser.add_argument("--strict-gt", action="store_true")
    parser.add_argument("--skip-bad-frames", action="store_true", default=True)
    parser.add_argument("--strict-frame-count", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def gt_to_pred_fields(gt: dict) -> dict[str, np.ndarray]:
    return {
        "global_orient": np.asarray(gt["root_pose"], dtype=np.float32),
        "body_pose": np.asarray(gt["body_pose"], dtype=np.float32),
        "jaw_pose": np.asarray(gt["jaw_pose"], dtype=np.float32),
        "left_hand_pose": np.asarray(gt["lhand_pose"], dtype=np.float32),
        "right_hand_pose": np.asarray(gt["rhand_pose"], dtype=np.float32),
        "betas": np.asarray(gt["shape"], dtype=np.float32),
        "expression": np.asarray(gt["expr"], dtype=np.float32),
        "transl": np.asarray(gt["trans"], dtype=np.float32),
    }


def apply_group_replacement(pred: dict, gt: dict, group_name: str) -> dict:
    out = copy.deepcopy(pred)
    gt_fields = gt_to_pred_fields(gt)
    for token in GROUPS[group_name]:
        if token == "global_orient":
            out["global_orient"] = gt_fields["global_orient"].copy()
        elif token == "transl":
            out["transl"] = gt_fields["transl"].copy()
        elif token == "transl_z":
            transl = np.asarray(out["transl"], dtype=np.float32).copy().reshape(3)
            transl[2] = gt_fields["transl"].reshape(3)[2]
            out["transl"] = transl
        elif token == "betas":
            out["betas"] = gt_fields["betas"].copy()
        elif token == "expression":
            out["expression"] = gt_fields["expression"].copy()
        elif token in ("jaw_pose", "left_hand_pose", "right_hand_pose"):
            out[token] = gt_fields[token].copy()
        elif token.startswith("body:"):
            joint = int(token.split(":", 1)[1])
            body_index = joint - 1
            body = np.asarray(out["body_pose"], dtype=np.float32).copy().reshape(21, 3)
            body_gt = gt_fields["body_pose"].reshape(21, 3)
            body[body_index] = body_gt[body_index]
            out["body_pose"] = body.reshape(-1)
        else:
            raise KeyError(f"Unknown replacement token {token}")
    return out


def collect_sequence_items(args: argparse.Namespace) -> tuple[dict[str, list[dict]], list[str]]:
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
    if not sequences:
        raise ValueError(describe_prediction_root(pred_root, args.params_subdir))

    sequence_data: dict[str, list[dict]] = {}
    for sequence in sequences:
        try:
            items, _ = build_sequence_items(sequence, args, video_dir, pred_root, image_index, ffprobe, warnings)
        except Exception as exc:
            warnings.append(f"{sequence}: skipped setup error: {exc}")
            continue
        if args.frame_stride > 1:
            items = items[:: args.frame_stride]
        sequence_data[sequence] = items

    if args.max_frames > 0:
        remaining = args.max_frames
        limited: dict[str, list[dict]] = {}
        for sequence in sorted(sequence_data, key=sequence_sort_key):
            if remaining <= 0:
                break
            take = sequence_data[sequence][:remaining]
            if take:
                limited[sequence] = take
            remaining -= len(take)
        sequence_data = limited
    return sequence_data, warnings


def joint_error_rows(
    predicted_joints: np.ndarray,
    target_joints: np.ndarray,
    sequence: str,
    prediction_frame: int,
) -> list[dict]:
    pred_rooted = predicted_joints - predicted_joints[0]
    target_rooted = target_joints - target_joints[0]
    errors = np.linalg.norm(pred_rooted - target_rooted, axis=1) * 1000.0
    rows = []
    for joint, name in BODY_NAMES.items():
        rows.append(
            {
                "sequence": sequence,
                "prediction_frame": prediction_frame,
                "joint": joint,
                "name": name,
                "root_relative_error_mm": float(errors[joint]),
            }
        )
    rows.append(
        {
            "sequence": sequence,
            "prediction_frame": prediction_frame,
            "joint": "left_hand_mean",
            "name": "left_hand_mean",
            "root_relative_error_mm": float(np.mean(errors[25:40])),
        }
    )
    rows.append(
        {
            "sequence": sequence,
            "prediction_frame": prediction_frame,
            "joint": "right_hand_mean",
            "name": "right_hand_mean",
            "root_relative_error_mm": float(np.mean(errors[40:55])),
        }
    )
    return rows


def parameter_error_rows(pred: dict, gt: dict, sequence: str, prediction_frame: int) -> list[dict]:
    rows = []
    pred_t = np.asarray(pred["transl"], dtype=np.float64).reshape(3)
    gt_t = np.asarray(gt["trans"], dtype=np.float64).reshape(3)
    delta_t = pred_t - gt_t
    for name, value in (
        ("transl_l2_m", np.linalg.norm(delta_t)),
        ("transl_x_abs_m", abs(delta_t[0])),
        ("transl_y_abs_m", abs(delta_t[1])),
        ("transl_z_abs_m", abs(delta_t[2])),
        ("global_orient_deg", geodesic_degrees(pred["global_orient"], gt["root_pose"])[0]),
    ):
        rows.append(
            {
                "sequence": sequence,
                "prediction_frame": prediction_frame,
                "parameter": name,
                "unit": "m" if name.endswith("_m") else "deg",
                "error": float(value),
            }
        )

    body_pred = np.asarray(pred["body_pose"], dtype=np.float64).reshape(21, 3)
    body_gt = np.asarray(gt["body_pose"], dtype=np.float64).reshape(21, 3)
    for joint, name in BODY_NAMES.items():
        value = geodesic_degrees(body_pred[joint - 1], body_gt[joint - 1])[0]
        rows.append(
            {
                "sequence": sequence,
                "prediction_frame": prediction_frame,
                "parameter": f"body_pose.{name}",
                "unit": "deg",
                "error": float(value),
            }
        )
    for pred_key, gt_key, prefix in (
        ("left_hand_pose", "lhand_pose", "left_hand_pose"),
        ("right_hand_pose", "rhand_pose", "right_hand_pose"),
    ):
        values = geodesic_degrees(
            np.asarray(pred[pred_key], dtype=np.float64).reshape(15, 3),
            np.asarray(gt[gt_key], dtype=np.float64).reshape(15, 3),
        )
        for index, value in enumerate(values):
            rows.append(
                {
                    "sequence": sequence,
                    "prediction_frame": prediction_frame,
                    "parameter": f"{prefix}.{index:02d}",
                    "unit": "deg",
                    "error": float(value),
                }
            )
        rows.append(
            {
                "sequence": sequence,
                "prediction_frame": prediction_frame,
                "parameter": f"{prefix}.mean",
                "unit": "deg",
                "error": float(np.mean(values)),
            }
        )
    return rows


def aggregate_simple(rows: list[dict], key: str, value: str) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    units: dict[str, str] = {}
    for row in rows:
        grouped[str(row[key])].append(float(row[value]))
        if "unit" in row:
            units[str(row[key])] = str(row["unit"])
    out = []
    for name, values in grouped.items():
        item = {
            key: name,
            "mean": mean(values),
            "median": percentile(values, 50),
            "p95": percentile(values, 95),
            "frames": len(values),
        }
        if name in units:
            item["unit"] = units[name]
        out.append(item)
    out.sort(key=lambda row: row["mean"], reverse=True)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()
    annotation_dir = (args.annotation_dir or root / "datasets" / "annotations" / "SignLanguage").resolve()
    pred_root = (args.pred_root or root / "outputs_fusion_signlanguage").resolve()
    group_names = list(GROUPS) if args.groups == ["all"] else args.groups
    unknown_groups = sorted(set(group_names) - set(GROUPS))
    if unknown_groups:
        raise KeyError(f"Unknown groups: {unknown_groups}. Available: {sorted(GROUPS)}")

    sequence_data, warnings = collect_sequence_items(args)
    wanted_ids = {item["ground_truth_id"] for items in sequence_data.values() for item in items}
    if not wanted_ids:
        raise ValueError("No frame pairs available")
    ground_truth, missing_gt_ids = load_ground_truth_records(
        annotation_dir / "smplx_annotation.json",
        wanted_ids,
        allow_missing=not args.strict_gt,
    )
    missing_gt = set(missing_gt_ids)
    if missing_gt_ids:
        warnings.append(f"missing GT records skipped: {len(missing_gt_ids)}")

    model = NumpySMPLX(args.model_path.resolve())
    skipped_frames: list[dict] = []
    baseline_rows: list[dict] = []
    counterfactual_rows: list[dict] = []
    joint_rows: list[dict] = []
    param_rows: list[dict] = []
    summary_rows: list[dict] = []
    sequence_baseline_rows: dict[str, list[dict]] = defaultdict(list)

    for sequence, items in sequence_data.items():
        items = [item for item in items if item["ground_truth_id"] not in missing_gt]
        if not items:
            continue
        print(f"Diagnosing {sequence}: {len(items)} frames", flush=True)
        for start in range(0, len(items), args.batch_size):
            batch_items = items[start : start + args.batch_size]
            if start == 0 or (start // args.batch_size) % 25 == 0:
                print(
                    f"  {sequence}: batch {start // args.batch_size + 1}/"
                    f"{(len(items) + args.batch_size - 1) // args.batch_size}",
                    flush=True,
                )
            batch_items, pred_params = load_prediction_batch(
                batch_items,
                strict_pred=args.strict_pred,
                skipped_frames=skipped_frames,
                warnings=warnings,
            )
            if not batch_items:
                continue
            gt_params = [ground_truth[item["ground_truth_id"]]["smplx_param"] for item in batch_items]

            pred_vertices, pred_joints = model.forward(pred_params, predicted=True)
            gt_vertices, gt_joints = model.forward(gt_params, predicted=False)
            baseline_metrics = []
            for index, item in enumerate(batch_items):
                metrics = geometry_metrics(
                    pred_vertices[index],
                    pred_joints[index],
                    gt_vertices[index],
                    gt_joints[index],
                    model,
                )
                baseline_metrics.append(metrics)
                row = {
                    "sequence": sequence,
                    "prediction_frame": item["prediction_frame"],
                    "ground_truth_id": item["ground_truth_id"],
                    **metrics,
                }
                baseline_rows.append(row)
                sequence_baseline_rows[sequence].append(row)
                joint_rows.extend(
                    joint_error_rows(pred_joints[index], gt_joints[index], sequence, item["prediction_frame"])
                )
                param_rows.extend(
                    parameter_error_rows(
                        pred_params[index],
                        gt_params[index],
                        sequence,
                        item["prediction_frame"],
                    )
                )

            for group_name in group_names:
                replaced = [
                    apply_group_replacement(pred, gt, group_name)
                    for pred, gt in zip(pred_params, gt_params)
                ]
                cf_vertices, cf_joints = model.forward(replaced, predicted=True)
                for index, item in enumerate(batch_items):
                    cf_metrics = geometry_metrics(
                        cf_vertices[index],
                        cf_joints[index],
                        gt_vertices[index],
                        gt_joints[index],
                        model,
                    )
                    row = {
                        "sequence": sequence,
                        "prediction_frame": item["prediction_frame"],
                        "ground_truth_id": item["ground_truth_id"],
                        "group": group_name,
                    }
                    for metric in METRICS:
                        row[f"{metric}_before"] = baseline_metrics[index][metric]
                        row[f"{metric}_after"] = cf_metrics[metric]
                        row[f"{metric}_improvement"] = baseline_metrics[index][metric] - cf_metrics[metric]
                    counterfactual_rows.append(row)

    if not baseline_rows:
        raise ValueError("No usable diagnostic rows")

    summary_rows.extend(summarize(sequence_baseline_rows[sequence], sequence) for sequence in sorted(sequence_baseline_rows, key=sequence_sort_key))
    summary_rows.append(summarize(baseline_rows, "ALL"))

    cf_summary = []
    for group_name in group_names:
        rows = [row for row in counterfactual_rows if row["group"] == group_name]
        if not rows:
            continue
        item: dict[str, str | int | float] = {"group": group_name, "frames": len(rows)}
        for metric in (
            "mpjpe_mm",
            "mpvpe_mm",
            "pa_mpvpe_mm",
            "body_mpjpe_mm",
            "visible_upper_mpjpe_mm",
            "visible_upper_mpvpe_mm",
            "hands_wrist_mpvpe_mm",
            "hands_pa_mpvpe_mm",
            "face_head_mpvpe_mm",
        ):
            before = [row[f"{metric}_before"] for row in rows]
            after = [row[f"{metric}_after"] for row in rows]
            improvement = [row[f"{metric}_improvement"] for row in rows]
            item[f"{metric}_before_mean"] = mean(before)
            item[f"{metric}_after_mean"] = mean(after)
            item[f"{metric}_improvement_mean"] = mean(improvement)
        cf_summary.append(item)
    cf_summary.sort(key=lambda row: row["visible_upper_mpvpe_mm_improvement_mean"], reverse=True)

    joint_summary = aggregate_simple(joint_rows, "name", "root_relative_error_mm")
    parameter_summary = aggregate_simple(param_rows, "parameter", "error")

    write_csv(args.output_dir / "baseline_geometry_summary.csv", summary_rows)
    write_csv(args.output_dir / "baseline_geometry_frame_metrics.csv", baseline_rows)
    write_csv(args.output_dir / "counterfactual_group_summary.csv", cf_summary)
    write_csv(args.output_dir / "counterfactual_group_frame_metrics.csv", counterfactual_rows)
    write_csv(args.output_dir / "joint_error_summary.csv", joint_summary)
    write_csv(args.output_dir / "joint_error_frame_metrics.csv", joint_rows)
    write_csv(args.output_dir / "parameter_error_summary.csv", parameter_summary)
    write_csv(args.output_dir / "parameter_error_frame_metrics.csv", param_rows)
    write_csv(args.output_dir / "skipped_frames.csv", skipped_frames)

    report = {
        "paths": {
            "root": str(root),
            "annotation_dir": str(annotation_dir),
            "pred_root": str(pred_root.resolve()),
            "params_subdir": args.params_subdir,
            "model_path": str(args.model_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
        },
        "frames": len(baseline_rows),
        "groups": group_names,
        "warnings": warnings,
        "top_counterfactuals_by_visible_upper_mpvpe": cf_summary[:10],
        "top_joint_errors": joint_summary[:15],
        "top_parameter_errors": parameter_summary[:20],
        "notes": {
            "counterfactuals": "Each group is replaced with GT SMPL-X params one at a time. This is oracle-only root-cause analysis, not a deployable method.",
            "improvement": "Positive improvement means the metric decreased after replacing that parameter group with GT.",
            "joint_errors": "Root-relative SMPL-X joint errors, measured after subtracting the pelvis/root joint.",
        },
    }
    (args.output_dir / "error_source_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("group,visible_upper_mpvpe_before,after,improvement,mpvpe_improvement,hands_wrist_mpvpe_improvement", flush=True)
    for row in cf_summary[:15]:
        print(
            f"{row['group']},"
            f"{row['visible_upper_mpvpe_mm_before_mean']:.2f},"
            f"{row['visible_upper_mpvpe_mm_after_mean']:.2f},"
            f"{row['visible_upper_mpvpe_mm_improvement_mean']:.2f},"
            f"{row['mpvpe_mm_improvement_mean']:.2f},"
            f"{row['hands_wrist_mpvpe_mm_improvement_mean']:.2f}",
            flush=True,
        )
    print(f"\nWrote error-source diagnostic to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
