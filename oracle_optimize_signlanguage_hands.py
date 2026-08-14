#!/usr/bin/env python3
"""GT-based hand oracle diagnostics for SignLanguage SMPL-X predictions.

This measures how much hand metrics could improve by changing only hand-related
parameters. It is diagnostic only because it reads GT.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_signlanguage_geometry import (
    NumpySMPLX,
    build_sequence_items,
    discover_sequences,
    find_ffprobe,
    geometry_metrics,
    load_ground_truth_records,
    load_image_index,
    normalize_sequence_name,
    one_person,
    sequence_sort_key,
    summarize,
)
from oracle_optimize_signlanguage_global import (
    matrix_to_rotvec,
    rebuild_param_vector,
    rigid_rotation_row,
    rotvec_to_matrix,
)


LEFT_WRIST_SMPLX = 20
RIGHT_WRIST_SMPLX = 21
LEFT_WRIST_BODY_INDEX = LEFT_WRIST_SMPLX - 1
RIGHT_WRIST_BODY_INDEX = RIGHT_WRIST_SMPLX - 1
ARM_CHAIN_BODY_INDICES = (
    16 - 1,  # left shoulder
    17 - 1,  # right shoulder
    18 - 1,  # left elbow
    19 - 1,  # right elbow
    LEFT_WRIST_BODY_INDEX,
    RIGHT_WRIST_BODY_INDEX,
)
LEFT_HAND_JOINTS = np.arange(25, 40, dtype=np.int64)
RIGHT_HAND_JOINTS = np.arange(40, 55, dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--params-subdir", default="fused_params")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--strict-pred", action="store_true")
    parser.add_argument("--strict-gt", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_person(person: dict) -> dict:
    return {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in person.items()
    }


def set_body_pose_joint(person: dict, body_index: int, rotvec: np.ndarray) -> None:
    body = np.asarray(person["body_pose"], dtype=np.float32).reshape(21, 3).copy()
    body[body_index] = np.asarray(rotvec, dtype=np.float32).reshape(3)
    person["body_pose"] = body.reshape(-1)
    rebuild_param_vector(person)


def apply_wrist_delta(person: dict, wrist_body_index: int, delta_left: np.ndarray) -> None:
    body = np.asarray(person["body_pose"], dtype=np.float64).reshape(21, 3)
    current = rotvec_to_matrix(body[wrist_body_index])
    updated = matrix_to_rotvec(delta_left @ current).astype(np.float32)
    set_body_pose_joint(person, wrist_body_index, updated)


def fit_hand_wrist_delta(
    pred_joints: np.ndarray,
    target_joints: np.ndarray,
    wrist: int,
    hand_joints: np.ndarray,
) -> np.ndarray:
    source = pred_joints[hand_joints] - pred_joints[wrist]
    target = target_joints[hand_joints] - target_joints[wrist]
    row_rotation = rigid_rotation_row(source, target)
    return row_rotation.T


def replace_fingers(person: dict, gt_param: dict, side: str) -> None:
    if side == "left":
        person["left_hand_pose"] = np.asarray(gt_param["lhand_pose"], dtype=np.float32).reshape(-1)
    elif side == "right":
        person["right_hand_pose"] = np.asarray(gt_param["rhand_pose"], dtype=np.float32).reshape(-1)
    else:
        raise ValueError(side)
    rebuild_param_vector(person)


def replace_wrist(person: dict, gt_param: dict, side: str) -> None:
    gt_body = np.asarray(gt_param["body_pose"], dtype=np.float32).reshape(21, 3)
    if side == "left":
        set_body_pose_joint(person, LEFT_WRIST_BODY_INDEX, gt_body[LEFT_WRIST_BODY_INDEX])
    elif side == "right":
        set_body_pose_joint(person, RIGHT_WRIST_BODY_INDEX, gt_body[RIGHT_WRIST_BODY_INDEX])
    else:
        raise ValueError(side)


def replace_arm_chain(person: dict, gt_param: dict) -> None:
    body = np.asarray(person["body_pose"], dtype=np.float32).reshape(21, 3).copy()
    gt_body = np.asarray(gt_param["body_pose"], dtype=np.float32).reshape(21, 3)
    for index in ARM_CHAIN_BODY_INDICES:
        body[index] = gt_body[index]
    person["body_pose"] = body.reshape(-1)
    rebuild_param_vector(person)


def collect_items(args: argparse.Namespace) -> tuple[dict[str, list[dict]], list[str], dict]:
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
    metadata = {}
    for sequence in sequences:
        try:
            items, sequence_metadata = build_sequence_items(
                sequence, args, video_dir, pred_root, image_index, ffprobe, warnings
            )
        except Exception as exc:
            warnings.append(f"{sequence}: skipped setup error: {exc}")
            continue
        if args.max_frames_per_sequence > 0:
            items = items[: args.max_frames_per_sequence]
        sequence_data[sequence] = items
        metadata[sequence] = sequence_metadata
    return sequence_data, warnings, metadata


def load_rows(
    args: argparse.Namespace,
    sequence_data: dict[str, list[dict]],
    ground_truth: dict[int, dict],
) -> tuple[list[dict], list[dict]]:
    rows = []
    skipped = []
    for sequence, items in sequence_data.items():
        for item in items:
            if item["ground_truth_id"] not in ground_truth:
                skipped.append({**item, "reason": "missing_gt"})
                continue
            try:
                pred = one_person(item["file"])
            except Exception as exc:
                if args.strict_pred:
                    raise
                skipped.append({**item, "reason": str(exc)})
                continue
            rows.append(
                {
                    **item,
                    "person": pred,
                    "target_param": ground_truth[item["ground_truth_id"]]["smplx_param"],
                }
            )
    return rows, skipped


def forward_eval(rows: list[dict], model: NumpySMPLX, stage: str, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    frame_rows = []
    grouped: dict[str, list[dict]] = {}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        pred_vertices, pred_joints = model.forward([row["person"] for row in batch], predicted=True)
        target_vertices, target_joints = model.forward([row["target_param"] for row in batch], predicted=False)
        for index, row in enumerate(batch):
            metrics = geometry_metrics(
                pred_vertices[index],
                pred_joints[index],
                target_vertices[index],
                target_joints[index],
                model,
            )
            output = {
                "stage": stage,
                "sequence": row["sequence"],
                "prediction_frame": row["prediction_frame"],
                "annotation_frame": row["annotation_frame"],
                "ground_truth_id": row["ground_truth_id"],
                **metrics,
            }
            frame_rows.append(output)
            grouped.setdefault(row["sequence"], []).append(output)
    summary = []
    for sequence in sorted(grouped, key=sequence_sort_key):
        row = summarize(grouped[sequence], sequence)
        row["stage"] = stage
        summary.append(row)
    all_summary = summarize(frame_rows, "ALL")
    all_summary["stage"] = stage
    summary.append(all_summary)
    return frame_rows, summary


def with_wrist_oracle(rows: list[dict], model: NumpySMPLX, args: argparse.Namespace) -> list[dict]:
    output = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        pred_vertices, pred_joints = model.forward([row["person"] for row in batch], predicted=True)
        target_vertices, target_joints = model.forward([row["target_param"] for row in batch], predicted=False)
        for index, row in enumerate(batch):
            person = copy_person(row["person"])
            left_delta = fit_hand_wrist_delta(
                pred_joints[index], target_joints[index], LEFT_WRIST_SMPLX, LEFT_HAND_JOINTS
            )
            right_delta = fit_hand_wrist_delta(
                pred_joints[index], target_joints[index], RIGHT_WRIST_SMPLX, RIGHT_HAND_JOINTS
            )
            apply_wrist_delta(person, LEFT_WRIST_BODY_INDEX, left_delta)
            apply_wrist_delta(person, RIGHT_WRIST_BODY_INDEX, right_delta)
            output.append({**row, "person": person})
    return output


def with_gt_finger_oracle(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        person = copy_person(row["person"])
        replace_fingers(person, row["target_param"], "left")
        replace_fingers(person, row["target_param"], "right")
        output.append({**row, "person": person})
    return output


def with_gt_wrist_oracle(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        person = copy_person(row["person"])
        replace_wrist(person, row["target_param"], "left")
        replace_wrist(person, row["target_param"], "right")
        output.append({**row, "person": person})
    return output


def with_gt_arm_oracle(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        person = copy_person(row["person"])
        replace_arm_chain(person, row["target_param"])
        output.append({**row, "person": person})
    return output


def with_gt_arm_finger_oracle(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        person = copy_person(row["person"])
        replace_arm_chain(person, row["target_param"])
        replace_fingers(person, row["target_param"], "left")
        replace_fingers(person, row["target_param"], "right")
        output.append({**row, "person": person})
    return output


def with_gt_wrist_finger_oracle(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        person = copy_person(row["person"])
        replace_wrist(person, row["target_param"], "left")
        replace_wrist(person, row["target_param"], "right")
        replace_fingers(person, row["target_param"], "left")
        replace_fingers(person, row["target_param"], "right")
        output.append({**row, "person": person})
    return output


def main() -> None:
    args = parse_args()
    args.strict_frame_count = False
    args.skip_bad_frames = True
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotation_dir = (args.annotation_dir or args.root / "datasets" / "annotations" / "SignLanguage").resolve()
    sequence_data, warnings, metadata = collect_items(args)
    wanted_ids = {item["ground_truth_id"] for items in sequence_data.values() for item in items}
    gt_records, missing = load_ground_truth_records(
        annotation_dir / "smplx_annotation.json",
        wanted_ids,
        allow_missing=not args.strict_gt,
    )
    if missing:
        warnings.append(f"missing GT records skipped: {len(missing)}")
    rows, skipped = load_rows(args, sequence_data, gt_records)
    if not rows:
        raise ValueError("No usable pred/GT pairs")

    model = NumpySMPLX(args.model_path)
    stages = {
        "before": rows,
        "wrist_rigid_oracle": with_wrist_oracle(rows, model, args),
        "gt_wrist_oracle": with_gt_wrist_oracle(rows),
        "gt_arm_oracle": with_gt_arm_oracle(rows),
        "gt_fingers_oracle": with_gt_finger_oracle(rows),
        "gt_wrist_fingers_oracle": with_gt_wrist_finger_oracle(rows),
        "gt_arm_fingers_oracle": with_gt_arm_finger_oracle(rows),
    }

    all_frame_rows = []
    all_summary_rows = []
    for name, stage_rows in stages.items():
        frame_rows, summary_rows = forward_eval(stage_rows, model, name, args)
        write_csv(args.output_dir / f"{name}_frame_metrics.csv", frame_rows)
        write_csv(args.output_dir / f"{name}_summary.csv", summary_rows)
        all_frame_rows.extend(frame_rows)
        all_summary_rows.extend(summary_rows)
    write_csv(args.output_dir / "hand_oracle_summary.csv", all_summary_rows)
    write_csv(args.output_dir / "skipped_frames.csv", skipped)

    compact_metrics = [
        "hand_mpjpe_mm_mean",
        "hand_pa_mpjpe_mm_mean",
        "hands_wrist_mpjpe_mm_mean",
        "hands_pa_mpjpe_mm_mean",
        "hands_wrist_mpvpe_mm_mean",
        "hands_pa_mpvpe_mm_mean",
        "mpjpe_mm_mean",
        "mpvpe_mm_mean",
    ]
    compact = []
    for row in all_summary_rows:
        if row["sequence"] == "ALL":
            compact.append({"stage": row["stage"], "frames": row["frames"], **{m: row[m] for m in compact_metrics}})
    report = {
        "mode": "gt_hand_oracle_diagnostic",
        "warning": "Uses GT, diagnostic upper-bound only.",
        "paths": {
            "pred_root": str((args.pred_root or args.root / "outputs_fusion_signlanguage").resolve()),
            "params_subdir": args.params_subdir,
            "model_path": str(args.model_path.resolve()),
            "output_dir": str(args.output_dir.resolve()),
        },
        "frames": {
            "usable": len(rows),
            "skipped": len(skipped),
        },
        "warnings": warnings,
        "metadata": metadata,
        "all_summary": compact,
    }
    (args.output_dir / "hand_oracle_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("stage,frames," + ",".join(compact_metrics))
    for row in compact:
        print(",".join([row["stage"], str(row["frames"]), *[f"{float(row[m]):.4f}" for m in compact_metrics]]))
    print(f"Wrote hand oracle diagnostic to {args.output_dir}")


if __name__ == "__main__":
    main()
