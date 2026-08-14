#!/usr/bin/env python3
"""Build a GT-supervised hand correction dataset for SignLanguage predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_signlanguage_geometry import (
    build_sequence_items,
    discover_sequences,
    find_ffprobe,
    load_ground_truth_records,
    load_image_index,
    normalize_sequence_name,
    one_person,
    sequence_sort_key,
)
from oracle_optimize_signlanguage_global import matrix_to_rotvec, rotvec_to_matrix
from signlanguage_global_correction import base_feature


LEFT_WRIST_BODY_INDEX = 20 - 1
RIGHT_WRIST_BODY_INDEX = 21 - 1


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--params-subdir", default="fused_params")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--feature-preset", choices=("minimal", "upper_global", "full"), default="upper_global")
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-sequences", nargs="*", default=None)
    parser.add_argument("--test-sequences", nargs="*", default=None)
    parser.add_argument("--target", choices=("wrist", "fingers", "wrist_fingers"), default="wrist_fingers")
    parser.add_argument("--max-label-rot-deg", type=float, default=120.0)
    return parser.parse_args()


def rotation_delta(pred_rotvec: np.ndarray, target_rotvec: np.ndarray) -> np.ndarray:
    pred = rotvec_to_matrix(np.asarray(pred_rotvec, dtype=np.float64).reshape(3))
    target = rotvec_to_matrix(np.asarray(target_rotvec, dtype=np.float64).reshape(3))
    return matrix_to_rotvec(target @ pred.T).astype(np.float32)


def hand_target(person: dict, gt_param: dict, target_mode: str) -> np.ndarray:
    values = []
    pred_body = np.asarray(person["body_pose"], dtype=np.float64).reshape(21, 3)
    gt_body = np.asarray(gt_param["body_pose"], dtype=np.float64).reshape(21, 3)
    if target_mode in ("wrist", "wrist_fingers"):
        values.append(rotation_delta(pred_body[LEFT_WRIST_BODY_INDEX], gt_body[LEFT_WRIST_BODY_INDEX]))
        values.append(rotation_delta(pred_body[RIGHT_WRIST_BODY_INDEX], gt_body[RIGHT_WRIST_BODY_INDEX]))
    if target_mode in ("fingers", "wrist_fingers"):
        pred_left = np.asarray(person["left_hand_pose"], dtype=np.float64).reshape(15, 3)
        pred_right = np.asarray(person["right_hand_pose"], dtype=np.float64).reshape(15, 3)
        gt_left = np.asarray(gt_param["lhand_pose"], dtype=np.float64).reshape(15, 3)
        gt_right = np.asarray(gt_param["rhand_pose"], dtype=np.float64).reshape(15, 3)
        values.extend(rotation_delta(pred_left[i], gt_left[i]) for i in range(15))
        values.extend(rotation_delta(pred_right[i], gt_right[i]) for i in range(15))
    return np.concatenate(values, axis=0).astype(np.float32)


def split_sequences(args: argparse.Namespace, sequences: list[str]) -> dict[str, str]:
    if args.val_sequences is not None or args.test_sequences is not None:
        val = {normalize_sequence_name(x) for x in (args.val_sequences or [])}
        test = {normalize_sequence_name(x) for x in (args.test_sequences or [])}
        overlap = val & test
        if overlap:
            raise ValueError(f"Sequences cannot be both val and test: {sorted(overlap)}")
        return {seq: "test" if seq in test else "val" if seq in val else "train" for seq in sequences}
    count = len(sequences)
    test_count = max(1, int(round(count * args.test_ratio))) if count >= 3 else 0
    val_count = max(1, int(round(count * args.val_ratio))) if count >= 3 else 0
    if val_count + test_count >= count:
        test_count = max(0, min(test_count, count - 2))
        val_count = max(0, min(val_count, count - test_count - 1))
    train_end = count - val_count - test_count
    val_end = count - test_count
    return {
        sequence: "train" if idx < train_end else "val" if idx < val_end else "test"
        for idx, sequence in enumerate(sequences)
    }


def temporal_stack(rows: list[dict], radius: int) -> list[np.ndarray]:
    if radius <= 0:
        return [row["base_feature"] for row in rows]
    features = [row["base_feature"] for row in rows]
    output = []
    last = len(features) - 1
    for idx in range(len(features)):
        output.append(np.concatenate([features[min(max(idx + off, 0), last)] for off in range(-radius, radius + 1)]))
    return output


def main() -> None:
    args = parse_args()
    args.strict_frame_count = False
    args.skip_bad_frames = True
    root = args.root.resolve()
    video_dir = (args.video_dir or root / "datasets" / "videos" / "SignLanguage").resolve()
    annotation_dir = (args.annotation_dir or root / "datasets" / "annotations" / "SignLanguage").resolve()
    pred_root = (args.pred_root or root / "outputs_fusion_signlanguage").resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    image_index = load_image_index(annotation_dir / "keypoint_annotation.json")
    ffprobe = find_ffprobe()
    warnings = []
    if args.sequences == ["all"]:
        sequences = discover_sequences(pred_root, args.params_subdir)
    else:
        sequences = sorted([normalize_sequence_name(x) for x in args.sequences], key=sequence_sort_key)
    splits = split_sequences(args, sequences)

    sequence_items = {}
    for sequence in sequences:
        try:
            items, _ = build_sequence_items(sequence, args, video_dir, pred_root, image_index, ffprobe, warnings)
        except Exception as exc:
            warnings.append(f"{sequence}: skipped setup error: {exc}")
            continue
        sequence_items[sequence] = items

    wanted_ids = {item["ground_truth_id"] for items in sequence_items.values() for item in items}
    gt_records, missing = load_ground_truth_records(annotation_dir / "smplx_annotation.json", wanted_ids, allow_missing=True)
    if missing:
        warnings.append(f"missing GT records skipped: {len(missing)}")

    max_norm = np.deg2rad(args.max_label_rot_deg) if args.max_label_rot_deg > 0 else None
    all_rows = []
    skipped = []
    for sequence in sequences:
        rows = []
        for item in sequence_items.get(sequence, []):
            if item["ground_truth_id"] not in gt_records:
                skipped.append({**item, "reason": "missing_gt"})
                continue
            try:
                person = one_person(item["file"])
            except Exception as exc:
                skipped.append({**item, "reason": str(exc)})
                continue
            target = hand_target(person, gt_records[item["ground_truth_id"]]["smplx_param"], args.target)
            if max_norm is not None and np.max(np.linalg.norm(target.reshape(-1, 3), axis=1)) > max_norm:
                skipped.append({**item, "reason": "label_rotation_outlier"})
                continue
            rows.append(
                {
                    "sequence": sequence,
                    "frame": item["prediction_frame"],
                    "file": str(item["file"]),
                    "base_feature": base_feature(person, args.feature_preset),
                    "target": target,
                    "split": splits[sequence],
                }
            )
        rows.sort(key=lambda row: row["frame"])
        for row, feature in zip(rows, temporal_stack(rows, args.temporal_radius)):
            row["feature"] = feature.astype(np.float32)
            del row["base_feature"]
            all_rows.append(row)

    if not all_rows:
        raise ValueError("No usable training rows")
    x = np.stack([row["feature"] for row in all_rows]).astype(np.float32)
    y = np.stack([row["target"] for row in all_rows]).astype(np.float32)
    split_name = np.asarray([row["split"] for row in all_rows])
    split = np.asarray([{"train": 0, "val": 1, "test": 2}[name] for name in split_name], dtype=np.int64)
    np.savez_compressed(
        args.output,
        x=x,
        y=y,
        sequence=np.asarray([row["sequence"] for row in all_rows]),
        frame=np.asarray([row["frame"] for row in all_rows], dtype=np.int64),
        path=np.asarray([row["file"] for row in all_rows]),
        split=split,
        split_name=split_name,
        feature_preset=np.asarray(args.feature_preset),
        temporal_radius=np.asarray(args.temporal_radius, dtype=np.int64),
        target_mode=np.asarray(args.target),
    )
    report = {
        "output": str(args.output.resolve()),
        "pred_root": str(pred_root),
        "params_subdir": args.params_subdir,
        "feature_preset": args.feature_preset,
        "temporal_radius": args.temporal_radius,
        "target": args.target,
        "feature_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "frames": int(len(all_rows)),
        "split_counts": {name: int(np.sum(split_name == name)) for name in ("train", "val", "test")},
        "warnings": warnings,
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:20],
    }
    safe_report = json_safe(report)
    args.output.with_suffix(".json").write_text(json.dumps(safe_report, indent=2), encoding="utf-8")
    print(json.dumps(safe_report, indent=2))


if __name__ == "__main__":
    main()
