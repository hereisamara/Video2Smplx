#!/usr/bin/env python3
"""Build a GT-supervised upper-body plus translation correction dataset."""

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
from signlanguage_global_correction import UPPER_BODY_BODY_POSE_INDICES, base_feature


TARGET_JOINTS = {
    "arms": (16, 17, 18, 19, 20, 21),
    "torso_arms": (3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21),
    "upper_body": (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
}

JOINT_WEIGHTS = {
    3: 0.50,
    6: 0.75,
    9: 1.00,
    12: 1.00,
    13: 1.25,
    14: 1.25,
    15: 0.75,
    16: 2.00,
    17: 2.00,
    18: 2.25,
    19: 2.25,
    20: 2.50,
    21: 2.50,
}


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
    parser.add_argument("--params-subdir", default="fused_params_model_global_corrected")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--feature-preset", choices=("minimal", "upper_global", "full"), default="upper_global")
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-sequences", nargs="*", default=None)
    parser.add_argument("--test-sequences", nargs="*", default=None)
    parser.add_argument("--target", choices=tuple(TARGET_JOINTS), default="torso_arms")
    parser.add_argument("--max-label-rot-deg", type=float, default=120.0)
    parser.add_argument("--max-label-transl-m", type=float, default=1.5)
    parser.add_argument("--transl-weight", type=float, default=4.0)
    return parser.parse_args()


def rotation_delta(pred_rotvec: np.ndarray, target_rotvec: np.ndarray) -> np.ndarray:
    pred = rotvec_to_matrix(np.asarray(pred_rotvec, dtype=np.float64).reshape(3))
    target = rotvec_to_matrix(np.asarray(target_rotvec, dtype=np.float64).reshape(3))
    return matrix_to_rotvec(target @ pred.T).astype(np.float32)


def target_body_indices(target_mode: str) -> tuple[int, ...]:
    return tuple(joint - 1 for joint in TARGET_JOINTS[target_mode])


def target_weights(target_mode: str, transl_weight: float) -> np.ndarray:
    weights = np.asarray([JOINT_WEIGHTS[joint] for joint in TARGET_JOINTS[target_mode]], dtype=np.float32)
    weights = weights / float(np.mean(weights))
    rot_weights = np.repeat(weights, 3).astype(np.float32)
    transl_weights = np.full(3, float(transl_weight), dtype=np.float32)
    return np.concatenate([rot_weights, transl_weights], axis=0)


def correction_target(person: dict, gt_param: dict, target_mode: str) -> np.ndarray:
    pred_body = np.asarray(person["body_pose"], dtype=np.float64).reshape(21, 3)
    gt_body = np.asarray(gt_param["body_pose"], dtype=np.float64).reshape(21, 3)
    rot_values = [rotation_delta(pred_body[index], gt_body[index]) for index in target_body_indices(target_mode)]
    pred_transl = np.asarray(person["transl"], dtype=np.float64).reshape(3)
    gt_transl_key = "transl" if "transl" in gt_param else "trans"
    gt_transl = np.asarray(gt_param[gt_transl_key], dtype=np.float64).reshape(3)
    transl_delta = (gt_transl - pred_transl).astype(np.float32)
    return np.concatenate([*rot_values, transl_delta], axis=0).astype(np.float32)


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

    max_rot_norm = np.deg2rad(args.max_label_rot_deg) if args.max_label_rot_deg > 0 else None
    max_transl_norm = float(args.max_label_transl_m) if args.max_label_transl_m > 0 else None
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
            target = correction_target(person, gt_records[item["ground_truth_id"]]["smplx_param"], args.target)
            rot_target = target[:-3].reshape(-1, 3)
            transl_target = target[-3:]
            if max_rot_norm is not None and np.max(np.linalg.norm(rot_target, axis=1)) > max_rot_norm:
                skipped.append({**item, "reason": "label_rotation_outlier"})
                continue
            if max_transl_norm is not None and float(np.linalg.norm(transl_target)) > max_transl_norm:
                skipped.append({**item, "reason": "label_translation_outlier"})
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
    weights = target_weights(args.target, args.transl_weight)
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
        target_kind=np.asarray("upperbody_transl"),
        target_body_indices=np.asarray(target_body_indices(args.target), dtype=np.int64),
        target_smplx_joints=np.asarray(TARGET_JOINTS[args.target], dtype=np.int64),
        target_weight=weights,
        transl_weight=np.asarray(args.transl_weight, dtype=np.float32),
        default_feature_upper_body_indices=np.asarray(UPPER_BODY_BODY_POSE_INDICES, dtype=np.int64),
    )
    report = {
        "output": str(args.output.resolve()),
        "pred_root": str(pred_root),
        "params_subdir": args.params_subdir,
        "feature_preset": args.feature_preset,
        "temporal_radius": args.temporal_radius,
        "target": args.target,
        "target_kind": "upperbody_transl",
        "target_smplx_joints": list(TARGET_JOINTS[args.target]),
        "target_body_indices": list(target_body_indices(args.target)),
        "target_weight": weights.tolist(),
        "transl_weight": args.transl_weight,
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
