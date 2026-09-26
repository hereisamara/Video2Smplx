#!/usr/bin/env python3
"""Build a 2D-guided SignLanguage upper-body correction dataset.

Inputs are current SMPL-X predictions plus YOLO COCO-17 2D keypoints.  Labels
are GT-relative rotation deltas for global orientation and the arm chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.evaluation.evaluate_signlanguage_geometry import (
    NumpySMPLX,
    build_sequence_items,
    discover_sequences,
    find_ffprobe,
    load_ground_truth_records,
    load_image_index,
    normalize_sequence_name,
    one_person,
    sequence_sort_key,
)
from video2smplx.postprocessing.signlanguage import base_feature
from video2smplx.postprocessing.rotation import matrix_to_rotvec, rotvec_to_matrix


COCO_IDS = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
SMPLX_IDS = np.asarray([15, 16, 17, 18, 19, 20, 21, 1, 2], dtype=np.int64)
TARGET_SMPLX_JOINTS = (13, 14, 16, 17, 18, 19, 20, 21)
TARGET_BODY_INDICES = tuple(joint - 1 for joint in TARGET_SMPLX_JOINTS)
TARGET_NAMES = (
    "global_orient",
    "left_collar",
    "right_collar",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
TARGET_WEIGHTS = np.asarray([3.0, 1.2, 1.2, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0], dtype=np.float32)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--params-subdir", default="fused_params_model_global_hand_corrected")
    parser.add_argument("--yolo-keypoints", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--feature-preset", choices=("minimal", "upper_global", "full"), default="upper_global")
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-sequences", nargs="*", default=None)
    parser.add_argument("--test-sequences", nargs="*", default=None)
    parser.add_argument("--min-keypoint-conf", type=float, default=0.15)
    parser.add_argument("--max-label-rot-deg", type=float, default=90.0)
    return parser.parse_args()


def rotation_delta(pred_rotvec: np.ndarray, target_rotvec: np.ndarray) -> np.ndarray:
    pred = rotvec_to_matrix(np.asarray(pred_rotvec, dtype=np.float64).reshape(3))
    target = rotvec_to_matrix(np.asarray(target_rotvec, dtype=np.float64).reshape(3))
    return matrix_to_rotvec(target @ pred.T).astype(np.float32)


def correction_target(person: dict, gt_param: dict) -> np.ndarray:
    values = [rotation_delta(person["global_orient"], gt_param["root_pose"])]
    pred_body = np.asarray(person["body_pose"], dtype=np.float64).reshape(21, 3)
    gt_body = np.asarray(gt_param["body_pose"], dtype=np.float64).reshape(21, 3)
    values.extend(rotation_delta(pred_body[index], gt_body[index]) for index in TARGET_BODY_INDICES)
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
    return {seq: "train" if i < train_end else "val" if i < val_end else "test" for i, seq in enumerate(sequences)}


def load_yolo(path: Path) -> dict[str, dict[int, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for sequence, video in data.get("videos", {}).items():
        output[sequence] = {int(frame["frame"]): frame for frame in video.get("frames", [])}
    return output


def get_yolo_frame(yolo: dict[str, dict[int, dict]], sequence: str, frame: int) -> dict | None:
    frames = yolo.get(sequence)
    if not frames:
        return None
    if frame in frames:
        return frames[frame]
    # Be tolerant if a detector export uses zero-based frame numbers.
    if frame - 1 in frames:
        return frames[frame - 1]
    if frame + 1 in frames:
        return frames[frame + 1]
    return None


def normalize_yolo_frame(frame: dict, min_conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(float(frame.get("image_width", 1)), 1.0)
    height = max(float(frame.get("image_height", 1)), 1.0)
    xy = np.asarray(frame["keypoints_xy"], dtype=np.float32).reshape(17, 2)[COCO_IDS]
    conf = np.asarray(frame["keypoints_conf"], dtype=np.float32).reshape(17)[COCO_IDS]
    xy_norm = np.stack([(xy[:, 0] / width - 0.5), (xy[:, 1] / height - 0.5)], axis=1)
    valid = (conf >= min_conf).astype(np.float32)
    return xy_norm, conf * valid, np.asarray([width, height], dtype=np.float32)


def weighted_similarity_2d(source: np.ndarray, target: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    if float(weight.sum()) < 1e-6:
        projected = np.zeros_like(target, dtype=np.float32)
        return projected, np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weight = weight / weight.sum()
    src_mean = (source * weight[:, None]).sum(axis=0)
    tgt_mean = (target * weight[:, None]).sum(axis=0)
    src = source - src_mean
    tgt = target - tgt_mean
    cov = src.T @ (tgt * weight[:, None])
    u, singular, vt = np.linalg.svd(cov)
    sign = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ np.diag([1.0, sign]) @ vt
    denom = float((weight[:, None] * src * src).sum())
    scale = float(np.sum(singular * np.asarray([1.0, sign])) / denom) if denom > 1e-9 else 1.0
    projected = scale * src @ rotation + tgt_mean
    angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    geom = np.asarray([scale, np.cos(angle), np.sin(angle), tgt_mean[0] - src_mean[0], tgt_mean[1] - src_mean[1]], dtype=np.float32)
    return projected.astype(np.float32), geom


def guided_feature(person: dict, pred_joints: np.ndarray, yolo_frame: dict, feature_preset: str, min_conf: float) -> np.ndarray:
    yolo_xy, yolo_conf, image_size = normalize_yolo_frame(yolo_frame, min_conf)
    pred = pred_joints[SMPLX_IDS].astype(np.float32)
    root = pred_joints[0].astype(np.float32)
    pred_rooted = pred - root[None]
    body_scale = float(np.linalg.norm(pred_joints[16] - pred_joints[17]))
    if body_scale < 1e-6:
        body_scale = 1.0
    pred_2d_source = pred_rooted[:, :2] / body_scale
    projected, geom = weighted_similarity_2d(pred_2d_source, yolo_xy, yolo_conf)
    residual = (yolo_xy - projected) * yolo_conf[:, None]
    conf_stats = np.asarray([yolo_conf.mean(), yolo_conf.min(), yolo_conf.max(), float((yolo_conf > 0).mean())], dtype=np.float32)
    return np.concatenate(
        [
            base_feature(person, feature_preset),
            yolo_xy.reshape(-1),
            yolo_conf.reshape(-1),
            pred_rooted.reshape(-1) / body_scale,
            projected.reshape(-1),
            residual.reshape(-1),
            geom,
            conf_stats,
            image_size / 1000.0,
        ],
        axis=0,
    ).astype(np.float32)


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
    pred_root = args.pred_root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    image_index = load_image_index(annotation_dir / "keypoint_annotation.json")
    ffprobe = find_ffprobe()
    yolo = load_yolo(args.yolo_keypoints)
    model = NumpySMPLX(args.model_path.resolve())
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
    all_rows = []
    skipped = []
    for sequence in sequences:
        rows = []
        items = sequence_items.get(sequence, [])
        for start in range(0, len(items), 64):
            batch = items[start : start + 64]
            persons = []
            kept = []
            for item in batch:
                if item["ground_truth_id"] not in gt_records:
                    skipped.append({**item, "reason": "missing_gt"})
                    continue
                yolo_frame = get_yolo_frame(yolo, sequence, item["prediction_frame"])
                if yolo_frame is None:
                    skipped.append({**item, "reason": "missing_yolo"})
                    continue
                try:
                    persons.append(one_person(item["file"]))
                    kept.append(item)
                except Exception as exc:
                    skipped.append({**item, "reason": str(exc)})
            if not kept:
                continue
            pred_joints = model.forward_joints(persons, predicted=True)
            for idx, item in enumerate(kept):
                gt = gt_records[item["ground_truth_id"]]["smplx_param"]
                target = correction_target(persons[idx], gt)
                if max_rot_norm is not None and np.max(np.linalg.norm(target.reshape(-1, 3), axis=1)) > max_rot_norm:
                    skipped.append({**item, "reason": "label_rotation_outlier"})
                    continue
                try:
                    feature = guided_feature(
                        persons[idx],
                        pred_joints[idx],
                        yolo_frame,
                        args.feature_preset,
                        args.min_keypoint_conf,
                    )
                except Exception as exc:
                    skipped.append({**item, "reason": f"feature_error: {exc}"})
                    continue
                rows.append({"sequence": sequence, "frame": item["prediction_frame"], "file": str(item["file"]), "base_feature": feature, "target": target, "split": splits[sequence]})
        rows.sort(key=lambda row: row["frame"])
        for row, feature in zip(rows, temporal_stack(rows, args.temporal_radius)):
            row["feature"] = feature.astype(np.float32)
            del row["base_feature"]
            all_rows.append(row)
        print(f"{sequence}: usable {len(rows)}", flush=True)

    if not all_rows:
        reason_counts: dict[str, int] = {}
        for item in skipped:
            reason = str(item.get("reason", "unknown"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print(json.dumps({"warnings": warnings[:20], "skipped_count": len(skipped), "reason_counts": reason_counts, "skipped_sample": skipped[:20]}, indent=2), flush=True)
        raise ValueError("No usable training rows")
    x = np.stack([row["feature"] for row in all_rows]).astype(np.float32)
    y = np.stack([row["target"] for row in all_rows]).astype(np.float32)
    split_name = np.asarray([row["split"] for row in all_rows])
    split = np.asarray([{"train": 0, "val": 1, "test": 2}[name] for name in split_name], dtype=np.int64)
    weights = np.repeat(TARGET_WEIGHTS / TARGET_WEIGHTS.mean(), 3).astype(np.float32)
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
        target_kind=np.asarray("2d_guided_upperbody"),
        target_names=np.asarray(TARGET_NAMES),
        target_body_indices=np.asarray(TARGET_BODY_INDICES, dtype=np.int64),
        target_smplx_joints=np.asarray(TARGET_SMPLX_JOINTS, dtype=np.int64),
        target_weight=weights,
        yolo_keypoints=str(args.yolo_keypoints.resolve()),
        min_keypoint_conf=np.asarray(args.min_keypoint_conf, dtype=np.float32),
    )
    report = {
        "output": str(args.output.resolve()),
        "pred_root": str(pred_root),
        "params_subdir": args.params_subdir,
        "yolo_keypoints": str(args.yolo_keypoints.resolve()),
        "feature_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "target_names": TARGET_NAMES,
        "target_smplx_joints": TARGET_SMPLX_JOINTS,
        "frames": int(len(all_rows)),
        "split_counts": {name: int(np.sum(split_name == name)) for name in ("train", "val", "test")},
        "warnings": warnings[:50],
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:20],
    }
    args.output.with_suffix(".json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
