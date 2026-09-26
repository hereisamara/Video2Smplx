#!/usr/bin/env python3
"""GT-free upper-body 2D global-orientation alignment for SignLanguage.

This postprocess uses 2D upper-body/hand/face keypoints from
keypoint_annotation.json as image evidence and updates only `global_orient` by
default. It does not read 3D GT SMPL-X annotations.

The optimization is weak-perspective: for each candidate global rotation, a
2D scale and offset are solved analytically against visible upper-body
keypoints, then a small coordinate search refines the rotation correction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import pickle
import shutil
from pathlib import Path

import numpy as np

from tools.evaluation.evaluate_signlanguage_geometry import (
    NumpySMPLX,
    build_sequence_items,
    discover_sequences,
    find_ffprobe,
    load_image_index,
    normalize_sequence_name,
    one_person,
    sequence_sort_key,
)


PARAM_VECTOR_KEYS = (
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "betas",
    "expression",
    "transl",
)

BODY_OBS = (
    (0, 15, 1.0),  # nose -> head
    (1, 23, 0.5),  # left eye
    (2, 24, 0.5),  # right eye
    (5, 16, 2.0),  # left shoulder
    (6, 17, 2.0),  # right shoulder
    (7, 18, 2.0),  # left elbow
    (8, 19, 2.0),  # right elbow
    (9, 20, 3.0),  # left wrist
    (10, 21, 3.0),  # right wrist
)

# COCO/WholeBody hand order: wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4.
# SMPL-X exposes 3 articulated joints per finger, so skip wrist and fingertips.
HAND_21_TO_SMPLX_15 = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--params-subdir", default="fused_params")
    parser.add_argument("--output-subdir", default="fused_params_upperbody_2d_global")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fit-scope", choices=("global", "sequence", "frame"), default="sequence")
    parser.add_argument("--max-rot-deg", type=float, default=18.0)
    parser.add_argument("--min-observations", type=int, default=6)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=5, help="Odd moving-average window for frame mode.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write-params", action="store_true")
    return parser.parse_args()


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    denom = 2.0 * math.sin(theta)
    if abs(denom) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return axis / denom * theta


def matrix_to_euler_xyz_deg(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy >= 1e-8:
        x = math.atan2(rotation[2, 1], rotation[2, 2])
        y = math.atan2(-rotation[2, 0], sy)
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x = math.atan2(-rotation[1, 2], rotation[1, 1])
        y = math.atan2(-rotation[2, 0], sy)
        z = 0.0
    return np.degrees(np.asarray([x, y, z], dtype=np.float64))


def rebuild_param_vector(person: dict) -> None:
    if all(key in person for key in PARAM_VECTOR_KEYS):
        person["smplx_param_vector"] = np.concatenate(
            [np.asarray(person[key]).reshape(-1) for key in PARAM_VECTOR_KEYS]
        ).astype(np.float32)


def apply_delta_orient(person: dict, delta: np.ndarray) -> None:
    current = rotvec_to_matrix(np.asarray(person["global_orient"], dtype=np.float64).reshape(3))
    person["global_orient"] = matrix_to_rotvec(delta @ current).astype(np.float32)
    rebuild_param_vector(person)


def write_person(path: Path, person: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([person], handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_keypoint_records(keypoint_path: Path, wanted_image_ids: set[int]) -> tuple[dict[int, dict], list[int]]:
    records: dict[int, dict] = {}
    pattern = b'"image_id"'
    decoder = json.JSONDecoder()
    with keypoint_path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            start = 0
            while len(records) < len(wanted_image_ids):
                pos = mapped.find(pattern, start)
                if pos < 0:
                    break
                object_start = mapped.rfind(b"{", 0, pos)
                if object_start < 0:
                    start = pos + len(pattern)
                    continue
                text = mapped[object_start : object_start + 200000].decode("ascii")
                try:
                    record, consumed = decoder.raw_decode(text)
                except json.JSONDecodeError:
                    start = pos + len(pattern)
                    continue
                image_id = int(record.get("image_id", -1))
                if image_id in wanted_image_ids:
                    records[image_id] = record
                start = object_start + consumed
        finally:
            mapped.close()
    missing = sorted(wanted_image_ids - records.keys())
    return records, missing


def add_body_observations(record: dict, observations: list[tuple[int, np.ndarray, float]], threshold: float) -> None:
    keypoints = np.asarray(record.get("keypoints", []), dtype=np.float64).reshape(-1, 3)
    if len(keypoints) < 11:
        return
    for coco_idx, smplx_idx, base_weight in BODY_OBS:
        x, y, score = keypoints[coco_idx]
        if score >= threshold:
            observations.append((smplx_idx, np.asarray([x, y], dtype=np.float64), base_weight * score))


def add_hand_observations(
    record: dict,
    observations: list[tuple[int, np.ndarray, float]],
    key: str,
    score_key: str,
    smplx_start: int,
    threshold: float,
) -> None:
    points = np.asarray(record.get(key, []), dtype=np.float64).reshape(-1, 3)
    scores = np.asarray(record.get(score_key, []), dtype=np.float64).reshape(-1)
    if len(points) < 21:
        return
    for local_idx, hand_idx in enumerate(HAND_21_TO_SMPLX_15):
        score = float(scores[hand_idx]) if len(scores) > hand_idx else float(points[hand_idx, 2])
        if score >= threshold:
            observations.append(
                (
                    smplx_start + local_idx,
                    points[hand_idx, :2].astype(np.float64),
                    2.5 * score,
                )
            )


def observations_from_record(record: dict, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[tuple[int, np.ndarray, float]] = []
    add_body_observations(record, observations, threshold)
    add_hand_observations(record, observations, "lefthand_kpts", "lefthand_kpts_score", 25, threshold)
    add_hand_observations(record, observations, "righthand_kpts", "righthand_kpts_score", 40, threshold)
    if not observations:
        return np.empty(0, dtype=np.int64), np.empty((0, 2)), np.empty(0)
    indices = np.asarray([item[0] for item in observations], dtype=np.int64)
    points = np.asarray([item[1] for item in observations], dtype=np.float64)
    weights = np.asarray([item[2] for item in observations], dtype=np.float64)
    return indices, points, weights


def weak_project(points3d: np.ndarray) -> np.ndarray:
    return np.column_stack([points3d[:, 0], -points3d[:, 1]])


def fit_scale_offset(source2d: np.ndarray, target2d: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    weights = weights / max(float(weights.sum()), 1e-12)
    source_mean = np.sum(source2d * weights[:, None], axis=0)
    target_mean = np.sum(target2d * weights[:, None], axis=0)
    source_centered = source2d - source_mean
    target_centered = target2d - target_mean
    denom = float(np.sum(weights * np.sum(source_centered**2, axis=1)))
    if denom < 1e-12:
        return 1.0, target_mean - source_mean
    scale = float(np.sum(weights * np.sum(source_centered * target_centered, axis=1)) / denom)
    offset = target_mean - scale * source_mean
    return scale, offset


def reprojection_loss(local_joints: np.ndarray, indices: np.ndarray, target2d: np.ndarray, weights: np.ndarray, delta: np.ndarray) -> tuple[float, float, np.ndarray]:
    transformed = (delta @ local_joints[indices].T).T
    source2d = weak_project(transformed)
    scale, offset = fit_scale_offset(source2d, target2d, weights)
    residual = scale * source2d + offset - target2d
    loss = float(np.sqrt(np.sum(weights * np.sum(residual**2, axis=1)) / max(float(weights.sum()), 1e-12)))
    return loss, scale, offset


def optimize_delta(local_joints: np.ndarray, indices: np.ndarray, target2d: np.ndarray, weights: np.ndarray, max_rot_deg: float) -> tuple[np.ndarray, dict]:
    limit = math.radians(max_rot_deg)
    delta_vec = np.zeros(3, dtype=np.float64)
    best_delta = rotvec_to_matrix(delta_vec)
    best_loss, best_scale, best_offset = reprojection_loss(local_joints, indices, target2d, weights, best_delta)
    step = min(math.radians(8.0), limit)
    while step > math.radians(0.15):
        improved = False
        for axis in range(3):
            for sign in (-1.0, 1.0):
                candidate = delta_vec.copy()
                candidate[axis] += sign * step
                if np.linalg.norm(candidate) > limit:
                    continue
                candidate_delta = rotvec_to_matrix(candidate)
                loss, scale, offset = reprojection_loss(local_joints, indices, target2d, weights, candidate_delta)
                if loss < best_loss:
                    delta_vec = candidate
                    best_delta = candidate_delta
                    best_loss = loss
                    best_scale = scale
                    best_offset = offset
                    improved = True
        if not improved:
            step *= 0.5
    return best_delta, {"loss": best_loss, "scale": best_scale, "offset": best_offset.tolist()}


def smooth_rotvecs(rotvecs: list[np.ndarray], window: int) -> list[np.ndarray]:
    if window <= 1:
        return rotvecs
    if window % 2 == 0:
        window += 1
    radius = window // 2
    arr = np.asarray(rotvecs, dtype=np.float64)
    output = []
    for index in range(len(arr)):
        start = max(0, index - radius)
        end = min(len(arr), index + radius + 1)
        output.append(np.mean(arr[start:end], axis=0))
    return output


def copy_non_param_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file() and not path.name.endswith("_params.pkl"):
            shutil.copy2(path, target_dir / path.name)


def main() -> None:
    args = parse_args()
    args.strict_frame_count = False
    root = args.root.resolve()
    video_dir = (args.video_dir or root / "datasets" / "videos" / "SignLanguage").resolve()
    annotation_dir = (args.annotation_dir or root / "datasets" / "annotations" / "SignLanguage").resolve()
    pred_root = (args.pred_root or root / "outputs_fusion_signlanguage").resolve()
    output_dir = (args.output_dir or pred_root / "evaluation" / f"upperbody_2d_global_{args.params_subdir}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_index = load_image_index(annotation_dir / "keypoint_annotation.json")
    if args.sequences == ["all"]:
        sequences = discover_sequences(pred_root, args.params_subdir)
    else:
        sequences = sorted([normalize_sequence_name(x) for x in args.sequences], key=sequence_sort_key)
    sequence_data = {}
    warnings = []
    for sequence in sequences:
        try:
            items, _ = build_sequence_items(sequence, args, video_dir, pred_root, image_index, find_ffprobe(), warnings)
        except Exception as exc:
            warnings.append(f"{sequence}: skipped setup error: {exc}")
            continue
        sequence_data[sequence] = items

    wanted_ids = {item["ground_truth_id"] for items in sequence_data.values() for item in items}
    keypoint_records, missing_keypoints = load_keypoint_records(annotation_dir / "keypoint_annotation.json", wanted_ids)
    if missing_keypoints:
        warnings.append(f"Missing {len(missing_keypoints)} 2D keypoint records; first IDs: {missing_keypoints[:10]}")

    model = NumpySMPLX(args.model_path.resolve())
    fit_rows = []
    skipped = []
    for sequence, items in sequence_data.items():
        sequence_rows = []
        for start in range(0, len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            kept = []
            params = []
            observations = []
            for item in batch:
                record = keypoint_records.get(item["ground_truth_id"])
                if record is None:
                    skipped.append({**item, "reason": "missing_2d_keypoints"})
                    continue
                indices, points2d, weights = observations_from_record(record, args.confidence_threshold)
                if len(indices) < args.min_observations:
                    skipped.append({**item, "reason": f"too_few_observations:{len(indices)}"})
                    continue
                try:
                    person = one_person(item["file"])
                except Exception as exc:
                    skipped.append({**item, "reason": str(exc)})
                    continue
                kept.append(item)
                params.append(person)
                observations.append((indices, points2d, weights))
            if not kept:
                continue
            _, joints_batch = model.forward(params, predicted=True)
            for item, person, joints, obs in zip(kept, params, joints_batch, observations):
                indices, points2d, weights = obs
                local_joints = joints - np.asarray(person["transl"], dtype=np.float64).reshape(3)
                delta, fit = optimize_delta(local_joints, indices, points2d, weights, args.max_rot_deg)
                row = {
                    **item,
                    "person": person,
                    "delta": delta,
                    "delta_rotvec": matrix_to_rotvec(delta),
                    "initial_loss_px": reprojection_loss(local_joints, indices, points2d, weights, np.eye(3))[0],
                    "final_loss_px": fit["loss"],
                    "observations": len(indices),
                    "scale": fit["scale"],
                }
                sequence_rows.append(row)
        fit_rows.extend(sequence_rows)

    if args.fit_scope == "global":
        global_rotvec = np.median(np.asarray([row["delta_rotvec"] for row in fit_rows]), axis=0)
        for row in fit_rows:
            row["applied_delta"] = rotvec_to_matrix(global_rotvec)
    elif args.fit_scope == "sequence":
        for sequence in sequences:
            rows = [row for row in fit_rows if row["sequence"] == sequence]
            if not rows:
                continue
            sequence_rotvec = np.median(np.asarray([row["delta_rotvec"] for row in rows]), axis=0)
            for row in rows:
                row["applied_delta"] = rotvec_to_matrix(sequence_rotvec)
    else:
        for sequence in sequences:
            rows = [row for row in fit_rows if row["sequence"] == sequence]
            rows.sort(key=lambda row: row["prediction_frame"])
            smoothed = smooth_rotvecs([row["delta_rotvec"] for row in rows], args.smooth_window)
            for row, rotvec in zip(rows, smoothed):
                row["applied_delta"] = rotvec_to_matrix(rotvec)

    correction_rows = []
    copied_dirs = set()
    for row in fit_rows:
        person = {key: value.copy() if hasattr(value, "copy") else value for key, value in row["person"].items()}
        apply_delta_orient(person, row["applied_delta"])
        source_dir = Path(row["file"]).parent
        target_dir = source_dir.parent / args.output_subdir
        if not args.no_write_params:
            write_person(target_dir / Path(row["file"]).name, person)
            if target_dir not in copied_dirs:
                copy_non_param_files(source_dir, target_dir)
                copied_dirs.add(target_dir)
        delta_rotvec = matrix_to_rotvec(row["applied_delta"])
        delta_euler = matrix_to_euler_xyz_deg(row["applied_delta"])
        correction_rows.append(
            {
                "sequence": row["sequence"],
                "prediction_frame": row["prediction_frame"],
                "ground_truth_id": row["ground_truth_id"],
                "observations": row["observations"],
                "initial_loss_px": row["initial_loss_px"],
                "fit_loss_px": row["final_loss_px"],
                "delta_rotvec_x": delta_rotvec[0],
                "delta_rotvec_y": delta_rotvec[1],
                "delta_rotvec_z": delta_rotvec[2],
                "delta_euler_x_deg": delta_euler[0],
                "delta_euler_y_deg": delta_euler[1],
                "delta_euler_z_deg": delta_euler[2],
            }
        )

    write_csv(output_dir / "upperbody_2d_global_corrections.csv", correction_rows)
    write_csv(output_dir / "skipped_frames.csv", skipped)
    summary = {
        "mode": "upperbody_2d_global_align",
        "uses_3d_gt": False,
        "fit_scope": args.fit_scope,
        "params_subdir": args.params_subdir,
        "output_subdir": args.output_subdir,
        "frames_fit": len(fit_rows),
        "skipped": len(skipped),
        "mean_initial_loss_px": float(np.mean([row["initial_loss_px"] for row in fit_rows])) if fit_rows else None,
        "mean_fit_loss_px": float(np.mean([row["final_loss_px"] for row in fit_rows])) if fit_rows else None,
        "warnings": warnings,
    }
    (output_dir / "upperbody_2d_global_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not args.no_write_params:
        print(f"Wrote corrected PKLs under each sequence: {args.output_subdir}")


if __name__ == "__main__":
    main()
