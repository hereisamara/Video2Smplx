#!/usr/bin/env python3
"""Oracle global-alignment diagnostic for SignLanguage SMPL-X predictions.

This uses paired GT SMPL-X to estimate the best correction when changing only
prediction `global_orient` and/or `transl`. It is intended to measure the upper
bound of a no-retraining global-alignment postprocess. Do not treat it as a
deployable method, because it reads GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
from pathlib import Path

import numpy as np

from evaluate_signlanguage_geometry import (
    METRICS,
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
    write_csv,
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


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    if abs(math.pi - theta) < 1e-5:
        axis = np.sqrt(np.maximum(np.diag(rotation) + 1.0, 0.0) / 2.0)
        axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
        axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
        axis[2] = math.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
        norm = np.linalg.norm(axis)
        return theta * axis / norm if norm > 1e-12 else np.array([theta, 0.0, 0.0])
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(theta))
    return axis * theta


def matrix_to_euler_xyz_deg(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-8
    if not singular:
        x = math.atan2(rotation[2, 1], rotation[2, 2])
        y = math.atan2(-rotation[2, 0], sy)
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x = math.atan2(-rotation[1, 2], rotation[1, 1])
        y = math.atan2(-rotation[2, 0], sy)
        z = 0.0
    return np.degrees(np.asarray([x, y, z], dtype=np.float64))


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
    parser.add_argument("--output-subdir", default="fused_params_oracle_global")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--strict-frame-count", action="store_true")
    parser.add_argument(
        "--fit-scope",
        choices=("global", "sequence", "frame"),
        default="sequence",
        help="Use one rotation for all frames, one per sequence, or one per frame.",
    )
    parser.add_argument(
        "--joint-set",
        choices=("upper_body", "visible_upper", "body", "all"),
        default="visible_upper",
        help="Joints used to fit the oracle global rotation.",
    )
    parser.add_argument(
        "--optimize",
        choices=("orient", "transl", "both"),
        default="both",
        help="Which global parameters to modify.",
    )
    parser.add_argument(
        "--translation-mode",
        choices=("keep", "gt-root", "upper-centroid", "sequence-median-root", "sequence-median-upper-centroid"),
        default="gt-root",
        help="GT-root modes only affect raw/global metrics; root-aligned MPJPE/MPVPE ignore transl.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-write-params", action="store_true", help="Only write reports/CSVs; do not save optimized PKLs.")
    parser.add_argument(
        "--fast-labels-only",
        action="store_true",
        help="Compute joints-only oracle labels and write optimized PKLs, skipping full MPJPE/MPVPE evaluation.",
    )
    parser.add_argument("--strict-gt", action="store_true")
    parser.add_argument("--strict-pred", action="store_true")
    return parser.parse_args()


def rebuild_param_vector(person: dict) -> None:
    if all(key in person for key in PARAM_VECTOR_KEYS):
        person["smplx_param_vector"] = np.concatenate(
            [np.asarray(person[key]).reshape(-1) for key in PARAM_VECTOR_KEYS],
            axis=0,
        ).astype(np.float32)


def load_prediction(path: Path, strict: bool, skipped: list[dict], item: dict) -> dict | None:
    try:
        return one_person(path)
    except Exception as exc:
        if strict:
            raise
        skipped.append(
            {
                "sequence": item["sequence"],
                "prediction_frame": item["prediction_frame"],
                "file": str(path),
                "reason": str(exc),
            }
        )
        return None


def joint_indices(model: NumpySMPLX, name: str) -> np.ndarray:
    if name == "upper_body":
        return np.asarray(
            [
                3,
                6,
                9,
                12,
                15,
                13,
                14,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                *range(25, 55),
            ],
            dtype=np.int64,
        )
    if name == "visible_upper":
        return model.visible_upper_joint_indices
    if name == "body":
        return np.arange(0, 22, dtype=np.int64)
    return np.arange(0, 55, dtype=np.int64)


def rigid_rotation_row(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return R where row-vector source @ R approximates target."""
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    return u @ np.diag(sign) @ vt


def apply_delta_orient(person: dict, delta_left: np.ndarray) -> None:
    current = rotvec_to_matrix(np.asarray(person["global_orient"], dtype=np.float64).reshape(3))
    person["global_orient"] = matrix_to_rotvec(delta_left @ current).astype(np.float32)
    rebuild_param_vector(person)


def set_translation(person: dict, transl: np.ndarray) -> None:
    person["transl"] = np.asarray(transl, dtype=np.float32).reshape(3)
    rebuild_param_vector(person)


def sequence_pairs(
    args: argparse.Namespace,
    video_dir: Path,
    pred_root: Path,
    image_index: dict[str, list[dict]],
    ffprobe: str,
) -> tuple[dict[str, list[dict]], dict, list[str]]:
    warnings: list[str] = []
    if args.sequences == ["all"]:
        sequences = discover_sequences(pred_root, args.params_subdir)
    else:
        sequences = sorted([normalize_sequence_name(x) for x in args.sequences], key=sequence_sort_key)
    data = {}
    metadata = {}
    for sequence in sequences:
        items, sequence_metadata = build_sequence_items(
            sequence,
            args,
            video_dir,
            pred_root,
            image_index,
            ffprobe,
            warnings,
        )
        data[sequence] = items
        metadata[sequence] = sequence_metadata
    return data, metadata, warnings


def load_forward_rows(
    sequence_data: dict[str, list[dict]],
    ground_truth: dict[int, dict],
    model: NumpySMPLX,
    args: argparse.Namespace,
    skipped: list[dict],
) -> list[dict]:
    rows = []
    for sequence, items in sequence_data.items():
        for start in range(0, len(items), args.batch_size):
            batch_items = items[start : start + args.batch_size]
            kept_items = []
            pred_params = []
            target_params = []
            for item in batch_items:
                if item["ground_truth_id"] not in ground_truth:
                    skipped.append(
                        {
                            "sequence": item["sequence"],
                            "prediction_frame": item["prediction_frame"],
                            "file": str(item["file"]),
                            "reason": f"missing GT SMPL-X record {item['ground_truth_id']}",
                        }
                    )
                    continue
                pred = load_prediction(item["file"], args.strict_pred, skipped, item)
                if pred is None:
                    continue
                kept_items.append(item)
                pred_params.append(pred)
                target_params.append(ground_truth[item["ground_truth_id"]]["smplx_param"])
            if not kept_items:
                continue
            if getattr(args, "fast_labels_only", False):
                pred_vertices = [None] * len(pred_params)
                target_vertices = [None] * len(target_params)
                pred_joints = model.forward_joints(pred_params, predicted=True)
                target_joints = model.forward_joints(target_params, predicted=False)
            else:
                pred_vertices, pred_joints = model.forward(pred_params, predicted=True)
                target_vertices, target_joints = model.forward(target_params, predicted=False)
            for index, item in enumerate(kept_items):
                rows.append(
                    {
                        **item,
                        "person": pred_params[index],
                        "target_param": target_params[index],
                        "pred_vertices": pred_vertices[index],
                        "pred_joints": pred_joints[index],
                        "target_vertices": target_vertices[index],
                        "target_joints": target_joints[index],
                    }
                )
    return rows


def fit_rotation(rows: list[dict], indices: np.ndarray) -> np.ndarray:
    source = []
    target = []
    for row in rows:
        source.append(row["pred_joints"][indices] - row["pred_joints"][0])
        target.append(row["target_joints"][indices] - row["target_joints"][0])
    row_rotation = rigid_rotation_row(np.concatenate(source, axis=0), np.concatenate(target, axis=0))
    # The fitted row-vector rotation corresponds to left-multiplying global_orient
    # by its transpose under the convention used by the SMPL-X forward pass.
    return row_rotation.T


def fit_frame_rotation(row: dict, indices: np.ndarray) -> np.ndarray:
    row_rotation = rigid_rotation_row(
        row["pred_joints"][indices] - row["pred_joints"][0],
        row["target_joints"][indices] - row["target_joints"][0],
    )
    return row_rotation.T


def write_person(path: Path, person: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([person], handle)


def copy_non_param_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file() and not path.name.endswith("_params.pkl"):
            shutil.copy2(path, target_dir / path.name)


def evaluate_rows(rows: list[dict], model: NumpySMPLX, label: str, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    frame_rows = []
    summary_rows = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        metrics = geometry_metrics(
            row["pred_vertices"],
            row["pred_joints"],
            row["target_vertices"],
            row["target_joints"],
            model,
        )
        output = {
            "stage": label,
            "sequence": row["sequence"],
            "prediction_frame": row["prediction_frame"],
            "annotation_frame": row["annotation_frame"],
            "ground_truth_id": row["ground_truth_id"],
            **metrics,
        }
        frame_rows.append(output)
        grouped.setdefault(row["sequence"], []).append(output)
    for sequence in sorted(grouped, key=sequence_sort_key):
        summary = summarize(grouped[sequence], sequence)
        summary["stage"] = label
        summary_rows.append(summary)
    all_summary = summarize(frame_rows, "ALL")
    all_summary["stage"] = label
    summary_rows.append(all_summary)
    return frame_rows, summary_rows


def forward_modified(rows: list[dict], model: NumpySMPLX, args: argparse.Namespace) -> list[dict]:
    output = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        pred_params = [row["optimized_person"] for row in batch]
        target_params = [row["target_param"] for row in batch]
        pred_vertices, pred_joints = model.forward(pred_params, predicted=True)
        target_vertices, target_joints = model.forward(target_params, predicted=False)
        for index, row in enumerate(batch):
            updated = dict(row)
            updated["pred_vertices"] = pred_vertices[index]
            updated["pred_joints"] = pred_joints[index]
            updated["target_vertices"] = target_vertices[index]
            updated["target_joints"] = target_joints[index]
            output.append(updated)
    return output


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    video_dir = (args.video_dir or root / "datasets" / "videos" / "SignLanguage").resolve()
    annotation_dir = (args.annotation_dir or root / "datasets" / "annotations" / "SignLanguage").resolve()
    pred_root = (args.pred_root or root / "outputs_fusion_signlanguage").resolve()
    output_dir = (args.output_dir or pred_root / "evaluation" / f"oracle_global_{args.params_subdir}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_index = load_image_index(annotation_dir / "keypoint_annotation.json")
    sequence_data, metadata, warnings = sequence_pairs(
        args,
        video_dir,
        pred_root,
        image_index,
        find_ffprobe(),
    )
    wanted_ids = {item["ground_truth_id"] for items in sequence_data.values() for item in items}
    ground_truth, missing_gt = load_ground_truth_records(
        annotation_dir / "smplx_annotation.json",
        wanted_ids,
        allow_missing=not args.strict_gt,
    )
    if missing_gt:
        warnings.append(f"Skipping {len(missing_gt)} missing GT records; first IDs: {missing_gt[:10]}")

    model = NumpySMPLX(args.model_path.resolve())
    skipped: list[dict] = []
    rows = load_forward_rows(sequence_data, ground_truth, model, args, skipped)
    if not rows:
        raise ValueError("No usable pred/GT frame pairs")

    indices = joint_indices(model, args.joint_set)
    if args.fit_scope == "global":
        global_delta = fit_rotation(rows, indices)
        deltas_by_sequence = {sequence: global_delta for sequence in sequence_data}
        frame_deltas = {}
    elif args.fit_scope == "sequence":
        deltas_by_sequence = {}
        frame_deltas = {}
        for sequence in sequence_data:
            sequence_rows = [row for row in rows if row["sequence"] == sequence]
            if sequence_rows:
                deltas_by_sequence[sequence] = fit_rotation(sequence_rows, indices)
    else:
        deltas_by_sequence = {}
        frame_deltas = {
            (row["sequence"], row["prediction_frame"]): fit_frame_rotation(row, indices)
            for row in rows
        }

    sequence_translation_deltas: dict[str, np.ndarray] = {}
    if args.translation_mode in ("sequence-median-root", "sequence-median-upper-centroid"):
        for sequence in sequence_data:
            values = []
            for row in rows:
                if row["sequence"] != sequence:
                    continue
                delta = frame_deltas.get((row["sequence"], row["prediction_frame"]), deltas_by_sequence.get(sequence, np.eye(3)))
                person = dict(row["person"])
                if args.optimize in ("orient", "both"):
                    apply_delta_orient(person, delta)
                if args.fast_labels_only:
                    pred_joints = model.forward_joints([person], predicted=True)[0]
                else:
                    pred_joints = model.forward([person], predicted=True)[1][0]
                if args.translation_mode == "sequence-median-root":
                    values.append(row["target_joints"][0] - pred_joints[0])
                else:
                    values.append(
                        row["target_joints"][indices].mean(axis=0)
                        - pred_joints[indices].mean(axis=0)
                    )
            if values:
                sequence_translation_deltas[sequence] = np.median(np.asarray(values), axis=0)

    optimized_rows = []
    correction_rows = []
    forward_cache: dict[int, np.ndarray] = {}
    for row in rows:
        person = {key: value.copy() if hasattr(value, "copy") else value for key, value in row["person"].items()}
        delta = frame_deltas.get(
            (row["sequence"], row["prediction_frame"]),
            deltas_by_sequence.get(row["sequence"], np.eye(3)),
        )
        original_orient = np.asarray(person["global_orient"], dtype=np.float64).reshape(3)
        original_transl = np.asarray(person["transl"], dtype=np.float64).reshape(3)
        if args.optimize in ("orient", "both"):
            apply_delta_orient(person, delta)
        transl_delta = np.zeros(3, dtype=np.float64)
        if args.optimize in ("transl", "both") and args.translation_mode != "keep":
            if args.translation_mode in ("sequence-median-root", "sequence-median-upper-centroid"):
                transl_delta = sequence_translation_deltas.get(row["sequence"], np.zeros(3, dtype=np.float64))
            else:
                cache_key = id(person)
                if cache_key not in forward_cache:
                    if args.fast_labels_only:
                        forward_cache[cache_key] = model.forward_joints([person], predicted=True)[0]
                    else:
                        forward_cache[cache_key] = model.forward([person], predicted=True)[1][0]
                pred_joints_after_orient = forward_cache[cache_key]
                if args.translation_mode == "gt-root":
                    transl_delta = row["target_joints"][0] - pred_joints_after_orient[0]
                else:
                    transl_delta = (
                        row["target_joints"][indices].mean(axis=0)
                        - pred_joints_after_orient[indices].mean(axis=0)
                    )
            set_translation(person, original_transl + transl_delta)
        optimized = dict(row)
        optimized["optimized_person"] = person
        optimized_rows.append(optimized)
        delta_rotvec = matrix_to_rotvec(delta)
        delta_euler = matrix_to_euler_xyz_deg(delta)

        correction_rows.append(
            {
                "sequence": row["sequence"],
                "prediction_frame": row["prediction_frame"],
                "ground_truth_id": row["ground_truth_id"],
                "delta_rotvec_x": delta_rotvec[0],
                "delta_rotvec_y": delta_rotvec[1],
                "delta_rotvec_z": delta_rotvec[2],
                "delta_euler_x_deg": delta_euler[0],
                "delta_euler_y_deg": delta_euler[1],
                "delta_euler_z_deg": delta_euler[2],
                "delta_transl_x": transl_delta[0],
                "delta_transl_y": transl_delta[1],
                "delta_transl_z": transl_delta[2],
                "original_global_orient_x": original_orient[0],
                "original_global_orient_y": original_orient[1],
                "original_global_orient_z": original_orient[2],
                "optimized_global_orient_x": np.asarray(person["global_orient"]).reshape(3)[0],
                "optimized_global_orient_y": np.asarray(person["global_orient"]).reshape(3)[1],
                "optimized_global_orient_z": np.asarray(person["global_orient"]).reshape(3)[2],
                "original_transl_x": original_transl[0],
                "original_transl_y": original_transl[1],
                "original_transl_z": original_transl[2],
                "optimized_transl_x": np.asarray(person["transl"]).reshape(3)[0],
                "optimized_transl_y": np.asarray(person["transl"]).reshape(3)[1],
                "optimized_transl_z": np.asarray(person["transl"]).reshape(3)[2],
            }
        )

    if args.fast_labels_only:
        write_csv(output_dir / "oracle_global_corrections.csv", correction_rows)
        write_csv(output_dir / "skipped_frames.csv", skipped)

        if not args.no_write_params:
            copied_dirs = set()
            for row in optimized_rows:
                source_dir = Path(row["file"]).parent
                target_dir = source_dir.parent / args.output_subdir
                write_person(target_dir / Path(row["file"]).name, row["optimized_person"])
                if target_dir not in copied_dirs:
                    copy_non_param_files(source_dir, target_dir)
                    copied_dirs.add(target_dir)

        sequence_corrections = []
        for sequence, delta in sorted(deltas_by_sequence.items(), key=lambda item: sequence_sort_key(item[0])):
            sequence_corrections.append(
                {
                    "sequence": sequence,
                    "delta_rotvec": matrix_to_rotvec(delta).astype(float).tolist(),
                    "delta_euler_xyz_deg": matrix_to_euler_xyz_deg(delta).astype(float).tolist(),
                }
            )
        report = {
            "mode": "oracle_gt_global_alignment_fast_labels_only",
            "warning": "This uses GT and is for training-label generation only. Full MPJPE/MPVPE evaluation was skipped.",
            "paths": {
                "pred_root": str(pred_root),
                "params_subdir": args.params_subdir,
                "output_subdir": args.output_subdir,
                "output_dir": str(output_dir),
                "write_params": not args.no_write_params,
                "model_path": str(args.model_path.resolve()),
            },
            "settings": {
                "fit_scope": args.fit_scope,
                "joint_set": args.joint_set,
                "optimize": args.optimize,
                "translation_mode": args.translation_mode,
                "fast_labels_only": True,
            },
            "frames": {
                "usable": len(rows),
                "skipped": len(skipped),
            },
            "warnings": warnings,
            "metadata": metadata,
            "sequence_corrections": sequence_corrections,
        }
        (output_dir / "oracle_global_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote fast oracle correction labels to {output_dir}")
        if args.no_write_params:
            print("Did not write optimized PKLs because --no-write-params was set")
        else:
            print(f"Wrote optimized PKLs under each sequence: {args.output_subdir}")
        print(f"Usable frames: {len(rows)}; skipped: {len(skipped)}")
        return

    optimized_rows = forward_modified(optimized_rows, model, args)
    before_frame, before_summary = evaluate_rows(rows, model, "before", args)
    after_frame, after_summary = evaluate_rows(optimized_rows, model, "after", args)
    write_csv(output_dir / "oracle_global_frame_metrics_before.csv", before_frame)
    write_csv(output_dir / "oracle_global_frame_metrics_after.csv", after_frame)
    write_csv(output_dir / "oracle_global_summary_before.csv", before_summary)
    write_csv(output_dir / "oracle_global_summary_after.csv", after_summary)
    write_csv(output_dir / "oracle_global_corrections.csv", correction_rows)
    write_csv(output_dir / "skipped_frames.csv", skipped)

    if not args.no_write_params:
        for row in optimized_rows:
            source_dir = Path(row["file"]).parent
            target_dir = source_dir.parent / args.output_subdir
            write_person(target_dir / Path(row["file"]).name, row["optimized_person"])
        copied_dirs = set()
        for row in optimized_rows:
            source_dir = Path(row["file"]).parent
            target_dir = source_dir.parent / args.output_subdir
            if target_dir not in copied_dirs:
                copy_non_param_files(source_dir, target_dir)
                copied_dirs.add(target_dir)

    sequence_corrections = []
    for sequence, delta in sorted(deltas_by_sequence.items(), key=lambda item: sequence_sort_key(item[0])):
        sequence_translation_values = np.asarray(
            [
                [row["delta_transl_x"], row["delta_transl_y"], row["delta_transl_z"]]
                for row in correction_rows
                if row["sequence"] == sequence
            ],
            dtype=np.float64,
        )
        if len(sequence_translation_values):
            translation_median = np.median(sequence_translation_values, axis=0)
            translation_mean = np.mean(sequence_translation_values, axis=0)
        else:
            translation_median = np.zeros(3, dtype=np.float64)
            translation_mean = np.zeros(3, dtype=np.float64)
        sequence_corrections.append(
            {
                "sequence": sequence,
                "delta_rotvec": matrix_to_rotvec(delta).astype(float).tolist(),
                "delta_euler_xyz_deg": matrix_to_euler_xyz_deg(delta).astype(float).tolist(),
                "translation_delta_median": translation_median.astype(float).tolist(),
                "translation_delta_mean": translation_mean.astype(float).tolist(),
            }
        )

    before_all = next(row for row in before_summary if row["sequence"] == "ALL")
    after_all = next(row for row in after_summary if row["sequence"] == "ALL")
    improvements = {
        key: float(before_all[f"{key}_mean"] - after_all[f"{key}_mean"])
        for key in METRICS
        if f"{key}_mean" in before_all
    }
    report = {
        "mode": "oracle_gt_global_alignment",
        "warning": "This uses GT and is for diagnosis/upper-bound measurement only.",
        "paths": {
            "pred_root": str(pred_root),
            "params_subdir": args.params_subdir,
            "output_subdir": args.output_subdir,
            "output_dir": str(output_dir),
            "write_params": not args.no_write_params,
            "model_path": str(args.model_path.resolve()),
        },
        "settings": {
            "fit_scope": args.fit_scope,
            "joint_set": args.joint_set,
            "optimize": args.optimize,
            "translation_mode": args.translation_mode,
        },
        "frames": {
            "usable": len(rows),
            "skipped": len(skipped),
        },
        "warnings": warnings,
        "metadata": metadata,
        "sequence_corrections": sequence_corrections,
        "before_all": before_all,
        "after_all": after_all,
        "improvement_mm": improvements,
    }
    (output_dir / "oracle_global_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fields = [
        "sequence",
        "frames",
        "visible_upper_mpjpe_mm_mean",
        "visible_upper_pa_mpjpe_mm_mean",
        "visible_upper_mpvpe_mm_mean",
        "visible_upper_pa_mpvpe_mm_mean",
        "hands_wrist_mpjpe_mm_mean",
        "hands_pa_mpjpe_mm_mean",
        "hands_wrist_mpvpe_mm_mean",
        "hands_pa_mpvpe_mm_mean",
        "face_head_mpvpe_mm_mean",
        "face_pa_mpvpe_mm_mean",
        "mpjpe_mm_mean",
        "mpvpe_mm_mean",
        "raw_mpjpe_mm_mean",
    ]
    print("before")
    print(",".join(fields))
    for row in before_summary:
        print(",".join(str(row[field]) for field in fields))
    print("\nafter")
    print(",".join(fields))
    for row in after_summary:
        print(",".join(str(row[field]) for field in fields))
    print(f"\nWrote oracle global-alignment diagnostic to {output_dir}")
    if args.no_write_params:
        print("Did not write optimized PKLs because --no-write-params was set")
    else:
        print(f"Wrote optimized PKLs under each sequence: {args.output_subdir}")


if __name__ == "__main__":
    main()
