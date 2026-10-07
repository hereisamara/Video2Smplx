#!/usr/bin/env python3
"""Evaluate Signify predictions against GT SMPL-X OBJ meshes.

Signify data in this project is expected as:

    frames/<sequence>/low_121.png
    smplx_gt/<sequence>/00242.obj

The default pairing is `gt_obj_frame = prediction_frame * 2`, matching the
downloaded Signify frame/GT archives. Predictions are standard Video2Smplx PKLs:

    <pred-root>/<sequence>/<params-subdir>/000121_params.pkl

Metrics reuse the same geometry definitions as evaluate_signlanguage_geometry:
root-aligned MPJPE/MPVPE, PA versions, visible upper, hand, and face subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from tools.evaluation.evaluate_signlanguage_geometry import (
    METRICS,
    NumpySMPLX,
    geometry_metrics,
    load_prediction_batch,
    summarize,
    write_csv,
)


SEQUENCE_RE = re.compile(r"^(.*?)(\d+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True, help="Extracted Signify frames root.")
    parser.add_argument("--gt-root", type=Path, required=True, help="Extracted Signify smplx_gt root.")
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--params-subdir", default="smplestx_params")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--frame-prefix", default="low_")
    parser.add_argument("--gt-frame-multiplier", type=float, default=2.0)
    parser.add_argument("--gt-frame-offset", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--strict-pred", action="store_true")
    parser.add_argument("--strict-gt", action="store_true")
    return parser.parse_args()


def sequence_sort_key(value: Path | str) -> tuple[str, int]:
    name = value.name if isinstance(value, Path) else str(value)
    match = SEQUENCE_RE.match(name)
    if not match:
        return (name, -1)
    suffix = match.group(2)
    return (match.group(1), int(suffix) if suffix else -1)


def discover_sequences(frames_root: Path, gt_root: Path, pred_root: Path, params_subdir: str) -> list[str]:
    frame_sequences = {path.name for path in frames_root.iterdir() if path.is_dir()}
    gt_sequences = {path.name for path in gt_root.iterdir() if path.is_dir()}
    pred_sequences = {
        path.name
        for path in pred_root.iterdir()
        if path.is_dir() and (path / params_subdir).is_dir() and any((path / params_subdir).glob("*_params.pkl"))
    }
    return sorted(frame_sequences & gt_sequences & pred_sequences, key=sequence_sort_key)


def prediction_frame_number(path: Path) -> int:
    match = re.search(r"(\d+)_params\.pkl$", path.name)
    if not match:
        raise ValueError(f"Cannot parse prediction frame from {path.name}")
    return int(match.group(1))


def gt_frame_number(prediction_frame: int, multiplier: float, offset: float) -> int:
    return int(round(prediction_frame * multiplier + offset))


def obj_path_for_frame(gt_root: Path, sequence: str, frame_number: int) -> Path:
    return gt_root / sequence / f"{frame_number:05d}.obj"


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float32)


def regress_joints(model: NumpySMPLX, vertices: np.ndarray) -> np.ndarray:
    return np.einsum("jv,vc->jc", model.joint_regressor, vertices, optimize=True).astype(np.float32)


def build_sequence_items(args: argparse.Namespace, sequence: str) -> list[dict]:
    params_dir = args.pred_root / sequence / args.params_subdir
    files = sorted(params_dir.glob("*_params.pkl"), key=prediction_frame_number)
    if not files:
        raise FileNotFoundError(f"{sequence}: no prediction PKLs in {params_dir}")

    items = []
    for file in files:
        pred_frame = prediction_frame_number(file)
        gt_frame = gt_frame_number(pred_frame, args.gt_frame_multiplier, args.gt_frame_offset)
        gt_path = obj_path_for_frame(args.gt_root, sequence, gt_frame)
        items.append(
            {
                "sequence": sequence,
                "file": file,
                "prediction_frame": pred_frame,
                "ground_truth_frame": gt_frame,
                "ground_truth_file": gt_path,
            }
        )
    return items


def main() -> None:
    args = parse_args()
    args.frames_root = args.frames_root.resolve()
    args.gt_root = args.gt_root.resolve()
    args.pred_root = args.pred_root.resolve()
    args.model_path = args.model_path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sequences == ["all"]:
        sequences = discover_sequences(args.frames_root, args.gt_root, args.pred_root, args.params_subdir)
    else:
        sequences = sorted(args.sequences, key=sequence_sort_key)
    if not sequences:
        raise ValueError("No usable Signify sequences found")

    model = NumpySMPLX(args.model_path)
    frame_rows = []
    summary_rows = []
    skipped_frames = []
    warnings = []
    gt_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    metadata = {}

    for sequence in sequences:
        try:
            items = build_sequence_items(args, sequence)
        except Exception as exc:
            warnings.append(f"{sequence}: skipped setup error: {exc}")
            continue

        sequence_rows = []
        print(f"Evaluating {sequence}: {len(items)} prediction frames", flush=True)
        for start in range(0, len(items), args.batch_size):
            batch_items = items[start : start + args.batch_size]
            kept_items, predicted_params = load_prediction_batch(
                batch_items,
                strict_pred=args.strict_pred,
                skipped_frames=skipped_frames,
                warnings=warnings,
            )
            if not kept_items:
                continue
            try:
                predicted_vertices, predicted_joints = model.forward(predicted_params, predicted=True)
            except Exception as exc:
                if args.strict_pred:
                    raise
                for item in kept_items:
                    skipped_frames.append(
                        {
                            "sequence": sequence,
                            "prediction_frame": item["prediction_frame"],
                            "file": str(item["file"]),
                            "reason": f"prediction_forward_error: {exc}",
                        }
                    )
                continue

            for index, item in enumerate(kept_items):
                gt_path = Path(item["ground_truth_file"])
                try:
                    if gt_path not in gt_cache:
                        gt_vertices = load_obj_vertices(gt_path)
                        if gt_vertices.shape[0] != predicted_vertices.shape[1]:
                            raise ValueError(
                                f"GT vertex count {gt_vertices.shape[0]} != predicted {predicted_vertices.shape[1]}"
                            )
                        gt_cache[gt_path] = (gt_vertices, regress_joints(model, gt_vertices))
                    target_vertices, target_joints = gt_cache[gt_path]
                except Exception as exc:
                    if args.strict_gt:
                        raise
                    skipped_frames.append(
                        {
                            "sequence": sequence,
                            "prediction_frame": item["prediction_frame"],
                            "file": str(item["file"]),
                            "ground_truth_file": str(gt_path),
                            "reason": f"gt_error: {exc}",
                        }
                    )
                    continue

                metrics = geometry_metrics(
                    predicted_vertices[index],
                    predicted_joints[index],
                    target_vertices,
                    target_joints,
                    model,
                )
                row = {
                    "sequence": sequence,
                    "prediction_frame": item["prediction_frame"],
                    "ground_truth_frame": item["ground_truth_frame"],
                    "ground_truth_file": str(gt_path),
                    **metrics,
                }
                sequence_rows.append(row)
                frame_rows.append(row)
        if sequence_rows:
            summary_rows.append(summarize(sequence_rows, sequence))
            metadata[sequence] = {"prediction_frames": len(items), "evaluated_frames": len(sequence_rows)}

    if not frame_rows:
        raise ValueError("No Signify frame pairs could be evaluated")
    summary_rows.append(summarize(frame_rows, "ALL"))
    write_csv(args.output_dir / "geometry_frame_metrics.csv", frame_rows)
    write_csv(args.output_dir / "geometry_summary.csv", summary_rows)
    write_csv(args.output_dir / "skipped_frames.csv", skipped_frames)

    compact_keys = [
        "sequence",
        "frames",
        "mpjpe_mm_mean",
        "pa_mpjpe_mm_mean",
        "mpvpe_mm_mean",
        "pa_mpvpe_mm_mean",
        "body_mpjpe_mm_mean",
        "body_pa_mpjpe_mm_mean",
        "hand_mpjpe_mm_mean",
        "hand_pa_mpjpe_mm_mean",
        "visible_upper_mpvpe_mm_mean",
        "hands_wrist_mpvpe_mm_mean",
        "hands_pa_mpvpe_mm_mean",
        "face_head_mpvpe_mm_mean",
        "face_pa_mpvpe_mm_mean",
    ]
    print(",".join(compact_keys))
    for row in summary_rows:
        print(",".join(str(row.get(key, "")) for key in compact_keys))

    report = {
        "paths": {
            "frames_root": str(args.frames_root),
            "gt_root": str(args.gt_root),
            "pred_root": str(args.pred_root),
            "params_subdir": args.params_subdir,
            "model_path": str(args.model_path),
            "output_dir": str(args.output_dir.resolve()),
        },
        "pairing": {
            "gt_frame": "round(prediction_frame * gt_frame_multiplier + gt_frame_offset)",
            "gt_frame_multiplier": args.gt_frame_multiplier,
            "gt_frame_offset": args.gt_frame_offset,
        },
        "metric_notes": {
            metric: "same definition as evaluate_signlanguage_geometry.py"
            for metric in METRICS
        },
        "sequences": metadata,
        "warnings": warnings[:50],
        "skipped_count": len(skipped_frames),
        "skipped_sample": skipped_frames[:20],
    }
    (args.output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote Signify evaluation to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
