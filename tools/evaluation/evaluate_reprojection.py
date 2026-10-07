#!/usr/bin/env python3
"""Evaluate visible 2D pose agreement between an input and rendered videos.

This is a video-only proxy metric. It does not measure metric 3D error. Because
this project's standalone renders use a different camera and crop than the
input, the rendered pose is similarity-aligned from its shoulder pair to the
input shoulder pair before residual joint errors are computed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# MediaPipe imports Matplotlib indirectly. Keep its cache inside a writable
# temporary location unless the caller has selected another one.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/video2smplx-matplotlib")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from video2smplx.reprojection import (
    LANDMARK_NAMES,
    compare_landmarks,
    normalized_to_pixels,
)


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL = Path(".cache/pose_landmarker_lite.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure similarity-aligned 2D joint reprojection error."
    )
    parser.add_argument("--reference", type=Path, required=True, help="Original input video.")
    parser.add_argument(
        "--outputs", type=Path, nargs="+", required=True, help="One or more rendered SMPL-X videos."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reprojection_results"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the official Pose Landmarker Lite model when --model is missing.",
    )
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument(
        "--frame-step", type=int, default=1, help="Evaluate every Nth frame (default: every frame)."
    )
    return parser.parse_args()


def ensure_model(path: Path, download: bool) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(
            f"Pose model not found: {path}\n"
            "Pass --download-model once, or provide an existing file with --model."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    metadata = {
        "path": str(path.resolve()),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return metadata


def create_landmarker(model_path: Path):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path), delegate=python.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.2,
        min_tracking_confidence=0.2,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_video(path: Path, landmarker, frame_step: int) -> tuple[dict, dict[int, dict | None]]:
    metadata = video_metadata(path)
    capture = cv2.VideoCapture(str(path))
    detections: dict[int, dict | None] = {}
    frame_index = 0
    print(f"Detecting pose: {path.name} ({metadata['frame_count']} frames)")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                detections[frame_index] = {
                    "xy": np.asarray([(point.x, point.y) for point in landmarks], dtype=np.float64),
                    "visibility": np.asarray(
                        [point.visibility for point in landmarks], dtype=np.float64
                    ),
                }
            else:
                detections[frame_index] = None
        frame_index += 1
    capture.release()
    metadata["decoded_frame_count"] = frame_index
    metadata["sampled_frame_count"] = len(detections)
    metadata["detected_frame_count"] = sum(item is not None for item in detections.values())
    return metadata, detections


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarize_output(
    reference_meta: dict,
    reference_detections: dict[int, dict | None],
    output_meta: dict,
    output_detections: dict[int, dict | None],
    visibility_threshold: float,
) -> tuple[dict, list[dict], list[dict]]:
    rows: list[dict] = []
    per_joint_errors: dict[int, list[float]] = defaultdict(list)
    all_normalized_errors: list[float] = []
    all_pixel_errors: list[float] = []
    comparable_frames = 0
    common_indices = sorted(set(reference_detections) & set(output_detections))

    for frame_index in common_indices:
        reference = reference_detections[frame_index]
        rendered = output_detections[frame_index]
        row = {
            "frame_index": frame_index,
            "reference_detected": reference is not None,
            "output_detected": rendered is not None,
            "comparable": False,
            "joint_count": 0,
            "mean_error_px": None,
            "mean_error_shoulder": None,
            "pck_0_2": None,
            "pck_0_5": None,
        }
        if reference is not None and rendered is not None:
            reference_px = normalized_to_pixels(
                reference["xy"], reference_meta["width"], reference_meta["height"]
            )
            rendered_px = normalized_to_pixels(
                rendered["xy"], output_meta["width"], output_meta["height"]
            )
            comparison = compare_landmarks(
                reference_px,
                reference["visibility"],
                rendered_px,
                rendered["visibility"],
                visibility_threshold=visibility_threshold,
            )
            if comparison is not None:
                comparable_frames += 1
                errors_px = comparison["errors_px"]
                errors_normalized = comparison["errors_normalized"]
                all_pixel_errors.extend(errors_px.tolist())
                all_normalized_errors.extend(errors_normalized.tolist())
                for joint_index, error in zip(comparison["joint_indices"], errors_normalized):
                    per_joint_errors[int(joint_index)].append(float(error))
                row.update(
                    {
                        "comparable": True,
                        "joint_count": len(errors_px),
                        "mean_error_px": float(np.mean(errors_px)),
                        "mean_error_shoulder": float(np.mean(errors_normalized)),
                        "pck_0_2": float(np.mean(errors_normalized <= 0.2)),
                        "pck_0_5": float(np.mean(errors_normalized <= 0.5)),
                    }
                )
        rows.append(row)

    normalized = np.asarray(all_normalized_errors, dtype=np.float64)
    pixels = np.asarray(all_pixel_errors, dtype=np.float64)
    sampled = len(common_indices)
    summary = {
        "output": output_meta,
        "frame_alignment": "frame_index",
        "sampled_frame_pairs": sampled,
        "comparable_frames": comparable_frames,
        "comparable_frame_rate": comparable_frames / sampled if sampled else 0.0,
        "joint_observations": len(normalized),
        "mean_error_px": finite_or_none(np.mean(pixels)) if len(pixels) else None,
        "median_error_px": finite_or_none(np.median(pixels)) if len(pixels) else None,
        "mean_error_shoulder": finite_or_none(np.mean(normalized)) if len(normalized) else None,
        "median_error_shoulder": finite_or_none(np.median(normalized)) if len(normalized) else None,
        "pck_0_2": finite_or_none(np.mean(normalized <= 0.2)) if len(normalized) else None,
        "pck_0_5": finite_or_none(np.mean(normalized <= 0.5)) if len(normalized) else None,
    }
    joint_rows = []
    for joint_index, errors in sorted(per_joint_errors.items()):
        values = np.asarray(errors, dtype=np.float64)
        joint_rows.append(
            {
                "joint_index": joint_index,
                "joint_name": LANDMARK_NAMES[joint_index],
                "observations": len(values),
                "mean_error_shoulder": float(np.mean(values)),
                "median_error_shoulder": float(np.median(values)),
                "pck_0_2": float(np.mean(values <= 0.2)),
                "pck_0_5": float(np.mean(values <= 0.5)),
            }
        )
    return summary, rows, joint_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict) -> None:
    name = Path(summary["output"]["path"]).name
    mean_nme = summary["mean_error_shoulder"]
    pck = summary["pck_0_2"]
    nme_text = "n/a" if mean_nme is None else f"{mean_nme:.4f} shoulder widths"
    pck_text = "n/a" if pck is None else f"{100 * pck:.1f}%"
    print(
        f"{name}: mean={nme_text}, PCK@0.2={pck_text}, "
        f"coverage={summary['comparable_frames']}/{summary['sampled_frame_pairs']} frames"
    )


def main() -> int:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be at least 1")
    for path in (args.reference, *args.outputs):
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

    model_path = ensure_model(args.model, args.download_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with create_landmarker(model_path) as landmarker:
        reference_meta, reference_detections = detect_video(
            args.reference, landmarker, args.frame_step
        )
        result = {
            "metric": "shoulder-aligned visible-joint 2D reprojection error",
            "units": "reference shoulder widths",
            "reference": reference_meta,
            "visibility_threshold": args.visibility_threshold,
            "frame_step": args.frame_step,
            "limitations": [
                "Video-only proxy; not metric 3D ground-truth error.",
                "Standalone render cameras differ from the input camera, so shoulder-based similarity alignment is applied.",
                "Only joints visible in both videos are scored; dense fingers and facial expression are not evaluated.",
                "Frames are paired by frame index, not timestamps.",
            ],
            "outputs": [],
        }

        for output_path in args.outputs:
            output_meta, output_detections = detect_video(output_path, landmarker, args.frame_step)
            summary, frame_rows, joint_rows = summarize_output(
                reference_meta,
                reference_detections,
                output_meta,
                output_detections,
                args.visibility_threshold,
            )
            result["outputs"].append(summary)
            stem = output_path.stem
            write_csv(args.output_dir / f"{stem}_frames.csv", frame_rows)
            write_csv(args.output_dir / f"{stem}_joints.csv", joint_rows)
            print_summary(summary)

    report_path = args.output_dir / "reprojection_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
