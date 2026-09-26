#!/usr/bin/env python3
"""Create report-ready torso reprojection charts and landmark overlays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/video2smplx-matplotlib")

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from video2smplx.reprojection import (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    TORSO_LANDMARKS,
    compare_landmarks,
    normalized_to_pixels,
)


COLORS = {
    "reference": (40, 180, 40),       # BGR green
    "standard": (190, 45, 190),      # BGR magenta
    "stabilized": (30, 140, 240),    # BGR orange
}
DRAW_JOINTS = (0, 11, 12, 13, 14, 23, 24)
CONNECTIONS = ((0, 11), (0, 12), (11, 12), (11, 13), (12, 14), (11, 23), (12, 24), (23, 24))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--stabilized", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=(80, 160, 240))
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    return parser.parse_args()


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
    return frame


def create_landmarker(model: Path):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model.resolve()), delegate=python.BaseOptions.Delegate.CPU
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.2,
        min_pose_presence_confidence=0.2,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect(frame: np.ndarray, landmarker) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks:
        raise RuntimeError("Pose was not detected in a requested visualization frame")
    landmarks = result.pose_landmarks[0]
    xy = np.asarray([(point.x, point.y) for point in landmarks], dtype=np.float64)
    visibility = np.asarray([point.visibility for point in landmarks], dtype=np.float64)
    return normalized_to_pixels(xy, frame.shape[1], frame.shape[0]), visibility


def draw_pose(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness=3):
    points_int = np.rint(points).astype(int)
    for first, second in CONNECTIONS:
        cv2.line(image, tuple(points_int[first]), tuple(points_int[second]), color, thickness, cv2.LINE_AA)
    for index in DRAW_JOINTS:
        cv2.circle(image, tuple(points_int[index]), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, tuple(points_int[index]), 5, color, -1, cv2.LINE_AA)


def add_legend(image: np.ndarray):
    entries = (("Input", "reference"), ("Standard", "standard"), ("Stabilized", "stabilized"))
    x = 20
    for label, key in entries:
        cv2.line(image, (x, 28), (x + 32, 28), COLORS[key], 5, cv2.LINE_AA)
        cv2.putText(image, label, (x + 40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        x += 180


def create_overlays(args, landmarker):
    panels = []
    for frame_index in args.frames:
        reference_frame = read_frame(args.reference, frame_index)
        standard_frame = read_frame(args.standard, frame_index)
        stabilized_frame = read_frame(args.stabilized, frame_index)
        reference_xy, reference_visibility = detect(reference_frame, landmarker)
        standard_xy, standard_visibility = detect(standard_frame, landmarker)
        stabilized_xy, stabilized_visibility = detect(stabilized_frame, landmarker)

        standard = compare_landmarks(
            reference_xy,
            reference_visibility,
            standard_xy,
            standard_visibility,
            visibility_threshold=args.visibility_threshold,
            scored_landmarks=TORSO_LANDMARKS,
        )
        stabilized = compare_landmarks(
            reference_xy,
            reference_visibility,
            stabilized_xy,
            stabilized_visibility,
            visibility_threshold=args.visibility_threshold,
            scored_landmarks=TORSO_LANDMARKS,
        )
        if standard is None or stabilized is None:
            raise RuntimeError(f"Frame {frame_index} lacks enough visible torso landmarks")

        panel = reference_frame.copy()
        overlay = panel.copy()
        draw_pose(overlay, reference_xy, COLORS["reference"], thickness=4)
        draw_pose(overlay, standard["aligned_rendered_xy"], COLORS["standard"], thickness=3)
        draw_pose(overlay, stabilized["aligned_rendered_xy"], COLORS["stabilized"], thickness=2)
        panel = cv2.addWeighted(overlay, 0.82, panel, 0.18, 0)
        add_legend(panel)
        title = (
            f"Frame {frame_index} | standard {np.mean(standard['errors_normalized']):.3f} | "
            f"stabilized {np.mean(stabilized['errors_normalized']):.3f}"
        )
        cv2.rectangle(panel, (0, panel.shape[0] - 44), (panel.shape[1], panel.shape[0]), (255, 255, 255), -1)
        cv2.putText(panel, title, (16, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
        panels.append(panel)

    combined = np.concatenate(panels, axis=0)
    output_path = args.output_dir / "torso_reprojection_overlays.png"
    cv2.imwrite(str(output_path), combined)
    return output_path


def weighted_group(rows: list[dict], names: set[str]) -> tuple[float, float]:
    selected = [row for row in rows if row["joint_name"] in names]
    counts = np.asarray([int(row["observations"]) for row in selected], dtype=float)
    errors = np.asarray([float(row["mean_error_shoulder"]) for row in selected])
    pck = np.asarray([float(row["pck_0_2"]) for row in selected])
    return float(np.average(errors, weights=counts)), float(np.average(pck, weights=counts))


def create_chart(args):
    # The per-joint CSVs hold the torso-only aggregate used in the report.
    import csv

    torso_names = {"nose", "left_elbow", "right_elbow", "left_hip", "right_hip"}
    metrics = []
    for video in (args.standard, args.stabilized):
        csv_path = args.report.parent / f"{video.stem}_joints.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            metrics.append(weighted_group(list(csv.DictReader(handle)), torso_names))

    labels = ("Standard", "Stabilized")
    colors = ("#b52eb5", "#f08c1e")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    errors = [metric[0] for metric in metrics]
    pcks = [100 * metric[1] for metric in metrics]
    axes[0].bar(labels, errors, color=colors, width=0.58)
    axes[0].set_title("Mean upper-torso error (lower is better)")
    axes[0].set_ylabel("Reference shoulder widths")
    axes[0].set_ylim(0, max(errors) * 1.35)
    axes[1].bar(labels, pcks, color=colors, width=0.58)
    axes[1].set_title("PCK@0.2 (higher is better)")
    axes[1].set_ylabel("Correct torso landmarks (%)")
    axes[1].set_ylim(0, 105)
    for axis, values, fmt in ((axes[0], errors, "{:.3f}"), (axes[1], pcks, "{:.1f}%")):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        for index, value in enumerate(values):
            axis.text(index, value + axis.get_ylim()[1] * 0.025, fmt.format(value), ha="center", fontweight="bold")
    fig.suptitle("Upper-torso 2D pose agreement (hands excluded)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path = args.output_dir / "torso_metric_summary.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.report.open(encoding="utf-8") as handle:
        json.load(handle)  # Validate that the source report exists and is valid.
    chart = create_chart(args)
    with create_landmarker(args.model) as landmarker:
        overlays = create_overlays(args, landmarker)
    print(chart.resolve())
    print(overlays.resolve())


if __name__ == "__main__":
    main()
