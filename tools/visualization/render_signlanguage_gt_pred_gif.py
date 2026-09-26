#!/usr/bin/env python3
"""Render SignLanguage GT-vs-pred SMPL-X joint comparison as GIF/PNG.

This avoids OpenCV/ffmpeg and uses only NumPy + PIL. The result is a compact
debug visualization: GT skeleton, predicted skeleton, and overlay.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.evaluation.evaluate_signlanguage_geometry import (
    NumpySMPLX,
    find_ffprobe,
    load_ground_truth_records,
    load_image_index,
    nearest_annotation_index,
    prediction_frame_number,
    video_metadata,
)


BODY_COLOR = (45, 105, 220)
PRED_COLOR = (225, 115, 35)
LIGHT_BODY = (150, 180, 245)
LIGHT_PRED = (245, 180, 130)
TEXT = (35, 35, 35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sequence", default="SignLanguage_S2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=111)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--panel-size", type=int, default=420)
    parser.add_argument("--duration-ms", type=int, default=80)
    parser.add_argument("--sample-frames", type=int, nargs="+", default=[1, 20, 40, 60, 80, 100])
    return parser.parse_args()


def load_person(path: Path) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError(f"Expected one predicted person in {path}")
    return data[0]


def collect_pairs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    image_index = load_image_index(args.annotation_dir / "keypoint_annotation.json")
    fps, video_frames = video_metadata(args.video_path, find_ffprobe())
    pred_files = sorted(args.pred_dir.glob("*_params.pkl"), key=prediction_frame_number)
    if args.max_frames:
        pred_files = pred_files[: args.max_frames]
    pred_files = pred_files[:: args.stride]

    pairs = []
    wanted_ids = set()
    annotation_images = image_index[args.sequence]
    for file in pred_files:
        pred_frame = prediction_frame_number(file)
        pred_index = pred_frame - 1
        annotation_index = nearest_annotation_index(pred_index, fps, args.annotation_fps)
        image = annotation_images[annotation_index]
        ground_truth_id = int(image["id"])
        wanted_ids.add(ground_truth_id)
        pairs.append(
            {
                "file": file,
                "prediction_frame": pred_frame,
                "annotation_frame": annotation_index + 1,
                "ground_truth_id": ground_truth_id,
            }
        )

    gt, missing_gt_ids = load_ground_truth_records(
        args.annotation_dir / "smplx_annotation.json",
        wanted_ids,
        allow_missing=True,
    )
    if missing_gt_ids:
        raise KeyError(f"Missing GT SMPL-X records for visualization: {missing_gt_ids[:10]}")
    metadata = {
        "video_fps": fps,
        "video_frames": video_frames,
        "prediction_frames": len(pred_files),
        "stride": args.stride,
    }
    return pairs, {"ground_truth": gt, "metadata": metadata}


def load_joints(args: argparse.Namespace, model: NumpySMPLX, pairs: list[dict], gt_records: dict) -> list[dict]:
    frames = []
    for pair in pairs:
        pred_vertices, pred_joints = model.forward([load_person(pair["file"])], predicted=True)
        gt_vertices, gt_joints = model.forward([gt_records[pair["ground_truth_id"]]["smplx_param"]], predicted=False)
        frames.append(
            {
                **pair,
                "pred_joints": pred_joints[0],
                "gt_joints": gt_joints[0],
            }
        )
    return frames


def visible_joint_indices() -> np.ndarray:
    return np.asarray(
        [
            0,
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


def skeleton_edges(parents: np.ndarray, indices: np.ndarray) -> list[tuple[int, int]]:
    index_set = set(int(x) for x in indices)
    edges = []
    for child in indices:
        parent = int(parents[int(child)])
        if parent in index_set:
            edges.append((parent, int(child)))
    return edges


def transform_points(points: np.ndarray, center: np.ndarray, scale: float, panel_size: int) -> np.ndarray:
    margin = panel_size * 0.11
    usable = panel_size - 2 * margin
    xy = points[:, [0, 1]].copy()
    xy[:, 1] *= -1.0
    normalized = (xy - center[None]) / scale
    pixel = normalized * usable + panel_size / 2
    return pixel


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    joints: np.ndarray,
    indices: np.ndarray,
    edges: list[tuple[int, int]],
    center: np.ndarray,
    scale: float,
    panel_size: int,
    color: tuple[int, int, int],
    light_color: tuple[int, int, int],
    width: int = 4,
) -> None:
    pixels = transform_points(joints, center, scale, panel_size)
    for parent, child in edges:
        draw.line(
            [tuple(pixels[parent]), tuple(pixels[child])],
            fill=light_color if child >= 25 else color,
            width=width if child < 25 else max(2, width - 1),
        )
    radius = 4
    for index in indices:
        x, y = pixels[int(index)]
        joint_color = light_color if int(index) >= 25 else color
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=joint_color)


def label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int]) -> None:
    draw.text(xy, text, fill=TEXT)


def make_panel(size: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (size, size), (248, 248, 246))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(215, 215, 210), width=1)
    label(draw, title, (12, 10))
    return image, draw


def compose_frame(
    frame: dict,
    indices: np.ndarray,
    edges: list[tuple[int, int]],
    center: np.ndarray,
    scale: float,
    panel_size: int,
) -> Image.Image:
    gt_local = frame["gt_joints"] - frame["gt_joints"][0]
    pred_local = frame["pred_joints"] - frame["pred_joints"][0]

    gt_panel, gt_draw = make_panel(panel_size, "GT")
    draw_skeleton(gt_draw, gt_local, indices, edges, center, scale, panel_size, BODY_COLOR, LIGHT_BODY)

    pred_panel, pred_draw = make_panel(panel_size, "Pred no stabilization")
    draw_skeleton(pred_draw, pred_local, indices, edges, center, scale, panel_size, PRED_COLOR, LIGHT_PRED)

    overlay_panel, overlay_draw = make_panel(panel_size, "Overlay")
    draw_skeleton(overlay_draw, gt_local, indices, edges, center, scale, panel_size, BODY_COLOR, LIGHT_BODY, width=3)
    draw_skeleton(overlay_draw, pred_local, indices, edges, center, scale, panel_size, PRED_COLOR, LIGHT_PRED, width=3)
    label(
        overlay_draw,
        f"pred {frame['prediction_frame']:06d} / ann {frame['annotation_frame']:06d}",
        (12, panel_size - 28),
    )

    output = Image.new("RGB", (panel_size * 3, panel_size), "white")
    output.paste(gt_panel, (0, 0))
    output.paste(pred_panel, (panel_size, 0))
    output.paste(overlay_panel, (panel_size * 2, 0))
    return output


def compute_view(frames: list[dict], indices: np.ndarray) -> tuple[np.ndarray, float]:
    all_points = []
    for frame in frames:
        all_points.append((frame["gt_joints"] - frame["gt_joints"][0])[indices])
        all_points.append((frame["pred_joints"] - frame["pred_joints"][0])[indices])
    points = np.concatenate(all_points, axis=0)
    xy = points[:, [0, 1]].copy()
    xy[:, 1] *= -1.0
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    scale = float(np.max(max_xy - min_xy))
    return center, max(scale, 1e-6)


def make_sample_sheet(frames_by_number: dict[int, Image.Image], output_path: Path) -> None:
    if not frames_by_number:
        return
    images = list(frames_by_number.values())
    width = images[0].width
    height = images[0].height
    sheet = Image.new("RGB", (width, height * len(images)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (frame_number, image) in enumerate(frames_by_number.items()):
        y = row * height
        sheet.paste(image, (0, y))
        draw.text((12, y + 32), f"frame {frame_number}", fill=TEXT)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = NumpySMPLX(args.model_path)
    pairs, loaded = collect_pairs(args)
    frames = load_joints(args, model, pairs, loaded["ground_truth"])

    indices = visible_joint_indices()
    edges = skeleton_edges(model.parents, indices)
    center, scale = compute_view(frames, indices)

    rendered = [
        compose_frame(frame, indices, edges, center, scale, args.panel_size)
        for frame in frames
    ]
    gif_path = args.output_dir / "SignLanguage_S2_gt_pred_no_stabilization.gif"
    rendered[0].save(
        gif_path,
        save_all=True,
        append_images=rendered[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )

    samples = {}
    rendered_by_frame = {frame["prediction_frame"]: image for frame, image in zip(frames, rendered)}
    for frame_number in args.sample_frames:
        if frame_number in rendered_by_frame:
            samples[frame_number] = rendered_by_frame[frame_number]
    sheet_path = args.output_dir / "SignLanguage_S2_gt_pred_no_stabilization_samples.png"
    make_sample_sheet(samples, sheet_path)

    report = {
        "gif": str(gif_path),
        "sample_sheet": str(sheet_path),
        "sequence": args.sequence,
        "frames": len(frames),
        "model_path": str(args.model_path),
        "pred_dir": str(args.pred_dir),
        "annotation_dir": str(args.annotation_dir),
        "metadata": loaded["metadata"],
        "note": "Root-aligned SMPL-X visible upper-body joint skeletons. Blue=GT, orange=prediction.",
    }
    report_path = args.output_dir / "visualization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
