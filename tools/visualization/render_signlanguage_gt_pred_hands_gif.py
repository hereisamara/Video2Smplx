#!/usr/bin/env python3
"""Render SignLanguage GT-vs-pred hand-only SMPL-X joint comparison."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.evaluation.evaluate_signlanguage_geometry import (
    NumpySMPLX,
    find_ffprobe,
    load_ground_truth_records,
    load_image_index,
    nearest_annotation_index,
    prediction_frame_number,
    video_metadata,
)


GT_COLOR = (45, 105, 220)
PRED_COLOR = (225, 115, 35)
TEXT = (35, 35, 35)
GRID = (226, 226, 222)
BG = (248, 248, 246)
PANEL_BORDER = (210, 210, 205)

LEFT_HAND = {
    "name": "left",
    "wrist": 20,
    "indices": np.arange(25, 40, dtype=np.int64),
}
RIGHT_HAND = {
    "name": "right",
    "wrist": 21,
    "indices": np.arange(40, 55, dtype=np.int64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sequence", default="SignLanguage_S2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="pred")
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--panel-size", type=int, default=360)
    parser.add_argument("--duration-ms", type=int, default=90)
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
    pred_files = pred_files[:: max(1, args.stride)]

    annotation_images = image_index[args.sequence]
    pairs = []
    wanted_ids = set()
    for file in pred_files:
        pred_frame = prediction_frame_number(file)
        pred_index = pred_frame - 1
        annotation_index = nearest_annotation_index(pred_index, fps, args.annotation_fps)
        if annotation_index >= len(annotation_images):
            continue
        image = annotation_images[annotation_index]
        gt_id = int(image["id"])
        wanted_ids.add(gt_id)
        pairs.append(
            {
                "file": file,
                "prediction_frame": pred_frame,
                "annotation_frame": annotation_index + 1,
                "ground_truth_id": gt_id,
            }
        )

    gt, missing = load_ground_truth_records(
        args.annotation_dir / "smplx_annotation.json",
        wanted_ids,
        allow_missing=True,
    )
    if missing:
        pairs = [pair for pair in pairs if pair["ground_truth_id"] in gt]
    return pairs, {
        "ground_truth": gt,
        "metadata": {
            "video_fps": fps,
            "video_frames": video_frames,
            "prediction_frames": len(pred_files),
            "usable_pairs": len(pairs),
            "missing_gt": len(missing),
            "stride": args.stride,
        },
    }


def load_joints(model: NumpySMPLX, pairs: list[dict], gt_records: dict) -> list[dict]:
    frames = []
    for pair in pairs:
        pred_joints = model.forward_joints([load_person(pair["file"])], predicted=True)[0]
        gt_joints = model.forward_joints([gt_records[pair["ground_truth_id"]]["smplx_param"]], predicted=False)[0]
        frames.append({**pair, "pred_joints": pred_joints, "gt_joints": gt_joints})
    return frames


def hand_edges(hand: dict) -> list[tuple[int, int]]:
    wrist = hand["wrist"]
    idx = [int(x) for x in hand["indices"]]
    # SMPL-X hand order in this evaluator follows index, middle, pinky, ring, thumb groups.
    chains = [
        [wrist, idx[0], idx[1], idx[2]],
        [wrist, idx[3], idx[4], idx[5]],
        [wrist, idx[6], idx[7], idx[8]],
        [wrist, idx[9], idx[10], idx[11]],
        [wrist, idx[12], idx[13], idx[14]],
    ]
    return [(chain[i], chain[i + 1]) for chain in chains for i in range(len(chain) - 1)]


def hand_points(joints: np.ndarray, hand: dict) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray([hand["wrist"], *hand["indices"]], dtype=np.int64)
    root = joints[hand["wrist"]]
    local = joints[indices] - root
    return indices, local


def view_from_frames(frames: list[dict]) -> tuple[np.ndarray, float]:
    points = []
    for frame in frames:
        for hand in (LEFT_HAND, RIGHT_HAND):
            _, gt = hand_points(frame["gt_joints"], hand)
            _, pred = hand_points(frame["pred_joints"], hand)
            points.extend([gt, pred])
    all_points = np.concatenate(points, axis=0)
    xy = all_points[:, [0, 1]].copy()
    xy[:, 1] *= -1.0
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    scale = float(np.max(max_xy - min_xy))
    return center, max(scale, 1e-6)


def project(local_points: np.ndarray, center: np.ndarray, scale: float, panel_size: int) -> np.ndarray:
    margin = panel_size * 0.15
    usable = panel_size - 2 * margin
    xy = local_points[:, [0, 1]].copy()
    xy[:, 1] *= -1.0
    return ((xy - center[None]) / scale) * usable + panel_size / 2.0


def draw_grid(draw: ImageDraw.ImageDraw, size: int) -> None:
    step = size // 4
    for x in range(step, size, step):
        draw.line((x, 0, x, size), fill=GRID, width=1)
    for y in range(step, size, step):
        draw.line((0, y, size, y), fill=GRID, width=1)


def draw_hand(
    draw: ImageDraw.ImageDraw,
    joints: np.ndarray,
    hand: dict,
    center: np.ndarray,
    scale: float,
    panel_size: int,
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    indices, local = hand_points(joints, hand)
    pixel_by_joint = {
        int(index): tuple(point)
        for index, point in zip(indices, project(local, center, scale, panel_size))
    }
    for parent, child in hand_edges(hand):
        draw.line([pixel_by_joint[parent], pixel_by_joint[child]], fill=color, width=width)
    radius = max(3, width + 1)
    for point in pixel_by_joint.values():
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def make_panel(size: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(image)
    draw_grid(draw, size)
    draw.rectangle((0, 0, size - 1, size - 1), outline=PANEL_BORDER, width=1)
    draw.text((12, 10), title, fill=TEXT)
    return image, draw


def compose_frame(frame: dict, center: np.ndarray, scale: float, panel_size: int) -> Image.Image:
    panels = []
    specs = [
        ("Left GT", "gt_joints", LEFT_HAND, GT_COLOR),
        ("Left overlay", None, LEFT_HAND, None),
        ("Right GT", "gt_joints", RIGHT_HAND, GT_COLOR),
        ("Right overlay", None, RIGHT_HAND, None),
    ]
    for title, source, hand, color in specs:
        image, draw = make_panel(panel_size, title)
        if source is not None:
            draw_hand(draw, frame[source], hand, center, scale, panel_size, color, width=4)
        else:
            draw_hand(draw, frame["gt_joints"], hand, center, scale, panel_size, GT_COLOR, width=3)
            draw_hand(draw, frame["pred_joints"], hand, center, scale, panel_size, PRED_COLOR, width=3)
            draw.text((12, panel_size - 28), "blue GT  orange pred", fill=TEXT)
        panels.append(image)

    output = Image.new("RGB", (panel_size * 2, panel_size * 2 + 30), "white")
    output.paste(panels[0], (0, 0))
    output.paste(panels[1], (panel_size, 0))
    output.paste(panels[2], (0, panel_size))
    output.paste(panels[3], (panel_size, panel_size))
    draw = ImageDraw.Draw(output)
    draw.text(
        (12, panel_size * 2 + 7),
        f"frame {frame['prediction_frame']:06d} / annotation {frame['annotation_frame']:06d}",
        fill=TEXT,
    )
    return output


def make_sample_sheet(samples: dict[int, Image.Image], output_path: Path) -> None:
    if not samples:
        return
    images = list(samples.values())
    width = images[0].width
    height = images[0].height
    columns = 2
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (width * columns, height * rows), "white")
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = NumpySMPLX(args.model_path)
    pairs, loaded = collect_pairs(args)
    frames = load_joints(model, pairs, loaded["ground_truth"])
    if not frames:
        raise ValueError("No usable GT/pred frame pairs for hand visualization")

    center, scale = view_from_frames(frames)
    rendered = [compose_frame(frame, center, scale, args.panel_size) for frame in frames]
    base_name = f"{args.sequence}_gt_pred_hands_{args.tag}"
    gif_path = args.output_dir / f"{base_name}.gif"
    rendered[0].save(
        gif_path,
        save_all=True,
        append_images=rendered[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )

    rendered_by_frame = {frame["prediction_frame"]: image for frame, image in zip(frames, rendered)}
    samples = {
        frame_number: rendered_by_frame[frame_number]
        for frame_number in args.sample_frames
        if frame_number in rendered_by_frame
    }
    sheet_path = args.output_dir / f"{base_name}_samples.png"
    make_sample_sheet(samples, sheet_path)

    report = {
        "gif": str(gif_path),
        "sample_sheet": str(sheet_path),
        "sequence": args.sequence,
        "frames": len(frames),
        "pred_dir": str(args.pred_dir),
        "model_path": str(args.model_path),
        "metadata": loaded["metadata"],
        "note": "Wrist-anchored hand-only SMPL-X joint skeletons. Blue=GT, orange=prediction.",
    }
    report_path = args.output_dir / f"{base_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
