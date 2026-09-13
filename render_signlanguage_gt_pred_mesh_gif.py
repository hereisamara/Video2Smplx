#!/usr/bin/env python3
"""Render SignLanguage GT-vs-pred SMPL-X mesh comparison as GIF/PNG.

This is a lightweight software renderer: it uses the SMPL-X vertices/faces,
orthographic projection, simple z-sorted triangle drawing, and PIL GIF output.
It does not require pyrender, OpenCV, ffmpeg, or a GPU.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate_signlanguage_geometry import (
    NumpySMPLX,
    find_ffprobe,
    load_ground_truth_records,
    load_image_index,
    nearest_annotation_index,
    prediction_frame_number,
    video_metadata,
)


GT_COLOR = np.asarray([52, 111, 218], dtype=np.float32)
PRED_COLOR = np.asarray([227, 116, 32], dtype=np.float32)
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
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--panel-size", type=int, default=300)
    parser.add_argument("--duration-ms", type=int, default=90)
    parser.add_argument("--sample-frames", type=int, nargs="+", default=[1, 20, 40, 60, 80, 100])
    parser.add_argument("--view", choices=("front", "back"), default="front")
    parser.add_argument("--tag", default="no_stabilization")
    parser.add_argument("--pred-subtitle", default="no stabilization")
    return parser.parse_args()


def load_person(path: Path) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError(f"Expected one predicted person in {path}")
    return data[0]


def load_faces(model_path: Path) -> np.ndarray:
    data = np.load(model_path, allow_pickle=True)
    return np.asarray(data["f"], dtype=np.int32)


def collect_pairs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    image_index = load_image_index(args.annotation_dir / "keypoint_annotation.json")
    fps, video_frames = video_metadata(args.video_path, find_ffprobe())
    pred_files = sorted(args.pred_dir.glob("*_params.pkl"), key=prediction_frame_number)
    if args.max_frames:
        pred_files = pred_files[: args.max_frames]
    pred_files = pred_files[:: args.stride]

    annotation_images = image_index[args.sequence]
    pairs = []
    wanted_ids = set()
    for file in pred_files:
        prediction_frame = prediction_frame_number(file)
        prediction_index = prediction_frame - 1
        annotation_index = nearest_annotation_index(prediction_index, fps, args.annotation_fps)
        image = annotation_images[annotation_index]
        ground_truth_id = int(image["id"])
        wanted_ids.add(ground_truth_id)
        pairs.append(
            {
                "file": file,
                "prediction_frame": prediction_frame,
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
        "rendered_frames": len(pred_files),
        "stride": args.stride,
    }
    return pairs, {"ground_truth": gt, "metadata": metadata}


def load_meshes(args: argparse.Namespace, model: NumpySMPLX, pairs: list[dict], gt_records: dict) -> list[dict]:
    frames = []
    for pair in pairs:
        pred_vertices, pred_joints = model.forward([load_person(pair["file"])], predicted=True)
        gt_vertices, gt_joints = model.forward([gt_records[pair["ground_truth_id"]]["smplx_param"]], predicted=False)
        frames.append(
            {
                **pair,
                "pred_vertices": pred_vertices[0] - pred_joints[0, 0],
                "gt_vertices": gt_vertices[0] - gt_joints[0, 0],
            }
        )
    return frames


def display_vertices(vertices: np.ndarray, view: str) -> np.ndarray:
    output = vertices.copy()
    if view == "front":
        # SMPL-X predictions are back-facing in the native XY projection used
        # by this software renderer; rotate the display camera around vertical.
        output[:, 0] *= -1.0
        output[:, 2] *= -1.0
    return output


def compute_view(frames: list[dict], view: str) -> tuple[np.ndarray, float]:
    vertices = []
    for frame in frames:
        vertices.append(display_vertices(frame["gt_vertices"], view))
        vertices.append(display_vertices(frame["pred_vertices"], view))
    all_vertices = np.concatenate(vertices, axis=0)
    xy = all_vertices[:, [0, 1]]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    scale = float(np.max(max_xy - min_xy))
    return center, max(scale, 1e-6)


def project(vertices: np.ndarray, center: np.ndarray, scale: float, size: int) -> np.ndarray:
    margin = size * 0.08
    usable = size - 2 * margin
    xy = vertices[:, [0, 1]]
    return (xy - center[None]) / scale * usable + size / 2


def shaded_color(base: np.ndarray, shade: float, alpha: int) -> tuple[int, int, int, int]:
    color = np.clip(base * shade + 255.0 * (1.0 - shade) * 0.10, 0, 255)
    return (int(color[0]), int(color[1]), int(color[2]), alpha)


def draw_label(image: Image.Image, title: str, subtitle: str | None = None) -> None:
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), title, fill=TEXT)
    if subtitle:
        draw.text((10, image.height - 22), subtitle, fill=TEXT)


def render_mesh(
    vertices_in: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
    size: int,
    base_color: np.ndarray,
    title: str,
    view: str,
    subtitle: str | None = None,
    alpha: int = 255,
) -> Image.Image:
    vertices = display_vertices(vertices_in, view)
    pixels = project(vertices, center, scale, size)
    triangles_3d = vertices[faces]
    triangles_2d = pixels[faces]

    v1 = triangles_3d[:, 1] - triangles_3d[:, 0]
    v2 = triangles_3d[:, 2] - triangles_3d[:, 0]
    normals = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normals, axis=1)
    normal_norm[normal_norm < 1e-8] = 1.0
    normals = normals / normal_norm[:, None]
    shade = 0.52 + 0.42 * np.abs(normals[:, 2])
    z_order = np.argsort(triangles_3d[:, :, 2].mean(axis=1))

    image = Image.new("RGBA", (size, size), (248, 248, 246, 0 if alpha < 255 else 255))
    draw = ImageDraw.Draw(image, "RGBA")
    if alpha == 255:
        draw.rectangle((0, 0, size - 1, size - 1), fill=(248, 248, 246, 255), outline=(218, 218, 212, 255))
    for index in z_order:
        points = [tuple(point) for point in triangles_2d[index]]
        draw.polygon(points, fill=shaded_color(base_color, float(shade[index]), alpha))
    if alpha == 255:
        draw_label(image, title, subtitle)
    return image


def compose_frame(
    frame: dict,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
    panel_size: int,
    view: str,
    pred_subtitle: str,
) -> Image.Image:
    gt = render_mesh(frame["gt_vertices"], faces, center, scale, panel_size, GT_COLOR, "GT mesh", view)
    pred = render_mesh(
        frame["pred_vertices"],
        faces,
        center,
        scale,
        panel_size,
        PRED_COLOR,
        "Pred mesh",
        view,
        pred_subtitle,
    )

    overlay = Image.new("RGBA", (panel_size, panel_size), (248, 248, 246, 255))
    overlay.alpha_composite(
        render_mesh(frame["gt_vertices"], faces, center, scale, panel_size, GT_COLOR, "", view, alpha=128)
    )
    overlay.alpha_composite(
        render_mesh(frame["pred_vertices"], faces, center, scale, panel_size, PRED_COLOR, "", view, alpha=128)
    )
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(218, 218, 212), width=1)
    overlay_draw.text((10, 8), "Overlay", fill=TEXT)
    overlay_draw.text(
        (10, panel_size - 22),
        f"pred {frame['prediction_frame']:06d} / ann {frame['annotation_frame']:06d}",
        fill=TEXT,
    )

    output = Image.new("RGB", (panel_size * 3, panel_size), "white")
    output.paste(gt.convert("RGB"), (0, 0))
    output.paste(pred.convert("RGB"), (panel_size, 0))
    output.paste(overlay.convert("RGB"), (panel_size * 2, 0))
    return output


def make_sample_sheet(frames_by_number: dict[int, Image.Image], output_path: Path) -> None:
    if not frames_by_number:
        return
    images = list(frames_by_number.values())
    sheet = Image.new("RGB", (images[0].width, images[0].height * len(images)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (frame_number, image) in enumerate(frames_by_number.items()):
        y = row * image.height
        sheet.paste(image, (0, y))
        draw.text((10, y + 28), f"frame {frame_number}", fill=TEXT)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = NumpySMPLX(args.model_path)
    faces = load_faces(args.model_path)
    pairs, loaded = collect_pairs(args)
    frames = load_meshes(args, model, pairs, loaded["ground_truth"])
    center, scale = compute_view(frames, args.view)

    rendered = [
        compose_frame(frame, faces, center, scale, args.panel_size, args.view, args.pred_subtitle)
        for frame in frames
    ]
    gif_path = args.output_dir / f"{args.sequence}_gt_pred_mesh_{args.tag}.gif"
    rendered[0].save(
        gif_path,
        save_all=True,
        append_images=rendered[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )

    by_frame = {frame["prediction_frame"]: image for frame, image in zip(frames, rendered)}
    samples = {frame: by_frame[frame] for frame in args.sample_frames if frame in by_frame}
    sheet_path = args.output_dir / f"{args.sequence}_gt_pred_mesh_{args.tag}_samples.png"
    make_sample_sheet(samples, sheet_path)

    report = {
        "gif": str(gif_path),
        "sample_sheet": str(sheet_path),
        "sequence": args.sequence,
        "frames": len(frames),
        "faces": int(len(faces)),
        "model_path": str(args.model_path),
        "pred_dir": str(args.pred_dir),
        "metadata": loaded["metadata"],
        "view": args.view,
        "tag": args.tag,
        "pred_subtitle": args.pred_subtitle,
        "note": "Root-aligned SMPL-X surface mesh render. Blue=GT, orange=prediction.",
    }
    report_path = args.output_dir / "mesh_visualization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
