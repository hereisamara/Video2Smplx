#!/usr/bin/env python3
"""Render before/after SignLanguage SMPL-X mesh comparisons.

This script makes compact subjective comparison GIFs using the same lightweight
software renderer as the existing GT-vs-pred mesh visualizer. It compares:

    GT mesh | before prediction | after prediction | before overlay | after overlay

The render is root-aligned by default, matching the MPJPE/MPVPE evaluation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tools.evaluation.evaluate_signlanguage_geometry import (
    NumpySMPLX,
    discover_sequences,
    find_ffprobe,
    load_ground_truth_records,
    load_image_index,
    nearest_annotation_index,
    normalize_sequence_name,
    prediction_frame_number,
    sequence_sort_key,
    video_metadata,
)


GT_COLOR = np.asarray([52, 111, 218], dtype=np.float32)
BEFORE_COLOR = np.asarray([222, 104, 40], dtype=np.float32)
AFTER_COLOR = np.asarray([42, 150, 96], dtype=np.float32)
TEXT = (35, 35, 35)
BG = (248, 248, 246)
BORDER = (218, 218, 212)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("datasets/videos/SignLanguage"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("datasets/annotations/SignLanguage"))
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-subdir", default="fused_params_model_global_hand_corrected")
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-subdir", default="fused_params_2d_upper_corrected_s0p75")
    parser.add_argument("--model-path", type=Path, default=Path("signlanguage_global_correction_server/models/SMPLX_FEMALE.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["random"],
        help="Sequence numbers/names, or 'random' to sample from usable sequences.",
    )
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--annotation-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--panel-size", type=int, default=260)
    parser.add_argument("--duration-ms", type=int, default=90)
    parser.add_argument("--sample-frames", type=int, nargs="+", default=[1, 30, 60, 90, 120])
    parser.add_argument("--view", choices=("front", "back"), default="front")
    parser.add_argument("--alignment", choices=("root", "world"), default="root")
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--tag", default="before_after")
    return parser.parse_args()


def load_person(path: Path) -> dict:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or len(data) != 1:
        count = len(data) if hasattr(data, "__len__") else "unknown"
        raise ValueError(f"Expected exactly one predicted person in {path}, got {count}")
    return data[0]


def load_faces(model_path: Path) -> np.ndarray:
    data = np.load(model_path, allow_pickle=True)
    return np.asarray(data["f"], dtype=np.int32)


def usable_sequences(args: argparse.Namespace) -> list[str]:
    before = set(discover_sequences(args.before_root, args.before_subdir))
    after = set(discover_sequences(args.after_root, args.after_subdir))
    videos = {path.parent.name for path in args.video_dir.glob("SignLanguage_S*/SignLanguage_S*.mp4")}
    usable = sorted(before & after & videos, key=sequence_sort_key)
    if not usable:
        raise ValueError(
            "No usable sequences found. Check before/after roots, params subdirs, and video-dir."
        )
    return usable


def select_sequences(args: argparse.Namespace) -> list[str]:
    available = usable_sequences(args)
    requested = [value.lower() for value in args.sequences]
    if requested == ["random"] or requested == ["all"]:
        rng = random.Random(args.random_seed)
        count = min(args.sample_count, len(available))
        return sorted(rng.sample(available, count), key=sequence_sort_key)

    normalized = [normalize_sequence_name(value) for value in args.sequences]
    available_set = set(available)
    missing = [sequence for sequence in normalized if sequence not in available_set]
    if missing:
        raise ValueError(f"Requested sequences are not usable: {missing}")
    return sorted(normalized, key=sequence_sort_key)


def common_prediction_files(before_dir: Path, after_dir: Path) -> list[tuple[int, Path, Path]]:
    before_by_frame = {prediction_frame_number(path): path for path in before_dir.glob("*_params.pkl")}
    after_by_frame = {prediction_frame_number(path): path for path in after_dir.glob("*_params.pkl")}
    common = sorted(set(before_by_frame) & set(after_by_frame))
    return [(frame, before_by_frame[frame], after_by_frame[frame]) for frame in common]


def collect_pairs(args: argparse.Namespace, sequence: str, image_index: dict[str, list[dict]]) -> tuple[list[dict], dict]:
    video_path = args.video_dir / sequence / f"{sequence}.mp4"
    fps, video_frames = video_metadata(video_path, find_ffprobe())
    before_dir = args.before_root / sequence / args.before_subdir
    after_dir = args.after_root / sequence / args.after_subdir
    pred_files = common_prediction_files(before_dir, after_dir)
    if args.max_frames:
        pred_files = pred_files[: args.max_frames]
    pred_files = pred_files[:: max(1, args.stride)]
    if not pred_files:
        raise ValueError(f"{sequence}: no common prediction frames in before/after dirs")

    annotation_images = image_index[sequence]
    pairs = []
    wanted_ids = set()
    for prediction_frame, before_file, after_file in pred_files:
        prediction_index = prediction_frame - 1
        annotation_index = nearest_annotation_index(prediction_index, fps, args.annotation_fps)
        if annotation_index >= len(annotation_images):
            continue
        image = annotation_images[annotation_index]
        ground_truth_id = int(image["id"])
        wanted_ids.add(ground_truth_id)
        pairs.append(
            {
                "prediction_frame": prediction_frame,
                "annotation_frame": annotation_index + 1,
                "ground_truth_id": ground_truth_id,
                "before_file": before_file,
                "after_file": after_file,
            }
        )

    gt, missing = load_ground_truth_records(
        args.annotation_dir / "smplx_annotation.json",
        wanted_ids,
        allow_missing=True,
    )
    if missing:
        pairs = [pair for pair in pairs if pair["ground_truth_id"] in gt]
    if not pairs:
        raise ValueError(f"{sequence}: no usable GT/prediction frame pairs")
    return pairs, {
        "ground_truth": gt,
        "metadata": {
            "video_path": str(video_path),
            "video_fps": fps,
            "video_frames": video_frames,
            "candidate_frames": len(pred_files),
            "usable_pairs": len(pairs),
            "missing_gt": len(missing),
            "stride": args.stride,
        },
    }


def align_vertices(vertices: np.ndarray, joints: np.ndarray, alignment: str) -> np.ndarray:
    if alignment == "root":
        return vertices - joints[0]
    return vertices


def load_meshes(
    args: argparse.Namespace,
    model: NumpySMPLX,
    pairs: list[dict],
    gt_records: dict[int, dict],
) -> list[dict]:
    frames = []
    for pair in pairs:
        try:
            before_vertices, before_joints = model.forward([load_person(pair["before_file"])], predicted=True)
            after_vertices, after_joints = model.forward([load_person(pair["after_file"])], predicted=True)
            gt_vertices, gt_joints = model.forward(
                [gt_records[pair["ground_truth_id"]]["smplx_param"]],
                predicted=False,
            )
        except Exception as exc:
            print(f"[skip] frame {pair['prediction_frame']}: {exc}")
            continue
        frames.append(
            {
                **pair,
                "gt_vertices": align_vertices(gt_vertices[0], gt_joints[0], args.alignment),
                "before_vertices": align_vertices(before_vertices[0], before_joints[0], args.alignment),
                "after_vertices": align_vertices(after_vertices[0], after_joints[0], args.alignment),
            }
        )
    if not frames:
        raise ValueError("No frames could be rendered after loading prediction PKLs")
    return frames


def display_vertices(vertices: np.ndarray, view: str) -> np.ndarray:
    output = vertices.copy()
    if view == "front":
        output[:, 0] *= -1.0
        output[:, 2] *= -1.0
    return output


def compute_view(frames: list[dict], view: str) -> tuple[np.ndarray, float]:
    vertices = []
    for frame in frames:
        vertices.append(display_vertices(frame["gt_vertices"], view))
        vertices.append(display_vertices(frame["before_vertices"], view))
        vertices.append(display_vertices(frame["after_vertices"], view))
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

    image = Image.new("RGBA", (size, size), (*BG, 0 if alpha < 255 else 255))
    draw = ImageDraw.Draw(image, "RGBA")
    if alpha == 255:
        draw.rectangle((0, 0, size - 1, size - 1), fill=(*BG, 255), outline=(*BORDER, 255))
    for index in z_order:
        draw.polygon(
            [tuple(point) for point in triangles_2d[index]],
            fill=shaded_color(base_color, float(shade[index]), alpha),
        )
    if title:
        draw_label(image, title, subtitle)
    return image


def overlay_panel(
    gt_vertices: np.ndarray,
    pred_vertices: np.ndarray,
    pred_color: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
    size: int,
    view: str,
    title: str,
    subtitle: str,
) -> Image.Image:
    overlay = Image.new("RGBA", (size, size), (*BG, 255))
    overlay.alpha_composite(render_mesh(gt_vertices, faces, center, scale, size, GT_COLOR, "", view, alpha=120))
    overlay.alpha_composite(render_mesh(pred_vertices, faces, center, scale, size, pred_color, "", view, alpha=135))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, size - 1, size - 1), outline=BORDER, width=1)
    draw.text((10, 8), title, fill=TEXT)
    draw.text((10, size - 22), subtitle, fill=TEXT)
    return overlay


def compose_frame(
    frame: dict,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
    panel_size: int,
    view: str,
    before_label: str,
    after_label: str,
) -> Image.Image:
    gt = render_mesh(frame["gt_vertices"], faces, center, scale, panel_size, GT_COLOR, "GT mesh", view)
    before = render_mesh(
        frame["before_vertices"],
        faces,
        center,
        scale,
        panel_size,
        BEFORE_COLOR,
        "Before",
        view,
        before_label,
    )
    after = render_mesh(
        frame["after_vertices"],
        faces,
        center,
        scale,
        panel_size,
        AFTER_COLOR,
        "After",
        view,
        after_label,
    )
    before_overlay = overlay_panel(
        frame["gt_vertices"],
        frame["before_vertices"],
        BEFORE_COLOR,
        faces,
        center,
        scale,
        panel_size,
        view,
        "Before overlay",
        f"pred {frame['prediction_frame']:06d}",
    )
    after_overlay = overlay_panel(
        frame["gt_vertices"],
        frame["after_vertices"],
        AFTER_COLOR,
        faces,
        center,
        scale,
        panel_size,
        view,
        "After overlay",
        f"ann {frame['annotation_frame']:06d}",
    )

    output = Image.new("RGB", (panel_size * 5, panel_size), "white")
    for column, image in enumerate([gt, before, after, before_overlay, after_overlay]):
        output.paste(image.convert("RGB"), (panel_size * column, 0))
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


def render_sequence(
    args: argparse.Namespace,
    sequence: str,
    model: NumpySMPLX,
    faces: np.ndarray,
    image_index: dict[str, list[dict]],
) -> dict:
    sequence_out = args.output_dir / sequence
    sequence_out.mkdir(parents=True, exist_ok=True)
    pairs, loaded = collect_pairs(args, sequence, image_index)
    frames = load_meshes(args, model, pairs, loaded["ground_truth"])
    center, scale = compute_view(frames, args.view)
    rendered = [
        compose_frame(
            frame,
            faces,
            center,
            scale,
            args.panel_size,
            args.view,
            args.before_label,
            args.after_label,
        )
        for frame in frames
    ]

    gif_path = sequence_out / f"{sequence}_{args.tag}.gif"
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
    sheet_path = sequence_out / f"{sequence}_{args.tag}_samples.png"
    make_sample_sheet(samples, sheet_path)

    return {
        "sequence": sequence,
        "gif": str(gif_path),
        "sample_sheet": str(sheet_path) if samples else None,
        "frames": len(frames),
        "metadata": loaded["metadata"],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequences = select_sequences(args)
    print(f"[render] selected sequences: {' '.join(sequences)}")

    model = NumpySMPLX(args.model_path)
    faces = load_faces(args.model_path)
    image_index = load_image_index(args.annotation_dir / "keypoint_annotation.json")

    results = []
    for sequence in sequences:
        print(f"[render] {sequence}")
        results.append(render_sequence(args, sequence, model, faces, image_index))

    report = {
        "results": results,
        "selected_sequences": sequences,
        "before_root": str(args.before_root),
        "before_subdir": args.before_subdir,
        "after_root": str(args.after_root),
        "after_subdir": args.after_subdir,
        "model_path": str(args.model_path),
        "alignment": args.alignment,
        "view": args.view,
        "tag": args.tag,
        "note": "Root-aligned unless alignment=world. Blue=GT, orange=before, green=after.",
    }
    report_path = args.output_dir / f"{args.tag}_render_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
