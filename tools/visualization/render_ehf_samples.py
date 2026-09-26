#!/usr/bin/env python3
"""Render EHF image / GT aligned mesh / predicted mesh comparison samples."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ehf-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 2, 4, 13, 35, 96])
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument(
        "--eval-module-dir",
        type=Path,
        default=Path("/Users/khineaindrayhtun/Downloads/videos"),
    )
    return parser.parse_args()


def load_person(path: Path) -> dict:
    with path.open("rb") as handle:
        prediction = pickle.load(handle)
    if not isinstance(prediction, list) or not prediction:
        raise ValueError(f"No predicted person in {path}")
    return prediction[0]


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.eval_module_dir))
    from evaluate_ehf_geometry import read_aligned_ply_vertices
    from evaluate_smplx_geometry import NumpySMPLX
    from render_smplx_frame_comparisons import label_panel, load_faces, render_mesh

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = NumpySMPLX(args.model_path)
    faces = load_faces(args.model_path)
    rows: list[Image.Image] = []
    manifest = []

    for frame in args.frames:
        image_path = args.ehf_dir / f"{frame:02d}_img.jpg"
        gt_path = args.ehf_dir / f"{frame:02d}_align.ply"
        pred_path = args.pred_dir / f"{frame:06d}_params.pkl"
        if not pred_path.exists():
            pred_path = args.pred_dir / f"{frame:03d}_params.pkl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction for frame {frame}: {pred_path}")

        source = fit_image(image_path, args.image_size)
        gt_vertices = read_aligned_ply_vertices(gt_path)
        gt_joints = np.einsum("jv,vc->jc", model.joint_regressor, gt_vertices, optimize=True)

        pred_vertices_batch, pred_joints_batch = model.forward([load_person(pred_path)], predicted=True)
        pred_vertices = pred_vertices_batch[0]
        pred_joints = pred_joints_batch[0]

        gt_render = gt_vertices - gt_joints[0]
        pred_render = pred_vertices - pred_joints[0]
        # Display-only flip: PIL raster coordinates grow downward, while EHF's
        # metric coordinates make the body easier to inspect with y inverted.
        gt_render = gt_render.copy()
        pred_render = pred_render.copy()
        gt_render[:, 1] *= -1.0
        pred_render[:, 1] *= -1.0
        combined = np.concatenate([gt_render, pred_render], axis=0)
        center = combined.mean(axis=0)
        center[1] = (combined[:, 1].min() + combined[:, 1].max()) / 2
        scale = float(np.max(np.ptp(combined[:, [0, 1]], axis=0)))

        gt_panel = render_mesh(gt_render, faces, center, scale, args.image_size, (55, 110, 220))
        pred_panel = render_mesh(pred_render, faces, center, scale, args.image_size, (230, 120, 40))
        overlay = Image.new("RGBA", (args.image_size, args.image_size), "white")
        overlay.alpha_composite(render_mesh(gt_render, faces, center, scale, args.image_size, (55, 110, 220), alpha=120))
        overlay.alpha_composite(render_mesh(pred_render, faces, center, scale, args.image_size, (230, 120, 40), alpha=120))

        panels = [
            label_panel(source, f"EHF {frame:02d} image"),
            label_panel(gt_panel, "GT aligned PLY"),
            label_panel(pred_panel, "Pred aligned rootrot"),
            label_panel(overlay.convert("RGB"), "overlay: GT blue, pred orange"),
        ]
        row = Image.new("RGB", (args.image_size * len(panels), args.image_size), "white")
        for index, panel in enumerate(panels):
            row.paste(panel, (index * args.image_size, 0))
        rows.append(row)
        manifest.append(
            {
                "frame": frame,
                "image": str(image_path),
                "gt_mesh": str(gt_path),
                "prediction": str(pred_path),
            }
        )

    sheet = Image.new("RGB", (args.image_size * 4, args.image_size * len(rows)), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * args.image_size))

    sheet_path = args.output_dir / "EHF_aligned_rootrot_samples.png"
    sheet.save(sheet_path)
    report = {
        "model_path": str(args.model_path),
        "pred_dir": str(args.pred_dir),
        "frames": args.frames,
        "sheet": str(sheet_path),
        "manifest": manifest,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
