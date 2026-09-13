#!/usr/bin/env python3
"""Extract YOLO pose keypoints for SignLanguage videos.

The output JSON is intentionally simple so it can be reused by training and
inference scripts without depending on Ultralytics at that stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_signlanguage_geometry import normalize_sequence_name, sequence_sort_key, video_metadata, find_ffprobe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("datasets/videos/SignLanguage"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo26l-pose.pt")
    parser.add_argument("--sequences", nargs="+", default=["all"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def discover_video_sequences(video_dir: Path) -> list[str]:
    sequences = []
    for directory in video_dir.glob("SignLanguage_S*"):
        if (directory / f"{directory.name}.mp4").exists():
            sequences.append(directory.name)
    return sorted(sequences, key=sequence_sort_key)


def select_person(result) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return np.zeros((17, 2), dtype=float), np.zeros(17, dtype=float), None, 0.0
    xy = keypoints.xy.detach().cpu().numpy()
    conf = keypoints.conf.detach().cpu().numpy() if keypoints.conf is not None else np.ones(xy.shape[:2])
    scores = np.nan_to_num(conf, nan=0.0).sum(axis=1)
    if getattr(result, "boxes", None) is not None and result.boxes.conf is not None:
        box_conf = result.boxes.conf.detach().cpu().numpy()
        scores = scores + box_conf[: len(scores)] * 2.0
    index = int(np.argmax(scores))
    bbox = None
    person_conf = float(np.nanmean(conf[index]))
    if getattr(result, "boxes", None) is not None and result.boxes.xyxy is not None and len(result.boxes.xyxy) > index:
        bbox = result.boxes.xyxy[index].detach().cpu().numpy().astype(float)
        person_conf = float(result.boxes.conf[index].detach().cpu()) if result.boxes.conf is not None else person_conf
    return xy[index].astype(float), np.nan_to_num(conf[index], nan=0.0).astype(float), bbox, person_conf


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    video_dir = args.video_dir.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.sequences == ["all"]:
        sequences = discover_video_sequences(video_dir)
    else:
        sequences = sorted([normalize_sequence_name(x) for x in args.sequences], key=sequence_sort_key)
    ffprobe = find_ffprobe()
    model = YOLO(args.model)

    output = {
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "keypoint_format": "COCO17 xy pixel coordinates, confidence",
        "videos": {},
    }
    for sequence in sequences:
        video_path = video_dir / sequence / f"{sequence}.mp4"
        if not video_path.exists():
            print(f"[skip] missing {video_path}", flush=True)
            continue
        fps, frame_count = video_metadata(video_path, ffprobe)
        print(f"[video] {sequence} fps={fps:.3f} frames={frame_count}", flush=True)
        frames = []
        for frame_index, result in enumerate(
            model.predict(
                source=str(video_path),
                stream=True,
                verbose=False,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
            ),
            start=1,
        ):
            xy, conf, bbox, person_conf = select_person(result)
            height, width = result.orig_shape
            frames.append(
                {
                    "frame": frame_index,
                    "image_width": int(width),
                    "image_height": int(height),
                    "keypoints_xy": xy.tolist(),
                    "keypoints_conf": conf.tolist(),
                    "bbox_xyxy": bbox.tolist() if bbox is not None else None,
                    "person_conf": person_conf,
                }
            )
            if frame_index == 1 or frame_index % 250 == 0:
                print(f"  {sequence}: frame {frame_index}", flush=True)
        output["videos"][sequence] = {
            "video_path": str(video_path),
            "fps": fps,
            "frame_count": frame_count,
            "frames": frames,
        }
    args.output.write_text(json.dumps(output), encoding="utf-8")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
