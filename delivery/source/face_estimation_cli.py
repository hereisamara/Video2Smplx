#!/usr/bin/env python3
"""Face estimation CLI.

Runs EMOCA on a video or frame directory and writes one per-frame face PKL
under <output>/face_params.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video2smplx.frames import extract_frames_cv2, iter_video_frames_cv2, list_frames
from video2smplx.runners.emoca import EMOCARunner


DEFAULT_EMOCA_DIR = ROOT / "EMOCA-Inference"


def frame_source(args: argparse.Namespace, output: Path):
    if args.frames_dir:
        frames = list_frames(Path(args.frames_dir))
        if not frames:
            raise RuntimeError(f"No image frames found in {args.frames_dir}")
        return frames, len(frames), "frames_dir"

    if not args.video:
        raise ValueError("Provide --video or --frames-dir.")

    video = Path(args.video)
    if args.materialize_frames:
        frames = extract_frames_cv2(video, output / "frames", fps=args.fps)
        if not frames:
            raise RuntimeError(f"No frames extracted from {video}")
        return frames, len(frames), "materialized_video"

    return iter_video_frames_cv2(video, fps=args.fps), None, "streamed_video"


def has_face(result: dict) -> bool:
    return result.get("exp") is not None or result.get("jaw_pose") is not None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate SMPL-X face expression and jaw pose with EMOCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=None, help="Input video path.")
    parser.add_argument("--frames-dir", default=None, help="Existing image frame directory.")
    parser.add_argument("--output", required=True, help="Output root directory.")
    parser.add_argument("--fps", type=int, default=30, help="Frame sampling FPS.")
    parser.add_argument("--materialize-frames", action="store_true", help="Save decoded frames.")
    parser.add_argument("--emoca-dir", default=str(DEFAULT_EMOCA_DIR))
    parser.add_argument("--model-name", default="EMOCA_v2_lr_mse_20")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--scale", type=float, default=1.25)
    parser.add_argument("--face-detector-threshold", type=float, default=0.5)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    face_dir = output / "face_params"
    face_dir.mkdir(parents=True, exist_ok=True)
    frames, frame_count, source_mode = frame_source(args, output)

    print("[load] EMOCA")
    runner = EMOCARunner(
        Path(args.emoca_dir),
        model_name=args.model_name,
        device=args.device,
        crop_size=args.crop_size,
        scale=args.scale,
        face_detector_threshold=args.face_detector_threshold,
    )

    started = time.perf_counter()
    total = 0
    valid = 0
    errors: list[str] = []

    for frame in tqdm(frames, total=frame_count, desc="Face estimation"):
        try:
            result = runner.predict(frame)
        except Exception as exc:
            errors.append(f"{frame.frame_id:06d}: {type(exc).__name__}: {exc}")
            result = runner.empty_result()
        total += 1
        valid += int(has_face(result))
        out_path = face_dir / f"{frame.frame_id:06d}_face.pkl"
        with out_path.open("wb") as f:
            pickle.dump(result, f)

    report = {
        "program": "face_estimation_cli.py",
        "source_mode": source_mode,
        "video": args.video,
        "frames_dir": args.frames_dir,
        "output_dir": str(output),
        "face_params": str(face_dir),
        "frames_total": total,
        "frames_with_face": valid,
        "seconds": time.perf_counter() - started,
        "errors": errors[:100],
        "error_count": len(errors),
        "settings": vars(args),
    }
    (output / "face_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
