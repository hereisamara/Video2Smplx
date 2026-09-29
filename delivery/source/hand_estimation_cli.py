#!/usr/bin/env python3
"""Hand estimation CLI.

Runs WiLoR on a video or frame directory and writes one per-frame hand PKL
under <output>/hand_params.
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
from video2smplx.runners.wilor import WiLoRRunner


DEFAULT_WILOR_DIR = ROOT / "WiLoR-Inference"


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


def has_hand(result: dict) -> bool:
    return result.get("left_hand_pose") is not None or result.get("right_hand_pose") is not None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate MANO/SMPL-X hand pose parameters with WiLoR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=None, help="Input video path.")
    parser.add_argument("--frames-dir", default=None, help="Existing image frame directory.")
    parser.add_argument("--output", required=True, help="Output root directory.")
    parser.add_argument("--fps", type=int, default=30, help="Frame sampling FPS.")
    parser.add_argument("--materialize-frames", action="store_true", help="Save decoded frames.")
    parser.add_argument("--wilor-dir", default=str(DEFAULT_WILOR_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument(
        "--params-only",
        action="store_true",
        help="Skip final WiLoR vertices, joints, and projection after MANO regression.",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    hand_dir = output / "hand_params"
    hand_dir.mkdir(parents=True, exist_ok=True)
    frames, frame_count, source_mode = frame_source(args, output)

    print("[load] WiLoR")
    runner = WiLoRRunner(
        Path(args.wilor_dir),
        device=args.device,
        rescale_factor=args.rescale_factor,
        batch_size=args.batch_size,
        params_only=args.params_only,
    )

    started = time.perf_counter()
    total = 0
    valid = 0
    errors: list[str] = []

    for frame in tqdm(frames, total=frame_count, desc="Hand estimation"):
        try:
            result = runner.predict(frame)
        except Exception as exc:
            errors.append(f"{frame.frame_id:06d}: {type(exc).__name__}: {exc}")
            result = runner.empty_result()
        total += 1
        valid += int(has_hand(result))
        out_path = hand_dir / f"{frame.frame_id:06d}_hand.pkl"
        with out_path.open("wb") as f:
            pickle.dump(result, f)

    report = {
        "program": "hand_estimation_cli.py",
        "source_mode": source_mode,
        "video": args.video,
        "frames_dir": args.frames_dir,
        "output_dir": str(output),
        "hand_params": str(hand_dir),
        "frames_total": total,
        "frames_with_hand": valid,
        "seconds": time.perf_counter() - started,
        "errors": errors[:100],
        "error_count": len(errors),
        "settings": vars(args),
    }
    (output / "hand_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
