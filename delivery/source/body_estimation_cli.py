#!/usr/bin/env python3
"""Body estimation CLI.

Runs SMPLest-X on a video or an existing frame directory and writes one
per-frame PKL file under <output>/body_params.
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
from video2smplx.runners.smplestx import SmplestXRunner


DEFAULT_SMPLESTX_DIR = ROOT / "SMPLest-X-Inference"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate SMPL-X body parameters with SMPLest-X.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=None, help="Input video path.")
    parser.add_argument("--frames-dir", default=None, help="Existing image frame directory.")
    parser.add_argument("--output", required=True, help="Output root directory.")
    parser.add_argument("--fps", type=int, default=30, help="Frame sampling FPS.")
    parser.add_argument("--materialize-frames", action="store_true", help="Save decoded frames.")
    parser.add_argument("--smplestx-dir", default=str(DEFAULT_SMPLESTX_DIR))
    parser.add_argument("--ckpt", default="smplest_x_h", help="SMPLest-X checkpoint name.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multi-person", action="store_true")
    parser.add_argument("--detector-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="SMPLest-X crop batch size.")
    parser.add_argument("--inference-mode", action="store_true")
    parser.add_argument("--single-gpu-model", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    body_dir = output / "body_params"
    body_dir.mkdir(parents=True, exist_ok=True)

    frames, frame_count, source_mode = frame_source(args, output)

    print("[load] SMPLest-X")
    runner = SmplestXRunner(
        Path(args.smplestx_dir),
        ckpt_name=args.ckpt,
        device=args.device,
        multi_person=args.multi_person,
        detector_stride=args.detector_stride,
        use_inference_mode=args.inference_mode,
        single_gpu_model=args.single_gpu_model,
        model_batch_size=args.batch_size,
    )

    started = time.perf_counter()
    total = 0
    valid = 0
    empty = 0
    errors: list[str] = []

    progress = tqdm(total=frame_count, desc="Body estimation")
    try:
        batch: list = []
        for frame in frames:
            batch.append(frame)
            if len(batch) < max(1, args.batch_size):
                continue
            people_batch = runner.predict_batch(batch)
            for batch_frame, people in zip(batch, people_batch):
                total += 1
                if people:
                    valid += 1
                else:
                    empty += 1
                out_path = body_dir / f"{batch_frame.frame_id:06d}_params.pkl"
                with out_path.open("wb") as f:
                    pickle.dump(people, f)
                progress.update(1)
            batch = []

        if batch:
            people_batch = runner.predict_batch(batch)
            for batch_frame, people in zip(batch, people_batch):
                total += 1
                if people:
                    valid += 1
                else:
                    empty += 1
                out_path = body_dir / f"{batch_frame.frame_id:06d}_params.pkl"
                with out_path.open("wb") as f:
                    pickle.dump(people, f)
                progress.update(1)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        progress.close()

    report = {
        "program": "body_estimation_cli.py",
        "source_mode": source_mode,
        "video": args.video,
        "frames_dir": args.frames_dir,
        "output_dir": str(output),
        "body_params": str(body_dir),
        "frames_total": total,
        "frames_with_person": valid,
        "empty_frames": empty,
        "seconds": time.perf_counter() - started,
        "errors": errors,
        "settings": vars(args),
    }
    (output / "body_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
