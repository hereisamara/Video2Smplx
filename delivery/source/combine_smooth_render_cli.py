#!/usr/bin/env python3
"""Parameter combination, temporal smoothing, NPZ export, and rendering CLI."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video2smplx.fusion import PARAM_VECTOR_KEYS, extract_frame_id, fuse_frame, validate_person
from zero_filter_render import stage_render, stage_smooth, stage_zero_transl


DEFAULT_SMPLX_MODEL = (
    ROOT
    / "SMPLest-X-Inference"
    / "human_models"
    / "human_model_files"
    / "smplx"
    / "SMPLX_NEUTRAL.npz"
)


def indexed_files(folder: Path, pattern: str) -> dict[int, Path]:
    items: dict[int, Path] = {}
    if not folder:
        return items
    for path in sorted(folder.glob(pattern)):
        frame_id = extract_frame_id(path.name)
        if frame_id is not None:
            items[frame_id] = path
    return items


def load_pickle(path: Path | None, default: Any) -> Any:
    if path is None:
        return default
    with path.open("rb") as f:
        return pickle.load(f)


def first_person(frame_data: list[Any]) -> dict[str, Any] | None:
    if frame_data and isinstance(frame_data[0], dict):
        return frame_data[0]
    return None


def stack_field(frames: list[list[Any]], key: str) -> np.ndarray:
    values = []
    max_dim = 0
    for frame in frames:
        person = first_person(frame)
        if person is None or key not in person:
            values.append(None)
            continue
        arr = np.asarray(person[key], dtype=np.float32).reshape(-1)
        values.append(arr)
        max_dim = max(max_dim, arr.shape[0])

    if max_dim == 0:
        return np.full((len(frames), 0), np.nan, dtype=np.float32)

    out = np.full((len(frames), max_dim), np.nan, dtype=np.float32)
    for idx, arr in enumerate(values):
        if arr is not None:
            out[idx, : arr.shape[0]] = arr
    return out


def export_npz(frames: list[list[Any]], frame_ids: list[int], output_path: Path) -> None:
    payload = {
        "frame_id": np.asarray(frame_ids, dtype=np.int32),
        "valid": np.asarray([first_person(frame) is not None for frame in frames], dtype=bool),
    }
    for key in PARAM_VECTOR_KEYS:
        payload[key] = stack_field(frames, key)
    payload["smplx_param_vector"] = stack_field(frames, "smplx_param_vector")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def make_side_by_side(input_video: Path, render_video: Path, output_video: Path, fps: int) -> None:
    import cv2

    cap_left = cv2.VideoCapture(str(input_video))
    cap_right = cv2.VideoCapture(str(render_video))
    if not cap_left.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video}")
    if not cap_right.isOpened():
        raise FileNotFoundError(f"Could not open rendered video: {render_video}")

    ok_left, left = cap_left.read()
    ok_right, right = cap_right.read()
    if not ok_left or not ok_right:
        raise RuntimeError("Could not read first frame for side-by-side video.")

    target_h = min(left.shape[0], right.shape[0])
    def resize_to_height(frame):
        scale = target_h / frame.shape[0]
        width = max(1, int(round(frame.shape[1] * scale)))
        return cv2.resize(frame, (width, target_h), interpolation=cv2.INTER_AREA)

    left = resize_to_height(left)
    right = resize_to_height(right)
    size = (left.shape[1] + right.shape[1], target_h)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open side-by-side writer: {output_video}")

    def write_pair(a, b):
        a = resize_to_height(a)
        b = resize_to_height(b)
        if a.shape[0] != target_h or b.shape[0] != target_h:
            raise RuntimeError("Unexpected resize height mismatch.")
        writer.write(np.concatenate([a, b], axis=1))

    write_pair(left, right)
    while True:
        ok_left, left = cap_left.read()
        ok_right, right = cap_right.read()
        if not ok_left or not ok_right:
            break
        write_pair(left, right)

    cap_left.release()
    cap_right.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse body/hand/face estimates, smooth, export SMPL-X NPZ, and render.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--body-dir", required=True, help="Directory of body *_params.pkl files.")
    parser.add_argument("--hand-dir", default=None, help="Directory of hand *_hand.pkl files.")
    parser.add_argument("--face-dir", default=None, help="Directory of face *_face.pkl files.")
    parser.add_argument("--output", required=True, help="Output root directory.")
    parser.add_argument("--smplx-model", default=str(DEFAULT_SMPLX_MODEL), help="SMPL-X model file/directory.")
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--smooth-poly", type=int, default=3)
    parser.add_argument("--viewport", type=int, default=800)
    parser.add_argument("--skip-zero-transl", action="store_true")
    parser.add_argument("--skip-smooth", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--input-video", default=None, help="Optional input video for side-by-side comparison.")
    args = parser.parse_args()

    started = time.perf_counter()
    output = Path(args.output).resolve()
    fused_dir = output / "fused_params"
    rendered_dir = output / "rendered"
    render_params_dir = rendered_dir / "params"
    fused_dir.mkdir(parents=True, exist_ok=True)
    render_params_dir.mkdir(parents=True, exist_ok=True)

    body_files = indexed_files(Path(args.body_dir), "*_params.pkl")
    hand_files = indexed_files(Path(args.hand_dir), "*_hand.pkl") if args.hand_dir else {}
    face_files = indexed_files(Path(args.face_dir), "*_face.pkl") if args.face_dir else {}
    if not body_files:
        raise FileNotFoundError(f"No body *_params.pkl files found in {args.body_dir}")

    frame_ids = sorted(body_files)
    fused_frames: list[list[Any]] = []
    validation_errors: list[str] = []
    hand_matches = 0
    face_matches = 0

    for frame_id in tqdm(frame_ids, desc="Fuse params"):
        body = load_pickle(body_files[frame_id], [])
        hand = load_pickle(hand_files.get(frame_id), None)
        face = load_pickle(face_files.get(frame_id), None)
        hand_matches += int(hand is not None)
        face_matches += int(face is not None)
        fused = fuse_frame(body, hand, face)
        if fused and isinstance(fused[0], dict):
            for err in validate_person(fused[0]):
                validation_errors.append(f"{frame_id:06d}: {err}")
        fused_frames.append(fused)
        with (fused_dir / f"{frame_id:06d}_params.pkl").open("wb") as f:
            pickle.dump(fused, f)

    render_frames = copy.deepcopy(fused_frames)
    if not args.skip_zero_transl:
        render_frames = stage_zero_transl(render_frames)
    if not args.skip_smooth:
        valid_count = sum(first_person(frame) is not None for frame in render_frames)
        window = min(args.smooth_window, valid_count)
        if window % 2 == 0:
            window -= 1
        if window > args.smooth_poly:
            render_frames = stage_smooth(render_frames, window, args.smooth_poly)
        else:
            print(f"[smooth] skipped: valid_count={valid_count}, window={window}")

    for frame_id, frame_data in zip(frame_ids, render_frames):
        with (render_params_dir / f"{frame_id:06d}_params.pkl").open("wb") as f:
            pickle.dump(frame_data, f)

    npz_path = output / "smplx_params.npz"
    export_npz(render_frames, frame_ids, npz_path)

    rendered_video = rendered_dir / "smplx_render.mp4"
    side_by_side = output / "side_by_side_input_render.mp4"
    if not args.skip_render:
        stage_render(
            all_data=render_frames,
            smplx_model_path=str(args.smplx_model),
            output_video_path=str(rendered_video),
            gender=args.gender,
            video_fps=args.fps,
            viewport_size=args.viewport,
        )
        if args.input_video:
            make_side_by_side(Path(args.input_video), rendered_video, side_by_side, args.fps)
    elif rendered_video.exists() and args.input_video:
        make_side_by_side(Path(args.input_video), rendered_video, side_by_side, args.fps)

    if not args.skip_render and not rendered_video.exists():
        raise RuntimeError(f"Render requested but output was not created: {rendered_video}")
    if args.input_video and not side_by_side.exists() and not args.skip_render:
        raise RuntimeError(f"Side-by-side requested but output was not created: {side_by_side}")

    report = {
        "program": "combine_smooth_render_cli.py",
        "body_dir": str(Path(args.body_dir).resolve()),
        "hand_dir": str(Path(args.hand_dir).resolve()) if args.hand_dir else None,
        "face_dir": str(Path(args.face_dir).resolve()) if args.face_dir else None,
        "output_dir": str(output),
        "fused_params": str(fused_dir),
        "render_params": str(render_params_dir),
        "smplx_params_npz": str(npz_path),
        "rendered_video": str(rendered_video) if rendered_video.exists() else None,
        "side_by_side_video": str(side_by_side) if side_by_side.exists() else None,
        "frames_total": len(frame_ids),
        "hand_matches": hand_matches,
        "face_matches": face_matches,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors[:100],
        "seconds": time.perf_counter() - started,
        "settings": vars(args),
    }
    (output / "combine_render_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))

    if args.skip_render and args.input_video:
        print("[note] side-by-side video requires a rendered video; render was skipped.")
    if args.skip_render:
        print("[note] render skipped by --skip-render.")


if __name__ == "__main__":
    main()
