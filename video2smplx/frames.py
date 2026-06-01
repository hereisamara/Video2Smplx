"""Frame extraction and discovery for Video2Smplx."""

from __future__ import annotations

from pathlib import Path

from video2smplx.contracts import FrameInput
from video2smplx.fusion import extract_frame_id


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_frames_cv2(video: Path, output_dir: Path, fps: int | None = None) -> list[FrameInput]:
    """Extract frames in-process with OpenCV and return normalized frame inputs."""
    import cv2

    video = Path(video).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_interval = 1
    if fps and source_fps > 0 and fps < source_fps:
        frame_interval = max(1, round(source_fps / fps))

    frames: list[FrameInput] = []
    source_idx = 0
    output_idx = 1
    while True:
        ok, image_bgr = cap.read()
        if not ok:
            break
        source_idx += 1
        if (source_idx - 1) % frame_interval != 0:
            continue

        frame_path = output_dir / f"{output_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), image_bgr)
        frames.append(FrameInput(frame_id=output_idx, path=frame_path))
        output_idx += 1

    cap.release()
    return frames


def iter_video_frames_cv2(video: Path, fps: int | None = None):
    """Yield decoded video frames as in-memory BGR arrays."""
    import cv2

    video = Path(video).resolve()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_interval = 1
    if fps and source_fps > 0 and fps < source_fps:
        frame_interval = max(1, round(source_fps / fps))

    source_idx = 0
    output_idx = 1
    try:
        while True:
            ok, image_bgr = cap.read()
            if not ok:
                break
            source_idx += 1
            if (source_idx - 1) % frame_interval != 0:
                continue

            yield FrameInput(frame_id=output_idx, image_bgr=image_bgr)
            output_idx += 1
    finally:
        cap.release()


def list_frames(frame_dir: Path) -> list[FrameInput]:
    """List already extracted frames with ids derived from their filenames."""
    frame_dir = Path(frame_dir).resolve()
    frames: list[FrameInput] = []
    for path in sorted(frame_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frame_id = extract_frame_id(path.name)
        if frame_id is not None:
            frames.append(FrameInput(frame_id=frame_id, path=path))
    return frames
