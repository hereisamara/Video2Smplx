"""Small data contracts for the integrated in-process pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameInput:
    frame_id: int
    path: Path | None = None
    image_bgr: Any | None = None


@dataclass
class FramePrediction:
    frame: FrameInput
    body: list[dict[str, Any]]
    hands: dict[str, Any]
    face: dict[str, Any]
    fused: list[Any]
