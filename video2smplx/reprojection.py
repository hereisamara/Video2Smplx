"""Metrics for comparing a reference 2D pose with a rendered 2D pose.

The output videos produced by this repository use a visualization camera that
does not match the input camera.  Consequently, rendered landmarks are aligned
to the reference with a 2D similarity transform estimated from the shoulders
before residual pose error is measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# MediaPipe Pose landmark names and indices.
LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Torso-focused reporting set used when hands and lower-body motion are outside
# the evaluation scope. Shoulders remain alignment anchors rather than scores.
TORSO_LANDMARKS = (0, 13, 14, 23, 24)

# Shoulders are alignment anchors, so they are deliberately excluded from the
# scored joints. Dense facial landmarks are excluded because SMPL-X face quality
# is not represented by MediaPipe Pose's few face points.
SCORED_LANDMARKS = (
    0,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
)


@dataclass(frozen=True)
class Similarity2D:
    """A point transform ``y = scale * rotation @ x + translation``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation


def shoulder_similarity(source: np.ndarray, target: np.ndarray) -> Similarity2D:
    """Return the exact orientation/scale/translation mapping of shoulder pairs.

    ``source`` and ``target`` must be shaped ``(2, 2)`` in left/right order.
    A ValueError is raised for a degenerate shoulder pair.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != (2, 2) or target.shape != (2, 2):
        raise ValueError("source and target shoulders must both have shape (2, 2)")

    source_vector = source[1] - source[0]
    target_vector = target[1] - target[0]
    source_width = float(np.linalg.norm(source_vector))
    target_width = float(np.linalg.norm(target_vector))
    if source_width <= 1e-8 or target_width <= 1e-8:
        raise ValueError("cannot align a degenerate shoulder pair")

    source_angle = np.arctan2(source_vector[1], source_vector[0])
    target_angle = np.arctan2(target_vector[1], target_vector[0])
    angle = target_angle - source_angle
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    scale = target_width / source_width
    source_midpoint = source.mean(axis=0)
    target_midpoint = target.mean(axis=0)
    translation = target_midpoint - scale * (rotation @ source_midpoint)
    return Similarity2D(scale, rotation, translation)


def normalized_to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized x/y MediaPipe coordinates to image pixels."""

    points = np.asarray(landmarks, dtype=np.float64)
    return points * np.array((width, height), dtype=np.float64)


def compare_landmarks(
    reference_xy: np.ndarray,
    reference_visibility: np.ndarray,
    rendered_xy: np.ndarray,
    rendered_visibility: np.ndarray,
    visibility_threshold: float = 0.5,
    scored_landmarks: tuple[int, ...] = SCORED_LANDMARKS,
) -> dict | None:
    """Compare one frame after aligning rendered shoulders to reference shoulders.

    Coordinates must already be expressed in their respective image pixels.
    Returned normalized errors use reference shoulder width as the denominator.
    Only joints visible in both images are included. Returns ``None`` when the
    shoulders cannot be aligned or no scored joint is mutually visible.
    """

    reference_xy = np.asarray(reference_xy, dtype=np.float64)
    rendered_xy = np.asarray(rendered_xy, dtype=np.float64)
    reference_visibility = np.asarray(reference_visibility, dtype=np.float64)
    rendered_visibility = np.asarray(rendered_visibility, dtype=np.float64)
    anchors = np.array((LEFT_SHOULDER, RIGHT_SHOULDER))
    if np.any(reference_visibility[anchors] < visibility_threshold) or np.any(
        rendered_visibility[anchors] < visibility_threshold
    ):
        return None

    try:
        transform = shoulder_similarity(rendered_xy[anchors], reference_xy[anchors])
    except ValueError:
        return None
    aligned = transform.apply(rendered_xy)
    shoulder_width = float(np.linalg.norm(reference_xy[RIGHT_SHOULDER] - reference_xy[LEFT_SHOULDER]))

    candidate = np.asarray(scored_landmarks, dtype=np.int64)
    visible = (reference_visibility[candidate] >= visibility_threshold) & (
        rendered_visibility[candidate] >= visibility_threshold
    )
    joint_indices = candidate[visible]
    if len(joint_indices) == 0:
        return None

    errors_px = np.linalg.norm(aligned[joint_indices] - reference_xy[joint_indices], axis=1)
    errors_normalized = errors_px / shoulder_width
    return {
        "joint_indices": joint_indices,
        "errors_px": errors_px,
        "errors_normalized": errors_normalized,
        "aligned_rendered_xy": aligned,
        "shoulder_width_px": shoulder_width,
        "scale": transform.scale,
    }
