"""GT-free 2D pose features used by the upper-body correction model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from video2smplx.postprocessing.signlanguage import base_feature


COCO_IDS = np.asarray([0, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
SMPLX_IDS = np.asarray([15, 16, 17, 18, 19, 20, 21, 1, 2], dtype=np.int64)


def load_yolo(path: Path) -> dict[str, dict[int, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for sequence, video in data.get("videos", {}).items():
        output[sequence] = {int(frame["frame"]): frame for frame in video.get("frames", [])}
    return output


def get_yolo_frame(yolo: dict[str, dict[int, dict]], sequence: str, frame: int) -> dict | None:
    frames = yolo.get(sequence)
    if not frames:
        return None
    if frame in frames:
        return frames[frame]
    if frame - 1 in frames:
        return frames[frame - 1]
    if frame + 1 in frames:
        return frames[frame + 1]
    return None


def normalize_yolo_frame(frame: dict, min_conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(float(frame.get("image_width", 1)), 1.0)
    height = max(float(frame.get("image_height", 1)), 1.0)
    xy = np.asarray(frame["keypoints_xy"], dtype=np.float32).reshape(17, 2)[COCO_IDS]
    conf = np.asarray(frame["keypoints_conf"], dtype=np.float32).reshape(17)[COCO_IDS]
    xy_norm = np.stack([(xy[:, 0] / width - 0.5), (xy[:, 1] / height - 0.5)], axis=1)
    valid = (conf >= min_conf).astype(np.float32)
    return xy_norm, conf * valid, np.asarray([width, height], dtype=np.float32)


def weighted_similarity_2d(
    source: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    if float(weight.sum()) < 1e-6:
        projected = np.zeros_like(target, dtype=np.float32)
        return projected, np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weight = weight / weight.sum()
    src_mean = (source * weight[:, None]).sum(axis=0)
    tgt_mean = (target * weight[:, None]).sum(axis=0)
    src = source - src_mean
    tgt = target - tgt_mean
    cov = src.T @ (tgt * weight[:, None])
    u, singular, vt = np.linalg.svd(cov)
    sign = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ np.diag([1.0, sign]) @ vt
    denom = float((weight[:, None] * src * src).sum())
    scale = float(np.sum(singular * np.asarray([1.0, sign])) / denom) if denom > 1e-9 else 1.0
    projected = scale * src @ rotation + tgt_mean
    angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    geom = np.asarray(
        [
            scale,
            np.cos(angle),
            np.sin(angle),
            tgt_mean[0] - src_mean[0],
            tgt_mean[1] - src_mean[1],
        ],
        dtype=np.float32,
    )
    return projected.astype(np.float32), geom


def guided_feature(
    person: dict,
    pred_joints: np.ndarray,
    yolo_frame: dict,
    feature_preset: str,
    min_conf: float,
) -> np.ndarray:
    yolo_xy, yolo_conf, image_size = normalize_yolo_frame(yolo_frame, min_conf)
    pred = pred_joints[SMPLX_IDS].astype(np.float32)
    root = pred_joints[0].astype(np.float32)
    pred_rooted = pred - root[None]
    body_scale = float(np.linalg.norm(pred_joints[16] - pred_joints[17]))
    if body_scale < 1e-6:
        body_scale = 1.0
    pred_2d_source = pred_rooted[:, :2] / body_scale
    projected, geom = weighted_similarity_2d(pred_2d_source, yolo_xy, yolo_conf)
    residual = (yolo_xy - projected) * yolo_conf[:, None]
    conf_stats = np.asarray(
        [yolo_conf.mean(), yolo_conf.min(), yolo_conf.max(), float((yolo_conf > 0).mean())],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            base_feature(person, feature_preset),
            yolo_xy.reshape(-1),
            yolo_conf.reshape(-1),
            pred_rooted.reshape(-1) / body_scale,
            projected.reshape(-1),
            residual.reshape(-1),
            geom,
            conf_stats,
            image_size / 1000.0,
        ],
        axis=0,
    ).astype(np.float32)
