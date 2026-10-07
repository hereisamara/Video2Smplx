#!/usr/bin/env python3
"""Evaluate SignLanguage SMPL-X predictions with MPJPE/MPVPE and PA metrics.

The default paths match the server layout:

    datasets/videos/SignLanguage/SignLanguage_S*/SignLanguage_S*.mp4
    datasets/annotations/SignLanguage/{keypoint_annotation,smplx_annotation}.json
    outputs_fusion_signlanguage/SignLanguage_S*/fused_params/*_params.pkl

Predicted video frames are paired to the 30 FPS annotations by nearest
timestamp. The geometry forward pass is a NumPy implementation of SMPL-X LBS,
so the script only needs NumPy plus ffprobe and the licensed SMPL-X NPZ model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mmap
import pickle
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ANNOTATION_FPS = 30.0
SEQUENCE_RE = re.compile(r"^SignLanguage_S(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Dataset/run root.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Default: <root>/datasets/videos/SignLanguage",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Default: <root>/datasets/annotations/SignLanguage",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=None,
        help="Default: <root>/outputs_fusion_signlanguage",
    )
    parser.add_argument("--params-subdir", default="fused_params")
    parser.add_argument("--model-path", type=Path, required=True, help="SMPLX_*.npz or SMPLX_*.pkl model file.")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["all"],
        help="Sequence numbers/names, for example: 2 3 SignLanguage_S4. Default: all predicted sequences.",
    )
    parser.add_argument("--annotation-fps", type=float, default=ANNOTATION_FPS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <pred-root>/evaluation/<model-stem>_<params-subdir>",
    )
    parser.add_argument(
        "--strict-frame-count",
        action="store_true",
        help="Fail if the number of prediction PKLs differs from video frame count.",
    )
    parser.add_argument(
        "--skip-bad-frames",
        action="store_true",
        help="Deprecated: bad prediction frames are skipped by default; kept for compatibility.",
    )
    parser.add_argument(
        "--strict-pred",
        action="store_true",
        help="Fail if a prediction PKL is empty, malformed, or unreadable.",
    )
    parser.add_argument(
        "--strict-gt",
        action="store_true",
        help="Fail if any paired frame is missing from smplx_annotation.json.",
    )
    return parser.parse_args()


def sequence_sort_key(path_or_name: Path | str) -> tuple[int, str]:
    name = path_or_name.name if isinstance(path_or_name, Path) else path_or_name
    match = SEQUENCE_RE.match(name)
    return (int(match.group(1)) if match else 10**9, name)


def normalize_sequence_name(value: str) -> str:
    if value.isdigit():
        return f"SignLanguage_S{int(value)}"
    return value


def discover_sequences(pred_root: Path, params_subdir: str) -> list[str]:
    sequences = []
    for directory in pred_root.glob("SignLanguage_S*"):
        params_dir = directory / params_subdir
        if directory.is_dir() and params_dir.is_dir() and any(params_dir.glob("*_params.pkl")):
            sequences.append(directory.name)
    return sorted(sequences, key=sequence_sort_key)


def describe_prediction_root(pred_root: Path, params_subdir: str, limit: int = 8) -> str:
    if not pred_root.exists():
        return f"{pred_root} does not exist"
    sequence_dirs = sorted(
        [directory for directory in pred_root.glob("SignLanguage_S*") if directory.is_dir()],
        key=sequence_sort_key,
    )
    if not sequence_dirs:
        return f"{pred_root} exists, but contains no SignLanguage_S* directories"

    examples = []
    empty_examples = []
    nonempty = 0
    for directory in sequence_dirs[:limit]:
        subdirs = sorted(child.name for child in directory.iterdir() if child.is_dir())
        examples.append(f"{directory.name}: {subdirs[:10]}")
    for directory in sequence_dirs:
        params_dir = directory / params_subdir
        if params_dir.is_dir() and any(params_dir.glob("*_params.pkl")):
            nonempty += 1
        elif params_dir.is_dir() and len(empty_examples) < limit:
            empty_examples.append(directory.name)
    empty_message = f" Empty '{params_subdir}' examples: {empty_examples}." if empty_examples else ""
    return (
        f"{pred_root} contains {len(sequence_dirs)} SignLanguage_S* directories, "
        f"but {nonempty} contain '{params_subdir}/*_params.pkl'. Example sequence subdirs: "
        + "; ".join(examples)
        + empty_message
    )


def load_image_index(keypoint_path: Path) -> dict[str, list[dict]]:
    """Read only the `images` prefix of the large keypoint JSON."""
    markers = (b', "annotations":', b',"annotations":')
    chunks: list[bytes] = []
    tail = b""
    with keypoint_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"Could not find annotations marker in {keypoint_path}")
            chunks.append(chunk)
            joined_tail = tail + chunk
            positions = [joined_tail.find(marker) for marker in markers]
            positions = [position for position in positions if position >= 0]
            if positions:
                local_pos = min(positions)
                absolute_pos = sum(map(len, chunks[:-1])) - len(tail) + local_pos
                prefix = b"".join(chunks)[:absolute_pos] + b"}"
                break
            tail = joined_tail[-max(map(len, markers)) :]

    images = json.loads(prefix)["images"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for image in images:
        sequence = image["file_name"].split("/", 1)[0]
        grouped[sequence].append(image)
    for sequence in grouped:
        grouped[sequence].sort(key=lambda item: item["file_name"])
    return dict(grouped)


def find_ffprobe() -> str:
    candidates = [shutil.which("ffprobe"), "/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("ffprobe is required to read video FPS/frame count")


def video_metadata(video_path: Path, ffprobe: str) -> tuple[float, int]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    frames = int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else -1
    return numerator / denominator, frames


def nearest_annotation_index(prediction_index: int, video_fps: float, annotation_fps: float) -> int:
    # Half-up rounding avoids Python banker's rounding at exact timestamp ties.
    return math.floor(prediction_index * annotation_fps / video_fps + 0.5)


def load_ground_truth_records(
    smplx_path: Path,
    wanted_ids: set[int],
    *,
    allow_missing: bool = False,
) -> tuple[dict[int, dict], list[int]]:
    """Extract selected top-level records without loading the full SMPL-X JSON."""
    records: dict[int, dict] = {}
    pattern = re.compile(rb'"([0-9]+)"\s*:\s*\{\s*"smplx_param"')
    decoder = json.JSONDecoder()
    with smplx_path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for match in pattern.finditer(mapped):
                frame_id = int(match.group(1))
                if frame_id not in wanted_ids:
                    continue
                colon = mapped.find(b":", match.start(), match.end())
                # A record is roughly 4 KB; 64 KB leaves margin for formatting.
                text = mapped[colon + 1 : colon + 1 + 65536].decode("ascii").lstrip()
                record, _ = decoder.raw_decode(text)
                records[frame_id] = record
                if len(records) == len(wanted_ids):
                    break
        finally:
            mapped.close()
    missing = sorted(wanted_ids - records.keys())
    if missing and not allow_missing:
        raise KeyError(f"Missing {len(missing)} SMPL-X records; first IDs: {missing[:10]}")
    return records, missing


def one_person(prediction_path: Path) -> dict:
    with prediction_path.open("rb") as handle:
        prediction = pickle.load(handle)
    if not isinstance(prediction, list) or len(prediction) != 1:
        count = len(prediction) if hasattr(prediction, "__len__") else "unknown"
        raise ValueError(f"Expected exactly one predicted person in {prediction_path}, got {count}")
    return prediction[0]


def load_prediction_batch(
    batch_items: list[dict],
    *,
    strict_pred: bool,
    skipped_frames: list[dict],
    warnings: list[str],
) -> tuple[list[dict], list[dict]]:
    kept_items = []
    predicted_params = []
    for item in batch_items:
        try:
            predicted_params.append(one_person(item["file"]))
            kept_items.append(item)
        except Exception as exc:
            if strict_pred:
                raise
            skipped_frames.append(
                {
                    "sequence": item["sequence"],
                    "prediction_frame": item["prediction_frame"],
                    "file": str(item["file"]),
                    "reason": str(exc),
                }
            )
            warnings.append(
                f"{item['sequence']} frame {item['prediction_frame']}: skipped prediction: {exc}"
            )
    return kept_items, predicted_params


class NumpySMPLX:
    """Minimal SMPL-X linear-blend-skinning forward pass for evaluation."""

    def __init__(self, model_path: Path, num_betas: int = 10, num_expression: int = 10):
        data = load_smplx_model_data(model_path)
        self.v_template = as_numpy(data["v_template"], dtype=np.float32)
        all_shape_dirs = as_numpy(data["shapedirs"], dtype=np.float32)
        self.shape_dirs = np.concatenate(
            [all_shape_dirs[:, :, :num_betas], all_shape_dirs[:, :, 300 : 300 + num_expression]],
            axis=2,
        )
        self.pose_dirs = as_numpy(data["posedirs"], dtype=np.float32).reshape(-1, 486).T
        self.joint_regressor = as_numpy(data["J_regressor"], dtype=np.float32)
        self.weights = as_numpy(data["weights"], dtype=np.float32)
        self.parents = as_numpy(data["kintree_table"], dtype=np.int64)[0]
        self.parents[0] = -1
        self.num_betas = num_betas
        self.num_expression = num_expression

        dominant_joint = np.argmax(self.weights, axis=1)
        expression_dirs = self.shape_dirs[:, :, num_betas:]
        self.face_vertex_indices = np.flatnonzero(np.any(expression_dirs != 0, axis=(1, 2)))
        self.visible_upper_joint_indices = np.asarray(
            [
                0,
                3,
                6,
                9,
                12,
                15,
                13,
                14,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                *range(25, 55),
            ],
            dtype=np.int64,
        )
        self.visible_upper_vertex_indices = np.flatnonzero(
            np.isin(dominant_joint, self.visible_upper_joint_indices)
        )
        self.dexavatar_upper_minus_face_vertex_indices = np.setdiff1d(
            self.visible_upper_vertex_indices,
            self.face_vertex_indices,
            assume_unique=True,
        )
        self.left_hand_vertex_indices = np.flatnonzero(
            np.isin(dominant_joint, np.asarray([20, *range(25, 40)]))
        )
        self.right_hand_vertex_indices = np.flatnonzero(
            np.isin(dominant_joint, np.asarray([21, *range(40, 55)]))
        )

    @staticmethod
    def rodrigues(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, 3)
        theta = np.linalg.norm(vectors, axis=1)
        result = np.repeat(np.eye(3, dtype=np.float32)[None], len(vectors), axis=0)
        active = theta > 1e-8
        axes = vectors[active] / theta[active, None]
        skew = np.zeros((len(axes), 3, 3), dtype=np.float32)
        skew[:, 0, 1] = -axes[:, 2]
        skew[:, 0, 2] = axes[:, 1]
        skew[:, 1, 0] = axes[:, 2]
        skew[:, 1, 2] = -axes[:, 0]
        skew[:, 2, 0] = -axes[:, 1]
        skew[:, 2, 1] = axes[:, 0]
        angles = theta[active]
        result[active] = (
            np.eye(3, dtype=np.float32)[None]
            + np.sin(angles)[:, None, None] * skew
            + (1.0 - np.cos(angles))[:, None, None] * (skew @ skew)
        )
        return result

    def forward(self, params: list[dict], predicted: bool) -> tuple[np.ndarray, np.ndarray]:
        root, body, jaw, left, right, betas, expression, translation = self._split_params(params, predicted)

        batch = len(params)
        eyes = np.zeros((batch, 6), dtype=np.float32)
        full_pose = np.concatenate([root, body, jaw, eyes, left, right], axis=1).astype(np.float32)
        if full_pose.shape[1] != 55 * 3:
            raise ValueError(f"Expected 165 full-pose values, got {full_pose.shape[1]}")
        rotations = self.rodrigues(full_pose.reshape(-1, 3)).reshape(batch, 55, 3, 3)

        coefficients = np.concatenate([betas, expression], axis=1).astype(np.float32)
        shaped = self.v_template[None] + np.einsum(
            "bl,vcl->bvc", coefficients, self.shape_dirs, optimize=True
        )
        joints = np.einsum("jv,bvc->bjc", self.joint_regressor, shaped, optimize=True)
        pose_feature = (rotations[:, 1:] - np.eye(3, dtype=np.float32)).reshape(batch, -1)
        pose_offsets = (pose_feature @ self.pose_dirs).reshape(batch, -1, 3)
        posed = shaped + pose_offsets

        transforms, posed_joints = self._joint_transforms(joints, rotations)

        rest_homogeneous = np.concatenate(
            [joints, np.zeros((batch, 55, 1), dtype=np.float32)], axis=2
        )[..., None]
        correction = transforms @ rest_homogeneous
        relative_transforms = transforms.copy()
        relative_transforms[:, :, :, 3] -= correction[:, :, :, 0]

        vertex_transforms = np.einsum(
            "vj,bjkl->bvkl", self.weights, relative_transforms, optimize=True
        )
        posed_homogeneous = np.concatenate(
            [posed, np.ones((batch, posed.shape[1], 1), dtype=np.float32)], axis=2
        )
        vertices = (vertex_transforms @ posed_homogeneous[..., None])[:, :, :3, 0]
        translation = np.asarray(translation, dtype=np.float32)
        return vertices + translation[:, None], posed_joints + translation[:, None]

    def forward_joints(self, params: list[dict], predicted: bool) -> np.ndarray:
        """Forward only SMPL-X joints, skipping pose blend shapes and vertices."""
        root, body, jaw, left, right, betas, expression, translation = self._split_params(params, predicted)

        batch = len(params)
        eyes = np.zeros((batch, 6), dtype=np.float32)
        full_pose = np.concatenate([root, body, jaw, eyes, left, right], axis=1).astype(np.float32)
        if full_pose.shape[1] != 55 * 3:
            raise ValueError(f"Expected 165 full-pose values, got {full_pose.shape[1]}")
        rotations = self.rodrigues(full_pose.reshape(-1, 3)).reshape(batch, 55, 3, 3)

        coefficients = np.concatenate([betas, expression], axis=1).astype(np.float32)
        shaped = self.v_template[None] + np.einsum(
            "bl,vcl->bvc", coefficients, self.shape_dirs, optimize=True
        )
        joints = np.einsum("jv,bvc->bjc", self.joint_regressor, shaped, optimize=True)
        _, posed_joints = self._joint_transforms(joints, rotations)
        translation = np.asarray(translation, dtype=np.float32)
        return posed_joints + translation[:, None]

    def _split_params(self, params: list[dict], predicted: bool) -> tuple[np.ndarray, ...]:
        if predicted:
            root = np.stack([x["global_orient"] for x in params])
            body = np.stack([x["body_pose"] for x in params])
            jaw = np.stack([x["jaw_pose"] for x in params])
            left = np.stack([x["left_hand_pose"] for x in params])
            right = np.stack([x["right_hand_pose"] for x in params])
            betas = np.stack([x["betas"] for x in params])[:, : self.num_betas]
            expression = np.stack([x["expression"] for x in params])[:, : self.num_expression]
            translation = np.stack([x["transl"] for x in params])
        else:
            root = np.stack([x["root_pose"] for x in params])
            body = np.stack([x["body_pose"] for x in params])
            jaw = np.stack([x["jaw_pose"] for x in params])
            left = np.stack([x["lhand_pose"] for x in params])
            right = np.stack([x["rhand_pose"] for x in params])
            betas = np.stack([x["shape"] for x in params])[:, : self.num_betas]
            expression = np.stack([x["expr"] for x in params])[:, : self.num_expression]
            translation = np.stack([x["trans"] for x in params])
        return root, body, jaw, left, right, betas, expression, translation

    def _joint_transforms(self, joints: np.ndarray, rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = joints.shape[0]
        relative_joints = joints.copy()
        relative_joints[:, 1:] -= joints[:, self.parents[1:]]
        local = np.zeros((batch, 55, 4, 4), dtype=np.float32)
        local[:, :, :3, :3] = rotations
        local[:, :, :3, 3] = relative_joints
        local[:, :, 3, 3] = 1.0
        transforms = np.empty_like(local)
        transforms[:, 0] = local[:, 0]
        for joint_index in range(1, 55):
            transforms[:, joint_index] = transforms[:, self.parents[joint_index]] @ local[:, joint_index]
        posed_joints = transforms[:, :, :3, 3]
        return transforms, posed_joints


def as_numpy(value: object, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Convert SMPL-X npz/pkl values, sparse matrices, and chumpy arrays to ndarray."""
    if hasattr(value, "toarray"):
        array = value.toarray()
    elif hasattr(value, "todense"):
        array = value.todense()
    elif hasattr(value, "r"):
        array = value.r
    else:
        array = value
    return np.asarray(array, dtype=dtype)


def load_smplx_model_data(model_path: Path) -> dict:
    suffix = model_path.suffix.lower()
    if suffix == ".npz":
        return dict(np.load(model_path, allow_pickle=True))
    if suffix == ".pkl":
        with model_path.open("rb") as handle:
            return pickle.load(handle, encoding="latin1")
    raise ValueError(f"Unsupported model file {model_path}; expected .npz or .pkl")


def procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_mean = predicted.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_centered = predicted - pred_mean
    target_centered = target - target_mean
    covariance = pred_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    denom = np.sum(pred_centered**2)
    scale = float(np.sum(singular_values * sign) / denom) if denom > 1e-12 else 1.0
    return scale * pred_centered @ rotation + target_mean


def mean_root_error(
    predicted: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray | list[int] | slice,
    pred_root: np.ndarray,
    target_root: np.ndarray,
) -> float:
    pred_local = predicted[indices] - pred_root
    target_local = target[indices] - target_root
    return float(np.mean(np.linalg.norm(pred_local - target_local, axis=1)))


def mean_pa_error(predicted: np.ndarray, target: np.ndarray, indices: np.ndarray | list[int] | slice) -> float:
    aligned = procrustes_align(predicted[indices], target[indices])
    return float(np.mean(np.linalg.norm(aligned - target[indices], axis=1)))


def geometry_metrics(
    predicted_vertices: np.ndarray,
    predicted_joints: np.ndarray,
    target_vertices: np.ndarray,
    target_joints: np.ndarray,
    model: NumpySMPLX,
) -> dict[str, float]:
    mm = 1000.0
    root = 0
    body_indices = np.arange(0, 22)
    hand_indices = np.arange(25, 55)
    left_hand_indices = np.arange(25, 40)
    right_hand_indices = np.arange(40, 55)

    raw_joint_error = np.linalg.norm(predicted_joints - target_joints, axis=1)
    raw_vertex_error = np.linalg.norm(predicted_vertices - target_vertices, axis=1)

    predicted_joints_root = predicted_joints - predicted_joints[root]
    target_joints_root = target_joints - target_joints[root]
    predicted_vertices_root = predicted_vertices - predicted_joints[root]
    target_vertices_root = target_vertices - target_joints[root]
    joint_error = np.linalg.norm(predicted_joints_root - target_joints_root, axis=1)
    vertex_error = np.linalg.norm(predicted_vertices_root - target_vertices_root, axis=1)

    visible_j = model.visible_upper_joint_indices
    visible_v = model.visible_upper_vertex_indices
    face_v = model.face_vertex_indices
    dex_upper_minus_face_v = model.dexavatar_upper_minus_face_vertex_indices

    hand_joint_errors = []
    pa_hand_joint_errors = []
    hand_vertex_errors = []
    pa_hand_vertex_errors = []
    hand_specs = (
        (20, np.arange(25, 40), model.left_hand_vertex_indices),
        (21, np.arange(40, 55), model.right_hand_vertex_indices),
    )
    for wrist, joint_indices, vertex_indices in hand_specs:
        hand_joint_errors.append(
            np.linalg.norm(
                (predicted_joints[joint_indices] - predicted_joints[wrist])
                - (target_joints[joint_indices] - target_joints[wrist]),
                axis=1,
            )
        )
        pred_skeleton = np.concatenate([predicted_joints[wrist : wrist + 1], predicted_joints[joint_indices]])
        target_skeleton = np.concatenate([target_joints[wrist : wrist + 1], target_joints[joint_indices]])
        aligned_skeleton = procrustes_align(pred_skeleton, target_skeleton)
        pa_hand_joint_errors.append(np.linalg.norm(aligned_skeleton[1:] - target_skeleton[1:], axis=1))

        hand_vertex_errors.append(
            np.linalg.norm(
                (predicted_vertices[vertex_indices] - predicted_joints[wrist])
                - (target_vertices[vertex_indices] - target_joints[wrist]),
                axis=1,
            )
        )
        aligned_vertices = procrustes_align(predicted_vertices[vertex_indices], target_vertices[vertex_indices])
        pa_hand_vertex_errors.append(np.linalg.norm(aligned_vertices - target_vertices[vertex_indices], axis=1))

    face_head_error = np.linalg.norm(
        (predicted_vertices[face_v] - predicted_joints[15])
        - (target_vertices[face_v] - target_joints[15]),
        axis=1,
    )

    metrics = {
        "mpjpe_mm": float(np.mean(joint_error) * mm),
        "pa_mpjpe_mm": mean_pa_error(predicted_joints, target_joints, np.arange(55)) * mm,
        "mpvpe_mm": float(np.mean(vertex_error) * mm),
        "pa_mpvpe_mm": mean_pa_error(predicted_vertices, target_vertices, np.arange(len(predicted_vertices))) * mm,
        "body_mpjpe_mm": float(np.mean(joint_error[body_indices]) * mm),
        "body_pa_mpjpe_mm": mean_pa_error(predicted_joints, target_joints, body_indices) * mm,
        "hand_mpjpe_mm": float(np.mean(joint_error[hand_indices]) * mm),
        "hand_pa_mpjpe_mm": mean_pa_error(predicted_joints, target_joints, hand_indices) * mm,
        "left_hand_mpjpe_mm": float(np.mean(joint_error[left_hand_indices]) * mm),
        "left_hand_pa_mpjpe_mm": mean_pa_error(predicted_joints, target_joints, left_hand_indices) * mm,
        "right_hand_mpjpe_mm": float(np.mean(joint_error[right_hand_indices]) * mm),
        "right_hand_pa_mpjpe_mm": mean_pa_error(predicted_joints, target_joints, right_hand_indices) * mm,
        "visible_upper_mpjpe_mm": mean_root_error(
            predicted_joints, target_joints, visible_j, predicted_joints[root], target_joints[root]
        )
        * mm,
        "visible_upper_pa_mpjpe_mm": mean_pa_error(predicted_joints, target_joints, visible_j) * mm,
        "visible_upper_mpvpe_mm": mean_root_error(
            predicted_vertices, target_vertices, visible_v, predicted_joints[root], target_joints[root]
        )
        * mm,
        "visible_upper_pa_mpvpe_mm": mean_pa_error(predicted_vertices, target_vertices, visible_v) * mm,
        "hands_wrist_mpjpe_mm": float(np.mean(np.concatenate(hand_joint_errors)) * mm),
        "hands_pa_mpjpe_mm": float(np.mean(np.concatenate(pa_hand_joint_errors)) * mm),
        "hands_wrist_mpvpe_mm": float(np.mean(np.concatenate(hand_vertex_errors)) * mm),
        "hands_pa_mpvpe_mm": float(np.mean(np.concatenate(pa_hand_vertex_errors)) * mm),
        "face_head_mpvpe_mm": float(np.mean(face_head_error) * mm),
        "face_pa_mpvpe_mm": mean_pa_error(predicted_vertices, target_vertices, face_v) * mm,
        "raw_mpjpe_mm": float(np.mean(raw_joint_error) * mm),
        "raw_mpvpe_mm": float(np.mean(raw_vertex_error) * mm),
        "dex_ubody_minus_face_tr_v2v_mm": float(np.mean(raw_vertex_error[dex_upper_minus_face_v]) * mm),
        "dex_left_hand_tr_v2v_mm": float(np.mean(raw_vertex_error[model.left_hand_vertex_indices]) * mm),
        "dex_right_hand_tr_v2v_mm": float(np.mean(raw_vertex_error[model.right_hand_vertex_indices]) * mm),
    }
    return metrics


METRICS = (
    "mpjpe_mm",
    "pa_mpjpe_mm",
    "mpvpe_mm",
    "pa_mpvpe_mm",
    "body_mpjpe_mm",
    "body_pa_mpjpe_mm",
    "hand_mpjpe_mm",
    "hand_pa_mpjpe_mm",
    "left_hand_mpjpe_mm",
    "left_hand_pa_mpjpe_mm",
    "right_hand_mpjpe_mm",
    "right_hand_pa_mpjpe_mm",
    "visible_upper_mpjpe_mm",
    "visible_upper_pa_mpjpe_mm",
    "visible_upper_mpvpe_mm",
    "visible_upper_pa_mpvpe_mm",
    "hands_wrist_mpjpe_mm",
    "hands_pa_mpjpe_mm",
    "hands_wrist_mpvpe_mm",
    "hands_pa_mpvpe_mm",
    "face_head_mpvpe_mm",
    "face_pa_mpvpe_mm",
    "raw_mpjpe_mm",
    "raw_mpvpe_mm",
    "dex_ubody_minus_face_tr_v2v_mm",
    "dex_left_hand_tr_v2v_mm",
    "dex_right_hand_tr_v2v_mm",
)


def summarize(rows: list[dict], sequence: str) -> dict:
    output: dict[str, str | int | float] = {"sequence": sequence, "frames": len(rows)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        output[f"{metric}_mean"] = float(np.mean(values))
        output[f"{metric}_median"] = float(np.median(values))
        output[f"{metric}_p95"] = float(np.percentile(values, 95))
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction_frame_number(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def build_sequence_items(
    sequence: str,
    args: argparse.Namespace,
    video_dir: Path,
    pred_root: Path,
    image_index: dict[str, list[dict]],
    ffprobe: str,
    warnings: list[str],
) -> tuple[list[dict], dict]:
    video_path = video_dir / sequence / f"{sequence}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"{sequence}: missing video {video_path}")
    if sequence not in image_index:
        raise KeyError(f"{sequence}: missing annotation image index")

    fps, video_frames = video_metadata(video_path, ffprobe)
    files = sorted((pred_root / sequence / args.params_subdir).glob("*_params.pkl"), key=prediction_frame_number)
    if not files:
        raise FileNotFoundError(f"{sequence}: no prediction PKLs in {pred_root / sequence / args.params_subdir}")
    if args.strict_frame_count and video_frames >= 0 and len(files) != video_frames:
        raise ValueError(f"{sequence}: {len(files)} predictions but {video_frames} video frames")
    if video_frames >= 0 and len(files) != video_frames:
        warnings.append(f"{sequence}: {len(files)} predictions, video has {video_frames} frames")

    annotation_images = image_index[sequence]
    items = []
    for file in files:
        prediction_index = prediction_frame_number(file) - 1
        annotation_index = nearest_annotation_index(prediction_index, fps, args.annotation_fps)
        if annotation_index >= len(annotation_images):
            message = f"{sequence}: frame {prediction_index + 1} maps to missing annotation {annotation_index + 1}"
            if args.skip_bad_frames:
                warnings.append(message)
                continue
            raise IndexError(message)
        image = annotation_images[annotation_index]
        items.append(
            {
                "sequence": sequence,
                "file": file,
                "prediction_frame": prediction_index + 1,
                "annotation_frame": annotation_index + 1,
                "ground_truth_id": int(image["id"]),
                "annotation_file": image["file_name"],
            }
        )

    metadata = {
        "video_path": str(video_path),
        "video_fps": fps,
        "video_frames": video_frames,
        "prediction_frames": len(files),
        "evaluated_frames": len(items),
        "annotation_frames": len(annotation_images),
        "first_ground_truth_id": items[0]["ground_truth_id"] if items else None,
        "last_ground_truth_id": items[-1]["ground_truth_id"] if items else None,
    }
    return items, metadata


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    video_dir = (args.video_dir or root / "datasets" / "videos" / "SignLanguage").resolve()
    annotation_dir = (args.annotation_dir or root / "datasets" / "annotations" / "SignLanguage").resolve()
    pred_root = (args.pred_root or root / "outputs_fusion_signlanguage").resolve()
    model_path = args.model_path.resolve()
    output_dir = (
        args.output_dir
        or pred_root / "evaluation" / f"{model_path.stem.lower()}_{args.params_subdir}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    keypoint_path = annotation_dir / "keypoint_annotation.json"
    smplx_path = annotation_dir / "smplx_annotation.json"
    image_index = load_image_index(keypoint_path)
    ffprobe = find_ffprobe()

    if args.sequences == ["all"]:
        sequences = discover_sequences(pred_root, args.params_subdir)
    else:
        sequences = sorted([normalize_sequence_name(x) for x in args.sequences], key=sequence_sort_key)
    if not sequences:
        raise ValueError(describe_prediction_root(pred_root, args.params_subdir))

    warnings: list[str] = []
    sequence_data: dict[str, list[dict]] = {}
    metadata: dict[str, dict] = {}
    wanted_ids: set[int] = set()
    for sequence in sequences:
        try:
            items, sequence_metadata = build_sequence_items(
                sequence, args, video_dir, pred_root, image_index, ffprobe, warnings
            )
        except Exception as exc:
            if not args.skip_bad_frames:
                raise
            warnings.append(f"{sequence}: skipped sequence setup error: {exc}")
            continue
        sequence_data[sequence] = items
        metadata[sequence] = sequence_metadata
        wanted_ids.update(item["ground_truth_id"] for item in items)

    if not wanted_ids:
        raise ValueError("No frame pairs available for evaluation")

    print(f"Loading {len(wanted_ids)} GT SMPL-X records from {smplx_path}", file=sys.stderr)
    ground_truth, missing_gt_ids = load_ground_truth_records(
        smplx_path,
        wanted_ids,
        allow_missing=not args.strict_gt,
    )
    skipped_frames: list[dict] = []
    if missing_gt_ids:
        missing_gt_set = set(missing_gt_ids)
        warnings.append(
            f"Skipping {len(missing_gt_ids)} frame pairs because their SMPL-X GT records are missing; "
            f"first IDs: {missing_gt_ids[:10]}"
        )
        for sequence, items in list(sequence_data.items()):
            kept_items = []
            for item in items:
                if item["ground_truth_id"] in missing_gt_set:
                    skipped_frames.append(
                        {
                            "sequence": sequence,
                            "prediction_frame": item["prediction_frame"],
                            "file": str(item["file"]),
                            "reason": f"missing GT SMPL-X record {item['ground_truth_id']}",
                        }
                    )
                else:
                    kept_items.append(item)
            sequence_data[sequence] = kept_items
            if sequence in metadata:
                metadata[sequence]["evaluated_frames"] = len(kept_items)
                metadata[sequence]["missing_gt_frames"] = len(items) - len(kept_items)
        sequence_data = {sequence: items for sequence, items in sequence_data.items() if items}
        if not sequence_data:
            raise ValueError("No frame pairs remain after removing missing GT records")
    model = NumpySMPLX(model_path)

    frame_rows: list[dict] = []
    summary_rows: list[dict] = []
    for sequence, items in sequence_data.items():
        sequence_rows = []
        print(f"Evaluating {sequence}: {len(items)} frames", file=sys.stderr)
        for start in range(0, len(items), args.batch_size):
            batch_items = items[start : start + args.batch_size]
            try:
                batch_items, predicted_params = load_prediction_batch(
                    batch_items,
                    strict_pred=args.strict_pred,
                    skipped_frames=skipped_frames,
                    warnings=warnings,
                )
                if not batch_items:
                    continue
                target_params = [ground_truth[item["ground_truth_id"]]["smplx_param"] for item in batch_items]
                predicted_vertices, predicted_joints = model.forward(predicted_params, predicted=True)
                target_vertices, target_joints = model.forward(target_params, predicted=False)
            except Exception as exc:
                if args.strict_pred:
                    raise
                for item in batch_items:
                    skipped_frames.append(
                        {
                            "sequence": sequence,
                            "prediction_frame": item["prediction_frame"],
                            "file": str(item["file"]),
                            "reason": str(exc),
                        }
                    )
                warnings.append(f"{sequence}: skipped batch at item {start}: {exc}")
                continue

            for index, item in enumerate(batch_items):
                metrics = geometry_metrics(
                    predicted_vertices[index],
                    predicted_joints[index],
                    target_vertices[index],
                    target_joints[index],
                    model,
                )
                row = {
                    "sequence": sequence,
                    "prediction_frame": item["prediction_frame"],
                    "annotation_frame": item["annotation_frame"],
                    "ground_truth_id": item["ground_truth_id"],
                    "annotation_file": item["annotation_file"],
                    **metrics,
                }
                sequence_rows.append(row)
                frame_rows.append(row)
        if sequence_rows:
            summary_rows.append(summarize(sequence_rows, sequence))

    summary_rows.append(summarize(frame_rows, "ALL"))
    write_csv(output_dir / "geometry_frame_metrics.csv", frame_rows)
    write_csv(output_dir / "geometry_summary.csv", summary_rows)
    write_csv(output_dir / "skipped_frames.csv", skipped_frames)

    report = {
        "paths": {
            "root": str(root),
            "video_dir": str(video_dir),
            "annotation_dir": str(annotation_dir),
            "pred_root": str(pred_root),
            "params_subdir": args.params_subdir,
            "model_path": str(model_path),
            "output_dir": str(output_dir),
        },
        "metric_notes": {
            "mpjpe_mm": "pelvis/root translation aligned over all 55 SMPL-X joints",
            "pa_mpjpe_mm": "similarity Procrustes aligned over all 55 SMPL-X joints",
            "mpvpe_mm": "pelvis/root translation aligned over all SMPL-X vertices",
            "pa_mpvpe_mm": "similarity Procrustes aligned over all SMPL-X vertices",
            "body_mpjpe_mm": "root-aligned joints 0-21",
            "body_pa_mpjpe_mm": "PA over joints 0-21",
            "hand_mpjpe_mm": "root-aligned articulated hand joints 25-54",
            "hand_pa_mpjpe_mm": "PA over articulated hand joints 25-54",
            "hands_wrist_mpjpe_mm": "each hand aligned to its own wrist before measuring finger joints",
            "hands_pa_mpjpe_mm": "PA fitted independently for each hand skeleton including wrist",
            "visible_upper_*": "upper-body/head/hand subset useful for cropped signing videos",
            "raw_*": "no translation, rotation, or scale alignment",
            "dex_*_tr_v2v_mm": (
                "DexAvatar-style regional raw vertex-to-vertex error; "
                "upper-body mask is visible upper-body vertices excluding expression/face vertices"
            ),
        },
        "joint_sets": {
            "all": "55 native SMPL-X joints",
            "body": "0-21",
            "left_hand": "25-39",
            "right_hand": "40-54",
            "hands": "25-54",
            "visible_upper": [int(x) for x in model.visible_upper_joint_indices],
        },
        "vertex_masks": {
            "visible_upper_count": int(len(model.visible_upper_vertex_indices)),
            "dex_ubody_minus_face_count": int(len(model.dexavatar_upper_minus_face_vertex_indices)),
            "face_count": int(len(model.face_vertex_indices)),
            "left_hand_count": int(len(model.left_hand_vertex_indices)),
            "right_hand_count": int(len(model.right_hand_vertex_indices)),
        },
        "temporal_alignment": {
            "method": "nearest annotation timestamp using half-up rounding",
            "annotation_fps": args.annotation_fps,
            "formula": "floor((prediction_frame - 1) * annotation_fps / video_fps + 0.5)",
        },
        "warnings": warnings,
        "skipped_frames": len(skipped_frames),
        "sequences": metadata,
        "summary": summary_rows,
    }
    with (output_dir / "geometry_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    fields = [
        "sequence",
        "frames",
        "mpjpe_mm_mean",
        "pa_mpjpe_mm_mean",
        "mpvpe_mm_mean",
        "pa_mpvpe_mm_mean",
        "body_mpjpe_mm_mean",
        "body_pa_mpjpe_mm_mean",
        "hand_mpjpe_mm_mean",
        "hand_pa_mpjpe_mm_mean",
        "dex_ubody_minus_face_tr_v2v_mm_mean",
        "dex_left_hand_tr_v2v_mm_mean",
        "dex_right_hand_tr_v2v_mm_mean",
    ]
    print(",".join(fields))
    for row in summary_rows:
        print(",".join(str(row[field]) for field in fields))
    print(f"\nWrote geometry evaluation to {output_dir}")


if __name__ == "__main__":
    main()
