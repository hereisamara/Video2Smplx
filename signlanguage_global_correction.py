#!/usr/bin/env python3
"""Shared utilities for SignLanguage global-correction training scripts."""

from __future__ import annotations

import copy
import pickle
import re
import shutil
from pathlib import Path

import numpy as np

from evaluate_signlanguage_geometry import one_person, sequence_sort_key
from oracle_optimize_signlanguage_global import matrix_to_rotvec, rebuild_param_vector, rotvec_to_matrix


PARAM_VECTOR_KEYS = (
    "global_orient",
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "betas",
    "expression",
    "transl",
)

UPPER_BODY_SMPLX_JOINTS = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
UPPER_BODY_BODY_POSE_INDICES = tuple(joint - 1 for joint in UPPER_BODY_SMPLX_JOINTS)
SEQUENCE_RE = re.compile(r"^SignLanguage_S(\d+)$")


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


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)_params\.pkl$", path.name)
    if not match:
        raise ValueError(f"Cannot parse frame number from {path.name}")
    return int(match.group(1))


def list_prediction_files(pred_root: Path, sequence: str, params_subdir: str) -> list[Path]:
    params_dir = pred_root / sequence / params_subdir
    if not params_dir.is_dir():
        return []
    return sorted(params_dir.glob("*_params.pkl"), key=frame_number)


def copy_person(person: dict) -> dict:
    return {
        key: value.copy() if hasattr(value, "copy") else copy.deepcopy(value)
        for key, value in person.items()
    }


def write_person(path: Path, person: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump([person], handle)


def copy_sidecar_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file() and not path.name.endswith("_params.pkl"):
            shutil.copy2(path, target_dir / path.name)


def bbox_feature(person: dict) -> np.ndarray:
    xyxy = person.get("bbox_xyxy")
    xywh = person.get("bbox_xywh")
    if xyxy is None:
        xyxy_values = np.zeros(4, dtype=np.float32)
        xyxy_present = 0.0
    else:
        xyxy_values = np.asarray(xyxy, dtype=np.float32).reshape(-1)[:4]
        if xyxy_values.size < 4:
            xyxy_values = np.pad(xyxy_values, (0, 4 - xyxy_values.size))
        xyxy_present = 1.0
    if xywh is None:
        xywh_values = np.zeros(4, dtype=np.float32)
        xywh_present = 0.0
    else:
        xywh_values = np.asarray(xywh, dtype=np.float32).reshape(-1)[:4]
        if xywh_values.size < 4:
            xywh_values = np.pad(xywh_values, (0, 4 - xywh_values.size))
        xywh_present = 1.0
    return np.concatenate(
        [xyxy_values, xywh_values, np.asarray([xyxy_present, xywh_present], dtype=np.float32)]
    )


def param_vector(person: dict) -> np.ndarray:
    if all(key in person for key in PARAM_VECTOR_KEYS):
        return np.concatenate(
            [np.asarray(person[key], dtype=np.float32).reshape(-1) for key in PARAM_VECTOR_KEYS],
            axis=0,
        )
    if "smplx_param_vector" not in person:
        raise KeyError("Prediction is missing SMPL-X parameter fields and smplx_param_vector")
    return np.asarray(person["smplx_param_vector"], dtype=np.float32).reshape(-1)


def base_feature(person: dict, preset: str) -> np.ndarray:
    global_orient = np.asarray(person["global_orient"], dtype=np.float32).reshape(3)
    transl = np.asarray(person["transl"], dtype=np.float32).reshape(3)
    bbox = bbox_feature(person)
    if preset == "minimal":
        return np.concatenate([global_orient, transl, bbox], axis=0)
    if preset == "full":
        return np.concatenate([param_vector(person), bbox], axis=0)
    if preset != "upper_global":
        raise ValueError(f"Unknown feature preset: {preset}")

    body_pose = np.asarray(person["body_pose"], dtype=np.float32).reshape(21, 3)
    upper_body_pose = body_pose[np.asarray(UPPER_BODY_BODY_POSE_INDICES, dtype=np.int64)].reshape(-1)
    left_hand = np.asarray(person["left_hand_pose"], dtype=np.float32).reshape(-1)
    right_hand = np.asarray(person["right_hand_pose"], dtype=np.float32).reshape(-1)
    jaw = np.asarray(person["jaw_pose"], dtype=np.float32).reshape(-1)
    betas = np.asarray(person["betas"], dtype=np.float32).reshape(-1)[:10]
    expression = np.asarray(person["expression"], dtype=np.float32).reshape(-1)[:10]
    return np.concatenate(
        [
            global_orient,
            transl,
            upper_body_pose,
            left_hand,
            right_hand,
            jaw,
            betas,
            expression,
            bbox,
        ],
        axis=0,
    )


def apply_predicted_correction(
    person: dict,
    delta_rotvec: np.ndarray,
    delta_transl: np.ndarray,
    *,
    orient_scale: float = 1.0,
    transl_scale: float = 1.0,
    max_delta_degrees: float | None = 30.0,
) -> dict:
    corrected = copy_person(person)
    delta_rotvec = np.asarray(delta_rotvec, dtype=np.float64).reshape(3) * float(orient_scale)
    if max_delta_degrees is not None and max_delta_degrees > 0:
        max_norm = np.deg2rad(float(max_delta_degrees))
        norm = float(np.linalg.norm(delta_rotvec))
        if norm > max_norm:
            delta_rotvec = delta_rotvec * (max_norm / norm)

    current = rotvec_to_matrix(np.asarray(corrected["global_orient"], dtype=np.float64).reshape(3))
    delta = rotvec_to_matrix(delta_rotvec)
    corrected["global_orient"] = matrix_to_rotvec(delta @ current).astype(np.float32)
    corrected["transl"] = (
        np.asarray(corrected["transl"], dtype=np.float32).reshape(3)
        + np.asarray(delta_transl, dtype=np.float32).reshape(3) * float(transl_scale)
    ).astype(np.float32)
    rebuild_param_vector(corrected)
    return corrected


def correction_target(original: dict, corrected: dict) -> np.ndarray:
    original_rotation = rotvec_to_matrix(np.asarray(original["global_orient"], dtype=np.float64).reshape(3))
    corrected_rotation = rotvec_to_matrix(np.asarray(corrected["global_orient"], dtype=np.float64).reshape(3))
    delta_rotation = corrected_rotation @ original_rotation.T
    delta_rotvec = matrix_to_rotvec(delta_rotation)
    delta_transl = (
        np.asarray(corrected["transl"], dtype=np.float64).reshape(3)
        - np.asarray(original["transl"], dtype=np.float64).reshape(3)
    )
    return np.concatenate([delta_rotvec, delta_transl], axis=0).astype(np.float32)


def load_one_person_or_none(path: Path) -> dict | None:
    try:
        return one_person(path)
    except Exception:
        return None
