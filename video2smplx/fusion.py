"""Pure fusion helpers for SMPLest-X, WiLoR, and EMOCA outputs.

This module intentionally has no subprocess, filesystem, or model-loading
logic. It is the stable contract layer between the three specialist projects
and the final SMPL-X parameter stream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import re
from typing import Any, Mapping, MutableMapping

import numpy as np


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

EXPECTED_DIMS = {
    "global_orient": {3},
    "body_pose": {63},
    "left_hand_pose": {45},
    "right_hand_pose": {45},
    "jaw_pose": {3},
    "betas": {10},
    "expression": {10, 50},
    "transl": {3},
}


@dataclass
class FusionStats:
    total_frames: int = 0
    wilor_matched: int = 0
    wilor_missing: int = 0
    emoca_matched: int = 0
    emoca_missing: int = 0
    empty_smplestx_frames: int = 0
    validation_errors: int = 0
    errors: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_frame_id(filename: str) -> int | None:
    """Return the last integer found in a filename, or None."""
    matches = re.findall(r"\d+", filename)
    return int(matches[-1]) if matches else None


def normalize_emoca_frame_id(filename: str) -> int | None:
    """Normalize EMOCA's frame_{N*100}_params.pkl names to frame N."""
    raw = extract_frame_id(filename)
    if raw is None:
        return None
    return raw // 100


def flattened_array(value: Any) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def validate_person(person: Mapping[str, Any]) -> list[str]:
    """Return validation messages for malformed SMPL-X parameter fields."""
    errors: list[str] = []
    for key, allowed_dims in EXPECTED_DIMS.items():
        if key not in person:
            errors.append(f"missing key '{key}'")
            continue
        dim = flattened_array(person[key]).shape[0]
        if dim not in allowed_dims:
            expected = "/".join(str(v) for v in sorted(allowed_dims))
            errors.append(f"key '{key}' has dim {dim}, expected {expected}")
    return errors


def rebuild_smplx_param_vector(person: MutableMapping[str, Any]) -> np.ndarray:
    """Rebuild and assign the concatenated SMPL-X parameter vector."""
    missing = [key for key in PARAM_VECTOR_KEYS if key not in person]
    if missing:
        raise KeyError(f"cannot rebuild smplx_param_vector; missing: {missing}")

    vector = np.concatenate(
        [flattened_array(person[key]) for key in PARAM_VECTOR_KEYS],
        axis=0,
    )
    person["smplx_param_vector"] = vector
    return vector


def fuse_person(
    base_person: Mapping[str, Any],
    wilor_params: Mapping[str, Any] | None = None,
    emoca_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse specialist hand and face outputs into one person parameter dict."""
    person = copy.deepcopy(dict(base_person))

    if wilor_params:
        right_hand = wilor_params.get("right_hand_pose")
        left_hand = wilor_params.get("left_hand_pose")
        if right_hand is not None:
            person["right_hand_pose"] = flattened_array(right_hand)
        if left_hand is not None:
            person["left_hand_pose"] = flattened_array(left_hand)

    if emoca_params:
        expression = emoca_params.get("exp")
        jaw_pose = emoca_params.get("jaw_pose")
        if expression is not None:
            person["expression"] = flattened_array(expression)
        if jaw_pose is not None:
            person["jaw_pose"] = flattened_array(jaw_pose)

    rebuild_smplx_param_vector(person)
    return person


def fuse_frame(
    smplestx_data: list[Any],
    wilor_params: Mapping[str, Any] | None = None,
    emoca_params: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Fuse one SMPLest-X frame list, preserving non-primary people unchanged."""
    if not smplestx_data:
        return []

    fused = copy.deepcopy(smplestx_data)
    if isinstance(fused[0], Mapping):
        fused[0] = fuse_person(fused[0], wilor_params, emoca_params)
    return fused

