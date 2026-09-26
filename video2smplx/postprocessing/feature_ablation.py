"""Feature layout transforms supported by correction checkpoints."""

from __future__ import annotations

import numpy as np


DEFAULT_BASE_FEATURE_DIM = 168


def transform_features(
    x: np.ndarray,
    *,
    mode: str,
    temporal_radius: int,
    base_feature_dim: int = DEFAULT_BASE_FEATURE_DIM,
) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float32)
    block_count = 2 * int(temporal_radius) + 1
    if block_count <= 0:
        raise ValueError(f"Invalid temporal radius: {temporal_radius}")
    if x.shape[1] % block_count != 0:
        raise ValueError(f"Feature dim {x.shape[1]} is not divisible by temporal block count {block_count}")
    per_frame_dim = x.shape[1] // block_count
    if base_feature_dim > per_frame_dim:
        raise ValueError(f"base_feature_dim {base_feature_dim} > per_frame_dim {per_frame_dim}")

    blocks = x.reshape(x.shape[0], block_count, per_frame_dim)
    if mode == "full":
        output = blocks.reshape(x.shape[0], -1).copy()
    elif mode == "smplx_only":
        output = blocks[:, :, :base_feature_dim].reshape(x.shape[0], -1).copy()
    elif mode == "detector_zeroed":
        edited = blocks.copy()
        edited[:, :, base_feature_dim:] = 0.0
        output = edited.reshape(x.shape[0], -1)
    elif mode == "detector_only":
        edited = blocks.copy()
        edited[:, :, :base_feature_dim] = 0.0
        output = edited.reshape(x.shape[0], -1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    metadata = {
        "mode": mode,
        "block_count": int(block_count),
        "base_feature_dim": int(base_feature_dim),
        "per_frame_feature_dim": int(per_frame_dim),
        "input_feature_dim": int(x.shape[1]),
        "output_feature_dim": int(output.shape[1]),
    }
    return output.astype(np.float32), metadata
