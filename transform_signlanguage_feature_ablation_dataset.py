#!/usr/bin/env python3
"""Create feature ablation datasets from a 2D-guided SignLanguage NPZ.

The original 2D-guided feature layout is temporal stacks of per-frame features:

    [SMPL-X base feature | detector/projected 2D geometry features]

This script keeps the same labels/splits but rewrites x for controlled
ablations, for example `smplx_only` with no detector keypoint inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_BASE_FEATURE_DIM = 168


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("full", "smplx_only", "detector_zeroed", "detector_only"),
        required=True,
    )
    parser.add_argument("--base-feature-dim", type=int, default=DEFAULT_BASE_FEATURE_DIM)
    parser.add_argument(
        "--temporal-radius",
        type=int,
        default=-1,
        help="Default: read temporal_radius from input dataset.",
    )
    return parser.parse_args()


def scalar_from_npz(data, key: str, default):
    if key not in data.files:
        return default
    value = data[key]
    if hasattr(value, "shape") and value.shape == ():
        return value.item()
    return value


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


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    temporal_radius = (
        int(scalar_from_npz(data, "temporal_radius", 1))
        if args.temporal_radius < 0
        else int(args.temporal_radius)
    )
    x, metadata = transform_features(
        data["x"],
        mode=args.mode,
        temporal_radius=temporal_radius,
        base_feature_dim=args.base_feature_dim,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: data[key] for key in data.files if key != "x"}
    arrays.update(
        {
            "x": x,
            "feature_ablation": np.asarray(args.mode),
            "source_dataset": np.asarray(str(args.input.resolve())),
            "base_feature_dim": np.asarray(args.base_feature_dim, dtype=np.int64),
            "source_feature_dim": np.asarray(metadata["input_feature_dim"], dtype=np.int64),
        }
    )
    np.savez_compressed(args.output, **arrays)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        **metadata,
        "samples": int(x.shape[0]),
        "split_counts": {
            split: int(np.sum(data["split_name"].astype(str) == split))
            for split in ("train", "val", "test")
            if "split_name" in data.files
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
