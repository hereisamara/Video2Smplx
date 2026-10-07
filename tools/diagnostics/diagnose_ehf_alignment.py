#!/usr/bin/env python3
"""GT-only diagnostic for EHF alignment failure modes.

This does not implement a postprocess. It reports how much of the error is
explained by rigid rotation and similarity scale.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--ehf-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--eval-module-dir",
        type=Path,
        default=Path("/Users/khineaindrayhtun/Downloads/videos"),
    )
    parser.add_argument("--transform", choices=("identity", "x-180"), default="x-180")
    return parser.parse_args()


def rigid_align_no_scale(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_mean = predicted.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_centered = predicted - pred_mean
    target_centered = target - target_mean
    u, _, vt = np.linalg.svd(pred_centered.T @ target_centered)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    return pred_centered @ rotation + target_mean


def rigid_rotation_no_scale(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_centered = predicted - predicted.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(pred_centered.T @ target_centered)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    return u @ np.diag(sign) @ vt


def similarity_scale(predicted: np.ndarray, target: np.ndarray) -> float:
    pred_centered = predicted - predicted.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    u, singular_values, vt = np.linalg.svd(pred_centered.T @ target_centered)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    return float(np.sum(singular_values * sign) / np.sum(pred_centered**2))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.eval_module_dir))
    from evaluate_ehf_geometry import (
        coordinate_transform_matrix,
        frame_id_from_prediction,
        read_aligned_ply_vertices,
    )
    from evaluate_smplx_geometry import NumpySMPLX, procrustes_align

    model = NumpySMPLX(args.model_path)
    transform = coordinate_transform_matrix(args.transform)
    files = sorted(args.pred_dir.glob("*_params.pkl"), key=frame_id_from_prediction)

    root = []
    rigid = []
    similarity = []
    scales = []
    rotvecs = []
    for path in files:
        person = pickle.load(path.open("rb"))[0]
        _, pred_joints_batch = model.forward([person], predicted=True)
        pred_joints = pred_joints_batch[0] @ transform.T
        gt_vertices = read_aligned_ply_vertices(
            args.ehf_dir / f"{frame_id_from_prediction(path):02d}_align.ply"
        )
        gt_joints = np.einsum("jv,vc->jc", model.joint_regressor, gt_vertices, optimize=True)

        pred_root = pred_joints - pred_joints[0]
        gt_root = gt_joints - gt_joints[0]
        root.append(np.linalg.norm(pred_root - gt_root, axis=1).mean() * 1000)
        rigid_rotation = rigid_rotation_no_scale(pred_root, gt_root)
        rigid_joints = (pred_root - pred_root.mean(axis=0)) @ rigid_rotation + gt_root.mean(axis=0)
        rigid.append(np.linalg.norm(rigid_joints - gt_root, axis=1).mean() * 1000)
        rotvecs.append(Rotation.from_matrix(rigid_rotation.T).as_rotvec())
        sim_joints = procrustes_align(pred_root, gt_root)
        similarity.append(np.linalg.norm(sim_joints - gt_root, axis=1).mean() * 1000)
        scales.append(similarity_scale(pred_root, gt_root))

    def line(name: str, values: list[float]) -> str:
        arr = np.asarray(values, dtype=np.float64)
        return (
            f"{name}: mean={np.mean(arr):.2f}, median={np.median(arr):.2f}, "
            f"p95={np.percentile(arr, 95):.2f}"
        )

    print(line("root_aligned_mpjpe_mm", root))
    print(line("rigid_rotation_mpjpe_mm", rigid))
    print(line("similarity_pa_mpjpe_mm", similarity))
    print(line("similarity_scale", scales))
    rotvec_arr = np.asarray(rotvecs, dtype=np.float64)
    median_rotvec = np.median(rotvec_arr, axis=0)
    print(
        "median_rigid_rotvec_rad: "
        f"[{median_rotvec[0]:.6f}, {median_rotvec[1]:.6f}, {median_rotvec[2]:.6f}]"
    )
    print(
        "median_rigid_euler_xyz_deg: "
        + str(np.round(Rotation.from_rotvec(median_rotvec).as_euler("xyz", degrees=True), 3).tolist())
    )


if __name__ == "__main__":
    main()
