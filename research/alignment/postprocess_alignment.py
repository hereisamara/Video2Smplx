#!/usr/bin/env python3
"""Post-process SMPL-X predictions with prediction-only camera alignment.

This script intentionally does not read 3D ground truth. It can either apply a
fixed camera-convention transform or refine global orientation/translation from
2D OpenPose keypoints and camera intrinsics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from video2smplx.fusion import rebuild_smplx_param_vector


TRANSFORMS = {
    "none": np.eye(3, dtype=np.float64),
    "x-180": np.diag([1.0, -1.0, -1.0]),
    "opencv-to-opengl": np.diag([1.0, -1.0, -1.0]),
    "y-180": np.diag([-1.0, 1.0, -1.0]),
    "z-180": np.diag([-1.0, -1.0, 1.0]),
}

# OpenPose BODY_25 -> SMPL-X joint indices. For mid-hip we average both hips.
BODY25_TO_SMPLX = {
    1: (12,),  # Neck
    2: (17,),  # RShoulder
    3: (19,),  # RElbow
    4: (21,),  # RWrist
    5: (16,),  # LShoulder
    6: (18,),  # LElbow
    7: (20,),  # LWrist
    8: (1, 2),  # MidHip
    9: (2,),  # RHip
    10: (5,),  # RKnee
    11: (8,),  # RAnkle
    12: (1,),  # LHip
    13: (4,),  # LKnee
    14: (7,),  # LAnkle
}


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraCalibration:
    camera: Camera
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None


@dataclass
class FitResult:
    frame: int
    file: str
    status: str
    observations: int
    initial_reprojection_px: float | None
    final_reprojection_px: float | None
    delta_rotvec: list[float]
    transl: list[float] | None
    message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align SMPL-X parameter pickles without 3D ground truth. "
            "Use --mode fixed for a camera-convention transform, or --mode fit2d "
            "to refine alignment from OpenPose BODY_25 keypoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder of *_params.pkl files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for aligned *_params.pkl files.")
    parser.add_argument("--mode", choices=("fixed", "fit2d"), default="fit2d")
    parser.add_argument(
        "--initial-transform",
        choices=sorted(TRANSFORMS),
        default="none",
        help="Fixed camera-convention transform applied before optional 2D fitting.",
    )
    parser.add_argument(
        "--camera-rotation-transform",
        choices=("none", "world-to-camera", "camera-to-world"),
        default="none",
        help=(
            "Compose a calibrated camera extrinsic rotation from --camera-file. "
            "This is useful for EHF-style calibrated datasets and does not use GT meshes."
        ),
    )
    parser.add_argument(
        "--extra-rotvec",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="Additional fixed rotation vector in radians composed after the selected transform.",
    )
    parser.add_argument(
        "--rotation-mode",
        choices=("none", "per-frame", "median"),
        default="median",
        help="How to apply the 2D-fitted orientation correction.",
    )
    parser.add_argument(
        "--translation-mode",
        choices=("keep", "fit"),
        default="fit",
        help="Whether to keep original translation or refit translation from 2D.",
    )
    parser.add_argument("--keypoints-dir", type=Path, help="Folder containing OpenPose *_2Djnt.json files.")
    parser.add_argument("--camera-file", type=Path, help="EHF-style camera file with intrinsics.")
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--smplx-model", type=Path, help="Path to SMPLX_*.npz or SMPL-X model folder.")
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch"),
        default="numpy",
        help="SMPL-X forward backend used by fit2d.",
    )
    parser.add_argument(
        "--numpy-smplx-module",
        type=Path,
        default=Path("/Users/khineaindrayhtun/Downloads/videos/evaluate_smplx_geometry.py"),
        help="Python file that defines NumpySMPLX for the numpy backend.",
    )
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--max-rot-deg", type=float, default=45.0)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for quick diagnostics.")
    parser.add_argument(
        "--flat-hand-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SMPL-X layer flat_hand_mean setting; keep consistent with rendering/evaluation.",
    )
    parser.add_argument("--report-name", default="alignment_report.json")
    return parser.parse_args()


def frame_id_from_name(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.name)
    return int(matches[-1]) if matches else None


def flatten(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def read_pickle(path: Path) -> list[Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def parse_ehf_camera(path: Path) -> CameraCalibration:
    text = path.read_text(encoding="utf-8")

    def rows_after(label: str, count: int) -> list[float]:
        idx = text.find(label)
        if idx < 0:
            raise ValueError(f"missing camera section: {label}")
        tail = text[idx:].splitlines()[1:]
        values: list[float] = []
        for line in tail:
            if "[[" in line or "[" in line:
                values.extend(float(v) for v in re.findall(r"[-+]?\d+\.\d+(?:e[-+]?\d+)?|[-+]?\d+", line))
            if len(values) >= count:
                return values[:count]
        raise ValueError(f"not enough values after camera section: {label}")

    focal = rows_after("Focal length", 2)
    principal = rows_after("Principal Point", 2)
    rotation = None
    translation = None
    try:
        rotation = Rotation.from_rotvec(rows_after("Rotation", 3)).as_matrix()
    except ValueError:
        rotation = None
    try:
        translation = np.asarray(rows_after("Translation", 3), dtype=np.float64)
    except ValueError:
        translation = None
    return CameraCalibration(
        camera=Camera(fx=focal[0], fy=focal[1], cx=principal[0], cy=principal[1]),
        rotation=rotation,
        translation=translation,
    )


def make_camera(args: argparse.Namespace) -> Camera:
    if args.fx is not None and args.fy is not None and args.cx is not None and args.cy is not None:
        return Camera(args.fx, args.fy, args.cx, args.cy)
    if args.camera_file:
        return parse_ehf_camera(args.camera_file).camera
    raise ValueError("fit2d requires either --camera-file or all of --fx --fy --cx --cy")


def make_global_transform(args: argparse.Namespace) -> np.ndarray:
    transform = TRANSFORMS[args.initial_transform].copy()
    if args.camera_rotation_transform == "none":
        output = transform
    else:
        if args.camera_file is None:
            raise ValueError("--camera-rotation-transform requires --camera-file")
        calibration = parse_ehf_camera(args.camera_file)
        if calibration.rotation is None:
            raise ValueError(f"No camera rotation found in: {args.camera_file}")
        if args.camera_rotation_transform == "world-to-camera":
            camera_rotation = calibration.rotation
        else:
            camera_rotation = calibration.rotation.T
        output = camera_rotation @ transform
    if args.extra_rotvec is not None:
        output = Rotation.from_rotvec(np.asarray(args.extra_rotvec, dtype=np.float64)).as_matrix() @ output
    return output


def find_openpose_json(keypoints_dir: Path, frame_id: int, prediction_name: str) -> Path | None:
    candidates = [
        keypoints_dir / f"{frame_id:02d}_2Djnt.json",
        keypoints_dir / f"{frame_id:03d}_2Djnt.json",
        keypoints_dir / f"{frame_id:06d}_2Djnt.json",
        keypoints_dir / f"{frame_id:06d}_keypoints.json",
        keypoints_dir / prediction_name.replace("_params.pkl", "_2Djnt.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(keypoints_dir.glob(f"*{frame_id:02d}*_2Djnt.json"))
    return matches[0] if matches else None


def load_body25(path: Path, confidence_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    people = data.get("people") or []
    if not people:
        raise ValueError("no people in OpenPose JSON")
    keypoints = np.asarray(people[0].get("pose_keypoints_2d", []), dtype=np.float64).reshape(-1, 3)
    if keypoints.shape[0] < 25:
        raise ValueError(f"expected BODY_25 keypoints, got {keypoints.shape[0]}")
    conf = keypoints[:, 2]
    valid = conf >= confidence_threshold
    return keypoints[:, :2], valid


def select_joint_pairs(joints: np.ndarray, keypoints: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    weights: list[float] = []
    for openpose_idx, smplx_indices in BODY25_TO_SMPLX.items():
        if openpose_idx >= len(valid) or not valid[openpose_idx]:
            continue
        if max(smplx_indices) >= joints.shape[0]:
            continue
        source.append(np.mean(joints[list(smplx_indices)], axis=0))
        target.append(keypoints[openpose_idx])
        weights.append(1.0)
    if not source:
        return np.empty((0, 3)), np.empty((0, 2)), np.empty((0,))
    return np.asarray(source), np.asarray(target), np.asarray(weights)


def project(points: np.ndarray, camera: Camera) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-4)
    return np.column_stack((camera.fx * points[:, 0] / z + camera.cx, camera.fy * points[:, 1] / z + camera.cy))


def mean_reprojection(points_3d: np.ndarray, target_2d: np.ndarray, camera: Camera) -> float:
    if len(points_3d) == 0:
        return math.nan
    return float(np.linalg.norm(project(points_3d, camera) - target_2d, axis=1).mean())


def apply_transform_to_person(person: dict[str, Any], transform: np.ndarray) -> None:
    if "global_orient" in person:
        current = Rotation.from_rotvec(flatten(person["global_orient"])).as_matrix()
        person["global_orient"] = Rotation.from_matrix(transform @ current).as_rotvec().astype(np.float32)
    if "transl" in person:
        person["transl"] = (transform @ flatten(person["transl"])).astype(np.float32)
    rebuild_smplx_param_vector(person)


def torch_available() -> tuple[Any, Any]:
    try:
        import torch
        import smplx
    except ImportError as exc:
        raise RuntimeError(
            "fit2d mode requires torch and smplx. Run this script in the same "
            "environment used for SMPL-X rendering/inference, or use --mode fixed."
        ) from exc
    return torch, smplx


def load_torch_smplx_model(args: argparse.Namespace):
    torch, smplx = torch_available()
    if args.smplx_model is None:
        raise ValueError("fit2d requires --smplx-model")
    model = smplx.SMPLX(
        model_path=str(args.smplx_model),
        gender=args.gender,
        use_pca=False,
        flat_hand_mean=args.flat_hand_mean,
        num_betas=10,
        num_expression_coeffs=50,
    ).to(args.device)
    model.eval()
    return "torch", torch, model


def load_numpy_smplx_model(args: argparse.Namespace):
    if args.smplx_model is None:
        raise ValueError("fit2d requires --smplx-model")
    module_path = args.numpy_smplx_module
    if module_path is not None and module_path.is_file():
        spec = importlib_util.spec_from_file_location("video2smplx_numpy_eval", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load NumpySMPLX module from: {module_path}")
        module = importlib_util.module_from_spec(spec)
        # Its imports are relative to the evaluator directory.
        sys.path.insert(0, str(module_path.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            try:
                sys.path.remove(str(module_path.parent))
            except ValueError:
                pass
    else:
        try:
            import evaluate_smplx_geometry as module
        except ImportError as exc:
            raise RuntimeError(
                "numpy backend requires evaluate_smplx_geometry.py with NumpySMPLX; "
                "pass --numpy-smplx-module /path/to/evaluate_smplx_geometry.py"
            ) from exc

    if not hasattr(module, "NumpySMPLX"):
        raise RuntimeError(f"NumpySMPLX not found in: {module_path}")
    return "numpy", None, module.NumpySMPLX(args.smplx_model)


def load_fit_model(args: argparse.Namespace):
    if args.backend == "torch":
        return load_torch_smplx_model(args)
    return load_numpy_smplx_model(args)


def tensor_param(torch, person: dict[str, Any], key: str, dim: int, device: str):
    value = flatten(person.get(key, np.zeros(dim, dtype=np.float32)))
    if value.shape[0] < dim:
        value = np.pad(value, (0, dim - value.shape[0]))
    value = value[:dim].astype(np.float32)
    return torch.tensor(value, dtype=torch.float32, device=device).view(1, -1)


def smplx_joints_torch(torch, model, person: dict[str, Any], global_orient: np.ndarray, transl: np.ndarray, device: str) -> np.ndarray:
    params = {
        "global_orient": torch.tensor(global_orient.astype(np.float32), dtype=torch.float32, device=device).view(1, 3),
        "body_pose": tensor_param(torch, person, "body_pose", 63, device),
        "left_hand_pose": tensor_param(torch, person, "left_hand_pose", 45, device),
        "right_hand_pose": tensor_param(torch, person, "right_hand_pose", 45, device),
        "jaw_pose": tensor_param(torch, person, "jaw_pose", 3, device),
        "betas": tensor_param(torch, person, "betas", 10, device),
        "expression": tensor_param(torch, person, "expression", 50, device),
        "transl": torch.tensor(transl.astype(np.float32), dtype=torch.float32, device=device).view(1, 3),
    }
    with torch.no_grad():
        output = model(return_verts=False, **params)
    return output.joints.detach().cpu().numpy()[0]


def smplx_joints_numpy(model, person: dict[str, Any], global_orient: np.ndarray, transl: np.ndarray) -> np.ndarray:
    fitted = copy.deepcopy(person)
    fitted["global_orient"] = global_orient.astype(np.float32)
    fitted["transl"] = transl.astype(np.float32)
    _, joints = model.forward([fitted], predicted=True)
    return joints[0]


def smplx_joints(backend: str, torch, model, person: dict[str, Any], global_orient: np.ndarray, transl: np.ndarray, device: str) -> np.ndarray:
    if backend == "torch":
        return smplx_joints_torch(torch, model, person, global_orient, transl, device)
    return smplx_joints_numpy(model, person, global_orient, transl)


def fit_frame(
    backend: str,
    torch,
    model,
    person: dict[str, Any],
    keypoints: np.ndarray,
    valid: np.ndarray,
    camera: Camera,
    args: argparse.Namespace,
    fixed_delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    base_orient = flatten(person["global_orient"]).astype(np.float64)
    base_transl = flatten(person.get("transl", np.zeros(3))).astype(np.float64)

    if fixed_delta is not None:
        start_delta = fixed_delta.astype(np.float64)
        fit_rotation = False
    else:
        start_delta = np.zeros(3, dtype=np.float64)
        fit_rotation = args.rotation_mode != "none"
    fit_translation = args.translation_mode == "fit"

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        offset = 0
        if fit_rotation:
            delta = values[:3]
            offset = 3
        else:
            delta = start_delta
        if fit_translation:
            transl = values[offset : offset + 3]
        else:
            transl = base_transl
        orient = Rotation.from_matrix(
            Rotation.from_rotvec(delta).as_matrix() @ Rotation.from_rotvec(base_orient).as_matrix()
        ).as_rotvec()
        return orient, transl

    x0_parts = []
    lower = []
    upper = []
    max_rot = math.radians(args.max_rot_deg)
    if fit_rotation:
        x0_parts.append(start_delta)
        lower.extend([-max_rot, -max_rot, -max_rot])
        upper.extend([max_rot, max_rot, max_rot])
    if fit_translation:
        x0_parts.append(base_transl)
        lower.extend([-100.0, -100.0, 0.05])
        upper.extend([100.0, 100.0, 100.0])
    if not x0_parts:
        joints = smplx_joints(backend, torch, model, person, base_orient, base_transl, args.device)
        source, target, _ = select_joint_pairs(joints, keypoints, valid)
        error = mean_reprojection(source, target, camera)
        return base_orient, base_transl, error, error, len(source)

    x0 = np.concatenate(x0_parts)
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)

    def residual(values: np.ndarray) -> np.ndarray:
        orient, transl = unpack(values)
        joints = smplx_joints(backend, torch, model, person, orient, transl, args.device)
        source, target, weights = select_joint_pairs(joints, keypoints, valid)
        if len(source) < 4:
            return np.full(8, 1e3, dtype=np.float64)
        projected = project(source, camera)
        return ((projected - target) * np.sqrt(weights[:, None])).reshape(-1) / max(camera.fx, camera.fy)

    initial_orient, initial_transl = unpack(x0)
    initial_joints = smplx_joints(backend, torch, model, person, initial_orient, initial_transl, args.device)
    initial_source, initial_target, _ = select_joint_pairs(initial_joints, keypoints, valid)
    initial_error = mean_reprojection(initial_source, initial_target, camera)

    result = least_squares(
        residual,
        x0,
        bounds=(lower_arr, upper_arr),
        max_nfev=args.max_nfev,
        loss="soft_l1",
        f_scale=0.02,
    )
    final_orient, final_transl = unpack(result.x)
    final_joints = smplx_joints(backend, torch, model, person, final_orient, final_transl, args.device)
    final_source, final_target, _ = select_joint_pairs(final_joints, keypoints, valid)
    final_error = mean_reprojection(final_source, final_target, camera)
    return final_orient, final_transl, initial_error, final_error, len(final_source)


def fit2d(args: argparse.Namespace, files: list[Path]) -> list[FitResult]:
    if args.keypoints_dir is None:
        raise ValueError("fit2d requires --keypoints-dir")
    camera = make_camera(args)
    backend, torch, model = load_fit_model(args)

    staged: list[tuple[Path, list[Any], dict[str, Any], np.ndarray, np.ndarray, float, float, int]] = []
    first_pass: list[FitResult] = []

    for path in files:
        frame_id = frame_id_from_name(path)
        if frame_id is None:
            first_pass.append(FitResult(-1, path.name, "skipped", 0, None, None, [0.0, 0.0, 0.0], None, "no frame id"))
            continue
        keypoint_path = find_openpose_json(args.keypoints_dir, frame_id, path.name)
        if keypoint_path is None:
            first_pass.append(FitResult(frame_id, path.name, "skipped", 0, None, None, [0.0, 0.0, 0.0], None, "missing keypoints"))
            continue

        data = read_pickle(path)
        aligned = copy.deepcopy(data)
        if not aligned or not isinstance(aligned[0], dict):
            first_pass.append(FitResult(frame_id, path.name, "skipped", 0, None, None, [0.0, 0.0, 0.0], None, "empty frame"))
            continue
        person = aligned[0]
        apply_transform_to_person(person, make_global_transform(args))
        keypoints, valid = load_body25(keypoint_path, args.confidence_threshold)
        orient, transl, initial_error, final_error, observations = fit_frame(
            backend, torch, model, person, keypoints, valid, camera, args
        )
        base = flatten(person["global_orient"])
        delta = Rotation.from_matrix(
            Rotation.from_rotvec(orient).as_matrix() @ Rotation.from_rotvec(base).as_matrix().T
        ).as_rotvec()
        staged.append((path, aligned, person, keypoints, valid, initial_error, final_error, observations))
        first_pass.append(
            FitResult(
                frame_id,
                path.name,
                "fit",
                observations,
                initial_error,
                final_error,
                delta.astype(float).tolist(),
                transl.astype(float).tolist(),
            )
        )

    if args.rotation_mode == "median":
        deltas = np.asarray([row.delta_rotvec for row in first_pass if row.status == "fit"], dtype=np.float64)
        fixed_delta = np.median(deltas, axis=0) if len(deltas) else np.zeros(3, dtype=np.float64)
    else:
        fixed_delta = None

    results: list[FitResult] = []
    staged_by_name = {item[0].name: item for item in staged}
    first_by_name = {item.file: item for item in first_pass}
    for path in files:
        first = first_by_name.get(path.name)
        if first is None or first.status != "fit":
            if first is not None:
                results.append(first)
            continue
        _, aligned, person, keypoints, valid, _, _, _ = staged_by_name[path.name]
        if args.rotation_mode == "median":
            orient, transl, initial_error, final_error, observations = fit_frame(
                backend, torch, model, person, keypoints, valid, camera, args, fixed_delta=fixed_delta
            )
            delta = fixed_delta
        else:
            orient = Rotation.from_matrix(
                Rotation.from_rotvec(np.asarray(first.delta_rotvec)).as_matrix()
                @ Rotation.from_rotvec(flatten(person["global_orient"])).as_matrix()
            ).as_rotvec()
            transl = np.asarray(first.transl if first.transl is not None else flatten(person.get("transl", np.zeros(3))))
            initial_error = first.initial_reprojection_px
            final_error = first.final_reprojection_px
            observations = first.observations

        person["global_orient"] = orient.astype(np.float32)
        person["transl"] = transl.astype(np.float32)
        rebuild_smplx_param_vector(person)
        write_pickle(args.output_dir / path.name, aligned)
        results.append(
            FitResult(
                first.frame,
                path.name,
                "written",
                observations,
                initial_error,
                final_error,
                np.asarray(delta, dtype=float).tolist(),
                transl.astype(float).tolist(),
            )
        )
    return results


def fixed_transform(args: argparse.Namespace, files: list[Path]) -> list[FitResult]:
    results: list[FitResult] = []
    transform = make_global_transform(args)
    for path in files:
        data = read_pickle(path)
        aligned = copy.deepcopy(data)
        if aligned and isinstance(aligned[0], dict):
            apply_transform_to_person(aligned[0], transform)
            status = "written"
        else:
            status = "empty"
        write_pickle(args.output_dir / path.name, aligned)
        results.append(
            FitResult(
                frame=frame_id_from_name(path) or -1,
                file=path.name,
                status=status,
                observations=0,
                initial_reprojection_px=None,
                final_reprojection_px=None,
                delta_rotvec=Rotation.from_matrix(transform).as_rotvec().astype(float).tolist(),
                transl=flatten(aligned[0]["transl"]).astype(float).tolist() if aligned and isinstance(aligned[0], dict) and "transl" in aligned[0] else None,
            )
        )
    return results


def summarize(results: list[FitResult], args: argparse.Namespace) -> dict[str, Any]:
    written = [row for row in results if row.status == "written"]
    initial = np.asarray([row.initial_reprojection_px for row in written if row.initial_reprojection_px is not None], dtype=np.float64)
    final = np.asarray([row.final_reprojection_px for row in written if row.final_reprojection_px is not None], dtype=np.float64)
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "mode": args.mode,
        "initial_transform": args.initial_transform,
        "camera_rotation_transform": args.camera_rotation_transform,
        "extra_rotvec": args.extra_rotvec,
        "rotation_mode": args.rotation_mode,
        "translation_mode": args.translation_mode,
        "files": len(results),
        "written": len(written),
        "initial_reprojection_px_mean": float(np.mean(initial)) if len(initial) else None,
        "final_reprojection_px_mean": float(np.mean(final)) if len(final) else None,
        "frames": [row.__dict__ for row in results],
    }
    return summary


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("*_params.pkl"))
    if args.max_frames is not None:
        files = files[: args.max_frames]
    if not files:
        raise FileNotFoundError(f"No *_params.pkl files found in: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "fixed":
            results = fixed_transform(args, files)
        else:
            results = fit2d(args, files)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    report = summarize(results, args)
    report_path = args.output_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    main()
