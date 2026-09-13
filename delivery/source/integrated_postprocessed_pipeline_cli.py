#!/usr/bin/env python3
"""One-process integrated Video2Smplx pipeline with optional post-processing.

This entrypoint is for fair FPS testing and deployment-style execution. It runs
the existing integrated body/hand/face fusion pipeline once, then applies the
trained SignLanguage post-processors in the same Python process.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apply_signlanguage_2d_guided_upperbody_corrector import (  # noqa: E402
    apply_delta as apply_upper_delta,
    normalize_indices,
)
from apply_signlanguage_hand_correction import apply_hand_delta  # noqa: E402
from build_signlanguage_2d_guided_upperbody_dataset import (  # noqa: E402
    guided_feature,
    get_yolo_frame,
    load_yolo,
)
from delivery.source.combine_smooth_render_cli import (  # noqa: E402
    export_npz,
    first_person,
    make_side_by_side,
)
from evaluate_signlanguage_geometry import NumpySMPLX  # noqa: E402
from extract_signlanguage_yolo_pose_keypoints import select_person  # noqa: E402
from signlanguage_global_correction import (  # noqa: E402
    apply_predicted_correction,
    base_feature,
    copy_person,
    frame_number,
    write_person,
)
from train_signlanguage_global_correction import build_model, import_torch  # noqa: E402
from transform_signlanguage_feature_ablation_dataset import transform_features  # noqa: E402
from video2smplx.integrated_pipeline import (  # noqa: E402
    DEFAULT_SMPLX_MODEL,
    EMOCA_DIR,
    SMPLESTX_DIR,
    WILOR_DIR,
    run_integrated_pipeline,
)
from zero_filter_render import stage_render, stage_smooth, stage_zero_transl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--sequence", default=None, help="Sequence name for SignLanguage correctors.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smplestx-dir", default=str(SMPLESTX_DIR))
    parser.add_argument("--wilor-dir", default=str(WILOR_DIR))
    parser.add_argument("--emoca-dir", default=str(EMOCA_DIR))
    parser.add_argument("--smplestx-ckpt", default="smplest_x_h")
    parser.add_argument("--emoca-model", default="EMOCA_v2_lr_mse_20")
    parser.add_argument("--smplx-model", default=str(DEFAULT_SMPLX_MODEL))
    parser.add_argument("--eval-model", default="signlanguage_global_correction_server/models/SMPLX_FEMALE.npz")
    parser.add_argument("--multi-person", action="store_true")
    parser.add_argument("--smplestx-detector-stride", type=int, default=5)
    parser.add_argument("--smplestx-batch-size", type=int, default=8)
    parser.add_argument("--smplestx-inference-mode", action="store_true", default=True)
    parser.add_argument("--no-smplestx-inference-mode", dest="smplestx_inference_mode", action="store_false")
    parser.add_argument("--smplestx-single-gpu-model", action="store_true", default=True)
    parser.add_argument("--no-smplestx-single-gpu-model", dest="smplestx_single_gpu_model", action="store_false")
    parser.add_argument("--parallel-models", action="store_true", default=True)
    parser.add_argument("--no-parallel-models", dest="parallel_models", action="store_false")
    parser.add_argument("--stabilize-global-orient", action="store_true")
    parser.add_argument("--stabilize-shape", action="store_true")
    parser.add_argument("--stabilize-lower-body", action="store_true")
    parser.add_argument("--stabilize-torso", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--smooth-poly", type=int, default=3)
    parser.add_argument("--viewport", type=int, default=800)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--disable-postprocessing", action="store_true")
    parser.add_argument(
        "--postprocess-mode",
        choices=("none", "global", "global_hand", "fast", "accurate"),
        default="accurate",
        help=(
            "Postprocess stack: none=base fusion only, global=global_orient+transl, "
            "fast/global_hand=global+hand, accurate=global+hand+2D upper-body."
        ),
    )
    parser.add_argument("--global-ckpt", default="outputs_fusion_signlanguage_no_stablized/evaluation/global_correction_model_upper_r1/best_model.pt")
    parser.add_argument("--hand-ckpt", default="/project/lt200246-mmacma/khtun/signlanguage_hand_correction_outputs/evaluation/hand_correction_model_wrist_fingers_r1/best_model.pt")
    parser.add_argument("--upper2d-ckpt", default="/project/lt200246-mmacma/khtun/signlanguage_2d_guided_upperbody_outputs/evaluation/2d_guided_upperbody_model_r1/best_model.pt")
    parser.add_argument("--yolo-model", default="yolov8x-pose.pt")
    parser.add_argument(
        "--precomputed-yolo-keypoints",
        default=None,
        help="Reusable YOLO pose keypoint JSON. If set, postprocessing skips YOLO inference.",
    )
    parser.add_argument("--upper-scale", type=float, default=0.75)
    parser.add_argument("--min-keypoint-conf", type=float, default=0.15)
    parser.add_argument("--post-batch-size", type=int, default=512)
    parser.add_argument(
        "--warmup-exclude-frames",
        type=int,
        default=2,
        help="Number of first per-frame model timings to exclude from steady-state FPS.",
    )
    return parser.parse_args()


def list_param_files(params_dir: Path) -> list[Path]:
    return sorted(params_dir.glob("*_params.pkl"), key=frame_number)


def load_frame(path: Path) -> list[Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def temporal_stack(rows: list[dict[str, Any]], radius: int) -> np.ndarray:
    if radius <= 0:
        return np.stack([row["base_feature"] for row in rows]).astype(np.float32)
    features = [row["base_feature"] for row in rows]
    stacked = []
    last = len(features) - 1
    for index in range(len(features)):
        stacked.append(
            np.concatenate(
                [features[min(max(index + offset, 0), last)] for offset in range(-radius, radius + 1)],
                axis=0,
            )
        )
    return np.stack(stacked).astype(np.float32)


def load_checkpoint_model(checkpoint_path: Path, device_name: str):
    torch, nn, _, _ = import_torch()
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        nn,
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["layers"]),
        float(checkpoint["dropout"]),
    ).to(device)
    output_dim = int(checkpoint.get("output_dim", 6))
    if hasattr(model[-1], "in_features") and output_dim != model[-1].out_features:
        model[-1] = nn.Linear(model[-1].in_features, output_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    return {
        "torch": torch,
        "device": device,
        "checkpoint": checkpoint,
        "model": model,
        "mean": mean,
        "std": std,
        "feature_preset": str(checkpoint.get("feature_preset", "upper_global")),
        "temporal_radius": int(checkpoint.get("temporal_radius", 1)),
        "output_dim": output_dim,
    }


def predict_mlp(model_info: dict[str, Any], features: np.ndarray, batch_size: int) -> np.ndarray:
    torch = model_info["torch"]
    model = model_info["model"]
    device = model_info["device"]
    x = (features.astype(np.float32) - model_info["mean"]) / model_info["std"]
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).to(device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0, model_info["output_dim"]), dtype=np.float32)


def load_person_rows(params_dir: Path, feature_preset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    skipped = []
    for source_path in list_param_files(params_dir):
        frame_data = load_frame(source_path)
        person = first_person(frame_data)
        if person is None:
            skipped.append({"file": str(source_path), "reason": "empty_or_multi_person"})
            continue
        try:
            feature = base_feature(person, feature_preset)
        except Exception as exc:
            skipped.append({"file": str(source_path), "reason": f"feature_error: {exc}"})
            continue
        rows.append(
            {
                "frame": frame_number(source_path),
                "source_path": source_path,
                "person": person,
                "base_feature": feature,
            }
        )
    rows.sort(key=lambda row: row["frame"])
    return rows, skipped


def copy_or_write_all(source_dir: Path, output_dir: Path, corrected: dict[int, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in list_param_files(source_dir):
        frame = frame_number(source_path)
        target_path = output_dir / source_path.name
        if frame in corrected:
            write_person(target_path, corrected[frame])
        else:
            shutil.copy2(source_path, target_path)


def apply_global_inprocess(input_dir: Path, output_dir: Path, checkpoint: Path, device: str, batch_size: int) -> dict[str, Any]:
    info = load_checkpoint_model(checkpoint, device)
    rows, skipped = load_person_rows(input_dir, info["feature_preset"])
    features = temporal_stack(rows, info["temporal_radius"])
    deltas = predict_mlp(info, features, batch_size)
    corrected = {}
    for row, delta in zip(rows, deltas):
        corrected[row["frame"]] = apply_predicted_correction(
            row["person"],
            delta[:3],
            delta[3:],
            orient_scale=1.0,
            transl_scale=1.0,
            max_delta_degrees=30.0,
        )
    copy_or_write_all(input_dir, output_dir, corrected)
    return {"written_frames": len(corrected), "skipped": skipped[:20], "checkpoint": str(checkpoint)}


def apply_hand_inprocess(input_dir: Path, output_dir: Path, checkpoint: Path, device: str, batch_size: int) -> dict[str, Any]:
    info = load_checkpoint_model(checkpoint, device)
    target_mode = str(info["checkpoint"].get("target_mode", "wrist_fingers"))
    rows, skipped = load_person_rows(input_dir, info["feature_preset"])
    features = temporal_stack(rows, info["temporal_radius"])
    deltas = predict_mlp(info, features, batch_size)
    corrected = {}
    for row, delta in zip(rows, deltas):
        corrected[row["frame"]] = apply_hand_delta(row["person"], delta, target_mode, 90.0, 1.0)
    copy_or_write_all(input_dir, output_dir, corrected)
    return {"written_frames": len(corrected), "skipped": skipped[:20], "checkpoint": str(checkpoint), "target_mode": target_mode}


def extract_yolo_inprocess(video: Path, sequence: str, yolo_model: Path, device: str, output_json: Path) -> dict[str, dict[int, dict]]:
    from ultralytics import YOLO

    model = YOLO(str(yolo_model))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_index, result in enumerate(
        model.predict(
            source=str(video),
            stream=True,
            verbose=False,
            device=device,
            imgsz=960,
            conf=0.25,
        ),
        start=1,
    ):
        xy, conf, bbox, person_conf = select_person(result)
        height, width = result.orig_shape
        frames.append(
            {
                "frame": frame_index,
                "image_width": int(width),
                "image_height": int(height),
                "keypoints_xy": xy.tolist(),
                "keypoints_conf": conf.tolist(),
                "bbox_xyxy": bbox.tolist() if bbox is not None else None,
                "person_conf": person_conf,
            }
        )
    payload = {
        "model": str(yolo_model),
        "imgsz": 960,
        "conf": 0.25,
        "keypoint_format": "COCO17 xy pixel coordinates, confidence",
        "videos": {
            sequence: {
                "video_path": str(video),
                "frames": frames,
            }
        },
    }
    output_json.write_text(json.dumps(payload), encoding="utf-8")
    return {sequence: {int(frame["frame"]): frame for frame in frames}}


def apply_upper2d_inprocess(
    input_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    yolo: dict[str, dict[int, dict]],
    sequence: str,
    model_path: Path,
    device: str,
    batch_size: int,
    scale: float,
    min_keypoint_conf: float,
) -> dict[str, Any]:
    info = load_checkpoint_model(checkpoint, device)
    ckpt = info["checkpoint"]
    feature_ablation = str(ckpt.get("feature_ablation", "full"))
    base_feature_dim = int(ckpt.get("base_feature_dim", 168))
    body_indices = normalize_indices(ckpt.get("target_body_indices"))
    needs_detector = feature_ablation != "smplx_only"
    smplx = NumpySMPLX(model_path.resolve()) if needs_detector else None

    output_dir.mkdir(parents=True, exist_ok=True)
    files = list_param_files(input_dir)
    rows = []
    skipped = []
    for start in range(0, len(files), 64):
        batch_files = files[start : start + 64]
        persons = []
        kept = []
        for source_path in batch_files:
            person = first_person(load_frame(source_path))
            if person is None:
                shutil.copy2(source_path, output_dir / source_path.name)
                skipped.append({"file": str(source_path), "reason": "bad_or_empty_prediction_copied"})
                continue
            persons.append(person)
            kept.append(source_path)
        if not kept:
            continue
        pred_joints = smplx.forward_joints(persons, predicted=True) if needs_detector else None
        for index, source_path in enumerate(kept):
            frame = frame_number(source_path)
            try:
                if needs_detector:
                    yolo_frame = get_yolo_frame(yolo, sequence, frame)
                    if yolo_frame is None:
                        shutil.copy2(source_path, output_dir / source_path.name)
                        skipped.append({"file": str(source_path), "reason": "missing_yolo_copied"})
                        continue
                    feature = guided_feature(
                        persons[index],
                        pred_joints[index],
                        yolo_frame,
                        info["feature_preset"],
                        min_keypoint_conf,
                    )
                else:
                    feature = base_feature(persons[index], info["feature_preset"])
            except Exception as exc:
                shutil.copy2(source_path, output_dir / source_path.name)
                skipped.append({"file": str(source_path), "reason": f"feature_error_copied: {exc}"})
                continue
            rows.append(
                {
                    "frame": frame,
                    "source_path": source_path,
                    "person": persons[index],
                    "base_feature": feature,
                }
            )
    rows.sort(key=lambda row: row["frame"])
    features = temporal_stack(rows, info["temporal_radius"])
    if needs_detector and feature_ablation != "full":
        features, _ = transform_features(
            features,
            mode=feature_ablation,
            temporal_radius=info["temporal_radius"],
            base_feature_dim=base_feature_dim,
        )
    deltas = predict_mlp(info, features, batch_size)
    corrected = {}
    upper_args = argparse.Namespace(
        scale=scale,
        global_scale=1.0,
        arms_scale=1.0,
        max_global_delta_degrees=35.0,
        max_arm_delta_degrees=45.0,
    )
    for row, delta in zip(rows, deltas):
        corrected[row["frame"]] = apply_upper_delta(row["person"], delta, body_indices, upper_args)
    copy_or_write_all(input_dir, output_dir, corrected)
    return {
        "written_frames": len(corrected),
        "skipped": skipped[:20],
        "checkpoint": str(checkpoint),
        "feature_ablation": feature_ablation,
        "body_indices": body_indices,
    }


def render_param_dir(
    params_dir: Path,
    output_dir: Path,
    smplx_model: Path,
    input_video: Path,
    fps: int,
    smooth_window: int,
    smooth_poly: int,
    viewport: int,
    skip_render: bool,
) -> dict[str, Any]:
    files = list_param_files(params_dir)
    if not files:
        raise FileNotFoundError(f"No *_params.pkl files in {params_dir}")
    frames = [load_frame(path) for path in files]
    frame_ids = [frame_number(path) for path in files]
    render_dir = output_dir / "rendered"
    render_params_dir = render_dir / "params"
    render_params_dir.mkdir(parents=True, exist_ok=True)

    render_frames = stage_zero_transl(copy.deepcopy(frames))
    valid_count = sum(first_person(frame) is not None for frame in render_frames)
    window = min(smooth_window, valid_count)
    if window % 2 == 0:
        window -= 1
    if window > smooth_poly:
        render_frames = stage_smooth(render_frames, window, smooth_poly)
    for frame_id, frame_data in zip(frame_ids, render_frames):
        with (render_params_dir / f"{frame_id:06d}_params.pkl").open("wb") as handle:
            pickle.dump(frame_data, handle)

    npz_path = output_dir / "smplx_params.npz"
    export_npz(render_frames, frame_ids, npz_path)
    rendered_video = render_dir / "smplx_render.mp4"
    side_by_side = output_dir / "side_by_side_input_render.mp4"
    if not skip_render:
        stage_render(
            render_frames,
            smplx_model_path=str(smplx_model),
            output_video_path=str(rendered_video),
            video_fps=fps,
            viewport_size=viewport,
        )
        make_side_by_side(input_video, rendered_video, side_by_side, fps)
    return {
        "frames": len(frame_ids),
        "params_dir": str(params_dir),
        "render_params": str(render_params_dir),
        "smplx_params_npz": str(npz_path),
        "rendered_video": str(rendered_video) if rendered_video.exists() else None,
        "side_by_side_video": str(side_by_side) if side_by_side.exists() else None,
    }


def add_timing(timings: list[dict[str, Any]], stage: str, start: float, frames: int | None = None) -> None:
    seconds = time.perf_counter() - start
    row = {"stage": stage, "seconds": seconds}
    if frames:
        row["seconds_per_frame"] = seconds / frames
        row["fps"] = frames / seconds if seconds > 0 else 0.0
    timings.append(row)
    print(f"[timing] {stage}: {seconds:.3f}s", flush=True)


def write_report(path: Path, report: dict[str, Any], timings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report["timings"] = timings
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        keys = ["stage", "seconds", "seconds_per_frame", "fps"]
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(timings)


def steady_state_model_summary(base_summary: dict[str, Any], warmup_exclude: int) -> dict[str, Any]:
    timing_report = base_summary.get("timings") or {}
    per_frame = timing_report.get("per_frame") or []
    usable = [
        row
        for row in per_frame
        if row.get("status") == "ok" and row.get("total_sec") is not None
    ]
    warmup = max(0, int(warmup_exclude))
    steady = usable[warmup:] if warmup < len(usable) else []
    seconds = sum(float(row["total_sec"]) for row in steady)
    frames = len(steady)
    return {
        "warmup_excluded_frames": min(warmup, len(usable)),
        "steady_state_frames": frames,
        "steady_state_model_seconds": seconds,
        "steady_state_model_seconds_per_frame": seconds / frames if frames else None,
        "steady_state_model_fps": frames / seconds if seconds > 0 else None,
        "first_frame_model_seconds": float(usable[0]["total_sec"]) if usable else None,
        "second_frame_model_seconds": float(usable[1]["total_sec"]) if len(usable) > 1 else None,
    }


def initial_load_summary(base_summary: dict[str, Any]) -> dict[str, Any]:
    timing_report = base_summary.get("timings") or {}
    steps = timing_report.get("pipeline_steps") or []
    load_names = {
        "import_runners",
        "load_smplestx",
        "load_wilor",
        "load_emoca",
    }
    selected = [
        {
            "stage": str(step.get("name")),
            "seconds": float(step.get("seconds", 0.0)),
        }
        for step in steps
        if step.get("name") in load_names
    ]
    return {
        "definition": "import_runners + load_smplestx + load_wilor + load_emoca",
        "initial_load_seconds": sum(row["seconds"] for row in selected),
        "initial_load_steps": selected,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    video = Path(args.video).resolve()
    sequence = args.sequence or args.name or video.stem
    output.mkdir(parents=True, exist_ok=True)
    timings: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "program": "integrated_postprocessed_pipeline_cli.py",
        "video": str(video),
        "output": str(output),
        "sequence": sequence,
        "postprocess_mode": "none" if args.disable_postprocessing else args.postprocess_mode,
        "postprocessing_enabled": not args.disable_postprocessing and args.postprocess_mode != "none",
        "settings": vars(args),
    }

    start = time.perf_counter()
    base_summary = run_integrated_pipeline(
        video=video,
        output=output,
        name=sequence,
        fps=args.fps,
        smplestx_ckpt=args.smplestx_ckpt,
        emoca_model=args.emoca_model,
        device=args.device,
        multi_person=args.multi_person,
        skip_render=True,
        smplx_model=Path(args.smplx_model),
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
        viewport=args.viewport,
        smplestx_dir=Path(args.smplestx_dir),
        wilor_dir=Path(args.wilor_dir),
        emoca_dir=Path(args.emoca_dir),
        stabilize_lower_body=args.stabilize_lower_body,
        stabilize_global_orient=args.stabilize_global_orient,
        stabilize_torso=args.stabilize_torso,
        stabilize_shape=args.stabilize_shape,
        smplestx_detector_stride=args.smplestx_detector_stride,
        smplestx_batch_size=args.smplestx_batch_size,
        smplestx_inference_mode=args.smplestx_inference_mode,
        smplestx_single_gpu_model=args.smplestx_single_gpu_model,
        parallel_models=args.parallel_models,
    )
    frames = int(base_summary["frames"])
    add_timing(timings, "base_integrated_inference_fusion_no_render", start, frames)
    report["base_summary"] = base_summary

    base_params = output / "fused_params"
    final_params = base_params
    post_report = {}
    post_mode = "none" if args.disable_postprocessing else args.postprocess_mode
    if post_mode != "none":
        start = time.perf_counter()
        global_dir = output / "fused_params_model_global_corrected"
        post_report["global"] = apply_global_inprocess(
            base_params,
            global_dir,
            Path(args.global_ckpt),
            args.device,
            args.post_batch_size,
        )
        add_timing(timings, "post_global_orient_transl", start, frames)
        final_params = global_dir

        if post_mode in ("global_hand", "fast", "accurate"):
            start = time.perf_counter()
            hand_dir = output / "fused_params_model_global_hand_corrected"
            post_report["hand"] = apply_hand_inprocess(
                global_dir,
                hand_dir,
                Path(args.hand_ckpt),
                args.device,
                args.post_batch_size,
            )
            add_timing(timings, "post_hand_wrist_fingers", start, frames)
            final_params = hand_dir

        if post_mode == "accurate":
            start = time.perf_counter()
            if args.precomputed_yolo_keypoints:
                yolo_json = Path(args.precomputed_yolo_keypoints)
                yolo = load_yolo(yolo_json)
                post_report["yolo"] = {
                    "json": str(yolo_json),
                    "frames": len(yolo.get(sequence, {})),
                    "source": "precomputed",
                }
                add_timing(timings, "post_load_precomputed_yolo_2d", start, frames)
            else:
                yolo_json = output / "runtime" / f"yolo_pose_{sequence}.json"
                yolo = extract_yolo_inprocess(video, sequence, Path(args.yolo_model), args.device, yolo_json)
                post_report["yolo"] = {
                    "json": str(yolo_json),
                    "frames": len(yolo.get(sequence, {})),
                    "source": "inprocess_detector",
                }
                add_timing(timings, "post_yolo_pose_2d", start, frames)

            start = time.perf_counter()
            upper_dir = output / "fused_params_2d_upper_corrected_s0p75"
            post_report["upper2d"] = apply_upper2d_inprocess(
                final_params,
                upper_dir,
                Path(args.upper2d_ckpt),
                yolo,
                sequence,
                Path(args.eval_model),
                args.device,
                args.post_batch_size,
                args.upper_scale,
                args.min_keypoint_conf,
            )
            add_timing(timings, "post_2d_guided_upper_body", start, frames)
            final_params = upper_dir

    report["postprocessing"] = post_report

    start = time.perf_counter()
    if post_mode == "none":
        final_output = output / "final_base"
    elif post_mode == "accurate":
        final_output = output / "final_postprocessed"
    elif post_mode == "global_hand":
        final_output = output / "final_global_hand"
    else:
        final_output = output / f"final_{post_mode}"
    render_report = render_param_dir(
        final_params,
        final_output,
        Path(args.smplx_model),
        video,
        args.fps,
        args.smooth_window,
        args.smooth_poly,
        args.viewport,
        args.skip_render,
    )
    add_timing(timings, "final_npz_smooth_render", start, frames)
    report["final_outputs"] = render_report

    total_seconds = sum(float(row["seconds"]) for row in timings)
    inference_no_render = sum(
        float(row["seconds"])
        for row in timings
        if row["stage"] != "final_npz_smooth_render"
    )
    load_summary = initial_load_summary(base_summary)
    initial_load_seconds = float(load_summary["initial_load_seconds"])
    no_load_total = max(0.0, total_seconds - initial_load_seconds)
    no_load_inference = max(0.0, inference_no_render - initial_load_seconds)
    report["summary"] = {
        "frames": frames,
        "total_seconds": total_seconds,
        "total_seconds_per_frame": total_seconds / frames,
        "total_fps": frames / total_seconds if total_seconds > 0 else 0.0,
        "inference_post_no_render_seconds": inference_no_render,
        "inference_post_no_render_seconds_per_frame": inference_no_render / frames,
        "inference_post_no_render_fps": frames / inference_no_render if inference_no_render > 0 else 0.0,
        "initial_load": load_summary,
        "no_initial_load_total_seconds": no_load_total,
        "no_initial_load_total_seconds_per_frame": no_load_total / frames,
        "no_initial_load_total_fps": frames / no_load_total if no_load_total > 0 else None,
        "no_initial_load_no_render_seconds": no_load_inference,
        "no_initial_load_no_render_seconds_per_frame": no_load_inference / frames,
        "no_initial_load_no_render_fps": frames / no_load_inference if no_load_inference > 0 else None,
        "steady_state": steady_state_model_summary(base_summary, args.warmup_exclude_frames),
    }
    report_path = output / "runtime" / "integrated_postprocessed_runtime_report.json"
    write_report(report_path, report, timings)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"[done] report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
