"""One-process Video2Smplx pipeline.

This path loads SMPLest-X, WiLoR, and EMOCA once as Python runner objects,
then performs body/hand/face inference and fusion per frame in memory.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

from video2smplx.frames import extract_frames_cv2, iter_video_frames_cv2, list_frames
from video2smplx.fusion import FusionStats, fuse_frame, validate_person
from video2smplx.stabilization import (
    GlobalOrientStabilizer,
    LowerBodyStabilizer,
    ShapeStabilizer,
    TorsoStabilizer,
)


ROOT = Path(__file__).resolve().parents[1]
SMPLESTX_DIR = ROOT / "SMPLest-X-Inference"
WILOR_DIR = ROOT / "WiLoR-Inference"
EMOCA_DIR = ROOT / "EMOCA-Inference"
DEFAULT_SMPLX_MODEL = (
    SMPLESTX_DIR
    / "human_models"
    / "human_model_files"
    / "smplx"
    / "SMPLX_NEUTRAL.npz"
)
MODEL_TIMING_KEYS = ("smplestx", "wilor", "emoca")


def _clear_frame_dir(frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for path in frame_dir.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            path.unlink()


def _has_hand_data(hands: dict) -> bool:
    return hands.get("right_hand_pose") is not None or hands.get("left_hand_pose") is not None


def _has_face_data(face: dict) -> bool:
    return face.get("exp") is not None or face.get("jaw_pose") is not None


def _is_valid_fused_frame(frame_data: list) -> bool:
    return bool(frame_data and isinstance(frame_data[0], dict))


def _warning_sample(warnings: list[str], limit: int = 8) -> str:
    if not warnings:
        return "none"
    shown = warnings[:limit]
    suffix = ""
    if len(warnings) > limit:
        suffix = f"\n  ... {len(warnings) - limit} more warnings in fusion_report.json"
    return "\n  " + "\n  ".join(shown) + suffix


def _format_timing_ms(seconds: float | None) -> str:
    if seconds is None:
        return "skipped"
    return f"{seconds * 1000:.1f} ms"


def _timing_average(total_seconds: float, count: int) -> float | None:
    if count == 0:
        return None
    return total_seconds / count


def _effective_smooth_window(
    requested_window: int,
    valid_frames: int,
    polyorder: int,
) -> int | None:
    if valid_frames <= polyorder:
        return None

    window = min(requested_window, valid_frames)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 1
        if window % 2 == 0:
            window += 1
    if window > valid_frames:
        return None
    return window


def run_integrated_pipeline(
    video: Path,
    output: Path,
    name: str | None = None,
    fps: int = 30,
    reuse_frames: bool = False,
    frames_dir: Path | None = None,
    materialize_frames: bool = False,
    smplestx_ckpt: str = "smplest_x_h",
    emoca_model: str = "EMOCA_v2_lr_mse_20",
    device: str = "cuda",
    multi_person: bool = False,
    skip_render: bool = False,
    smplx_model: Path = DEFAULT_SMPLX_MODEL,
    smooth_window: int = 15,
    smooth_poly: int = 3,
    viewport: int = 800,
    smplestx_dir: Path = SMPLESTX_DIR,
    wilor_dir: Path = WILOR_DIR,
    emoca_dir: Path = EMOCA_DIR,
    stabilize_lower_body: bool = False,
    stabilize_global_orient: bool = False,
    stabilize_torso: bool = False,
    stabilize_shape: bool = False,
) -> dict:
    from tqdm import tqdm

    video = Path(video).resolve()
    output = Path(output).resolve()
    smplestx_dir = Path(smplestx_dir).resolve()
    wilor_dir = Path(wilor_dir).resolve()
    emoca_dir = Path(emoca_dir).resolve()
    if smplx_model == DEFAULT_SMPLX_MODEL and smplestx_dir != SMPLESTX_DIR:
        smplx_model = (
            smplestx_dir
            / "human_models"
            / "human_model_files"
            / "smplx"
            / "SMPLX_NEUTRAL.npz"
        )
    run_name = name or video.stem
    frames_dir = Path(frames_dir).resolve() if frames_dir else output / "frames"
    fused_dir = output / "fused_params"
    rendered_dir = output / "rendered"
    report_path = fused_dir / "fusion_report.json"
    timing_report_path = fused_dir / "model_timing_report.json"

    if not video.exists() and not reuse_frames:
        raise FileNotFoundError(f"Video not found: {video}")

    fused_dir.mkdir(parents=True, exist_ok=True)
    if reuse_frames:
        frames = list_frames(frames_dir)
        frame_iterable = frames
        frame_count = len(frames)
        frame_source = str(frames_dir)
        if not frames:
            raise RuntimeError(f"No frames available in: {frames_dir}")
    elif materialize_frames:
        _clear_frame_dir(frames_dir)
        frames = extract_frames_cv2(video, frames_dir, fps=fps)
        frame_iterable = frames
        frame_count = len(frames)
        frame_source = str(frames_dir)
        if not frames:
            raise RuntimeError(f"No frames extracted from: {video}")
    else:
        frame_iterable = iter_video_frames_cv2(video, fps=fps)
        frame_count = None
        frame_source = "in-memory video stream"

    print("\n" + "#" * 72)
    print("  INTEGRATED VIDEO2SMPLX PIPELINE")
    print("#" * 72)
    print(f"  name       : {run_name}")
    print(f"  video      : {video}")
    print(f"  output     : {output}")
    print(f"  smplestx   : {smplestx_dir}")
    print(f"  wilor      : {wilor_dir}")
    print(f"  emoca      : {emoca_dir}")
    if frame_count is None:
        print(f"  frames     : {frame_source}")
    else:
        print(f"  frames     : {frame_source} ({frame_count} frames)")
    print(f"  device     : {device}")
    print(f"  lower body : {'first-frame stabilized' if stabilize_lower_body else 'dynamic'}")
    print(f"  root orient: {'first-frame stabilized' if stabilize_global_orient else 'dynamic'}")
    print(f"  torso      : {'first-frame stabilized' if stabilize_torso else 'dynamic'}")
    print(f"  shape      : {'first-frame stabilized' if stabilize_shape else 'dynamic'}")
    print("#" * 72)

    from video2smplx.runners.emoca import EMOCARunner
    from video2smplx.runners.smplestx import SmplestXRunner
    from video2smplx.runners.wilor import WiLoRRunner

    print("\n[load] SMPLest-X")
    smplestx = SmplestXRunner(
        smplestx_dir,
        ckpt_name=smplestx_ckpt,
        device=device,
        multi_person=multi_person,
    )
    print("\n[load] WiLoR")
    wilor = WiLoRRunner(wilor_dir, device=device)
    print("\n[load] EMOCA")
    emoca = EMOCARunner(emoca_dir, model_name=emoca_model, device=device)

    stats = FusionStats()
    fused_outputs: list[tuple[str, list]] = []
    model_time_totals = {key: 0.0 for key in MODEL_TIMING_KEYS}
    model_time_counts = {key: 0 for key in MODEL_TIMING_KEYS}
    per_frame_timings: list[dict] = []
    lower_body_stabilizer = LowerBodyStabilizer() if stabilize_lower_body else None
    global_orient_stabilizer = (
        GlobalOrientStabilizer() if stabilize_global_orient else None
    )
    torso_stabilizer = TorsoStabilizer() if stabilize_torso else None
    shape_stabilizer = ShapeStabilizer() if stabilize_shape else None

    for frame in tqdm(frame_iterable, total=frame_count, desc="Integrated inference"):
        stats.total_frames += 1
        out_name = f"{frame.frame_id:06d}_params.pkl"
        frame_timings: dict[str, float | None] = {
            key: None for key in MODEL_TIMING_KEYS
        }
        frame_start = time.perf_counter()
        frame_status = "ok"
        try:
            model_start = time.perf_counter()
            try:
                body = smplestx.predict(frame)
            finally:
                frame_timings["smplestx"] = time.perf_counter() - model_start
            if not body:
                frame_status = "empty_smplestx"
                fused_outputs.append((out_name, []))
                stats.empty_smplestx_frames += 1
                continue
            if lower_body_stabilizer is not None:
                body = lower_body_stabilizer.apply(body, frame.frame_id)
            if global_orient_stabilizer is not None:
                body = global_orient_stabilizer.apply(body, frame.frame_id)
            if torso_stabilizer is not None:
                body = torso_stabilizer.apply(body, frame.frame_id)
            if shape_stabilizer is not None:
                body = shape_stabilizer.apply(body, frame.frame_id)

            model_start = time.perf_counter()
            try:
                hands = wilor.predict(frame)
            finally:
                frame_timings["wilor"] = time.perf_counter() - model_start
            if _has_hand_data(hands):
                stats.wilor_matched += 1
            else:
                stats.wilor_missing += 1

            model_start = time.perf_counter()
            try:
                face = emoca.predict(frame)
            finally:
                frame_timings["emoca"] = time.perf_counter() - model_start
            if _has_face_data(face):
                stats.emoca_matched += 1
            else:
                stats.emoca_missing += 1

            fused = fuse_frame(body, hands, face)
            validation_errors = validate_person(fused[0])
            if validation_errors:
                stats.validation_errors += len(validation_errors)
                stats.warnings.append(
                    f"{out_name}: {'; '.join(validation_errors)}"
                )

            fused_outputs.append((out_name, fused))
        except Exception as exc:
            frame_status = "error"
            stats.errors += 1
            stats.warnings.append(f"{frame.frame_id:06d}: {type(exc).__name__}: {exc}")
            fused_outputs.append((out_name, []))
        finally:
            total_seconds = time.perf_counter() - frame_start
            for key, seconds in frame_timings.items():
                if seconds is None:
                    continue
                model_time_totals[key] += seconds
                model_time_counts[key] += 1
            per_frame_timings.append(
                {
                    "frame_id": frame.frame_id,
                    "file": out_name,
                    "status": frame_status,
                    "timings_sec": frame_timings,
                    "total_sec": total_seconds,
                }
            )
            tqdm.write(
                "[timing] "
                f"frame={frame.frame_id:06d} "
                f"smplestx={_format_timing_ms(frame_timings['smplestx'])} "
                f"wilor={_format_timing_ms(frame_timings['wilor'])} "
                f"emoca={_format_timing_ms(frame_timings['emoca'])} "
                f"total={_format_timing_ms(total_seconds)} "
                f"status={frame_status}"
            )

    if stats.total_frames == 0:
        raise RuntimeError(f"No frames decoded from: {video}")

    for out_name, fused in fused_outputs:
        with open(fused_dir / out_name, "wb") as f:
            pickle.dump(fused, f)

    with open(report_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    timing_average_sec = {
        key: _timing_average(model_time_totals[key], model_time_counts[key])
        for key in MODEL_TIMING_KEYS
    }
    timing_report = {
        "frames": stats.total_frames,
        "model_counts": model_time_counts,
        "model_totals_sec": model_time_totals,
        "model_averages_sec": timing_average_sec,
        "model_averages_ms": {
            key: None if seconds is None else seconds * 1000
            for key, seconds in timing_average_sec.items()
        },
        "per_frame": per_frame_timings,
    }
    with open(timing_report_path, "w") as f:
        json.dump(timing_report, f, indent=2)

    valid_fused_frames = sum(
        1 for _, fused in fused_outputs if _is_valid_fused_frame(fused)
    )
    print("\n[fusion]")
    print(f"  Frames decoded     : {stats.total_frames}")
    print(f"  Valid fused frames : {valid_fused_frames}")
    print(f"  Empty body frames  : {stats.empty_smplestx_frames}")
    print(f"  Frame errors       : {stats.errors}")
    print(f"  Fusion report      : {report_path}")
    print(f"  Timing report      : {timing_report_path}")
    if lower_body_stabilizer is not None:
        print(f"  Lower-body fixed   : {lower_body_stabilizer.applied_frames} frames")
    if global_orient_stabilizer is not None:
        print(f"  Root orient fixed  : {global_orient_stabilizer.applied_frames} frames")
    if torso_stabilizer is not None:
        print(f"  Torso fixed        : {torso_stabilizer.applied_frames} frames")
    if shape_stabilizer is not None:
        print(f"  Shape fixed        : {shape_stabilizer.applied_frames} frames")

    print("\n[timing averages]")
    print(
        "  "
        + "  ".join(
            f"{key}={_format_timing_ms(timing_average_sec[key])}"
            f" (n={model_time_counts[key]})"
            for key in MODEL_TIMING_KEYS
        )
    )

    if valid_fused_frames == 0:
        raise RuntimeError(
            "Integrated inference produced 0 valid fused frames, so render/smooth was not run.\n"
            f"Fusion report: {report_path}\n"
            "First recorded warnings/errors:"
            f"{_warning_sample(stats.warnings)}"
        )

    final_video = rendered_dir / "smplest_wilor_emoca.mp4"
    if not skip_render:
        from zero_filter_render import (
            stage_render as render_smplx_video,
            stage_smooth,
            stage_zero_transl,
        )

        rendered_dir.mkdir(parents=True, exist_ok=True)
        render_data = copy.deepcopy([fused for _, fused in fused_outputs])
        render_data = stage_zero_transl(render_data)
        effective_smooth_window = _effective_smooth_window(
            smooth_window,
            valid_fused_frames,
            smooth_poly,
        )
        if effective_smooth_window is None:
            print(
                "\n[render] Skipping smoothing because there are not enough "
                f"valid frames ({valid_fused_frames}) for polyorder={smooth_poly}."
            )
        else:
            if effective_smooth_window != smooth_window:
                print(
                    "\n[render] Adjusting smooth_window from "
                    f"{smooth_window} to {effective_smooth_window} for "
                    f"{valid_fused_frames} valid frames."
                )
            render_data = stage_smooth(
                render_data,
                effective_smooth_window,
                smooth_poly,
            )
        params_out_dir = rendered_dir / "params"
        params_out_dir.mkdir(parents=True, exist_ok=True)
        for (out_name, _), data in zip(fused_outputs, render_data):
            with open(params_out_dir / out_name, "wb") as f:
                pickle.dump(data, f)
        render_smplx_video(
            all_data=render_data,
            smplx_model_path=str(smplx_model),
            output_video_path=str(final_video),
            video_fps=fps,
            viewport_size=viewport,
        )

    summary = {
        "name": run_name,
        "frames": stats.total_frames,
        "output": str(output),
        "fused_params": str(fused_dir),
        "fusion_report": str(report_path),
        "timing_report": str(timing_report_path),
        "rendered_video": str(final_video) if not skip_render else None,
        "valid_fused_frames": valid_fused_frames,
        "stats": stats.to_dict(),
        "timings": timing_report,
        "lower_body_stabilization": {
            "enabled": stabilize_lower_body,
            "reference_frame_id": (
                lower_body_stabilizer.reference_frame_id
                if lower_body_stabilizer is not None
                else None
            ),
            "applied_frames": (
                lower_body_stabilizer.applied_frames
                if lower_body_stabilizer is not None
                else 0
            ),
        },
        "global_orient_stabilization": {
            "enabled": stabilize_global_orient,
            "reference_frame_id": (
                global_orient_stabilizer.reference_frame_id
                if global_orient_stabilizer is not None
                else None
            ),
            "applied_frames": (
                global_orient_stabilizer.applied_frames
                if global_orient_stabilizer is not None
                else 0
            ),
        },
        "torso_stabilization": {
            "enabled": stabilize_torso,
            "reference_frame_id": (
                torso_stabilizer.reference_frame_id
                if torso_stabilizer is not None
                else None
            ),
            "applied_frames": (
                torso_stabilizer.applied_frames
                if torso_stabilizer is not None
                else 0
            ),
        },
        "shape_stabilization": {
            "enabled": stabilize_shape,
            "reference_frame_id": (
                shape_stabilizer.reference_frame_id
                if shape_stabilizer is not None
                else None
            ),
            "applied_frames": (
                shape_stabilizer.applied_frames
                if shape_stabilizer is not None
                else 0
            ),
        },
    }
    print("\n" + "#" * 72)
    print("  INTEGRATED PIPELINE COMPLETE")
    print(f"  Fused params   : {fused_dir}")
    print(f"  Fusion report  : {report_path}")
    print(f"  Timing report  : {timing_report_path}")
    if not skip_render:
        print(f"  Rendered video : {final_video}")
    print("#" * 72)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SMPLest-X, WiLoR, and EMOCA as in-process runners.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--name", default=None, help="Run name.")
    parser.add_argument("--fps", type=int, default=30, help="Extraction/output FPS.")
    parser.add_argument("--frames_dir", default=None, help="Optional extracted frame directory.")
    parser.add_argument("--reuse_frames", action="store_true", help="Use --frames_dir instead of extracting.")
    parser.add_argument(
        "--materialize_frames",
        action="store_true",
        help="Write decoded frames to --frames_dir before inference. Default streams frames in memory.",
    )
    parser.add_argument("--smplestx_ckpt", default="smplest_x_h", help="SMPLest-X checkpoint name.")
    parser.add_argument("--emoca_model", default="EMOCA_v2_lr_mse_20", help="EMOCA model name.")
    parser.add_argument("--smplestx_dir", default=str(SMPLESTX_DIR), help="SMPLest-X project/assets directory.")
    parser.add_argument("--wilor_dir", default=str(WILOR_DIR), help="WiLoR project/assets directory.")
    parser.add_argument("--emoca_dir", default=str(EMOCA_DIR), help="EMOCA project/assets directory.")
    parser.add_argument("--device", default="cuda", help="Device for model inference.")
    parser.add_argument("--multi_person", action="store_true", help="Keep all SMPLest-X people.")
    parser.add_argument(
        "--stabilize_lower_body",
        action="store_true",
        help="Hold primary person's lower-body SMPL-X body_pose joints from the first valid frame.",
    )
    parser.add_argument(
        "--stabilize_global_orient",
        action="store_true",
        help="Hold primary person's SMPL-X root/global orientation from the first valid frame.",
    )
    parser.add_argument(
        "--stabilize_torso",
        action="store_true",
        help="Hold primary person's SMPL-X spine body_pose joints from the first valid frame.",
    )
    parser.add_argument(
        "--stabilize_shape",
        action="store_true",
        help="Hold primary person's SMPL-X shape coefficients from the first valid frame.",
    )
    parser.add_argument("--skip_render", action="store_true", help="Only write fused params.")
    parser.add_argument("--smplx_model", default=str(DEFAULT_SMPLX_MODEL), help="SMPL-X model path for render.")
    parser.add_argument("--smooth_window", type=int, default=15)
    parser.add_argument("--smooth_poly", type=int, default=3)
    parser.add_argument("--viewport", type=int, default=800)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_integrated_pipeline(
        video=Path(args.video),
        output=Path(args.output),
        name=args.name,
        fps=args.fps,
        reuse_frames=args.reuse_frames,
        frames_dir=Path(args.frames_dir) if args.frames_dir else None,
        materialize_frames=args.materialize_frames,
        smplestx_ckpt=args.smplestx_ckpt,
        emoca_model=args.emoca_model,
        device=args.device,
        multi_person=args.multi_person,
        skip_render=args.skip_render,
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
    )


if __name__ == "__main__":
    main()
