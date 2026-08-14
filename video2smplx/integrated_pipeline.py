"""One-process Video2Smplx pipeline.

This path loads SMPLest-X, WiLoR, and EMOCA once as Python runner objects,
then performs body/hand/face inference and fusion per frame in memory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
PARALLEL_MODEL_WORKERS = len(MODEL_TIMING_KEYS)


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


def _add_step_timing(
    step_timings: list[dict],
    name: str,
    start_time: float,
    metadata: dict | None = None,
) -> float:
    seconds = time.perf_counter() - start_time
    step_timings.append(
        {
            "name": name,
            "seconds": seconds,
            "milliseconds": seconds * 1000,
            "metadata": metadata or {},
        }
    )
    return seconds


def _iter_batches(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _timed_predict(predict_fn, frame) -> dict:
    start_time = time.perf_counter()
    try:
        return {
            "result": predict_fn(frame),
            "seconds": time.perf_counter() - start_time,
            "exception": None,
        }
    except Exception as exc:
        return {
            "result": None,
            "seconds": time.perf_counter() - start_time,
            "exception": exc,
        }


def _predict_frame_parallel(executor, smplestx, wilor, emoca, frame):
    futures = {
        "smplestx": executor.submit(_timed_predict, smplestx.predict, frame),
        "wilor": executor.submit(_timed_predict, wilor.predict, frame),
        "emoca": executor.submit(_timed_predict, emoca.predict, frame),
    }
    results = {}
    timings = {}
    errors = []
    for key in MODEL_TIMING_KEYS:
        prediction = futures[key].result()
        timings[key] = prediction["seconds"]
        if prediction["exception"] is not None:
            errors.append(f"{key}: {type(prediction['exception']).__name__}: {prediction['exception']}")
            continue
        results[key] = prediction["result"]

    if errors:
        raise RuntimeError("; ".join(errors))

    return results["smplestx"], results["wilor"], results["emoca"], timings


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
    smplestx_detector_stride: int = 1,
    smplestx_batch_size: int = 1,
    smplestx_inference_mode: bool = False,
    smplestx_single_gpu_model: bool = False,
    parallel_models: bool = False,
    save_raw_smplestx: bool = False,
    smplestx_only: bool = False,
) -> dict:
    from tqdm import tqdm

    if smplestx_only:
        save_raw_smplestx = True
        parallel_models = False
        skip_render = True

    pipeline_start = time.perf_counter()
    step_timings: list[dict] = []
    step_start = pipeline_start
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
    raw_smplestx_dir = output / "smplestx_params"
    rendered_dir = output / "rendered"
    report_dir = raw_smplestx_dir if smplestx_only else fused_dir
    report_path = report_dir / ("smplestx_report.json" if smplestx_only else "fusion_report.json")
    timing_report_path = report_dir / "model_timing_report.json"

    if not video.exists() and not reuse_frames:
        raise FileNotFoundError(f"Video not found: {video}")
    _add_step_timing(
        step_timings,
        "setup_paths",
        step_start,
        {"output": str(output), "video": str(video)},
    )

    step_start = time.perf_counter()
    if not smplestx_only:
        fused_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_smplestx:
        raw_smplestx_dir.mkdir(parents=True, exist_ok=True)
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
    _add_step_timing(
        step_timings,
        "prepare_frames",
        step_start,
        {
            "mode": (
                "reuse_frames"
                if reuse_frames
                else "materialize_frames"
                if materialize_frames
                else "stream_frames"
            ),
            "frame_count": frame_count,
            "frame_source": frame_source,
        },
    )

    print("\n" + "#" * 72)
    print("  INTEGRATED VIDEO2SMPLX PIPELINE")
    print("#" * 72)
    print(f"  name       : {run_name}")
    print(f"  video      : {video}")
    print(f"  output     : {output}")
    print(f"  smplestx   : {smplestx_dir}")
    print(f"  mode       : {'SMPLest-X only' if smplestx_only else 'SMPLest-X + WiLoR + EMOCA fusion'}")
    if not smplestx_only:
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
    print(f"  sx det str : {max(1, smplestx_detector_stride)}")
    print(f"  sx batch   : {max(1, smplestx_batch_size)}")
    print(f"  sx infer   : {'inference_mode' if smplestx_inference_mode else 'no_grad'}")
    print(f"  sx model   : {'single GPU module' if smplestx_single_gpu_model else 'DataParallel wrapper'}")
    print(f"  model exec : {'parallel per frame' if parallel_models else 'sequential/batched'}")
    if smplestx_only:
        print("  note       : WiLoR/EMOCA/fusion/render are skipped")
    if parallel_models and smplestx_batch_size > 1:
        print("  note       : --parallel_models runs SMPLest-X one frame at a time")
    print("#" * 72)

    step_start = time.perf_counter()
    from video2smplx.runners.smplestx import SmplestXRunner
    if not smplestx_only:
        from video2smplx.runners.emoca import EMOCARunner
        from video2smplx.runners.wilor import WiLoRRunner
    _add_step_timing(step_timings, "import_runners", step_start)

    print("\n[load] SMPLest-X")
    step_start = time.perf_counter()
    smplestx = SmplestXRunner(
        smplestx_dir,
        ckpt_name=smplestx_ckpt,
        device=device,
        multi_person=multi_person,
        detector_stride=smplestx_detector_stride,
        use_inference_mode=smplestx_inference_mode,
        single_gpu_model=smplestx_single_gpu_model,
        model_batch_size=smplestx_batch_size,
    )
    _add_step_timing(step_timings, "load_smplestx", step_start)
    wilor = None
    emoca = None
    if not smplestx_only:
        print("\n[load] WiLoR")
        step_start = time.perf_counter()
        wilor = WiLoRRunner(wilor_dir, device=device)
        _add_step_timing(step_timings, "load_wilor", step_start)
        print("\n[load] EMOCA")
        step_start = time.perf_counter()
        emoca = EMOCARunner(emoca_dir, model_name=emoca_model, device=device)
        _add_step_timing(step_timings, "load_emoca", step_start)

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
    smplestx_batch_size = max(1, smplestx_batch_size)

    def _apply_stabilizers(body, frame_id: int):
        if lower_body_stabilizer is not None:
            body = lower_body_stabilizer.apply(body, frame_id)
        if global_orient_stabilizer is not None:
            body = global_orient_stabilizer.apply(body, frame_id)
        if torso_stabilizer is not None:
            body = torso_stabilizer.apply(body, frame_id)
        if shape_stabilizer is not None:
            body = shape_stabilizer.apply(body, frame_id)
        return body

    def _record_frame_timing(
        frame,
        out_name: str,
        frame_status: str,
        frame_timings: dict[str, float | None],
        total_seconds: float,
        metadata: dict | None = None,
    ) -> None:
        for key, seconds in frame_timings.items():
            if seconds is None:
                continue
            model_time_totals[key] += seconds
            model_time_counts[key] += 1

        timing_record = {
            "frame_id": frame.frame_id,
            "file": out_name,
            "status": frame_status,
            "timings_sec": frame_timings,
            "total_sec": total_seconds,
            "parallel_models": parallel_models,
        }
        if metadata:
            timing_record.update(metadata)
        per_frame_timings.append(timing_record)
        tqdm.write(
            "[timing] "
            f"frame={frame.frame_id:06d} "
            f"smplestx={_format_timing_ms(frame_timings['smplestx'])} "
            f"wilor={_format_timing_ms(frame_timings['wilor'])} "
            f"emoca={_format_timing_ms(frame_timings['emoca'])} "
            f"total={_format_timing_ms(total_seconds)} "
            f"status={frame_status}"
        )
        progress.update(1)

    inference_start = time.perf_counter()
    raw_smplestx_outputs: list[tuple[str, list]] = []
    progress = tqdm(total=frame_count, desc="Integrated inference")
    try:
        if parallel_models:
            with ThreadPoolExecutor(max_workers=PARALLEL_MODEL_WORKERS) as executor:
                for frame in frame_iterable:
                    stats.total_frames += 1
                    out_name = f"{frame.frame_id:06d}_params.pkl"
                    frame_timings: dict[str, float | None] = {
                        key: None for key in MODEL_TIMING_KEYS
                    }
                    frame_start = time.perf_counter()
                    frame_status = "ok"
                    try:
                        body, hands, face, parallel_timings = _predict_frame_parallel(
                            executor,
                            smplestx,
                            wilor,
                            emoca,
                            frame,
                        )
                        frame_timings.update(parallel_timings)
                        if not body:
                            frame_status = "empty_smplestx"
                            fused_outputs.append((out_name, []))
                            if save_raw_smplestx:
                                raw_smplestx_outputs.append((out_name, []))
                            stats.empty_smplestx_frames += 1
                            continue

                        body = _apply_stabilizers(body, frame.frame_id)
                        if save_raw_smplestx:
                            raw_smplestx_outputs.append((out_name, copy.deepcopy(body)))

                        if _has_hand_data(hands):
                            stats.wilor_matched += 1
                        else:
                            stats.wilor_missing += 1

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
                        stats.warnings.append(
                            f"{frame.frame_id:06d}: {type(exc).__name__}: {exc}"
                        )
                        fused_outputs.append((out_name, []))
                    finally:
                        _record_frame_timing(
                            frame=frame,
                            out_name=out_name,
                            frame_status=frame_status,
                            frame_timings=frame_timings,
                            total_seconds=time.perf_counter() - frame_start,
                            metadata={
                                "smplestx_batch_size": 1,
                                "smplestx_batch_total_sec": frame_timings["smplestx"],
                            },
                        )
        else:
            for frame_batch in _iter_batches(frame_iterable, smplestx_batch_size):
                batch_start = time.perf_counter()
                batch_error: Exception | None = None
                try:
                    body_batch = smplestx.predict_batch(frame_batch)
                    if len(body_batch) != len(frame_batch):
                        raise RuntimeError(
                            "SMPLest-X batch result length mismatch: "
                            f"{len(body_batch)} results for {len(frame_batch)} frames"
                        )
                except Exception as exc:
                    batch_error = exc
                    body_batch = [[] for _ in frame_batch]
                smplestx_seconds = time.perf_counter() - batch_start
                smplestx_seconds_per_frame = smplestx_seconds / len(frame_batch)

                for frame, body in zip(frame_batch, body_batch):
                    stats.total_frames += 1
                    out_name = f"{frame.frame_id:06d}_params.pkl"
                    frame_timings: dict[str, float | None] = {
                        key: None for key in MODEL_TIMING_KEYS
                    }
                    frame_timings["smplestx"] = smplestx_seconds_per_frame
                    frame_start = time.perf_counter()
                    frame_status = "ok"
                    try:
                        if batch_error is not None:
                            raise batch_error
                        if not body:
                            frame_status = "empty_smplestx"
                            if not smplestx_only:
                                fused_outputs.append((out_name, []))
                            if save_raw_smplestx:
                                raw_smplestx_outputs.append((out_name, []))
                            stats.empty_smplestx_frames += 1
                            continue

                        body = _apply_stabilizers(body, frame.frame_id)
                        if save_raw_smplestx:
                            raw_smplestx_outputs.append((out_name, copy.deepcopy(body)))
                        if smplestx_only:
                            frame_status = "ok_smplestx_only"
                            continue

                        model_start = time.perf_counter()
                        try:
                            assert wilor is not None
                            hands = wilor.predict(frame)
                        finally:
                            frame_timings["wilor"] = time.perf_counter() - model_start
                        if _has_hand_data(hands):
                            stats.wilor_matched += 1
                        else:
                            stats.wilor_missing += 1

                        model_start = time.perf_counter()
                        try:
                            assert emoca is not None
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
                        stats.warnings.append(
                            f"{frame.frame_id:06d}: {type(exc).__name__}: {exc}"
                        )
                        if not smplestx_only:
                            fused_outputs.append((out_name, []))
                    finally:
                        total_seconds = (
                            smplestx_seconds_per_frame
                            + time.perf_counter()
                            - frame_start
                        )
                        _record_frame_timing(
                            frame=frame,
                            out_name=out_name,
                            frame_status=frame_status,
                            frame_timings=frame_timings,
                            total_seconds=total_seconds,
                            metadata={
                                "smplestx_batch_size": len(frame_batch),
                                "smplestx_batch_total_sec": smplestx_seconds,
                            },
                        )
    finally:
        progress.close()

    inference_total_sec = time.perf_counter() - inference_start
    step_timings.append(
        {
            "name": "integrated_inference",
            "seconds": inference_total_sec,
            "milliseconds": inference_total_sec * 1000,
            "metadata": {
                "frames": stats.total_frames,
                "smplestx_batch_size": smplestx_batch_size,
                "parallel_models": parallel_models,
            },
        }
    )

    if stats.total_frames == 0:
        raise RuntimeError(f"No frames decoded from: {video}")

    if not smplestx_only:
        step_start = time.perf_counter()
        for out_name, fused in fused_outputs:
            with open(fused_dir / out_name, "wb") as f:
                pickle.dump(fused, f)
        _add_step_timing(
            step_timings,
            "write_fused_params",
            step_start,
            {"files": len(fused_outputs), "directory": str(fused_dir)},
        )

    if save_raw_smplestx:
        step_start = time.perf_counter()
        for out_name, raw_smplestx in raw_smplestx_outputs:
            with open(raw_smplestx_dir / out_name, "wb") as f:
                pickle.dump(raw_smplestx, f)
        _add_step_timing(
            step_timings,
            "write_raw_smplestx_params",
            step_start,
            {"files": len(raw_smplestx_outputs), "directory": str(raw_smplestx_dir)},
        )

    step_start = time.perf_counter()
    with open(report_path, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)
    _add_step_timing(
        step_timings,
        "write_fusion_report",
        step_start,
        {"path": str(report_path)},
    )

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
        "inference_total_sec": inference_total_sec,
        "inference_total_ms": inference_total_sec * 1000,
        "parallel_models": parallel_models,
        "parallel_model_workers": PARALLEL_MODEL_WORKERS if parallel_models else 0,
        "render_total_sec": None,
        "render_total_ms": None,
        "pipeline_total_sec": None,
        "pipeline_total_ms": None,
        "pipeline_steps": step_timings,
        "per_frame": per_frame_timings,
    }
    step_start = time.perf_counter()
    with open(timing_report_path, "w") as f:
        json.dump(timing_report, f, indent=2)
    _add_step_timing(
        step_timings,
        "write_initial_timing_report",
        step_start,
        {"path": str(timing_report_path)},
    )

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
    if save_raw_smplestx:
        print(f"  Raw SMPLest-X      : {raw_smplestx_dir}")
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
    print(f"  inference_total={_format_timing_ms(inference_total_sec)}")

    if smplestx_only:
        step_timings.append(
            {
                "name": "render_total",
                "seconds": 0.0,
                "milliseconds": 0.0,
                "metadata": {"skip_render": True, "reason": "smplestx_only"},
            }
        )
        pipeline_total_sec = time.perf_counter() - pipeline_start
        timing_report["render_total_sec"] = None
        timing_report["render_total_ms"] = None
        timing_report["pipeline_total_sec"] = pipeline_total_sec
        timing_report["pipeline_total_ms"] = pipeline_total_sec * 1000
        timing_report["pipeline_steps"] = step_timings
        step_start = time.perf_counter()
        with open(timing_report_path, "w") as f:
            json.dump(timing_report, f, indent=2)
        _add_step_timing(
            step_timings,
            "write_final_timing_report",
            step_start,
            {"path": str(timing_report_path)},
        )
        timing_report["pipeline_steps"] = step_timings
        with open(timing_report_path, "w") as f:
            json.dump(timing_report, f, indent=2)

        summary = {
            "name": run_name,
            "frames": stats.total_frames,
            "output": str(output),
            "smplestx_params": str(raw_smplestx_dir),
            "smplestx_report": str(report_path),
            "timing_report": str(timing_report_path),
            "stats": stats.to_dict(),
            "timings": timing_report,
            "smplestx_only": True,
            "smplestx_optimization": {
                "detector_stride": max(1, smplestx_detector_stride),
                "batch_size": smplestx_batch_size,
                "inference_mode": smplestx_inference_mode,
                "single_gpu_model": smplestx_single_gpu_model,
            },
        }
        print("\n" + "#" * 72)
        print("  SMPLEST-X ONLY PIPELINE COMPLETE")
        print(f"  SMPLest-X params : {raw_smplestx_dir}")
        print(f"  SMPLest-X report : {report_path}")
        print(f"  Timing report    : {timing_report_path}")
        print(f"  Inference time   : {_format_timing_ms(inference_total_sec)}")
        print(f"  Total time       : {_format_timing_ms(pipeline_total_sec)}")
        print("\n[pipeline steps]")
        for step in step_timings:
            print(f"  {step['name']}: {_format_timing_ms(step['seconds'])}")
        print("#" * 72)
        return summary

    if valid_fused_frames == 0:
        raise RuntimeError(
            "Integrated inference produced 0 valid fused frames, so render/smooth was not run.\n"
            f"Fusion report: {report_path}\n"
            "First recorded warnings/errors:"
            f"{_warning_sample(stats.warnings)}"
        )

    final_video = rendered_dir / "smplest_wilor_emoca.mp4"
    render_total_sec = None
    if not skip_render:
        render_start = time.perf_counter()
        step_start = time.perf_counter()
        from zero_filter_render import (
            stage_render as render_smplx_video,
            stage_smooth,
            stage_zero_transl,
        )
        _add_step_timing(step_timings, "render_import_helpers", step_start)

        step_start = time.perf_counter()
        rendered_dir.mkdir(parents=True, exist_ok=True)
        render_data = copy.deepcopy([fused for _, fused in fused_outputs])
        _add_step_timing(
            step_timings,
            "render_prepare_data",
            step_start,
            {"frames": len(render_data), "directory": str(rendered_dir)},
        )
        step_start = time.perf_counter()
        render_data = stage_zero_transl(render_data)
        _add_step_timing(step_timings, "render_zero_translation", step_start)
        effective_smooth_window = _effective_smooth_window(
            smooth_window,
            valid_fused_frames,
            smooth_poly,
        )
        if effective_smooth_window is None:
            step_timings.append(
                {
                    "name": "render_smoothing",
                    "seconds": 0.0,
                    "milliseconds": 0.0,
                    "metadata": {
                        "skipped": True,
                        "reason": "not enough valid frames",
                        "valid_frames": valid_fused_frames,
                        "smooth_poly": smooth_poly,
                    },
                }
            )
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
            step_start = time.perf_counter()
            render_data = stage_smooth(
                render_data,
                effective_smooth_window,
                smooth_poly,
            )
            _add_step_timing(
                step_timings,
                "render_smoothing",
                step_start,
                {
                    "smooth_window": effective_smooth_window,
                    "smooth_poly": smooth_poly,
                },
            )
        step_start = time.perf_counter()
        params_out_dir = rendered_dir / "params"
        params_out_dir.mkdir(parents=True, exist_ok=True)
        for (out_name, _), data in zip(fused_outputs, render_data):
            with open(params_out_dir / out_name, "wb") as f:
                pickle.dump(data, f)
        _add_step_timing(
            step_timings,
            "render_write_params",
            step_start,
            {"files": len(fused_outputs), "directory": str(params_out_dir)},
        )
        step_start = time.perf_counter()
        render_smplx_video(
            all_data=render_data,
            smplx_model_path=str(smplx_model),
            output_video_path=str(final_video),
            video_fps=fps,
            viewport_size=viewport,
        )
        _add_step_timing(
            step_timings,
            "render_video",
            step_start,
            {"path": str(final_video), "viewport": viewport, "fps": fps},
        )
        render_total_sec = time.perf_counter() - render_start
        step_timings.append(
            {
                "name": "render_total",
                "seconds": render_total_sec,
                "milliseconds": render_total_sec * 1000,
                "metadata": {"skip_render": False},
            }
        )
    else:
        step_timings.append(
            {
                "name": "render_total",
                "seconds": 0.0,
                "milliseconds": 0.0,
                "metadata": {"skip_render": True},
            }
        )

    pipeline_total_sec = time.perf_counter() - pipeline_start
    timing_report["render_total_sec"] = render_total_sec
    timing_report["render_total_ms"] = (
        None if render_total_sec is None else render_total_sec * 1000
    )
    timing_report["pipeline_total_sec"] = pipeline_total_sec
    timing_report["pipeline_total_ms"] = pipeline_total_sec * 1000
    timing_report["pipeline_steps"] = step_timings
    step_start = time.perf_counter()
    with open(timing_report_path, "w") as f:
        json.dump(timing_report, f, indent=2)
    _add_step_timing(
        step_timings,
        "write_final_timing_report",
        step_start,
        {"path": str(timing_report_path)},
    )
    timing_report["pipeline_steps"] = step_timings
    with open(timing_report_path, "w") as f:
        json.dump(timing_report, f, indent=2)

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
        "smplestx_optimization": {
            "detector_stride": max(1, smplestx_detector_stride),
            "batch_size": smplestx_batch_size,
            "inference_mode": smplestx_inference_mode,
            "single_gpu_model": smplestx_single_gpu_model,
        },
        "parallel_models": parallel_models,
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
    print(f"  Inference time : {_format_timing_ms(inference_total_sec)}")
    if render_total_sec is not None:
        print(f"  Render time    : {_format_timing_ms(render_total_sec)}")
    print(f"  Total time     : {_format_timing_ms(pipeline_total_sec)}")
    print("\n[pipeline steps]")
    for step in step_timings:
        print(f"  {step['name']}: {_format_timing_ms(step['seconds'])}")
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
        "--smplestx_detector_stride",
        type=int,
        default=1,
        help=(
            "Run SMPLest-X person detection every N frames and reuse the last "
            "bbox between detector frames. 1 preserves current behavior."
        ),
    )
    parser.add_argument(
        "--smplestx_batch_size",
        type=int,
        default=1,
        help=(
            "Batch this many SMPLest-X crops before model forward. "
            "1 preserves current behavior."
        ),
    )
    parser.add_argument(
        "--smplestx_inference_mode",
        action="store_true",
        help="Use torch.inference_mode() for SMPLest-X forward instead of torch.no_grad().",
    )
    parser.add_argument(
        "--smplestx_single_gpu_model",
        action="store_true",
        help="Use the unwrapped SMPLest-X module after checkpoint load instead of DataParallel.",
    )
    parser.add_argument(
        "--parallel_models",
        action="store_true",
        help=(
            "Run SMPLest-X, WiLoR, and EMOCA predict calls concurrently per frame. "
            "Default keeps the sequential/batched path."
        ),
    )
    parser.add_argument(
        "--save_raw_smplestx",
        action="store_true",
        help="Also save SMPLest-X-only per-frame PKLs to <output>/smplestx_params before fusion.",
    )
    parser.add_argument(
        "--smplestx_only",
        action="store_true",
        help="Run only SMPLest-X and save <output>/smplestx_params; skip WiLoR, EMOCA, fusion, and render.",
    )
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
        smplestx_detector_stride=args.smplestx_detector_stride,
        smplestx_batch_size=args.smplestx_batch_size,
        smplestx_inference_mode=args.smplestx_inference_mode,
        smplestx_single_gpu_model=args.smplestx_single_gpu_model,
        parallel_models=args.parallel_models,
        save_raw_smplestx=args.save_raw_smplestx,
        smplestx_only=args.smplestx_only,
    )


if __name__ == "__main__":
    main()
