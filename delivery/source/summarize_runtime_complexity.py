#!/usr/bin/env python3
"""Summarize empirical runtime complexity for the final postprocessed pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_stage_timings(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "stage": row["stage"],
                    "seconds": float(row["seconds"]),
                    "status": row.get("status", ""),
                }
            )
    return rows


def file_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size / (1024 * 1024)


def checkpoint_param_count(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        total = 0
        for value in state.values():
            if hasattr(value, "numel"):
                total += int(value.numel())
        return total
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create runtime/FPS summary for a final postprocessed run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-root", required=True, help="Root containing the sequence folder.")
    parser.add_argument("--sequence", required=True, help="Sequence name, e.g. SignLanguage_S2.")
    parser.add_argument("--timing-csv", required=True, help="Stage timing CSV written by the Slurm script.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-ckpt", default=None)
    parser.add_argument("--hand-ckpt", default=None)
    parser.add_argument("--upper2d-ckpt", default=None)
    parser.add_argument("--yolo-model", default=None)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    sequence = args.sequence
    run = run_root / sequence
    final_run = run / "final_postprocessed"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    body_report = read_json(run / "body_report.json")
    hand_report = read_json(run / "hand_report.json")
    face_report = read_json(run / "face_report.json")
    base_combine_report = read_json(run / "base_fusion" / "combine_render_report.json")
    final_combine_report = read_json(final_run / "combine_render_report.json")
    stage_timings = read_stage_timings(Path(args.timing_csv))

    frames = int(
        body_report.get("frames_total")
        or final_combine_report.get("frames_total")
        or base_combine_report.get("frames_total")
        or 0
    )
    if frames <= 0:
        raise ValueError("Could not determine frame count from reports.")

    total_seconds = sum(row["seconds"] for row in stage_timings)
    runtime_rows = []
    for row in stage_timings:
        seconds = row["seconds"]
        runtime_rows.append(
            {
                "stage": row["stage"],
                "seconds": seconds,
                "minutes": seconds / 60.0,
                "seconds_per_frame": seconds / frames,
                "fps": frames / seconds if seconds > 0 else 0.0,
                "percent_total": 100.0 * seconds / total_seconds if total_seconds > 0 else 0.0,
                "status": row["status"],
            }
        )
    runtime_rows.append(
        {
            "stage": "TOTAL",
            "seconds": total_seconds,
            "minutes": total_seconds / 60.0,
            "seconds_per_frame": total_seconds / frames,
            "fps": frames / total_seconds if total_seconds > 0 else 0.0,
            "percent_total": 100.0,
            "status": "ok",
        }
    )

    models = [
        ("global_corrector", Path(args.global_ckpt) if args.global_ckpt else None),
        ("hand_corrector", Path(args.hand_ckpt) if args.hand_ckpt else None),
        ("upper2d_corrector", Path(args.upper2d_ckpt) if args.upper2d_ckpt else None),
        ("yolo_pose_model", Path(args.yolo_model) if args.yolo_model else None),
    ]
    model_rows = []
    for name, path in models:
        model_rows.append(
            {
                "model": name,
                "path": str(path.resolve()) if path and path.exists() else str(path) if path else "",
                "file_size_mb": file_size_mb(path),
                "parameter_count": checkpoint_param_count(path),
            }
        )

    outputs = {
        "smplx_params_npz": str(final_run / "smplx_params.npz"),
        "rendered_video": str(final_run / "rendered" / "smplx_render.mp4"),
        "side_by_side_video": str(final_run / "side_by_side_input_render.mp4"),
    }
    output_sizes = {
        key: file_size_mb(Path(value))
        for key, value in outputs.items()
    }

    summary = {
        "sequence": sequence,
        "frames": frames,
        "run_root": str(run_root),
        "final_run": str(final_run),
        "runtime_complexity": "empirical wall-clock runtime on the allocated server GPU",
        "stage_timings": runtime_rows,
        "models": model_rows,
        "reports": {
            "body": body_report,
            "hand": hand_report,
            "face": face_report,
            "base_combine": base_combine_report,
            "final_combine": final_combine_report,
        },
        "outputs": outputs,
        "output_sizes_mb": output_sizes,
    }

    write_csv(output_dir / "runtime_stage_summary.csv", runtime_rows)
    write_csv(output_dir / "model_complexity_summary.csv", model_rows)
    (output_dir / "runtime_complexity_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Runtime Complexity Summary",
        "",
        f"Sequence: `{sequence}`",
        f"Frames: `{frames}`",
        "",
        "## Stage Runtime",
        "",
        "| Stage | Seconds | Sec/Frame | FPS | % Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in runtime_rows:
        md_lines.append(
            "| {stage} | {seconds:.2f} | {seconds_per_frame:.4f} | {fps:.2f} | {percent_total:.1f} |".format(
                **row
            )
        )
    md_lines.extend(
        [
            "",
            "## Model/Corrector Size",
            "",
            "| Model | File Size MB | Parameter Count |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in model_rows:
        size = "" if row["file_size_mb"] is None else f"{row['file_size_mb']:.2f}"
        params = "" if row["parameter_count"] is None else str(row["parameter_count"])
        md_lines.append(f"| {row['model']} | {size} | {params} |")
    md_lines.extend(
        [
            "",
            "## Final Outputs",
            "",
            f"- SMPL-X NPZ: `{outputs['smplx_params_npz']}`",
            f"- Rendered video: `{outputs['rendered_video']}`",
            f"- Side-by-side video: `{outputs['side_by_side_video']}`",
        ]
    )
    (output_dir / "runtime_complexity_summary.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
