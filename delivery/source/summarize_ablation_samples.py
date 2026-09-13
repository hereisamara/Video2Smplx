#!/usr/bin/env python3
"""Summarize ablation sample outputs, metrics, and runtime reports."""

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


METRICS = [
    ("MPJPE", "mpjpe_mm_mean"),
    ("PA-MPJPE", "pa_mpjpe_mm_mean"),
    ("MPVPE", "mpvpe_mm_mean"),
    ("PA-MPVPE", "pa_mpvpe_mm_mean"),
    ("Visible Upper MPVPE", "visible_upper_mpvpe_mm_mean"),
    ("Hand Wrist MPVPE", "hands_wrist_mpvpe_mm_mean"),
    ("Hand PA-MPVPE", "hands_pa_mpvpe_mm_mean"),
    ("Face MPVPE", "face_head_mpvpe_mm_mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_all_metrics(evaluation_dir: str) -> dict[str, str]:
    if not evaluation_dir:
        return {}
    path = Path(evaluation_dir) / "geometry_summary.csv"
    if not path.exists():
        return {}
    for row in read_csv(path):
        if row.get("sequence") == "ALL":
            return row
    return {}


def file_size_mb(path_value: str) -> str:
    path = Path(path_value)
    if not path.exists():
        return ""
    return f"{path.stat().st_size / (1024 * 1024):.2f}"


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def summarize_row(row: dict[str, str]) -> dict[str, str]:
    report = read_json(Path(row["runtime_report"]))
    summary = report.get("summary", {})
    steady = summary.get("steady_state", {})
    metrics = read_all_metrics(row.get("evaluation_dir", ""))
    output = {
        **row,
        "frames": str(summary.get("frames", metrics.get("frames", ""))),
        "total_fps": fmt(summary.get("total_fps")),
        "no_initial_load_no_render_fps": fmt(summary.get("no_initial_load_no_render_fps")),
        "steady_state_model_fps": fmt(steady.get("steady_state_model_fps")),
        "smplx_npz_mb": file_size_mb(row["smplx_npz"]),
        "rendered_video_mb": file_size_mb(row["rendered_video"]),
        "side_by_side_video_mb": file_size_mb(row["side_by_side_video"]),
    }
    for _, key in METRICS:
        output[key] = fmt(metrics.get(key))
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Ablation Sample Summary",
        "",
        "Each row is one rendered sample run. Stabilization flags are recorded so the project requirement can be audited.",
        "",
        "## Outputs",
        "",
        "| Sequence | Variant | Postprocess | Stabilized | Frames | No-load No-render FPS | Steady Model FPS | Rendered Video | Side-by-side Video |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        stabilized = "global+shape" if row["stabilize_global_orient"] == "1" or row["stabilize_shape"] == "1" else "none"
        lines.append(
            "| {sequence} | {variant} | {postprocess_mode} | {stabilized} | {frames} | {fps} | {steady} | `{rendered}` | `{side}` |".format(
                sequence=row["sequence"],
                variant=row["variant"],
                postprocess_mode=row["postprocess_mode"],
                stabilized=stabilized,
                frames=row.get("frames", ""),
                fps=row.get("no_initial_load_no_render_fps", ""),
                steady=row.get("steady_state_model_fps", ""),
                rendered=row["rendered_video"],
                side=row["side_by_side_video"],
            )
        )

    metric_rows = [row for row in rows if any(row.get(key) for _, key in METRICS)]
    if metric_rows:
        lines.extend(
            [
                "",
                "## Geometry Metrics",
                "",
                "| Sequence | Variant | MPJPE | PA-MPJPE | MPVPE | PA-MPVPE | Visible Upper MPVPE | Hand Wrist MPVPE | Hand PA-MPVPE | Face MPVPE |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in metric_rows:
            lines.append(
                "| {sequence} | {variant} | {mpjpe} | {pa_mpjpe} | {mpvpe} | {pa_mpvpe} | {upper} | {hand} | {hand_pa} | {face} |".format(
                    sequence=row["sequence"],
                    variant=row["variant"],
                    mpjpe=row.get("mpjpe_mm_mean", ""),
                    pa_mpjpe=row.get("pa_mpjpe_mm_mean", ""),
                    mpvpe=row.get("mpvpe_mm_mean", ""),
                    pa_mpvpe=row.get("pa_mpvpe_mm_mean", ""),
                    upper=row.get("visible_upper_mpvpe_mm_mean", ""),
                    hand=row.get("hands_wrist_mpvpe_mm_mean", ""),
                    hand_pa=row.get("hands_pa_mpvpe_mm_mean", ""),
                    face=row.get("face_head_mpvpe_mm_mean", ""),
                )
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Manifest CSV: `{path.parent / 'ablation_sample_manifest.csv'}`",
            f"- Summary CSV: `{path.parent / 'ablation_sample_summary.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_row(row) for row in read_csv(args.manifest)]
    write_csv(args.output_dir / "ablation_sample_summary.csv", rows)
    write_markdown(args.output_dir / "ablation_sample_summary.md", rows)
    print(f"[done] wrote {args.output_dir / 'ablation_sample_summary.md'}")


if __name__ == "__main__":
    main()
