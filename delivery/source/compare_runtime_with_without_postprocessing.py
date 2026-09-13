#!/usr/bin/env python3
"""Compare runtime/FPS with and without post-processing from fair benchmark runs."""

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seconds_by_stage(sequence_dir: Path) -> dict[str, float]:
    path = sequence_dir / "runtime" / "fair_stage_timings.csv"
    if not path.exists():
        path = sequence_dir / "runtime" / "stage_timings.csv"
    rows = read_csv(path)
    return {row["stage"]: float(row["seconds"]) for row in rows}


def frames_for_sequence(sequence_dir: Path) -> int:
    reports = [
        sequence_dir / "body_report.json",
        sequence_dir / "base_no_post" / "combine_render_report.json",
        sequence_dir / "final_postprocessed" / "combine_render_report.json",
    ]
    for path in reports:
        data = read_json(path)
        frames = int(data.get("frames_total") or 0)
        if frames > 0:
            return frames
    raise ValueError(f"Could not determine frame count for {sequence_dir}")


def metric_row(sequence: str, frames: int, label: str, seconds: float) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "frames": frames,
        "runtime_type": label,
        "seconds": seconds,
        "minutes": seconds / 60.0,
        "seconds_per_frame": seconds / frames if frames else 0.0,
        "fps": frames / seconds if seconds > 0 else 0.0,
    }


def summarize_sequence(run_root: Path, sequence: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequence_dir = run_root / sequence
    frames = frames_for_sequence(sequence_dir)
    stages = seconds_by_stage(sequence_dir)

    body = stages.get("body_smplestx", 0.0)
    hand = stages.get("hand_wilor", 0.0)
    face = stages.get("face_emoca", 0.0)
    base_no_render = stages.get("base_combine_smooth_npz_no_render", 0.0)
    base_render = stages.get("base_render_npz_side_by_side", 0.0)
    global_corr = stages.get("correct_global_orient_transl", 0.0)
    hand_corr = stages.get("correct_hand_wrist_fingers", 0.0)
    yolo = stages.get("extract_yolo_pose_2d", 0.0)
    upper = stages.get("correct_2d_guided_upper_body", 0.0)
    final_render = stages.get("final_render_npz_side_by_side", 0.0)

    base_inference = body + hand + face + base_no_render
    base_with_render = base_inference + base_render
    post_overhead = global_corr + hand_corr + yolo + upper
    with_post_no_render = base_inference + post_overhead
    with_post_with_render = with_post_no_render + final_render

    rows = [
        metric_row(sequence, frames, "without_post_no_render", base_inference),
        metric_row(sequence, frames, "without_post_with_render", base_with_render),
        metric_row(sequence, frames, "postprocessing_overhead_only", post_overhead),
        metric_row(sequence, frames, "with_post_no_render", with_post_no_render),
        metric_row(sequence, frames, "with_post_with_render", with_post_with_render),
    ]
    stage_rows = [
        {"sequence": sequence, "frames": frames, "stage": stage, "seconds": seconds}
        for stage, seconds in stages.items()
    ]
    return rows, stage_rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, dict[str, float]] = {}
    for row in rows:
        item = by_type.setdefault(row["runtime_type"], {"frames": 0.0, "seconds": 0.0})
        item["frames"] += float(row["frames"])
        item["seconds"] += float(row["seconds"])
    output = []
    for runtime_type, item in by_type.items():
        output.append(
            metric_row(
                "ALL",
                int(item["frames"]),
                runtime_type,
                item["seconds"],
            )
        )
    return output


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Sequence | Runtime Type | Frames | Seconds | Sec/Frame | FPS |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {sequence} | {runtime_type} | {frames} | {seconds:.2f} | {seconds_per_frame:.4f} | {fps:.2f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for sequence in args.sequences:
        rows, stages = summarize_sequence(run_root, sequence)
        summary_rows.extend(rows)
        stage_rows.extend(stages)
    all_rows = aggregate(summary_rows)
    final_rows = summary_rows + all_rows

    write_csv(output_dir / "runtime_with_without_postprocessing.csv", final_rows)
    write_csv(output_dir / "runtime_stage_breakdown_fair.csv", stage_rows)
    payload = {
        "run_root": str(run_root),
        "sequences": args.sequences,
        "summary": final_rows,
        "stages": stage_rows,
        "interpretation": {
            "without_post_no_render": "body + hand + face + base combine/smooth/NPZ, no rendering",
            "without_post_with_render": "without_post_no_render + base render/side-by-side",
            "postprocessing_overhead_only": "global corrector + hand corrector + YOLO 2D pose + 2D upper-body corrector",
            "with_post_no_render": "without_post_no_render + postprocessing_overhead_only",
            "with_post_with_render": "with_post_no_render + final render/side-by-side",
        },
    }
    (output_dir / "runtime_with_without_postprocessing.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    md = "\n".join(
        [
            "# Runtime With vs Without Post-Processing",
            "",
            markdown_table(final_rows),
            "",
            "## Definitions",
            "",
            "- `without_post_no_render`: body + hand + face + base combine/smooth/NPZ.",
            "- `without_post_with_render`: same as above plus base SMPL-X render and side-by-side video.",
            "- `postprocessing_overhead_only`: global corrector + hand corrector + YOLO 2D pose + 2D-guided upper-body corrector.",
            "- `with_post_no_render`: base no-render pipeline plus post-processing.",
            "- `with_post_with_render`: post-processed pipeline plus final SMPL-X render and side-by-side video.",
            "",
            "Use `with_post_no_render` for real prediction throughput if rendering is not part of deployment.",
            "Use `with_post_with_render` for full deliverable artifact generation.",
            "",
        ]
    )
    (output_dir / "runtime_with_without_postprocessing.md").write_text(md, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
