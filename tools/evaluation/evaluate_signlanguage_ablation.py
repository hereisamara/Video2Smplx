#!/usr/bin/env python3
"""Run SignLanguage ablation evaluations across multiple prediction roots.

Each method is specified as:

    --method "Name|/path/to/pred_root|params_subdir"

The script reads the train/val/test sequence split from a correction dataset
NPZ, calls evaluate_signlanguage_geometry.py for each method/split pair, and
writes a single ablation_summary.csv plus a Markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


METRIC_KEYS = [
    "mpjpe_mm_mean",
    "pa_mpjpe_mm_mean",
    "mpvpe_mm_mean",
    "pa_mpvpe_mm_mean",
    "body_mpjpe_mm_mean",
    "body_pa_mpjpe_mm_mean",
    "hand_mpjpe_mm_mean",
    "hand_pa_mpjpe_mm_mean",
    "visible_upper_mpvpe_mm_mean",
    "hands_wrist_mpvpe_mm_mean",
    "hands_pa_mpvpe_mm_mean",
    "face_head_mpvpe_mm_mean",
    "face_pa_mpvpe_mm_mean",
]

DISPLAY_METRICS = [
    ("MPJPE", "mpjpe_mm_mean"),
    ("PA-MPJPE", "pa_mpjpe_mm_mean"),
    ("MPVPE", "mpvpe_mm_mean"),
    ("PA-MPVPE", "pa_mpvpe_mm_mean"),
    ("Visible Upper MPVPE", "visible_upper_mpvpe_mm_mean"),
    ("Hand Wrist MPVPE", "hands_wrist_mpvpe_mm_mean"),
    ("Hand PA-MPVPE", "hands_pa_mpvpe_mm_mean"),
    ("Face MPVPE", "face_head_mpvpe_mm_mean"),
    ("Face PA-MPVPE", "face_pa_mpvpe_mm_mean"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset NPZ containing sequence/split_name arrays.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method spec: 'Name|pred_root|params_subdir'. Can be passed multiple times.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip method/split pairs whose params are missing instead of failing.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Method name used for delta columns. Default: first evaluated method.",
    )
    return parser.parse_args()


def parse_method(spec: str) -> dict[str, str | Path]:
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(f"Bad --method spec {spec!r}; expected 'Name|pred_root|params_subdir'")
    name, root, subdir = [part.strip() for part in parts]
    if not name or not root or not subdir:
        raise ValueError(f"Bad --method spec {spec!r}; name/root/subdir cannot be empty")
    return {"method": name, "pred_root": Path(root).expanduser(), "params_subdir": subdir}


def unique_ordered(values: np.ndarray) -> list[str]:
    seen = set()
    output = []
    for value in values.astype(str).tolist():
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load_split_sequences(dataset: Path, splits: list[str]) -> dict[str, list[str]]:
    data = np.load(dataset, allow_pickle=True)
    sequences = data["sequence"].astype(str)
    split_names = data["split_name"].astype(str)
    return {split: unique_ordered(sequences[split_names == split]) for split in splits}


def has_any_params(pred_root: Path, params_subdir: str, sequences: list[str]) -> bool:
    for sequence in sequences[:10]:
        params_dir = pred_root / sequence / params_subdir
        if params_dir.is_dir() and any(params_dir.glob("*_params.pkl")):
            return True
    for sequence in sequences:
        params_dir = pred_root / sequence / params_subdir
        if params_dir.is_dir() and any(params_dir.glob("*_params.pkl")):
            return True
    return False


def all_row(summary_csv: Path) -> dict[str, str]:
    with summary_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sequence"] == "ALL":
                return row
    raise ValueError(f"No ALL row in {summary_csv}")


def safe_dir_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        elif char.isspace():
            keep.append("_")
    return "".join(keep).strip("_") or "method"


def evaluate_method_split(
    method: dict[str, str | Path],
    split: str,
    sequences: list[str],
    args: argparse.Namespace,
) -> dict[str, str]:
    method_name = str(method["method"])
    pred_root = Path(method["pred_root"])
    params_subdir = str(method["params_subdir"])
    output_dir = args.output_root / f"{safe_dir_name(method_name)}_{split}"
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_signlanguage_geometry.py")),
        "--pred-root",
        str(pred_root),
        "--params-subdir",
        params_subdir,
        "--model-path",
        str(args.model_path),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(output_dir),
        "--sequences",
        *sequences,
    ]
    print(f"[eval] {method_name} | {split} | {len(sequences)} sequences", flush=True)
    subprocess.run(command, check=True)
    row = all_row(output_dir / "geometry_summary.csv")
    return {
        "method": method_name,
        "split": split,
        "sequence_count": str(len(sequences)),
        "pred_root": str(pred_root),
        "params_subdir": params_subdir,
        **row,
    }


def numeric(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def add_delta_columns(rows: list[dict[str, str]], baseline_name: str | None) -> None:
    if not rows:
        return
    baseline_by_split: dict[str, dict[str, str]] = {}
    if baseline_name is None:
        baseline_name = rows[0]["method"]
    for row in rows:
        if row["method"] == baseline_name:
            baseline_by_split[row["split"]] = row

    for row in rows:
        baseline = baseline_by_split.get(row["split"])
        for key in METRIC_KEYS:
            delta_key = f"delta_vs_{baseline_name}_{key}"
            if baseline is None:
                row[delta_key] = ""
                continue
            current_value = numeric(row, key)
            baseline_value = numeric(baseline, key)
            row[delta_key] = (
                f"{current_value - baseline_value:.6f}"
                if current_value is not None and baseline_value is not None
                else ""
            )


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: str | None) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return str(value)


def write_markdown(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("No rows evaluated.\n", encoding="utf-8")
        return

    lines = ["# SignLanguage Ablation Summary", ""]
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        lines.append(f"## {split}")
        headers = ["Method", "Frames", *[name for name, _ in DISPLAY_METRICS]]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---", "---:"] + ["---:"] * len(DISPLAY_METRICS)) + " |")
        for row in split_rows:
            values = [row["method"], row.get("frames", "")]
            values.extend(fmt(row.get(key)) for _, key in DISPLAY_METRICS)
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    methods = [parse_method(spec) for spec in args.method]
    split_sequences = load_split_sequences(args.dataset, args.splits)
    (args.output_root / "split_sequences.json").write_text(
        json.dumps(split_sequences, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "methods.json").write_text(
        json.dumps([{**m, "pred_root": str(m["pred_root"])} for m in methods], indent=2),
        encoding="utf-8",
    )

    rows = []
    skipped = []
    for method in methods:
        for split, sequences in split_sequences.items():
            if not sequences:
                continue
            if args.skip_missing and not has_any_params(Path(method["pred_root"]), str(method["params_subdir"]), sequences):
                message = {
                    "method": str(method["method"]),
                    "split": split,
                    "pred_root": str(method["pred_root"]),
                    "params_subdir": str(method["params_subdir"]),
                    "reason": "no prediction PKLs found for this split",
                }
                skipped.append(message)
                print(f"[skip] {message}", flush=True)
                continue
            rows.append(evaluate_method_split(method, split, sequences, args))

    add_delta_columns(rows, args.baseline)
    write_csv(rows, args.output_root / "ablation_summary.csv")
    write_markdown(rows, args.output_root / "ablation_summary.md")
    (args.output_root / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    if rows:
        print((args.output_root / "ablation_summary.md").read_text(encoding="utf-8"))
    if skipped:
        print(f"[warn] skipped {len(skipped)} method/split pairs; see skipped.json")
    print(f"[done] {args.output_root}")


if __name__ == "__main__":
    main()
