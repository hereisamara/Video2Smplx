#!/usr/bin/env python3
"""Evaluate 2D-guided SignLanguage outputs by train/val/test split."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="2d_guided_upperbody_dataset_r1.npz")
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--params-subdirs", nargs="+", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def unique_ordered(values: np.ndarray) -> list[str]:
    seen = set()
    output = []
    for value in values.astype(str).tolist():
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def all_row(summary_csv: Path) -> dict:
    with summary_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sequence"] == "ALL":
                return row
    raise ValueError(f"No ALL row in {summary_csv}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    data = np.load(args.dataset, allow_pickle=True)
    sequences = data["sequence"].astype(str)
    split_names = data["split_name"].astype(str)

    split_sequences = {
        split: unique_ordered(sequences[split_names == split])
        for split in args.splits
    }
    (args.output_root / "split_sequences.json").write_text(json.dumps(split_sequences, indent=2), encoding="utf-8")

    rows = []
    for params_subdir in args.params_subdirs:
        for split, sequence_list in split_sequences.items():
            if not sequence_list:
                continue
            output_dir = args.output_root / f"{params_subdir}_{split}"
            command = [
                sys.executable,
                "evaluate_signlanguage_geometry.py",
                "--pred-root",
                str(args.pred_root),
                "--params-subdir",
                params_subdir,
                "--model-path",
                str(args.model_path),
                "--batch-size",
                str(args.batch_size),
                "--output-dir",
                str(output_dir),
                "--sequences",
                *sequence_list,
            ]
            print(f"[eval] {params_subdir} {split}: {len(sequence_list)} sequences", flush=True)
            subprocess.run(command, check=True)
            row = all_row(output_dir / "geometry_summary.csv")
            row = {"params_subdir": params_subdir, "split": split, "sequence_count": len(sequence_list), **row}
            rows.append(row)

    if rows:
        with (args.output_root / "split_geometry_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        keys = [
            "params_subdir",
            "split",
            "sequence_count",
            "frames",
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
        print(",".join(keys))
        for row in rows:
            print(",".join(str(row.get(key, "")) for key in keys))


if __name__ == "__main__":
    main()
