#!/usr/bin/env python3
"""Print DexAvatar-style regional metrics from a geometry_summary.csv file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEXAVATAR_BENCHMARK_ROWS = [
    ("FrankMoCap", 78.07, 20.47, 19.62),
    ("PIXIE", 60.11, 25.02, 22.42),
    ("PyMAF-X", 68.61, 21.46, 19.19),
    ("SMPLify-SL", 56.07, 22.23, 18.83),
    ("SGNify", 55.63, 19.22, 17.50),
    ("OSX", 47.32, 18.34, 18.12),
    ("Neural Sign Actors", 46.42, 16.17, 15.23),
    ("EVA*", 40.38, 13.73, 13.68),
    ("DexAvatar", 30.13, 13.53, 13.08),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_csv",
        type=Path,
        help="Path to geometry_summary.csv produced by evaluate_signlanguage_geometry.py.",
    )
    parser.add_argument("--label", default="Ours", help="Label for the local result row.")
    return parser.parse_args()


def format_value(value: str | float) -> str:
    return f"{float(value):.2f}"


def main() -> None:
    args = parse_args()
    with args.summary_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    all_rows = [row for row in rows if row["sequence"] == "ALL"]
    if not all_rows:
        raise ValueError(f"No ALL row in {args.summary_csv}")
    row = all_rows[0]

    print("DexAvatar-style regional raw V2V comparison")
    print("Important: published rows are SGNify benchmark TR-V2V; local row is SignLanguage with matching-style regional raw V2V masks.")
    print()
    print("| Method | Dataset | UBody(-F) V2V | LHand V2V | RHand V2V |")
    print("|---|---|---:|---:|---:|")
    for method, upper, left, right in DEXAVATAR_BENCHMARK_ROWS:
        print(f"| {method} | SGNify | {upper:.2f} | {left:.2f} | {right:.2f} |")
    print(
        f"| {args.label} | SignLanguage | "
        f"{format_value(row['dex_ubody_minus_face_tr_v2v_mm_mean'])} | "
        f"{format_value(row['dex_left_hand_tr_v2v_mm_mean'])} | "
        f"{format_value(row['dex_right_hand_tr_v2v_mm_mean'])} |"
    )


if __name__ == "__main__":
    main()
