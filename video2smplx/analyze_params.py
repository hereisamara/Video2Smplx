"""Inspect temporal variation in fused SMPL-X parameter folders."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np

from video2smplx.fusion import extract_frame_id, flattened_array
from video2smplx.stabilization import LOWER_BODY_JOINT_INDICES, TORSO_JOINT_INDICES


BODY_JOINT_GROUPS = {
    "body_pose_lower": LOWER_BODY_JOINT_INDICES,
    "body_pose_torso": TORSO_JOINT_INDICES,
    "body_pose_neck_head": (11, 14),
    "body_pose_collar_shoulder": (12, 13, 15, 16),
    "body_pose_arms": (17, 18, 19, 20),
}


def _person_from_file(path: Path) -> dict[str, Any] | None:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if data and isinstance(data[0], dict):
        return data[0]
    return None


def _flat_key(person: dict[str, Any], key: str, expected_dim: int | None = None) -> np.ndarray | None:
    if key not in person:
        return None
    values = flattened_array(person[key]).astype(np.float64)
    if expected_dim is not None and values.shape[0] != expected_dim:
        return None
    return values


def _body_pose_group(person: dict[str, Any], joint_indices: tuple[int, ...]) -> np.ndarray | None:
    body_pose = _flat_key(person, "body_pose", expected_dim=63)
    if body_pose is None:
        return None
    return body_pose.reshape(21, 3)[list(joint_indices)].reshape(-1)


def _collect_series(
    people: list[tuple[int, dict[str, Any]]],
    extractor: Callable[[dict[str, Any]], np.ndarray | None],
) -> tuple[list[int], np.ndarray | None]:
    frame_ids: list[int] = []
    values: list[np.ndarray] = []
    for frame_id, person in people:
        value = extractor(person)
        if value is None:
            continue
        frame_ids.append(frame_id)
        values.append(value)
    if not values:
        return frame_ids, None
    return frame_ids, np.stack(values, axis=0)


def _series_stats(frame_ids: list[int], values: np.ndarray, eps: float) -> dict[str, Any]:
    delta_from_first = values - values[0]
    l2_from_first = np.linalg.norm(delta_from_first, axis=1)
    max_abs_by_frame = np.max(np.abs(delta_from_first), axis=1)
    changed = np.where(max_abs_by_frame > eps)[0]

    if values.shape[0] > 1:
        step_delta = np.diff(values, axis=0)
        l2_step = np.linalg.norm(step_delta, axis=1)
        max_abs_step = np.max(np.abs(step_delta), axis=1)
        max_l2_step = float(np.max(l2_step))
        mean_l2_step = float(np.mean(l2_step))
        max_abs_step_value = float(np.max(max_abs_step))
    else:
        max_l2_step = 0.0
        mean_l2_step = 0.0
        max_abs_step_value = 0.0

    return {
        "frames_with_value": len(frame_ids),
        "dim": int(values.shape[1]),
        "max_abs_delta_from_first": float(np.max(max_abs_by_frame)),
        "max_l2_delta_from_first": float(np.max(l2_from_first)),
        "mean_l2_delta_from_first": float(np.mean(l2_from_first)),
        "max_abs_step": max_abs_step_value,
        "max_l2_step": max_l2_step,
        "mean_l2_step": mean_l2_step,
        "changed_frames": int(changed.shape[0]),
        "first_changed_frame_id": frame_ids[int(changed[0])] if changed.shape[0] else None,
    }


def analyze_params_dir(params_dir: Path, eps: float = 1e-7) -> dict[str, Any]:
    params_dir = Path(params_dir).resolve()
    files = sorted(params_dir.glob("*.pkl"))
    people: list[tuple[int, dict[str, Any]]] = []
    empty_frames = 0

    for idx, path in enumerate(files, start=1):
        person = _person_from_file(path)
        if person is None:
            empty_frames += 1
            continue
        frame_id = extract_frame_id(path.name) or idx
        people.append((frame_id, person))

    extractors: list[tuple[str, Callable[[dict[str, Any]], np.ndarray | None]]] = [
        ("global_orient", lambda p: _flat_key(p, "global_orient", 3)),
        ("transl", lambda p: _flat_key(p, "transl", 3)),
        ("betas", lambda p: _flat_key(p, "betas", 10)),
        ("body_pose_all", lambda p: _flat_key(p, "body_pose", 63)),
        ("left_hand_pose", lambda p: _flat_key(p, "left_hand_pose", 45)),
        ("right_hand_pose", lambda p: _flat_key(p, "right_hand_pose", 45)),
        ("jaw_pose", lambda p: _flat_key(p, "jaw_pose", 3)),
        ("expression", lambda p: _flat_key(p, "expression")),
    ]
    for group_name, joint_indices in BODY_JOINT_GROUPS.items():
        extractors.append((group_name, lambda p, idxs=joint_indices: _body_pose_group(p, idxs)))

    groups: dict[str, Any] = {}
    for name, extractor in extractors:
        frame_ids, values = _collect_series(people, extractor)
        if values is None:
            groups[name] = {"frames_with_value": 0}
            continue
        groups[name] = _series_stats(frame_ids, values, eps)

    candidates = []
    for name, stats in groups.items():
        if stats.get("frames_with_value", 0) == 0:
            continue
        if stats.get("max_abs_delta_from_first", 0.0) > eps:
            candidates.append(
                {
                    "name": name,
                    "max_abs_delta_from_first": stats["max_abs_delta_from_first"],
                    "max_l2_step": stats["max_l2_step"],
                    "first_changed_frame_id": stats["first_changed_frame_id"],
                }
            )
    candidates.sort(key=lambda item: item["max_l2_step"], reverse=True)

    return {
        "params_dir": str(params_dir),
        "pkl_files": len(files),
        "valid_frames": len(people),
        "empty_frames": empty_frames,
        "eps": eps,
        "groups": groups,
        "motion_source_candidates": candidates,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("\nPARAMETER MOTION ANALYSIS")
    print(f"  params_dir   : {report['params_dir']}")
    print(f"  pkl files    : {report['pkl_files']}")
    print(f"  valid frames : {report['valid_frames']}")
    print(f"  empty frames : {report['empty_frames']}")
    print(f"  eps          : {report['eps']}")

    print("\n  group                         frames  max_abs_first  max_l2_step  changed  first_changed")
    print("  " + "-" * 88)
    for name, stats in report["groups"].items():
        if stats.get("frames_with_value", 0) == 0:
            print(f"  {name:<29} {'0':>6}  missing")
            continue
        first_changed = stats["first_changed_frame_id"]
        first_changed_text = "-" if first_changed is None else str(first_changed)
        print(
            f"  {name:<29} "
            f"{stats['frames_with_value']:>6}  "
            f"{stats['max_abs_delta_from_first']:>13.6g}  "
            f"{stats['max_l2_step']:>11.6g}  "
            f"{stats['changed_frames']:>7}  "
            f"{first_changed_text:>13}"
        )

    print("\n  largest frame-to-frame movers")
    print("  " + "-" * 40)
    for item in report["motion_source_candidates"][:8]:
        print(
            f"  {item['name']:<29} "
            f"step={item['max_l2_step']:.6g}  "
            f"first_delta={item['max_abs_delta_from_first']:.6g}  "
            f"first_changed={item['first_changed_frame_id']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze temporal variation in fused/rendered SMPL-X parameter PKLs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--params_dir", required=True, help="Directory containing frame *_params.pkl files.")
    parser.add_argument("--eps", type=float, default=1e-7, help="Threshold used to count a frame as changed.")
    parser.add_argument("--json_output", default=None, help="Optional path to write the full analysis JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = analyze_params_dir(Path(args.params_dir), eps=args.eps)
    _print_report(report)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
