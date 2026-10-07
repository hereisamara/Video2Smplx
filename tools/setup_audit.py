#!/usr/bin/env python3
"""
Lightweight environment and asset audit for the Video2Smplx pipeline.

This script intentionally avoids importing heavy third-party packages so it can
run before the ML environments are installed.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def heading(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def status_line(label: str, ok: bool, detail: str) -> bool:
    state = "OK" if ok else "MISSING"
    print(f"[{state:7}] {label}: {detail}")
    return ok


def run_command(args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False, "command not found"

    text = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        lines = text.splitlines()
        return False, lines[0] if lines else f"exit code {result.returncode}"
    return True, text or "ok"


def find_conda_envs() -> tuple[bool, list[str], str]:
    conda = shutil.which("conda")
    if not conda:
        return False, [], "conda not found"

    ok, output = run_command(["conda", "env", "list"])
    if not ok:
        return False, [], output

    envs: list[str] = []
    for line in output.splitlines() if "\n" in output else [output]:
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        envs.append(parts[0])
    return True, envs, conda


def check_paths(paths: list[tuple[str, Path, bool]]) -> int:
    missing = 0
    for label, path, required in paths:
        exists = path.exists()
        if not exists and required:
            missing += 1
        kind = "file" if path.suffix else "path"
        optional = "" if required else " (optional)"
        status_line(f"{label}{optional}", exists, f"{kind}: {path}")
    return missing


def main() -> int:
    failures = 0

    heading("System")
    machine = platform.machine()
    system = platform.system()
    python_version = sys.version.split()[0]
    failures += 0 if status_line("OS", system == "Linux", f"{system} {platform.release()}") else 1
    failures += 0 if status_line("Arch", machine in {"x86_64", "AMD64"}, machine) else 1
    failures += 0 if status_line("Python", sys.version_info[:2] in {(3, 8), (3, 10)}, python_version) else 1

    heading("Commands")
    failures += 0 if status_line("conda", shutil.which("conda") is not None, shutil.which("conda") or "not found") else 1
    failures += 0 if status_line("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found") else 1
    failures += 0 if status_line("nvidia-smi", shutil.which("nvidia-smi") is not None, shutil.which("nvidia-smi") or "not found") else 1

    heading("Conda Environments")
    conda_ok, envs, conda_detail = find_conda_envs()
    failures += 0 if status_line("conda env query", conda_ok, conda_detail) else 1
    if conda_ok:
        failures += 0 if status_line("env ubuntu", "ubuntu" in envs, ", ".join(envs) or "no envs listed") else 1
        failures += 0 if status_line("env work38d", "work38d" in envs, ", ".join(envs) or "no envs listed") else 1

    heading("Source Tree")
    failures += check_paths([
        ("pipeline.py", ROOT / "pipeline.py", True),
        ("SMPLest-X entrypoint", ROOT / "SMPLest-X-Inference" / "main" / "inference.py", True),
        ("WiLoR entrypoint", ROOT / "WiLoR-Inference" / "demo_params_unified.py", True),
        ("EMOCA entrypoint", ROOT / "EMOCA-Inference" / "gdl_apps" / "EMOCA" / "demos" / "visualize3.py", True),
    ])

    heading("Model Assets")
    failures += check_paths([
        ("SMPLest-X config", ROOT / "SMPLest-X-Inference" / "pretrained_models" / "smplest_x_h" / "config_base.py", True),
        ("SMPLest-X weights", ROOT / "SMPLest-X-Inference" / "pretrained_models" / "smplest_x_h" / "smplest_x_h.pth.tar", True),
        ("YOLOv8 detector", ROOT / "SMPLest-X-Inference" / "pretrained_models" / "yolov8x.pt", True),
        ("SMPL-X neutral model", ROOT / "SMPLest-X-Inference" / "human_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz", True),
        ("SMPL-X male model", ROOT / "SMPLest-X-Inference" / "human_models" / "human_model_files" / "smplx" / "SMPLX_MALE.npz", False),
        ("SMPL-X female model", ROOT / "SMPLest-X-Inference" / "human_models" / "human_model_files" / "smplx" / "SMPLX_FEMALE.npz", False),
        ("WiLoR checkpoint", ROOT / "WiLoR-Inference" / "pretrained_models" / "wilor_final.ckpt", True),
        ("WiLoR detector", ROOT / "WiLoR-Inference" / "pretrained_models" / "detector.pt", True),
        ("MANO model", ROOT / "WiLoR-Inference" / "mano_data" / "MANO_RIGHT.pkl", True),
        ("MANO mean params", ROOT / "WiLoR-Inference" / "mano_data" / "mano_mean_params.npz", True),
        ("EMOCA model dir", ROOT / "EMOCA-Inference" / "assets" / "EMOCA" / "models" / "EMOCA_v2_lr_mse_20", True),
        ("DECA data", ROOT / "EMOCA-Inference" / "assets" / "DECA" / "data" / "deca_model.tar", True),
        ("FLAME geometry", ROOT / "EMOCA-Inference" / "assets" / "FLAME" / "geometry" / "generic_model.pkl", True),
        ("FaceRecognition weights", ROOT / "EMOCA-Inference" / "assets" / "FaceRecognition" / "resnet50_ft_weight.pkl", True),
    ])

    heading("Input / Output")
    check_paths([
        ("Shared input frames dir", ROOT / "demo" / "input", False),
        ("Shared WiLoR output dir", ROOT / "demo" / "result_params_unified" / "params", False),
        ("EMOCA demo output dir", ROOT / "EMOCA-Inference" / "demo" / "output", False),
    ])

    print("\nSummary")
    print("-------")
    if failures == 0:
        print("Audit passed. The local setup looks ready for a full pipeline run.")
        return 0

    print(f"Audit found {failures} blocking issue(s).")
    print("Most common blockers are unsupported OS/GPU, missing conda environments, and missing model assets.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
