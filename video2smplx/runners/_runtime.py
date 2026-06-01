"""Runtime helpers for importing legacy project folders in one process."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys


def add_project_to_path(project_dir: Path) -> None:
    project_path = str(Path(project_dir).resolve())
    if project_path in sys.path:
        sys.path.remove(project_path)
    sys.path.insert(0, project_path)


def require_files(paths: list[Path], label: str) -> None:
    missing = [Path(path) for path in paths if not Path(path).exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"{label} required files are missing:\n{missing_text}")


@contextmanager
def project_cwd(project_dir: Path):
    previous = Path.cwd()
    os.chdir(str(project_dir))
    try:
        yield
    finally:
        os.chdir(str(previous))
