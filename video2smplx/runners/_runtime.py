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


@contextmanager
def project_cwd(project_dir: Path):
    previous = Path.cwd()
    os.chdir(str(project_dir))
    try:
        yield
    finally:
        os.chdir(str(previous))

