from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: Path) -> Path:
    """Find the project root by walking up the filesystem.

    We detect the root via simple directory markers (configs/, data/, results/, src/).
    This keeps path resolution robust when the code is executed from different CWDs.
    """

    p = start
    if p.is_file():
        p = p.parent

    for cand in [p] + list(p.parents):
        if (cand / "configs").is_dir() and (cand / "data").is_dir():
            return cand
        # common with src-layout: .../src/metamind -> go up to repo root
        if cand.name == "src" and (cand.parent / "configs").is_dir():
            return cand.parent

    return p


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    results_dir: Path
    logs_dir: Path
    configs_dir: Path

    @staticmethod
    def from_root(root: Path) -> "ProjectPaths":
        return ProjectPaths(
            root=root,
            data_dir=root / "data",
            results_dir=root / "results",
            logs_dir=root / "results" / "logs",
            configs_dir=root / "configs",
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)