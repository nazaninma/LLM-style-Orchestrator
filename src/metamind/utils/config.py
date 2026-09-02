from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


from pathlib import Path

def resolve_config_path(root: Path, user_path: str) -> Path:
    p = Path(user_path)

    # 1) absolute path
    if p.is_absolute() and p.exists():
        return p

    # 2) as provided (relative to CWD)
    if p.exists():
        return p.resolve()

    # 3) relative to project root
    candidate = (root / p).resolve()
    if candidate.exists():
        return candidate

    # 4) if user passed just a filename, look in configs/
    candidate2 = (root / "configs" / p.name).resolve()
    if candidate2.exists():
        return candidate2

    # fallback (so error message shows best guess)
    return candidate


@dataclass
class AppConfig:
    # General
    seed: int = 42
    log_level: str = "INFO"
    results_dir: str = "results"

    # Experiment defaults
    default_problem: str = "smoke_test"
    default_method: str = "noop"

    # Free-form nested settings (for future phases: method params, problem params, etc.)
    settings: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: Path) -> "AppConfig":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        if path.suffix.lower() in [".yaml", ".yml"]:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError("Config must be .yaml/.yml or .json")

        return AppConfig(
            seed=int(data.get("seed", 42)),
            log_level=str(data.get("log_level", "INFO")),
            results_dir=str(data.get("results_dir", "results")),
            default_problem=str(data.get("default_problem", "smoke_test")),
            default_method=str(data.get("default_method", "noop")),
            settings=dict(data.get("settings", {})),
        )

    def apply_overrides(self, overrides: List[Tuple[str, str]]) -> None:
        """
        overrides: list of ("path.to.key", "value") where value is parsed as json when possible.
        Example: ("seed","123"), ("settings.tsp.cities","30")
        """
        for key_path, raw_value in overrides:
            value = _smart_parse(raw_value)
            _set_by_dotted_path(self, key_path, value)


def _smart_parse(raw: str) -> Any:
    raw = raw.strip()
    # try JSON parsing: numbers, true/false, arrays, objects
    try:
        return json.loads(raw)
    except Exception:
        return raw  # fallback string


def _set_by_dotted_path(cfg: AppConfig, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if len(parts) == 1:
        attr = parts[0]
        if not hasattr(cfg, attr):
            raise KeyError(f"Unknown config field: {attr}")
        setattr(cfg, attr, value)
        return

    # If begins with "settings", set inside dict tree
    if parts[0] != "settings":
        raise KeyError("Only nested overrides under settings.* are allowed (except top-level fields).")

    d = cfg.settings
    for p in parts[1:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value