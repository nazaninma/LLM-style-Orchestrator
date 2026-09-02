from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from metamind.types import RunResult
from .exceptions import ParameterValidationError
from .progress import EarlyStoppingConfig, ProgressCallback, default_progress_callback


log = logging.getLogger(__name__)


@dataclass
class RunContext:
    """
    Shared execution context for all methods.
    """
    run_dir: Path
    seed: int
    time_limit_sec: Optional[float] = None
    minimize: bool = True  # most optimization is minimization
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    def checkpoint_path(self, name: str = "checkpoint.json") -> Path:
        return self.run_dir / name


class Problem(Protocol):
    """
    A minimal protocol for problems. In later phases you'll expand this.
    For now, methods should treat 'data' as a dict and use keys they need.
    """
    name: str
    data: Dict[str, Any]


class BaseMethod(Protocol):
    """
    All CI methods must implement this standardized API.
    """
    name: str

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def solve(
        self,
        problem: Dict[str, Any],
        params: Dict[str, Any],
        ctx: RunContext,
        progress_cb: ProgressCallback = default_progress_callback,
    ) -> RunResult:
        ...


def require_float(params: Dict[str, Any], key: str, default: Optional[float] = None,
                  min_v: Optional[float] = None, max_v: Optional[float] = None) -> float:
    if key not in params:
        if default is None:
            raise ParameterValidationError(f"Missing required parameter: {key}")
        v = float(default)
    else:
        v = float(params[key])

    if min_v is not None and v < min_v:
        raise ParameterValidationError(f"Parameter '{key}' must be >= {min_v}. Got {v}")
    if max_v is not None and v > max_v:
        raise ParameterValidationError(f"Parameter '{key}' must be <= {max_v}. Got {v}")
    return v


def require_int(params: Dict[str, Any], key: str, default: Optional[int] = None,
                min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    if key not in params:
        if default is None:
            raise ParameterValidationError(f"Missing required parameter: {key}")
        v = int(default)
    else:
        v = int(params[key])

    if min_v is not None and v < min_v:
        raise ParameterValidationError(f"Parameter '{key}' must be >= {min_v}. Got {v}")
    if max_v is not None and v > max_v:
        raise ParameterValidationError(f"Parameter '{key}' must be <= {max_v}. Got {v}")
    return v


def require_bool(params: Dict[str, Any], key: str, default: Optional[bool] = None) -> bool:
    if key not in params:
        if default is None:
            raise ParameterValidationError(f"Missing required parameter: {key}")
        return bool(default)
    v = params[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Checkpoint saved: %s", path)


def load_checkpoint(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Failed to load checkpoint: %s", path)
        return None