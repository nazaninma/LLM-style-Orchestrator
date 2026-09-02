from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class RunMetadata:
    run_id: str
    timestamp_utc: str
    seed: int
    problem_name: str
    method_name: str


@dataclass
class RunResult:
    """Generic result container for any problem/method run."""
    metadata: RunMetadata
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # e.g., solution vectors, confusion matrix, etc.
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d