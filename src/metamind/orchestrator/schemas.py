from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


@dataclass
class OrchestratorDecision:
    problem_type: str  # optimization / classification / clustering / tsp / function_optimization
    selected_method: str
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    backup_method: Optional[str] = None
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrchestratorRecommendation:
    summary: str
    suggestions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)