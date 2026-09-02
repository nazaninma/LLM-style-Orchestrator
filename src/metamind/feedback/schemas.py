from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FeedbackSuggestion:
    action: str                 # "rerun" | "switch_method" | "stop"
    reason: str
    method: Optional[str] = None
    method_params: Optional[Dict[str, Any]] = None
    confidence: float = 0.5
