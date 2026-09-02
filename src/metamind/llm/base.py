from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class LLMConfig:
    """
    Configuration for LLM usage.

    This project can use OpenAI-compatible gateways (like OpenRouter) via `base_url`.
    
    """

    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_output_tokens: int = 800
    api_key_env: str = "METIS_API_KEY"
    base_url: Optional[str] = "https://api.metisai.ir/openai/v1"


class LLMClient(Protocol):
    """Minimal interface used by the orchestrator."""

    def generate_json(self, *, system: str, user: str) -> Dict[str, Any]:
        ...
