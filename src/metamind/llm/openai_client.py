from __future__ import annotations

import json
import os
from typing import Any, Dict

from .base import LLMClient, LLMConfig


class OpenAIResponsesClient:
    """OpenAI Responses API client wrapper.

    Uses the official `openai` Python SDK.

    We intentionally keep this wrapper small and dependency-light so the rest
    of the codebase doesn't depend on OpenAI SDK details.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        from openai import OpenAI  # type: ignore
        api_key = "tpsg-uzuqH4I2nQKDNx8cwWGWTSVu5ZqpIgM"

        if not api_key:
            raise RuntimeError("API key not found (set OPENAI_API_KEY or OPENROUTER_API_KEY)")

        self.client = OpenAI(
            api_key=api_key,
            base_url=cfg.base_url   # باید https://openrouter.ai/api/v1 باشد
        )

    def generate_json(self, *, system: str, user: str) -> Dict[str, Any]:
        """Return a dict parsed from model output.

        The model is instructed to output *only* strict JSON.
        """

        instructions = (
            system.strip()
            + "\n\n"
            + "You MUST output only valid JSON. Do not wrap in markdown. Do not add commentary."
        )

        resp = self.client.responses.create(
            model=self.cfg.model,
            instructions=instructions,
            input=user.strip(),
            temperature=float(self.cfg.temperature),
            max_output_tokens=int(self.cfg.max_output_tokens),
        )

        text = getattr(resp, "output_text", "")
        if not text:
            # fallback: try to reconstruct from output items if SDK changes
            try:
                text = "".join([getattr(o, "text", "") for o in resp.output])  # type: ignore
            except Exception:
                text = ""

        text = text.strip()
        if not text:
            raise ValueError("LLM returned empty output")

        # Some models may occasionally include leading/trailing non-json text.
        # We do a best-effort extraction of the first JSON object.
        obj_text = _extract_json_object(text)
        return json.loads(obj_text)


def _extract_json_object(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s

    # Find first '{' and last '}' and slice.
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError(f"Could not find JSON object in LLM output: {s[:200]}")
    return s[i : j + 1]
