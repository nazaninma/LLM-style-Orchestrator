from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from metamind.methods import list_methods
from metamind.types import RunResult
from .schemas import OrchestratorDecision, OrchestratorRecommendation
from .parser import parse_problem
from .selector import select_method
from .configurator import configure_params
from .interpreter import interpret_result
from .recommender import recommend
from metamind.llm import LLMClient


class Orchestrator:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.available = set(list_methods().keys())
        self.llm = llm

    def decide(self, problem: Dict[str, Any]) -> Tuple[OrchestratorDecision, Dict[str, Any]]:
        parsed = parse_problem(problem)

        # If a real LLM client is provided, prefer it. Fall back to the heuristic pipeline
        # on any error (robust for offline/no-key environments).
        if self.llm is not None:
            # try:
                llm_decision = self._llm_decide(parsed)
                return llm_decision, parsed
            # except Exception:
                # fall back
                # pass

        selected, backup, reasoning, confidence = select_method(parsed, self.available)
        params = configure_params(selected, parsed)

        decision = OrchestratorDecision(
            problem_type=parsed["problem_type"],
            selected_method=selected,
            backup_method=backup,
            reasoning=reasoning,
            parameters=params,
            confidence=confidence,
        )
        return decision, parsed

    def feedback(self, parsed: Dict[str, Any], result: RunResult) -> Tuple[Dict[str, Any], OrchestratorRecommendation]:
        if self.llm is not None:
            try:
                interpretation, rec = self._llm_feedback(parsed, result)
                return interpretation, rec
            except Exception:
                pass

        interpretation = interpret_result(result)
        rec = recommend(parsed, interpretation)
        return interpretation, rec

    def _llm_decide(self, parsed: Dict[str, Any]) -> OrchestratorDecision:
        system = (
            "You are MetaMind, an orchestrator that selects and configures Computational Intelligence methods. "
            "Choose the most suitable method and reasonable parameters." 
        )
        user = {
            "task": "Select a CI method and parameters",
            "available_methods": sorted(list(self.available)),
            "problem": parsed,
            "output_schema": {
                "problem_type": "string",
                "selected_method": "string (must be one of available_methods)",
                "reasoning": "string",
                "parameters": "object",
                "backup_method": "string or null (must be in available_methods)",
                "confidence": "number 0..1",
            },
        }

        out = self.llm.generate_json(system=system, user=_json_dumps(user))
        selected = str(out.get("selected_method", "")).strip().lower()
        if selected not in self.available:
            raise ValueError(f"LLM selected unknown method: {selected}")
        backup = out.get("backup_method", None)
        if isinstance(backup, str):
            backup = backup.strip().lower() or None
            if backup and backup not in self.available:
                backup = None

        params = out.get("parameters", {})
        if not isinstance(params, dict):
            params = {}

        return OrchestratorDecision(
            problem_type=str(out.get("problem_type", parsed.get("problem_type", "unknown"))),
            selected_method=selected,
            reasoning=str(out.get("reasoning", "")),
            parameters=params,
            backup_method=backup,
            confidence=float(out.get("confidence", 0.5)),
        )

    def _llm_feedback(self, parsed: Dict[str, Any], result: RunResult) -> Tuple[Dict[str, Any], OrchestratorRecommendation]:
        system = (
            "You are MetaMind, an orchestrator that interprets results and suggests improvements. "
            "Return concise, actionable suggestions." 
        )
        user = {
            "task": "Interpret results and recommend improvements",
            "problem": parsed,
            "result": {
                "metrics": result.metrics or {},
                "notes": result.notes,
                "metadata": {
                    "time_sec": getattr(result.metadata, "time_sec", None),
                    "n_iterations": getattr(result.metadata, "n_iterations", None),
                },
            },
            "output_schema": {
                "interpretation": "object (any keys you want)",
                "recommendation": {
                    "summary": "string",
                    "suggestions": "object",
                },
            },
        }

        out = self.llm.generate_json(system=system, user=_json_dumps(user))
        interpretation = out.get("interpretation", {})
        if not isinstance(interpretation, dict):
            interpretation = {}
        rec = out.get("recommendation", {})
        if not isinstance(rec, dict):
            rec = {}
        return interpretation, OrchestratorRecommendation(
            summary=str(rec.get("summary", "")),
            suggestions=(rec.get("suggestions", {}) if isinstance(rec.get("suggestions", {}), dict) else {}),
        )


def _json_dumps(obj):
    import json
    import numpy as np
    from pathlib import Path

    def _default(o):
        # numpy arrays -> list
        if isinstance(o, np.ndarray):
            return o.tolist()

        # numpy scalars -> python scalar
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()

        # pathlib paths -> str
        if isinstance(o, Path):
            return str(o)

        # sets/tuples -> list
        if isinstance(o, (set, tuple)):
            return list(o)

        # fallback
        return str(o)

    return json.dumps(
        obj,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=_default,
    )