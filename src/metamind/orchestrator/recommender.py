from __future__ import annotations

from typing import Any, Dict

from .schemas import OrchestratorRecommendation


def recommend(parsed: Dict[str, Any], interpretation: Dict[str, Any]) -> OrchestratorRecommendation:
    """
    Basic rule-based recommendation engine (Phase 3).
    Later, this is where LLM feedback will plug in.
    """
    method = interpretation.get("method_used", "")
    best = interpretation.get("best_fitness", None)
    err = interpretation.get("error_to_optimum", None)

    suggestions: Dict[str, Any] = {}

    if method == "random_search":
        suggestions["try_increasing_n_samples"] = True
        suggestions["try_disabling_early_stopping"] = True

        if err is not None and err > 1.0:
            suggestions["n_samples_recommended"] = 20000
            suggestions["reason"] = "Random search needs more samples to approach optimum."
        else:
            suggestions["n_samples_recommended"] = 8000

    summary = "Baseline recommendations generated."
    return OrchestratorRecommendation(summary=summary, suggestions=suggestions)