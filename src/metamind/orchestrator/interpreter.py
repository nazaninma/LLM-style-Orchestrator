from __future__ import annotations

from typing import Any, Dict

from metamind.types import RunResult


def interpret_result(result: RunResult) -> Dict[str, Any]:
    """
    Produces a compact summary for the orchestrator feedback loop.
    """
    metrics = result.metrics or {}
    summary = {
        "method_used": result.metadata.method_name,
        "problem_name": result.metadata.problem_name,
        "best_fitness": metrics.get("best_fitness", None),
        "time_sec": metrics.get("time_sec", None),
        "iterations_completed": metrics.get("iterations_completed", None),
        "error_to_optimum": metrics.get("error_to_optimum", None),
    }
    return summary