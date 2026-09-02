from __future__ import annotations
from typing import Any, Dict, Optional


def summarize_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": problem.get("name", "unknown"),
        "problem_type": (problem.get("problem_type") or problem.get("type") or "unknown"),
        "meta": problem.get("meta", {}) or {},
        "ml_keys": sorted(list((problem.get("ml", {}) or {}).keys())),
        "objective_keys": sorted(list((problem.get("objective", {}) or {}).keys())),
    }


def summarize_result(result_obj: Any) -> Dict[str, Any]:
    # result_obj معمولاً RunResult است
    metrics = getattr(result_obj, "metrics", None) or {}
    artifacts = getattr(result_obj, "artifacts", None) or {}

    # فقط چیزهای کوچک را نگه دار (convergence_history بزرگ است)
    artifacts_small: Dict[str, Any] = {}
    for k, v in artifacts.items():
        if k == "convergence_history":
            artifacts_small[k] = {
                "len": len(v) if isinstance(v, list) else None,
                "first": v[0] if isinstance(v, list) and v else None,
                "last": v[-1] if isinstance(v, list) and v else None,
            }
        elif isinstance(v, (int, float, str, bool)) or v is None:
            artifacts_small[k] = v

    return {
        "metrics": metrics,
        "artifacts": artifacts_small,
        "notes": getattr(result_obj, "notes", ""),
        "method": getattr(getattr(result_obj, "metadata", None), "method_name", None),
        "seed": getattr(getattr(result_obj, "metadata", None), "seed", None),
    }


def summarize_evaluation(summary_json: Dict[str, Any]) -> Dict[str, Any]:
    # این همان چیزی است که EvaluationEngine برمی‌گرداند (eval_summary.json)
    return {
        "problem_name": summary_json.get("problem_name"),
        "problem_type": summary_json.get("problem_type"),
        "repeats": summary_json.get("repeats"),
        "methods": summary_json.get("methods"),
        "stats_top": (summary_json.get("stats") or [])[:3],  # فقط top3
    }
