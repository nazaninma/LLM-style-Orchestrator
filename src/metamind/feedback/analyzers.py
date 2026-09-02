from __future__ import annotations
from typing import Any, Dict, Optional
from .schemas import FeedbackSuggestion


class HeuristicAnalyzer:
    """
    بدون LLM: با قوانین ساده پیشنهاد می‌دهد.
    """

    def suggest(
        self,
        problem_summary: Dict[str, Any],
        run_summary: Optional[Dict[str, Any]] = None,
        eval_summary: Optional[Dict[str, Any]] = None,
    ) -> FeedbackSuggestion:
        ptype = str(problem_summary.get("problem_type", "unknown")).lower()

        # اگر evaluation داریم، بهترین روش را از stats انتخاب کن
        if eval_summary and eval_summary.get("stats_top"):
            top = eval_summary["stats_top"][0]
            best_method = top.get("method")
            return FeedbackSuggestion(
                action="switch_method",
                method=best_method,
                method_params={},  # فقط روش را عوض می‌کنیم
                reason=f"Evaluation suggests best method='{best_method}' based on primary metric.",
                confidence=0.7,
            )

        # اگر single-run داریم: اگر iterations_completed کم و early_stop فعال، پارامتر را tweak کن
        if run_summary:
            metrics = run_summary.get("metrics", {}) or {}
            iters = float(metrics.get("iterations_completed", 0) or 0)

            # function_optimization: اگر زود متوقف شد => افزایش iter یا n_particles
            if ptype == "function_optimization":
                tweak = {}
                if iters < 0.8 * 300:  # فرض baseline 300
                    tweak = {"max_iterations": 500, "n_particles": 60}
                return FeedbackSuggestion(
                    action="rerun",
                    method=run_summary.get("method"),
                    method_params=tweak,
                    reason="Heuristic: increase search budget to improve convergence.",
                    confidence=0.6,
                )

        return FeedbackSuggestion(action="stop", reason="No further improvement rule triggered.", confidence=0.5)


class LLMAnalyzer:
    """
    این کلاس را طوری می‌نویسیم که اگر API/مدل نبود، fail نکند
    و caller بتواند fallback کند.
    """
    def __init__(self, client: Any):
        self.client = client

    def suggest(
        self,
        problem_summary: Dict[str, Any],
        run_summary: Optional[Dict[str, Any]] = None,
        eval_summary: Optional[Dict[str, Any]] = None,
    ) -> FeedbackSuggestion:
        prompt = {
            "task": "analyze_and_suggest",
            "problem": problem_summary,
            "run": run_summary,
            "evaluation": eval_summary,
            "output_format": {
                "action": "rerun|switch_method|stop",
                "method": "optional str",
                "method_params": "optional dict",
                "reason": "str",
                "confidence": "0..1"
            }
        }
        data = self.client.complete_json(prompt)  # باید dict بدهد

        return FeedbackSuggestion(
            action=str(data.get("action", "stop")),
            method=data.get("method"),
            method_params=data.get("method_params") or {},
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.5)),
        )
