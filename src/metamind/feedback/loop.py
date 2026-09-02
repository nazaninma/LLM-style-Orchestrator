from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

from metamind.methods import get_method
from metamind.methods.base import RunContext
from metamind.utils import save_json, save_results_csv, save_results_json

from .summarizer import summarize_problem, summarize_result, summarize_evaluation
from .schemas import FeedbackSuggestion
from .analyzers import HeuristicAnalyzer

log = logging.getLogger("metamind")


def _merge_params(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base or {})
    if override:
        out.update(override)
    return out

def _noop_progress_cb(ev):
  return

class FeedbackLoopEngine:
    def __init__(self, analyzer: Optional[Any] = None):
        self.analyzer = analyzer or HeuristicAnalyzer()


    def run(
        self,
        *,
        run_dir: Path,
        problem: Dict[str, Any],
        base_ctx: RunContext,
        selected_method: str,
        base_params: Dict[str, Any],
        last_result: Optional[Any] = None,
        eval_summary: Optional[Dict[str, Any]] = None,
        max_rounds: int = 3,
        min_confidence: float = 0.55,
    ) -> Dict[str, Any]:
        fb_dir = run_dir / "feedback_loop"
        fb_dir.mkdir(parents=True, exist_ok=True)

        problem_sum = summarize_problem(problem)
        eval_sum = summarize_evaluation(eval_summary) if eval_summary else None

        history: List[Dict[str, Any]] = []
        cur_method = selected_method
        cur_params = dict(base_params)

        cur_result = last_result

        for r in range(1, max_rounds + 1):
            run_sum = summarize_result(cur_result) if cur_result else None

            suggestion: FeedbackSuggestion = self.analyzer.suggest(
                problem_summary=problem_sum,
                run_summary=run_sum,
                eval_summary=eval_sum,
            )

            # ذخیره پیشنهاد
            save_json(fb_dir / f"suggestion_round_{r}.json", {
                "round": r,
                "suggestion": suggestion.__dict__,
                "current_method": cur_method,
                "current_params": cur_params,
            })

            log.info("Feedback round %s: action=%s conf=%.2f", r, suggestion.action, suggestion.confidence)

            if suggestion.confidence < min_confidence or suggestion.action == "stop":
                history.append({"round": r, "action": "stop", "reason": suggestion.reason})
                break

            if suggestion.action == "switch_method" and suggestion.method:
                cur_method = suggestion.method.strip().lower()
                if suggestion.method_params:
                    cur_params = _merge_params(cur_params, suggestion.method_params)
                history.append({"round": r, "action": "switch_method", "method": cur_method, "reason": suggestion.reason})
                # بعد از تغییر روش، یک اجرا انجام بده
            elif suggestion.action == "rerun":
                if suggestion.method_params:
                    cur_params = _merge_params(cur_params, suggestion.method_params)
                history.append({"round": r, "action": "rerun", "method": cur_method, "reason": suggestion.reason})
            else:
                history.append({"round": r, "action": "stop", "reason": "Unknown action"})
                break

            # اجرای مجدد
            sub_run = fb_dir / f"round_{r:02d}_{cur_method}"
            sub_run.mkdir(parents=True, exist_ok=True)

            ctx = RunContext(
                run_dir=sub_run,
                seed=int(base_ctx.seed) + r,  # تغییر seed برای تنوع
                time_limit_sec=base_ctx.time_limit_sec,
                minimize=base_ctx.minimize,
                early_stopping=base_ctx.early_stopping,
            )

            method = get_method(cur_method)
            cur_result = method.solve(problem=problem, params=cur_params, ctx=ctx, progress_cb=_noop_progress_cb,)

            save_results_json(sub_run, [cur_result])
            save_results_csv(sub_run, [cur_result])

            # ذخیره summary هر round
            save_json(sub_run / "round_summary.json", {
                "round": r,
                "method": cur_method,
                "params": cur_params,
                "result": summarize_result(cur_result),
            })

        out = {
            "final_method": cur_method,
            "final_params": cur_params,
            "history": history,
        }
        save_json(fb_dir / "feedback_final.json", out)
        return out
