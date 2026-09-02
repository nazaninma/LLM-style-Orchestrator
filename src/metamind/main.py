from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
import uuid

from metamind.utils import (
    AppConfig,
    ProjectPaths,
    setup_logging,
    make_run_dir,
    save_results_csv,
    save_results_json,
    save_json,
)
from metamind.utils.paths import find_project_root
from metamind.methods import get_method, list_methods
from metamind.methods.base import RunContext
from metamind.methods.progress import ProgressEvent, EarlyStoppingConfig
from metamind.orchestrator import Orchestrator
from metamind.llm import LLMConfig, OpenAIResponsesClient
from metamind.utils.config import resolve_config_path
from metamind.problems.loader import load_problem


def _parse_overrides(items: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in items:
        if "=" not in s:
            raise ValueError(f"Invalid --set '{s}'. Use key=value.")
        k, v = s.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="MetaMind - Main Runner")
    parser.add_argument("--config", type=str, default="configs/example.yaml", help="Path to config YAML/JSON")
    parser.add_argument("--set", action="append", default=[], help="Override config: key=value or settings.x.y=value")
    parser.add_argument("--run-id", type=str, default="", help="Optional run id. If empty, auto-generated.")
    args = parser.parse_args()

    # Robust project root detection (works from repo root, src/, or installed package)
    root = find_project_root(Path(__file__).resolve())
    paths = ProjectPaths.from_root(root)
    paths.ensure_dirs()

    # robust config path resolve (works from src or root)
    cfg_path = resolve_config_path(root, args.config)
    cfg = AppConfig.load(cfg_path)

    overrides = _parse_overrides(args.set)
    cfg.apply_overrides(overrides)

    run_id = args.run_id.strip() or (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    )

    results_root = root / cfg.results_dir
    logs_dir = results_root / "logs"
    results_root.mkdir(parents=True, exist_ok=True)

    setup_logging(logs_dir=logs_dir, run_id=run_id, level=cfg.log_level, console=True)
    log = logging.getLogger("metamind")

    log.info("Run started. run_id=%s", run_id)
    log.info("Config file: %s", str(cfg_path))
    log.info(
        "Config: seed=%s, default_problem=%s, default_method=%s",
        cfg.seed,
        cfg.default_problem,
        cfg.default_method,
    )
    log.info("Settings: %s", cfg.settings)
    log.info("Available methods: %s", sorted(list_methods().keys()))

    run_dir = make_run_dir(results_root, run_id)

    # -----------------------
    # Build problem (Phase 5)
    # -----------------------
    problem = load_problem(cfg, root=root)

    # -----------------------
    # Orchestrator decision (optionally backed by a real LLM)
    # -----------------------
    llm_cfg_raw = (cfg.settings.get("llm", {}) or {})
    llm_cfg = LLMConfig(
        enabled=bool(llm_cfg_raw.get("enabled", False)),
        provider=str(llm_cfg_raw.get("provider", "openai")),
        model=str(llm_cfg_raw.get("model", "gpt-4o-mini")),
        temperature=float(llm_cfg_raw.get("temperature", 0.2)),
        max_output_tokens=int(llm_cfg_raw.get("max_output_tokens", 800)),
        api_key_env=str(llm_cfg_raw.get("api_key_env", "OPENAI_API_KEY")),
        base_url=(str(llm_cfg_raw.get("base_url")).strip() or None) if llm_cfg_raw.get("base_url") is not None else None,
    )

    llm_client = None
    if llm_cfg.enabled:
        try:
            if llm_cfg.provider.lower().strip() != "openai":
                raise ValueError(f"Unsupported LLM provider: {llm_cfg.provider}")
            llm_client = OpenAIResponsesClient(llm_cfg)
            log.info("LLM enabled for orchestrator: provider=%s model=%s api_key_env=%s", llm_cfg.provider, llm_cfg.model, llm_cfg.api_key_env)
        except Exception as e:
            log.warning("LLM requested but could not be initialized (%s). Falling back to heuristic orchestrator.", e)
            llm_client = None

    orch = Orchestrator(llm=llm_client)
    decision, parsed = orch.decide(problem)
    save_json(run_dir / "decision.json", decision.to_dict())
    log.info("Decision: %s", decision.to_dict())

    # -----------------------
    # Early stopping / context
    # -----------------------
    es = cfg.settings.get("early_stopping", {}) or {}
    early_cfg = EarlyStoppingConfig(
        enabled=bool(es.get("enabled", False)),
        patience=int(es.get("patience", 30)),
        min_delta=float(es.get("min_delta", 1e-9)),
    )

    # ✅ ctx همیشه باید قبل از evaluation ساخته شود
    ctx = RunContext(
        run_dir=run_dir,
        seed=int(cfg.seed),
        time_limit_sec=float(cfg.settings.get("time_limit_sec", 0)) or None,
        minimize=True,
        early_stopping=early_cfg,
    )

    # -----------------------
    # Phase 6: Evaluation mode
    # -----------------------
    ev = (cfg.settings.get("evaluation", {}) or {})
    if bool(ev.get("enabled", False)):
        from metamind.evaluation.engine import EvaluationEngine

        engine = EvaluationEngine(results_root=results_root)
        eval_summary = engine.run(run_dir=run_dir, problem=problem, base_ctx=ctx, settings=cfg.settings)

        # -----------------------
        # Phase 7: Feedback loop (optional, after evaluation)
        # -----------------------
        fb = (cfg.settings.get("feedback_loop", {}) or {})
        if bool(fb.get("enabled", False)):
            from metamind.feedback.loop import FeedbackLoopEngine
            from metamind.feedback.analyzers import HeuristicAnalyzer

            loop = FeedbackLoopEngine(analyzer=HeuristicAnalyzer())
            loop.run(
                run_dir=run_dir,
                problem=problem,
                base_ctx=ctx,
                selected_method=str(cfg.default_method).strip().lower(),
                base_params=cfg.settings.get("method_params", {}) or {},
                last_result=None,
                eval_summary=eval_summary,
                max_rounds=int(fb.get("max_rounds", 3)),
                min_confidence=float(fb.get("min_confidence", 0.55)),
            )

        log.info("Evaluation run finished successfully.")
        return 0

    # -----------------------
    # Normal single-run mode
    # -----------------------
    if bool(cfg.settings.get("force_method", False)):
        forced = str(cfg.default_method).strip().lower()
        if forced:
            decision.selected_method = forced
            decision.reasoning += f" | Forced by config default_method={forced}"
            log.info("Forced method enabled. Using selected_method=%s", forced)
            save_json(run_dir / "decision.json", decision.to_dict())

    # Merge method params:
    # 1) orchestrator suggested
    # 2) config override settings.method_params (wins)
    method_params = dict(decision.parameters or {})
    method_params.update(cfg.settings.get("method_params", {}) or {})
    decision.parameters = method_params
    save_json(run_dir / "decision.json", decision.to_dict())

    def progress_cb(ev: ProgressEvent) -> None:
        if ev.iteration == 0 or (ev.iteration % 200 == 0):
            log.info("Progress: iter=%s best=%s metrics=%s", ev.iteration, ev.best_fitness, ev.metrics)

    method = get_method(decision.selected_method)
    result = method.solve(problem=problem, params=method_params, ctx=ctx, progress_cb=progress_cb)

    json_path = save_results_json(run_dir, [result])
    csv_path = save_results_csv(run_dir, [result])

    interpretation, rec = orch.feedback(parsed, result)
    save_json(run_dir / "interpretation.json", interpretation)
    save_json(run_dir / "recommendation.json", rec.to_dict())

    # -----------------------
    # Phase 7: Feedback loop (optional, after single-run)
    # -----------------------
    fb = (cfg.settings.get("feedback_loop", {}) or {})
    if bool(fb.get("enabled", False)):
        from metamind.feedback.loop import FeedbackLoopEngine
        from metamind.feedback.analyzers import HeuristicAnalyzer

        loop = FeedbackLoopEngine(analyzer=HeuristicAnalyzer())
        loop.run(
            run_dir=run_dir,
            problem=problem,
            base_ctx=ctx,
            selected_method=str(decision.selected_method).strip().lower(),
            base_params=method_params,
            last_result=result,
            eval_summary=None,
            max_rounds=int(fb.get("max_rounds", 3)),
            min_confidence=float(fb.get("min_confidence", 0.55)),
        )

    log.info("Saved results: %s", json_path)
    log.info("Saved results: %s", csv_path)
    log.info("Saved decision/feedback files in: %s", run_dir)
    log.info("Run finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
