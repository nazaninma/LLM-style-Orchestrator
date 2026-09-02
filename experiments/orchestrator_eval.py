
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from metamind.orchestrator import Orchestrator
from metamind.llm import make_llm_client
from metamind.methods import get_method
from metamind.methods.base import RunContext
from metamind.evaluation.engine import EvaluationEngine
from metamind.utils import save_json

log = logging.getLogger("metamind")

DEFAULT_CONFIGS = [
    "tsp_eval.yaml",
    "tsp_berlin52_eval.yaml",
    "rastrigin_eval.yaml",
    "ackley_eval.yaml",
    "rosenbrock_eval.yaml",
    "titanic_eval.yaml",
    "iris_eval.yaml",
    "blobs_eval.yaml",
]

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def run_orchestrator_evaluation(project_root: Path, out_dir: Path, configs: List[str] | None = None, repeats: int = 5) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = configs or DEFAULT_CONFIGS

    # Try to create real LLM client if configured; else heuristic orchestrator
    llm = make_llm_client()
    orch = Orchestrator(llm=llm)

    engine = EvaluationEngine(results_root=out_dir)

    summary_rows: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []

    for cfg_name in configs:
        cfg_path = project_root / "configs" / cfg_name
        if not cfg_path.exists():
            log.warning("Config not found: %s", cfg_path)
            continue

        settings = load_yaml(cfg_path)
        # build problem via main loader
        from metamind.problems.loader import load_problem_bundle

        bundle = load_problem_bundle(project_root=project_root, config=settings, seed=42)
        problem = bundle["problem"]
        base_ctx = bundle["ctx"]

        problem_name = f"{problem.get('problem_type')}::{problem.get('name')}::{cfg_name}"

        # Run full evaluation once to establish best fixed method under current evaluation config
        fixed_dir = out_dir / "fixed_method_baseline" / cfg_name.replace(".yaml", "")
        fixed_dir.mkdir(parents=True, exist_ok=True)
        fixed = engine.run(run_dir=fixed_dir, problem=problem, base_ctx=base_ctx, settings=settings)
        best_fixed_method = fixed.get("summary", {}).get("best_method")

        # Orchestrator evaluation: 5 independent decisions + execution
        decisions: List[Dict[str, Any]] = []
        chosen_metrics: List[Dict[str, Any]] = []
        chosen_methods: List[str] = []

        for i in range(repeats):
            # vary seed to encourage variation where randomness exists
            seed = int(base_ctx.seed) + i
            sub_dir = out_dir / "orchestrator_runs" / cfg_name.replace(".yaml", "") / f"trial_{i+1:02d}_seed_{seed}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            ctx = RunContext(
                run_dir=sub_dir,
                seed=seed,
                time_limit_sec=base_ctx.time_limit_sec,
                minimize=base_ctx.minimize,
                early_stopping=base_ctx.early_stopping,
            )

            decision, parsed = orch.decide(problem)
            dec_dict = decision.model_dump() if hasattr(decision, "model_dump") else decision.__dict__
            decisions.append(dec_dict)
            chosen_methods.append(decision.selected_method)

            # Execute chosen method with suggested parameters
            method = get_method(decision.selected_method)
            res = method.solve(problem=problem, params=decision.parameters or {}, ctx=ctx, progress_cb=lambda e: None)

            # Compute derived metrics via engine helper (reuse internal)
            from metamind.evaluation.derived_metrics import add_derived_metrics
            met = add_derived_metrics(problem, res.metrics or {}, res.artifacts or {})
            chosen_metrics.append(met)

            # save per-trial
            save_json(sub_dir, {"decision": dec_dict, "metrics": met, "parsed": parsed}, filename="orchestrator_trial.json")

        # Selection accuracy: % trials where selected == best_fixed_method
        acc = None
        if best_fixed_method:
            acc = sum(1 for m in chosen_methods if m == best_fixed_method) / len(chosen_methods)

        detail = {
            "config": cfg_name,
            "problem_name": problem_name,
            "best_fixed_method": best_fixed_method,
            "selection_accuracy": acc,
            "chosen_methods": chosen_methods,
            "decisions": decisions,
            "chosen_metrics": chosen_metrics,
        }
        all_details.append(detail)

        summary_rows.append({
            "config": cfg_name,
            "problem_type": problem.get("problem_type"),
            "problem": problem.get("name"),
            "best_fixed_method": best_fixed_method,
            "selection_accuracy": acc,
            "chosen_method_mode": max(set(chosen_methods), key=chosen_methods.count),
        })

    save_json(out_dir, {"summary": summary_rows, "details": all_details}, filename="orchestrator_evaluation.json")
    return {"summary": summary_rows, "details": all_details}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "results" / "orchestrator_evaluation"
    run_orchestrator_evaluation(project_root, out_dir, repeats=5)
