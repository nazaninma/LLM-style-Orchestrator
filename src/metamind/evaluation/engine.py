from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from metamind.methods import get_method, list_methods
from metamind.methods.base import RunContext
from metamind.methods.progress import ProgressEvent

from metamind.utils import save_json, save_results_csv, save_results_json

from .aggregator import aggregate_runs
from .derived_metrics import add_derived_metrics
from .plotting import plot_convergence
from .plotting import plot_confusion_matrix


log = logging.getLogger("metamind")


def _method_compat(method_name: str, problem_type: str) -> bool:
    m = (method_name or "").lower().strip()
    p = (problem_type or "").lower().strip()

    if p == "function_optimization":
        return m in ("pso", "ga", "gp", "fuzzy", "random_search")
    if p == "tsp":
        return m in ("aco", "ga_tsp", "hopfield", "random_search")
    if p == "classification":
        return m in ("perceptron", "mlp")
    if p == "clustering":
        return m in ("kmeans", "som")
    return False


def _make_seeds(base_seed: int, repeats: int) -> List[int]:
    return [base_seed + i for i in range(repeats)]


def _flatten_stats(stats) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in stats:
        row = {
            "method": s.method,
            "n_runs": s.n_runs,
            "n_success": s.n_success,
            "success_rate": s.success_rate,
            "primary_metric": s.primary_metric,
            "primary_mean": s.primary_mean,
            "primary_std": s.primary_std,
            "primary_ci_low": getattr(s, "primary_ci_low", None),
            "primary_ci_high": getattr(s, "primary_ci_high", None),
            "maximize": s.maximize,
        }
        for k in ("time_sec", "n_iterations", "best_fitness", "test_accuracy", "test_f1", "inertia", "silhouette"):
            if k in s.metrics_mean:
                row[f"{k}_mean"] = s.metrics_mean[k]
                row[f"{k}_std"] = s.metrics_std.get(k, 0.0)
        rows.append(row)
    return rows


class EvaluationEngine:
    def __init__(self, results_root: Path):
        self.results_root = results_root

    def run(self, *, run_dir: Path, problem: Dict[str, Any], base_ctx: RunContext, settings: Dict[str, Any]) -> Dict[str, Any]:
        ev_cfg = (settings.get("evaluation", {}) or {})
        repeats = int(ev_cfg.get("repeats", 5))
        make_plots = bool(ev_cfg.get("make_plots", True))
        save_per_run = bool(ev_cfg.get("save_per_run", True))
        cv_folds = int(ev_cfg.get("cv_folds", 0))

        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        problem_type = (problem.get("problem_type") or problem.get("type") or "unknown").strip().lower()
        problem_name = str(problem.get("name", "unknown_problem"))

        # which methods?
        all_methods = sorted(list_methods().keys())
        wanted = ev_cfg.get("methods", None)
        if wanted:
            methods = [str(m).strip().lower() for m in wanted if str(m).strip()]
        else:
            methods = all_methods

        # filter by compatibility + existence
        methods = [m for m in methods if (m in all_methods and _method_compat(m, problem_type))]
        if not methods:
            raise ValueError(
                f"No compatible methods for problem_type={problem_type}. "
                f"Check evaluation.methods in config."
            )

        # seeds
        seeds = ev_cfg.get("seeds", None)
        if seeds and isinstance(seeds, list) and seeds:
            seeds = [int(s) for s in seeds]
            repeats = len(seeds)
        else:
            seeds = _make_seeds(int(base_ctx.seed), repeats)

        # method params override from config
        cfg_method_params = settings.get("method_params", {}) or {}

        per_method_runs: Dict[str, List[Dict[str, Any]]] = {}
        per_run_results_flat: List[Dict[str, Any]] = []

        
        def _cv_accuracy_if_needed(method_name: str, params: Dict[str, Any], seed: int) -> float | None:
            if problem_type != "classification" or cv_folds <= 1:
                return None
            ml = problem.get("ml", {}) or {}
            X_all = ml.get("X")
            y_all = ml.get("y")
            if X_all is None or y_all is None:
                return None
            try:
                import numpy as np
                from sklearn.model_selection import StratifiedKFold
            except Exception:
                return None
            X_all = np.asarray(X_all, dtype=float)
            y_all = np.asarray(y_all, dtype=int).reshape(-1)
            if X_all.shape[0] != y_all.shape[0]:
                return None
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            accs = []
            for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all), start=1):
                fold_problem = dict(problem)
                fold_ml = dict(ml)
                fold_ml["X_train"] = X_all[tr_idx]
                fold_ml["y_train"] = y_all[tr_idx]
                fold_ml["X_val"] = X_all[te_idx]
                fold_ml["y_val"] = y_all[te_idx]
                fold_ml["X_test"] = X_all[te_idx]
                fold_ml["y_test"] = y_all[te_idx]
                fold_problem["ml"] = fold_ml
                fold_ctx = RunContext(
                    run_dir=run_dir / "evaluation" / "_cv_tmp" / method_name / f"seed_{seed}" / f"fold_{fold_i}",
                    seed=int(seed + fold_i),
                    time_limit_sec=base_ctx.time_limit_sec,
                    minimize=base_ctx.minimize,
                    early_stopping=base_ctx.early_stopping,
                )
                fold_ctx.run_dir.mkdir(parents=True, exist_ok=True)
                m = get_method(method_name)
                res = m.solve(problem=fold_problem, params=params, ctx=fold_ctx, progress_cb=lambda e: None)
                v = (res.metrics or {}).get("test_accuracy")
                if isinstance(v, (int, float)):
                    accs.append(float(v))
            if not accs:
                return None
            return float(sum(accs) / len(accs))
        def progress_cb(ev: ProgressEvent) -> None:
            if ev.iteration == 0:
                log.info("Eval progress: iter=%s best=%s metrics=%s", ev.iteration, ev.best_fitness, ev.metrics)

        for method_name in methods:
            per_method_runs[method_name] = []

            for i, seed in enumerate(seeds):
                sub_run = eval_dir / method_name / f"rep_{i+1:02d}_seed_{seed}"
                sub_run.mkdir(parents=True, exist_ok=True)

                ctx = RunContext(
                    run_dir=sub_run,
                    seed=int(seed),
                    time_limit_sec=base_ctx.time_limit_sec,
                    minimize=base_ctx.minimize,
                    early_stopping=base_ctx.early_stopping,
                )

                method = get_method(method_name)
                params = dict(cfg_method_params)

                log.info("Evaluation run: method=%s rep=%s/%s seed=%s", method_name, i + 1, len(seeds), seed)
                try:
                    # reset per-run counters (e.g., function evaluations)
                    if problem_type == "function_optimization":
                        obj = problem.get("objective", {}) or {}
                        counter = obj.get("counter")
                        if isinstance(counter, dict):
                            counter["count"] = 0
                    result = method.solve(problem=problem, params=params, ctx=ctx, progress_cb=progress_cb)
                    error: str | None = None
                except Exception as e:
                    log.exception("Evaluation run failed: method=%s seed=%s", method_name, seed)
                    error = f"{type(e).__name__}: {e}"
                    result = None
                    
                if result is not None and problem_type == "classification":
                    arts = result.artifacts or {}
                    y_true = arts.get("y_true")
                    y_pred = arts.get("y_pred")

                    # بعضی متدها ممکنه y_pred ندهند و فقط y_score بدهند
                    # در این حالت می‌توان threshold=0.5 زد (اگر خواستی)
                    if y_true is not None and y_pred is not None:
                        try:
                            plot_confusion_matrix(
                                out_dir=eval_dir,  # یا sub_run اگر می‌خوای per-run ذخیره شه
                                method=method_name,
                                y_true=y_true,
                                y_pred=y_pred,
                                title_prefix=problem_name
                            )
                        except Exception as e:
                            log.warning("Could not plot confusion matrix for %s: %s", method_name, e)


                if save_per_run and result is not None:
                    save_results_json(sub_run, [result])
                    save_results_csv(sub_run, [result])

                if result is None:
                    item = {
                        "method": method_name,
                        "seed": int(seed),
                        "error": error,
                        "metrics": {},
                        "artifacts": {},
                        "notes": None,
                        "metadata": {},
                    }
                else:
                    metrics0 = dict(result.metrics or {})
                    cv_mean = _cv_accuracy_if_needed(method_name, params, int(seed))
                    if cv_mean is not None:
                        metrics0["cv_accuracy_mean"] = float(cv_mean)
                    item = {
                        "method": method_name,
                        "seed": int(seed),
                        "error": None,
                        "metrics": add_derived_metrics(problem, metrics0, result.artifacts or {}),
                        "artifacts": result.artifacts or {},
                        "notes": result.notes,
                        "metadata": asdict(result.metadata),
                    }
                per_method_runs[method_name].append(item)

                if result is not None:
                    row = {"method": method_name, "seed": int(seed)}
                    for k, v in (add_derived_metrics(problem, metrics0, result.artifacts or {})).items():
                        if isinstance(v, (int, float)):
                            row[k] = float(v)
                    per_run_results_flat.append(row)

        # write flat runs csv
        if per_run_results_flat:
            csv_path = eval_dir / "eval_runs_flat.csv"
            keys = sorted({k for row in per_run_results_flat for k in row.keys()})
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(per_run_results_flat)

        # aggregate
        stats = aggregate_runs(problem_type, per_method_runs)

        # Wilcoxon signed-rank test vs best method (paired by seed)
        wilcoxon_vs_best: Dict[str, Any] = {}
        if stats:
            best_method = stats[0].method
            primary_metric = stats[0].primary_metric
            maximize = stats[0].maximize

            def _series(method_name: str) -> Dict[int, float]:
                out: Dict[int, float] = {}
                for it in per_method_runs.get(method_name, []):
                    if it.get("error"):
                        continue
                    seed_v = int(it.get("seed", 0))
                    v = (it.get("metrics", {}) or {}).get(primary_metric)
                    if isinstance(v, (int, float)):
                        out[seed_v] = float(v)
                return out

            base_s = _series(best_method)
            for m in methods:
                if m == best_method:
                    continue
                comp_s = _series(m)
                common = sorted(set(base_s.keys()) & set(comp_s.keys()))
                if len(common) < 2:
                    wilcoxon_vs_best[m] = {"p_value": None, "n_pairs": len(common), "best_method": best_method, "metric": primary_metric, "maximize": maximize}
                    continue
                x = [base_s[s] for s in common]
                y = [comp_s[s] for s in common]
                try:
                    from scipy.stats import wilcoxon

                    # test y vs x; if maximize, we invert to make "lower is better" for the test direction
                    if maximize:
                        x_t = [-v for v in x]
                        y_t = [-v for v in y]
                    else:
                        x_t, y_t = x, y
                    stat = wilcoxon(y_t, x_t, zero_method="wilcox", alternative="two-sided")
                    p = float(getattr(stat, "pvalue", None) or stat[1])
                except Exception:
                    p = None
                wilcoxon_vs_best[m] = {"p_value": p, "n_pairs": len(common), "best_method": best_method, "metric": primary_metric, "maximize": maximize}

        summary = {
            "problem_name": problem_name,
            "problem_type": problem_type,
            "repeats": repeats,
            "methods": methods,
            "stats": [asdict(s) for s in stats],
            "wilcoxon_vs_best": wilcoxon_vs_best,
        }

        table_rows = _flatten_stats(stats)
        if table_rows:
            table_csv = eval_dir / "eval_table.csv"
            keys = sorted({k for row in table_rows for k in row.keys()})
            with table_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(table_rows)

        save_json(eval_dir / "eval_summary.json", summary)
        save_json(eval_dir / "eval_runs.json", {"rows": per_method_runs})
        save_json(eval_dir / "eval_table.json", {"rows": table_rows})

        plot_paths: List[str] = []
        if make_plots:
            paths = plot_convergence(eval_dir, problem_name, problem_type, per_method_runs)
            plot_paths = [str(p) for p in paths]
            save_json(eval_dir / "eval_plots.json", {"plots": plot_paths})

        log.info("Evaluation finished. Outputs in: %s", str(eval_dir))
        return summary
