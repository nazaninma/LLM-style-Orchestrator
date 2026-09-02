
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numbers
import math

def _is_num(x: Any) -> bool:
    """Robust numeric check.

    Accepts Python numerics and numpy scalar numerics, but rejects bool and
    NaN/Inf values.
    """
    if isinstance(x, bool):
        return False
    if not isinstance(x, numbers.Real):
        return False
    try:
        xf = float(x)
    except Exception:
        return False
    return not (math.isnan(xf) or math.isinf(xf))

def _first_idx_ge(xs: List[float], thr: float) -> Optional[int]:
    for i, v in enumerate(xs):
        if v >= thr:
            return i
    return None

def convergence_speed_from_history(history: List[float], minimize: bool = True) -> Optional[int]:
    """Iterations to reach 90% of final quality (doc). Returns 1-based iteration count."""
    if not history or len(history) < 2:
        return None
    # ensure best-so-far
    best = []
    cur = history[0]
    for v in history:
        if minimize:
            cur = min(cur, v)
        else:
            cur = max(cur, v)
        best.append(cur)

    start = best[0]
    final = best[-1]
    denom = (start - final) if minimize else (final - start)
    if denom == 0:
        return 1
    # normalized progress in [0,1]
    prog = []
    for v in best:
        p = (start - v) / denom if minimize else (v - start) / denom
        prog.append(p)
    idx = _first_idx_ge(prog, 0.9)
    return (idx + 1) if idx is not None else None

def compute_tsp_derived(problem: Dict[str, Any], metrics: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    tsp = problem.get("tsp", {}) or {}
    D = tsp.get("distance")
    if D is None:
        D = tsp.get("dist")
    if D is None:
        D = tsp.get("D")
    optimal = tsp.get("optimal", None)
    # tour length
    tl = metrics.get("tour_length") or metrics.get("best_length") or metrics.get("best_cost")
    if not _is_num(tl):
        tour = artifacts.get("best_solution") or artifacts.get("tour") or metrics.get("best_solution")
        if tour is not None and D is not None:
            try:
                s = 0.0
                tour = list(tour)
                n = len(tour)
                for i in range(n):
                    a = int(tour[i]); b = int(tour[(i+1)%n])
                    s += float(D[a][b])
                tl = s
            except Exception:
                tl = None
    if _is_num(tl):
        out["tour_length"] = float(tl)

    # gap%
    if _is_num(optimal) and _is_num(tl) and float(optimal) != 0.0:
        gap = (float(tl) - float(optimal)) / float(optimal) * 100.0
        out["known_optimal"] = float(optimal)
        out["gap_percentage"] = float(gap)
        # Project document: "within 5% of optimal". Use absolute gap to be
        # robust to minor numeric differences or mismatched baselines.
        out["success"] = bool(abs(gap) <= 5.0)
    else:
        out["success"] = None

    # convergence speed
    hist = artifacts.get("convergence_history") or artifacts.get("history") or metrics.get("convergence_history")
    if isinstance(hist, list) and hist and all(_is_num(x) for x in hist):
        cs = convergence_speed_from_history([float(x) for x in hist], minimize=True)
        if cs is not None:
            out["convergence_speed_iter"] = int(cs)
    return out

def compute_functionopt_derived(problem: Dict[str, Any], metrics: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    best = metrics.get("best_fitness") or metrics.get("best_cost") or metrics.get("fitness")
    if _is_num(best):
        out["best_fitness"] = float(best)
    optimum = (problem.get("meta", {}) or {}).get("optimum_hint", 0.0)
    if _is_num(best) and _is_num(optimum):
        err = abs(float(best) - float(optimum))
        out["error"] = float(err)
        out["success"] = bool(err < 50)
    else:
        out["success"] = None

    # function evaluations from counter
    obj = problem.get("objective", {}) or {}
    counter = obj.get("counter")
    if isinstance(counter, dict) and _is_num(counter.get("count", None)):
        out["function_evaluations"] = int(counter["count"])
    return out

def compute_classification_derived(problem: Dict[str, Any], metrics: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # AUC requires y_true and y_score probabilities
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    y_true = artifacts.get("y_true")
    y_score = artifacts.get("y_score") or artifacts.get("y_prob")
    if roc_auc_score and y_true is not None and y_score is not None:
        try:
            auc = float(roc_auc_score(y_true, y_score))
            out["auc_roc"] = auc
        except Exception:
            pass
    # cross-val score (if computed)
    cv = metrics.get("cv_accuracy_mean") or metrics.get("cross_validation_score")
    if _is_num(cv):
        out["cross_validation_score"] = float(cv)
    return out

def compute_clustering_derived(problem: Dict[str, Any], metrics: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except Exception:
        return out

    ml = problem.get("ml", {}) or {}
    X = ml.get("X")
    y_true = ml.get("y")
    labels = artifacts.get("labels") or artifacts.get("assignments") or artifacts.get("cluster_labels")
    if X is None or labels is None:
        return out
    try:
        import numpy as np
        X_arr = np.asarray(X, dtype=float)
        lab = np.asarray(labels, dtype=int)
        if X_arr.shape[0] == lab.shape[0]:
            # Always metrics
            if len(set(lab.tolist())) > 1:
                out["silhouette"] = float(silhouette_score(X_arr, lab))
                out["davies_bouldin"] = float(davies_bouldin_score(X_arr, lab))
                out["calinski_harabasz"] = float(calinski_harabasz_score(X_arr, lab))
            # Inertia may already be there for kmeans
            if _is_num(metrics.get("inertia")):
                out["inertia"] = float(metrics["inertia"])
            # When true labels available
            if y_true is not None:
                yt = np.asarray(y_true, dtype=int)
                if yt.shape[0] == lab.shape[0]:
                    out["ari"] = float(adjusted_rand_score(yt, lab))
                    out["nmi"] = float(normalized_mutual_info_score(yt, lab))
    except Exception:
        pass
    return out

def add_derived_metrics(problem: Dict[str, Any], result_metrics: Dict[str, Any], result_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    pt = (problem.get("problem_type") or problem.get("type") or "").lower().strip()
    derived: Dict[str, Any] = {}
    if pt == "tsp":
        derived = compute_tsp_derived(problem, result_metrics, result_artifacts)
    elif pt == "function_optimization":
        derived = compute_functionopt_derived(problem, result_metrics, result_artifacts)
    elif pt == "classification":
        derived = compute_classification_derived(problem, result_metrics, result_artifacts)
    elif pt == "clustering":
        derived = compute_clustering_derived(problem, result_metrics, result_artifacts)
    # default success for problems where success isn't defined in the document
    if "success" not in derived and pt not in ("tsp", "function_optimization"):
        derived["success"] = True

    # merge without overwriting existing explicit metrics (explicit wins)
    out = dict(derived)
    for k,v in result_metrics.items():
        out[k]=v
    return out
