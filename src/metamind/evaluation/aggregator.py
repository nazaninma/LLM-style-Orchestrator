from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import numbers


def _is_number(x: Any) -> bool:
    """Robust numeric check (accepts numpy scalars too)."""
    if isinstance(x, bool):
        return False
    if not isinstance(x, numbers.Real):
        return False
    try:
        xf = float(x)
    except Exception:
        return False
    return not (math.isnan(xf) or math.isinf(xf))


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return float(m), 0.0
    var = sum((v - m) ** 2 for v in xs) / (len(xs) - 1)
    return float(m), float(math.sqrt(var))


def ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    """95% confidence interval for the mean (t-distribution when possible)."""
    if n <= 1:
        return mean, mean
    # fall back to normal approx if scipy isn't present
    try:
        from scipy.stats import t

        tval = float(t.ppf(0.975, df=n - 1))
    except Exception:
        tval = 1.96
    half = tval * (std / math.sqrt(n))
    return float(mean - half), float(mean + half)


def pick_primary_score(problem_type: str, metrics: Dict[str, Any]) -> Tuple[str, Optional[float], bool]:
    """
    Returns: (metric_name, value, maximize?)
    maximize? tells ranking direction.
    """
    p = (problem_type or "").lower().strip()

    # function / tsp -> follow project document primary metrics
    if p == "tsp":
        v = metrics.get("tour_length")
        if _is_number(v):
            return "tour_length", float(v), False
        v = metrics.get("best_length") or metrics.get("best_cost")
        if _is_number(v):
            return "tour_length", float(v), False
        return "tour_length", None, False

    if p == "function_optimization":
        v = metrics.get("best_fitness")
        if _is_number(v):
            return "best_fitness", float(v), False
        v = metrics.get("fitness") or metrics.get("best_cost")
        if _is_number(v):
            return "best_fitness", float(v), False
        return "best_fitness", None, False

    # classification -> project document ranks mainly by accuracy
    if p == "classification":
        for k in ("test_accuracy", "accuracy", "val_accuracy"):
            v = metrics.get(k, None)
            if _is_number(v):
                return "test_accuracy", float(v), True
        for k in ("test_f1", "f1", "val_f1"):
            v = metrics.get(k, None)
            if _is_number(v):
                return "test_accuracy", float(v), True
        return "test_accuracy", None, True

    # clustering -> project document uses Silhouette (higher better) as always metric
    if p == "clustering":
        for k in ("silhouette", "silhouette_score"):
            v = metrics.get(k, None)
            if _is_number(v):
                return "silhouette", float(v), True
        v = metrics.get("inertia")
        if _is_number(v):
            return "silhouette", float(v), True  # still maximize placeholder
        return "silhouette", None, True

    # unknown
    for k, v in metrics.items():
        if _is_number(v):
            return k, float(v), False
    return "score", None, False


@dataclass
class MethodStats:
    method: str
    primary_metric: str
    primary_mean: float
    primary_std: float
    primary_ci_low: float
    primary_ci_high: float
    maximize: bool
    n_runs: int
    n_success: int
    n_fail: int
    success_rate: float
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]


def aggregate_runs(problem_type: str, runs: Dict[str, List[Dict[str, Any]]]) -> List[MethodStats]:
    """
    runs: method -> list of {"metrics": {...}, "artifacts": {...}, ...}
    """
    out: List[MethodStats] = []

    for method, items in runs.items():
        if not items:
            continue

        # determine primary metric from first run (or best available)
        primary_key, _, maximize = pick_primary_score(problem_type, items[0].get("metrics", {}) or {})

        # gather numeric metrics across runs (doc-defined success if available)
        all_metric_keys = set()
        n_error = 0
        n_valid = 0
        n_success_quality = 0

        for it in items:
            if it.get("error"):
                n_error += 1
                continue
            n_valid += 1
            m = (it.get("metrics", {}) or {})
            succ = m.get("success", None)
            # success can be bool or numeric (e.g., numpy.bool_ / 0/1)
            try:
                is_succ = bool(succ) if isinstance(succ, (bool, int, float)) else False
            except Exception:
                is_succ = False
            if is_succ:
                n_success_quality += 1
            for k, v in m.items():
                if _is_number(v):
                    all_metric_keys.add(k)

        metrics_mean: Dict[str, float] = {}
        metrics_std: Dict[str, float] = {}
        for k in sorted(all_metric_keys):
            vals = []
            for it in items:
                v = (it.get("metrics", {}) or {}).get(k, None)
                if _is_number(v):
                    vals.append(float(v))
            m, s = mean_std(vals)
            metrics_mean[k] = m
            metrics_std[k] = s

        primary_vals: List[float] = []
        for it in items:
            if it.get("error"):
                continue
            v = (it.get("metrics", {}) or {}).get(primary_key, None)
            if _is_number(v):
                primary_vals.append(float(v))
        pm, ps = mean_std(primary_vals)
        lo, hi = ci95(pm, ps, max(1, len(primary_vals)))

        out.append(
            MethodStats(
                method=method,
                primary_metric=primary_key,
                primary_mean=float(pm),
                primary_std=float(ps),
                primary_ci_low=float(lo),
                primary_ci_high=float(hi),
                maximize=bool(maximize),
                n_runs=len(items),
                n_success=int(n_success_quality),
                n_fail=int(n_error),
                success_rate=float(n_success_quality / n_valid) if n_valid > 0 else 0.0,
                metrics_mean=metrics_mean,
                metrics_std=metrics_std,
            )
        )

    # sort by primary metric
    def key_fn(ms: MethodStats):
        return -ms.primary_mean if ms.maximize else ms.primary_mean

    out.sort(key=key_fn)
    return out