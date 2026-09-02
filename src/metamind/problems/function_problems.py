from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
import math
import numpy as np


class FunctionProblem:
    """
    Phase 5.6 - Function Optimization Problem Builder
    """

    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle

    def build(self) -> Dict[str, Any]:
        meta_cfg = self.bundle.get("problem_meta", {}) or {}

        default_name = str(self.bundle.get("_default_problem", "sphere"))
        fn_name = str(meta_cfg.get("function", default_name)).strip().lower()
        dim = int(meta_cfg.get("dimension", 10))

        # default bounds
        bounds: List[Tuple[float, float]] = meta_cfg.get("bounds", None)
        if bounds is None:
            if fn_name == "rastrigin":
                bounds = [(-5.12, 5.12)] * dim
            elif fn_name == "rosenbrock":
                bounds = [(-5.0, 10.0)] * dim
            else:
                bounds = [(-5.12, 5.12)] * dim

        def sphere(x) -> float:
          x = np.asarray(x, dtype=float)
          return float(np.sum(x * x))

        def rastrigin(x) -> float:
            x = np.asarray(x, dtype=float)
            A = 10.0
            return float(A * x.size + np.sum(x * x - A * np.cos(2 * math.pi * x)))

        def rosenbrock(x) -> float:
            x = np.asarray(x, dtype=float)
            return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

        def ackley(x) -> float:
            x = np.asarray(x, dtype=float)
            a = 20.0
            b = 0.2
            c = 2 * math.pi
            n = x.size
            s1 = np.sum(x * x)
            s2 = np.sum(np.cos(c * x))
            term1 = -a * np.exp(-b * math.sqrt(s1 / n))
            term2 = -np.exp(s2 / n)
            return float(term1 + term2 + a + math.e)
        fns: Dict[str, Callable[[np.ndarray], float]] = {
            "sphere": sphere,
            "rastrigin": rastrigin,
            "rosenbrock": rosenbrock,
            "ackley": ackley,
        }
        if fn_name not in fns:
            raise ValueError(f"Unknown function '{fn_name}'. Supported: {sorted(fns.keys())}")


        # wrap objective with an evaluation counter (required metric: Function Evaluations)
        counter = {"count": 0}

        base_fn = fns[fn_name]

        def counted_fn(x) -> float:
            counter["count"] += 1
            return base_fn(x)
        meta = {
            "source": "builtin",
            "function": fn_name,
            "dimension": dim,
            "bounds": bounds,
            "optimum_hint": 0.0,
        }

        return {
            "name": fn_name,
            "problem_type": "function_optimization",
            "meta": meta,
            "objective": {
                "type": "function",
                "fn": counted_fn,
                "counter": counter,
                "bounds": bounds,
            },
        }