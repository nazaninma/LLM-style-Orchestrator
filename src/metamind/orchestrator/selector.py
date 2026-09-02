from __future__ import annotations

from typing import Any, Dict, Tuple


def _first_available(candidates: list[str], available: set[str], default: str) -> str:
    for c in candidates:
        if c in available:
            return c
    return default


def select_method(parsed: Dict[str, Any], available_methods: set[str]) -> Tuple[str, str | None, str, float]:
    """
    Returns: (selected_method, backup_method, reasoning, confidence)

    This project uses a rule-based selector (LLM-style orchestration can be added later).
    The rules are aligned with the CI methods required by the assignment.
    """
    ptype = parsed.get("problem_type", "unknown")
    meta = parsed.get("meta", {}) or {}

    selected = "random_search"
    backup: str | None = "noop"
    confidence = 0.55
    reasoning = "Default baseline selection."

    if ptype == "tsp":
        n = int(meta.get("cities", meta.get("n_cities", 0)) or 0)

        if n and n <= 60:
            # small instances: ACO strong; Hopfield can be competitive; GA_TSP as fallback
            selected = _first_available(["aco", "hopfield", "ga_tsp", "random_search"], available_methods, "random_search")
            backup = _first_available(["hopfield", "ga_tsp", "aco", "random_search"], available_methods, None)  # type: ignore
            reasoning = "TSP small instance: prefer ACO; Hopfield/GA_TSP as backups for diversification."
            confidence = 0.82
        else:
            selected = _first_available(["ga_tsp", "aco", "hopfield", "random_search"], available_methods, "random_search")
            backup = _first_available(["aco", "hopfield", "ga_tsp", "random_search"], available_methods, None)  # type: ignore
            reasoning = "TSP larger instance: prefer GA_TSP for scalability; ACO/Hopfield as backups."
            confidence = 0.75

    elif ptype == "function_optimization":
        fname = str(meta.get("function", "")).lower()
        dim = int(meta.get("dimension", 0) or 0)

        if fname in ("rastrigin", "ackley"):
            selected = _first_available(["pso", "gp", "ga", "fuzzy", "random_search"], available_methods, "random_search")
            backup = _first_available(["gp", "ga", "fuzzy", "pso"], available_methods, None)  # type: ignore
            reasoning = f"{fname.title()} is multi-modal; PSO is a strong default; GP/GA/Fuzzy as backups."
            confidence = 0.80
        elif fname == "rosenbrock":
            selected = _first_available(["ga", "gp", "pso", "fuzzy", "random_search"], available_methods, "random_search")
            backup = _first_available(["gp", "pso", "fuzzy", "ga"], available_methods, None)  # type: ignore
            reasoning = "Rosenbrock has a narrow curved valley; GA/GP often handle it well; PSO/Fuzzy as backups."
            confidence = 0.68
        else:
            # generic smooth-ish
            selected = _first_available(["gp", "pso", "ga", "fuzzy", "random_search"], available_methods, "random_search")
            backup = _first_available(["pso", "ga", "fuzzy", "gp"], available_methods, None)  # type: ignore
            reasoning = "Generic continuous optimization: GP offers diverse search; PSO/GA/Fuzzy as backups."
            confidence = 0.72

    elif ptype == "classification":
        selected = _first_available(["mlp", "perceptron", "random_search", "noop"], available_methods, "noop")
        backup = _first_available(["perceptron", "mlp"], available_methods, None)  # type: ignore
        reasoning = "Classification: MLP is usually stronger than a linear perceptron; perceptron as backup."
        confidence = 0.82

    elif ptype == "clustering":
        selected = _first_available(["som", "kmeans", "random_search", "noop"], available_methods, "noop")
        backup = _first_available(["kmeans", "som"], available_methods, None)  # type: ignore
        reasoning = "Clustering: SOM is suitable for structure discovery; KMeans as backup."
        confidence = 0.82

    else:
        selected = _first_available(["random_search", "noop"], available_methods, "noop")
        backup = None
        reasoning = "Unknown problem type: falling back to baselines."
        confidence = 0.50

    if selected not in available_methods:
        selected = "noop" if "noop" in available_methods else next(iter(available_methods))

    if backup == selected:
        backup = None

    return selected, backup, reasoning, confidence
