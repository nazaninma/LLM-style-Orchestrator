from __future__ import annotations

from typing import Any, Dict


def configure_params(selected_method: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Return reasonable default parameters for each algorithm.

    These defaults are conservative and can be overridden by:
      settings.method_params in the YAML config.
    """
    meta = parsed.get("meta", {}) or {}

    if selected_method == "noop":
        return {}

    if selected_method == "random_search":
        dim = int(meta.get("dimension", 10) or 10)
        return {
            "n_samples": int(meta.get("n_samples", 5000)),
            "checkpoint_every": int(meta.get("checkpoint_every", 500)),
            "dimension": dim,
        }

    # Continuous optimization
    if selected_method == "pso":
        return {"n_particles": 50, "max_iterations": 500, "w": 0.7, "c1": 1.5, "c2": 1.5, "w_decay": True, "velocity_clamp": 0.5}

    if selected_method == "ga":
        return {"population_size": 120, "generations": 500, "crossover_rate": 0.85, "mutation_rate": 0.15, "tournament_k": 3}

    if selected_method == "gp":
        return {"population_size": 80, "generations": 300, "program_len": 12, "crossover_rate": 0.8, "mutation_rate": 0.2, "elitism": 2}

    if selected_method == "fuzzy":
        return {"max_iterations": 2000, "step0": 0.25, "min_step": 1e-6, "max_step": 1.0, "report_every": 50}

    # TSP
    if selected_method == "aco":
        return {"max_iterations": 400, "n_ants": 40, "alpha": 1.0, "beta": 3.0, "rho": 0.3, "q": 1.0}

    if selected_method == "ga_tsp":
        return {"population_size": 200, "generations": 600, "crossover_rate": 0.9, "mutation_rate": 0.2}

    if selected_method == "hopfield":
        return {"max_iterations": 3000, "dt": 0.01, "A": 500.0, "B": 500.0, "C": 200.0, "D": 1.0, "temperature": 0.5, "report_every": 100}

    # Classification
    if selected_method == "perceptron":
        return {"max_epochs": 200, "lr": 0.05}

    if selected_method == "mlp":
        return {"hidden_sizes": [16, 8], "max_epochs": 400, "lr": 0.01, "batch_size": 32}

    # Clustering
    if selected_method == "kmeans":
        return {"k": int(meta.get("k", 3) or 3), "max_iterations": 200}

    if selected_method == "som":
        return {"grid": [10, 10], "max_iterations": 2000, "lr0": 0.5, "sigma0": 3.0}

    # Fallback
    return {}
