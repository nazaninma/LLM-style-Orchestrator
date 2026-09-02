from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .function_problems import FunctionProblem
from .tsp_problems import TSPProblem
from .classification_problems import ClassificationProblem
from .clustering_problems import ClusteringProblem


def load_problem(cfg, root: Path) -> Dict[str, Any]:
    """
    Main entry point for building problems based on config.

    We build a merged bundle (dict) from cfg.settings and inject:
      - _seed
      - _default_problem

    Then dispatch based on settings.problem_type.
    """
    settings = cfg.settings or {}
    ptype = settings.get("problem_type")
    if not ptype:
        raise ValueError("settings.problem_type must be defined in config")

    bundle = dict(settings)
    bundle["_seed"] = int(cfg.seed)
    bundle["_default_problem"] = str(cfg.default_problem)
    bundle["_root"] = str(root)

    if ptype == "function_optimization":
        return FunctionProblem(bundle).build()

    if ptype == "tsp":
        return TSPProblem(bundle).build()

    if ptype == "classification":
        return ClassificationProblem(bundle).build()

    if ptype == "clustering":
        return ClusteringProblem(bundle).build()

    raise ValueError(f"Unknown problem_type: {ptype}")