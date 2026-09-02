from __future__ import annotations

from typing import Any, Dict


def parse_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize problem dict into a parsed representation for the orchestrator.
    Supports both:
      - problem["problem_type"]
      - problem["type"]
    """
    if not isinstance(problem, dict):
        raise TypeError("problem must be a dict")

    ptype = (problem.get("problem_type") or problem.get("type") or "").strip().lower()
    if not ptype:
        ptype = "unknown"

    parsed: Dict[str, Any] = {
        "problem_type": ptype,
        "name": problem.get("name", "unknown_problem"),
        "meta": problem.get("meta", {}) or {},
        "ml": problem.get("ml", {}) or {},
        "raw": problem,
    }

    if ptype == "clustering":
        ml = parsed["ml"]
        if "X" not in ml:
            raise ValueError("Clustering problem requires ml['X']")

    if ptype == "classification":
        ml = parsed["ml"]
        if not (("X_train" in ml and "y_train" in ml) or ("X" in ml and "y" in ml)):
            raise ValueError("Classification problem requires ml data (X/y or X_train/y_train)")

    return parsed