from __future__ import annotations

from typing import Dict

from .base import BaseMethod
from .noop import NoOpMethod
from .random_search import RandomSearchMethod
from .pso import PSOMethod
from .ga_continuous import GAContinuousMethod
from .aco_tsp import ACOTSPMethod
from .ga_tsp import GATSPMethod
from .hopfield_tsp import HopfieldTSPMethod
from .gp import GPMethod
from .fuzzy_controller import FuzzyControllerMethod
from .perceptron import PerceptronMethod
from .mlp import MLPMethod
from .som import SOMMethod
from .kmeans import KMeansMethod


_METHODS: Dict[str, BaseMethod] = {
    # Baselines
    "noop": NoOpMethod(),
    "random_search": RandomSearchMethod(),

    # Continuous optimization
    "pso": PSOMethod(),
    "ga": GAContinuousMethod(),
    "gp": GPMethod(),
    "fuzzy": FuzzyControllerMethod(),

    # TSP / combinatorial
    "aco": ACOTSPMethod(),
    "ga_tsp": GATSPMethod(),
    "hopfield": HopfieldTSPMethod(),

    # ML / clustering
    "perceptron": PerceptronMethod(),
    "mlp": MLPMethod(),
    "som": SOMMethod(),
    "kmeans": KMeansMethod(),
}


def list_methods() -> Dict[str, BaseMethod]:
    return dict(_METHODS)


def get_method(name: str) -> BaseMethod:
    key = name.strip().lower()
    if key not in _METHODS:
        available = ", ".join(sorted(_METHODS.keys()))
        raise KeyError(f"Unknown method '{name}'. Available: {available}")
    return _METHODS[key]
