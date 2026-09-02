from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class ProgressEvent:
    iteration: int
    best_fitness: float
    metrics: Dict[str, float]
    payload: Dict[str, Any]


ProgressCallback = Callable[[ProgressEvent], None]


def default_progress_callback(_: ProgressEvent) -> None:
    # no-op default
    return


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 20
    min_delta: float = 1e-9


class EarlyStopper:
    """
    Generic early stopping:
    - Tracks best_fitness improvement
    - Stops if no improvement >= min_delta for 'patience' iterations
    """

    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best: Optional[float] = None
        self.bad_count: int = 0

    def step(self, current_best: float) -> bool:
        if not self.cfg.enabled:
            return False

        if self.best is None:
            self.best = current_best
            self.bad_count = 0
            return False

        # For minimization: improvement means smaller
        improved = (self.best - current_best) >= self.cfg.min_delta
        if improved:
            self.best = current_best
            self.bad_count = 0
            return False

        self.bad_count += 1
        return self.bad_count >= self.cfg.patience