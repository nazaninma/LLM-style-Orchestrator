from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple, List

import numpy as np

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_float, require_int
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper
from .ml_utils import standardize_fit, standardize_transform

log = logging.getLogger(__name__)


class SOMMethod:
    """
    2D Self-Organizing Map for clustering.

    ✅ New format (Phase 5/6):
      problem = {
        "name": ...,
        "problem_type": "clustering",
        "ml": {"X": ...}
      }

    ✅ Backward compatible (old format):
      problem = {
        "name": ...,
        "cluster": {"X": ...}
      }
    """
    name = "som"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        grid_x = require_int(params, "grid_x", default=10, min_v=2, max_v=200)
        grid_y = require_int(params, "grid_y", default=10, min_v=2, max_v=200)
        epochs = require_int(params, "epochs", default=100, min_v=1, max_v=200000)

        lr = require_float(params, "lr", default=0.5, min_v=1e-6, max_v=5.0)
        sigma = require_float(params, "sigma", default=3.0, min_v=0.1, max_v=100.0)

        return {"grid_x": grid_x, "grid_y": grid_y, "epochs": epochs, "lr": lr, "sigma": sigma}

    def _extract_X(self, problem: Dict[str, Any]) -> np.ndarray:
        # ✅ New pipeline format
        ml = problem.get("ml", {}) or {}
        if "X" in ml:
            return np.asarray(ml["X"], dtype=float)

        # ✅ Backward compatible
        cluster = problem.get("cluster", {}) or {}
        if "X" in cluster:
            return np.asarray(cluster["X"], dtype=float)

        raise ValueError("SOM expects X in problem['ml']['X'] (new) or problem['cluster']['X'] (old).")

    def solve(
        self,
        problem: Dict[str, Any],
        params: Dict[str, Any],
        ctx: RunContext,
        progress_cb: ProgressCallback = default_progress_callback,
    ) -> RunResult:
        p = self.validate_params(params)

        # ✅ unified X access
        X = self._extract_X(problem)

        # standardize always (consistent with clustering pipeline)
        scaler = standardize_fit(X)
        X = standardize_transform(X, scaler)

        if X.ndim != 2:
            raise ValueError(f"SOM expects 2D X. Got shape={getattr(X, 'shape', None)}")

        n, d = X.shape
        rng = np.random.default_rng(ctx.seed)

        gx, gy = p["grid_x"], p["grid_y"]
        weights = rng.normal(0, 0.5, size=(gx, gy, d))

        # precompute neuron coords
        coords = np.array([(i, j) for i in range(gx) for j in range(gy)], dtype=float).reshape(gx, gy, 2)

        def bmu(x: np.ndarray) -> Tuple[int, int, float]:
            diff = weights - x.reshape(1, 1, d)
            dist2 = np.sum(diff * diff, axis=2)
            idx = np.unravel_index(np.argmin(dist2), dist2.shape)
            return int(idx[0]), int(idx[1]), float(dist2[idx])

        def neighborhood(dist2: np.ndarray, sigma: float) -> np.ndarray:
            return np.exp(-dist2 / (2.0 * sigma * sigma))

        start = time.time()
        history_qe: List[float] = []  # quantization error per epoch
        early = EarlyStopper(ctx.early_stopping)

        for ep in range(p["epochs"]):
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("SOM time limit reached at epoch=%s", ep)
                break

            # decay lr and sigma
            lr = p["lr"] * (1.0 - ep / max(1, p["epochs"] - 1))
            sigma = p["sigma"] * (1.0 - ep / max(1, p["epochs"] - 1))
            sigma = max(0.1, sigma)

            # shuffle samples
            idx = np.arange(n)
            rng.shuffle(idx)

            qe_sum = 0.0
            for i in idx:
                x = X[i]
                bx, by, dist2_b = bmu(x)
                qe_sum += dist2_b

                # neighborhood influence
                diff = coords - np.array([bx, by], dtype=float).reshape(1, 1, 2)
                grid_dist2 = np.sum(diff * diff, axis=2)
                h = neighborhood(grid_dist2, sigma).reshape(gx, gy, 1)

                weights += lr * h * (x.reshape(1, 1, d) - weights)

            qe = float(qe_sum / n)
            history_qe.append(qe)

            if ep == 0 or ep % max(1, p["epochs"] // 20) == 0 or ep == p["epochs"] - 1:
                # ✅ progress_cb safe call
                if progress_cb is not None:
                    progress_cb(
                        ProgressEvent(
                            iteration=ep,
                            best_fitness=qe,
                            metrics={"quantization_error": qe},
                            payload={},
                        )
                    )

            if early.step(qe):
                log.info("SOM early stopping at epoch=%s qe=%s", ep, qe)
                break

        elapsed = time.time() - start

        # assign cluster ids as BMU indices
        assignments: List[int] = []
        for i in range(n):
            bx, by, _ = bmu(X[i])
            assignments.append(bx * gy + by)

        meta = RunMetadata(
            run_id=ctx.run_dir.name,
            timestamp_utc=problem.get("timestamp_utc", ""),
            seed=ctx.seed,
            problem_name=problem.get("name", "unknown_problem"),
            method_name=self.name,
        )

        final_qe = float(history_qe[-1] if history_qe else 0.0)

        metrics = {
            "quantization_error": final_qe,
            "time_sec": float(elapsed),
            "iterations_completed": float(len(history_qe)),
        }

        # ✅ IMPORTANT: for Phase 6 plotting (generic key)
        artifacts = {
            "assignments": assignments,
            "weights_shape": list(weights.shape),
            "history_quantization_error": history_qe,     # keep old key
            "convergence_history": history_qe,            # ✅ plotting reads this
        }

        return RunResult(
            metadata=meta,
            metrics=metrics,
            artifacts=artifacts,
            notes="som clustering",
        )
