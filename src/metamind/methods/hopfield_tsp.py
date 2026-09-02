from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_float, require_int
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper

log = logging.getLogger(__name__)


class HopfieldTSPMethod:
    """
    Hopfield network heuristic for TSP (continuous-state Hopfield-Tank style).

    Expects problem dict with:
      - name: str
      - tsp: {"D": ndarray [n,n], "coords": ndarray [n,2] (optional), "optimum": float (optional)}
    """

    name = "hopfield"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        max_iterations = require_int(params, "max_iterations", default=2000, min_v=50, max_v=200_000)
        dt = require_float(params, "dt", default=0.01, min_v=1e-5, max_v=1.0)
        # Penalty / energy weights
        A = require_float(params, "A", default=500.0, min_v=0.0, max_v=1e6)
        B = require_float(params, "B", default=500.0, min_v=0.0, max_v=1e6)
        C = require_float(params, "C", default=200.0, min_v=0.0, max_v=1e6)
        D_w = require_float(params, "D", default=1.0, min_v=0.0, max_v=1e3)
        # temperature for softmax
        temp = require_float(params, "temperature", default=0.5, min_v=0.05, max_v=5.0)
        report_every = require_int(params, "report_every", default=100, min_v=1, max_v=1_000_000)
        return {
            "max_iterations": max_iterations,
            "dt": dt,
            "A": A,
            "B": B,
            "C": C,
            "D": D_w,
            "temperature": temp,
            "report_every": report_every,
        }

    @staticmethod
    def _tour_length(D: np.ndarray, tour: List[int]) -> float:
        n = len(tour)
        s = 0.0
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            s += float(D[a, b])
        return float(s)

    def solve(
        self,
        problem: Dict[str, Any],
        params: Dict[str, Any],
        ctx: RunContext,
        progress_cb: ProgressCallback = default_progress_callback,
    ) -> RunResult:
        p = self.validate_params(params)
        tsp = problem["tsp"]
        Dmat: np.ndarray = np.asarray(tsp["D"], dtype=float)
        n = int(Dmat.shape[0])
        optimum = tsp.get("optimum", None)

        rng = np.random.default_rng(ctx.seed)

        # State matrix V: V[i, j] ~ probability city i is in position j
        V = rng.random((n, n))
        V = V / (V.sum(axis=0, keepdims=True) + 1e-12)

        stopper = EarlyStopper(ctx.early_stopping)

        best_tour: Optional[List[int]] = None
        best_len: float = float("inf")

        start = time.time()

        # Precompute shifted position index for neighbor term
        # We'll use cyclic next position
        for it in range(p["max_iterations"]):
            if ctx.time_limit_sec and (time.time() - start) > ctx.time_limit_sec:
                break

            # Row and column sums
            row_sum = V.sum(axis=1, keepdims=True)  # city assigned across positions
            col_sum = V.sum(axis=0, keepdims=True)  # position filled across cities

            # Constraint gradients (soft penalties)
            dE_city = p["A"] * (row_sum - 1.0)  # shape (n,1)
            dE_pos = p["B"] * (col_sum - 1.0)   # shape (1,n)

            # Global activity penalty encourages exactly n ones total (soft)
            dE_global = p["C"] * (V.sum() - n)  # scalar

            # Distance term: sum over i,k and adjacent positions j,j+1
            # dE_dist/dV[i,j] ~ D * sum_k D[i,k] * (V[k,j+1] + V[k,j-1])
            V_next = np.roll(V, -1, axis=1)
            V_prev = np.roll(V, 1, axis=1)
            dist_term = p["D"] * (Dmat @ (V_next + V_prev))  # (n,n)

            # Total gradient
            grad = dE_city + dE_pos + dE_global + dist_term  # broadcast to (n,n)

            # Update (gradient descent on energy) with softmax-like squashing
            U = V - p["dt"] * grad
            # Squash to (0,1) and re-normalize columns (positions)
            U = U / p["temperature"]
            U = U - U.max(axis=0, keepdims=True)  # stability
            V = np.exp(U)
            V = V / (V.sum(axis=0, keepdims=True) + 1e-12)

            if it % p["report_every"] == 0 or it == p["max_iterations"] - 1:
                # Decode tour by position: pick city with max prob in each position, then fix duplicates
                pos_choice = np.argmax(V, axis=0).tolist()  # length n, may have duplicates
                # Repair into permutation:
                seen = set()
                tour = [-1] * n
                missing = [i for i in range(n) if i not in pos_choice]
                miss_i = 0
                for j, c in enumerate(pos_choice):
                    if c not in seen:
                        tour[j] = c
                        seen.add(c)
                    else:
                        tour[j] = missing[miss_i]
                        miss_i += 1
                tour_len = self._tour_length(Dmat, tour)

                if tour_len < best_len:
                    best_len = tour_len
                    best_tour = tour

                gap = None
                if optimum is not None:
                    try:
                        gap = float(best_len) - float(optimum)
                    except Exception:
                        gap = None

                progress_cb(
                    ProgressEvent(iteration=it, best_fitness=best_len, metrics=dict(({"gap": gap} if gap is not None else {}), **{"best_len": float(best_len)}), payload={"tour": best_tour})
                )

                if stopper.step(best_len):
                    break

        runtime = time.time() - start

        if best_tour is None:
            # fallback: greedy decode
            best_tour = np.argmax(V, axis=0).tolist()
            best_len = self._tour_length(Dmat, best_tour)

        meta = RunMetadata(
            run_id=ctx.run_dir.name,
            timestamp_utc=problem.get("timestamp_utc", ""),
            seed=ctx.seed,
            problem_name=problem.get("name", "unknown_problem"),
            method_name=self.name,
        )

        metrics = {
            "best_fitness": float(best_len),
            **({"gap": float(best_len - float(optimum))} if optimum is not None else {}),
            "time_sec": float(runtime),
            "n_iterations": float(it + 1),
        }

        artifacts = {
            "tour": best_tour,
            "tour_length": float(best_len),
        }

        return RunResult(metadata=meta, metrics=metrics, artifacts=artifacts)
