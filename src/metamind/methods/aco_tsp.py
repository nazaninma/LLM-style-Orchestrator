from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Tuple

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_float, require_int
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper
from .tsp_utils import tour_length, make_random_tour, two_opt_local_search

log = logging.getLogger(__name__)


class ACOTSPMethod:
    """
    Ant Colony Optimization for TSP.

    Expects problem dict with:
      - name: str
      - tsp: {"dist": 2D matrix OR "coords": [(x,y),...], "optimal": float (optional)}
    """
    name = "aco"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        n_ants = require_int(params, "n_ants", default=40, min_v=5, max_v=2000)
        n_iterations = require_int(params, "n_iterations", default=200, min_v=10, max_v=100000)

        alpha = require_float(params, "alpha", default=1.0, min_v=0.0, max_v=10.0)   # pheromone influence
        beta = require_float(params, "beta", default=2.0, min_v=0.0, max_v=10.0)     # heuristic influence
        rho = require_float(params, "rho", default=0.5, min_v=0.0, max_v=0.99)       # evaporation
        q = require_float(params, "q", default=1.0, min_v=0.0, max_v=1e6)            # deposit factor

        elite_weight = require_int(params, "elite_weight", default=5, min_v=0, max_v=1000)
        use_2opt = bool(params.get("use_2opt", True))

        return {
            "n_ants": n_ants,
            "n_iterations": n_iterations,
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "q": q,
            "elite_weight": elite_weight,
            "use_2opt": use_2opt,
        }

    def solve(
        self,
        problem: Dict[str, Any],
        params: Dict[str, Any],
        ctx: RunContext,
        progress_cb: ProgressCallback = default_progress_callback,
    ) -> RunResult:
        p = self.validate_params(params)

        tsp = problem["tsp"]
        dist: List[List[float]] = tsp["dist"]
        optimum = tsp.get("optimal", None)

        n = len(dist)
        rng = random.Random(ctx.seed)

        # initialize pheromone
        tau0 = 1.0
        pher = [[tau0] * n for _ in range(n)]

        # heuristic: eta = 1 / distance
        eps = 1e-12
        eta = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    eta[i][j] = 1.0 / (dist[i][j] + eps)

        def prob_next(cur: int, unvisited: set[int]) -> Tuple[List[int], List[float]]:
            choices = []
            weights = []
            for j in unvisited:
                choices.append(j)
                w = (pher[cur][j] ** p["alpha"]) * (eta[cur][j] ** p["beta"])
                weights.append(w)
            return choices, weights

        def weighted_choice(choices: List[int], weights: List[float]) -> int:
            s = sum(weights)
            if s <= 0:
                return rng.choice(choices)
            r = rng.random() * s
            cum = 0.0
            for c, w in zip(choices, weights):
                cum += w
                if cum >= r:
                    return c
            return choices[-1]

        start = time.time()
        best_tour = make_random_tour(n, rng)
        best_len = tour_length(best_tour, dist)
        history: List[float] = [best_len]

        early = EarlyStopper(ctx.early_stopping)

        for it in range(p["n_iterations"]):
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("ACO time limit reached at iter=%s", it)
                break

            tours: List[List[int]] = []
            lens: List[float] = []

            # build tours
            for _ in range(p["n_ants"]):
                start_city = rng.randrange(n)
                tour = [start_city]
                unvisited = set(range(n))
                unvisited.remove(start_city)

                cur = start_city
                while unvisited:
                    choices, weights = prob_next(cur, unvisited)
                    nxt = weighted_choice(choices, weights)
                    tour.append(nxt)
                    unvisited.remove(nxt)
                    cur = nxt

                if p["use_2opt"]:
                    tour = two_opt_local_search(tour, dist, max_improve=100)

                L = tour_length(tour, dist)
                tours.append(tour)
                lens.append(L)

                if L < best_len:
                    best_len = L
                    best_tour = tour[:]

            # evaporate pheromone
            rho = p["rho"]
            for i in range(n):
                for j in range(n):
                    pher[i][j] *= (1.0 - rho)

            # deposit pheromone from all ants
            for tour, L in zip(tours, lens):
                deposit = p["q"] / (L + eps)
                for k in range(n - 1):
                    a, b = tour[k], tour[k + 1]
                    pher[a][b] += deposit
                    pher[b][a] += deposit
                # close loop
                pher[tour[-1]][tour[0]] += deposit
                pher[tour[0]][tour[-1]] += deposit

            # elite reinforcement
            if p["elite_weight"] > 0:
                deposit = p["elite_weight"] * p["q"] / (best_len + eps)
                for k in range(n - 1):
                    a, b = best_tour[k], best_tour[k + 1]
                    pher[a][b] += deposit
                    pher[b][a] += deposit
                pher[best_tour[-1]][best_tour[0]] += deposit
                pher[best_tour[0]][best_tour[-1]] += deposit

            history.append(float(best_len))

            if it == 0 or it % max(1, p["n_iterations"] // 50) == 0 or it == p["n_iterations"] - 1:
                gap = None if optimum is None else (best_len - float(optimum)) / float(optimum) * 100.0
                progress_cb(
                    ProgressEvent(
                        iteration=it,
                        best_fitness=float(best_len),
                        metrics={
                            "best_tour_length": float(best_len),
                            **({"gap_percent": float(gap)} if gap is not None else {}),
                        },
                        payload={"best_tour": best_tour},
                    )
                )

            if early.step(float(best_len)):
                log.info("ACO early stopping at iter=%s best_len=%s", it, best_len)
                break

        elapsed = time.time() - start

        meta = RunMetadata(
            run_id=ctx.run_dir.name,
            timestamp_utc=problem.get("timestamp_utc", ""),
            seed=ctx.seed,
            problem_name=problem.get("name", "unknown_problem"),
            method_name=self.name,
        )

        metrics = {
            "tour_length": float(best_len),
            "time_sec": float(elapsed),
            "iterations_completed": float(len(history)),
        }
        if optimum is not None:
            metrics["gap_to_optimal_percent"] = float((best_len - float(optimum)) / float(optimum) * 100.0)

        return RunResult(
            metadata=meta,
            metrics=metrics,
            artifacts={
                "best_tour": best_tour,
                "convergence_history": history,
            },
            notes="aco tsp",
        )