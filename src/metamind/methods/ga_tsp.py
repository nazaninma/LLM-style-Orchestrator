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


class GATSPMethod:
    """
    Genetic Algorithm for TSP using permutation representation.
    Operators:
      - Tournament selection
      - Order Crossover (OX)
      - Swap mutation
      - Optional 2-opt local search on offspring

    Expects problem dict with:
      - name: str
      - tsp: {"dist": 2D matrix OR "coords": [(x,y)...], "optimal": float (optional)}
    """

    name = "ga_tsp"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        population_size = require_int(params, "population_size", default=150, min_v=10, max_v=20000)
        generations = require_int(params, "generations", default=400, min_v=10, max_v=200000)

        crossover_rate = require_float(params, "crossover_rate", default=0.9, min_v=0.0, max_v=1.0)
        mutation_rate = require_float(params, "mutation_rate", default=0.2, min_v=0.0, max_v=1.0)

        tournament_size = require_int(params, "tournament_size", default=3, min_v=2, max_v=50)
        elitism = require_int(params, "elitism", default=2, min_v=0, max_v=5000)

        use_2opt = bool(params.get("use_2opt", True))

        return {
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "elitism": elitism,
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

        def is_better(a: float, b: float) -> bool:
            return a < b if ctx.minimize else a > b

        def fitness(tour: List[int]) -> float:
            return float(tour_length(tour, dist))

        def tournament_select(pop: List[List[int]], fit: List[float]) -> List[int]:
            k = p["tournament_size"]
            idxs = [rng.randrange(len(pop)) for _ in range(k)]
            best = idxs[0]
            for j in idxs[1:]:
                if is_better(fit[j], fit[best]):
                    best = j
            return pop[best][:]

        def order_crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
            # OX: pick slice from parent, fill remaining with order from other
            i = rng.randrange(n)
            k = rng.randrange(i, n)
            def make_child(p1: List[int], p2: List[int]) -> List[int]:
                child = [-1] * n
                child[i:k] = p1[i:k]
                used = set(child[i:k])
                fill = [x for x in p2 if x not in used]
                idx = 0
                for t in range(n):
                    if child[t] == -1:
                        child[t] = fill[idx]
                        idx += 1
                return child
            return make_child(a, b), make_child(b, a)

        def mutate_swap(tour: List[int]) -> List[int]:
            out = tour[:]
            if rng.random() < p["mutation_rate"]:
                i, j = rng.randrange(n), rng.randrange(n)
                out[i], out[j] = out[j], out[i]
            return out

        start = time.time()

        population: List[List[int]] = [make_random_tour(n, rng) for _ in range(p["population_size"])]
        if p["use_2opt"]:
            population = [two_opt_local_search(t, dist, max_improve=50) for t in population]

        fit = [fitness(t) for t in population]
        best_idx = min(range(len(fit)), key=lambda i: fit[i])
        best_tour = population[best_idx][:]
        best_len = float(fit[best_idx])

        history: List[float] = [best_len]
        early = EarlyStopper(ctx.early_stopping)

        for gen in range(p["generations"]):
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("GA_TSP time limit reached at gen=%s", gen)
                break

            elitism_k = min(p["elitism"], p["population_size"])
            ranked = sorted(range(len(population)), key=lambda i: fit[i])
            new_pop: List[List[int]] = [population[i][:] for i in ranked[:elitism_k]]

            while len(new_pop) < p["population_size"]:
                p1 = tournament_select(population, fit)
                p2 = tournament_select(population, fit)

                if rng.random() < p["crossover_rate"]:
                    c1, c2 = order_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                c1 = mutate_swap(c1)
                if p["use_2opt"]:
                    c1 = two_opt_local_search(c1, dist, max_improve=30)
                new_pop.append(c1)

                if len(new_pop) < p["population_size"]:
                    c2 = mutate_swap(c2)
                    if p["use_2opt"]:
                        c2 = two_opt_local_search(c2, dist, max_improve=30)
                    new_pop.append(c2)

            population = new_pop
            fit = [fitness(t) for t in population]

            best_idx = min(range(len(fit)), key=lambda i: fit[i])
            if is_better(fit[best_idx], best_len):
                best_len = float(fit[best_idx])
                best_tour = population[best_idx][:]

            history.append(best_len)

            if gen == 0 or gen % max(1, p["generations"] // 50) == 0 or gen == p["generations"] - 1:
                gap = None if optimum is None else (best_len - float(optimum)) / float(optimum) * 100.0
                progress_cb(
                    ProgressEvent(
                        iteration=gen,
                        best_fitness=float(best_len),
                        metrics={
                            "best_tour_length": float(best_len),
                            **({"gap_percent": float(gap)} if gap is not None else {}),
                        },
                        payload={"best_tour": best_tour},
                    )
                )

            if early.step(float(best_len)):
                log.info("GA_TSP early stopping at gen=%s best_len=%s", gen, best_len)
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
            notes="ga tsp permutation",
        )