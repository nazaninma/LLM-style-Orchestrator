from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Tuple

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_float, require_int, save_checkpoint, load_checkpoint
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper

log = logging.getLogger(__name__)


class GAContinuousMethod:
    """
    Real-coded Genetic Algorithm for continuous function minimization.

    Expects problem dict with:
      - name: str
      - objective: {"fn": callable, "bounds": List[(lo,hi)], "optimum": float (optional)}
    """

    name = "ga"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        population_size = require_int(params, "population_size", default=100, min_v=10, max_v=10000)
        generations = require_int(params, "generations", default=500, min_v=10, max_v=1_000_000)

        crossover_rate = require_float(params, "crossover_rate", default=0.8, min_v=0.0, max_v=1.0)
        mutation_rate = require_float(params, "mutation_rate", default=0.1, min_v=0.0, max_v=1.0)

        tournament_size = require_int(params, "tournament_size", default=3, min_v=2, max_v=50)
        elitism = require_int(params, "elitism", default=2, min_v=0, max_v=1000)

        mutation_sigma = require_float(params, "mutation_sigma", default=0.1, min_v=0.0, max_v=10.0)
        checkpoint_every = require_int(params, "checkpoint_every", default=0, min_v=0, max_v=1_000_000)

        return {
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "elitism": elitism,
            "mutation_sigma": mutation_sigma,
            "checkpoint_every": checkpoint_every,
        }

    def solve(
        self,
        problem: Dict[str, Any],
        params: Dict[str, Any],
        ctx: RunContext,
        progress_cb: ProgressCallback = default_progress_callback,
    ) -> RunResult:
        p = self.validate_params(params)

        obj = problem["objective"]
        fn = obj["fn"]
        bounds: List[Tuple[float, float]] = obj["bounds"]
        optimum = obj.get("optimum", None)

        dim = len(bounds)
        rng = random.Random(ctx.seed)

        def clip(x: float, lo: float, hi: float) -> float:
            return lo if x < lo else hi if x > hi else x

        def rand_individual() -> List[float]:
            return [rng.uniform(lo, hi) for (lo, hi) in bounds]

        def eval_f(x: List[float]) -> float:
            return float(fn(x))

        def is_better(a: float, b: float) -> bool:
            return a < b if ctx.minimize else a > b

        def tournament_select(pop: List[List[float]], fit: List[float]) -> List[float]:
            k = p["tournament_size"]
            idxs = [rng.randrange(len(pop)) for _ in range(k)]
            best_idx = idxs[0]
            for j in idxs[1:]:
                if is_better(fit[j], fit[best_idx]):
                    best_idx = j
            return pop[best_idx][:]

        def blend_crossover(a: List[float], b: List[float], alpha: float = 0.5) -> Tuple[List[float], List[float]]:
            # BLX-alpha style
            c1, c2 = [], []
            for d in range(dim):
                lo = min(a[d], b[d])
                hi = max(a[d], b[d])
                span = hi - lo
                low = lo - alpha * span
                high = hi + alpha * span
                v1 = clip(rng.uniform(low, high), bounds[d][0], bounds[d][1])
                v2 = clip(rng.uniform(low, high), bounds[d][0], bounds[d][1])
                c1.append(v1)
                c2.append(v2)
            return c1, c2

        def mutate(x: List[float]) -> List[float]:
            # gaussian mutation per gene with probability mutation_rate
            out = x[:]
            for d in range(dim):
                if rng.random() < p["mutation_rate"]:
                    span = bounds[d][1] - bounds[d][0]
                    sigma = p["mutation_sigma"] * span
                    out[d] = clip(out[d] + rng.gauss(0.0, sigma), bounds[d][0], bounds[d][1])
            return out

        ckpt_path = ctx.checkpoint_path("ga_ckpt.json")
        ckpt = load_checkpoint(ckpt_path)

        start_gen = 0
        start = time.time()

        if ckpt:
            log.info("Resuming GA from checkpoint: %s", ckpt_path)
            start_gen = int(ckpt.get("gen", 0))
            population = ckpt["population"]
            fitness = ckpt["fitness"]
            best_x = ckpt["best_x"]
            best_f = float(ckpt["best_f"])
            history: List[float] = list(ckpt.get("history", []))
        else:
            population = [rand_individual() for _ in range(p["population_size"])]
            fitness = [eval_f(ind) for ind in population]
            best_idx = min(range(len(fitness)), key=lambda i: fitness[i]) if ctx.minimize else max(
                range(len(fitness)), key=lambda i: fitness[i]
            )
            best_x = population[best_idx][:]
            best_f = float(fitness[best_idx])
            history = [best_f]

        early = EarlyStopper(ctx.early_stopping)

        for gen in range(start_gen, p["generations"]):
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("GA time limit reached at gen=%s", gen)
                break

            # elitism: keep top k
            elitism_k = min(p["elitism"], p["population_size"])
            ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=not ctx.minimize)
            new_pop: List[List[float]] = [population[i][:] for i in ranked[:elitism_k]]

            # generate rest
            while len(new_pop) < p["population_size"]:
                parent1 = tournament_select(population, fitness)
                parent2 = tournament_select(population, fitness)

                if rng.random() < p["crossover_rate"]:
                    child1, child2 = blend_crossover(parent1, parent2, alpha=0.5)
                else:
                    child1, child2 = parent1[:], parent2[:]

                child1 = mutate(child1)
                if len(new_pop) < p["population_size"]:
                    new_pop.append(child1)

                if len(new_pop) < p["population_size"]:
                    child2 = mutate(child2)
                    new_pop.append(child2)

            population = new_pop
            fitness = [eval_f(ind) for ind in population]

            # update best
            best_idx = min(range(len(fitness)), key=lambda i: fitness[i]) if ctx.minimize else max(
                range(len(fitness)), key=lambda i: fitness[i]
            )
            if is_better(fitness[best_idx], best_f):
                best_f = float(fitness[best_idx])
                best_x = population[best_idx][:]

            history.append(float(best_f))

            # progress
            if gen == 0 or gen % max(1, p["generations"] // 50) == 0 or gen == p["generations"] - 1:
                gap = None if optimum is None else abs(float(best_f) - float(optimum))
                progress_cb(
                    ProgressEvent(
                        iteration=gen,
                        best_fitness=float(best_f),
                        metrics={
                            "best_fitness": float(best_f),
                            **({"gap": float(gap)} if gap is not None else {}),
                        },
                        payload={"best_x": best_x},
                    )
                )

            # early stopping
            if early.step(float(best_f)):
                log.info("GA early stopping at gen=%s best=%s", gen, best_f)
                break

            # checkpoint
            if p["checkpoint_every"] > 0 and gen > 0 and (gen % p["checkpoint_every"] == 0):
                save_checkpoint(
                    ckpt_path,
                    {
                        "gen": gen,
                        "population": population,
                        "fitness": fitness,
                        "best_x": best_x,
                        "best_f": float(best_f),
                        "history": history[-5000:],
                    },
                )

        elapsed = time.time() - start

        meta = RunMetadata(
            run_id=ctx.run_dir.name,
            timestamp_utc=problem.get("timestamp_utc", ""),
            seed=ctx.seed,
            problem_name=problem.get("name", "unknown_problem"),
            method_name=self.name,
        )

        metrics = {
            "best_fitness": float(best_f),
            "time_sec": float(elapsed),
            "iterations_completed": float(len(history)),
        }
        if optimum is not None:
            metrics["error_to_optimum"] = float(abs(float(best_f) - float(optimum)))

        return RunResult(
            metadata=meta,
            metrics=metrics,
            artifacts={
                "best_x": best_x,
                "convergence_history": history,
                "checkpoint_path": str(ckpt_path),
            },
            notes="ga continuous optimizer",
        )