from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Tuple

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_float, require_int
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper

log = logging.getLogger(__name__)


class GPMethod:
    """
    A lightweight 'Genetic Programming'-style optimizer for continuous domains.

    Implementation note:
    For this course project scaffold, GP is implemented as an expression-guided
    evolutionary search that evolves *programs* which generate candidate vectors.

    Each individual is a small program represented by a list of primitive ops
    that transforms a latent random vector into a candidate solution. This keeps
    the spirit of GP (program evolution) while staying compatible with the
    unified optimization API.

    Expects problem dict with:
      - objective: {"fn": callable, "bounds": List[(lo,hi)]}
    """

    name = "gp"

    PRIMS = ("add", "sub", "mul", "sin", "cos", "tanh", "clip")

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        population_size = require_int(params, "population_size", default=80, min_v=10, max_v=5000)
        generations = require_int(params, "generations", default=300, min_v=5, max_v=200_000)
        program_len = require_int(params, "program_len", default=12, min_v=3, max_v=200)
        tournament_k = require_int(params, "tournament_k", default=3, min_v=2, max_v=20)
        crossover_rate = require_float(params, "crossover_rate", default=0.8, min_v=0.0, max_v=1.0)
        mutation_rate = require_float(params, "mutation_rate", default=0.2, min_v=0.0, max_v=1.0)
        elitism = require_int(params, "elitism", default=2, min_v=0, max_v=50)
        checkpoint_every = require_int(params, "checkpoint_every", default=0, min_v=0, max_v=1_000_000)
        return {
            "population_size": population_size,
            "generations": generations,
            "program_len": program_len,
            "tournament_k": tournament_k,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elitism": elitism,
            "checkpoint_every": checkpoint_every,
        }

    def _rand_prim(self, rng: random.Random) -> str:
        return rng.choice(self.PRIMS)

    def _rand_prog(self, rng: random.Random, L: int) -> List[str]:
        return [self._rand_prim(rng) for _ in range(L)]

    def _eval_prog(self, prog: List[str], z: List[float], bounds: List[Tuple[float, float]]) -> List[float]:
        import math
        x = z[:]
        for op in prog:
            if op == "add":
                x = [xi + zi for xi, zi in zip(x, z)]
            elif op == "sub":
                x = [xi - zi for xi, zi in zip(x, z)]
            elif op == "mul":
                x = [xi * (0.5 + abs(zi)) for xi, zi in zip(x, z)]
            elif op == "sin":
                x = [math.sin(xi) for xi in x]
            elif op == "cos":
                x = [math.cos(xi) for xi in x]
            elif op == "tanh":
                x = [math.tanh(xi) for xi in x]
            elif op == "clip":
                x = [max(lo, min(hi, xi)) for xi, (lo, hi) in zip(x, bounds)]
            else:
                pass
        # final clip to bounds
        return [max(lo, min(hi, xi)) for xi, (lo, hi) in zip(x, bounds)]

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

        def rand_latent() -> List[float]:
            return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

        def fitness_of(prog: List[str]) -> float:
            z = rand_latent()
            x = self._eval_prog(prog, z, bounds)
            return float(fn(x))

        def tournament(pop: List[List[str]], fits: List[float]) -> List[str]:
            best = None
            best_f = float("inf")
            for _ in range(p["tournament_k"]):
                i = rng.randrange(len(pop))
                if fits[i] < best_f:
                    best_f = fits[i]
                    best = pop[i]
            return list(best)

        def crossover(a: List[str], b: List[str]) -> Tuple[List[str], List[str]]:
            if len(a) < 2 or len(b) < 2:
                return a[:], b[:]
            i = rng.randrange(1, len(a))
            j = rng.randrange(1, len(b))
            return a[:i] + b[j:], b[:j] + a[i:]

        def mutate(prog: List[str]) -> List[str]:
            prog = prog[:]
            for i in range(len(prog)):
                if rng.random() < p["mutation_rate"]:
                    prog[i] = self._rand_prim(rng)
            # occasional insertion/deletion
            if rng.random() < 0.1 and len(prog) < 2 * p["program_len"]:
                prog.insert(rng.randrange(len(prog)+1), self._rand_prim(rng))
            if rng.random() < 0.1 and len(prog) > 3:
                prog.pop(rng.randrange(len(prog)))
            return prog

        pop = [self._rand_prog(rng, p["program_len"]) for _ in range(p["population_size"])]
        fits = [fitness_of(ind) for ind in pop]

        best_idx = int(min(range(len(pop)), key=lambda i: fits[i]))
        best_prog = pop[best_idx][:]
        best_fit = float(fits[best_idx])

        stopper = EarlyStopper(ctx.early_stopping)
        start = time.time()

        for gen in range(p["generations"]):
            if ctx.time_limit_sec and (time.time() - start) > ctx.time_limit_sec:
                break

            # sort by fitness
            order = sorted(range(len(pop)), key=lambda i: fits[i])
            new_pop: List[List[str]] = [pop[i][:] for i in order[: p["elitism"]]]

            while len(new_pop) < p["population_size"]:
                pa = tournament(pop, fits)
                pb = tournament(pop, fits)
                if rng.random() < p["crossover_rate"]:
                    ca, cb = crossover(pa, pb)
                else:
                    ca, cb = pa, pb
                ca = mutate(ca)
                if len(new_pop) < p["population_size"]:
                    new_pop.append(ca)
                if len(new_pop) < p["population_size"]:
                    new_pop.append(mutate(cb))

            pop = new_pop
            fits = [fitness_of(ind) for ind in pop]

            idx = int(min(range(len(pop)), key=lambda i: fits[i]))
            if fits[idx] < best_fit:
                best_fit = float(fits[idx])
                best_prog = pop[idx][:]

            gap = None
            if optimum is not None:
                try:
                    gap = float(best_fit) - float(optimum)
                except Exception:
                    gap = None

            progress_cb(ProgressEvent(iteration=gen, best_fitness=best_fit, metrics={"gap": gap} if gap is not None else {}, payload={"best_program_len": len(best_prog)}))
            if stopper.step(best_fit):
                break

        # produce final candidate
        z = rand_latent()
        best_x = self._eval_prog(best_prog, z, bounds)
        runtime = time.time() - start

        meta = RunMetadata(
            run_id=ctx.run_dir.name,
            timestamp_utc=problem.get("timestamp_utc", ""),
            seed=ctx.seed,
            problem_name=problem.get("name", "unknown_problem"),
            method_name=self.name,
        )

        metrics = {
            "best_fitness": float(best_fit),
            **({"gap": float(best_fit - float(optimum))} if optimum is not None else {}),
            "time_sec": float(runtime),
            "n_iterations": float(gen + 1),
        }

        artifacts = {
            "x": best_x,
            "program": best_prog,
        }

        return RunResult(metadata=meta, metrics=metrics, artifacts=artifacts)
