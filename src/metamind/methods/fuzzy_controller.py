from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Tuple

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_float, require_int
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper

log = logging.getLogger(__name__)


class FuzzyControllerMethod:
    """
    Simple fuzzy-controller driven stochastic hill-climber for continuous optimization.

    The controller adjusts the step size based on:
      - recent improvement (good / small / none)
      - progress through iterations (early / mid / late)

    This is intentionally lightweight but provides a distinct CI-style method
    aligned with the project requirement.

    Expects:
      - objective: {"fn": callable, "bounds": List[(lo,hi)], "optimum": float (optional)}
    """

    name = "fuzzy"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        max_iterations = require_int(params, "max_iterations", default=2000, min_v=50, max_v=500_000)
        step0 = require_float(params, "step0", default=0.25, min_v=1e-4, max_v=10.0)
        min_step = require_float(params, "min_step", default=1e-6, min_v=1e-12, max_v=1.0)
        max_step = require_float(params, "max_step", default=1.0, min_v=1e-3, max_v=100.0)
        restart_every = require_int(params, "restart_every", default=0, min_v=0, max_v=1_000_000)
        report_every = require_int(params, "report_every", default=50, min_v=1, max_v=1_000_000)
        return {
            "max_iterations": max_iterations,
            "step0": step0,
            "min_step": min_step,
            "max_step": max_step,
            "restart_every": restart_every,
            "report_every": report_every,
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

        def clip_vec(x: List[float]) -> List[float]:
            return [max(lo, min(hi, xi)) for xi, (lo, hi) in zip(x, bounds)]

        def rand_vec() -> List[float]:
            return [rng.uniform(lo, hi) for (lo, hi) in bounds]

        x = rand_vec()
        best = float(fn(x))
        best_x = x[:]

        step = float(p["step0"])
        stopper = EarlyStopper(ctx.early_stopping)
        start = time.time()

        # fuzzy membership helpers
        def tri(x: float, a: float, b: float, c: float) -> float:
            if x <= a or x >= c:
                return 0.0
            if x == b:
                return 1.0
            if x < b:
                return (x - a) / (b - a)
            return (c - x) / (c - b)

        def clamp(v: float, lo: float, hi: float) -> float:
            return lo if v < lo else hi if v > hi else v

        last_best = best

        for it in range(p["max_iterations"]):
            if ctx.time_limit_sec and (time.time() - start) > ctx.time_limit_sec:
                break

            # random perturbation with current step size
            proposal = [xi + step * rng.gauss(0.0, 1.0) for xi in x]
            proposal = clip_vec(proposal)
            f = float(fn(proposal))

            improved = (best - f)  # positive means improvement (minimization)
            if f < best:
                best = f
                best_x = proposal[:]
                x = proposal
            else:
                # occasionally accept to escape local minima (very mild SA)
                if rng.random() < 0.02:
                    x = proposal

            # fuzzy control: determine step multiplier
            # normalize improvement by |best|+1 to be scale robust
            norm_imp = improved / (abs(best) + 1.0)

            # membership for improvement: none / small / good
            m_none = tri(norm_imp, -1e-6, 0.0, 1e-6)
            m_small = tri(norm_imp, 0.0, 1e-4, 5e-4)
            m_good = tri(norm_imp, 1e-4, 1e-2, 1.0)

            t = it / max(1, p["max_iterations"] - 1)
            m_early = tri(t, 0.0, 0.0, 0.5)  # ramps down
            m_late = tri(t, 0.5, 1.0, 1.0)   # ramps up
            m_mid = tri(t, 0.25, 0.5, 0.75)

            # Rules (heuristic):
            # - early & none  -> increase step (explore)
            # - early & good  -> keep step
            # - mid & small   -> slightly decrease
            # - late & none   -> decrease (fine tune) but may trigger restart
            # - late & good   -> decrease a bit (exploit)
            inc = m_early * m_none
            keep = m_early * m_good + m_mid * m_good
            dec = m_mid * (m_small + m_none) + m_late * (m_small + m_good + m_none)

            # defuzzify to multiplier
            mult = (1.25 * inc + 1.0 * keep + 0.85 * dec) / (inc + keep + dec + 1e-12)
            step = clamp(step * mult, p["min_step"], p["max_step"])

            # optional restart if stuck late
            if p["restart_every"] and (it + 1) % p["restart_every"] == 0:
                x = rand_vec()

            gap = None
            if optimum is not None:
                try:
                    gap = float(best) - float(optimum)
                except Exception:
                    gap = None

            if it % p["report_every"] == 0 or it == p["max_iterations"] - 1:
                progress_cb(ProgressEvent(iteration=it, best_fitness=best, metrics=dict(({"gap": gap} if gap is not None else {}), **{"step": float(step)}), payload={"x": best_x}))

            if stopper.step(best):
                break

        runtime = time.time() - start
        meta = RunMetadata(
            run_id=ctx.run_dir.name,
            timestamp_utc=problem.get("timestamp_utc", ""),
            seed=ctx.seed,
            problem_name=problem.get("name", "unknown_problem"),
            method_name=self.name,
        )

        metrics = {
            "best_fitness": float(best),
            **({"gap": float(best - float(optimum))} if optimum is not None else {}),
            "time_sec": float(runtime),
            "n_iterations": float(it + 1),
            "step_final": float(step),
        }

        artifacts = {
            "x": best_x,
        }

        return RunResult(metadata=meta, metrics=metrics, artifacts=artifacts)
