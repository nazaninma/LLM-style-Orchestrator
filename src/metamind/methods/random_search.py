from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Tuple

from metamind.types import RunResult, RunMetadata
from .base import RunContext, require_int, require_float, save_checkpoint, load_checkpoint
from .progress import ProgressEvent, ProgressCallback, default_progress_callback, EarlyStopper

log = logging.getLogger(__name__)


class RandomSearchMethod:
    """
    A simple baseline optimizer to test:
    - parameter validation
    - progress callbacks
    - logging
    - early stopping
    - checkpoint save/resume

    Expects problem dict with:
      - name: str
      - objective: {"type": "function", "fn": callable, "bounds": List[Tuple[float,float]], "optimum": float (optional)}
    """
    name = "random_search"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        n_samples = require_int(params, "n_samples", default=2000, min_v=10, max_v=2_000_000)
        checkpoint_every = require_int(params, "checkpoint_every", default=200, min_v=0, max_v=1_000_000)
        return {"n_samples": n_samples, "checkpoint_every": checkpoint_every}

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

        rng = random.Random(ctx.seed)
        start = time.time()

        ckpt_path = ctx.checkpoint_path("random_search_ckpt.json")
        ckpt = load_checkpoint(ckpt_path)

        i0 = 0
        best_x = None
        best_f = float("inf") if ctx.minimize else float("-inf")
        history: List[float] = []

        if ckpt:
            i0 = int(ckpt.get("i", 0))
            best_f = float(ckpt.get("best_f", best_f))
            best_x = ckpt.get("best_x", best_x)
            history = list(ckpt.get("history", []))
            log.info("Resuming from checkpoint: i=%s best_f=%s", i0, best_f)

        early = EarlyStopper(ctx.early_stopping)

        def is_better(a: float, b: float) -> bool:
            return a < b if ctx.minimize else a > b

        for i in range(i0, p["n_samples"]):
            # time limit (optional)
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("Time limit reached. Stopping at i=%s", i)
                break

            x = [rng.uniform(lo, hi) for (lo, hi) in bounds]
            f = float(fn(x))

            if best_x is None or is_better(f, best_f):
                best_f = f
                best_x = x

            history.append(best_f)

            # progress callback
            if (i % max(1, p["n_samples"] // 50)) == 0 or i == p["n_samples"] - 1:
                gap = None if optimum is None else abs(best_f - float(optimum))
                progress_cb(
                    ProgressEvent(
                        iteration=i,
                        best_fitness=best_f,
                        metrics={"best_fitness": best_f, **({ "gap": float(gap)} if gap is not None else {})},
                        payload={"best_x": best_x},
                    )
                )

            # early stopping
            if early.step(best_f):
                log.info("Early stopping triggered at i=%s best_f=%s", i, best_f)
                break

            # checkpointing
            if p["checkpoint_every"] > 0 and i > 0 and (i % p["checkpoint_every"] == 0):
                save_checkpoint(
                    ckpt_path,
                    {"i": i, "best_f": best_f, "best_x": best_x, "history": history[-2000:]},
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
            metrics["error_to_optimum"] = float(abs(best_f - float(optimum)))

        return RunResult(
            metadata=meta,
            metrics=metrics,
            artifacts={
                "best_x": best_x,
                "convergence_history": history,
                "checkpoint_path": str(ckpt_path),
            },
            notes="random_search baseline",
        )