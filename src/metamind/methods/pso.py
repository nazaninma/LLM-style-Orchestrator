from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Tuple

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_bool, require_float, require_int, save_checkpoint, load_checkpoint
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper

log = logging.getLogger(__name__)


class PSOMethod:
    """
    Particle Swarm Optimization for continuous function minimization.

    Expects problem dict with:
      - name: str
      - objective: {"fn": callable, "bounds": List[(lo,hi)], "optimum": float (optional)}
    """

    name = "pso"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        n_particles = require_int(params, "n_particles", default=50, min_v=5, max_v=5000)
        max_iterations = require_int(params, "max_iterations", default=500, min_v=10, max_v=1_000_000)

        w = require_float(params, "w", default=0.7, min_v=0.0, max_v=1.2)
        c1 = require_float(params, "c1", default=1.5, min_v=0.0, max_v=4.0)
        c2 = require_float(params, "c2", default=1.5, min_v=0.0, max_v=4.0)

        w_decay = require_bool(params, "w_decay", default=True)
        velocity_clamp = require_float(params, "velocity_clamp", default=0.5, min_v=0.0, max_v=10.0)

        checkpoint_every = require_int(params, "checkpoint_every", default=0, min_v=0, max_v=1_000_000)

        return {
            "n_particles": n_particles,
            "max_iterations": max_iterations,
            "w": w,
            "c1": c1,
            "c2": c2,
            "w_decay": w_decay,
            "velocity_clamp": velocity_clamp,
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

        def rand_vec() -> List[float]:
            return [rng.uniform(lo, hi) for (lo, hi) in bounds]

        def zero_vec() -> List[float]:
            return [0.0] * dim

        def eval_f(x: List[float]) -> float:
            return float(fn(x))

        def is_better(a: float, b: float) -> bool:
            return a < b if ctx.minimize else a > b

        # checkpointing support
        ckpt_path = ctx.checkpoint_path("pso_ckpt.json")
        ckpt = load_checkpoint(ckpt_path)

        start_iter = 0
        start = time.time()

        # Initialize swarm
        if ckpt:
            log.info("Resuming PSO from checkpoint: %s", ckpt_path)
            start_iter = int(ckpt.get("iter", 0))
            positions = ckpt["positions"]
            velocities = ckpt["velocities"]
            pbest_pos = ckpt["pbest_pos"]
            pbest_fit = ckpt["pbest_fit"]
            gbest_pos = ckpt["gbest_pos"]
            gbest_fit = float(ckpt["gbest_fit"])
            history: List[float] = list(ckpt.get("history", []))
        else:
            positions = [rand_vec() for _ in range(p["n_particles"])]
            velocities = [zero_vec() for _ in range(p["n_particles"])]

            pbest_pos = [pos[:] for pos in positions]
            pbest_fit = [eval_f(pos) for pos in positions]

            # global best
            gbest_idx = min(range(len(pbest_fit)), key=lambda i: pbest_fit[i]) if ctx.minimize else max(
                range(len(pbest_fit)), key=lambda i: pbest_fit[i]
            )
            gbest_pos = pbest_pos[gbest_idx][:]
            gbest_fit = float(pbest_fit[gbest_idx])
            history = [gbest_fit]

        early = EarlyStopper(ctx.early_stopping)

        # main loop
        for it in range(start_iter, p["max_iterations"]):
            # time limit
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("PSO time limit reached at iter=%s", it)
                break

            # optionally decay inertia weight
            if p["w_decay"] and p["max_iterations"] > 1:
                w = p["w"] * (1.0 - it / (p["max_iterations"] - 1))
            else:
                w = p["w"]

            for i in range(p["n_particles"]):
                r1 = rng.random()
                r2 = rng.random()

                # update velocity and position per dimension
                for d in range(dim):
                    vel = (
                        w * velocities[i][d]
                        + p["c1"] * r1 * (pbest_pos[i][d] - positions[i][d])
                        + p["c2"] * r2 * (gbest_pos[d] - positions[i][d])
                    )

                    # clamp velocity
                    vc = p["velocity_clamp"]
                    if vc > 0:
                        vel = clip(vel, -vc, vc)

                    velocities[i][d] = vel
                    positions[i][d] = clip(positions[i][d] + vel, bounds[d][0], bounds[d][1])

                fit = eval_f(positions[i])

                if is_better(fit, pbest_fit[i]):
                    pbest_fit[i] = fit
                    pbest_pos[i] = positions[i][:]

                if is_better(fit, gbest_fit):
                    gbest_fit = fit
                    gbest_pos = positions[i][:]

            history.append(float(gbest_fit))

            # progress callback occasionally
            if it == 0 or it % max(1, p["max_iterations"] // 50) == 0 or it == p["max_iterations"] - 1:
                gap = None if optimum is None else abs(float(gbest_fit) - float(optimum))
                progress_cb(
                    ProgressEvent(
                        iteration=it,
                        best_fitness=float(gbest_fit),
                        metrics={
                            "best_fitness": float(gbest_fit),
                            **({"gap": float(gap)} if gap is not None else {}),
                        },
                        payload={"best_x": gbest_pos},
                    )
                )

            # early stopping
            if early.step(float(gbest_fit)):
                log.info("PSO early stopping at iter=%s best=%s", it, gbest_fit)
                break

            # checkpoint
            if p["checkpoint_every"] > 0 and it > 0 and (it % p["checkpoint_every"] == 0):
                save_checkpoint(
                    ckpt_path,
                    {
                        "iter": it,
                        "positions": positions,
                        "velocities": velocities,
                        "pbest_pos": pbest_pos,
                        "pbest_fit": pbest_fit,
                        "gbest_pos": gbest_pos,
                        "gbest_fit": float(gbest_fit),
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
            "best_fitness": float(gbest_fit),
            "time_sec": float(elapsed),
            "iterations_completed": float(len(history)),
        }
        if optimum is not None:
            metrics["error_to_optimum"] = float(abs(float(gbest_fit) - float(optimum)))

        return RunResult(
            metadata=meta,
            metrics=metrics,
            artifacts={
                "best_x": gbest_pos,
                "convergence_history": history,
                "checkpoint_path": str(ckpt_path),
            },
            notes="pso continuous optimizer",
        )