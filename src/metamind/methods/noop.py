from __future__ import annotations

import logging
from typing import Any, Dict

from metamind.types import RunResult, RunMetadata
from .base import RunContext
from .progress import ProgressEvent, ProgressCallback, default_progress_callback

log = logging.getLogger(__name__)


class NoOpMethod:
    name = "noop"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # no parameters needed
        return {}

    def solve(
        self,
        problem: Dict[str, Any],
        params: Dict[str, Any],
        ctx: RunContext,
        progress_cb: ProgressCallback = default_progress_callback,
    ) -> RunResult:
        log.info("NoOpMethod running. problem_keys=%s", list(problem.keys()))
        progress_cb(ProgressEvent(iteration=0, best_fitness=0.0, metrics={"status_ok": 1.0}, payload={}))

        meta = RunMetadata(
            run_id=ctx.run_dir.name,
            timestamp_utc=problem.get("timestamp_utc", ""),
            seed=ctx.seed,
            problem_name=problem.get("name", "unknown_problem"),
            method_name=self.name,
        )
        return RunResult(
            metadata=meta,
            metrics={"status_ok": 1.0},
            artifacts={"message": "NoOp executed successfully."},
            notes="noop",
        )