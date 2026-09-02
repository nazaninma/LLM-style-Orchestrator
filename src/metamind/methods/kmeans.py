from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_int, require_float
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper

log = logging.getLogger(__name__)


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    # ساده و قابل قبول برای دیتاست‌های کوچک (مثل iris)
    n = len(X)
    if n <= 1:
        return 0.0
    unique = np.unique(labels)
    if len(unique) <= 1:
        return 0.0

    # precompute distances
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    s_all = []
    for i in range(n):
        same = labels == labels[i]
        other_clusters = [c for c in unique if c != labels[i]]

        # a(i): mean intra-cluster distance
        if same.sum() > 1:
            a = D[i, same].sum() / (same.sum() - 1)
        else:
            a = 0.0

        # b(i): min mean distance to other cluster
        b = float("inf")
        for c in other_clusters:
            mask = labels == c
            if mask.sum() == 0:
                continue
            b = min(b, D[i, mask].mean())

        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        s_all.append(s)

    return float(np.mean(s_all))


class KMeansMethod:
    name = "kmeans"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        k = require_int(params, "k", default=3, min_v=2, max_v=1000)
        max_iter = require_int(params, "max_iter", default=200, min_v=1, max_v=100000)
        n_init = require_int(params, "n_init", default=5, min_v=1, max_v=1000)
        tol = require_float(params, "tol", default=1e-4, min_v=0.0, max_v=1.0)
        return {"k": k, "max_iter": max_iter, "n_init": n_init, "tol": tol}

    def solve(
        self,
        problem: Dict[str, Any],
        params: Dict[str, Any],
        ctx: RunContext,
        progress_cb: ProgressCallback = default_progress_callback,
    ) -> RunResult:
        p = self.validate_params(params)
        ml = problem["ml"]
        X = np.asarray(ml["X"], dtype=float)

        rng = np.random.default_rng(ctx.seed)
        early = EarlyStopper(ctx.early_stopping)

        best_inertia = float("inf")
        best_labels = None
        best_centroids = None
        best_sil = None

        start = time.time()

        def run_once() -> tuple[float, np.ndarray, np.ndarray, float]:
            n, d = X.shape
            # init centroids with random points
            centroids = X[rng.choice(n, size=p["k"], replace=False)]

            labels = np.zeros(n, dtype=int)
            prev_inertia = None

            for it in range(p["max_iter"]):
                # assign
                dist2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
                labels = dist2.argmin(axis=1)

                # update
                new_centroids = np.zeros_like(centroids)
                for j in range(p["k"]):
                    mask = labels == j
                    if mask.any():
                        new_centroids[j] = X[mask].mean(axis=0)
                    else:
                        # empty cluster -> reinit
                        new_centroids[j] = X[rng.integers(0, n)]
                centroids = new_centroids

                # inertia
                inertia = float(((X - centroids[labels]) ** 2).sum())

                if prev_inertia is not None and abs(prev_inertia - inertia) < p["tol"]:
                    break
                prev_inertia = inertia

            sil = _silhouette_score(X, labels)
            return inertia, labels, centroids, sil

        for init in range(p["n_init"]):
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("KMeans time limit reached at init=%s", init)
                break

            inertia, labels, centroids, sil = run_once()

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids
                best_sil = sil

            progress_cb(
                ProgressEvent(
                    iteration=init,
                    best_fitness=best_inertia,  # minimize inertia
                    metrics={"best_inertia": float(best_inertia), "silhouette": float(best_sil)},
                    payload={},
                )
            )

            if early.step(best_inertia):
                log.info("KMeans early stopping at init=%s inertia=%s", init, best_inertia)
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
            "inertia": float(best_inertia),
            "silhouette": float(best_sil if best_sil is not None else 0.0),
            "time_sec": float(elapsed),
            "iterations_completed": float(p["n_init"]),
        }

        return RunResult(
            metadata=meta,
            metrics=metrics,
            artifacts={
                "labels": best_labels.tolist() if best_labels is not None else [],
                "centroids": best_centroids.tolist() if best_centroids is not None else [],
            },
            notes="kmeans clustering",
        )