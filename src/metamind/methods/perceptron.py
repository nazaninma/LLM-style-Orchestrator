from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from metamind.types import RunMetadata, RunResult
from .base import RunContext, require_float, require_int
from .progress import ProgressCallback, ProgressEvent, default_progress_callback, EarlyStopper
from .ml_utils import (
    train_test_split,
    standardize_fit,
    standardize_transform,
    accuracy,
    confusion_matrix_binary,
    precision_recall_f1,
)

log = logging.getLogger(__name__)


class PerceptronMethod:
    """
    Classic perceptron for binary classification (labels must be 0/1).

    Supports BOTH problem formats:

    Old format:
      - problem["ml"]["X"], problem["ml"]["y"]

    New (Phase 5.4) format:
      - problem["ml"]["X_train"], ["y_train"], optional ["X_val"], ["y_val"], ["X_test"], ["y_test"]
    """
    name = "perceptron"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        epochs = require_int(params, "epochs", default=30, min_v=1, max_v=100000)
        lr = require_float(params, "lr", default=0.1, min_v=1e-6, max_v=10.0)
        test_size = require_float(params, "test_size", default=0.2, min_v=0.05, max_v=0.5)
        return {"epochs": epochs, "lr": lr, "test_size": test_size}

    def _load_data(
        self, ml: Dict[str, Any], seed: int, test_size: float
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Returns: X_train,y_train,X_val,y_val,X_test,y_test,is_split_provided
        """
        # New format
        if "X_train" in ml and "y_train" in ml:
            X_train = np.asarray(ml["X_train"], dtype=float)
            y_train = np.asarray(ml["y_train"], dtype=int)

            X_val = np.asarray(ml["X_val"], dtype=float) if "X_val" in ml else None
            y_val = np.asarray(ml["y_val"], dtype=int) if "y_val" in ml else None

            X_test = np.asarray(ml["X_test"], dtype=float) if "X_test" in ml else None
            y_test = np.asarray(ml["y_test"], dtype=int) if "y_test" in ml else None

            return X_train, y_train, X_val, y_val, X_test, y_test, True

        # Old format
        X = np.asarray(ml["X"], dtype=float)
        y = np.asarray(ml["y"], dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, seed=seed)
        return X_train, y_train, None, None, X_test, y_test, False

    def _maybe_standardize(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray],
        X_test: Optional[np.ndarray],
        ml: Dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Standardize unless the problem explicitly indicates it's already standardized.
        We support a flag in ml: ml["standardized"] = True/False.
        If you didn't set it, default is False => we standardize here.
        """
        already = bool(ml.get("standardized", False))
        if already:
            return X_train, X_val, X_test

        scaler = standardize_fit(X_train)
        X_train_s = standardize_transform(X_train, scaler)
        X_val_s = standardize_transform(X_val, scaler) if X_val is not None else None
        X_test_s = standardize_transform(X_test, scaler) if X_test is not None else None
        return X_train_s, X_val_s, X_test_s

    def solve(
        self,
        problem: Dict[str, Any],
        params: Dict[str, Any],
        ctx: RunContext,
        progress_cb: ProgressCallback = default_progress_callback,
    ) -> RunResult:
        p = self.validate_params(params)
        ml = problem["ml"]

        # 1) Load data (split-aware)
        X_train, y_train, X_val, y_val, X_test, y_test, split_provided = self._load_data(
            ml=ml, seed=ctx.seed, test_size=p["test_size"]
        )

        # 2) Standardize (only if not already standardized by the problem builder)
        X_train, X_val, X_test = self._maybe_standardize(X_train, X_val, X_test, ml)

        # Basic checks
        if X_test is None or y_test is None:
            raise ValueError("Perceptron requires a test set. Provide X_test/y_test or use old format X/y.")
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D array")

        n, d = X_train.shape
        rng = np.random.default_rng(ctx.seed)

        w = rng.normal(0, 0.01, size=d)
        b = 0.0

        def score(Xb: np.ndarray) -> np.ndarray:
            return (Xb @ w + b).reshape(-1)

        def predict(Xb: np.ndarray) -> np.ndarray:
            scores = score(Xb)
            return (scores >= 0).astype(int)

        early = EarlyStopper(ctx.early_stopping)
        history_acc_test = []

        start = time.time()
        for ep in range(p["epochs"]):
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("Perceptron time limit reached at epoch=%s", ep)
                break

            idx = np.arange(n)
            rng.shuffle(idx)

            # SGD over samples
            for i in idx:
                xi = X_train[i]
                yi = y_train[i]
                yhat = 1 if (xi @ w + b) >= 0 else 0
                err = yi - yhat
                if err != 0:
                    w += p["lr"] * err * xi
                    b += p["lr"] * err

            # evaluate on TEST for progress + early stop
            y_pred_test = predict(X_test)
            acc_test = accuracy(y_test, y_pred_test)
            history_acc_test.append(float(acc_test))

            if ep == 0 or ep % max(1, p["epochs"] // 20) == 0 or ep == p["epochs"] - 1:
                progress_cb(
                    ProgressEvent(
                        iteration=ep,
                        best_fitness=1.0 - float(acc_test),  # minimize error
                        metrics={"test_accuracy": float(acc_test), "test_error": float(1.0 - float(acc_test))},
                        payload={},
                    )
                )

            # early stop on error (minimize)
            if early.step(1.0 - float(acc_test)):
                log.info("Perceptron early stopping at epoch=%s test_acc=%s", ep, acc_test)
                break

        elapsed = time.time() - start

        # 3) Final metrics (train/val/test)
        def eval_split(Xs: np.ndarray, ys: np.ndarray, prefix: str) -> Dict[str, Any]:
            ypred = predict(Xs)
            acc = accuracy(ys, ypred)
            cm = confusion_matrix_binary(ys, ypred)
            prf = precision_recall_f1(cm)
            return {
                f"{prefix}_accuracy": float(acc),
                f"{prefix}_precision": float(prf["precision"]),
                f"{prefix}_recall": float(prf["recall"]),
                f"{prefix}_f1": float(prf["f1"]),
                f"{prefix}_confusion": cm,
            }

        metrics: Dict[str, Any] = {}
        artifacts: Dict[str, Any] = {
            "weights": w.tolist(),
            "bias": float(b),
            "history_test_accuracy": history_acc_test,
        }

        # Train
        metrics.update(eval_split(X_train, y_train, "train"))

        # Val (optional)
        if X_val is not None and y_val is not None and len(y_val) > 0:
            metrics.update(eval_split(X_val, y_val, "val"))

        # Test (required)
        metrics.update(eval_split(X_test, y_test, "test"))
        artifacts["confusion_matrix"] = metrics["test_confusion"]
        # For AUC-ROC
        artifacts["y_true"] = y_test.astype(int).reshape(-1).tolist()
        artifacts["y_score"] = score(X_test).tolist()

        metrics.update(
            {
                "time_sec": float(elapsed),
                "iterations_completed": float(len(history_acc_test)),
            }
        )

        meta = RunMetadata(
            run_id=ctx.run_dir.name,
            timestamp_utc=problem.get("timestamp_utc", ""),
            seed=ctx.seed,
            problem_name=problem.get("name", "unknown_problem"),
            method_name=self.name,
        )

        return RunResult(
            metadata=meta,
            metrics=metrics,
            artifacts=artifacts,
            notes="perceptron binary classifier",
        )