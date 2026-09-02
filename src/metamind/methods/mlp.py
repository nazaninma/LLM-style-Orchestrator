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


class MLPMethod:
    """
    Simple 1-hidden-layer MLP for binary classification (labels must be 0/1).
    Uses sigmoid output + binary cross-entropy.

    Supports BOTH problem formats:

    Old format:
      - problem["ml"]["X"], problem["ml"]["y"]

    New (Phase 5.4) format:
      - problem["ml"]["X_train"], ["y_train"], optional ["X_val"], ["y_val"], ["X_test"], ["y_test"]
    """
    name = "mlp"

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        epochs = require_int(params, "epochs", default=80, min_v=1, max_v=200000)
        lr = require_float(params, "lr", default=0.01, min_v=1e-6, max_v=10.0)
        hidden = require_int(params, "hidden_units", default=32, min_v=2, max_v=5000)
        batch_size = require_int(params, "batch_size", default=32, min_v=1, max_v=100000)
        test_size = require_float(params, "test_size", default=0.2, min_v=0.05, max_v=0.5)
        l2 = require_float(params, "l2", default=0.0, min_v=0.0, max_v=10.0)
        return {
            "epochs": epochs,
            "lr": lr,
            "hidden_units": hidden,
            "batch_size": batch_size,
            "test_size": test_size,
            "l2": l2,
        }

    def _load_data(
        self, ml: Dict[str, Any], seed: int, test_size: float
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Returns: X_train,y_train,X_val,y_val,X_test,y_test,is_split_provided
        """
        # New format
        if "X_train" in ml and "y_train" in ml:
            X_train = np.asarray(ml["X_train"], dtype=float)
            y_train = np.asarray(ml["y_train"], dtype=int).reshape(-1, 1)

            X_val = np.asarray(ml["X_val"], dtype=float) if "X_val" in ml else None
            y_val = np.asarray(ml["y_val"], dtype=int).reshape(-1, 1) if "y_val" in ml else None

            X_test = np.asarray(ml["X_test"], dtype=float) if "X_test" in ml else None
            y_test = np.asarray(ml["y_test"], dtype=int).reshape(-1, 1) if "y_test" in ml else None

            return X_train, y_train, X_val, y_val, X_test, y_test, True

        # Old format
        X = np.asarray(ml["X"], dtype=float)
        y = np.asarray(ml["y"], dtype=int).reshape(-1, 1)
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
            raise ValueError("MLP requires a test set. Provide X_test/y_test or use old format X/y.")
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D array")

        n, d = X_train.shape
        rng = np.random.default_rng(ctx.seed)

        # 3) Initialize network
        H = p["hidden_units"]
        W1 = rng.normal(0, 0.1, size=(d, H))
        b1 = np.zeros((1, H))
        W2 = rng.normal(0, 0.1, size=(H, 1))
        b2 = np.zeros((1, 1))

        def relu(z): return np.maximum(0, z)
        def relu_grad(z): return (z > 0).astype(float)
        def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

        def forward(Xb):
            z1 = Xb @ W1 + b1
            a1 = relu(z1)
            z2 = a1 @ W2 + b2
            a2 = sigmoid(z2)
            return z1, a1, a2

        def bce(y_true, y_prob):
            eps = 1e-12
            y_prob = np.clip(y_prob, eps, 1 - eps)
            return float(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean())

        def predict(Xb):
            _, _, yprob = forward(Xb)
            return (yprob >= 0.5).astype(int).reshape(-1)

        def predict_proba(Xb):
            _, _, yprob = forward(Xb)
            return yprob.reshape(-1)

        early = EarlyStopper(ctx.early_stopping)
        history_loss = []
        history_acc = []

        # 4) Training loop
        start = time.time()
        for ep in range(p["epochs"]):
            if ctx.time_limit_sec is not None and (time.time() - start) > ctx.time_limit_sec:
                log.warning("MLP time limit reached at epoch=%s", ep)
                break

            idx = np.arange(n)
            rng.shuffle(idx)

            for s in range(0, n, p["batch_size"]):
                batch = idx[s:s + p["batch_size"]]
                Xb = X_train[batch]
                yb = y_train[batch]

                z1, a1, yhat = forward(Xb)

                dz2 = (yhat - yb) / max(1, len(batch))
                dW2 = a1.T @ dz2 + p["l2"] * W2
                db2 = dz2.sum(axis=0, keepdims=True)

                da1 = dz2 @ W2.T
                dz1 = da1 * relu_grad(z1)
                dW1 = Xb.T @ dz1 + p["l2"] * W1
                db1 = dz1.sum(axis=0, keepdims=True)

                W2 -= p["lr"] * dW2
                b2 -= p["lr"] * db2
                W1 -= p["lr"] * dW1
                b1 -= p["lr"] * db1

            # Evaluate on TEST each epoch (for progress + early stop)
            _, _, yprob_test = forward(X_test)
            loss = bce(y_test, yprob_test)
            ypred_test = (yprob_test >= 0.5).astype(int).reshape(-1)
            ytrue_test = y_test.astype(int).reshape(-1)

            acc_test = accuracy(ytrue_test, ypred_test)
            history_loss.append(loss)
            history_acc.append(acc_test)

            if ep == 0 or ep % max(1, p["epochs"] // 20) == 0 or ep == p["epochs"] - 1:
                progress_cb(
                    ProgressEvent(
                        iteration=ep,
                        best_fitness=loss,
                        metrics={"loss": float(loss), "test_accuracy": float(acc_test)},
                        payload={},
                    )
                )

            # Early stopping on loss (minimize)
            if early.step(loss):
                log.info("MLP early stopping at epoch=%s loss=%s", ep, loss)
                break

        elapsed = time.time() - start

        # 5) Final metrics (train/val/test)
        def eval_split(Xs: np.ndarray, ys: np.ndarray, prefix: str) -> Dict[str, Any]:
            ypred = predict(Xs)
            ytrue = ys.astype(int).reshape(-1)
            acc = accuracy(ytrue, ypred)
            cm = confusion_matrix_binary(ytrue, ypred)
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
            "history_loss": history_loss,
            "history_accuracy": history_acc,
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
        artifacts["y_score"] = predict_proba(X_test).tolist()

        metrics.update(
            {
                "time_sec": float(elapsed),
                "iterations_completed": float(len(history_loss)),
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
            notes="mlp binary classifier",
        )