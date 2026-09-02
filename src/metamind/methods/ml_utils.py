from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return {"mean": mean, "std": std}


def standardize_transform(X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return (X - scaler["mean"]) / scaler["std"]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    # assumes labels {0,1}
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(cm: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}