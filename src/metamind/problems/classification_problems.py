from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _resolve_path(root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _read_csv(root: Path, path: str | Path) -> List[Dict[str, str]]:
    path = _resolve_path(root, path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return float((xs[mid - 1] + xs[mid]) / 2.0)


def _standardize_train_apply(X_train: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_other - mu) / sigma, np.stack([mu, sigma], axis=0)


def split_indices(n: int, ratios=(0.7, 0.15, 0.15), seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sum(ratios) <= 0.999 or sum(ratios) >= 1.001:
        raise ValueError("ratios must sum to 1.0 (e.g., 0.7,0.15,0.15)")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def build_titanic_features(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ages, fares = [], []
    for r in rows:
        a = (r.get("Age", "") or "").strip()
        if a:
            try:
                ages.append(float(a))
            except ValueError:
                pass
        f = (r.get("Fare", "") or "").strip()
        if f:
            try:
                fares.append(float(f))
            except ValueError:
                pass

    age_med = _median(ages)
    fare_med = _median(fares)

    feat_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]
    X, y = [], []

    for r in rows:
        pclass = float(r.get("Pclass", "0") or 0)

        sex_raw = (r.get("Sex", "") or "").strip().lower()
        sex = 1.0 if sex_raw == "female" else 0.0

        age_raw = (r.get("Age", "") or "").strip()
        if age_raw:
            try:
                age = float(age_raw)
            except ValueError:
                age = age_med
        else:
            age = age_med

        sibsp = float(r.get("SibSp", "0") or 0)
        parch = float(r.get("Parch", "0") or 0)

        fare_raw = (r.get("Fare", "") or "").strip()
        if fare_raw:
            try:
                fare = float(fare_raw)
            except ValueError:
                fare = fare_med
        else:
            fare = fare_med

        emb = (r.get("Embarked", "") or "").strip().upper()
        emb_c = 1.0 if emb == "C" else 0.0
        emb_q = 1.0 if emb == "Q" else 0.0
        emb_s = 1.0 if emb == "S" else 0.0

        X.append([pclass, sex, age, sibsp, parch, fare, emb_c, emb_q, emb_s])

        if "Survived" not in r or r["Survived"] == "":
            raise ValueError("Titanic train.csv must contain Survived column.")
        y.append(int(float(r["Survived"])))

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), feat_names


class ClassificationProblem:
    """
    Phase 5.6 - Classification Problem Builder
    Output (split-aware):
      problem_type=classification
      meta: dataset, paths, n_features, split, standardized, ...
      ml: X_train/y_train/X_val/y_val/X_test/y_test + standardized flag
    """

    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle

    def build(self) -> Dict[str, Any]:
        root = Path(str(self.bundle.get("_root", Path.cwd())))
        meta_cfg = self.bundle.get("problem_meta", {}) or {}
        seed = int(self.bundle.get("_seed", 42))
        default_name = str(self.bundle.get("_default_problem", "classification"))

        dataset = str(meta_cfg.get("dataset", default_name))
        train_csv = meta_cfg.get("train_csv", f"data/{dataset}/train.csv")
        split = meta_cfg.get("split", [0.7, 0.15, 0.15])
        standardize = bool(meta_cfg.get("standardize", True))

        rows = _read_csv(root, train_csv)

        # فعلاً titanıc را به صورت رسمی ساپورت می‌کنیم (مثل پروژه‌ات)
        # اگر خواستی بعداً dataset های دیگر را هم اضافه می‌کنیم.
        if dataset.lower() != "titanic":
            raise ValueError(f"Unsupported classification dataset={dataset}. Currently supported: titanic")

        X, y, feature_names = build_titanic_features(rows)

        tr, va, te = split_indices(len(X), ratios=tuple(split), seed=seed)
        X_train, y_train = X[tr], y[tr]
        X_val, y_val = X[va], y[va]
        X_test, y_test = X[te], y[te]

        standardized_flag = False
        if standardize:
            # fit on train, apply to val/test
            X_train_std, scaler = _standardize_train_apply(X_train, X_train)
            X_val_std, _ = _standardize_train_apply(X_train, X_val)
            X_test_std, _ = _standardize_train_apply(X_train, X_test)
            X_train, X_val, X_test = X_train_std, X_val_std, X_test_std
            standardized_flag = True

        meta = {
            "dataset": dataset,
            "source": "csv",
            "train_csv": str(_resolve_path(root, train_csv)),
            "split": list(split),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": list(feature_names),
            "standardized": standardized_flag,
            "label": "Survived",
            "n_classes": 2,
        }

        return {
            "name": dataset,
            "problem_type": "classification",
            "meta": meta,
            "ml": {
                "X": X,
                "y": y,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                "feature_names": list(feature_names),
                "standardized": standardized_flag,
            },
        }