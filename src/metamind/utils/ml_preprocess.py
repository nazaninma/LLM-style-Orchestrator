from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import csv
import math
import numpy as np


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def _read_csv(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
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


def _standardize_train_apply(X_train: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns X_train_scaled, X_other_scaled, (mu, sigma)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_other - mu) / sigma, np.stack([mu, sigma], axis=0)


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


def build_titanic_features(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Builds Titanic features from Kaggle train.csv or test.csv.
    If Survived exists => y returned, else y=None.
    Features:
      - Pclass (1-3)
      - Sex (male=0, female=1)
      - Age (median-imputed)
      - SibSp
      - Parch
      - Fare (median-imputed)
      - Embarked (one-hot: C,Q,S)
    """
    # collect for medians
    ages = []
    fares = []
    for r in rows:
        a = r.get("Age", "").strip()
        if a:
            try:
                ages.append(float(a))
            except ValueError:
                pass
        f = r.get("Fare", "").strip()
        if f:
            try:
                fares.append(float(f))
            except ValueError:
                pass

    age_med = _median(ages)
    fare_med = _median(fares)

    feat_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]

    X = []
    y = []  # may remain empty for test.csv

    for r in rows:
        # Pclass
        pclass = float(r.get("Pclass", "0") or 0)

        # Sex
        sex_raw = (r.get("Sex", "") or "").strip().lower()
        sex = 1.0 if sex_raw == "female" else 0.0  # male/unknown -> 0

        # Age
        age_raw = (r.get("Age", "") or "").strip()
        if age_raw:
            try:
                age = float(age_raw)
            except ValueError:
                age = age_med
        else:
            age = age_med

        # SibSp, Parch
        sibsp = float(r.get("SibSp", "0") or 0)
        parch = float(r.get("Parch", "0") or 0)

        # Fare
        fare_raw = (r.get("Fare", "") or "").strip()
        if fare_raw:
            try:
                fare = float(fare_raw)
            except ValueError:
                fare = fare_med
        else:
            fare = fare_med

        # Embarked one-hot
        emb = (r.get("Embarked", "") or "").strip().upper()
        emb_c = 1.0 if emb == "C" else 0.0
        emb_q = 1.0 if emb == "Q" else 0.0
        emb_s = 1.0 if emb == "S" else 0.0
        # if missing, leave all zeros (or you can set S=1 as common default)

        X.append([pclass, sex, age, sibsp, parch, fare, emb_c, emb_q, emb_s])

        if "Survived" in r and r["Survived"] != "":
            y.append(int(float(r["Survived"])))

    X = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int) if len(y) == len(rows) else None
    return X, y_arr, feat_names


def prepare_titanic_problem(
    train_csv: str | Path,
    split=(0.7, 0.15, 0.15),
    seed: int = 42,
    standardize: bool = True,
) -> SplitData:
    rows = _read_csv(train_csv)
    X, y, feat_names = build_titanic_features(rows)
    if y is None:
        raise ValueError("Provided train_csv has no Survived column/labels.")

    tr, va, te = split_indices(len(X), ratios=split, seed=seed)

    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    X_test, y_test = X[te], y[te]

    if standardize:
        X_train, X_val, _ = _standardize_train_apply(X_train, X_val)
        X_train, X_test, _ = _standardize_train_apply(X_train, X_test)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_names,
    )