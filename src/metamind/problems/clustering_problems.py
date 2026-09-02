from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _resolve_path(root: Path, p: str | Path) -> Path:
    """Resolve file paths relative to the project root."""
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


class ClusteringProblem:
    """
    Phase 5.6 - Clustering Problem Builder
    """

    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle

    def build(self) -> Dict[str, Any]:
        root = Path(str(self.bundle.get("_root", Path.cwd())))
        meta_cfg = self.bundle.get("problem_meta", {}) or {}

        source = str(meta_cfg.get("source", "csv")).lower()

        default_name = str(self.bundle.get("_default_problem", "iris"))
        dataset = str(meta_cfg.get("dataset", default_name))

        csv_path = meta_cfg.get("csv", f"data/{dataset}/{dataset}.csv")
        feature_cols = meta_cfg.get(
            "feature_cols",
            ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )
        label_col = meta_cfg.get("label_col", "species")  # optional
        standardize = bool(meta_cfg.get("standardize", True))

        if source == "generated":
            # synthetic blobs (like sklearn.datasets.make_blobs)
            n_samples = int(meta_cfg.get("n_samples", 300))
            n_features = int(meta_cfg.get("n_features", 2))
            centers = int(meta_cfg.get("centers", meta_cfg.get("k", 4)))
            cluster_std = float(meta_cfg.get("cluster_std", 1.0))
            seed = int(self.bundle.get("_seed", 42))
            try:
                from sklearn.datasets import make_blobs

                X_arr, y_true = make_blobs(
                    n_samples=n_samples,
                    n_features=n_features,
                    centers=centers,
                    cluster_std=cluster_std,
                    random_state=seed,
                )
            except Exception:
                rng = np.random.default_rng(seed)
                X_arr = rng.standard_normal((n_samples, n_features))
                y_true = None

            if standardize:
                mu = X_arr.mean(axis=0)
                sigma = X_arr.std(axis=0)
                sigma = np.where(sigma == 0, 1.0, sigma)
                X_arr = (X_arr - mu) / sigma

            meta = {
                "dataset": dataset,
                "source": "generated",
                "n_samples": int(X_arr.shape[0]),
                "n_features": int(X_arr.shape[1]),
                "centers": centers,
                "cluster_std": cluster_std,
                "standardized": standardize,
            }
            return {
                "name": dataset,
                "problem_type": "clustering",
                "meta": meta,
                "ml": {
                    "X": np.asarray(X_arr, dtype=float),
                    "y": np.asarray(y_true, dtype=int) if y_true is not None else None,
                    "feature_cols": [f"x{i}" for i in range(int(X_arr.shape[1]))],
                    "label_col": None,
                    "standardized": standardize,
                },
            }

        rows = _read_csv(root, csv_path)

        X: List[List[float]] = []
        y: List[Any] = []

        for r in rows:
            try:
                X.append([float(r[c]) for c in feature_cols])
            except KeyError as e:
                raise KeyError(f"Missing feature column {e} in CSV. csv={_resolve_path(root, csv_path)}") from e
            except ValueError as e:
                raise ValueError(f"Non-numeric value in features row: {r}") from e

            if label_col and label_col in r:
                y.append(r[label_col])

        X_arr = np.asarray(X, dtype=float)

        y_arr: Optional[np.ndarray]
        if label_col and len(y) == len(rows):
            y_arr = np.asarray(y, dtype=object)
        else:
            y_arr = None

        # Encode non-numeric labels to integers for ARI/NMI metrics
        if y_arr is not None:
            try:
                # if already numeric, keep
                if not np.issubdtype(y_arr.dtype, np.number):
                    uniq = sorted({str(v) for v in y_arr.tolist()})
                    mapping = {u:i for i,u in enumerate(uniq)}
                    y_arr = np.asarray([mapping[str(v)] for v in y_arr.tolist()], dtype=int)
                    meta_cfg["_label_mapping"] = mapping
            except Exception:
                pass

        if standardize:
            mu = X_arr.mean(axis=0)
            sigma = X_arr.std(axis=0)
            sigma = np.where(sigma == 0, 1.0, sigma)
            X_arr = (X_arr - mu) / sigma

        meta = {
            "dataset": dataset,
            "source": "csv",
            "csv": str(_resolve_path(root, csv_path)),
            "n_samples": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
            "feature_cols": list(feature_cols),
            "label_col": label_col,
            "standardized": standardize,
        }

        return {
            "name": dataset,
            "problem_type": "clustering",
            "meta": meta,
            "ml": {
                "X": X_arr,
                "y": y_arr,
                "feature_cols": list(feature_cols),
                "label_col": label_col,
                "standardized": standardize,
            },
        }