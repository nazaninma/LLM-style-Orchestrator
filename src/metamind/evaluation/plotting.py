from __future__ import annotations   # ← همیشه خط اول

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

from typing import Any, Dict, List, Optional

def plot_confusion_matrix(out_dir: Path, method: str, y_true, y_pred, title_prefix: str = "") -> Path:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)

    cm = confusion_matrix(yt, yp, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    im = ax.imshow(cm)  # رنگ پیشفرض خوبه، لازم نیست رنگ مشخص کنی

    ax.set_title(f"{title_prefix} Confusion Matrix - {method}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])

    # عدد داخل خانه‌ها
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = out_dir / f"confusion_{method}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _extract_series(artifacts: Dict[str, Any], problem_type: str) -> Optional[List[float]]:
    # 1) generic key (خیلی مهم)
    if "convergence_history" in artifacts and isinstance(artifacts["convergence_history"], list):
        try:
            return [float(v) for v in artifacts["convergence_history"]]
        except Exception:
            return None

    p = (problem_type or "").strip().lower()

    # 2) classification احتمالی
    for k in ("val_loss_history", "train_loss_history", "loss_history", "val_accuracy_history", "accuracy_history"):
        if k in artifacts and isinstance(artifacts[k], list):
            try:
                return [float(v) for v in artifacts[k]]
            except Exception:
                return None

    # 3) clustering احتمالی
    for k in ("inertia_history", "sse_history", "withinss_history", "silhouette_history"):
        if k in artifacts and isinstance(artifacts[k], list):
            try:
                return [float(v) for v in artifacts[k]]
            except Exception:
                return None

    return None


def _pad_to_same_length(series_list: List[List[float]]) -> np.ndarray:
    max_len = max(len(s) for s in series_list)
    arr = np.full((len(series_list), max_len), np.nan, dtype=float)
    for i, s in enumerate(series_list):
        arr[i, : len(s)] = np.asarray(s, dtype=float)
    return arr


def plot_convergence(
    out_dir: Path,
    problem_name: str,
    problem_type: str,
    per_method_runs: Dict[str, List[Dict[str, Any]]],
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for method, runs in per_method_runs.items():
        series_runs: List[List[float]] = []
        for r in runs:
            s = _extract_series(r.get("artifacts", {}) or {}, problem_type)
            if s is not None and len(s) > 1:
                series_runs.append(s)

        if not series_runs:
            continue

        arr = _pad_to_same_length(series_runs)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)

        x = np.arange(len(mean))

        plt.figure()
        plt.plot(x, mean)
        # باند عدم قطعیت (mean ± std)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.title(f"Convergence - {problem_name} - {method}")
        plt.xlabel("Iteration/Epoch")
        plt.ylabel("Metric (best/loss/accuracy)")
        plt.grid(True, alpha=0.3)

        p = out_dir / f"convergence_{method}.png"
        plt.savefig(p, dpi=140, bbox_inches="tight")
        plt.close()
        saved.append(p)

    return saved