from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from metamind.types import RunResult


def make_run_dir(results_root: Path, run_id: str) -> Path:
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_results_json(run_dir: Path, results: List[RunResult]) -> Path:
    out = run_dir / "results.json"
    save_json(out, [r.to_dict() for r in results])
    return out


def save_results_csv(run_dir: Path, results: List[RunResult]) -> Path:
    """
    Flattens metadata + metrics to a CSV table.
    Artifacts are ignored here (kept in JSON).
    """
    out = run_dir / "results.csv"
    rows: List[Dict[str, Any]] = []

    for r in results:
        row = {
            "run_id": r.metadata.run_id,
            "timestamp_utc": r.metadata.timestamp_utc,
            "seed": r.metadata.seed,
            "problem_name": r.metadata.problem_name,
            "method_name": r.metadata.method_name,
            "notes": r.notes or "",
        }
        # metrics flattened
        for k, v in r.metrics.items():
            row[f"metric__{k}"] = v
        rows.append(row)

    # gather all columns
    columns = []
    for row in rows:
        for k in row.keys():
            if k not in columns:
                columns.append(k)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    return out