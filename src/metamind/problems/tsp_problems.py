from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import math

import numpy as np

# Known optimal tour lengths for common TSPLIB instances (from project document)
KNOWN_OPTIMAL = {
    "eil51": 426,
    "berlin52": 7542,
    "kroa100": 21282,
}


def _resolve_path(root: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _parse_tsplib_coords(tsp_path: Path) -> Tuple[str, np.ndarray, str, int | None]:
    """
    Minimal TSPLIB parser for NODE_COORD_SECTION (EUC_2D).
    Returns: (name, coords[N,2])
    """
    name = tsp_path.stem
    lines = tsp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    edge_weight_type = "EUC_2D"
    declared_dimension: int | None = None

    in_coords = False
    coords: List[Tuple[float, float]] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if s.upper().startswith("NAME"):
            # NAME : eil51
            parts = s.replace(":", " ").split()
            if len(parts) >= 2:
                name = parts[-1].strip()

        if s.upper().startswith("EDGE_WEIGHT_TYPE"):
            # e.g. EDGE_WEIGHT_TYPE : EUC_2D
            parts = s.replace(":", " ").split()
            if len(parts) >= 2:
                edge_weight_type = parts[-1].strip().upper()

        if s.upper().startswith("DIMENSION"):
            # e.g. DIMENSION : 51
            parts = s.replace(":", " ").split()
            if len(parts) >= 2:
                try:
                    declared_dimension = int(float(parts[-1]))
                except Exception:
                    pass

        if s.upper().startswith("NODE_COORD_SECTION"):
            in_coords = True
            continue

        if s.upper().startswith("EOF"):
            break

        if in_coords:
            parts = s.split()
            if len(parts) >= 3:
                # idx x y
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
                except ValueError:
                    pass

    if not coords:
        raise ValueError(f"Could not parse NODE_COORD_SECTION from: {tsp_path}")

    return name, np.asarray(coords, dtype=float), edge_weight_type, declared_dimension


def _pairwise_euclidean(coords: np.ndarray, *, round_tsplib: bool = False) -> np.ndarray:
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            # TSPLIB's EUC_2D uses nint() rounding to nearest integer.
            if round_tsplib:
                d = float(int(d + 0.5))
            D[i, j] = d
            D[j, i] = d
    return D


class TSPProblem:
    """
    Phase 5.6 - TSP Problem Builder (TSPLIB)
    """

    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = bundle

    def build(self) -> Dict[str, Any]:
        root = Path(str(self.bundle.get("_root", Path.cwd())))
        meta_cfg = self.bundle.get("problem_meta", {}) or {}

        source = str(meta_cfg.get("source", "tsplib")).lower()
        data_dir = meta_cfg.get("data_dir", "data/tsplib")
        problem_id = meta_cfg.get("problem_id", str(self.bundle.get("_default_problem", "eil51")))
        tour_file = meta_cfg.get("tour_file", None)

        if source not in ("tsplib", "generated"):
            raise ValueError(f"Unsupported TSP source={source}. Supported: tsplib, generated")

        if source == "generated":
            n_cities = int(meta_cfg.get("n_cities", meta_cfg.get("cities", 50)))
            seed = int(self.bundle.get("_seed", 42))
            scale = float(meta_cfg.get("scale", 100.0))
            rng = np.random.default_rng(seed)
            coords = rng.random((n_cities, 2)) * scale
            name = f"generated_{n_cities}"
            dist = _pairwise_euclidean(coords)
            meta = {
                "source": source,
                "problem_id": name,
                "dimension": int(coords.shape[0]),
                "seed": seed,
                "scale": scale,
            }
            return {
                "name": name,
                "problem_type": "tsp",
                "meta": meta,
                "tsp": {"coords": coords, "distance": dist, "dist": dist, "D": dist, "optimal": float(meta_cfg.get("optimal")) if meta_cfg.get("optimal") is not None else None},
            }

        tsp_file = meta_cfg.get("tsp_file", f"{data_dir}/{problem_id}.tsp")
        tsp_path = _resolve_path(root, tsp_file)

        name, coords, edge_weight_type, declared_dim = _parse_tsplib_coords(tsp_path)

        # Validate file consistency: TSPLIB DIMENSION should match coords count.
        if declared_dim is not None and declared_dim != int(coords.shape[0]):
            allow = bool(meta_cfg.get("allow_dimension_mismatch", False))
            if not allow:
                raise ValueError(
                    f"TSPLIB file dimension mismatch for {tsp_path}: "
                    f"DIMENSION={declared_dim} but parsed {coords.shape[0]} coords. "
                    f"Use a complete .tsp file or set problem_meta.allow_dimension_mismatch=true."
                )

        # TSPLIB's EUC_2D uses integer-rounded distances; this is required to
        # match known optimal tour lengths in the project document.
        round_tsplib = (edge_weight_type == "EUC_2D")
        dist = _pairwise_euclidean(coords, round_tsplib=round_tsplib)

        tour_path: Optional[Path] = _resolve_path(root, tour_file) if tour_file else None

        meta = {
            "source": source,
            "problem_id": problem_id,
            "tsp_file": str(tsp_path),
            "tour_file": str(tour_path) if tour_path else None,
            "dimension": int(coords.shape[0]),
        }

        # Resolve known optimum using multiple identifiers (config id, TSPLIB NAME,
        # and the common "light_" prefix used in bundled demo files).
        pid = str(problem_id).lower()
        name_key = str(name).lower()
        pid2 = pid[6:] if pid.startswith("light_") else pid
        known_opt = KNOWN_OPTIMAL.get(pid) or KNOWN_OPTIMAL.get(pid2) or KNOWN_OPTIMAL.get(name_key)

        return {
            "name": name,
            "problem_type": "tsp",
            "meta": meta,
            "tsp": {
                "coords": coords,
                # keep both keys to maximize method compatibility
                "distance": dist,
                "dist": dist,
                "D": dist,
                "optimal": float(meta_cfg.get("optimal", known_opt)) if meta_cfg.get("optimal", known_opt) is not None else None,
            },
        }