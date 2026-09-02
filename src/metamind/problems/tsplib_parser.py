from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class TSPLIBInstance:
    name: str
    dimension: int
    edge_weight_type: str
    coords: List[Tuple[float, float]]


def _parse_key_val(line: str):
    # remove BOM if exists
    line = line.lstrip("\ufeff").strip()

    # Case 1: KEY: VALUE
    if ":" in line:
        k, v = line.split(":", 1)
        return k.strip().upper(), v.strip()

    # Case 2: KEY VALUE  (whitespace separated)
    parts = line.split()
    if len(parts) >= 2:
        k = parts[0].strip().upper()
        v = " ".join(parts[1:]).strip()
        return k, v

    return None


def parse_tsplib_tsp(path: str | Path) -> TSPLIBInstance:
    """
    Minimal TSPLIB .tsp parser.
    Supports:
      - TYPE: TSP
      - EDGE_WEIGHT_TYPE: EUC_2D
      - NODE_COORD_SECTION
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSPLIB file not found: {path}")

    name = path.stem
    dimension = None
    edge_weight_type = None
    coords_map: Dict[int, Tuple[float, float]] = {}

    in_coords = False

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        u = line.upper()

        if u == "NODE_COORD_SECTION":
            in_coords = True
            continue
        if u == "EOF":
            break

        if not in_coords:
            kv = _parse_key_val(line)
            if kv:
                k, v = kv
                if k == "NAME":
                    name = v
                elif k == "DIMENSION":
                    dimension = int(v)
                elif k == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = v.upper()
            continue

        # Coordinate row: "index x y"
        parts = line.split()
        if len(parts) >= 3:
            idx = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            coords_map[idx] = (x, y)

    if dimension is None:
        raise ValueError(f"Missing DIMENSION in TSPLIB file: {path}")
    if edge_weight_type is None:
        raise ValueError(f"Missing EDGE_WEIGHT_TYPE in TSPLIB file: {path}")
    if edge_weight_type != "EUC_2D":
        raise ValueError(
            f"Only EDGE_WEIGHT_TYPE=EUC_2D is supported right now. Got: {edge_weight_type} in {path}"
        )

    # TSPLIB indices usually start at 1
    coords: List[Tuple[float, float]] = []
    for i in range(1, dimension + 1):
        if i not in coords_map:
            raise ValueError(f"Missing coord for node {i} in {path}")
        coords.append(coords_map[i])

    return TSPLIBInstance(
        name=name,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        coords=coords,
    )


def parse_tsplib_tour(path: str | Path) -> List[int]:
    """
    Minimal TSPLIB .tour parser.
    Returns 0-based node indices as a tour list.
    Supports TOUR_SECTION with node ids, ends with -1 or EOF.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TOUR file not found: {path}")

    in_tour = False
    nodes: List[int] = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        u = line.upper()

        if u == "TOUR_SECTION":
            in_tour = True
            continue
        if u == "EOF":
            break

        if not in_tour:
            continue

        # Tour section may have single int per line
        try:
            val = int(float(line))
        except ValueError:
            continue

        if val == -1:
            break

        # convert to 0-based
        nodes.append(val - 1)

    if not nodes:
        raise ValueError(f"No tour nodes parsed from: {path}")

    return nodes