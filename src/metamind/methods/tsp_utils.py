from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional


def euclidean_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def tour_length(tour: List[int], dist: List[List[float]]) -> float:
    n = len(tour)
    total = 0.0
    for i in range(n - 1):
        total += dist[tour[i]][tour[i + 1]]
    total += dist[tour[-1]][tour[0]]  # return
    return total


def make_random_tour(n: int, rng: random.Random) -> List[int]:
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    # reverse segment i..k
    new_tour = tour[:i] + list(reversed(tour[i:k + 1])) + tour[k + 1:]
    return new_tour


def two_opt_local_search(tour: List[int], dist: List[List[float]], max_improve: int = 200) -> List[int]:
    """
    Simple 2-opt improvement. max_improve limits number of accepted improvements.
    """
    best = tour[:]
    best_len = tour_length(best, dist)
    n = len(best)
    improvements = 0

    improved = True
    while improved and improvements < max_improve:
        improved = False
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                cand = two_opt_swap(best, i, k)
                cand_len = tour_length(cand, dist)
                if cand_len < best_len:
                    best, best_len = cand, cand_len
                    improvements += 1
                    improved = True
                    break
            if improved:
                break

    return best