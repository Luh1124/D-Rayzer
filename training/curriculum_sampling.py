"""
E-RayZer paper Sec. 3.3 semantic curriculum: overlap schedule o(s) and delta_t from precomputed O_u(dt).

See: https://arxiv.org/abs/2512.10950
"""

from __future__ import annotations

import functools
import json
import multiprocessing as mp
import os
from typing import List, Sequence, Tuple

import numpy as np


class CurriculumProgress:
    """
    Training curriculum progress s in [0, 1], shared across DataLoader workers via mp.Value.
    Update from a Lightning callback; workers read in Dataset.__getitem__.
    Prefer num_workers=0 if your platform cannot share this (e.g. some spawn setups).
    """

    def __init__(self) -> None:
        self._v = mp.Value("d", 0.0)

    def set(self, s: float) -> None:
        s = float(max(0.0, min(1.0, s)))
        with self._v.get_lock():
            self._v.value = s

    def get(self) -> float:
        with self._v.get_lock():
            return float(self._v.value)


def semantic_overlap_schedule(s: float, o_max: float = 1.0, o_min: float = 0.75) -> float:
    """
    Target visual-overlap level o(s) = s * o_min + (1 - s) * o_max, s in [0, 1].
    Sec. 4.1: semantic scheduling uses approximately 1.0 -> 0.75 (o_max -> o_min).
    """
    s = float(np.clip(s, 0.0, 1.0))
    return s * o_min + (1.0 - s) * o_max


def interpolate_delta_t_for_o_target(
    o_target: float,
    delta_t_values: Sequence[int],
    o_u_values: Sequence[float],
) -> float:
    """
    Map target overlap ``o_target`` to a (possibly fractional) frame spacing by linear
    interpolation on the precomputed (delta_t_k, O_u_k) curve (paper Sec. 3.3).
    """
    if not delta_t_values:
        return 1.0
    pairs = sorted(zip(delta_t_values, o_u_values), key=lambda x: int(x[0]))
    dts = np.array([float(p[0]) for p in pairs], dtype=np.float64)
    ou = np.array([float(p[1]) for p in pairs], dtype=np.float64)

    o_hi = float(np.max(ou))
    o_lo = float(np.min(ou))
    if o_target >= o_hi:
        return float(dts[int(np.argmax(ou))])
    if o_target <= o_lo:
        return float(dts[int(np.argmin(ou))])

    for k in range(len(dts) - 1):
        oa, ob = float(ou[k]), float(ou[k + 1])
        da, db = float(dts[k]), float(dts[k + 1])
        lo, hi = min(oa, ob), max(oa, ob)
        if not (lo - 1e-9 <= o_target <= hi + 1e-9):
            continue
        if abs(ob - oa) < 1e-12:
            return da
        t = (o_target - oa) / (ob - oa)
        return float(da + t * (db - da))

    return float(dts[len(dts) // 2])


@functools.lru_cache(maxsize=2048)
def load_dino_profile(path: str) -> Tuple[List[int], List[float]]:
    """Cached per process (each DataLoader worker has its own cache)."""
    with open(path, "r", encoding="utf-8") as f:
        row = json.load(f)
    return list(row["delta_t_values"]), list(row["O_u"])


def resolve_profile_path(profile_dir: str, scene_name: str) -> str | None:
    p = os.path.join(profile_dir, f"{scene_name}.json")
    return p if os.path.isfile(p) else None
