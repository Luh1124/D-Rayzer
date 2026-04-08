"""
RayZer DL3DV benchmark scene IDs (same hash as DL3DV folder / E-RayZer ``manifest.scene_name``).

The bundled list is copied from RayZer ``data/dl3dv10k_benchmark.txt`` (paths like
``./dl3dv_benchmark/<hash>/opencv_cameras.json``). Use ``scene_hashes_from_rayzer_benchmark_file``.
"""

from __future__ import annotations

import os
from typing import FrozenSet, Optional


def scene_hashes_from_rayzer_benchmark_file(path: str) -> FrozenSet[str]:
    """
    Parse RayZer's ``dl3dv10k_benchmark.txt``: each line contains ``.../dl3dv_benchmark/<hash>/...``.
    Returns the set of ``<hash>`` strings.
    """
    if not path or not os.path.isfile(path):
        return frozenset()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\\", "/")
            if not line or line.startswith("#"):
                continue
            parts = line.split("/")
            for i, p in enumerate(parts):
                if p == "dl3dv_benchmark" and i + 1 < len(parts):
                    out.add(parts[i + 1])
                    break
    return frozenset(out)


def default_benchmark_path(repo_root: Optional[str] = None) -> str:
    root = repo_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(root, "data", "rayzer_dl3dv10k_benchmark.txt")
