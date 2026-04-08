#!/usr/bin/env python3
"""
Scan manifest.json packs from a list file: count (w, h) distribution and mismatches.

Does not modify data. Use to see how many scenes use non-square or varying resolutions.

Example:
  python scripts/audit_manifest_resolutions.py \\
    --manifest-list /root/data/DL3DV/dl3dv_train_h256/all_manifests.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

WH = Tuple[int, int]


def _load_manifest_paths(list_path: str) -> List[str]:
    with open(list_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _scene_wh_stats(manifest_path: str) -> Optional[Tuple[WH, Set[WH], int]]:
    """
    Returns (primary_wh, all_distinct_wh, n_frames) or None on error.
    primary_wh = (w,h) from first frame; all_distinct_wh = unique (w,h) across frames.
    """
    try:
        with open(manifest_path, "r", encoding="utf-8") as jf:
            man: Dict[str, Any] = json.load(jf)
    except (OSError, json.JSONDecodeError) as e:
        print(f"skip read error {manifest_path}: {e}", file=sys.stderr)
        return None
    frames = man.get("frames") or []
    if not frames:
        return None
    distinct: Set[WH] = set()
    for fr in frames:
        w, h = int(fr["w"]), int(fr["h"])
        distinct.add((w, h))
    fr0 = frames[0]
    primary = (int(fr0["w"]), int(fr0["h"]))
    return primary, distinct, len(frames)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-list", type=str, required=True)
    p.add_argument(
        "--max-manifests",
        type=int,
        default=None,
        help="Only scan first N manifests (for quick smoke test).",
    )
    args = p.parse_args()

    paths = _load_manifest_paths(args.manifest_list)
    if args.max_manifests is not None:
        paths = paths[: args.max_manifests]

    counter_primary: Counter[WH] = Counter()
    counter_inconsistent_scenes = 0  # multiple (w,h) within one manifest
    counter_read_fail = 0
    counter_empty = 0
    total_frames = 0

    for mp in paths:
        if not os.path.isfile(mp):
            counter_read_fail += 1
            continue
        stats = _scene_wh_stats(mp)
        if stats is None:
            counter_read_fail += 1
            continue
        primary, distinct, nfr = stats
        total_frames += nfr
        if not distinct:
            counter_empty += 1
            continue
        counter_primary[primary] += 1
        if len(distinct) > 1:
            counter_inconsistent_scenes += 1

    n_ok = sum(counter_primary.values())
    print(f"manifest_list: {args.manifest_list}")
    print(f"manifests listed: {len(paths)}")
    print(f"manifests with readable frames: {n_ok}")
    print(f"read failures / empty: {counter_read_fail}")
    print(f"scenes with multiple (w,h) inside same manifest: {counter_inconsistent_scenes}")
    print(f"total frames (summed): {total_frames}")
    print()
    print("Primary (w, h) per scene — count (share of readable manifests):")
    for (w, h), c in counter_primary.most_common():
        pct = 100.0 * c / n_ok if n_ok else 0.0
        print(f"  {w:4d} x {h:4d}  :  {c:6d}  ({pct:5.2f}%)")

    square_256 = counter_primary.get((256, 256), 0)
    if n_ok:
        print()
        print(f"Scenes with exact 256x256 (primary): {square_256} ({100.0 * square_256 / n_ok:.2f}%)")
        nonsq = n_ok - square_256
        print(f"Scenes with other primary (w,h): {nonsq} ({100.0 * nonsq / n_ok:.2f}%)")


if __name__ == "__main__":
    main()
