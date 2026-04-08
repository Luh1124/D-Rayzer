"""
Reorder a temporal multi-view strip into batch tensor order [input slots | target slots].

DINO curriculum only chooses stride dt and start frame i0; this module maps the V
time-ordered frames to V network slots (reproj: first num_input slots = inputs, rest = targets).

Extend with new modes by adding a branch in ``batch_slot_permutation``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


def batch_slot_permutation(
    num_views: int,
    num_input_views: int,
    num_target_views: int,
    mode: str,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Return ``perm`` of length ``num_views``: batch slot ``k`` shows the frame that was at
    temporal strip index ``perm[k]`` (strip = i0, i0+dt, ...).

    After reordering, slots ``0:num_input_views`` are inputs and ``num_input_views:`` targets.

    ``rng`` is required for stochastic layouts (e.g. ``ends_anchors_random_rest``); pass the same
    per-sample ``numpy.random.Generator`` as in the dataset for reproducibility.
    """
    v = int(num_views)
    n_in = int(num_input_views)
    n_tgt = int(num_target_views)
    if v != n_in + n_tgt:
        raise ValueError(f"view layout needs V = n_in + n_tgt, got {v} != {n_in}+{n_tgt}")
    mode = str(mode).lower().strip()

    if mode in ("temporal_halves", "identity", "sequential", "half"):
        return list(range(v))

    if mode in ("interleaved_gcd", "rayzer_gcd", "interleaved"):
        return _interleaved_gcd_permutation(v, n_in, n_tgt)

    if mode == "interleaved_even_odd":
        return _interleaved_even_odd_permutation(v, n_in, n_tgt)

    if mode in ("ends_anchors_random_rest", "anchors_random_middle"):
        if rng is None:
            rng = np.random.default_rng()
        return _ends_anchors_random_rest_permutation(v, n_in, n_tgt, rng)

    if mode in ("ends_bridge", "bridge"):
        return _ends_bridge_permutation(v, n_in, n_tgt)

    if mode in ("rayzer_tail_targets", "tail_targets"):
        # Same slot order as identity when n_in + n_tgt == v (targets = last n_tgt strip indices)
        return list(range(v))

    raise ValueError(
        f"Unknown view_layout mode {mode!r}. "
        f"Try: temporal_halves, interleaved_gcd, interleaved_even_odd, ends_anchors_random_rest, "
        f"ends_bridge, rayzer_tail_targets"
    )


def _ends_anchors_random_rest_permutation(
    v: int, n_in: int, n_tgt: int, rng: np.random.Generator
) -> List[int]:
    """
    Input strip indices always include temporal endpoints ``0`` and ``v-1``; the remaining
    ``n_in - 2`` inputs are sampled without replacement from ``{1, …, v-2}``. Target slots are
    the other strip indices. Batch order is ``sorted(inputs) | sorted(targets)`` (time-ordered
    within each block).

    Requires ``n_in >= 2`` (so both ends can be reference views). If ``v == 2``, only
    ``n_in == 2, n_tgt == 0`` is valid for strict anchoring; with ``n_in == n_tgt == 1`` this
    mode cannot place both ends in inputs — call site should use another layout or counts.
    """
    if n_in < 2:
        raise ValueError(
            "ends_anchors_random_rest needs num_input_views >= 2 (both strip endpoints as inputs)."
        )
    if v < 2:
        return list(range(v))
    ends = {0, v - 1}
    middle = list(range(1, v - 1))
    need_mid = n_in - 2
    if need_mid > len(middle):
        raise ValueError(
            f"ends_anchors_random_rest: need {need_mid} middle inputs but only {len(middle)} middle indices"
        )
    if need_mid < 0:
        raise ValueError("ends_anchors_random_rest: inconsistent n_in")
    if need_mid == 0:
        chosen_mid: List[int] = []
    else:
        chosen_mid = list(rng.choice(middle, size=need_mid, replace=False))
        chosen_mid = [int(x) for x in chosen_mid]
    inp = sorted(ends | set(chosen_mid))
    if len(inp) != n_in:
        raise ValueError("ends_anchors_random_rest: duplicate strip index in inputs (internal bug)")
    tgt = sorted(set(range(v)) - set(inp))
    if len(tgt) != n_tgt:
        raise ValueError(
            f"ends_anchors_random_rest: expected {n_tgt} targets, got {len(tgt)} (n_in={n_in}, v={v})"
        )
    return inp + tgt


def _interleaved_even_odd_permutation(v: int, n_in: int, n_tgt: int) -> List[int]:
    """
    Alternate indices along the temporal strip: batch slots are
    ``[even strip indices …][odd strip indices …]`` or the swap, so reference (input)
    views are spread through time (E-RayZer Fig.2: e.g. v=5, 3 refs + 2 targets → strip 0,2,4 | 1,3).

    Requires an even/odd *count* split: ``(n_in, n_tgt)`` is ``(ceil(v/2), floor(v/2))`` or the
    reverse. When ``v`` is even and ``n_in == n_tgt == v/2``, inputs use even strip indices by convention.
    """
    if n_in + n_tgt != v:
        raise ValueError(f"interleaved_even_odd: need n_in+n_tgt==v, got {n_in}+{n_tgt}!={v}")
    c = (v + 1) // 2  # ceil(v/2)
    f = v // 2  # floor(v/2)
    evens = list(range(0, v, 2))
    odds = list(range(1, v, 2))
    if c == f:
        if n_in != c or n_tgt != c:
            raise ValueError(
                f"interleaved_even_odd: for even v={v} need n_in=n_tgt={c}, got {n_in},{n_tgt}"
            )
        return evens + odds
    if n_in == c and n_tgt == f:
        return evens + odds
    if n_in == f and n_tgt == c:
        return odds + evens
    raise ValueError(
        f"interleaved_even_odd: incompatible split for v={v}; need "
        f"(n_in,n_tgt) in {{({c},{f}), ({f},{c})}}, got ({n_in},{n_tgt})"
    )


def _interleaved_gcd_permutation(v: int, n_in: int, n_tgt: int) -> List[int]:
    """RayZer ``_build_indices``: gcd-sized groups, first part of each group → input slots."""
    g = math.gcd(n_in, n_tgt)
    if g == 0:
        return list(range(v))
    group_size = v // g
    in_per_group = n_in // g
    tar_per_group = n_tgt // g
    input_indices: List[int] = []
    target_indices: List[int] = []
    for group_idx in range(g):
        start = group_idx * group_size
        block = list(range(start, start + group_size))
        input_indices.extend(block[:in_per_group])
        target_indices.extend(block[in_per_group : in_per_group + tar_per_group])
    input_indices.sort()
    target_indices.sort()
    return input_indices + target_indices


def _ends_bridge_permutation(v: int, n_in: int, n_tgt: int) -> List[int]:
    """
    Inputs: first (n_in - 1) strip frames + **last** strip frame (temporal endpoints + early context).
    Targets: middle ``n_tgt`` consecutive strip frames.

    Improves baseline spread vs pure halves when the strip is long and you want anchors at both ends.
    Requires ``n_in >= 2`` and ``n_tgt == v - n_in``.
    """
    if n_in < 2:
        return list(range(v))
    inp = list(range(n_in - 1)) + [v - 1]
    mid = list(range(n_in - 1, v - 1))
    if len(mid) != n_tgt:
        raise ValueError("ends_bridge: middle segment length mismatch; check n_in, n_tgt, v")
    return inp + mid


def apply_permutation_to_sequence(items: Sequence, perm: Sequence[int]) -> List:
    """Reorder sequence (length V) by ``perm`` (same convention as ``batch_slot_permutation``)."""
    return [items[int(perm[k])] for k in range(len(perm))]
