"""Load preprocessed manifest.json packs (see scripts/preprocess_dl3dv_training_pack.py)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Collection, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

from training.curriculum_sampling import (
    CurriculumProgress,
    interpolate_delta_t_for_o_target,
    load_dino_profile,
    resolve_profile_path,
    semantic_overlap_schedule,
)
from training.view_layout import apply_permutation_to_sequence, batch_slot_permutation

MANIFEST_INDEX_WORKERS = 16
_CURRICULUM_INDEX_CACHE_VERSION = 1


def load_manifest_paths(list_path: str) -> List[str]:
    with open(list_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _read_manifest_json(mp: str) -> dict:
    with open(mp, "r", encoding="utf-8") as jf:
        return json.load(jf)


def _parallel_over_paths(paths: List[str], map_one):
    if len(paths) < 8:
        return [map_one(p) for p in paths]
    with ThreadPoolExecutor(max_workers=MANIFEST_INDEX_WORKERS) as ex:
        return list(ex.map(map_one, paths))


def _load_curriculum_scene_from_manifest(mp: str) -> Optional[Dict[str, Any]]:
    man = _read_manifest_json(mp)
    frames = man.get("frames") or []
    if len(frames) < 2:
        return None
    return {
        "manifest_path": mp,
        "base": os.path.dirname(os.path.abspath(mp)),
        "scene_name": man["scene_name"],
        "frames": frames,
    }


def _build_curriculum_scenes_index(paths: List[str]) -> List[Dict[str, Any]]:
    raw = _parallel_over_paths(paths, _load_curriculum_scene_from_manifest)
    return [r for r in raw if r is not None]


def _fingerprint_manifest_index(manifest_list_path: str, paths: List[str]) -> str:
    h = hashlib.sha256()
    with open(os.path.abspath(os.path.expanduser(manifest_list_path)), "rb") as f:
        h.update(f.read())
    for raw in paths:
        p = os.path.abspath(os.path.expanduser(raw.strip()))
        h.update(p.encode("utf-8"))
        try:
            st = os.stat(p)
            h.update(str(st.st_mtime_ns).encode("ascii"))
            h.update(str(st.st_size).encode("ascii"))
        except OSError:
            h.update(b"missing")
    return h.hexdigest()


def _try_load_curriculum_index_cache(
    cache_path: str, expected_fp: str
) -> Optional[List[Dict[str, Any]]]:
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != _CURRICULUM_INDEX_CACHE_VERSION:
        return None
    if str(payload.get("fingerprint", "")) != expected_fp:
        return None
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        return None
    return scenes


def _save_curriculum_index_cache_atomic(
    cache_path: str, fingerprint: str, scenes: List[Dict[str, Any]]
) -> None:
    payload = {
        "version": _CURRICULUM_INDEX_CACHE_VERSION,
        "fingerprint": fingerprint,
        "scenes": scenes,
    }
    d = os.path.dirname(os.path.abspath(cache_path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".erayzer_curriculum_idx_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as wf:
            pickle.dump(payload, wf, protocol=4)
        os.replace(tmp, cache_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def _exclusive_file_lock(lock_path: str) -> Iterator[None]:
    """Exclusive lock for cache rebuild (fcntl on POSIX; no-op if fcntl unavailable)."""
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)) or ".", exist_ok=True)
    with open(lock_path, "a", encoding="utf-8") as lf:
        if _fcntl is not None:
            _fcntl.flock(lf.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(lf.fileno(), _fcntl.LOCK_UN)


def index_curriculum_scenes_from_manifest_list(
    manifest_list_path: str,
    *,
    use_cache: bool = True,
    cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load scene records (parallel I/O when len(manifests) >= 8).

    With ``use_cache`` (default), uses ``<list>.erayzer_curriculum_scenes.pkl`` (or
    ``cache_path``); invalidated when the list file or any manifest mtime/size changes.
    """
    paths = load_manifest_paths(manifest_list_path)
    if not paths:
        return []
    if not use_cache:
        return _build_curriculum_scenes_index(paths)

    fp = _fingerprint_manifest_index(manifest_list_path, paths)
    cpath = cache_path or (
        os.path.abspath(os.path.expanduser(manifest_list_path)) + ".erayzer_curriculum_scenes.pkl"
    )

    hit = _try_load_curriculum_index_cache(cpath, fp)
    if hit is not None:
        return hit

    with _exclusive_file_lock(cpath + ".lock"):
        hit = _try_load_curriculum_index_cache(cpath, fp)
        if hit is not None:
            return hit
        scenes = _build_curriculum_scenes_index(paths)
        try:
            _save_curriculum_index_cache_atomic(cpath, fp, scenes)
        except OSError:
            pass
        return scenes


def _manifest_flat_items(mp: str) -> List[Tuple[str, dict]]:
    man = _read_manifest_json(mp)
    base = os.path.dirname(os.path.abspath(mp))
    return [(base, fr) for fr in man.get("frames") or []]


def _resize_rgb_chw(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """x: [3,H,W] in [0,1] -> [3,out_h,out_w] bicubic."""
    if x.shape[1] == out_h and x.shape[2] == out_w:
        return x
    y = x.unsqueeze(0)
    y = F.interpolate(y, size=(out_h, out_w), mode="bicubic", align_corners=False)
    return y.squeeze(0).clamp(0.0, 1.0)


def _load_frame_sample(
    base: str,
    fr: dict,
    *,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Load one frame. Optional resize matches model input size; intrinsics/wh stay in original pixels."""
    im = Image.open(os.path.join(base, fr["image"])).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    x = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)
    if image_height is not None and image_width is not None:
        x = _resize_rgb_chw(x, int(image_height), int(image_width))
    return {
        "image": x,
        "w2c": torch.tensor(fr["w2c"], dtype=torch.float32),
        "c2w": torch.tensor(fr["c2w"], dtype=torch.float32),
        "intrinsics": torch.tensor(
            [fr["fx"], fr["fy"], fr["cx"], fr["cy"]], dtype=torch.float32
        ),
        "wh": torch.tensor([fr["w"], fr["h"]], dtype=torch.float32),
    }


class ManifestFrameDataset(Dataset):
    """
    One sample = one frame. All frames from all scenes in the manifest list are flattened.

    Expects each manifest.json next to rgb/ with entries:
      image, w2c, c2w, fx, fy, cx, cy, w, h
    """

    def __init__(
        self,
        manifest_list_path: str,
        *,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._image_height = image_height
        self._image_width = image_width
        paths = load_manifest_paths(manifest_list_path)
        chunks = _parallel_over_paths(paths, _manifest_flat_items)
        self._items: List[Tuple[str, dict]] = [item for ch in chunks for item in ch]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        base, fr = self._items[idx]
        return _load_frame_sample(
            base,
            fr,
            image_height=self._image_height,
            image_width=self._image_width,
        )


class ManifestCurriculumSequenceDataset(Dataset):
    """
    One sample = one temporally spaced multi-view strip (V frames) from a single scene.

    Semantic branch (paper Sec. 3.3): precomputed O_u(dt) from DINO JSON, schedule
    o(s)=s*o_min+(1-s)*o_max, interpolate integer stride dt, then frames
    i, i+dt, ..., i+(V-1)*dt.

    Optional ``view_layout`` (when ``num_input_views + num_target_views == V``) reorders
    the temporal strip into batch slots ``[inputs | targets]``; use ``interleaved_even_odd`` or
    ``ends_bridge`` to avoid contiguous temporal halves when training GS (see ``training/view_layout.py``).

    ``CurriculumProgress`` must be updated each step (``CurriculumRampCallback``).

    Pass ``prefetched_scenes`` (full index before allow/block lists) from a DataModule to
    build the manifest index once for both train and val loaders.
    """

    def __init__(
        self,
        manifest_list_path: str,
        curriculum_progress: CurriculumProgress,
        profile_dir: str | None = None,
        num_views: int = 10,
        o_max: float = 1.0,
        o_min: float = 0.75,
        fallback_delta_t: int = 4,
        length: int = 20_000,
        rng_base_seed: int = 0,
        view_layout: str = "temporal_halves",
        num_input_views: int = 5,
        num_target_views: int = 5,
        scene_name_allowlist: Optional[Collection[str]] = None,
        scene_name_blocklist: Optional[Collection[str]] = None,
        cache_manifest_index: bool = True,
        manifest_index_cache_path: Optional[str] = None,
        prefetched_scenes: Optional[List[Dict[str, Any]]] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._progress = curriculum_progress
        self._image_height = image_height
        self._image_width = image_width
        self.profile_dir = profile_dir
        self.num_views = int(num_views)
        self.o_max = float(o_max)
        self.o_min = float(o_min)
        self.fallback_delta_t = max(1, int(fallback_delta_t))
        self._length = int(length)
        self._rng_base_seed = int(rng_base_seed)
        self.view_layout = str(view_layout)
        self._layout_n_in = int(num_input_views)
        self._layout_n_tgt = int(num_target_views)

        if prefetched_scenes is not None:
            self._scenes = list(prefetched_scenes)
        else:
            self._scenes = index_curriculum_scenes_from_manifest_list(
                manifest_list_path,
                use_cache=cache_manifest_index,
                cache_path=manifest_index_cache_path,
            )
        if not self._scenes:
            raise ValueError(
                f"No usable scenes (>=2 frames) in manifest list: {manifest_list_path}"
            )

        if scene_name_allowlist is not None:
            allow = set(scene_name_allowlist)
            self._scenes = [s for s in self._scenes if s["scene_name"] in allow]
        if scene_name_blocklist is not None:
            block = set(scene_name_blocklist)
            self._scenes = [s for s in self._scenes if s["scene_name"] not in block]
        if not self._scenes:
            raise ValueError(
                "After scene allowlist/blocklist filtering, no scenes remain. "
                "Check benchmark list vs manifest scene_name."
            )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self._rng_base_seed + idx)

        s_idx = int(rng.integers(0, len(self._scenes)))
        scene = self._scenes[s_idx]
        frames = scene["frames"]
        base = scene["base"]
        N = len(frames)

        s = self._progress.get()
        o_target = semantic_overlap_schedule(s, self.o_max, self.o_min)

        prof_path = (
            resolve_profile_path(self.profile_dir, scene["scene_name"])
            if self.profile_dir
            else None
        )
        if prof_path is not None:
            dts, ous = load_dino_profile(prof_path)
            dt_float = interpolate_delta_t_for_o_target(o_target, dts, ous)
            dt_int = max(1, int(round(dt_float)))
        else:
            dt_int = self.fallback_delta_t

        v = min(self.num_views, N)
        if v < 2:
            v = min(2, N)

        if v > 1:
            dt_cap = max(1, (N - 1) // (v - 1))
            dt_int = max(1, min(dt_int, dt_cap))

        span = (v - 1) * dt_int
        i_max = N - 1 - span
        while i_max < 0 and v > 2:
            v -= 1
            span = (v - 1) * dt_int
            i_max = N - 1 - span
        if i_max < 0:
            v = 2
            dt_int = max(1, (N - 1) // max(1, v - 1))
            span = (v - 1) * dt_int
            i_max = N - 1 - span
        i0 = int(rng.integers(0, max(1, i_max + 1)))
        indices = [i0 + k * dt_int for k in range(v)]

        views = [
            _load_frame_sample(
                base,
                frames[j],
                image_height=self._image_height,
                image_width=self._image_width,
            )
            for j in indices
        ]

        if v == self._layout_n_in + self._layout_n_tgt:
            perm = batch_slot_permutation(
                v,
                self._layout_n_in,
                self._layout_n_tgt,
                self.view_layout,
                rng=rng,
            )
            views = apply_permutation_to_sequence(views, perm)
            indices = apply_permutation_to_sequence(indices, perm)

        return {
            "image": torch.stack([x["image"] for x in views], dim=0),
            "w2c": torch.stack([x["w2c"] for x in views], dim=0),
            "c2w": torch.stack([x["c2w"] for x in views], dim=0),
            "intrinsics": torch.stack([x["intrinsics"] for x in views], dim=0),
            "wh": torch.stack([x["wh"] for x in views], dim=0),
            "frame_indices": torch.tensor(indices, dtype=torch.int64),
            "delta_t": torch.tensor(dt_int, dtype=torch.int64),
            "o_target": torch.tensor(o_target, dtype=torch.float32),
            "view_layout": self.view_layout,
        }
