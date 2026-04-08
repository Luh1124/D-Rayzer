#!/usr/bin/env python3
"""
Build per-scene training packs: resized RGB + manifest.json (w2c + intrinsics at output resolution).

- Reads DL3DV-ALL-960P: transforms.json + images_4/
- Camera: Nerfstudio/OpenGL transform_matrix -> OpenCV c2w -> store w2c = inv(c2w) in manifest
- Geometry: intrinsics scaled from transforms full-res -> source PNG -> resize, then **center square crop by default** (Open-Rayzer style, e.g. 256x256).
  Use ``--no-square-crop`` for rectangular output (width varies by aspect ratio); training must then resize in the loader or handle varying H/W.

Output layout (per scene):
  <output_root>/<scene_hash>/
    rgb/frame_00001.png ...
    manifest.json

List file: one absolute path per line to manifest.json (same list for
scripts/compute_dino_semantic_overlap_profiles.py --manifest-list).

Parallelism: scene-level multiprocessing (CPU-bound resize). Default worker
count is min(8, CPU count); use --num-workers 1 for a single process.

Corrupt/truncated PNGs: by default the scene is skipped and partial output is
removed; use --allow-truncated-png to let PIL load truncated files (weaker).

Example:
python scripts/preprocess_dl3dv_training_pack.py --dataset-root /root/data/DL3DV/DL3DV-ALL-960P --output-root /root/data/DL3DV/dl3dv_train_h256 --out-list /root/data/DL3DV/dl3dv_train_h256/all_manifests.txt --target-height 256 --patch-size 16 --num-workers 16

Rectangular packs (no center crop): add ``--no-square-crop``.


Re-runs: scenes with a valid existing ``manifest.json`` (matching target_height, patch_size, square_crop, camera convention, and frame count) are skipped unless ``--overwrite``. Incomplete or mismatched output dirs are removed and re-processed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image, ImageFile, UnidentifiedImageError

from erayzer_core.data_prep.camera_convert import (
    intrinsics_after_center_square_crop,
    intrinsics_after_resize,
    intrinsics_fullres_to_pixels,
    nerf_gl_c2w_to_opencv_c2w,
)


def _iter_scene_dirs(dataset_root: str, image_subdir: str) -> Iterator[str]:
    skip = {".git", ".cache", "__pycache__"}
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        if "transforms.json" not in filenames:
            continue
        if image_subdir not in dirnames:
            continue
        yield dirpath


def _resize_geometry(
    image: Image.Image,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    target_height: int,
    patch_size: int,
    square_crop: bool,
) -> Tuple[Image.Image, float, float, float, float, int, int]:
    """Resize to fixed height, snap width to patch_size; optionally center-crop to a square."""
    W0, H0 = image.size
    resize_w = int(target_height / H0 * W0)
    resize_w = int(round(resize_w / patch_size) * patch_size)
    if resize_w < patch_size:
        resize_w = patch_size
    img = image.resize((resize_w, target_height), resample=Image.LANCZOS)
    fx, fy, cx, cy = intrinsics_after_resize(fx, fy, cx, cy, W0, H0, resize_w, target_height)

    if square_crop:
        min_side = min(target_height, resize_w)
        start_h = (target_height - min_side) // 2
        start_w = (resize_w - min_side) // 2
        img = img.crop((start_w, start_h, start_w + min_side, start_h + min_side))
        fx, fy, cx, cy = intrinsics_after_center_square_crop(fx, fy, cx, cy, start_w, start_h)
        final_w = final_h = min_side
    else:
        final_w, final_h = resize_w, target_height

    return img, fx, fy, cx, cy, final_w, final_h


def _process_scene(
    scene_dir: str,
    image_subdir: str,
    output_root: str,
    target_height: int,
    patch_size: int,
    square_crop: bool,
    use_opencv_c2w_direct: bool,
    overwrite: bool,
    allow_truncated_png: bool,
) -> Optional[str]:
    import numpy as np

    if allow_truncated_png:
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    scene_name = os.path.basename(os.path.normpath(scene_dir))
    out_scene = os.path.join(output_root, scene_name)
    manifest_path = os.path.join(out_scene, "manifest.json")
    rgb_dir = os.path.join(out_scene, "rgb")

    transforms_path = os.path.join(scene_dir, "transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Any] = json.load(f)

    frames_in = meta.get("frames") or []
    if not frames_in:
        print(f"Skip empty: {scene_dir}", file=sys.stderr)
        return None

    n_expected = len(frames_in)
    expected_cam = "opencv_c2w" if use_opencv_c2w_direct else "dl3dv_nerfstudio_to_opencv"

    if os.path.isfile(manifest_path) and not overwrite:
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                old: Dict[str, Any] = json.load(mf)
            old_frames = old.get("frames") or []
            if (
                len(old_frames) == n_expected
                and int(old.get("target_height", -1)) == target_height
                and int(old.get("patch_size", -1)) == patch_size
                and bool(old.get("square_crop")) == square_crop
                and old.get("camera_convention") == expected_cam
            ):
                return manifest_path
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
        shutil.rmtree(out_scene, ignore_errors=True)

    full_w = float(meta["w"])
    full_h = float(meta["h"])
    fl_x = float(meta["fl_x"])
    fl_y = float(meta["fl_y"])
    cx0 = float(meta["cx"])
    cy0 = float(meta["cy"])

    frames_out: List[Dict[str, Any]] = []
    os.makedirs(rgb_dir, exist_ok=True)

    for fr in frames_in:
        name = os.path.basename(fr["file_path"])
        src = os.path.join(scene_dir, image_subdir, name)
        if not os.path.isfile(src):
            print(f"Skip scene (missing {src})", file=sys.stderr)
            return None
        try:
            image = Image.open(src)
            image.load()
            image = image.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as e:
            print(f"Skip scene (bad image {src}): {e}", file=sys.stderr)
            shutil.rmtree(out_scene, ignore_errors=True)
            return None
        W_img, H_img = image.size
        fx, fy, cx, cy = intrinsics_fullres_to_pixels(
            fl_x, fl_y, cx0, cy0, full_w, full_h, W_img, H_img
        )
        image, fx, fy, cx, cy, fw, fh = _resize_geometry(
            image, fx, fy, cx, cy, target_height, patch_size, square_crop
        )
        out_name = name
        out_path = os.path.join(rgb_dir, out_name)
        image.save(out_path)

        tm = np.array(fr["transform_matrix"], dtype=np.float64)
        if use_opencv_c2w_direct:
            c2w = tm
        else:
            c2w = nerf_gl_c2w_to_opencv_c2w(tm)
        w2c = np.linalg.inv(c2w)

        frames_out.append(
            {
                "image": f"rgb/{out_name}",
                "w2c": w2c.tolist(),
                "c2w": c2w.tolist(),
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "w": fw,
                "h": fh,
            }
        )

    manifest = {
        "scene_name": scene_name,
        "source_scene_dir": os.path.abspath(scene_dir),
        "target_height": target_height,
        "patch_size": patch_size,
        "square_crop": square_crop,
        "camera_convention": "opencv_c2w" if use_opencv_c2w_direct else "dl3dv_nerfstudio_to_opencv",
        "frames": frames_out,
    }
    with open(manifest_path, "w", encoding="utf-8") as wf:
        json.dump(manifest, wf, indent=2)
    return manifest_path


def _process_scene_task(
    task: Tuple[str, str, str, int, int, bool, bool, bool, bool],
) -> Optional[str]:
    """Picklable entry for ProcessPoolExecutor (must stay at module scope)."""
    return _process_scene(*task)


def _default_num_workers() -> int:
    n = os.cpu_count() or 1
    return max(1, min(n, 8))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--out-list", type=str, required=True, help="List of absolute paths to manifest.json")
    p.add_argument("--image-subdir", type=str, default="images_4")
    p.add_argument("--target-height", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16, help="Snap resized width to multiple of this (ViT).")
    p.add_argument(
        "--square-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After resize, center-crop to min(H,W)xmin(H,W) (Open-Rayzer style). Default: on. Use --no-square-crop for rectangular WxH.",
    )
    p.add_argument(
        "--opencv-c2w-direct",
        action="store_true",
        help="If set, treat transform_matrix as already OpenCV c2w (skip Nerfstudio flip).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Scene-level processes. 1 = sequential. Default: min(8, CPU count).",
    )
    p.add_argument(
        "--allow-truncated-png",
        action="store_true",
        help="PIL: load truncated/corrupt PNGs (may have artifacts). Default: skip the whole scene on read errors.",
    )
    args = p.parse_args()
    square_crop: bool = bool(args.square_crop)

    root = os.path.abspath(args.dataset_root)
    out_root = os.path.abspath(args.output_root)
    os.makedirs(out_root, exist_ok=True)

    scene_dirs = sorted(_iter_scene_dirs(root, args.image_subdir))
    if args.max_scenes is not None:
        scene_dirs = scene_dirs[: args.max_scenes]

    tasks: List[Tuple[str, str, str, int, int, bool, bool, bool, bool]] = [
        (
            scene_dir,
            args.image_subdir,
            out_root,
            args.target_height,
            args.patch_size,
            square_crop,
            args.opencv_c2w_direct,
            args.overwrite,
            args.allow_truncated_png,
        )
        for scene_dir in scene_dirs
    ]

    num_workers = args.num_workers if args.num_workers is not None else _default_num_workers()
    num_workers = max(1, num_workers)

    if num_workers == 1:
        results: List[Optional[str]] = []
        for task in tqdm(tasks, desc="scenes", unit="scene", file=sys.stderr):
            results.append(_process_scene(*task))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            results = list(
                tqdm(
                    ex.map(_process_scene_task, tasks, chunksize=1),
                    total=len(tasks),
                    desc="scenes",
                    unit="scene",
                    file=sys.stderr,
                )
            )

    manifests = [os.path.abspath(mp) for mp in results if mp]

    parent = os.path.dirname(os.path.abspath(args.out_list))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out_list, "w", encoding="utf-8") as wf:
        wf.write("\n".join(manifests) + ("\n" if manifests else ""))
    print(f"Wrote {len(manifests)} manifests; list -> {args.out_list}", file=sys.stderr)


if __name__ == "__main__":
    main()
