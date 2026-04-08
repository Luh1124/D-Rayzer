#!/usr/bin/env python3
"""
Offline semantic overlap profiles O_u(Δt) for curriculum sampling (E-RayZer paper, Sec. 3.3).

Reads training packs from preprocess_dl3dv_training_pack.py (manifest.json + rgb/).
Uses DINOv3 global embeddings (HuggingFace Transformers).

If you change preprocessing (e.g. square-crop vs rectangular, target height), re-run this script
on the new ``--manifest-list`` so embeddings match training pixels. Do not use ``--skip-existing``
unless outputs are already from the same preprocess; or use a fresh ``--output-dir``.

Single GPU:
  python scripts/compute_dino_semantic_overlap_profiles.py \\
    --manifest-list /data/dl3dv_train_h256/all_manifests.txt \\
    --output-dir /data/curriculum_dino_profiles \\
    --device cuda:0

Multi-GPU (each rank loads the model on cuda:LOCAL_RANK; manifest lines are sharded by index):
  torchrun --standalone --nproc_per_node=4 scripts/compute_dino_semantic_overlap_profiles.py \\
    --manifest-list /data/dl3dv_train_h256/all_manifests.txt \\
    --output-dir /data/curriculum_dino_profiles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _distributed_info() -> Tuple[int, int, int]:
    """(rank, world_size, local_rank). Single process: (0, 1, 0)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute DINOv3 semantic overlap profiles per scene.")
    p.add_argument(
        "--manifest-list",
        type=str,
        required=True,
        help="Text file: one absolute path per line to manifest.json (training pack).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write <scene_name>.json profiles.",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HuggingFace model id (DINOv3 ViT).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device when not using torchrun (WORLD_SIZE=1). Ignored under multi-GPU; uses LOCAL_RANK.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--delta-t-min", type=int, default=1)
    p.add_argument("--delta-t-max", type=int, default=128)
    p.add_argument("--delta-t-step", type=int, default=1)
    p.add_argument(
        "--num-triplet-samples",
        type=int,
        default=64,
        help="Random starting indices i per Δt (paper: small uniform sample).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames per scene (first N only; for debugging).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenes whose output JSON already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _distributed_info()

    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from tqdm import tqdm

    from erayzer_core.curriculum.dino_semantic_overlap import (
        build_semantic_overlap_profile,
        embed_image_paths_dinov3,
        load_dinov3_encoder,
        manifest_to_image_paths,
        profile_to_dict,
    )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    if world_size > 1:
        if not torch.cuda.is_available():
            print("Multi-GPU mode requires CUDA.", file=sys.stderr)
            sys.exit(1)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    with open(args.manifest_list, "r", encoding="utf-8") as f:
        all_manifests: List[str] = [ln.strip() for ln in f if ln.strip()]

    manifests = [all_manifests[i] for i in range(len(all_manifests)) if i % world_size == rank]

    delta_ts = list(range(args.delta_t_min, args.delta_t_max + 1, args.delta_t_step))
    if not delta_ts:
        print("No delta_t values in range; check --delta-t-min / --delta-t-max.", file=sys.stderr)
        sys.exit(1)

    if rank == 0:
        mode = f"torchrun x{world_size}" if world_size > 1 else str(device)
        print(
            f"Loading {args.model_id} ({args.dtype}) | {mode} | "
            f"this rank: {len(manifests)}/{len(all_manifests)} manifests",
            file=sys.stderr,
        )

    model, processor = load_dinov3_encoder(args.model_id, device=device, dtype=dtype)
    gen = torch.Generator().manual_seed(args.seed + rank)

    done = 0
    skipped = 0
    pbar_disable = world_size > 1 and rank != 0
    for manifest_path in tqdm(
        manifests,
        desc="scenes" if world_size == 1 else f"rank{rank}",
        unit="scene",
        file=sys.stderr,
        disable=pbar_disable,
    ):
        if not os.path.isfile(manifest_path):
            print(f"[rank{rank}] Skip missing file: {manifest_path}", file=sys.stderr)
            skipped += 1
            continue
        scene_name, _root, paths = manifest_to_image_paths(manifest_path, max_frames=args.max_frames)
        out_path = os.path.join(args.output_dir, f"{scene_name}.json")
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue
        if len(paths) < 3:
            print(f"[rank{rank}] Skip short scene ({len(paths)} frames): {scene_name}", file=sys.stderr)
            skipped += 1
            continue

        emb = embed_image_paths_dinov3(
            model,
            processor,
            paths,
            device=device,
            batch_size=args.batch_size,
            dtype=dtype,
        )
        used_dt, used_ou = build_semantic_overlap_profile(
            emb,
            delta_ts,
            num_triplet_samples=args.num_triplet_samples,
            generator=gen,
        )
        meta = {
            "num_frames": len(paths),
            "num_triplet_samples_per_delta": args.num_triplet_samples,
            "seed": args.seed,
            "rank": rank,
            "world_size": world_size,
            "delta_t_min": args.delta_t_min,
            "delta_t_max": args.delta_t_max,
            "delta_t_step": args.delta_t_step,
        }
        row = profile_to_dict(
            scene_name,
            manifest_path,
            args.model_id,
            used_dt,
            used_ou,
            extra_meta=meta,
        )
        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(row, wf, indent=2)
        done += 1

    prefix = f"[rank{rank}] " if world_size > 1 else ""
    print(f"{prefix}wrote {done} profiles, skipped {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
