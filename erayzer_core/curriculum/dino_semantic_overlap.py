"""
DINOv3-based semantic visual overlap for sequence-level curricula (E-RayZer Sec. 3.3).

Pairwise overlap: o_sem(i, j) = cos(phi(I_i), phi(I_j)) with L2-normalized embeddings.
Triplet statistic: o_tri(i, dt) = 1/2 * (o(i, i+dt) + o(i+dt, i+2dt)).
Per-sequence profile: O_u(dt) = mean over uniformly sampled valid i.

See: https://arxiv.org/abs/2512.10950
DINOv3 via Transformers: https://huggingface.co/docs/transformers/model_doc/dinov3
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


def load_dinov3_encoder(
    model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load HuggingFace DINOv3 image encoder + processor (inference)."""
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return model, processor


def _pool_embedding(outputs) -> torch.Tensor:
    """[B, D] from model forward; prefers pooler_output, else CLS token."""
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    hs = outputs.last_hidden_state
    return hs[:, 0, :]


@torch.inference_mode()
def embed_image_paths_dinov3(
    model,
    processor,
    image_paths: Sequence[str],
    device: torch.device,
    batch_size: int = 8,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    L2-normalized global embeddings [N, D] for ordered frames.
    Images are loaded RGB; missing files raise FileNotFoundError.
    """
    out_list: List[torch.Tensor] = []
    n = len(image_paths)
    for start in range(0, n, batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            outputs = model(**inputs)
        emb = _pool_embedding(outputs).float()
        emb = F.normalize(emb, dim=-1)
        out_list.append(emb.cpu())
    return torch.cat(out_list, dim=0)


def _valid_i_indices(num_frames: int, delta_t: int) -> int:
    """Largest i such that i + 2*delta_t < num_frames."""
    return num_frames - 2 * delta_t - 1


def build_semantic_overlap_profile(
    frame_embeddings: torch.Tensor,
    delta_t_values: Sequence[int],
    num_triplet_samples: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[int], List[float]]:
    """
    Args:
        frame_embeddings: [N, D], L2-normalized (CPU or CUDA).
        delta_t_values: candidate frame spacings (integers >= 1).
        num_triplet_samples: number of random starting indices i per delta_t.

    Returns:
        (used_delta_t_list, O_u_values) — skips delta_t with no valid triplets.
    """
    emb = frame_embeddings.detach().float().cpu()
    n = emb.shape[0]

    used_dt: List[int] = []
    used_ou: List[float] = []
    g = generator or torch.Generator()

    for dt in delta_t_values:
        if dt < 1:
            continue
        max_i = _valid_i_indices(n, dt)
        if max_i < 0:
            continue
        k = min(num_triplet_samples, max_i + 1)
        if k <= 0:
            continue
        if max_i + 1 <= k:
            indices = torch.arange(0, max_i + 1, dtype=torch.long)
        else:
            perm = torch.randperm(max_i + 1, generator=g)
            indices = perm[:k]

        i = indices
        j = i + dt
        k_idx = i + 2 * dt
        e_i = emb[i]
        e_j = emb[j]
        e_k = emb[k_idx]
        o1 = (e_i * e_j).sum(dim=-1)
        o2 = (e_j * e_k).sum(dim=-1)
        o_tri = 0.5 * (o1 + o2)
        used_dt.append(int(dt))
        used_ou.append(float(o_tri.mean().item()))

    return used_dt, used_ou


def manifest_to_image_paths(manifest_path: str, max_frames: Optional[int] = None) -> Tuple[str, str, List[str]]:
    """
    Returns (scene_name, pack_root, absolute_rgb_paths) for preprocess_dl3dv_training_pack.py manifest.json.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scene_name = data["scene_name"]
    frames = data["frames"]
    if max_frames is not None:
        frames = frames[:max_frames]
    root = os.path.dirname(os.path.abspath(manifest_path))
    paths = [os.path.join(root, fr["image"]) for fr in frames]
    return scene_name, root, paths


def profile_to_dict(
    scene_name: str,
    manifest_path: str,
    model_id: str,
    delta_t_values: Sequence[int],
    o_u_values: Sequence[float],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "scene_name": scene_name,
        "manifest_path": manifest_path,
        "model_id": model_id,
        "overlap_type": "dino_cosine",
        "delta_t_values": list(delta_t_values),
        "O_u": list(o_u_values),
        "meta": extra_meta or {},
    }
    return row
