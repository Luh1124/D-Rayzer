"""
Differentiable inverse warping for multi-view self-supervision (target depth drives sampling in source).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def fxfycxcy_to_K(fxfycxcy: torch.Tensor) -> torch.Tensor:
    """(B, 4) or (B, V, 4) -> (B, 3, 3) or (B, V, 3, 3) with fx, fy, cx, cy in pixel units."""
    if fxfycxcy.dim() == 2:
        b = fxfycxcy.shape[0]
        z = fxfycxcy.new_zeros(b, 3, 3)
        z[:, 0, 0] = fxfycxcy[:, 0]
        z[:, 1, 1] = fxfycxcy[:, 1]
        z[:, 0, 2] = fxfycxcy[:, 2]
        z[:, 1, 2] = fxfycxcy[:, 3]
        z[:, 2, 2] = 1.0
        return z
    b, v, _ = fxfycxcy.shape
    z = fxfycxcy.new_zeros(b, v, 3, 3)
    z[:, :, 0, 0] = fxfycxcy[:, :, 0]
    z[:, :, 1, 1] = fxfycxcy[:, :, 1]
    z[:, :, 0, 2] = fxfycxcy[:, :, 2]
    z[:, :, 1, 2] = fxfycxcy[:, :, 3]
    z[:, :, 2, 2] = 1.0
    return z


def disparity_to_depth_z(
    disparity: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Positive disparity -> positive Z along optical axis (relative scale)."""
    return 1.0 / (disparity.clamp_min(eps))


def inverse_warp(
    img_src: torch.Tensor,
    depth_tgt: torch.Tensor,
    c2w_tgt: torch.Tensor,
    c2w_src: torch.Tensor,
    K_tgt: torch.Tensor,
    K_src: torch.Tensor,
    min_depth: float = 1e-2,
    max_depth: float = 1e3,
    padding_mode: str = "zeros",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each target pixel, unproject with target depth, transform to source camera, sample source RGB.

    Args:
        img_src: (B, 3, H, W)
        depth_tgt: (B, 1, H, W) Z in target camera frame (OpenCV: +Z forward)
        c2w_*: (B, 4, 4) camera-to-world
        K_*: (B, 3, 3) intrinsics in pixel units for this resolution

    Returns:
        warped: (B, 3, H, W), valid: (B, 1, H, W) float mask
    """
    b, _, h, w = depth_tgt.shape
    device = depth_tgt.device
    dtype = depth_tgt.dtype

    u = torch.arange(w, device=device, dtype=dtype).view(1, 1, w).expand(b, h, w)
    vpix = torch.arange(h, device=device, dtype=dtype).view(1, h, 1).expand(b, h, w)

    z = depth_tgt[:, 0].clamp(min=min_depth, max=max_depth)
    fx_t = K_tgt[:, 0, 0].view(b, 1, 1)
    fy_t = K_tgt[:, 1, 1].view(b, 1, 1)
    cx_t = K_tgt[:, 0, 2].view(b, 1, 1)
    cy_t = K_tgt[:, 1, 2].view(b, 1, 1)

    xc = (u - cx_t) / fx_t * z
    yc = (vpix - cy_t) / fy_t * z
    pc = torch.stack([xc, yc, z], dim=1)
    ones = torch.ones(b, 1, h, w, device=device, dtype=dtype)
    ph = torch.cat([pc, ones], dim=1)

    n = h * w
    ph_flat = ph.view(b, 4, n)
    pw = torch.bmm(c2w_tgt, ph_flat).view(b, 4, h, w)

    w2c_src = torch.linalg.inv(c2w_src)
    pw_flat = pw.view(b, 4, n)
    p_src = torch.bmm(w2c_src, pw_flat).view(b, 4, h, w)
    x_s = p_src[:, 0]
    y_s = p_src[:, 1]
    z_s = p_src[:, 2].clamp(min=1e-6)

    fx_s = K_src[:, 0, 0].view(b, 1, 1)
    fy_s = K_src[:, 1, 1].view(b, 1, 1)
    cx_s = K_src[:, 0, 2].view(b, 1, 1)
    cy_s = K_src[:, 1, 2].view(b, 1, 1)

    u_s = fx_s * x_s / z_s + cx_s
    v_s = fy_s * y_s / z_s + cy_s

    grid_x = 2.0 * u_s / max(w - 1, 1) - 1.0
    grid_y = 2.0 * v_s / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    warped = F.grid_sample(
        img_src,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    valid = (
        (z_s > min_depth)
        & (z_s < max_depth)
        & (u_s >= 0)
        & (u_s <= w - 1)
        & (v_s >= 0)
        & (v_s <= h - 1)
    ).to(dtype).unsqueeze(1)

    return warped, valid


def input_target_reprojection_pairs(
    num_input_views: int,
    num_target_views: int,
    mode: str,
) -> List[Tuple[int, int]]:
    """
    Pairs (tgt_idx, src_idx): supervise **target** RGB with inverse warp from **input** RGB,
    using **target** depth (avoids photometric self-consistency on inputs that eases shortcutting).

    View layout: [input_0 .. input_{I-1}, target_0 .. target_{T-1}] (concatenated along V).
    """
    n_in = int(num_input_views)
    n_tgt = int(num_target_views)
    if n_in < 1 or n_tgt < 1:
        return []
    tgt_indices = list(range(n_in, n_in + n_tgt))
    src_indices = list(range(0, n_in))
    if mode in ("input_target_all", "cross_all", "all"):
        return [(t, s) for t in tgt_indices for s in src_indices]
    if mode == "input_target_nearest":
        # Each target only warps from the last input (minimal, still no tgt-as-src)
        return [(t, n_in - 1) for t in tgt_indices]
    if mode == "input_target_first":
        return [(t, 0) for t in tgt_indices]
    raise ValueError(
        f"Unknown input/target reproj_pair_mode: {mode!r} "
        f"(use input_target_all, input_target_nearest, or input_target_first)"
    )


def pairwise_reprojection_pairs(v: int, mode: str) -> List[Tuple[int, int]]:
    if v < 2:
        return []
    if mode == "consecutive":
        pairs: List[Tuple[int, int]] = []
        for i in range(v - 1):
            pairs.append((i, i + 1))
            pairs.append((i + 1, i))
        return pairs
    if mode == "all":
        return [(t, s) for t in range(v) for s in range(v) if t != s]
    raise ValueError(f"Unknown reproj_pair_mode: {mode}")


def multi_view_photometric_loss(
    images: torch.Tensor,
    disparity: torch.Tensor,
    c2w: torch.Tensor,
    fxfycxcy_px: torch.Tensor,
    *,
    eps: float = 1e-4,
    pair_mode: str = "consecutive",
    min_depth: float = 1e-2,
    max_depth: float = 1e3,
) -> torch.Tensor:
    """
    Args:
        images: (B, V, 3, H, W) in [0, 1]
        disparity: (B, V, 1, H, W) positive
        c2w: (B, V, 4, 4)
        fxfycxcy_px: (B, V, 4) fx, fy, cx, cy in pixel units at (H, W)
    """
    _b, v, _, h, w = images.shape
    pairs = pairwise_reprojection_pairs(v, pair_mode)
    if not pairs:
        return images.new_tensor(0.0)

    K = fxfycxcy_to_K(fxfycxcy_px)
    total = images.new_tensor(0.0)
    n_terms = 0

    for tgt, src in pairs:
        d_tgt = disparity_to_depth_z(disparity[:, tgt], eps=eps)
        wimg, m = inverse_warp(
            images[:, src],
            d_tgt,
            c2w[:, tgt],
            c2w[:, src],
            K[:, tgt],
            K[:, src],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        diff = (images[:, tgt] - wimg).abs().mean(dim=1, keepdim=True)
        per = (diff * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp(min=1.0)
        total = total + per.mean()
        n_terms += 1

    return total / max(n_terms, 1)


def multi_view_photometric_loss_input_target(
    images: torch.Tensor,
    disparity: torch.Tensor,
    c2w: torch.Tensor,
    fxfycxcy_px: torch.Tensor,
    num_input_views: int,
    num_target_views: int,
    *,
    eps: float = 1e-4,
    pair_mode: str = "input_target_all",
    min_depth: float = 1e-2,
    max_depth: float = 1e3,
) -> torch.Tensor:
    """
    E-RayZer-style split: only **target** views carry the photometric reprojection loss.
    For each pair (tgt, src): use **target** disparity → depth, sample **input** image src,
    match **target** RGB at tgt (inverse warp / Monodepth2 style).

    Expected ``V == num_input_views + num_target_views`` and tensor layout
    ``[..., 0:n_in]`` = inputs, ``[..., n_in:n_in+n_tgt]`` = targets.
    """
    _b, v, _, _h, _w = images.shape
    n_in = int(num_input_views)
    n_tgt = int(num_target_views)
    if v != n_in + n_tgt:
        raise ValueError(
            f"input/target split expects V={n_in}+{n_tgt}={n_in + n_tgt}, got V={v}"
        )
    pairs = input_target_reprojection_pairs(n_in, n_tgt, pair_mode)
    if not pairs:
        return images.new_tensor(0.0)

    K = fxfycxcy_to_K(fxfycxcy_px)
    total = images.new_tensor(0.0)
    n_terms = 0

    for tgt, src in pairs:
        d_tgt = disparity_to_depth_z(disparity[:, tgt], eps=eps)
        wimg, m = inverse_warp(
            images[:, src],
            d_tgt,
            c2w[:, tgt],
            c2w[:, src],
            K[:, tgt],
            K[:, src],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        diff = (images[:, tgt] - wimg).abs().mean(dim=1, keepdim=True)
        per = (diff * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp(min=1.0)
        total = total + per.mean()
        n_terms += 1

    return total / max(n_terms, 1)


def disparity_smoothness_loss(
    disparity: torch.Tensor,
    image_rgb: torch.Tensor,
) -> torch.Tensor:
    """
    Edge-aware smoothness on disparity maps.

    disparity: (B, V, 1, H, W)
    image_rgb: (B, V, 3, H, W)
    """
    b, v, _, h, w = disparity.shape
    loss = disparity.new_tensor(0.0)
    count = 0
    for bi in range(b):
        for vi in range(v):
            d = disparity[bi, vi]
            im = image_rgb[bi, vi]
            gd_x = torch.abs(d[:, :, :, :-1] - d[:, :, :, 1:])
            gd_y = torch.abs(d[:, :, :-1, :] - d[:, :, 1:, :])
            gi_x = torch.mean(torch.abs(im[:, :, :, :-1] - im[:, :, :, 1:]), dim=0, keepdim=True)
            gi_y = torch.mean(torch.abs(im[:, :, :-1, :] - im[:, :, 1:, :]), dim=0, keepdim=True)
            loss = loss + (gd_x * torch.exp(-gi_x)).mean() + (gd_y * torch.exp(-gi_y)).mean()
            count += 1
    return loss / max(count, 1)


def scale_intrinsics_to_resolution(
    fxfycxcy: torch.Tensor,
    wh_src: torch.Tensor,
    h_out: int,
    w_out: int,
) -> torch.Tensor:
    """
    Scale fx,fy,cx,cy from original (W,H) in wh_src to (w_out, h_out).

    fxfycxcy: (B, V, 4) or (V, 4)
    wh_src: (B, V, 2) with [W_orig, H_orig] or (V, 2)
    """
    if fxfycxcy.dim() == 2:
        fxfycxcy = fxfycxcy.unsqueeze(0)
    if wh_src.dim() == 2:
        wh_src = wh_src.unsqueeze(0)
    b, v, _ = fxfycxcy.shape
    out = fxfycxcy.clone()
    w0 = wh_src[..., 0].clamp(min=1.0)
    h0 = wh_src[..., 1].clamp(min=1.0)
    sx = float(w_out) / w0
    sy = float(h_out) / h0
    out[..., 0] *= sx
    out[..., 1] *= sy
    out[..., 2] *= sx
    out[..., 3] *= sy
    return out
