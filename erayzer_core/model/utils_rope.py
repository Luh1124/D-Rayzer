"""
Rotary positional embeddings (RoPE) for attention Q/K.

- **1D RoPE**: standard LLaMA-style; one index per token along the sequence.
- **2D M-RoPE** (Qwen2-VL-style multimodal RoPE, see e.g.
  https://medium.com/everyday-ai/qwen2-vls-rope-variant-m-rope-8cfcc4672ea9 ):
  split ``head_dim`` into two halves; the first half is rotated with frequencies
  tied to the **row** index, the second half with frequencies tied to the **column**
  index. Prefix tokens (camera / register) use (0, 0).
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin_table(
    max_pos: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables of shape ``(max_pos + 1, dim)`` for positions ``0 .. max_pos``."""
    assert dim % 2 == 0, f"RoPE requires even dim, got {dim}"
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    t = torch.arange(max_pos + 1, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


def rope_cos_sin(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1D RoPE tables of shape ``(seq_len, head_dim)`` for positions ``0 .. seq_len-1``."""
    return rope_cos_sin_table(seq_len - 1, head_dim, device, dtype, theta)


def apply_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (batch, seq, n_heads, head_dim)
    cos, sin: (seq, head_dim)
    """
    c = cos.unsqueeze(0).unsqueeze(2)
    s = sin.unsqueeze(0).unsqueeze(2)
    q = (q * c) + (rotate_half(q) * s)
    k = (k * c) + (rotate_half(k) * s)
    return q, k


def mrpe_token_positions(
    batch: int,
    seq_len: int,
    num_prefix: int,
    hh: int,
    ww: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integer row/col indices per token for E-RayZer layouts.

    - **Frame attention** ``(b*v, n, d)`` with ``n = num_prefix + hh*ww``: use local layout.
    - **Global attention** ``(b, v*n, d)``: repeat the same local grid per view.

    Prefix tokens (camera + registers) use (0, 0).
    """
    n = num_prefix + hh * ww
    pos_h = torch.zeros(seq_len, dtype=torch.long, device=device)
    pos_w = torch.zeros(seq_len, dtype=torch.long, device=device)

    if seq_len == n:
        for local in range(num_prefix, n):
            pi = local - num_prefix
            pos_h[local] = pi // ww
            pos_w[local] = pi % ww
    elif n > 0 and seq_len % n == 0:
        for t in range(seq_len):
            local = t % n
            if local >= num_prefix:
                pi = local - num_prefix
                pos_h[t] = pi // ww
                pos_w[t] = pi % ww
    else:
        raise ValueError(
            f"M-RoPE: seq_len={seq_len} is not n={n} nor a multiple of n (patch grid mismatch)."
        )

    pos_h = pos_h.unsqueeze(0).expand(batch, seq_len)
    pos_w = pos_w.unsqueeze(0).expand(batch, seq_len)
    return pos_h, pos_w


def apply_rope_half(
    x_half: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    x_half: (B, L, nh, half_dim)
    cos_table, sin_table: (max_pos+1, half_dim)
    positions: (B, L) int64
    """
    c = cos_table[positions]
    s = sin_table[positions]
    c = c.unsqueeze(2)
    s = s.unsqueeze(2)
    return (x_half * c) + (rotate_half(x_half) * s)


def apply_rope_qk_2d_mrpe(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_h: torch.Tensor,
    pos_w: torch.Tensor,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    M-RoPE for vision: first half of head_dim encodes row, second half column.

    q, k: (B, L, nh, head_dim); head_dim must be divisible by 4 (each half even).
    pos_h, pos_w: (B, L) int64
    """
    dh = q.shape[-1]
    if dh % 2 != 0:
        raise ValueError(f"M-RoPE requires even head_dim, got {dh}")
    dh2 = dh // 2
    if dh2 % 2 != 0:
        raise ValueError(f"M-RoPE requires head_dim divisible by 4, got {dh}")

    mh = int(pos_h.max().item())
    mw = int(pos_w.max().item())
    device = q.device
    dtype = q.dtype

    cos_h, sin_h = rope_cos_sin_table(mh, dh2, device, dtype, theta)
    cos_w, sin_w = rope_cos_sin_table(mw, dh2, device, dtype, theta)

    qh, qw = q[..., :dh2], q[..., dh2:]
    kh, kw = k[..., :dh2], k[..., dh2:]

    qh = apply_rope_half(qh, cos_h, sin_h, pos_h)
    qw = apply_rope_half(qw, cos_w, sin_w, pos_w)
    kh = apply_rope_half(kh, cos_h, sin_h, pos_h)
    kw = apply_rope_half(kw, cos_w, sin_w, pos_w)

    q_out = torch.cat([qh, qw], dim=-1)
    k_out = torch.cat([kh, kw], dim=-1)
    return q_out, k_out
