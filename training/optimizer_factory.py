"""Training optimizers: AdamW-only or Muon + AdamW hybrid (PyTorch 2.11+)."""

from __future__ import annotations

from typing import Any, List, Union

import torch
from torch import nn


def uses_multiple_optimizers(module: nn.Module, tr: Any) -> bool:
    """True if muon_hybrid will return two optimizers (Lightning needs manual optimization)."""
    name = str(tr.get("optimizer", "adamw")).lower().replace("-", "_")
    if name not in ("muon_hybrid", "muonhybrid", "adamw_muon"):
        return False
    if getattr(torch.optim, "Muon", None) is None:
        return False
    mu_p, adam_p = split_muon_adamw_params(module)
    return bool(mu_p and adam_p)


def split_muon_adamw_params(module: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split params: 2D weights for Muon; biases, norms, embeddings, etc. for AdamW."""
    muon: list[nn.Parameter] = []
    adam: list[nn.Parameter] = []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


def build_optimizers(
    module: nn.Module,
    tr: Any,
    lr: float,
) -> Union[torch.optim.Optimizer, List[torch.optim.Optimizer]]:
    """
    Args:
        tr: ``cfg.training`` (EasyDict or dict).

    Modes:
        - ``optimizer: adamw`` -- AdamW on all parameters (default).
        - ``optimizer: muon_hybrid`` -- Muon on 2D weights, AdamW on the rest (PyTorch >= 2.11).
    """
    name = str(tr.get("optimizer", "adamw")).lower().replace("-", "_")
    wd = float(tr.get("weight_decay", 0.05))
    beta1 = float(tr.get("beta1", 0.9))
    beta2 = float(tr.get("beta2", 0.95))

    if name == "adamw":
        return torch.optim.AdamW(
            module.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=(beta1, beta2),
        )

    if name in ("muon_hybrid", "muonhybrid", "adamw_muon"):
        Muon = getattr(torch.optim, "Muon", None)
        if Muon is None:
            raise RuntimeError(
                "training.optimizer is muon_hybrid but torch.optim.Muon is missing "
                "(need PyTorch >= 2.11). Upgrade PyTorch or set training.optimizer: adamw."
            )
        mu_p, adam_p = split_muon_adamw_params(module)
        if not mu_p:
            return torch.optim.AdamW(
                module.parameters(),
                lr=lr,
                weight_decay=wd,
                betas=(beta1, beta2),
            )

        muon_lr = float(tr["muon_lr"]) if tr.get("muon_lr") is not None else lr
        muon_wd = float(tr["muon_weight_decay"]) if tr.get("muon_weight_decay") is not None else wd
        opt_muon = Muon(
            mu_p,
            lr=muon_lr,
            weight_decay=muon_wd,
            momentum=float(tr.get("muon_momentum", 0.95)),
            nesterov=bool(tr.get("muon_nesterov", True)),
            ns_steps=int(tr.get("muon_ns_steps", 5)),
            eps=float(tr.get("muon_eps", 1e-7)),
            adjust_lr_fn=_parse_adjust_lr_fn(tr.get("muon_adjust_lr_fn")),
        )
        if not adam_p:
            return opt_muon
        opt_adam = torch.optim.AdamW(
            adam_p,
            lr=lr,
            weight_decay=wd,
            betas=(beta1, beta2),
        )
        return [opt_muon, opt_adam]

    raise ValueError(f"Unknown training.optimizer: {name!r} (supported: adamw, muon_hybrid)")


def _parse_adjust_lr_fn(raw: Any) -> str | None:
    if raw in (None, "", "null", "none", "None"):
        return None
    s = str(raw).strip()
    if s.lower() in ("null", "none"):
        return None
    return s
