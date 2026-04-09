from __future__ import annotations

import importlib
import math
import os
from contextlib import nullcontext
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from lightning.pytorch import LightningModule

import erayzer_core  # noqa: F401 - side effect: registers model/utils aliases for ERayZer

from erayzer_core.model.erayzer import _tokenizer_hw_and_grid
from training.config_loader import load_merged_edict
from training.optimizer_factory import build_optimizers, uses_multiple_optimizers
from erayzer_core.model.reprojection_warp import (
    disparity_smoothness_loss,
    multi_view_photometric_loss,
    multi_view_photometric_loss_input_target,
    scale_intrinsics_to_resolution,
)


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    tr: Any,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Optional cosine decay (``training.lr_schedule: cosine``), with linear warmup using
    ``training.warmup`` steps when set (same key as in ``config/optimizer/adamw.yaml``).
    """
    T_total = tr.get("lr_cosine_t_max_steps")
    if T_total is None:
        T_total = int(tr.get("max_fwdbwd_passes", 152_000))
    T_total = max(1, int(T_total))
    warmup = int(tr.get("warmup", 0) or 0)
    warmup = max(0, min(warmup, T_total - 1))
    min_ratio = float(tr.get("lr_cosine_min_ratio", 0.0))
    base = float(optimizer.param_groups[0]["lr"])
    # Paper: cosine decay to zero at end → lr_cosine_min_ratio: 0
    eta_min = max(0.0, base * min_ratio)
    T_cosine = max(1, T_total - warmup)
    # Warmup starts at base * start_factor (default 1e-4 of peak; avoid 1e-8 which looks like "dead" LR on W&B).
    wstart = float(tr.get("lr_warmup_start_factor", 1e-4))
    wstart = max(1e-12, wstart)

    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_cosine, eta_min=eta_min
    )
    if warmup <= 0:
        return cos

    lin = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=wstart, end_factor=1.0, total_iters=warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [lin, cos], milestones=[warmup]
    )


def _ensure_training_edict(cfg: edict) -> None:
    t = cfg.get("training")
    if t is None:
        cfg.training = edict({})
    elif isinstance(t, dict) and not isinstance(t, edict):
        cfg.training = edict(t)


def _resize_multiview_images(
    image: torch.Tensor,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """[B,V,C,H,W] or [B,C,H,W] -> [B,V,C,out_h,out_w] in [0,1]."""
    if image.dim() == 4:
        image = image.unsqueeze(1)
    b, v, c, h0, w0 = image.shape
    x = image.reshape(b * v, c, h0, w0)
    x = F.interpolate(x, size=(out_h, out_w), mode="bicubic", align_corners=False)
    return x.reshape(b, v, c, out_h, out_w).clamp(0.0, 1.0)


def _rgb_l1_on_views(pred: torch.Tensor, gt: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """pred, gt: [B,V,3,H,W] in [0,1]."""
    return F.l1_loss(pred[:, start:end], gt[:, start:end])


def _batch_psnr(pred: torch.Tensor, gt: torch.Tensor, start: int, end: int) -> torch.Tensor:
    x = pred[:, start:end].clamp(0, 1).float()
    y = gt[:, start:end].clamp(0, 1).float()
    mse = F.mse_loss(x, y)
    if mse.item() <= 0:
        return pred.new_tensor(99.0)
    return pred.new_tensor(10.0 * math.log10(1.0 / max(mse.item(), 1e-10)))


class PlaceholderTrainModule(LightningModule):
    """Tiny module for dataloader smoke tests without loading E-RayZer."""

    def __init__(self, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self._dummy = nn.Parameter(torch.zeros(1))

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._dummy.sum() + 0.0 * batch["image"].mean()
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ERayZerTrainModule(LightningModule):
    """
    Full Lightning training for E-RayZer:

    - ``geometry_output: gaussians``: RGB L1 on rendered views (targets-only by default).
    - ``geometry_output: depth_reprojection``: input/target cross-view photometric loss.

    Uses ``training.skip_heavy_gaussian_export`` to avoid per-batch GaussianModel deepcopy during training.
    """

    def __init__(
        self,
        config_paths: Union[str, Sequence[str]],
        lr: Optional[float] = None,
        geometry_output_override: Optional[str] = None,
        optimizer_override: Optional[str] = None,
    ) -> None:
        super().__init__()
        paths = [config_paths] if isinstance(config_paths, str) else list(config_paths)
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(p)

        self.cfg = load_merged_edict(paths)
        _ensure_training_edict(self.cfg)
        if geometry_output_override is not None:
            self.cfg.model.geometry_output = geometry_output_override
        if optimizer_override is not None:
            self.cfg.training.optimizer = str(optimizer_override).strip()

        self._depth_mode = self.cfg.model.get("geometry_output") == "depth_reprojection"
        t = self.cfg.training
        self.lr = float(lr if lr is not None else t.get("lr", 1e-4))

        self.save_hyperparameters(
            {
                "config_paths": paths,
                "lr": self.lr,
                "geometry_output": self.cfg.model.get("geometry_output", "gaussians"),
            }
        )

        self.cfg.inference = False
        self.cfg.evaluation = False
        self.cfg.training.skip_heavy_gaussian_export = bool(t.get("skip_heavy_gaussian_export", True))

        module_name, class_name = self.cfg.model.class_name.rsplit(".", 1)
        ModelClass = importlib.import_module(module_name).__dict__[class_name]
        self.erayzer = ModelClass(self.cfg)

        tr = self.cfg.training
        self.reproj_w = float(tr.get("reproj_loss_weight", 1.0))
        self.smooth_w = float(tr.get("reproj_smooth_weight", 0.0))
        self.pair_mode = str(tr.get("reproj_pair_mode", "consecutive"))
        self.min_depth = float(tr.get("reproj_min_depth", 1e-2))
        self.max_depth = float(tr.get("reproj_max_depth", 1e3))
        self.use_gt_K = bool(tr.get("reproj_use_gt_intrinsics", False))
        self.use_gt_pose = bool(tr.get("reproj_use_gt_pose", False))
        self.reproj_input_target_split = bool(tr.get("reproj_use_input_target_split", True))
        self.num_input_views = int(tr.get("num_input_views", 5))
        self.num_target_views = int(tr.get("num_target_views", 5))
        self.train_h, self.train_w, _, _ = _tokenizer_hw_and_grid(self.cfg)

        self.l2_loss_weight = float(tr.get("l2_loss_weight", 1.0))
        self.gs_supervise_targets_only = bool(tr.get("gs_supervise_target_views_only", True))

        amp_dtype = str(tr.get("amp_dtype", "bf16")).lower()
        self._amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.use_amp = bool(tr.get("use_amp", True))

        dep = self.cfg.model.get("depth_output") or {}
        self.depth_eps = float(dep.get("eps", 1e-4)) if isinstance(dep, dict) else float(getattr(dep, "eps", 1e-4))

        self._manual_optimization = uses_multiple_optimizers(self.erayzer, self.cfg.training)
        self.automatic_optimization = not self._manual_optimization
        self._grad_clip_norm = float(tr.get("grad_clip_norm", 0.0))
        # When True, grad clip + optional skip (paper) run inside the module; Trainer must not clip again.
        self._uses_internal_grad_clip = True
        # Successful optimizer.step() count (paper: LR + curriculum align with effective updates, not skipped batches).
        self.register_buffer("_effective_update_count", torch.zeros((), dtype=torch.long))
        # Same scheduler objects as in configure_optimizers (manual opt must step these; lr_schedulers() can be unset early).
        self._step_lr_schedulers_list: Optional[list] = None

    def _autocast(self):
        if not self.use_amp or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype)

    def configure_optimizers(self):
        opts = build_optimizers(self.erayzer, self.cfg.training, self.lr)
        tr = self.cfg.training
        mode = str(tr.get("lr_schedule", "constant")).lower().replace("-", "_")
        if mode in ("cosine", "cosine_annealing"):
            if isinstance(opts, list):
                self._step_lr_schedulers_list = [_build_lr_scheduler(o, tr) for o in opts]
                scheds = [
                    {"scheduler": s, "interval": "step", "frequency": 1}
                    for s in self._step_lr_schedulers_list
                ]
                return opts, scheds
            s0 = _build_lr_scheduler(opts, tr)
            self._step_lr_schedulers_list = [s0]
            return {
                "optimizer": opts,
                "lr_scheduler": {
                    "scheduler": s0,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        if isinstance(opts, list):
            return opts, []
        return opts

    def lr_scheduler_step(self, scheduler, metric, *args, **kwargs) -> None:
        """Manual opt steps schedulers in training_step; auto opt skips LR when grad step was skipped."""
        if self._manual_optimization:
            return
        if getattr(self, "_skip_lr_scheduler", True):
            return
        return super().lr_scheduler_step(scheduler, metric, *args, **kwargs)

    def _bump_effective_updates(self) -> None:
        self._effective_update_count.add_(1)

    def _allowed_gradnorm_threshold(self) -> Optional[float]:
        """Pre-clip ||g|| gate: skip optimizer step if above this. None disables the gate (clip still applies)."""
        raw = self.cfg.training.get("allowed_gradnorm_factor", 5.0)
        if raw is None:
            return None
        if isinstance(raw, str) and raw.strip().lower() in ("none", "null", "off", "false"):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _manual_lr_scheduler_step_all(self) -> None:
        """Lightning does not step LR schedulers for manual optimization; call after successful opt.step()."""
        schedulers = getattr(self, "_step_lr_schedulers_list", None)
        if not schedulers:
            schedulers = self.lr_schedulers()
            if schedulers is None:
                return
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
        for sch in schedulers:
            sch.step()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, *args, **kwargs):
        """AdamW-only path: paper grad-norm gate (skip if > allowed_gradnorm_factor) then clip to grad_clip_norm."""
        if self._manual_optimization:
            return
        self._skip_lr_scheduler = True
        optimizer_closure()
        # list(): .parameters() is a one-shot iterator; two clip_grad_norm_ calls would exhaust it.
        params = list(self.erayzer.parameters())
        gn = torch.nn.utils.clip_grad_norm_(params, float("inf"))
        gn_f = float(gn.detach().cpu()) if isinstance(gn, torch.Tensor) else float(gn)
        thr = self._allowed_gradnorm_threshold()
        self.log("train/grad_norm_pre_clip", gn_f, prog_bar=False, reduce_fx="mean")
        if thr is not None and gn_f > thr:
            optimizer.zero_grad()
            self.log("train/skip_grad_step", 1.0, prog_bar=False, reduce_fx="mean")
            self.log("train/optimizer_stepped", 0.0, prog_bar=False, reduce_fx="mean")
            return
        if self._grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self._grad_clip_norm)
        optimizer.step()
        self._bump_effective_updates()
        self._skip_lr_scheduler = False
        self.log("train/optimizer_stepped", 1.0, prog_bar=False, reduce_fx="mean")
        self.log(
            "train/effective_updates",
            float(self._effective_update_count.item()),
            prog_bar=False,
            reduce_fx="mean",
        )

    def training_step(self, batch: dict, batch_idx: int) -> Union[torch.Tensor, None]:
        if self._depth_mode:
            loss = self._step_depth(batch, train=True)
        else:
            loss, _ = self._step_gs(batch, train=True)
            self.log(
                "train/rgb_l1",
                loss / max(self.l2_loss_weight, 1e-8),
                prog_bar=False,
            )

        self.log("train/loss", loss, prog_bar=True)

        if self._manual_optimization:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
            self.manual_backward(loss)
            params = list(self.erayzer.parameters())
            gn = torch.nn.utils.clip_grad_norm_(params, float("inf"))
            gn_f = float(gn.detach().cpu()) if isinstance(gn, torch.Tensor) else float(gn)
            thr = self._allowed_gradnorm_threshold()
            self.log("train/grad_norm_pre_clip", gn_f, prog_bar=False, reduce_fx="mean")
            if thr is not None and gn_f > thr:
                for opt in opts:
                    opt.zero_grad(set_to_none=True)
                self.log("train/skip_grad_step", 1.0, prog_bar=False, reduce_fx="mean")
                self.log("train/optimizer_stepped", 0.0, prog_bar=False, reduce_fx="mean")
                return None
            if self._grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, self._grad_clip_norm)
            for opt in opts:
                opt.step()
            self._bump_effective_updates()
            self._manual_lr_scheduler_step_all()
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            self.log("train/optimizer_stepped", 1.0, prog_bar=False, reduce_fx="mean")
            self.log(
                "train/effective_updates",
                float(self._effective_update_count.item()),
                prog_bar=False,
                reduce_fx="mean",
            )
            return None
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        if self._depth_mode:
            loss = self._step_depth(batch, train=False)
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        else:
            loss, psnr = self._step_gs(batch, train=False)
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)
            self.log("val/psnr_approx", psnr, prog_bar=False, sync_dist=True)

    def _step_depth(self, batch: dict, train: bool) -> torch.Tensor:
        images_rs = _resize_multiview_images(
            batch["image"].to(self.device),
            self.train_h,
            self.train_w,
        )
        v = images_rs.shape[1]
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            if self.reproj_input_target_split:
                n_in, n_tgt = self.num_input_views, self.num_target_views
                if v != n_in + n_tgt:
                    raise ValueError(
                        f"reproj expects V={n_in}+{n_tgt}, got {v}"
                    )
            elif v < 2:
                raise ValueError("depth training needs V>=2")

            with self._autocast():
                out = self.erayzer({"image": images_rs})

            if self.use_gt_K and "intrinsics" in batch and "wh" in batch:
                Kpx = scale_intrinsics_to_resolution(
                    batch["intrinsics"].to(self.device).float(),
                    batch["wh"].to(self.device).float(),
                    self.train_h,
                    self.train_w,
                )
            else:
                Kpx = out.fxfycxcy_pixels

            if self.use_gt_pose and "c2w" in batch:
                c2w = batch["c2w"].to(self.device).float()[:, :v]
            else:
                c2w = out.c2w

            if self.reproj_input_target_split:
                pair_mode = self.pair_mode
                if pair_mode in ("consecutive", "all"):
                    pair_mode = "input_target_all"
                l_photo = multi_view_photometric_loss_input_target(
                    images_rs,
                    out.pred_disparity,
                    c2w,
                    Kpx,
                    self.num_input_views,
                    self.num_target_views,
                    eps=self.depth_eps,
                    pair_mode=pair_mode,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )
            else:
                l_photo = multi_view_photometric_loss(
                    images_rs,
                    out.pred_disparity,
                    c2w,
                    Kpx,
                    eps=self.depth_eps,
                    pair_mode=self.pair_mode,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )

            if self.smooth_w > 0.0:
                if self.reproj_input_target_split:
                    disp_s = out.pred_disparity[
                        :, self.num_input_views : self.num_input_views + self.num_target_views
                    ]
                    img_s = images_rs[
                        :, self.num_input_views : self.num_input_views + self.num_target_views
                    ]
                    l_smooth = disparity_smoothness_loss(disp_s, img_s)
                else:
                    l_smooth = disparity_smoothness_loss(out.pred_disparity, images_rs)
            else:
                l_smooth = images_rs.new_tensor(0.0)

            loss = self.reproj_w * l_photo + self.smooth_w * l_smooth

        if train:
            self.log("train/reproj_photo", l_photo.detach(), prog_bar=False)
            self.log("train/reproj_smooth", l_smooth.detach(), prog_bar=False)
        return loss

    def _step_gs(self, batch: dict, train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        images_rs = _resize_multiview_images(
            batch["image"].to(self.device),
            self.train_h,
            self.train_w,
        )
        v = images_rs.shape[1]
        n_in, n_tgt = self.num_input_views, self.num_target_views

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            with self._autocast():
                payload = {"image": images_rs}
                if (
                    self.reproj_input_target_split
                    and v == n_in + n_tgt
                    and n_in > 0
                    and n_tgt > 0
                ):
                    payload["num_input_views"] = n_in
                    payload["num_target_views"] = n_tgt
                out = self.erayzer(payload)

            pred = out.render.float().clamp(0.0, 1.0)
            gt = images_rs

            if pred.shape[1] == n_tgt and v >= n_in + n_tgt:
                rgb_l1 = F.l1_loss(pred, gt[:, n_in : n_in + n_tgt])
            elif self.gs_supervise_targets_only and v >= n_in + n_tgt:
                start, end = n_in, n_in + n_tgt
                rgb_l1 = _rgb_l1_on_views(pred, gt, start, end)
            else:
                start, end = 0, v
                rgb_l1 = _rgb_l1_on_views(pred, gt, start, end)
            loss = self.l2_loss_weight * rgb_l1
            if pred.shape[1] == n_tgt and v >= n_in + n_tgt:
                psnr = _batch_psnr(pred, gt[:, n_in : n_in + n_tgt], 0, n_tgt)
            else:
                psnr = _batch_psnr(pred, gt, start, end)

        return loss, psnr.detach()

