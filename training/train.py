#!/usr/bin/env python3
"""
Baseline PyTorch Lightning trainer (placeholder model).

  pip install -r requirements-train.txt
  python -m training.train --manifest-list /data/dl3dv_train_h256/all_manifests.txt

DINO curriculum (E-RayZer paper Sec. 3.3, multi-frame batches):

  python -m training.train \\
    --manifest-list /data/dl3dv_train_h256/all_manifests.txt \\
    --use-dino-curriculum \\
    --dino-profile-dir /data/dl3dv_train_h256/dino_overlap_profiles \\
    --curriculum-ramp-steps 86000 --num-views 10

Preprocess + profiles first:
  python scripts/preprocess_dl3dv_training_pack.py ...
  python scripts/compute_dino_semantic_overlap_profiles.py ...

Relative disparity + differentiable reprojection (no GS render), multi-view only.
  Default: input/target split (see config reproj_use_input_target_split); batch order must be
  [inputs | targets] with V = num_input_views + num_target_views (e.g. 5+5=10).
  python -m training.train --manifest-list ... --use-dino-curriculum \\
    --geometry-output depth_reprojection --num-views 10 --batch-size 1 --max-steps 1000

Multiple YAML configs (deep merge, later wins), e.g. base + experiment overrides:
  python -m training.train ... --config config/experiments/rope_rms.yaml

By default the trainer prepends ``config/model``, ``data``, ``optimizer/adamw``, ``train``; each ``--config`` merges after that (see ``DEFAULT_ERAYZER_CONFIG_REL_PATHS``). Use ``--no-default-config`` if you pass a full ordered list yourself.

View order (input/target slots): ``--view-layout interleaved_even_odd``, ``interleaved_gcd``, or ``ends_bridge``; default in ``config/data/default.yaml`` ``training.view_layout``.

Before training, Lightning runs a short validation pass (``num_sanity_val_steps``, default 2 batches) to catch dataloader / ``validation_step`` errors; set ``0`` to disable or ``-1`` to run the full val loader once.

Legacy tiny module (no config / no E-RayZer):
  python -m training.train ... --placeholder-model
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.benchmark_scenes import default_benchmark_path
from training.callbacks import CurriculumRampCallback
from training.config_loader import (
    default_erayzer_config_paths,
    load_merged_dict,
    normalize_config_paths,
)
from training.datamodule import DL3DVCurriculumSequenceDataModule, DL3DVManifestDataModule
from training.module import ERayZerTrainModule, PlaceholderTrainModule

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _target_image_hw_from_root(root: dict) -> tuple[int, int]:
    """Read ``model.target_image.height/width`` for dataloader resize (default 256)."""
    ti = (root.get("model") or {}).get("target_image") or {}
    return int(ti.get("height", 256)), int(ti.get("width", 256))


def _build_loggers(
    log_name: str,
    log_dir: str,
    tr_cfg: dict,
    no_wandb: bool,
):
    """CSV always; W&B when enabled and ``wandb`` is installed."""
    csv_logger = CSVLogger(save_dir=log_dir, name=log_name)
    use_wandb = bool(tr_cfg.get("use_wandb", True)) and not no_wandb
    if not use_wandb:
        return csv_logger
    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        print(
            "warning: use_wandb is True but wandb is not installed; "
            "only CSV logging is used. pip install wandb or pass --no-wandb.",
            file=sys.stderr,
        )
        return csv_logger
    wb = WandbLogger(
        project=str(tr_cfg.get("wandb_project", "E-RayZer")),
        name=str(tr_cfg.get("wandb_exp_name") or log_name),
        save_dir=log_dir,
        offline=bool(tr_cfg.get("wandb_offline", False)),
        log_model=False,
    )
    return [csv_logger, wb]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-list", type=str, required=True)
    p.add_argument(
        "--use-dino-curriculum",
        action="store_true",
        help="Use paper Sec. 3.3 semantic spacing from DINO JSON profiles (multi-frame sequences).",
    )
    p.add_argument(
        "--dino-profile-dir",
        type=str,
        default=None,
        help="Directory of <scene_name>.json from compute_dino_semantic_overlap_profiles.py",
    )
    p.add_argument("--num-views", type=int, default=10, help="Frames per sequence (paper: 10 views).")
    p.add_argument("--o-max", type=float, default=1.0, help="Overlap schedule at s=0 (easy).")
    p.add_argument("--o-min", type=float, default=0.75, help="Overlap schedule at s=1 (harder).")
    p.add_argument("--fallback-delta-t", type=int, default=4, help="Integer stride if profile missing.")
    p.add_argument("--curriculum-ramp-steps", type=int, default=86_000, help="Steps to ramp s from 0 to 1.")
    p.add_argument("--dataset-length", type=int, default=20_000, help="Synthetic dataset length (__len__).")
    p.add_argument("--dataset-seed", type=int, default=0)
    p.add_argument(
        "--view-layout",
        type=str,
        default=None,
        help="Reorder temporal strip into [inputs|targets] slots: temporal_halves, interleaved_gcd, interleaved_even_odd, ends_anchors_random_rest, ends_bridge (see training/view_layout.py). Default: config training.view_layout or temporal_halves.",
    )
    p.add_argument(
        "--num-input-views",
        type=int,
        default=None,
        help="Must match config / reproj split when using interleaved layouts (default: yaml num_input_views).",
    )
    p.add_argument(
        "--num-target-views",
        type=int,
        default=None,
        help="Must match config / reproj split (default: yaml num_target_views).",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="PATH",
        help="Extra YAML to merge after the default layered stack (later wins). "
        "Use --no-default-config to merge only these paths.",
    )
    p.add_argument(
        "--no-default-config",
        action="store_true",
        help="Do not load config/{model,data,optimizer,train}/; merge only paths from --config (at least one required).",
    )
    p.add_argument(
        "--geometry-output",
        type=str,
        default=None,
        help="Override model.geometry_output, e.g. depth_reprojection (disparity head + reproj loss).",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Override training.optimizer (adamw | muon_hybrid; shorthand: muon -> muon_hybrid). Default: yaml.",
    )
    p.add_argument(
        "--placeholder-model",
        action="store_true",
        help="Ignore --config and use the tiny placeholder module (no E-RayZer).",
    )
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--log-dir", type=str, default="experiments/lightning_logs")
    p.add_argument(
        "--benchmark-rayzer-txt",
        type=str,
        default=None,
        help="RayZer dl3dv10k_benchmark.txt path for val scene IDs (default: repo data/rayzer_dl3dv10k_benchmark.txt if present).",
    )
    p.add_argument(
        "--no-val-benchmark",
        action="store_true",
        help="Disable benchmark-only val set; train on full manifest (no scene exclusion).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Lightning checkpoint path to resume training (.ckpt).",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases (CSV logging only).",
    )
    p.add_argument(
        "--num-sanity-val-steps",
        type=int,
        default=None,
        help="Validation batches before training (Trainer num_sanity_val_steps). "
        "Default: training.num_sanity_val_steps in yaml or 2. Use 0 to disable; -1 = entire val dataloader.",
    )
    p.add_argument(
        "--no-manifest-index-cache",
        action="store_true",
        help="Disable pickle cache for curriculum manifest index (always scan manifest.json files).",
    )
    p.add_argument(
        "--manifest-index-cache-path",
        type=str,
        default=None,
        help="Override path for curriculum scene index cache (default: <manifest-list>.erayzer_curriculum_scenes.pkl).",
    )
    args = p.parse_args()

    try:
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    except AttributeError:
        pass

    extra = normalize_config_paths(args.config)
    if args.no_default_config:
        if not extra:
            raise SystemExit("--no-default-config requires at least one --config PATH")
        config_paths = extra
    else:
        config_paths = default_erayzer_config_paths(_ROOT) + extra
    merged_root = load_merged_dict(config_paths)
    tr_cfg = merged_root.get("training") or {}

    cbs = [LearningRateMonitor(logging_interval="step")]
    view_layout = (
        args.view_layout
        if args.view_layout is not None
        else str(tr_cfg.get("view_layout", "temporal_halves"))
    )
    n_in_ds = (
        args.num_input_views
        if args.num_input_views is not None
        else int(tr_cfg.get("num_input_views", 5))
    )
    n_tgt_ds = (
        args.num_target_views
        if args.num_target_views is not None
        else int(tr_cfg.get("num_target_views", 5))
    )

    bench_txt = args.benchmark_rayzer_txt or tr_cfg.get("benchmark_rayzer_txt")
    if not bench_txt:
        bd = default_benchmark_path(_ROOT)
        if os.path.isfile(bd):
            bench_txt = bd
    use_val_bench = bool(tr_cfg.get("use_val_benchmark_scenes", True)) and not args.no_val_benchmark

    img_h, img_w = _target_image_hw_from_root(merged_root)

    if args.use_dino_curriculum:
        dm = DL3DVCurriculumSequenceDataModule(
            args.manifest_list,
            profile_dir=args.dino_profile_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_views=args.num_views,
            o_max=args.o_max,
            o_min=args.o_min,
            fallback_delta_t=args.fallback_delta_t,
            dataset_length=args.dataset_length,
            dataset_seed=args.dataset_seed,
            view_layout=view_layout,
            num_input_views=n_in_ds,
            num_target_views=n_tgt_ds,
            benchmark_rayzer_txt=bench_txt,
            use_val_benchmark_scenes=use_val_bench,
            val_dataset_length=int(tr_cfg.get("val_dataset_length", 1024)),
            cache_manifest_index=not args.no_manifest_index_cache,
            manifest_index_cache_path=args.manifest_index_cache_path,
            image_height=img_h,
            image_width=img_w,
        )
        cbs.append(
            CurriculumRampCallback(dm.curriculum_progress, ramp_steps=args.curriculum_ramp_steps)
        )
        log_name = "dl3dv_dino_curriculum"
    else:
        dm = DL3DVManifestDataModule(
            args.manifest_list,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_height=img_h,
            image_width=img_w,
        )
        log_name = "dl3dv_baseline"

    if args.placeholder_model:
        model = PlaceholderTrainModule(lr=args.lr)
    else:
        for p in config_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"--config not found: {p}")
        opt_override = args.optimizer
        if opt_override is not None and str(opt_override).lower().strip() == "muon":
            opt_override = "muon_hybrid"
        model = ERayZerTrainModule(
            config_paths,
            lr=args.lr,
            geometry_output_override=args.geometry_output,
            optimizer_override=opt_override,
        )

    logger = _build_loggers(log_name, args.log_dir, tr_cfg, args.no_wandb)
    if args.num_sanity_val_steps is not None:
        n_sanity = args.num_sanity_val_steps
    else:
        n_sanity = int(tr_cfg.get("num_sanity_val_steps", 2))

    trainer_kw: dict = dict(
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=cbs,
        num_sanity_val_steps=n_sanity,
    )
    gc = float(tr_cfg.get("grad_clip_norm", 0.0))
    # Trainer gradient clipping is incompatible with Lightning manual optimization (e.g. Muon hybrid).
    if gc > 0 and getattr(model, "automatic_optimization", True):
        trainer_kw["gradient_clip_val"] = gc
    _cuda_ok = args.accelerator in ("gpu", "cuda") or (
        args.accelerator == "auto" and torch.cuda.is_available()
    )
    if (
        str(tr_cfg.get("amp_dtype", "bf16")).lower() == "bf16"
        and bool(tr_cfg.get("use_amp", True))
        and _cuda_ok
    ):
        trainer_kw["precision"] = "bf16-mixed"

    trainer = Trainer(**trainer_kw)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
