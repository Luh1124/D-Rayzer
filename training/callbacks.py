from __future__ import annotations

from lightning.pytorch import Callback, LightningModule, Trainer

from training.curriculum_sampling import CurriculumProgress


class CurriculumRampCallback(Callback):
    """
    Curriculum progress ``s in [0, 1]`` for E-RayZer Sec. 3.3 semantic schedule
    (``o(s) = s*o_min + (1-s)*o_max`` in ``curriculum_sampling``).

    Default (``warmup_steps=0``, ``ramp_power=1``): linear
    ``s = min(1, max(0, step - warmup) / ramp_steps)``.

    Use larger ``ramp_steps`` or ``warmup_steps`` to slow how fast batches get harder.
    Use ``ramp_power > 1`` so ``s = t**power`` (``t`` linear in step): early training stays
    in the easier regime longer.

    Progress uses ``pl_module._effective_update_count`` when present (successful optimizer steps
    only, aligned with LR when grad-norm skips occur); otherwise ``trainer.global_step``.
    Updated at ``on_train_batch_end`` so the next batch's sampling sees the new ``s``.
    """

    def __init__(
        self,
        progress: CurriculumProgress,
        ramp_steps: int,
        *,
        warmup_steps: int = 0,
        ramp_power: float = 1.0,
    ) -> None:
        super().__init__()
        self.progress = progress
        self.ramp_steps = max(1, int(ramp_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.ramp_power = float(ramp_power) if float(ramp_power) > 0 else 1.0

    def _set_progress_from_step(self, step: int) -> None:
        t = float(step - self.warmup_steps) / float(self.ramp_steps)
        t = max(0.0, min(1.0, t))
        s = min(1.0, t**self.ramp_power)
        self.progress.set(s)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if not trainer.training:
            return
        eff = getattr(pl_module, "_effective_update_count", None)
        if eff is not None:
            step = int(eff.item())
        else:
            step = int(trainer.global_step)
        self._set_progress_from_step(step)


class EffectiveUpdatesStopCallback(Callback):
    """
    Stop training when ``pl_module._effective_update_count`` reaches ``max_effective_updates``
    (successful ``optimizer.step()`` only). Requires :class:`~training.module.ERayZerTrainModule`.
    """

    def __init__(self, max_effective_updates: int) -> None:
        super().__init__()
        self.max_effective_updates = max(1, int(max_effective_updates))

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if not trainer.training:
            return
        buf = getattr(pl_module, "_effective_update_count", None)
        if buf is None:
            return
        if int(buf.item()) >= self.max_effective_updates:
            trainer.should_stop = True
