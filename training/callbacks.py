from __future__ import annotations

from lightning.pytorch import Callback, LightningModule, Trainer

from training.curriculum_sampling import CurriculumProgress


class CurriculumRampCallback(Callback):
    """
    Linear ramp s = min(1, global_step / ramp_steps) for E-RayZer Sec. 3.3 semantic schedule.
    Early steps: s ~ 0 (high overlap target); later: s -> 1 (lower overlap, harder spacing).
    """

    def __init__(self, progress: CurriculumProgress, ramp_steps: int) -> None:
        super().__init__()
        self.progress = progress
        self.ramp_steps = max(1, int(ramp_steps))

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        s = min(1.0, float(step) / float(self.ramp_steps))
        self.progress.set(s)
