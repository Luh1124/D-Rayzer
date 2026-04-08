from __future__ import annotations

import os
from typing import Any, Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from training.benchmark_scenes import scene_hashes_from_rayzer_benchmark_file
from training.curriculum_sampling import CurriculumProgress
from training.dataset import (
    ManifestCurriculumSequenceDataset,
    ManifestFrameDataset,
    index_curriculum_scenes_from_manifest_list,
)


class _EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, i: int):
        raise IndexError


class DL3DVManifestDataModule(LightningDataModule):
    def __init__(
        self,
        manifest_list_path: str,
        batch_size: int = 4,
        num_workers: int = 4,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.manifest_list_path = manifest_list_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._image_height = image_height
        self._image_width = image_width
        self._train: ManifestFrameDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._train = ManifestFrameDataset(
                self.manifest_list_path,
                image_height=self._image_height,
                image_width=self._image_width,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class DL3DVCurriculumSequenceDataModule(LightningDataModule):
    """
    Sequence batches with DINO-based semantic spacing (paper Sec. 3.3; see ``ManifestCurriculumSequenceDataset``).
    Exposes ``curriculum_progress`` for ``CurriculumRampCallback``.
    """

    def __init__(
        self,
        manifest_list_path: str,
        profile_dir: str | None,
        batch_size: int = 4,
        num_workers: int = 0,
        num_views: int = 10,
        o_max: float = 1.0,
        o_min: float = 0.75,
        fallback_delta_t: int = 4,
        dataset_length: int = 20_000,
        dataset_seed: int = 0,
        view_layout: str = "temporal_halves",
        num_input_views: int = 5,
        num_target_views: int = 5,
        benchmark_rayzer_txt: Optional[str] = None,
        use_val_benchmark_scenes: bool = True,
        val_dataset_length: int = 1024,
        cache_manifest_index: bool = True,
        manifest_index_cache_path: Optional[str] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.manifest_list_path = manifest_list_path
        self.profile_dir = profile_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_length = dataset_length
        self.dataset_seed = dataset_seed
        self.val_dataset_length = val_dataset_length
        self.use_val_benchmark_scenes = use_val_benchmark_scenes
        self.benchmark_rayzer_txt = benchmark_rayzer_txt
        self.curriculum_progress = CurriculumProgress()
        self._train: ManifestCurriculumSequenceDataset | None = None
        self._val: ManifestCurriculumSequenceDataset | None = None
        self._cache_manifest_index = cache_manifest_index
        self._manifest_index_cache_path = manifest_index_cache_path

        self._ds_common: dict[str, Any] = {
            "manifest_list_path": manifest_list_path,
            "curriculum_progress": self.curriculum_progress,
            "profile_dir": profile_dir,
            "num_views": num_views,
            "o_max": o_max,
            "o_min": o_min,
            "fallback_delta_t": fallback_delta_t,
            "view_layout": view_layout,
            "num_input_views": num_input_views,
            "num_target_views": num_target_views,
            "image_height": image_height,
            "image_width": image_width,
        }

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            return

        bench_hashes = frozenset()
        if self.benchmark_rayzer_txt and os.path.isfile(self.benchmark_rayzer_txt):
            bench_hashes = scene_hashes_from_rayzer_benchmark_file(self.benchmark_rayzer_txt)

        train_block = bench_hashes if (self.use_val_benchmark_scenes and len(bench_hashes) > 0) else None

        scenes_full = index_curriculum_scenes_from_manifest_list(
            self.manifest_list_path,
            use_cache=self._cache_manifest_index,
            cache_path=self._manifest_index_cache_path,
        )
        if not scenes_full:
            raise ValueError(
                f"No usable scenes (>=2 frames) in manifest list: {self.manifest_list_path}"
            )

        self._train = ManifestCurriculumSequenceDataset(
            **self._ds_common,
            prefetched_scenes=scenes_full,
            length=self.dataset_length,
            rng_base_seed=self.dataset_seed,
            scene_name_blocklist=train_block,
        )

        self._val = None
        if self.use_val_benchmark_scenes and len(bench_hashes) > 0:
            self._val = ManifestCurriculumSequenceDataset(
                **self._ds_common,
                prefetched_scenes=scenes_full,
                length=self.val_dataset_length,
                rng_base_seed=self.dataset_seed + 100_003,
                scene_name_allowlist=bench_hashes,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val is None:
            return DataLoader(_EmptyDataset(), batch_size=1, num_workers=0)
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
