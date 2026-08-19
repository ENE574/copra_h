"""Multi-task DataModule: cycle train loaders (homogeneous batches per task)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from easydict import EasyDict
from pytorch_lightning.utilities import CombinedLoader

from pl_modules.data_module import DataModule, get_collate


def _parse_yaml(path: str) -> EasyDict:
    import yaml

    with open(path, "r") as f:
        return EasyDict(yaml.safe_load(f))


class MultiTaskDataModule(pl.LightningDataModule):
    """
    Train: CombinedLoader cycles one task per step (each batch uses one dataset's collate/offline root).
    Val/Test: primary task only (first entry unless ``primary_task`` is set).
    """

    def __init__(
        self,
        multitask_sources: List[Dict[str, Any]],
        col_group: str = "fold_0",
        primary_task: Optional[str] = None,
    ):
        super().__init__()
        if not multitask_sources:
            raise ValueError("multitask_sources must be a non-empty list")
        self.multitask_sources = multitask_sources
        self.col_group = col_group
        self.primary_task = primary_task or multitask_sources[0]["name"]
        self._task_modules: Dict[str, DataModule] = {}

    def setup(self, stage=None):
        for spec in self.multitask_sources:
            name = spec["name"]
            data_cfg = _parse_yaml(spec["data_config"])
            dm = DataModule(
                df_path=data_cfg.df_path,
                col_group=self.col_group,
                batch_size=int(spec.get("batch_size", getattr(data_cfg, "batch_size", 2))),
                num_workers=int(getattr(data_cfg, "num_workers", 0)),
                pin_memory=bool(getattr(data_cfg, "pin_memory", True)),
                cache_dir=getattr(data_cfg, "cache_dir", None),
                strategy=getattr(data_cfg, "strategy", "separate"),
                dataset_args=data_cfg,
                group_remap=getattr(data_cfg, "group_remap", None),
            )
            dm.setup(stage=stage)
            self._task_modules[name] = dm

    def train_dataloader(self):
        loaders = {name: dm.train_dataloader() for name, dm in self._task_modules.items()}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        return self._task_modules[self.primary_task].val_dataloader()

    def test_dataloader(self):
        return self._task_modules[self.primary_task].test_dataloader()

    @property
    def collate_fn(self):
        dm = self._task_modules[self.primary_task]
        return dm.collate_fn

    @property
    def dataset_args(self):
        return self._task_modules[self.primary_task].dataset_args
