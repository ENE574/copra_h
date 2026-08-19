"""Batch sampler that groups mutations from the same complex."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import BatchSampler, Sampler


def _unwrap_dataset(obj):
    """Resolve the underlying dataset from a sampler or dataset object."""
    if hasattr(obj, "data"):
        return obj
    if hasattr(obj, "dataset"):
        return obj.dataset
    if hasattr(obj, "data_source"):
        return obj.data_source
    raise TypeError(f"Cannot resolve dataset from {type(obj)!r}")


def _build_complex_to_indices(dataset) -> dict[str, list[int]]:
    complex_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        entry = dataset.data[idx]
        complex_id = str(entry.get("complex_group", entry.get("complex", entry.get("PDB", idx))))
        complex_to_indices[complex_id].append(idx)
    return dict(complex_to_indices)


class ComplexBatchSampler(BatchSampler):
    """
    Each batch contains ``batch_size`` mutations from a single complex.

    Subclasses ``BatchSampler`` so PyTorch Lightning can inject a
    ``DistributedSampler`` under DDP (``sampler`` kwarg).
    """

    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        drop_last: bool = False,
        min_batch_size: int = 2,
        shuffle: bool = True,
    ):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.min_batch_size = int(min_batch_size)
        self.shuffle = bool(shuffle)
        dataset = _unwrap_dataset(sampler)
        self.complex_to_indices = _build_complex_to_indices(dataset)
        self.index_to_complex = {
            idx: complex_id
            for complex_id, group in self.complex_to_indices.items()
            for idx in group
        }

    def _make_batches(self, indices: List[int]) -> List[List[int]]:
        if not indices:
            return []

        local_groups: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            complex_id = self.index_to_complex.get(idx)
            if complex_id is not None:
                local_groups[complex_id].append(idx)

        complex_ids = list(local_groups.keys())
        if self.shuffle:
            random.shuffle(complex_ids)

        batches: List[List[int]] = []
        for complex_id in complex_ids:
            group = local_groups[complex_id]
            if self.shuffle:
                random.shuffle(group)
            for start in range(0, len(group), self.batch_size):
                batch = group[start : start + self.batch_size]
                if len(batch) < self.min_batch_size:
                    if self.drop_last:
                        continue
                    if len(batch) == 0:
                        continue
                elif self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self._make_batches(list(self.sampler)):
            yield batch

    def __len__(self) -> int:
        n = len(self._make_batches(list(self.sampler)))
        return max(n, 1)
