#!/usr/bin/env python3
"""Normalize offline embedding tensors to a fixed feature dimension."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch


def _adjust_dim(tensor: torch.Tensor, target: int) -> torch.Tensor:
    if target <= 0 or tensor.shape[-1] == target:
        return tensor
    if tensor.shape[-1] > target:
        return tensor[..., :target]
    pad_shape = list(tensor.shape[:-1]) + [target - tensor.shape[-1]]
    pad = torch.zeros(pad_shape, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=-1)


def _rewrite_payload(payload, target: int) -> Tuple[object, bool]:
    if isinstance(payload, torch.Tensor):
        updated = _adjust_dim(payload, target)
        return updated, updated.shape != payload.shape
    if not isinstance(payload, dict):
        return payload, False
    changed = False
    updated = dict(payload)
    for key in ("token_embeddings", "sequence_embedding"):
        value = updated.get(key)
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            new_value = _adjust_dim(value, target)
            if new_value.shape != value.shape:
                updated[key] = new_value
                changed = True
    return updated, changed


def _iter_pt_files(root: Path, subdirs: Iterable[str]) -> Iterable[Path]:
    if subdirs:
        for sub in subdirs:
            base = root / sub
            if base.exists():
                yield from base.rglob("*.pt")
        return
    yield from root.rglob("*.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify offline embeddings to a fixed dim.")
    parser.add_argument(
        "--embedding_root",
        action="append",
        required=True,
        help="Embedding root to scan (repeatable).",
    )
    parser.add_argument("--dim", type=int, default=1024, help="Target embedding dim.")
    parser.add_argument(
        "--subdir",
        action="append",
        default=["protein_sequence", "protein_structure", "rna_sequence", "rna_structure"],
        help="Subdir under each root to scan (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report changes.")
    args = parser.parse_args()

    total = 0
    updated = 0
    skipped = 0
    for root in args.embedding_root:
        root_path = Path(root)
        for emb_path in _iter_pt_files(root_path, args.subdir):
            total += 1
            try:
                payload = torch.load(emb_path, map_location="cpu")
            except Exception:
                skipped += 1
                continue
            new_payload, changed = _rewrite_payload(payload, args.dim)
            if not changed:
                continue
            updated += 1
            if args.dry_run:
                print(f"would update {emb_path}")
                continue
            torch.save(new_payload, emb_path)

    print(
        f"scanned={total} updated={updated} skipped={skipped} dim={args.dim}"
    )


if __name__ == "__main__":
    main()
