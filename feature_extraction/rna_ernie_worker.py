#!/usr/bin/env python3
"""
Run RNAErnie embedding extraction in a fresh Python process.

extract_features.py loads PyTorch first; loading Paddle in the same process often
triggers ``Intel MKL function load error`` on CPU. This worker imports only
Paddle (then Torch only for saving .pt tensors after inference).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _read_fasta(path: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    seq_chunks: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_chunks)))
                name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if name is not None:
        records.append((name, "".join(seq_chunks)))
    return records


def _safe_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)


def _paddle_places(device: str) -> list[str]:
    d = (device or "cpu").lower().strip()
    if d in ("cuda", "gpu"):
        return ["gpu", "cpu"]
    if d.startswith("gpu:"):
        return [d, "gpu", "cpu"]
    return ["cpu"]


def _gpu_fail(exc: BaseException) -> bool:
    msg = str(exc).lower().replace(" ", "")
    return (
        "cudnn" in msg
        or "preconditionnotmet" in msg
        or "cudnn_dso_handle" in msg
        or "cannot load cudnn" in str(exc).lower()
    )


def _prepare_cpu_env(places: list[str]) -> None:
    if "cpu" not in places:
        return
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ["MKL_THREADING_LAYER"] = "INTEL"
    os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    places = _paddle_places(args.device)
    _prepare_cpu_env(places)

    import paddle
    from paddlenlp.transformers import ErnieModel

    sys.path.insert(0, str(repo_root / "RNAErnie"))
    from rna_ernie import BatchConverter

    batch_converter = BatchConverter(
        k_mer=1,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    model = None
    last_exc: BaseException | None = None
    place_used: str | None = None
    for place in places:
        try:
            paddle.set_device(place)
            model = ErnieModel.from_pretrained(args.model_dir)
            model.eval()
            place_used = place
            break
        except RuntimeError as exc:
            last_exc = exc
            if place == "cpu":
                raise
            if _gpu_fail(exc):
                continue
            raise
    if model is None:
        raise RuntimeError(f"RNAErnie worker: failed to load model: {last_exc!r}") from last_exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _read_fasta(args.fasta)

    import torch

    with paddle.no_grad():
        for names, seqs, inputs_ids in batch_converter(records):
            embeddings = model(inputs_ids)[0].detach().numpy()
            for i, name in enumerate(names):
                seq_len = len(seqs[i])
                token_embeddings = torch.tensor(embeddings[i, 1 : 1 + seq_len])
                seq_rep = token_embeddings.mean(0)
                payload = {
                    "token_embeddings": token_embeddings,
                    "sequence_embedding": seq_rep,
                }
                path = out_dir / f"{_safe_name(name)}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
