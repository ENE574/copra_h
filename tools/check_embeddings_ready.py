#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch


PROT_SEQ_MODELS = ["esm2", "prott5", "saprot"]
RNA_SEQ_MODELS = ["rinalmo", "rna_msm", "rna_fm"]
PROT_STR_MODELS = ["esm_if1", "protrek", "proteinmpnn"]
RNA_STR_MODELS = ["rna_ernie", "rnabert", "rhofold"]


def read_fasta_seq(path: Path) -> str:
    lines = path.read_text().splitlines()
    seq = []
    for line in lines:
        if not line or line.startswith(">"):
            continue
        seq.append(line.strip())
    return "".join(seq)


def check_embedding(pt_path: Path, seq_len: int, allow_shorter: bool) -> str:
    if not pt_path.exists():
        return "missing"
    payload = torch.load(pt_path, map_location="cpu")
    token_embeddings = payload.get("token_embeddings")
    if token_embeddings is None:
        return "no_token_embeddings"
    emb_len = token_embeddings.shape[0]
    if emb_len == seq_len:
        return ""
    if allow_shorter and emb_len < seq_len:
        return f"warn_shorter:{emb_len}!={seq_len}"
    return f"length_mismatch:{emb_len}!={seq_len}"


def check_group(fasta_dir: Path, emb_root: Path, seq_models, str_models, kind: str, allow_shorter_struct: bool) -> list:
    errors = []
    for fasta_path in sorted(fasta_dir.glob("*.fasta")):
        seq = read_fasta_seq(fasta_path)
        seq_len = len(seq)
        stem = fasta_path.stem
        seq_base = emb_root / f"{kind}_sequence"
        str_base = emb_root / f"{kind}_structure"
        for model in seq_models:
            pt_path = seq_base / model / f"{stem}.pt"
            err = check_embedding(pt_path, seq_len, allow_shorter=False)
            if err:
                errors.append(f"{stem} seq {model}: {err}")
        for model in str_models:
            pt_path = str_base / model / f"{stem}.pt"
            err = check_embedding(pt_path, seq_len, allow_shorter=allow_shorter_struct)
            if err:
                errors.append(f"{stem} str {model}: {err}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that 12-model embeddings exist and match sequence lengths.")
    parser.add_argument("--embedding_root", default="outputs/feature_extraction", help="Embedding root directory.")
    parser.add_argument("--inputs_root", default="outputs/feature_extraction/inputs", help="Inputs root directory.")
    parser.add_argument("--strict_structure", action="store_true", help="Enforce exact length match for structure embeddings (default allows shorter).")
    parser.add_argument("--show_all", action="store_true", help="Print all warnings, not just failures.")
    args = parser.parse_args()

    emb_root = Path(args.embedding_root)
    inputs_root = Path(args.inputs_root)
    prot_fasta_dir = inputs_root / "protein_single"
    rna_fasta_dir = inputs_root / "rna_single"

    if not prot_fasta_dir.exists():
        print(f"[FAIL] Missing {prot_fasta_dir}")
        return 1
    if not rna_fasta_dir.exists():
        print(f"[FAIL] Missing {rna_fasta_dir}")
        return 1

    errors = []
    errors.extend(check_group(prot_fasta_dir, emb_root, PROT_SEQ_MODELS, PROT_STR_MODELS, "protein", allow_shorter_struct=not args.strict_structure))
    errors.extend(check_group(rna_fasta_dir, emb_root, RNA_SEQ_MODELS, RNA_STR_MODELS, "rna", allow_shorter_struct=not args.strict_structure))

    hard_errors = [e for e in errors if "missing" in e or "no_token_embeddings" in e or "length_mismatch" in e]
    soft_warnings = [e for e in errors if e.startswith("warn_shorter")]

    if hard_errors:
        print("[FAIL] Embedding checks failed:")
        for err in hard_errors:
            print(f"  - {err}")
        if args.show_all and soft_warnings:
            print("[WARN] Shorter structure embeddings (allowed):")
            for err in soft_warnings:
                print(f"  - {err}")
        return 1

    if soft_warnings:
        print("[OK] No hard failures, but some structure embeddings are shorter than fasta (allowed).")
        if args.show_all:
            for err in soft_warnings:
                print(f"  - {err}")
        return 0

    print("[OK] All 12-model embeddings exist and lengths match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
