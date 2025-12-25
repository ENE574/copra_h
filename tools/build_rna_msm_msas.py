#!/usr/bin/env python3
import argparse
from pathlib import Path


def read_fasta_sequence(path: Path) -> str:
    seq = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    break
                continue
            seq.append(line)
    if not seq:
        raise ValueError(f"No sequence found in {path}")
    return "".join(seq)


def sanitize_rna_sequence(seq: str) -> str:
    seq = seq.strip().upper().replace("T", "U")
    allowed = set("AGCUXN-")
    return "".join(ch if ch in allowed else "N" for ch in seq)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build single-sequence .a2m_msa2 files for RNA-MSM.")
    parser.add_argument("--id_list", required=True, help="Path to rna_msm_ids.txt")
    parser.add_argument("--rna_single_dir", required=True, help="Directory with *_rna_*.fasta files")
    parser.add_argument("--out_dir", required=True, help="Output directory for .a2m_msa2 files")
    args = parser.parse_args()

    id_list = Path(args.id_list)
    rna_single_dir = Path(args.rna_single_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    with id_list.open("r", encoding="utf-8") as handle:
        for line in handle:
            rna_id = line.strip()
            if not rna_id:
                continue
            fasta_path = rna_single_dir / f"{rna_id}.fasta"
            if not fasta_path.exists():
                missing.append(rna_id)
                continue
            seq = sanitize_rna_sequence(read_fasta_sequence(fasta_path))
            out_path = out_dir / f"{rna_id}.a2m_msa2"
            out_path.write_text(f">{rna_id}\n{seq}\n", encoding="utf-8")

    if missing:
        print(f"Missing {len(missing)} fasta files. First few: {missing[:5]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
