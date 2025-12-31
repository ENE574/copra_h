#!/usr/bin/env python3
import argparse
from pathlib import Path


def read_fasta(path: Path):
    name = None
    seq = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    yield name, "".join(seq)
                name = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if name:
            yield name, "".join(seq)


def sanitize_rna(seq: str) -> str:
    allowed = set("AGCUNX-")
    seq = seq.strip().upper().replace("T", "U")
    return "".join(ch if ch in allowed else "N" for ch in seq)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate single-sequence RNA-MSM .a2m_msa2 files from a FASTA.")
    parser.add_argument(
        "--fasta",
        default="outputs/feature_extraction_PRI30k/inputs/rna.fasta",
        help="Path to RNA FASTA (default: outputs/feature_extraction_PRI30k/inputs/rna.fasta)",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/feature_extraction_PRI30k/inputs/rna_msm_msas",
        help="Output directory for .a2m_msa2 files",
    )
    parser.add_argument(
        "--ids_out",
        default="outputs/feature_extraction_PRI30k/inputs/rna_msm_ids_unique.txt",
        help="Output path for unique RNA-MSM id list",
    )
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    count = 0
    seen = set()
    ids = []
    for name, seq in read_fasta(fasta_path):
        if name in seen:
            continue
        seen.add(name)
        ids.append(name)
        seq = sanitize_rna(seq)
        out_path = out_dir / f"{name}.a2m_msa2"
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write(f">{name}\n{seq}\n")
        count += 1

    ids_out = Path(args.ids_out)
    ids_out.parent.mkdir(parents=True, exist_ok=True)
    ids_out.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")

    print(f"Wrote {count} MSA files to {out_dir}")
    print(f"Wrote {len(ids)} unique ids to {ids_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
