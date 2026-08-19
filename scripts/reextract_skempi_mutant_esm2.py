#!/usr/bin/env python3
"""Re-extract mutant ESM-2 sequence embeddings for SKEMPI rows updated by PDB sync."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.structure_dataset import _parse_chain_seq_entries
from feature_extraction.extractors import extract_esm2, safe_name


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for label, seq in records:
            handle.write(f">{label}\n{seq}\n")


def _changed_row_indices(report_path: Path) -> list[int]:
    report = pd.read_csv(report_path)
    return sorted(report["row_index"].unique())


def _build_mutant_records(df: pd.DataFrame, row_indices: list[int]) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for idx in row_indices:
        row = df.iloc[idx]
        pdb = str(row["PDB"]).strip()
        mutation = str(row["MUTATION"]).strip()
        complex_id = f"{pdb}_{mutation}"
        for chain, seq in _parse_chain_seq_entries(row["Mutation sequences"]):
            label = safe_name(f"{complex_id}_prot_{chain}")
            if label in seen:
                continue
            seen.add(label)
            records.append((label, seq))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/splits/skempi_pdb_seq_sync_report.csv",
    )
    parser.add_argument(
        "--csv",
        default="/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/splits/skempi.csv",
    )
    parser.add_argument(
        "--output-root",
        default="/media/SSD0/csd/lrg/copra_h/outputs/feature_extraction_SKEMPI/mutant",
    )
    parser.add_argument(
        "--model-location",
        default="/media/SSD0/csd/lrg/copra_h/weights/esm-2_weights/esm2_t33_650M_UR50D.pt",
    )
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    report_path = Path(args.report)
    csv_path = Path(args.csv)
    output_dir = Path(args.output_root) / "protein_sequence" / "esm2"

    row_indices = _changed_row_indices(report_path)
    df = pd.read_csv(csv_path)
    records = _build_mutant_records(df, row_indices)

    print(f"changed rows: {len(row_indices)}")
    print(f"fasta records: {len(records)}")
    print(f"output: {output_dir}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as handle:
        fasta_path = Path(handle.name)
    try:
        _write_fasta(fasta_path, records)
        extract_esm2(
            fasta_path=str(fasta_path),
            output_dir=str(output_dir),
            device=args.device,
            model_location=args.model_location,
            batch_size=args.batch_size,
        )
    finally:
        fasta_path.unlink(missing_ok=True)

    print("done")


if __name__ == "__main__":
    main()
