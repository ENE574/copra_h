import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from feature_extraction.extractors import safe_name


def _split_chain_sequences(seq_field: str) -> List[Tuple[str, str]]:
    chains = []
    for part in str(seq_field).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            chain, seq = part.split(":", 1)
            chain = chain.strip()
        else:
            chain, seq = "", part
        chains.append((chain, seq.strip()))
    return chains


def build_dataset_fastas(
    csv_path: str,
    output_dir: str,
    id_col: str,
    protein_seq_col: str,
    rna_seq_col: str,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_fasta = output_dir / "protein.fasta"
    rna_fasta = output_dir / "rna.fasta"
    protein_single_dir = output_dir / "protein_single"
    rna_single_dir = output_dir / "rna_single"
    protein_single_dir.mkdir(parents=True, exist_ok=True)
    rna_single_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    rna_ids = []
    with open(protein_fasta, "w", encoding="utf-8") as p_handle, open(
        rna_fasta, "w", encoding="utf-8"
    ) as r_handle:
        for _, row in df.iterrows():
            complex_id = str(row[id_col])
            prot_chains = _split_chain_sequences(row[protein_seq_col])
            rna_chains = _split_chain_sequences(row[rna_seq_col])
            for chain, seq in prot_chains:
                name = safe_name(f"{complex_id}_prot_{chain or 'X'}")
                p_handle.write(f">{name}\n{seq}\n")
                with open(protein_single_dir / f"{name}.fasta", "w", encoding="utf-8") as f_handle:
                    f_handle.write(f">{name}\n{seq}\n")
            for chain, seq in rna_chains:
                name = safe_name(f"{complex_id}_rna_{chain or 'X'}")
                rna_ids.append(name)
                r_handle.write(f">{name}\n{seq}\n")
                with open(rna_single_dir / f"{name}.fasta", "w", encoding="utf-8") as f_handle:
                    f_handle.write(f">{name}\n{seq}\n")

    rna_id_list = output_dir / "rna_msm_ids.txt"
    with open(rna_id_list, "w", encoding="utf-8") as handle:
        for name in rna_ids:
            handle.write(f"{name}\n")

    return {
        "protein_fasta": str(protein_fasta),
        "rna_fasta": str(rna_fasta),
        "protein_single_dir": str(protein_single_dir),
        "rna_single_dir": str(rna_single_dir),
        "rna_msm_ids": str(rna_id_list),
    }
