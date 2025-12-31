import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from feature_extraction.extractors import safe_name


def _load_pdb_list(pdb_list_path: str) -> List[str]:
    items = []
    with open(pdb_list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def _parse_pdb_list_entries(pdb_list_path: str) -> List[Tuple[str, str, str]]:
    entries: List[Tuple[str, str, str]] = []
    for item in _load_pdb_list(pdb_list_path):
        name = Path(item).name
        stem = Path(name).stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        pdb_id = parts[0].strip()
        prot_chain = "_".join(parts[1:-1]).strip()
        rna_chain = parts[-1].strip()
        if not pdb_id or not prot_chain or not rna_chain:
            continue
        entries.append((pdb_id, prot_chain, rna_chain))
    return entries


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
    protein_chain_col: str = "Protein chains",
    rna_chain_col: str = "RNA chains",
    pdb_list_path: Optional[str] = None,
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
    if pdb_list_path:
        allowed_pairs = set(_parse_pdb_list_entries(pdb_list_path))
        if not allowed_pairs:
            raise ValueError(f"No valid entries parsed from pdb_list_path: {pdb_list_path}")
        id_series = df[id_col].astype(str).str.strip()
        prot_series = df[protein_chain_col].astype(str).str.strip()
        rna_series = df[rna_chain_col].astype(str).str.strip()
        pair_tuples = list(zip(id_series, prot_series, rna_series))
        mask = [
            (pdb_id, prot_chain, rna_chain) in allowed_pairs
            for pdb_id, prot_chain, rna_chain in pair_tuples
        ]
        df = df[mask].copy()
        found_pairs = set(pair for pair, keep in zip(pair_tuples, mask) if keep)
        missing_pairs = allowed_pairs - found_pairs
        if missing_pairs:
            sample = ", ".join(sorted(f"{p}_{pc}_{rc}" for p, pc, rc in list(missing_pairs)[:5]))
            raise ValueError(
                f"{len(missing_pairs)} entries from pdb_list_path not found in CSV. Sample: {sample}"
            )

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

    rna_id_list_unique = output_dir / "rna_msm_ids_unique.txt"
    seen = set()
    unique_ids = []
    for name in rna_ids:
        if name in seen:
            continue
        seen.add(name)
        unique_ids.append(name)
    with open(rna_id_list_unique, "w", encoding="utf-8") as handle:
        for name in unique_ids:
            handle.write(f"{name}\n")

    return {
        "protein_fasta": str(protein_fasta),
        "rna_fasta": str(rna_fasta),
        "protein_single_dir": str(protein_single_dir),
        "rna_single_dir": str(rna_single_dir),
        "rna_msm_ids": str(rna_id_list),
        "rna_msm_ids_unique": str(rna_id_list_unique),
    }


def build_dataset_pdb_list(
    csv_path: str,
    pdb_dir: str,
    id_col: str,
    protein_chain_col: str,
    rna_chain_col: str,
    pdb_list_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    pdb_dir = Path(pdb_dir)

    pdb_files: List[str] = []
    missing: List[str] = []

    if pdb_list_path:
        for item in _load_pdb_list(pdb_list_path):
            p = Path(item)
            if not p.suffix:
                candidate = pdb_dir / f"{p.name}.cif"
                if candidate.exists():
                    pdb_files.append(str(candidate))
                    continue
                candidate = pdb_dir / f"{p.name}.pdb"
                if candidate.exists():
                    pdb_files.append(str(candidate))
                    continue
                missing.append(p.name)
                continue
            if not p.is_absolute():
                p = pdb_dir / p
            if p.exists():
                pdb_files.append(str(p))
            else:
                missing.append(p.name)
        return {
            "pdb_files": sorted(set(pdb_files)),
            "missing": missing,
        }

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        pdb_id = str(row.get(id_col, "")).strip()
        prot_chain = str(row.get(protein_chain_col, "")).strip()
        rna_chain = str(row.get(rna_chain_col, "")).strip()
        if not pdb_id or pdb_id.lower() == "nan":
            continue
        if not prot_chain or prot_chain.lower() == "nan":
            continue
        if not rna_chain or rna_chain.lower() == "nan":
            continue
        found = False
        for suffix in (".cif", ".pdb"):
            candidate = pdb_dir / f"{pdb_id}_{prot_chain}_{rna_chain}{suffix}"
            if candidate.exists():
                pdb_files.append(str(candidate))
                found = True
                break
        if not found:
            missing.append(f"{pdb_id}_{prot_chain}_{rna_chain}")

    return {
        "pdb_files": sorted(set(pdb_files)),
        "missing": missing,
    }
