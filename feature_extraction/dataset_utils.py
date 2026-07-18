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


def _load_csvs_concat(csv_path: str, extra_csv_paths: Optional[List[str]] = None) -> pd.DataFrame:
    paths: List[str] = [csv_path]
    if extra_csv_paths:
        paths.extend(extra_csv_paths)
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def _drop_echo_header_rows(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Remove rows where the id column repeats the header label (duplicate header lines in CSV)."""
    if id_col not in df.columns:
        return df
    return df[df[id_col].astype(str).str.strip() != id_col].copy()


def _mcsm_train_stem_pdb_candidates(pdb_dir: Path, raw_pdb_id: str, mut: str) -> List[Path]:
    """See ``_mutant_structure_path_candidates`` docstring (training-aligned stem)."""
    structure_key = f"{raw_pdb_id}_{mut}" if mut else raw_pdb_id
    stem = structure_key.split("_")[0]
    return [pdb_dir / f"{stem}{suf}" for suf in (".pdb", ".cif")]


def build_dataset_fastas(
    csv_path: str,
    output_dir: str,
    id_col: str,
    protein_seq_col: str,
    rna_seq_col: str,
    protein_chain_col: str = "Protein chains",
    rna_chain_col: str = "RNA chains",
    pdb_list_path: Optional[str] = None,
    extra_csv_paths: Optional[List[str]] = None,
    mutation_col: Optional[str] = None,
    partner_seq_col: Optional[str] = None,
    partner_chain_col: Optional[str] = None,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_fasta = output_dir / "protein.fasta"
    rna_fasta = output_dir / "rna.fasta"
    protein_single_dir = output_dir / "protein_single"
    rna_single_dir = output_dir / "rna_single"
    protein_single_dir.mkdir(parents=True, exist_ok=True)
    rna_single_dir.mkdir(parents=True, exist_ok=True)

    df = _load_csvs_concat(csv_path, extra_csv_paths)
    df = _drop_echo_header_rows(df, id_col)
    if mutation_col and mutation_col in df.columns:
        df = df.drop_duplicates(subset=[id_col, mutation_col], keep="first").copy()

    # Determine if partner is protein (PPI case): partner_seq_col is used to
    # signal that the "na" side is actually a second protein chain.  In that
    # case partner sequences must be written with ``_prot_`` names so offline
    # embedding loaders (partner_use_protein_embeddings=True) can find them
    # under protein_sequence / protein_structure.
    has_protein_partner = bool(partner_seq_col and partner_seq_col in df.columns)

    if pdb_list_path:
        allowed_pairs = set(_parse_pdb_list_entries(pdb_list_path))
        if not allowed_pairs:
            raise ValueError(f"No valid entries parsed from pdb_list_path: {pdb_list_path}")
        id_series = df[id_col].astype(str).str.strip()
        prot_series = df[protein_chain_col].astype(str).str.strip()
        rna_col = partner_chain_col if has_protein_partner else rna_chain_col
        rna_series = df[rna_col].astype(str).str.strip()
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
            pdb_key = str(row[id_col]).strip()
            if mutation_col and mutation_col in row.index and str(row.get(mutation_col, "")).strip():
                complex_id = f"{pdb_key}_{str(row[mutation_col]).strip()}"
            else:
                complex_id = pdb_key
            prot_chains = _split_chain_sequences(row[protein_seq_col])
            for chain, seq in prot_chains:
                name = safe_name(f"{complex_id}_prot_{chain or 'X'}")
                p_handle.write(f">{name}\n{seq}\n")
                with open(protein_single_dir / f"{name}.fasta", "w", encoding="utf-8") as f_handle:
                    f_handle.write(f">{name}\n{seq}\n")

            if has_protein_partner:
                # PPI: partner is also a protein — write as _prot_ so it gets
                # protein embeddings and can be found by partner_use_protein_embeddings.
                partner_chains = _split_chain_sequences(row[partner_seq_col])
                for chain, seq in partner_chains:
                    name = safe_name(f"{complex_id}_prot_{chain or 'X'}")
                    p_handle.write(f">{name}\n{seq}\n")
                    with open(protein_single_dir / f"{name}.fasta", "w", encoding="utf-8") as f_handle:
                        f_handle.write(f">{name}\n{seq}\n")
            else:
                rna_chains = _split_chain_sequences(row[rna_seq_col])
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
    extra_csv_paths: Optional[List[str]] = None,
    mutation_col: Optional[str] = None,
    partner_chain_col: Optional[str] = None,
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

    df = _load_csvs_concat(csv_path, extra_csv_paths)
    df = _drop_echo_header_rows(df, id_col)
    if mutation_col and mutation_col in df.columns:
        df = df.drop_duplicates(subset=[id_col, mutation_col], keep="first").copy()

    has_partner = bool(partner_chain_col and partner_chain_col in df.columns)

    for _, row in df.iterrows():
        pdb_id = str(row.get(id_col, "")).strip()
        prot_chain = str(row.get(protein_chain_col, "")).strip()
        rna_chain = str(row.get(rna_chain_col, "")).strip()
        partner_chain = str(row.get(partner_chain_col or "", "")).strip() if has_partner else ""
        if not pdb_id or pdb_id.lower() == "nan":
            continue
        if not prot_chain or prot_chain.lower() == "nan":
            continue
        # For PPI (has_partner), the partner_chain is the "rna_chain" equivalent
        if has_partner and not partner_chain:
            continue
        if not has_partner and (not rna_chain or rna_chain.lower() == "nan"):
            continue
        mut = ""
        if mutation_col and mutation_col in row.index:
            mut = str(row.get(mutation_col, "") or "").strip()
        if mutation_col:
            candidates = _mcsm_train_stem_pdb_candidates(pdb_dir, pdb_id, mut)
        else:
            candidates = []
            for suf in (".cif", ".pdb"):
                if has_partner:
                    candidates.append(pdb_dir / f"{pdb_id}_{prot_chain}_{partner_chain}{suf}")
                else:
                    candidates.append(pdb_dir / f"{pdb_id}_{prot_chain}_{rna_chain}{suf}")
                candidates.append(pdb_dir / f"{pdb_id}{suf}")
        found = False
        for candidate in candidates:
            if candidate.exists():
                pdb_files.append(str(candidate))
                found = True
                break
        if not found:
            stem = (f"{pdb_id}_{mut}" if mut else pdb_id).split("_")[0]
            missing.append(stem)

    return {
        "pdb_files": sorted(set(pdb_files)),
        "missing": missing,
    }
