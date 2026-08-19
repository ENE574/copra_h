#!/usr/bin/env python3
"""
Prepare DNA dG (protein-DNA binding affinity) dataset.

Recommended source: PDBbind v2020 general set (protein-DNA subset).
PDBbind is the gold-standard collection of binding affinities for biomolecular
complexes. The full version requires registration at http://www.pdbbind.org.cn/,
but any published version is acceptable.

This script processes the PDBbind INDEX file for the "refined set" or "general set"
to extract protein-DNA complexes with measured ΔG.

PDBbind INDEX file format (general set, v2020):
    Column 0: PDB ID
    Column 1: Resolution (Å)
    Column 2: Release year
    Column 3: -log(Kd/Ki)
    Column 4: Kd/Ki (M)
    Column 5: Reference
    Column 6: Ligand name (for protein-ligand) or "protein-protein"/"protein-DNA"/"protein-RNA"

Usage:
    python prepare_datasets/prepare_dna_dg_from_pdbbind.py \\
        --pdbbind_dir /path/to/PDBbind_v2020 \\
        --out_csv /media/SSD0/csd/lrg/copra_h/datasets/DNA_dG/splits/DNA_dG.csv

If you don't have PDBbind, the script provides instructions to download it.

Alternative: If you have a custom protein-DNA dG CSV, use --custom_csv instead.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_FOLDS = 5
SEED = 2024

# Column names expected by the IMF-Net pipeline for protein-DNA
COL_PDB = "PDB"
COL_PROT_CHAIN = "Protein chains"
COL_NA_CHAIN = "DNA chains"
COL_PROT_SEQ = "Protein sequences"
COL_NA_SEQ = "DNA sequences"
COL_LABEL = "△G(kcal/mol)"
COL_FOLD_PREFIX = "fold_"

# Default data paths
DEFAULT_PDB_DIR = "/media/SSD0/csd/lrg/copra_h/datasets/DNA_dG/PDBs"
DEFAULT_OUT_CSV = "/media/SSD0/csd/lrg/copra_h/datasets/DNA_dG/splits/DNA_dG.csv"

# PDBbind INDEX file columns
# The general set INDEX file has these columns (tab-separated):
# 0       1        2       3       4       5      6
# PDB     Res.(Å)  Year   -logKd   Kd(M)   Ref    Category
# where Category is "protein-DNA", "protein-protein", "protein-RNA", or ligand name
COL_PDB_ID = 0
COL_RES = 1
COL_YEAR = 2
COL_PK = 3  # -log10(Kd/Ki)
COL_KD = 4  # Kd/Ki in M
COL_CATEGORY = 6


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _pk_to_dg(pk: float, temp_k: float = 298.0) -> float:
    """
    Convert -log10(Kd) to ΔG (kcal/mol) using ΔG = RT * ln(10) * pK.

    ΔG = RT * ln(10) * pK ≈ 1.363 * pK (at 298K)
    R = 1.9872036e-3 kcal/(mol·K)
    ln(10) ≈ 2.302585
    """
    R = 1.9872036e-3
    return R * temp_k * math.log(10) * pk


def _parse_pdbbind_index(
    index_path: Path,
) -> List[Dict]:
    """
    Parse PDBbind INDEX file and extract protein-DNA entries.

    Returns a list of dicts with fields: pdb_id, pk, kd, dg (kcal/mol).
    """
    entries: List[Dict] = []
    with open(index_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            category = parts[COL_CATEGORY].strip().lower()
            if "protein-dna" not in category and "protein_dna" not in category:
                continue

            pdb_id = parts[COL_PDB_ID].strip().upper()
            try:
                pk = float(parts[COL_PK])
            except (ValueError, IndexError):
                continue

            dg = _pk_to_dg(pk)
            entries.append({
                "pdb_id": pdb_id,
                "pk": pk,
                "dg": dg,
            })

    return entries


def _find_pdb_files(pdb_dir: Path, pdb_ids: Set[str]) -> Dict[str, bool]:
    """Check which PDB IDs have structure files available."""
    result = {}
    for pdb_id in pdb_ids:
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        cif_path = pdb_dir / f"{pdb_id}.cif"
        result[pdb_id] = pdb_path.exists() or cif_path.exists()
    return result


def _resolve_structure_chain(
    pdb_path: Path, entity_type: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to determine protein/DNA chain IDs from a PDB file.
    This is a heuristic; for best results, the user should manually verify.

    Returns (prot_chain, dna_chain) or (None, None) if cannot determine.
    """
    if not pdb_path.exists():
        pdb_path = pdb_path.with_suffix(".cif")
        if not pdb_path.exists():
            return None, None

    protein_chain_ids: Set[str] = set()
    dna_chain_ids: Set[str] = set()
    protein_residues = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
    }
    dna_residues = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "T"}

    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # PDB format: columns 21-22 = chain ID
                    chain_id = line[21:22].strip()
                    if not chain_id:
                        continue
                    resname = line[17:20].strip().upper()
                    if resname in protein_residues:
                        protein_chain_ids.add(chain_id)
                    elif resname in dna_residues:
                        dna_chain_ids.add(chain_id)
    except Exception:
        return None, None

    if protein_chain_ids and dna_chain_ids:
        prot_chain = ",".join(sorted(protein_chain_ids))
        dna_chain = ",".join(sorted(dna_chain_ids))
        return prot_chain, dna_chain
    return None, None


def _assign_random_folds(
    pdb_ids: List[str], n_folds: int = NUM_FOLDS, seed: int = SEED
) -> Dict[str, List[str]]:
    """Assign each PDB to fold groups (one val fold per PDB, rest train)."""
    rng = np.random.default_rng(seed)
    assignments: Dict[str, List[str]] = {}
    for pdb in sorted(set(pdb_ids)):
        val_fold = int(rng.integers(0, n_folds))
        assignments[pdb] = [
            "val" if f == val_fold else "train" for f in range(n_folds)
        ]
    return assignments


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def prepare_dna_dg_from_pdbbind(
    pdbbind_dir: str,
    out_csv: str,
    pdb_out_dir: str,
    index_filename: str = "INDEX_general_PL_data.2020",
    num_folds: int = NUM_FOLDS,
    seed: int = SEED,
    download_pdb: bool = True,
) -> None:
    """
    Prepare DNA dG dataset from PDBbind.

    Steps:
      1. Read INDEX file for protein-DNA entries
      2. (Optionally) download missing PDB structures
      3. Resolve protein/DNA chain IDs from PDB files
      4. Write output CSV with fold splits
    """
    pdbbind_dir = Path(pdbbind_dir)
    out_csv = Path(out_csv)
    pdb_out_dir = Path(pdb_out_dir)

    index_path = pdbbind_dir / index_filename
    if not index_path.exists():
        # Try alternative patterns
        alt_patterns = [
            pdbbind_dir / "INDEX_general_PL_data*.txt",
            pdbbind_dir / "index" / index_filename,
        ]
        for pattern in alt_patterns:
            matches = list(Path(str(pattern).replace("*", "*")).parent.glob(
                str(pattern).split("/")[-1].replace("*", "*")
            )) if "*" in str(pattern) else []
            if matches:
                index_path = matches[0]
                break

    if not index_path.exists():
        print(
            f"[ERROR] PDBbind INDEX file not found at {index_path}.\n\n"
            "To obtain PDBbind:\n"
            "  1. Register at http://www.pdbbind.org.cn/\n"
            "  2. Download the 'general set' (free for academic use)\n"
            "  3. Extract to a directory and pass --pdbbind_dir\n\n"
            "You can also provide a custom CSV directly using --custom_csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Parsing PDBbind INDEX: {index_path}")
    entries = _parse_pdbbind_index(index_path)
    print(f"[INFO] Found {len(entries)} protein-DNA complexes in PDBbind.")

    # Check PDB availability
    pdb_out_dir.mkdir(parents=True, exist_ok=True)
    pdb_ids = {e["pdb_id"] for e in entries}
    pdb_base = pdbbind_dir / "protein_dna"  # common structure in PDBbind
    if not pdb_base.exists():
        pdb_base = pdbbind_dir  # fallback

    structure_map = {}
    for pdb_id in pdb_ids:
        pdb_path = pdb_base / f"{pdb_id}" / f"{pdb_id}_protein_dna.pdb"
        if not pdb_path.exists():
            pdb_path = pdb_out_dir / f"{pdb_id}.pdb"
        structure_map[pdb_id] = pdb_path if pdb_path.exists() else None

    available = sum(1 for v in structure_map.values() if v is not None)
    print(f"[INFO] {available}/{len(pdb_ids)} PDB structures found locally.")

    if download_pdb and available < len(pdb_ids):
        print("[INFO] Attempting to download missing PDB structures from rcsb.org ...")
        import urllib.request

        for pdb_id in pdb_ids:
            if structure_map.get(pdb_id) is not None:
                continue
            dest = pdb_out_dir / f"{pdb_id}.pdb"
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            try:
                urllib.request.urlretrieve(url, dest)
                structure_map[pdb_id] = dest
                print(f"  Downloaded {pdb_id}.pdb")
            except Exception as e:
                print(f"  Failed to download {pdb_id}: {e}")

    # Resolve chain IDs and write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fold_cols = [f"{COL_FOLD_PREFIX}{i}" for i in range(num_folds)]
    fold_assignments = _assign_random_folds(list(pdb_ids), num_folds, seed)

    written = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            COL_PDB, COL_PROT_CHAIN, COL_NA_CHAIN,
            COL_LABEL,
        ] + fold_cols
        writer.writerow(header)

        for entry in entries:
            pdb_id = entry["pdb_id"]
            pdb_path = structure_map.get(pdb_id)
            if pdb_path is None:
                continue

            prot_chain, dna_chain = _resolve_structure_chain(pdb_path, "prot_dna")
            if prot_chain is None or dna_chain is None:
                # Write with empty chains (user can fill manually)
                prot_chain = ""
                dna_chain = ""

            writer.writerow([
                pdb_id,
                prot_chain,
                dna_chain,
                f"{entry['dg']:.3f}",
            ] + fold_assignments[pdb_id])
            written += 1

    print(f"[INFO] Written {written} DNA dG entries to {out_csv}")
    print(f"[INFO] Fold columns: {fold_cols}")
    if written < len(entries):
        print(
            f"[WARN] {len(entries) - written} entries skipped due to missing PDBs."
        )
    print(
        "[NOTE] Chain IDs are heuristic. Please manually verify and update the CSV.\n"
        "You can use `grep '^ATOM' PDBs/XXXX.pdb | cut -c22,18-20 | sort -u` to check."
    )


def prepare_dna_dg_from_custom_csv(
    custom_csv: str,
    pdb_dir: str,
    out_csv: str,
    num_folds: int = NUM_FOLDS,
    seed: int = SEED,
) -> None:
    """
    Create DNA dG dataset from a user-provided CSV.

    Expected custom CSV columns:
        PDB, Protein chains, DNA chains, △G(kcal/mol) [, fold_0 ... fold_N]
    If fold columns are absent, they are generated.
    """
    import pandas as pd

    custom_csv = Path(custom_csv)
    out_csv = Path(out_csv)
    pdb_dir = Path(pdb_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(custom_csv)
    required = [COL_PDB, COL_LABEL]
    for col in required:
        if col not in df.columns:
            print(
                f"[ERROR] Custom CSV missing required column '{col}'. "
                f"Got: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Auto-fill chain columns if missing
    if COL_PROT_CHAIN not in df.columns:
        df[COL_PROT_CHAIN] = ""
    if COL_NA_CHAIN not in df.columns:
        df[COL_NA_CHAIN] = ""

    # Add fold columns if missing
    fold_cols = [f"{COL_FOLD_PREFIX}{i}" for i in range(num_folds)]
    existing_folds = [c for c in fold_cols if c in df.columns]
    if len(existing_folds) < num_folds:
        pdb_ids = df[COL_PDB].tolist()
        assignments = _assign_random_folds(pdb_ids, num_folds, seed)
        for i in range(num_folds):
            col = f"{COL_FOLD_PREFIX}{i}"
            df[col] = [assignments[p][i] for p in pdb_ids]

    # Check PDB availability and filter
    if COL_PROT_CHAIN in df.columns and COL_NA_CHAIN in df.columns:
        has_empty_chains = (
            df[COL_PROT_CHAIN].isna() | (df[COL_PROT_CHAIN] == "")
        ).any()
        if has_empty_chains:
            print(
                "[WARN] Some entries have empty chain IDs. "
                "The pipeline will auto-detect chain IDs from PDB files."
            )

    df.to_csv(out_csv, index=False)
    print(f"[INFO] Written {len(df)} DNA dG entries to {out_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare DNA dG (protein-DNA binding affinity) dataset"
    )
    parser.add_argument(
        "--pdbbind_dir",
        default=None,
        help="Path to PDBbind directory (containing INDEX file and structure subdirs)",
    )
    parser.add_argument(
        "--index_filename",
        default="INDEX_general_PL_data.2020",
        help="Filename of the PDBbind INDEX file",
    )
    parser.add_argument(
        "--custom_csv",
        default=None,
        help="Use a custom CSV instead of PDBbind (columns: PDB, Protein chains, DNA chains, △G(kcal/mol))",
    )
    parser.add_argument(
        "--pdb_dir",
        default=DEFAULT_PDB_DIR,
        help="Directory for PDB structures",
    )
    parser.add_argument(
        "--out_csv",
        default=DEFAULT_OUT_CSV,
        help="Output CSV path",
    )
    parser.add_argument(
        "--num_folds", type=int, default=NUM_FOLDS, help="Number of CV folds"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for fold splits"
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Skip downloading missing PDBs from RCSB",
    )

    args = parser.parse_args()

    if args.custom_csv:
        prepare_dna_dg_from_custom_csv(
            custom_csv=args.custom_csv,
            pdb_dir=args.pdb_dir,
            out_csv=args.out_csv,
            num_folds=args.num_folds,
            seed=args.seed,
        )
    elif args.pdbbind_dir:
        prepare_dna_dg_from_pdbbind(
            pdbbind_dir=args.pdbbind_dir,
            out_csv=args.out_csv,
            pdb_out_dir=args.pdb_dir,
            index_filename=args.index_filename,
            num_folds=args.num_folds,
            seed=args.seed,
            download_pdb=not args.no_download,
        )
    else:
        print(
            "[ERROR] Provide either --pdbbind_dir or --custom_csv.",
            file=sys.stderr,
        )
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
