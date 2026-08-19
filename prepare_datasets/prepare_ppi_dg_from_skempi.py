#!/usr/bin/env python3
"""
Prepare PPI dG (binding affinity) dataset from SKEMPI 2.0.

Downloads the SKEMPI 2.0 database, extracts wild-type binding free energy (ΔG)
for all unique PPI complexes, and outputs a CSV compatible with the IMF-Net
structure_dataset pipeline for PPI (entity_type=ppi).

The user already has the PDB structures in:
    /media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/PDBs/

Usage:
    python prepare_datasets/prepare_ppi_dg_from_skempi.py \\
        --skempi_url https://life.bsc.es/pid/skempi2/download/skempi_v2.csv \\
        --pdb_dir /media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/PDBs \\
        --out_csv /media/SSD0/csd/lrg/copra_h/datasets/PPI_dG/splits/PPI_dG.csv

If --skempi_url is omitted, the script looks for a local copy at
SKEMPI/skempi_v2.csv or SKEMPI/splits/skempi_v2.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKEMPI_V2_FILENAME = "skempi_v2.csv"
NUM_FOLDS = 5
SEED = 2024

# Mapping: SKEMPI 2.0 column names → our expected column names
# We'll pick up key fields from the SKEMPI CSV and rename.
COL_OUR_PDB = "PDB"
COL_OUR_PROT_CHAIN = "Protein chains"
COL_OUR_PARTNER_CHAIN = "Protein chains B"
COL_OUR_PROT_SEQ = "Protein sequences"
COL_OUR_PARTNER_SEQ = "Protein sequences B"
COL_OUR_LABEL = "△G(kcal/mol)"
COL_OUR_ENTITY_TYPE = "entity_type"

# Expected SKEMPI 2.0 CSV columns (v2)
SKEMPI_COL_PDB = "pdb_id"
SKEMPI_COL_PROT1 = "protein_1"  # name only, not used
SKEMPI_COL_PROT2 = "protein_2"
SKEMPI_COL_CHAIN1 = "chain_1"  # e.g., "A"
SKEMPI_COL_CHAIN2 = "chain_2"  # e.g., "B"
SKEMPI_COL_SEQ1 = "seq_1"  # WT sequence of protein 1
SKEMPI_COL_SEQ2 = "seq_2"  # WT sequence of protein 2
SKEMPI_COL_AFFINITY_WT = "affinity_wt"  # M⁻¹
SKEMPI_COL_AFFINITY_TYPE = "affinity_type"  # Kd, Ki, IC50
SKEMPI_COL_AFFINITY_UNIT = "affinity_unit"  # M⁻¹ (note the typo in original)
SKEMPI_COL_TEMP = "temperature"  # K


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _parse_kd_to_dg(kd_m: Optional[float], temp_k: float = 298.0) -> Optional[float]:
    """
    Convert equilibrium dissociation constant Kd (M) to ΔG (kcal/mol).

    ΔG = RT ln(Kd)   (R = 1.9872036e-3 kcal/(mol·K))
    If Kd is in M⁻¹ (association constant Ka), convert first: Kd = 1 / Ka.

    Returns ΔG in kcal/mol, or None if conversion is impossible.
    """
    if kd_m is None or kd_m <= 0:
        return None
    if math.isinf(kd_m) or math.isnan(kd_m):
        return None
    R = 1.9872036e-3  # kcal/(mol·K)
    return R * temp_k * math.log(kd_m)


def _compute_dg_from_affinity(
    affinity_wt: Optional[float],
    affinity_type: str,
    temp_k: float = 298.0,
) -> Optional[float]:
    """
    Convert SKEMPI affinity values to ΔG (kcal/mol).

    SKEMPI 2.0 stores `affinity_wt` in M⁻¹ (*association* constant Ka).
    Some entries may have different units or types.

    ΔG = -RT ln(Ka) = RT ln(Kd) = RT ln(1/Ka)
    ΔG = -RT ln(Ka)
    """
    if affinity_wt is None or affinity_wt <= 0:
        return None
    if math.isinf(affinity_wt) or math.isnan(affinity_wt):
        return None
    R = 1.9872036e-3  # kcal/(mol·K)
    atype = str(affinity_type).strip().lower() if affinity_type else "kd"
    if atype in ("kd", "ki", "ic50"):
        # affinity_wt is Kd (M). ΔG = RT ln(Kd)
        dg = R * temp_k * math.log(affinity_wt)
    else:
        # affinity_wt is Ka (M⁻¹). ΔG = -RT ln(Ka)
        dg = -R * temp_k * math.log(affinity_wt)
    return dg


def _make_fold_columns(n_folds: int = NUM_FOLDS, seed: int = SEED) -> List[str]:
    """Generate fold column names (fold_0, ..., fold_{N-1})."""
    return [f"fold_{i}" for i in range(n_folds)]


def _assign_random_folds(
    complexes: List[str], n_folds: int = NUM_FOLDS, seed: int = SEED
) -> Dict[str, List[str]]:
    """
    Assign each PDB complex to n_folds random fold groups deterministically.
    Returns {pdb_id: [fold_0_label, ..., fold_{N-1}_label]} where each label
    is 'train', 'val', or 'test'.
    We use 1 held-out fold for val and the rest for train.
    """
    rng = np.random.default_rng(seed)
    pdb_list = sorted(set(complexes))
    assignments: Dict[str, List[str]] = {}
    for pdb in pdb_list:
        val_fold = int(rng.integers(0, n_folds))
        fold_labels = []
        for f in range(n_folds):
            if f == val_fold:
                fold_labels.append("val")
            else:
                fold_labels.append("train")
        assignments[pdb] = fold_labels
    return assignments


# ---------------------------------------------------------------------------
# Main preparation logic
# ---------------------------------------------------------------------------

def prepare_ppi_dg(
    skempi_path: str,
    pdb_dir: str,
    output_csv: str,
    skempi_url: Optional[str] = None,
    num_folds: int = NUM_FOLDS,
    seed: int = SEED,
    force_download: bool = False,
) -> None:
    """
    Main function to prepare PPI dG dataset.

    Steps:
      1. Download/copy SKEMPI v2 CSV
      2. Parse to extract unique WT PPI complexes with ΔG
      3. Cross-reference against existing PDB files
      4. Write output CSV with fold splits
    """
    skempi_path = Path(skempi_path)
    pdb_dir = Path(pdb_dir)
    output_csv = Path(output_csv)

    # -- Step 1: acquire SKEMPI v2 CSV ----------------------------------------
    if not skempi_path.exists() or force_download:
        if skempi_url is None:
            print(
                f"[ERROR] SKEMPI v2 CSV not found at {skempi_path} and "
                "--skempi_url not provided. Download it manually from "
                "https://life.bsc.es/pid/skempi2/download/ or provide the URL.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[INFO] Downloading SKEMPI v2 from {skempi_url} ...")
        urllib.request.urlretrieve(skempi_url, skempi_path)
        print(f"[INFO] Downloaded to {skempi_path}")
    else:
        print(f"[INFO] Using existing SKEMPI v2 CSV: {skempi_path}")

    # -- Step 2: parse CSV ----------------------------------------------------
    print("[INFO] Parsing SKEMPI v2 CSV for WT PPI complexes ...")
    unique_complexes: Dict[str, dict] = {}  # pdb_id → dict of metadata
    rows_seen = 0
    with open(skempi_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_seen += 1
            pdb_id = row.get(SKEMPI_COL_PDB, "").strip().upper()
            if not pdb_id:
                continue

            # Only process PPI (two protein chains)
            chain1 = row.get(SKEMPI_COL_CHAIN1, "").strip()
            chain2 = row.get(SKEMPI_COL_CHAIN2, "").strip()
            if not chain1 or not chain2:
                continue

            seq1 = row.get(SKEMPI_COL_SEQ1, "").strip()
            seq2 = row.get(SKEMPI_COL_SEQ2, "").strip()
            if not seq1 or not seq2:
                continue

            # Compute ΔG from binding affinity
            try:
                aff_wt = float(row.get(SKEMPI_COL_AFFINITY_WT, "").strip() or "0")
            except (ValueError, TypeError):
                aff_wt = None
            if aff_wt is None or aff_wt <= 0:
                continue

            try:
                temp_k = float(row.get(SKEMPI_COL_TEMP, "").strip() or "298")
            except (ValueError, TypeError):
                temp_k = 298.0

            affinity_type = row.get(SKEMPI_COL_AFFINITY_TYPE, "kd").strip()
            dg = _compute_dg_from_affinity(aff_wt, affinity_type, temp_k)
            if dg is None or not math.isfinite(dg):
                continue

            # Check if PDB exists locally
            pdb_path = pdb_dir / f"{pdb_id}.pdb"
            cif_path = pdb_dir / f"{pdb_id}.cif"
            has_struct = pdb_path.exists() or cif_path.exists()

            # Keep the first (or best) entry per PDB; prefer more chains match
            if pdb_id not in unique_complexes:
                unique_complexes[pdb_id] = {
                    "pdb_id": pdb_id,
                    "prot_chain": chain1,
                    "partner_chain": chain2,
                    "prot_seq": seq1,
                    "partner_seq": seq2,
                    "dg": dg,
                    "has_struct": has_struct,
                    "n_mutations": 0,
                    "temp_k": temp_k,
                }
            else:
                existing = unique_complexes[pdb_id]
                existing["n_mutations"] += 1
                # Prefer entries where both chains match our expectation
                # (no special handling needed)

    print(
        f"[INFO] Found {len(unique_complexes)} unique PPI complexes "
        f"with valid ΔG out of {rows_seen} mutation rows."
    )

    # Filter to those with PDB files
    with_struct = {k: v for k, v in unique_complexes.items() if v["has_struct"]}
    print(
        f"[INFO] {len(with_struct)} complexes have PDB files in {pdb_dir}. "
        f"{len(unique_complexes) - len(with_struct)} lack PDBs and will be skipped."
    )

    if len(with_struct) == 0:
        print(
            "[WARN] No PPI complexes with PDB files found. "
            "Make sure PDBs are in the expected directory.",
            file=sys.stderr,
        )
        # We'll still write whatever we have

    # -- Step 3: assign fold splits -------------------------------------------
    all_pdbs = list(with_struct.keys())
    fold_assignments = _assign_random_folds(all_pdbs, n_folds=num_folds, seed=seed)

    # -- Step 4: write output CSV ---------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fold_cols = _make_fold_columns(num_folds)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            COL_OUR_PDB,
            COL_OUR_PROT_CHAIN,
            COL_OUR_PARTNER_CHAIN,
            COL_OUR_PROT_SEQ,
            COL_OUR_PARTNER_SEQ,
            COL_OUR_LABEL,
            COL_OUR_ENTITY_TYPE,
        ] + fold_cols
        writer.writerow(header)

        for pdb_id in sorted(all_pdbs):
            entry = with_struct[pdb_id]
            writer.writerow(
                [
                    entry["pdb_id"],
                    entry["prot_chain"],
                    entry["partner_chain"],
                    entry["prot_seq"],
                    entry["partner_seq"],
                    f"{entry['dg']:.3f}",
                    "ppi",
                ]
                + fold_assignments[pdb_id]
            )

    print(f"[INFO] Written {len(all_pdbs)} PPI dG entries to {output_csv}")
    print(f"[INFO] Fold columns: {fold_cols}")
    print(f"[INFO] ΔG range: {min(e['dg'] for e in with_struct.values()):.2f} – "
          f"{max(e['dg'] for e in with_struct.values()):.2f} kcal/mol")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare PPI dG dataset from SKEMPI 2.0"
    )
    parser.add_argument(
        "--skempi_url",
        default=None,
        help="URL to skempi_v2.csv (if not provided, looks locally)",
    )
    parser.add_argument(
        "--skempi_csv",
        default=None,
        help=(
            "Path to local skempi_v2.csv. "
            "If omitted, tries SKEMPI/skempi_v2.csv then SKEMPI/splits/skempi_v2.csv"
        ),
    )
    parser.add_argument(
        "--pdb_dir",
        default="/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/PDBs",
        help="Directory containing SKEMPI PDB structures",
    )
    parser.add_argument(
        "--out_csv",
        default="/media/SSD0/csd/lrg/copra_h/datasets/PPI_dG/splits/PPI_dG.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--num_folds", type=int, default=NUM_FOLDS, help="Number of CV folds"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for fold splits"
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download SKEMPI v2 CSV even if it exists locally",
    )

    args = parser.parse_args()

    # Resolve SKEMPI CSV path
    skempi_csv = args.skempi_csv
    if skempi_csv is None:
        base = Path("/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI")
        candidates = [
            base / SKEMPI_V2_FILENAME,
            base / "splits" / SKEMPI_V2_FILENAME,
        ]
        for c in candidates:
            if c.exists():
                skempi_csv = str(c)
                break
        if skempi_csv is None:
            skempi_csv = str(candidates[0])
            print(
                f"[INFO] SKEMPI v2 CSV not found locally; will download to {skempi_csv}"
            )

    prepare_ppi_dg(
        skempi_path=skempi_csv,
        pdb_dir=args.pdb_dir,
        output_csv=args.out_csv,
        skempi_url=args.skempi_url,
        num_folds=args.num_folds,
        seed=args.seed,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
