#!/usr/bin/env python3
"""Build COPRA-style copra CSV for PD304 (train) and PD36 (test).

PD datasets are protein-DNA dG (absolute binding free energy), no mutations.
The CSV needs: PDB, Protein chains, DNA chains, label, fold_0 (train/val/test)
so it matches the structure_dataset / DataModule contract.

Chains are auto-resolved from each PDB file (heuristic, same as prepare_dna_dg).
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

PROT_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
DNA_RES = {"DA", "DC", "DG", "DT", "DU", "A", "C", "G", "T", "U"}


def resolve_chains(pdb_path: Path):
    prot, dna = set(), set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if len(line) < 22:
                    continue
                chain = line[21:22].strip()
                res = line[17:20].strip().upper()
                if not chain:
                    continue
                if res in PROT_RES:
                    prot.add(chain)
                elif res in DNA_RES:
                    dna.add(chain)
    except Exception:
        pass
    return (",".join(sorted(prot)), ",".join(sorted(dna)))


def build(src_csv, pdb_dir, out_csv, label_col, split_mode):
    pdb_dir = Path(pdb_dir)
    df = {}
    with open(src_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            pid = row["PDB_ID"].strip()
            df[pid] = row
    pids = sorted(df.keys())

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PDB", "Protein chains", "DNA chains", "△G(kcal/mol)", "fold_0"])
        for pid in pids:
            pdb_path = pdb_dir / f"{pid}.pdb"
            if not pdb_path.exists():
                print(f"[WARN] missing PDB {pdb_path}, skip")
                continue
            prot, dna = resolve_chains(pdb_path)
            if not prot or not dna:
                print(f"[WARN] {pid}: prot={prot!r} dna={dna!r} (empty), skip")
                continue
            label = float(df[pid][label_col])
            if split_mode == "train":
                fold = "train"
            elif split_mode == "test":
                fold = "test"
            else:
                raise ValueError(split_mode)
            w.writerow([pid, prot, dna, f"{label:.4f}", fold])
    print(f"[INFO] wrote {out_csv} ({len(pids)} entries, mode={split_mode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_csv", required=True)
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--label_col", default="dG_kcal_per_mol")
    ap.add_argument("--split_mode", choices=["train", "test"], required=True)
    a = ap.parse_args()
    build(a.src_csv, a.pdb_dir, a.out_csv, a.label_col, a.split_mode)


if __name__ == "__main__":
    main()
