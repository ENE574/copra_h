#!/usr/bin/env python3
"""
Subset PRA310_foldx_pia_terms.csv to the PDBs listed in PRA201.csv (no FoldX rerun).

Example:

  python scripts/subset_pra201_foldx_pia_terms.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Write PRA201_foldx_pia_terms.csv from PRA310 FoldX CSV + PRA201 split.")
    p.add_argument(
        "--pra201-csv",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/datasets/PRA310/splits/PRA201.csv"),
    )
    p.add_argument(
        "--full-foldx-csv",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/datasets/PRA310/splits/PRA310_foldx_pia_terms.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/datasets/PRA310/splits/PRA201_foldx_pia_terms.csv"),
    )
    p.add_argument("--pdb-col", default="PDB", help="Column name in PRA201 CSV for structure id (matches sample_id).")
    args = p.parse_args()

    for path, label in (
        (args.pra201_csv, "PRA201 CSV"),
        (args.full_foldx_csv, "full FoldX CSV"),
    ):
        if not path.is_file():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 1

    with args.full_foldx_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        if not fieldnames or "sample_id" not in fieldnames:
            print("ERROR: full FoldX CSV must have a sample_id column.", file=sys.stderr)
            return 1
        by_sid: dict[str, dict[str, str]] = {}
        for row in r:
            sid = (row.get("sample_id") or "").strip()
            if sid:
                by_sid[sid.upper()] = row

    with args.pra201_csv.open(newline="", encoding="utf-8") as f:
        pr = csv.DictReader(f)
        if args.pdb_col not in (pr.fieldnames or []):
            print(f"ERROR: column {args.pdb_col!r} not in {args.pra201_csv}", file=sys.stderr)
            return 1
        order = [(row.get(args.pdb_col) or "").strip().upper() for row in pr]

    if not order:
        print("ERROR: no rows in PRA201 CSV.", file=sys.stderr)
        return 1

    missing = [sid for sid in order if sid not in by_sid]
    if missing:
        print(f"ERROR: {len(missing)} PDB(s) from PRA201 not found in full FoldX CSV, e.g.: {missing[:8]}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for sid in order:
            w.writerow(by_sid[sid])

    print(f"Wrote {len(order)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
