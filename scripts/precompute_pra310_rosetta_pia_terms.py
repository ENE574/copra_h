#!/usr/bin/env python3
"""
Precompute whole-complex Rosetta score terms for PRA310, written under **FoldX-aligned**
PIA column names (``Electro``, ``Energy_SolvP``, ``Energy_SolvH``, ``Energy_VdW``).

Uses the same Rosetta→label bridge as ``data/pyrosetta_physics.rosetta_terms_to_foldx_pia_labels``.

Designed to run in a minimal conda env, e.g.:
  conda activate pyrosetta_only
  python scripts/precompute_pra310_rosetta_pia_terms.py \\
    --csv /media/SSD0/csd/lrg/datasets/PRA310/splits/PRA310.csv \\
    --data-root /media/SSD0/csd/lrg/datasets/PRA310/PDBs \\
    --output /media/SSD0/csd/lrg/datasets/PRA310/splits/PRA310_rosetta_pia_terms.csv

Requires PyRosetta. Default init adds ``-ignore_unrecognized_res``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data.pia_physics_names import PIA_PHYSICS_NAMES  # noqa: E402
from data.pyrosetta_physics import rosetta_terms_to_foldx_pia_labels  # noqa: E402


def _score_terms(pose, sfxn) -> dict[str, float]:
    from pyrosetta.rosetta.core.scoring import ScoreType

    sfxn(pose)
    emap = pose.energies().total_energies()
    out: dict[str, float] = {}
    for name in ("fa_elec", "fa_sol", "fa_atr", "fa_rep"):
        if not hasattr(ScoreType, name):
            out[name] = 0.0
            continue
        st = getattr(ScoreType, name)
        try:
            out[name] = float(emap[st])
        except Exception:
            out[name] = 0.0
    return out


def resolve_pdb_path(data_root: Path, structure_id: str, mut: bool) -> Path:
    if mut:
        base = structure_id.split("_")[0]
        return data_root / f"{base}.pdb"
    return data_root / f"{structure_id}.pdb"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Score PDBs with PyRosetta; write FoldX-aligned PIA columns (Electro, Energy_SolvP, ...)."
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/datasets/PRA310/splits/PRA310.csv"),
        help="Dataset CSV (must contain PDB column; optional MUTATION if --mut).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/datasets/PRA310/PDBs"),
        help="Directory containing <PDB>.pdb (same as data_root in PRA310.yml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/datasets/PRA310/splits/PRA310_rosetta_pia_terms.csv"),
        help="Output CSV path.",
    )
    p.add_argument("--pdb-col", default="PDB", help="Column name for PDB id (default: PDB).")
    p.add_argument(
        "--mut",
        action="store_true",
        help="Mutation mode: structure id = PDB + '_' + MUTATION column; PDB file = <PDB>.pdb (base id).",
    )
    p.add_argument("--scorefxn", default="ref2015", help="Rosetta scorefunction name (default: ref2015).")
    p.add_argument(
        "--extra-options",
        default="-mute all -ignore_unrecognized_res",
        help="Extra PyRosetta init() options (default: -mute all -ignore_unrecognized_res).",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1
    if not args.data_root.is_dir():
        print(f"ERROR: data root not a directory: {args.data_root}", file=sys.stderr)
        return 1

    try:
        from pyrosetta import init, pose_from_file, create_score_function
    except ImportError:
        print("ERROR: pyrosetta not importable. Activate pyrosetta_only (or install PyRosetta).", file=sys.stderr)
        return 1

    init(extra_options=args.extra_options.strip())
    sfxn = create_score_function(args.scorefxn)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["sample_id", "pdb_path"] + list(PIA_PHYSICS_NAMES) + ["ok", "error"]

    cache: dict[str, tuple[dict[str, float], str, str]] = {}

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None  # type: ignore

    with open(args.csv, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None or args.pdb_col not in reader.fieldnames:
            print(f"ERROR: column {args.pdb_col!r} missing from CSV. Found: {reader.fieldnames}", file=sys.stderr)
            return 1
        if args.mut and "MUTATION" not in reader.fieldnames:
            print("ERROR: --mut requires a MUTATION column in the CSV.", file=sys.stderr)
            return 1

        rows = list(reader)

    iterator = rows
    if _tqdm is not None:
        iterator = _tqdm(rows, desc="Scoring")

    with open(args.output, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in iterator:
            pdb_id = (row.get(args.pdb_col) or "").strip()
            if not pdb_id:
                row_out = {c: "" for c in PIA_PHYSICS_NAMES}
                row_out.update({"sample_id": "", "pdb_path": "", "ok": "0", "error": "empty_pdb_id"})
                writer.writerow(row_out)
                continue

            structure_id = pdb_id
            if args.mut:
                mut = (row.get("MUTATION") or "").strip()
                structure_id = f"{pdb_id}_{mut}" if mut else pdb_id

            pdb_path = resolve_pdb_path(args.data_root, structure_id, args.mut)
            key = str(pdb_path.resolve())

            if key in cache:
                terms, ok, err = cache[key]
            elif not pdb_path.is_file():
                terms = {"fa_elec": 0.0, "fa_sol": 0.0, "fa_atr": 0.0, "fa_rep": 0.0}
                err = "missing_pdb_file"
                ok = "0"
                cache[key] = (terms, ok, err)
            else:
                try:
                    pose = pose_from_file(str(pdb_path))
                    terms = _score_terms(pose, sfxn)
                    err = ""
                    ok = "1"
                    cache[key] = (terms, ok, err)
                except Exception as e:
                    terms = {"fa_elec": 0.0, "fa_sol": 0.0, "fa_atr": 0.0, "fa_rep": 0.0}
                    err = repr(e)
                    ok = "0"
                    cache[key] = (terms, ok, err)

            pia = rosetta_terms_to_foldx_pia_labels(terms)
            row_out = {c: pia[c] for c in PIA_PHYSICS_NAMES}
            row_out.update(
                {
                    "sample_id": structure_id,
                    "pdb_path": str(pdb_path),
                    "ok": ok,
                    "error": err,
                }
            )
            writer.writerow(row_out)

    print(f"Wrote {args.output} ({len(rows)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
