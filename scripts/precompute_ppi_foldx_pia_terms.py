#!/usr/bin/env python3
"""FoldX ``Stability`` PIA labels for protein–protein datasets (S79, S90, P2P).

These are wild-type complexes (no mutations), so the physical supervision unit is
the **PDB structure** (the whole complex). For every ``*.pdb`` in ``--data-root`` we
run FoldX ``--command=Stability`` and write one row with the five PIA physics terms
used as self-supervision heads in the model:

    Electro, Energy_SolvP, Energy_SolvH, Energy_VdW, Energy_Hbond

This mirrors ``scripts/precompute_pra310_foldx_pia_terms.py`` but (a) iterates over
PDB files (not CSV rows, since the complex is the supervision unit), (b) omits
``--complexWithRNA`` for pure protein–protein complexes, and (c) matches the emitted
``*_ST.fxout`` back to the input PDB stem (some PDBs contain multiple MODELs and
FoldX emits several fxouts).

Output CSV columns: sample_id, pdb_path, <5 PIA terms>, ok, error.
sample_id = PDB stem (matches how the offline loader keys physics targets).

Examples (run from repo root /home/csd/lrg/copra_h):

  python scripts/precompute_ppi_foldx_pia_terms.py --dataset S79
  python scripts/precompute_ppi_foldx_pia_terms.py --dataset S90 --workers 8
  python scripts/precompute_ppi_foldx_pia_terms.py --dataset P2P --workers 8 --resume
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA = Path("/media/SSD0/csd/lrg/copra_h/datasets")
_DEFAULT_FOLDX = _REPO_ROOT / "foldx" / "foldx_20270131"
_DEFAULT_MOLECULES = _REPO_ROOT / "foldx" / "molecules"


# --------------------------------------------------------------------------- #
# FoldX physics helpers (standalone import, no repo on sys.path needed)
# --------------------------------------------------------------------------- #
def _load_foldx_physics():
    p = _REPO_ROOT / "data" / "foldx_physics.py"
    spec = importlib.util.spec_from_file_location("_foldx_physics_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_fx = _load_foldx_physics()
parse_foldx_fxout = _fx.parse_foldx_fxout
foldx_terms_to_pia_targets = _fx.foldx_terms_to_pia_targets
_try_parse_st_line = _fx._try_parse_foldx5_stability_st_line
# For pure protein–protein complexes, the five protein PIA terms are the
# physical supervision targets (the DNA_* terms only apply to protein–DNA).
PPI_PIA_COLS = list(_fx.PIA_PHYSICS_NAMES[:5])
PIA_COLS = PPI_PIA_COLS

# Per-dataset defaults (CSV is only used to discover the canonical PDB dir here;
# the supervision unit is the PDB file itself).
_DATASET_DEFAULTS = {
    "S79": {
        "pdb_dir": _DATA / "S79" / "PDBs",
        "output": _DATA / "S79" / "splits" / "S79_foldx_pia_terms.csv",
        "complex_with_rna": False,
    },
    "S90": {
        "pdb_dir": _DATA / "S90" / "PDBs",
        "output": _DATA / "S90" / "splits" / "S90_foldx_pia_terms.csv",
        "complex_with_rna": False,
    },
    "P2P": {
        "pdb_dir": _DATA / "P2P" / "PDBs",
        "output": _DATA / "P2P" / "splits" / "P2P_foldx_pia_terms.csv",
        "complex_with_rna": False,
    },
}


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _find_matching_st_line(run_dir: Path, stem: str):
    """Return the parsed PIA dict for the ST line whose pdb field references ``stem``.

    FoldX may emit several ``*_ST.fxout`` when a PDB contains multiple MODELs; we
    scan every candidate fxout and select the line whose first field contains
    ``{stem}.pdb``. ``parse_foldx_fxout`` already returns exactly the five protein
    PIA keys, so we use it directly (FoldX's ``foldx_terms_to_pia_targets`` expects
    the full 8-term set incl. DNA and would zero out pure-protein dicts).
    """
    cands = sorted(run_dir.glob("*_ST.fxout"))
    if not cands:
        cands = sorted(run_dir.glob("Average*.fxout")) or sorted(run_dir.glob("*.fxout"))
    target = f"{stem.lower()}.pdb"
    for fx in cands:
        try:
            # Prefer the line explicitly referencing this pdb stem.
            for line in fx.read_text(encoding="utf-8", errors="replace").splitlines():
                first = line.split("\t", 1)[0].strip().lower()
                if target in first:
                    parsed = _try_parse_st_line(line)
                    if parsed is not None:
                        return parsed, fx
            # Fallback: first parseable line in this fxout.
            parsed = parse_foldx_fxout(fx)
            if parsed:
                return parsed, fx
        except Exception:
            continue
    return None, None


def run_one(args_tuple):
    foldx, molecules, pdb_path, run_dir, command, timeout_s, extra, keep_runs = args_tuple
    stem = pdb_path.stem
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        if molecules.is_dir() and not (run_dir / "molecules").exists():
            shutil.copytree(molecules, run_dir / "molecules")
        local_pdb = run_dir / pdb_path.name
        shutil.copy2(pdb_path, local_pdb)

        cmd = [str(foldx), f"--command={command}", f"--pdb={local_pdb.name}"] + list(extra)
        r = subprocess.run(cmd, cwd=str(run_dir), capture_output=True, text=True,
                           timeout=timeout_s, check=False)
        if r.returncode != 0:
            return stem, None, f"exit_{r.returncode}:{(r.stderr or r.stdout or '')[-1500:]}"

        pia, fx = _find_matching_st_line(run_dir, stem)
        if pia is None:
            return stem, None, "no_fxout_output"
        return stem, pia, ""
    except subprocess.TimeoutExpired:
        return stem, None, "timeout"
    except Exception as e:  # noqa: BLE001
        return stem, None, repr(e)
    finally:
        # Keep work dirs by default: deleting thousands of dirs per turn trips the
        # environment's bulk-delete guard. Clean up outputs/foldx_ppi_runs once at the end.
        if not keep_runs:
            shutil.rmtree(run_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(_DATASET_DEFAULTS))
    p.add_argument("--foldx", type=Path, default=_DEFAULT_FOLDX)
    p.add_argument("--molecules", type=Path, default=_DEFAULT_MOLECULES)
    p.add_argument("--pdb-dir", type=Path, default=None, help="Override PDB dir.")
    p.add_argument("--output", type=Path, default=None, help="Override output CSV.")
    p.add_argument("--command", default="Stability")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--work-root", type=Path,
                   default=_REPO_ROOT / "outputs" / "foldx_ppi_runs")
    p.add_argument("--resume", action="store_true",
                   help="Skip PDB stems already present in the output CSV.")
    p.add_argument("--limit", type=int, default=0, help="Debug: only process N PDBs.")
    p.add_argument("--keep-runs", action="store_true", default=True,
                   help="Keep per-PDB FoldX work dirs (default: keep, to avoid the "
                        "bulk-delete guard). Pass --no-keep-runs to clean up per PDB.")
    p.add_argument("--no-keep-runs", dest="keep_runs", action="store_false")
    args = p.parse_args()

    if not args.foldx.is_file():
        print(f"ERROR: FoldX binary not found: {args.foldx}", file=sys.stderr)
        return 1

    dcfg = _DATASET_DEFAULTS[args.dataset]
    pdb_dir = args.pdb_dir or dcfg["pdb_dir"]
    output = args.output or dcfg["output"]
    if not pdb_dir.is_dir():
        print(f"ERROR: pdb dir not found: {pdb_dir}", file=sys.stderr)
        return 1

    extra = []
    if not dcfg["complex_with_rna"] and all("complexWithRNA" not in x for x in extra):
        # pure protein–protein: do NOT pass --complexWithRNA
        pass

    output.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    pdbs = sorted(pdb_dir.glob("*.pdb"))
    if args.limit:
        pdbs = pdbs[: args.limit]

    done: set[str] = set()
    if args.resume and output.is_file():
        with open(output, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("ok") == "1":
                    done.add(row["sample_id"])

    pending = [pb for pb in pdbs if pb.stem not in done]
    print(f"[{args.dataset}] {len(pdbs)} PDBs, {len(pending)} pending"
          + (f" ({len(done)} done, resume)" if done else ""))

    fieldnames = ["sample_id", "pdb_path"] + PIA_COLS + ["ok", "error"]

    # Write header now; append results as they complete.
    with open(output, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        def _emit(stem, pia, err, pb):
            if pia is None:
                row_out = {c: 0.0 for c in PIA_COLS}
                row_out.update({"sample_id": stem, "pdb_path": str(pb),
                                "ok": "0", "error": (err or "foldx_failed")[:5000]})
            else:
                row_out = {c: pia[c] for c in PIA_COLS}
                row_out.update({"sample_id": stem, "pdb_path": str(pb), "ok": "1", "error": ""})
            writer.writerow(row_out)
            f_out.flush()

        if args.workers <= 1:
            for pb in pending:
                run_dir = args.work_root / f"{pb.stem}"
                stem, pia, err = run_one(
                    (args.foldx, args.molecules, pb, run_dir, args.command, args.timeout, extra, args.keep_runs))
                _emit(stem, pia, err, pb)
        else:
            tasks = [(args.foldx, args.molecules, pb, args.work_root / f"{pb.stem}",
                      args.command, args.timeout, extra, args.keep_runs) for pb in pending]
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_one, t): t[2] for t in tasks}
                for fut in as_completed(futs):
                    pb = futs[fut]
                    stem, pia, err = fut.result()
                    _emit(stem, pia, err, pb)

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
