#!/usr/bin/env python3
"""
Run bundled FoldX 5 (``foldx/foldx_20270131``) on each PRA310 PDB and write PIA label CSV.

Uses ``--command=Stability`` and by default ``--complexWithRNA=1`` for protein–RNA complexes.
Parses ``*_ST.fxout`` via ``data/foldx_physics.parse_foldx_fxout``.

Example (from repo root):

  python scripts/precompute_pra310_foldx_pia_terms.py

Override paths only if needed:

  python scripts/precompute_pra310_foldx_pia_terms.py \\
    --foldx /path/to/FoldX \\
    --csv ... --data-root ... --output ...
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_foldx_physics():
    import importlib.util

    p = _REPO_ROOT / "data" / "foldx_physics.py"
    spec = importlib.util.spec_from_file_location("_foldx_physics_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_fx = _load_foldx_physics()
parse_foldx_fxout = _fx.parse_foldx_fxout
foldx_terms_to_pia_targets = _fx.foldx_terms_to_pia_targets
PIA_COLS = list(_fx.PIA_PHYSICS_NAMES)

_DEFAULT_FOLDX = _REPO_ROOT / "foldx" / "foldx_20270131"
_DEFAULT_MOLECULES = _REPO_ROOT / "foldx" / "molecules"


def resolve_pdb_path(data_root: Path, structure_id: str, mut: bool) -> Path:
    if mut:
        base = structure_id.split("_")[0]
        return data_root / f"{base}.pdb"
    return data_root / f"{structure_id}.pdb"


def _find_foldx_stability_fxout(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.glob("*_ST.fxout"))
    if cands:
        return cands[0]
    cands = sorted(run_dir.glob("Average*.fxout"))
    if cands:
        return cands[0]
    cands = sorted(run_dir.glob("*.fxout"))
    for p in cands:
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
            if "\t" in txt and ".pdb" in txt.split("\t", 1)[0].lower():
                return p
        except OSError:
            continue
    return None


def run_foldx_once(
    foldx: Path,
    rotabase: Path | None,
    pdb_copy: Path,
    run_dir: Path,
    command: str,
    timeout_s: int,
    extra_foldx_args: list[str],
) -> tuple[bool, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if rotabase is not None and rotabase.is_file():
        shutil.copy2(rotabase, run_dir / "rotabase.txt")
    if _DEFAULT_MOLECULES.is_dir() and not (run_dir / "molecules").exists():
        shutil.copytree(_DEFAULT_MOLECULES, run_dir / "molecules")

    local_pdb = run_dir / pdb_copy.name
    shutil.copy2(pdb_copy, local_pdb)

    cmd = [
        str(foldx),
        f"--command={command}",
        f"--pdb={local_pdb.name}",
    ] + list(extra_foldx_args)

    try:
        r = subprocess.run(
            cmd,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, repr(e)

    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-2000:]
        return False, f"exit_{r.returncode}:{tail}"

    out = _find_foldx_stability_fxout(run_dir)
    if out is None:
        return False, "no_fxout_output"
    try:
        raw = parse_foldx_fxout(out)
        foldx_terms_to_pia_targets(raw)
    except Exception as e:
        return False, repr(e)
    return True, ""


def main() -> int:
    p = argparse.ArgumentParser(description="Run FoldX (bundled or custom) per PDB; write PIA CSV.")
    p.add_argument(
        "--foldx",
        type=Path,
        default=_DEFAULT_FOLDX,
        help=f"FoldX executable (default: {_DEFAULT_FOLDX}).",
    )
    p.add_argument(
        "--rotabase",
        type=Path,
        default=None,
        help="Optional rotabase.txt (FoldX 5 often works without it).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/datasets/PRA310/splits/PRA310.csv"),
    )
    p.add_argument("--data-root", type=Path, default=Path("/media/SSD0/csd/lrg/datasets/PRA310/PDBs"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/datasets/PRA310/splits/PRA310_foldx_pia_terms.csv"),
    )
    p.add_argument("--pdb-col", default="PDB")
    p.add_argument("--mut", action="store_true")
    p.add_argument("--command", default="Stability", help="FoldX -command (default: Stability).")
    p.add_argument(
        "--no-complex-with-rna",
        action="store_true",
        help="Do not pass --complexWithRNA=1 (not recommended for PRA).",
    )
    p.add_argument(
        "--extra-foldx-arg",
        action="append",
        default=[],
        help="Additional FoldX CLI tokens (repeatable), e.g. --extra-foldx-arg=--water=-IGNORE",
    )
    p.add_argument("--work-root", type=Path, default=_REPO_ROOT / "outputs" / "foldx_runs_pra310")
    p.add_argument("--timeout", type=int, default=900, help="Seconds per FoldX subprocess.")
    p.add_argument("--keep-runs", action="store_true", help="Keep per-PDB work directories.")
    args = p.parse_args()

    if not args.foldx.is_file():
        print(f"ERROR: FoldX binary not found: {args.foldx}", file=sys.stderr)
        return 1
    if not args.csv.is_file():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 1
    if not args.data_root.is_dir():
        print(f"ERROR: data root not a directory: {args.data_root}", file=sys.stderr)
        return 1

    extra = list(args.extra_foldx_arg)
    if not args.no_complex_with_rna:
        if "--complexWithRNA=1" not in extra and all("complexWithRNA" not in x for x in extra):
            extra.append("--complexWithRNA=1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None or args.pdb_col not in reader.fieldnames:
            print(f"ERROR: missing column {args.pdb_col!r}. Found {reader.fieldnames}", file=sys.stderr)
            return 1
        if args.mut and "MUTATION" not in reader.fieldnames:
            print("ERROR: --mut requires MUTATION column.", file=sys.stderr)
            return 1
        rows = list(reader)

    fieldnames = ["sample_id", "pdb_path"] + PIA_COLS + ["ok", "error"]

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # type: ignore

    with open(args.output, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(rows, desc="FoldX"):
            pdb_id = (row.get(args.pdb_col) or "").strip()
            if not pdb_id:
                row_out = {c: "" for c in PIA_COLS}
                row_out.update({"sample_id": "", "pdb_path": "", "ok": "0", "error": "empty_pdb_id"})
                writer.writerow(row_out)
                continue

            structure_id = pdb_id
            if args.mut:
                mut = (row.get("MUTATION") or "").strip()
                structure_id = f"{pdb_id}_{mut}" if mut else pdb_id

            pdb_path = resolve_pdb_path(args.data_root, structure_id, args.mut)
            run_dir = (args.work_root / structure_id.replace("/", "_")).resolve()

            if not pdb_path.is_file():
                row_out = {c: 0.0 for c in PIA_COLS}
                row_out.update(
                    {
                        "sample_id": structure_id,
                        "pdb_path": str(pdb_path),
                        "ok": "0",
                        "error": "missing_pdb_file",
                    }
                )
                writer.writerow(row_out)
                continue

            if run_dir.exists() and not args.keep_runs:
                shutil.rmtree(run_dir, ignore_errors=True)

            ok, err = run_foldx_once(
                args.foldx,
                args.rotabase,
                pdb_path,
                run_dir,
                args.command,
                args.timeout,
                extra,
            )

            if not ok:
                row_out = {c: 0.0 for c in PIA_COLS}
                row_out.update(
                    {
                        "sample_id": structure_id,
                        "pdb_path": str(pdb_path),
                        "ok": "0",
                        "error": err[:5000] if err else "foldx_failed",
                    }
                )
                writer.writerow(row_out)
            else:
                fx = _find_foldx_stability_fxout(run_dir)
                assert fx is not None
                raw = parse_foldx_fxout(fx)
                pia = foldx_terms_to_pia_targets(raw)
                row_out = {c: pia[c] for c in PIA_COLS}
                row_out.update(
                    {
                        "sample_id": structure_id,
                        "pdb_path": str(pdb_path),
                        "ok": "1",
                        "error": "",
                    }
                )
                writer.writerow(row_out)

            if not args.keep_runs and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)

    print(f"Wrote {args.output} ({len(rows)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
