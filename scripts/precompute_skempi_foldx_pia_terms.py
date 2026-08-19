#!/usr/bin/env python3
"""FoldX BuildModel → Stability PIA labels for SKEMPI (incl. multi-point mutations).

Fills missing rows in ``SKEMPI_foldx_pia_terms.csv`` so all SKEMPI v2 entries
have mutation-specific FoldX Stability decomposition (PIA proxy).

Example::

  # Smoke test (3 missing mutations)
  python scripts/precompute_skempi_foldx_pia_terms.py --missing-only --limit 3

  # Fill all missing (~1880 multi-mut + resume-safe)
  python scripts/precompute_skempi_foldx_pia_terms.py --missing-only --resume
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
_RDE_ROOT = _REPO_ROOT.parent / "RDE-PPI"

_DEFAULT_FOLDX = _REPO_ROOT / "foldx" / "foldx_20270131"
_DEFAULT_MOLECULES = _REPO_ROOT / "foldx" / "molecules"
_DEFAULT_SKEMPI_CSV = _RDE_ROOT / "data" / "SKEMPI_v2" / "skempi_v2.csv"
_DEFAULT_PDB_DIR = _RDE_ROOT / "data" / "SKEMPI_v2" / "PDBs"
_DEFAULT_OUTPUT = Path(
    "/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/splits/SKEMPI_foldx_pia_terms.csv"
)
_DEFAULT_WORK = Path("/media/SSD0/csd/lrg/copra_h/outputs/foldx_skempi_buildmodel")

FIELDNAMES = [
    "sample_id",
    "pdb_path",
    "mutation",
    "mutant_pdb",
    "Electro",
    "Energy_SolvP",
    "Energy_SolvH",
    "Energy_VdW",
    "Energy_Hbond",
    "ok",
    "error",
]


def _load_foldx_physics():
    spec = importlib.util.spec_from_file_location(
        "_foldx_physics_standalone", _REPO_ROOT / "data" / "foldx_physics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_fx = _load_foldx_physics()
parse_foldx_fxout = _fx.parse_foldx_fxout
foldx_terms_to_pia_targets = _fx.foldx_terms_to_pia_targets
PIA_COLS = list(_fx.PIA_PHYSICS_NAMES)


def _load_skempi_entries(csv_path: Path, pdb_dir: Path) -> list[dict]:
    sys.path.insert(0, str(_RDE_ROOT))
    from rde.datasets.skempi import load_skempi_entries

    return load_skempi_entries(str(csv_path), str(pdb_dir))


def skempi_mutstr_to_foldx(mutstr: str) -> str:
    """``LI38G,RI48A`` -> ``LI38G;RI48A;`` (SKEMPI cleaned mutation format)."""
    parts = [m.strip() for m in mutstr.split(",") if m.strip()]
    return "".join(f"{m};" for m in parts)


def _find_stability_fxout(run_dir: Path) -> Path | None:
    cands = sorted(run_dir.glob("*_ST.fxout"))
    if cands:
        return cands[0]
    cands = sorted(run_dir.glob("*ST*.fxout"))
    return cands[0] if cands else None


def _find_mutant_pdb(run_dir: Path, pdbcode: str) -> Path | None:
    for pat in (f"{pdbcode}_1.pdb", f"{pdbcode.upper()}_1.pdb", f"{pdbcode.lower()}_1.pdb"):
        p = run_dir / pat
        if p.is_file():
            return p
    cands = [p for p in run_dir.glob("*.pdb") if not p.name.startswith("WT_")]
    cands = [p for p in cands if "_Repair" not in p.name and p.name != f"{pdbcode}.pdb"]
    return cands[0] if cands else None


def _setup_run_dir(run_dir: Path, rotabase: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if rotabase is not None and rotabase.is_file():
        shutil.copy2(rotabase, run_dir / "rotabase.txt")
    mol_dst = run_dir / "molecules"
    if _DEFAULT_MOLECULES.is_dir() and not mol_dst.exists():
        shutil.copytree(_DEFAULT_MOLECULES, mol_dst)


def _run_foldx(cmd: list[str], run_dir: Path, timeout_s: int) -> tuple[bool, str]:
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
    return True, ""


def process_one_sample(task: dict) -> dict:
    foldx = Path(task["foldx"])
    rotabase = Path(task["rotabase"]) if task.get("rotabase") else None
    run_dir = Path(task["run_dir"])
    wt_pdb = Path(task["pdb_path"])
    pdbcode = task["pdbcode"]
    mutstr = task["mutstr"]
    sample_id = task["sample_id"]
    timeout_bm = int(task["timeout_bm"])
    timeout_st = int(task["timeout_st"])

    row = {c: "" for c in FIELDNAMES}
    row["sample_id"] = sample_id
    row["pdb_path"] = str(wt_pdb)
    row["mutation"] = mutstr
    row["mutant_pdb"] = str(run_dir / f"{pdbcode}_1.pdb")
    for c in PIA_COLS:
        row[c] = 0.0
    row["ok"] = "0"
    row["error"] = ""

    if not wt_pdb.is_file():
        row["error"] = "missing_wt_pdb"
        return row

    # Resume from cached fxout
    fx_cached = _find_stability_fxout(run_dir)
    mut_cached = _find_mutant_pdb(run_dir, pdbcode)
    if fx_cached is not None and mut_cached is not None:
        try:
            pia = foldx_terms_to_pia_targets(parse_foldx_fxout(fx_cached))
            row["mutant_pdb"] = str(mut_cached.resolve())
            for c in PIA_COLS:
                row[c] = pia[c]
            row["ok"] = "1"
            return row
        except Exception as e:
            row["error"] = f"parse_cached:{e}"

    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    _setup_run_dir(run_dir, rotabase)

    local_wt = run_dir / wt_pdb.name
    shutil.copy2(wt_pdb, local_wt)
    (run_dir / "individual_list.txt").write_text(skempi_mutstr_to_foldx(mutstr) + "\n")

    ok, err = _run_foldx(
        [
            str(foldx),
            "--command=BuildModel",
            f"--pdb={local_wt.name}",
            "--mutant-file=individual_list.txt",
            "--numberOfRuns=1",
        ],
        run_dir,
        timeout_bm,
    )
    if not ok:
        row["error"] = f"buildmodel:{err[:500]}"
        return row

    mut_pdb = _find_mutant_pdb(run_dir, pdbcode)
    if mut_pdb is None:
        row["error"] = "no_mutant_pdb"
        return row
    row["mutant_pdb"] = str(mut_pdb.resolve())

    ok, err = _run_foldx(
        [str(foldx), "--command=Stability", f"--pdb={mut_pdb.name}"],
        run_dir,
        timeout_st,
    )
    if not ok:
        row["error"] = f"stability:{err[:500]}"
        return row

    fx = _find_stability_fxout(run_dir)
    if fx is None:
        row["error"] = "no_st_fxout"
        return row
    try:
        pia = foldx_terms_to_pia_targets(parse_foldx_fxout(fx))
    except Exception as e:
        row["error"] = f"parse:{e}"
        return row

    for c in PIA_COLS:
        row[c] = pia[c]
    row["ok"] = "1"
    row["error"] = ""
    return row


def load_existing_csv(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {r["sample_id"]: r for r in reader if r.get("sample_id")}


def write_csv(path: Path, rows: dict[str, dict], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique_order = list(dict.fromkeys(order))
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for sid in unique_order:
            if sid in rows:
                writer.writerow({k: rows[sid].get(k, "") for k in FIELDNAMES})
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description="SKEMPI FoldX PIA via BuildModel+Stability")
    p.add_argument("--foldx", type=Path, default=_DEFAULT_FOLDX)
    p.add_argument("--rotabase", type=Path, default=None)
    p.add_argument("--skempi-csv", type=Path, default=_DEFAULT_SKEMPI_CSV)
    p.add_argument("--pdb-dir", type=Path, default=_DEFAULT_PDB_DIR)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--work-root", type=Path, default=_DEFAULT_WORK)
    p.add_argument("--missing-only", action="store_true", help="Only process rows absent from output CSV.")
    p.add_argument("--resume", action="store_true", help="Keep work dirs; reuse cached mutant/fxout.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--timeout-bm", type=int, default=600)
    p.add_argument("--timeout-st", type=int, default=120)
    args = p.parse_args()

    if not args.foldx.is_file():
        print(f"ERROR: FoldX not found: {args.foldx}", file=sys.stderr)
        return 1
    if not args.skempi_csv.is_file():
        print(f"ERROR: SKEMPI csv not found: {args.skempi_csv}", file=sys.stderr)
        return 1

    entries = _load_skempi_entries(args.skempi_csv, args.pdb_dir)
    seen_e: set[str] = set()
    unique_entries = []
    for e in entries:
        sid = f"{e['pdbcode'].upper()}_{e['mutstr']}"
        if sid in seen_e:
            continue
        seen_e.add(sid)
        unique_entries.append(e)
    order = [f"{e['pdbcode'].upper()}_{e['mutstr']}" for e in unique_entries]
    existing = load_existing_csv(args.output)

    tasks = []
    for e in unique_entries:
        sample_id = f"{e['pdbcode'].upper()}_{e['mutstr']}"
        if args.missing_only and sample_id in existing and existing[sample_id].get("ok") == "1":
            continue
        if sample_id in existing and not args.missing_only and existing[sample_id].get("ok") == "1":
            continue
        safe_dir = sample_id.replace("/", "_").replace(",", "_")
        tasks.append(
            {
                "sample_id": sample_id,
                "pdbcode": e["pdbcode"].upper(),
                "mutstr": e["mutstr"],
                "pdb_path": str(Path(e["pdb_path"]).resolve()),
                "run_dir": str((args.work_root / safe_dir).resolve()),
                "foldx": str(args.foldx.resolve()),
                "rotabase": str(args.rotabase.resolve()) if args.rotabase else "",
                "timeout_bm": args.timeout_bm,
                "timeout_st": args.timeout_st,
            }
        )

    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(
        f"SKEMPI unique mutations: {len(unique_entries)}, "
        f"existing ok: {sum(1 for r in existing.values() if r.get('ok')=='1')}, "
        f"to run: {len(tasks)}"
    )

    merged = dict(existing)
    if not args.resume and tasks:
        args.work_root.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # type: ignore

    if args.workers <= 1:
        for i, task in enumerate(tqdm(tasks, desc="FoldX SKEMPI"), 1):
            row = process_one_sample(task)
            merged[row["sample_id"]] = row
            if i % 10 == 0 or i == len(tasks):
                write_csv(args.output, merged, order)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_one_sample, t): t["sample_id"] for t in tasks}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="FoldX SKEMPI"):
                row = fut.result()
                merged[row["sample_id"]] = row
                write_csv(args.output, merged, order)

    unique_order = list(dict.fromkeys(order))
    write_csv(args.output, merged, order)
    ok_n = sum(1 for sid in unique_order if merged.get(sid, {}).get("ok") == "1")
    print(f"Wrote {args.output}: {ok_n}/{len(unique_order)} unique ok ({len(unique_order)-ok_n} missing/failed)")
    return 0 if ok_n == len(unique_order) else 1


if __name__ == "__main__":
    raise SystemExit(main())
