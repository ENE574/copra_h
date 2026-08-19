#!/usr/bin/env python3
"""Retest existing MPD checkpoints on MPD48 without retraining (Phase 2)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def resolve_ckpt(spec: dict) -> Path | None:
    if spec.get("ckpt_path"):
        p = Path(spec["ckpt_path"])
        return p if p.is_file() else None
    run_dir = Path(spec["run_dir"])
    ckpt_dir = run_dir / "log_fold_0" / "checkpoint"
    if not ckpt_dir.is_dir():
        return None
    matches = sorted(ckpt_dir.glob(spec.get("ckpt_glob", "*.ckpt")))
    return matches[0] if matches else None


def run_retest(model_cfg: Path, data_cfg: Path, run_cfg: Path) -> int:
    cmd = [
        PYTHON_BIN,
        str(ROOT / "run.py"),
        "test",
        "ddG",
        "--model_config",
        str(model_cfg),
        "--data_config",
        str(data_cfg),
        "--run_config",
        str(run_cfg),
    ]
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def read_test_metrics(run_dir: Path) -> dict | None:
    res_path = run_dir / "res.json"
    if not res_path.is_file():
        return None
    rows = json.loads(res_path.read_text())
    if not rows:
        return None
    r = rows[0]
    return {
        "test/all_pearson": float(r.get("pearson", float("nan"))),
        "test/all_rmse": float(r.get("rmse", float("nan"))),
        "test/pc_pearson": float(r.get("pc_pearson", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retest MPD PEMPNI checkpoints on MPD48")
    parser.add_argument(
        "--grid",
        default=str(ROOT / "config/sweeps/mpd_pempni_final_align_grid.yml"),
    )
    parser.add_argument("--id", action="append", dest="ids", help="Retest only these id(s)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grid = load_yaml(Path(args.grid))
    base_run = load_yaml(ROOT / "config/runs/test_mpd_pempni_retest_template.yml")
    out_root = Path(base_run["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in grid.get("retest_checkpoints", []):
        rid = spec["id"]
        if args.ids and rid not in args.ids:
            continue

        ckpt = resolve_ckpt(spec)
        if ckpt is None:
            print(f"SKIP {rid}: checkpoint not found")
            rows.append({"id": rid, "status": "missing_ckpt", "description": spec.get("description")})
            continue

        run_cfg_path = out_root / "configs" / f"{rid}.yml"
        run_cfg = deepcopy(base_run)
        run_cfg["ckpt"] = str(ckpt)
        run_cfg["run_name"] = f"mpd_retest_{rid}_"
        dump_yaml(run_cfg_path, run_cfg)

        model_cfg = ROOT / spec["model_config"]
        data_cfg = ROOT / spec["data_config"]

        print(f"\n>>> {rid}: {spec.get('description', '')}")
        print(f"    ckpt: {ckpt}")

        if args.dry_run:
            rows.append({"id": rid, "status": "dry_run", "checkpoint": str(ckpt)})
            continue

        rc = run_retest(model_cfg, data_cfg, run_cfg_path)
        run_dirs = sorted(out_root.glob(f"mpd_retest_{rid}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        run_dir = run_dirs[0] if run_dirs else None
        entry = {
            "id": rid,
            "description": spec.get("description", ""),
            "checkpoint": str(ckpt),
            "return_code": rc,
            "run_dir": str(run_dir) if run_dir else None,
            "status": "ok" if rc == 0 else "failed",
        }
        if run_dir:
            entry["test_metrics"] = read_test_metrics(run_dir)
            if entry["test_metrics"] is None:
                # test ddG does not write res.json; parse from lightning csv if needed
                entry["status"] = "ok_no_res_json"
        rows.append(entry)

    def sort_key(r: dict) -> float:
        m = r.get("test_metrics") or {}
        v = m.get("test/all_pearson")
        return float(v) if v is not None else -999.0

    rows.sort(key=lambda r: -sort_key(r))
    summary_path = out_root / "retest_leaderboard.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 72)
    print("CHECKPOINT RETEST SUMMARY")
    print("=" * 72)
    for row in rows:
        m = row.get("test_metrics") or {}
        pearson = m.get("test/all_pearson")
        ps = f"{pearson:.3f}" if pearson is not None else "NA"
        print(f"  {row['id']:24s} test/all={ps}  {row.get('status')}")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
