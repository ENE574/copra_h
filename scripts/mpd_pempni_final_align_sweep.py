#!/usr/bin/env python3
"""Run MPD PEMPNI final-protocol alignment ablations (train + MPD48 test)."""

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


def deep_update(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def find_latest_run(output_dir: Path, run_prefix: str) -> Path | None:
    if not output_dir.is_dir():
        return None
    cands = sorted(output_dir.glob(f"{run_prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


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
        "test/all_spearman": float(r.get("spearman", float("nan"))),
        "test/all_rmse": float(r.get("rmse", float("nan"))),
        "test/all_mae": float(r.get("mae", float("nan"))),
        "test/pc_pearson": float(r.get("pc_pearson", float("nan"))),
        "checkpoint": r.get("checkpoint"),
        "checkpoint_monitor": r.get("checkpoint_monitor"),
    }


def metric_value(metrics: dict | None, key: str, default=float("-inf")) -> float:
    if not metrics or key not in metrics:
        return default
    try:
        v = float(metrics[key])
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def rank_trials(rows: list[dict], primary: str, tiebreaker: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            metric_value(r.get("test_metrics"), primary),
            -metric_value(r.get("test_metrics"), tiebreaker, default=float("inf")),
        ),
        reverse=True,
    )


def run_trial(model_cfg: Path, data_cfg: Path, run_cfg: Path) -> int:
    cmd = [
        PYTHON_BIN,
        str(ROOT / "run.py"),
        "finetune",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="MPD PEMPNI final protocol alignment sweep")
    parser.add_argument(
        "--grid",
        default=str(ROOT / "config/sweeps/mpd_pempni_final_align_grid.yml"),
    )
    parser.add_argument("--trial", action="append", help="Run only these trial id(s)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    grid = load_yaml(Path(args.grid))
    primary = grid.get("selection_metric", "test/all_pearson")
    tiebreaker = grid.get("tiebreaker_metric", "test/all_rmse")
    base_run = load_yaml(ROOT / grid["base_run_config"])

    sweep_root = Path(base_run["output_dir"])
    sweep_root.mkdir(parents=True, exist_ok=True)
    trial_root = sweep_root / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)

    run_prefix = base_run.get("run_name", "mpd_pempni_align_")
    results: list[dict] = []

    for spec in grid.get("trials", []):
        trial_id = spec["id"]
        if args.trial and trial_id not in args.trial:
            continue

        trial_dir = trial_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        model_src = ROOT / spec["model_config"]
        data_src = ROOT / spec["data_config"]
        run_out = trial_dir / "run.yml"
        model_out = trial_dir / "model.yml"

        run_cfg = deep_update(base_run, spec.get("run", {}))
        run_cfg["run_name"] = f"{run_prefix}{trial_id}_"
        dump_yaml(model_out, load_yaml(model_src))
        dump_yaml(run_out, run_cfg)

        record = {
            "trial_id": trial_id,
            "description": spec.get("description", ""),
            "model_config": str(model_out),
            "data_config": str(data_src),
            "run_config": str(run_out),
            "protocol": spec.get("data_config", ""),
            "hparams": spec,
        }

        if not args.select_only:
            rc = run_trial(model_out, data_src, run_out)
            record["return_code"] = rc
            if rc != 0:
                record["status"] = "failed"
                results.append(record)
                continue
            time.sleep(1.0)
            run_dir = find_latest_run(Path(run_cfg["output_dir"]), run_cfg["run_name"])
            record["run_dir"] = str(run_dir) if run_dir else None
        else:
            run_dir = find_latest_run(Path(run_cfg["output_dir"]), run_cfg["run_name"])
            record["run_dir"] = str(run_dir) if run_dir else None

        if record.get("run_dir"):
            record["test_metrics"] = read_test_metrics(Path(record["run_dir"]))
            record["status"] = "ok" if record.get("test_metrics") else "missing_test_metrics"
        else:
            record["status"] = "missing_run_dir"

        results.append(record)
        (sweep_root / "leaderboard.json").write_text(
            json.dumps(rank_trials(results, primary, tiebreaker), indent=2)
        )

    ranked = rank_trials(results, primary, tiebreaker)
    if not ranked:
        print("No trials to rank.")
        return

    best = ranked[0]
    payload = {
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection_metric": primary,
        "tiebreaker_metric": tiebreaker,
        "best_trial_id": best["trial_id"],
        "best_test_metrics": best.get("test_metrics"),
        "leaderboard": ranked,
    }
    (sweep_root / "best_result.json").write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 72)
    print("FINAL ALIGNMENT SWEEP COMPLETE")
    print("=" * 72)
    print(f"Best: {best['trial_id']}  {primary}={metric_value(best.get('test_metrics'), primary):.4f}")
    for row in ranked:
        m = row.get("test_metrics") or {}
        print(
            f"  {row['trial_id']:22s} "
            f"test/all={metric_value(m, 'test/all_pearson'):7.3f} "
            f"rmse={metric_value(m, 'test/all_rmse', default=float('inf')):6.2f} "
            f"ckpt={m.get('checkpoint', 'NA')}"
        )


if __name__ == "__main__":
    main()
