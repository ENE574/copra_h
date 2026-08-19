#!/usr/bin/env python3
"""Run MPD PEMPNI val hyperparameter sweep and pick the best trial."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)


def deep_update(base: dict, patch: dict) -> dict:
    out = deepcopy(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def rel_to_root(path: Path) -> str:
    path = path.resolve()
    root = ROOT.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def read_val_metrics(
    run_dir: Path,
    *,
    train_all_cap: float = 0.80,
    last_n_epochs: int = 5,
) -> dict | None:
    csv_paths = sorted((run_dir / "log_fold_0").glob("lightning_logs/version_*/metrics.csv"))
    if not csv_paths:
        val_path = run_dir / "val_metrics.json"
        return json.loads(val_path.read_text()) if val_path.is_file() else None

    df = pd.read_csv(csv_paths[-1])
    if "val/all_pearson" not in df.columns:
        return None
    ep = df.dropna(subset=["val/all_pearson"]).groupby("epoch", as_index=False).last()
    if ep.empty:
        return None

    peak_row = ep.loc[ep["val/all_pearson"].idxmax()]
    last_n = max(1, int(last_n_epochs))
    last_block = ep.tail(last_n)
    last5_mean = float(last_block["val/all_pearson"].mean())

    cap_key = f"val/all_pearson_train_cap_{train_all_cap:.2f}"
    cap_epoch = -1
    cap_val = float("nan")
    if "train/all_pearson" in ep.columns:
        capped = ep[ep["train/all_pearson"] <= train_all_cap]
        if not capped.empty:
            cap_row = capped.loc[capped["val/all_pearson"].idxmax()]
            cap_val = float(cap_row["val/all_pearson"])
            cap_epoch = int(cap_row["epoch"])

    out = {
        "best_val_epoch": int(peak_row["epoch"]),
        "val/all_pearson": float(peak_row["val/all_pearson"]),
        "val/all_pearson_peak": float(peak_row["val/all_pearson"]),
        "val/all_pearson_last5_mean": last5_mean,
        cap_key: cap_val,
        "best_val_epoch_train_cap": cap_epoch,
        "last_n_epochs": last_n,
        "train_all_cap": train_all_cap,
        "stopped_epoch": int(ep["epoch"].max()),
    }
    for col in (
        "val/all_spearman",
        "val/all_rmse",
        "val/all_mae",
        "val/pc_pearson",
        "val/pc_spearman",
        "val/pc_rmse",
        "val/pc_mae",
        "train/all_pearson",
    ):
        if col in peak_row and pd.notna(peak_row[col]):
            out[col] = float(peak_row[col])
    if "train/all_pearson" in last_block.columns:
        out["train/all_pearson_last5_mean"] = float(last_block["train/all_pearson"].mean())
    return out


def metric_value(metrics: dict | None, key: str, default=float("-inf")) -> float:
    if not metrics or key not in metrics:
        return default
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return default


def rank_trials(rows: list[dict], primary: str, tiebreaker: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            metric_value(r.get("val_metrics"), primary),
            metric_value(r.get("val_metrics"), tiebreaker),
        ),
        reverse=True,
    )


def find_latest_run(output_dir: Path, run_prefix: str) -> Path | None:
    if not output_dir.is_dir():
        return None
    cands = sorted(output_dir.glob(f"{run_prefix}*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def run_trial(
    trial_id: str,
    model_cfg: Path,
    data_cfg: Path,
    run_cfg: Path,
    dry_run: bool,
) -> int:
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
    print(f"Trial {trial_id}")
    print(" ".join(cmd))
    print("=" * 72)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="MPD PEMPNI val hyperparameter sweep")
    parser.add_argument(
        "--grid",
        default=str(ROOT / "config/sweeps/mpd_pempni_hparam_grid.yml"),
        help="Sweep grid YAML",
    )
    parser.add_argument("--trial", action="append", help="Run only these trial id(s)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--select-only", action="store_true", help="Rank existing runs, do not train")
    args = parser.parse_args()

    grid = load_yaml(Path(args.grid))
    primary = grid.get("selection_metric", "val/all_pearson")
    tiebreaker = grid.get("tiebreaker_metric", "val/pc_pearson")
    train_all_cap = float(grid.get("train_all_cap", 0.80))
    last_n_epochs = int(grid.get("last_n_epochs", 5))

    base_model = load_yaml(ROOT / grid["base_model_config"])
    base_run = load_yaml(ROOT / grid["base_run_config"])
    data_cfg = ROOT / grid["base_data_config"]

    sweep_root = Path(base_run["output_dir"])
    sweep_root.mkdir(parents=True, exist_ok=True)
    trial_root = sweep_root / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)

    run_prefix = base_run.get("run_name", "mpd_pempni_val_hp_")
    results: list[dict] = []

    for spec in grid["trials"]:
        trial_id = spec["id"]
        if args.trial and trial_id not in args.trial:
            continue

        trial_dir = trial_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        model_out = trial_dir / "model.yml"
        run_out = trial_dir / "run.yml"

        trial_train = spec.get("train", {})
        trial_run = spec.get("run", {})
        model_cfg = deep_update(base_model, {"train": trial_train}) if trial_train else deepcopy(base_model)
        run_cfg = deep_update(base_run, trial_run) if trial_run else deepcopy(base_run)
        run_cfg["run_name"] = f"{run_prefix}{trial_id}_"

        dump_yaml(model_out, model_cfg)
        dump_yaml(run_out, run_cfg)

        record = {
            "trial_id": trial_id,
            "model_config": rel_to_root(model_out),
            "run_config": rel_to_root(run_out),
            "hparams": spec,
        }

        if not args.select_only:
            rc = run_trial(trial_id, model_out, data_cfg, run_out, args.dry_run)
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
            val_metrics = read_val_metrics(
                Path(record["run_dir"]),
                train_all_cap=train_all_cap,
                last_n_epochs=last_n_epochs,
            )
            record["val_metrics"] = val_metrics
            record["status"] = "ok" if val_metrics else "missing_val_metrics"
        else:
            record["status"] = "missing_run_dir"

        results.append(record)
        leaderboard_path = sweep_root / "leaderboard.json"
        leaderboard_path.write_text(
            json.dumps(rank_trials(results, primary, tiebreaker), indent=2)
        )

    ranked = rank_trials(results, primary, tiebreaker)
    if not ranked:
        print("No trials to rank.")
        return

    best = ranked[0]
    best_payload = {
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection_metric": primary,
        "tiebreaker_metric": tiebreaker,
        "train_all_cap": train_all_cap,
        "last_n_epochs": last_n_epochs,
        "best_trial_id": best["trial_id"],
        "best_val_metrics": best.get("val_metrics"),
        "best_run_dir": best.get("run_dir"),
        "leaderboard": ranked,
    }

    best_model = trial_root / best["trial_id"] / "model.yml"
    best_hparams_out = sweep_root / "best_hparams.json"
    best_hparams_out.write_text(json.dumps(best_payload, indent=2))

    final_model_out = sweep_root / "best_model_for_final_train.yml"
    if best_model.is_file():
        final_model_out.write_text(best_model.read_text())

    final_run_template = ROOT / "config/runs/finetune_mpd_pempni_final.yml"
    print("\n" + "=" * 72)
    print("VAL HYPERPARAMETER SELECTION COMPLETE")
    print("=" * 72)
    print(f"Best trial: {best['trial_id']}")
    print(f"  {primary}: {metric_value(best.get('val_metrics'), primary):.4f}")
    print(f"  {tiebreaker}: {metric_value(best.get('val_metrics'), tiebreaker):.4f}")
    vm = best.get("val_metrics") or {}
    if "val/all_pearson_peak" in vm:
        print(f"  val/all_pearson_peak: {vm['val/all_pearson_peak']:.4f}")
    if "train/all_pearson" in vm:
        print(f"  train/all_pearson @ peak epoch: {vm['train/all_pearson']:.4f}")
    if best.get("run_dir"):
        print(f"  run_dir: {best['run_dir']}")
    print(f"Leaderboard: {sweep_root / 'leaderboard.json'}")
    print(f"Best model config: {final_model_out}")
    print("\nNext step (full MPD276 train + MPD48 test once):")
    print(
        "  python run.py finetune ddG \\\n"
        f"    --model_config {rel_to_root(final_model_out)} \\\n"
        "    --data_config config/datasets/MPD_pempni.yml \\\n"
        f"    --run_config {rel_to_root(final_run_template)}"
    )


if __name__ == "__main__":
    main()
