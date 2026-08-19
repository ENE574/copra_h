#!/usr/bin/env python3
"""Run RDE-Net and RDE-Linear SKEMPI baselines (official RDE-PPI protocol).

Usage (from repo root):
  PYTHON_BIN=/home/csd/anaconda3/envs/copra_h/bin/python \\
    python copra_h/scripts/run_rde_skempi_baselines.py --method net
  PYTHON_BIN=... python copra_h/scripts/run_rde_skempi_baselines.py --method linear
  PYTHON_BIN=... python copra_h/scripts/run_rde_skempi_baselines.py --method both

Outputs under RDE-PPI/outputs/skempi_baseline/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
RDE_ROOT = ROOT / "RDE-PPI"
OUT_DIR = RDE_ROOT / "outputs" / "skempi_baseline"
CKPT_NET = RDE_ROOT / "trained_models" / "DDG_RDE_Network_30k.pt"
CKPT_RDE = RDE_ROOT / "trained_models" / "RDE.pt"
SKEMPI_CSV = RDE_ROOT / "data" / "SKEMPI_v2" / "skempi_v2.csv"
SKEMPI_PDB = RDE_ROOT / "data" / "SKEMPI_v2" / "PDBs"


def _patch_torch_load() -> None:
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load  # type: ignore[method-assign]


def _mean_pc_corr(df: pd.DataFrame, min_n: int, strict_gt: bool) -> float:
    pcs = []
    for _, sub in df.groupby("complex"):
        n = len(sub)
        if strict_gt:
            if n <= min_n:
                continue
        elif n < min_n:
            continue
        r = sub["ddG"].corr(sub["ddG_pred"])
        if pd.notna(r):
            pcs.append(r)
    return float(np.mean(pcs)) if pcs else float("nan")


def _copra_style_pc(df: pd.DataFrame, min_n: int = 3) -> float:
    return _mean_pc_corr(df, min_n, strict_gt=True)


def _rde_style_pc(df: pd.DataFrame, min_n: int = 10) -> float:
    return _mean_pc_corr(df, min_n, strict_gt=False)


def _summarize(df: pd.DataFrame, label: str) -> dict:
    out = {
        "method": label,
        "n_samples": int(len(df)),
        "overall_pearson": float(df["ddG"].corr(df["ddG_pred"])),
        "overall_spearman": float(df["ddG"].corr(df["ddG_pred"], method="spearman")),
        "pc_pearson_rde_n10": _rde_style_pc(df, min_n=10),
        "pc_pearson_copra_n3": _copra_style_pc(df, min_n=3),
        "n_complexes_rde_n10": int(
            sum(len(g) >= 10 for _, g in df.groupby("complex"))
        ),
        "n_complexes_copra_n3": int(
            sum(len(g) > 3 for _, g in df.groupby("complex"))
        ),
    }
    return out


def run_rde_net(device: str, num_workers: int) -> pd.DataFrame:
    from tqdm.auto import tqdm

    from rde.models.rde_ddg import DDG_RDE_Network
    from rde.utils.misc import get_logger
    from rde.utils.skempi import SkempiDatasetManager, eval_skempi_three_modes
    from rde.utils.train import CrossValidation, ScalarMetricAccumulator, recursive_to, sum_weighted_losses

    logger = get_logger("test", None)
    ckpt = torch.load(CKPT_NET)
    config = ckpt["config"]
    num_cvfolds = len(ckpt["model"]["models"])

    dataset_mgr = SkempiDatasetManager(
        config, num_cvfolds=num_cvfolds, num_workers=num_workers, logger=logger
    )
    cv_mgr = CrossValidation(
        model_factory=DDG_RDE_Network, config=config, num_cvfolds=num_cvfolds
    ).to(device)
    cv_mgr.load_state_dict(ckpt["model"])

    results = []
    with torch.no_grad():
        for fold in range(num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            for batch in tqdm(
                dataset_mgr.get_val_loader(fold),
                desc=f"RDE-Net fold {fold + 1}/{num_cvfolds}",
            ):
                batch = recursive_to(batch, device)
                loss_dict, output_dict = model(batch)
                for complex_name, mutstr, ddg_true, ddg_pred in zip(
                    batch["complex"],
                    batch["mutstr"],
                    output_dict["ddG_true"],
                    output_dict["ddG_pred"],
                ):
                    results.append(
                        {
                            "complex": complex_name,
                            "mutstr": mutstr,
                            "num_muts": len(mutstr.split(",")),
                            "ddG": ddg_true.item(),
                            "ddG_pred": ddg_pred.item(),
                        }
                    )

    df = pd.DataFrame(results)
    df["method"] = "RDE-Net"
    return df


def run_rde_linear(
    device: str,
    num_workers: int,
    num_trials: int,
    num_folds: int,
    iters: int,
) -> pd.DataFrame:
    from rde.linear.calibrate import run_calibration
    from rde.linear.entropy import get_entropy
    from rde.utils.skempi import eval_skempi_three_modes

    linear_dir = OUT_DIR / "rde_linear"
    linear_dir.mkdir(parents=True, exist_ok=True)
    cache_path = linear_dir / "entropy.pt"
    skempi_cache = RDE_ROOT / "data" / "SKEMPI_v2_cache.pkl"

    if cache_path.is_file():
        entropy = torch.load(cache_path, weights_only=False)
        print(f"[INFO] Using entropy cache: {cache_path}")
    else:
        print("[INFO] Computing entropy (slow; caches to entropy.pt)...")
        entropy = get_entropy(
            ckpt_path=str(CKPT_RDE),
            device=device,
            skempi_dir=str(SKEMPI_PDB),
            skempi_cache_path=str(skempi_cache),
        )
        torch.save(entropy, cache_path)

    df = run_calibration(
        entropy,
        num_trials=num_trials,
        num_folds=num_folds,
        iters=iters,
        device=device,
        output_dir=str(linear_dir / "calibration"),
    )
    df["method"] = "RDE-Linear"
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("net", "linear", "both"), default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--linear_trials", type=int, default=10)
    parser.add_argument("--linear_folds", type=int, default=3)
    parser.add_argument("--linear_iters", type=int, default=2000)
    args = parser.parse_args()

    if not SKEMPI_CSV.is_file() or not SKEMPI_PDB.is_dir():
        print(f"Missing SKEMPI v2 under {RDE_ROOT / 'data/SKEMPI_v2'}", file=sys.stderr)
        return 1
    if not CKPT_NET.is_file() or not CKPT_RDE.is_file():
        print(f"Missing checkpoints under {RDE_ROOT / 'trained_models'}", file=sys.stderr)
        return 1

    os.chdir(RDE_ROOT)
    sys.path.insert(0, str(RDE_ROOT))
    _patch_torch_load()

    from rde.utils.skempi import eval_skempi_three_modes

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []

    if args.method in ("net", "both"):
        if not torch.cuda.is_available() and args.device.startswith("cuda"):
            print("[WARN] CUDA unavailable; using cpu for RDE-Net")
            args.device = "cpu"
        df_net = run_rde_net(args.device, args.num_workers)
        net_csv = OUT_DIR / "rde_net_results.csv"
        df_net.to_csv(net_csv, index=False)
        metrics = eval_skempi_three_modes(df_net)
        metrics.to_csv(OUT_DIR / "rde_net_metrics_official.csv", index=False)
        summaries.append(_summarize(df_net, "RDE-Net"))
        print("\n=== RDE-Net official metrics (mode=all) ===")
        print(metrics[metrics["mode"] == "all"].to_string(index=False))

    if args.method in ("linear", "both"):
        if not torch.cuda.is_available() and args.device.startswith("cuda"):
            args.device = "cpu"
        df_lin = run_rde_linear(
            args.device,
            args.num_workers,
            args.linear_trials,
            args.linear_folds,
            args.linear_iters,
        )
        lin_csv = OUT_DIR / "rde_linear_results.csv"
        df_lin.to_csv(lin_csv, index=False)
        metrics = eval_skempi_three_modes(df_lin)
        metrics.to_csv(OUT_DIR / "rde_linear_metrics_official.csv", index=False)
        summaries.append(_summarize(df_lin, "RDE-Linear"))
        print("\n=== RDE-Linear official metrics (mode=all) ===")
        print(metrics[metrics["mode"] == "all"].to_string(index=False))

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nWrote {summary_path}")
    for s in summaries:
        print(
            f"{s['method']}: all={s['overall_pearson']:.4f} | "
            f"pc(rde n>=10)={s['pc_pearson_rde_n10']:.4f} | "
            f"pc(copra n>3)={s['pc_pearson_copra_n3']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
