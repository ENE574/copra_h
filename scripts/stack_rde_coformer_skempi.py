#!/usr/bin/env python3
"""Tier-0 stacking: RDE-Net OOF + CoFormer OOF on aligned SKEMPI rows.

Aligns CoFormer 5-fold test predictions to copra_h SKEMPI CSV via per-fold
sort on (PDB, DDG). Maps RDE-Net preds via PDB|MUTATION (mean if duplicated).

Usage:
  python copra_h/scripts/stack_rde_coformer_skempi.py
  python copra_h/scripts/stack_rde_coformer_skempi.py \\
    --coformer-run skempi_unified_tuneB_2026-06-18-11-48-05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SKEMPI_CSV = Path("/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/splits/skempi.csv")
DEFAULT_CF_ROOT = Path("/media/SSD0/csd/lrg/copra_h/outputs/SKEMPI_unified_cv")
DEFAULT_CF_RUN = "skempi_unified_2026-06-17-17-09-45"
DEFAULT_RDE_CSV = ROOT / "RDE-PPI/outputs/skempi_baseline/rde_net_results.csv"
DEFAULT_OUT = ROOT / "RDE-PPI/outputs/skempi_baseline/stacking"


def _pc_mean(df: pd.DataFrame, pred: str, group: str, min_n: int) -> float:
    pcs = []
    for _, sub in df.groupby(group):
        if len(sub) < min_n:
            continue
        r = sub["ddG"].corr(sub[pred])
        if pd.notna(r):
            pcs.append(r)
    return float(np.mean(pcs)) if pcs else float("nan")


def _metrics(df: pd.DataFrame, pred: str, group: str = "PDB") -> dict:
    return {
        "n": int(len(df)),
        "all_pearson": float(df["ddG"].corr(df[pred])),
        "all_spearman": float(df["ddG"].corr(df[pred], method="spearman")),
        "pc_pearson_rde_n10": _pc_mean(df, pred, group, min_n=10),
        "pc_pearson_copra_n3": _pc_mean(df, pred, group, min_n=3),
        "rmse": float(np.sqrt(np.mean((df["ddG"] - df[pred]) ** 2))),
        "mae": float(np.mean(np.abs(df["ddG"] - df[pred]))),
    }


def load_coformer_oof(run_dir: Path, sk: pd.DataFrame, num_folds: int = 5) -> pd.Series:
    preds = np.full(len(sk), np.nan, dtype=np.float64)
    for fold in range(num_folds):
        val_idx = sk.index[sk[f"fold_{fold}"] == "val"].to_numpy()
        val = sk.loc[val_idx].sort_values(["PDB", "DDG"]).reset_index(drop=True)
        test_csv = run_dir / f"log_fold_{fold}/pred/results_test.csv"
        if not test_csv.is_file():
            raise FileNotFoundError(test_csv)
        t = pd.read_csv(test_csv).sort_values(["complex", "y_true"]).reset_index(drop=True)
        if len(val) != len(t):
            raise ValueError(f"fold {fold}: val rows {len(val)} != test csv {len(t)}")
        if np.max(np.abs(val["DDG"].to_numpy() - t["y_true"].to_numpy())) > 1e-3:
            raise ValueError(f"fold {fold}: DDG mismatch after sort alignment")
        orig_idx = sk.loc[val_idx].sort_values(["PDB", "DDG"]).index.to_numpy()
        preds[orig_idx] = t["y_pred"].to_numpy()
    if np.isnan(preds).any():
        raise ValueError(f"missing CoFormer preds for {np.isnan(preds).sum()} rows")
    return pd.Series(preds, index=sk.index, name="ddG_pred_cf")


def load_rde_preds(rde_csv: Path, sk: pd.DataFrame) -> pd.Series:
    rde = pd.read_csv(rde_csv)
    rde["key"] = (
        rde["complex"].str.split("_").str[0] + "|" + rde["mutstr"].str.replace(" ", "")
    )
    agg = (
        rde.groupby("key", as_index=False)
        .agg(ddG_pred_rde=("ddG_pred", "mean"), ddG_rde=("ddG", "first"), n_rde=("ddG", "size"))
    )
    sk = sk.copy()
    sk["key"] = sk["PDB"] + "|" + sk["MUTATION"].astype(str)
    merged = sk[["key"]].merge(agg, on="key", how="left")
    missing = merged["ddG_pred_rde"].isna().sum()
    if missing:
        raise ValueError(f"RDE missing for {missing}/{len(sk)} copra SKEMPI rows")
    return merged["ddG_pred_rde"]


def grid_alpha_blend(df: pd.DataFrame) -> tuple[float, dict]:
    best_a, best_m = 0.0, -np.inf
    for a in np.linspace(0.0, 1.0, 101):
        pred = a * df["ddG_pred_rde"] + (1.0 - a) * df["ddG_pred_cf"]
        p = df["ddG"].corr(pred)
        if pd.notna(p) and p > best_m:
            best_m, best_a = p, a
    out = df.copy()
    out["ddG_pred"] = best_a * out["ddG_pred_rde"] + (1.0 - best_a) * out["ddG_pred_cf"]
    m = _metrics(out, "ddG_pred")
    m["alpha_rde"] = best_a
    m["alpha_cf"] = 1.0 - best_a
    return best_a, m


def meta_cv_ridge(df: pd.DataFrame, num_folds: int = 5) -> tuple[pd.Series, dict]:
    oof = np.full(len(df), np.nan, dtype=np.float64)
    for fold in range(num_folds):
        tr = df["oof_fold"] != fold
        te = df["oof_fold"] == fold
        x_tr = df.loc[tr, ["ddG_pred_rde", "ddG_pred_cf"]].to_numpy()
        y_tr = df.loc[tr, "ddG"].to_numpy()
        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(x_tr, y_tr)
        oof[te.to_numpy()] = reg.predict(
            df.loc[te, ["ddG_pred_rde", "ddG_pred_cf"]].to_numpy()
        )
    pred = pd.Series(oof, index=df.index, name="ddG_pred_stack_cv")
    out = df.copy()
    out["ddG_pred"] = pred
    return pred, _metrics(out, "ddG_pred")


def meta_cv_linear(df: pd.DataFrame, num_folds: int = 5) -> dict:
    oof = np.full(len(df), np.nan, dtype=np.float64)
    for fold in range(num_folds):
        tr = df["oof_fold"] != fold
        te = df["oof_fold"] == fold
        reg = LinearRegression(fit_intercept=True)
        reg.fit(
            df.loc[tr, ["ddG_pred_rde", "ddG_pred_cf"]].to_numpy(),
            df.loc[tr, "ddG"].to_numpy(),
        )
        oof[te.to_numpy()] = reg.predict(
            df.loc[te, ["ddG_pred_rde", "ddG_pred_cf"]].to_numpy()
        )
    out = df.copy()
    out["ddG_pred"] = oof
    m = _metrics(out, "ddG_pred")
    reg_full = LinearRegression(fit_intercept=True).fit(
        df[["ddG_pred_rde", "ddG_pred_cf"]].to_numpy(), df["ddG"].to_numpy()
    )
    m["coef_rde"] = float(reg_full.coef_[0])
    m["coef_cf"] = float(reg_full.coef_[1])
    m["intercept"] = float(reg_full.intercept_)
    return m


def residual_stack(df: pd.DataFrame) -> dict:
    reg = LinearRegression(fit_intercept=True)
    x = (df["ddG_pred_cf"] - df["ddG_pred_cf"].mean()).to_numpy()[:, None]
    reg.fit(x, (df["ddG"] - df["ddG_pred_rde"]).to_numpy())
    pred = df["ddG_pred_rde"] + reg.predict(x)
    out = df.copy()
    out["ddG_pred"] = pred
    m = _metrics(out, "ddG_pred")
    m["residual_coef"] = float(reg.coef_[0])
    m["residual_intercept"] = float(reg.intercept_)
    return m


def build_oof_table(
    sk_csv: Path,
    rde_csv: Path,
    coformer_run_dir: Path,
    num_folds: int,
) -> pd.DataFrame:
    sk = pd.read_csv(sk_csv)
    sk["ddG_pred_cf"] = load_coformer_oof(coformer_run_dir, sk, num_folds=num_folds)
    sk["ddG_pred_rde"] = load_rde_preds(rde_csv, sk)
    sk["ddG"] = sk["DDG"].astype(float)
    sk["oof_fold"] = -1
    for fold in range(num_folds):
        sk.loc[sk[f"fold_{fold}"] == "val", "oof_fold"] = fold
    if (sk["oof_fold"] < 0).any():
        raise ValueError("some rows never appear in a val fold")
    return sk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skempi-csv", type=Path, default=DEFAULT_SKEMPI_CSV)
    parser.add_argument("--rde-csv", type=Path, default=DEFAULT_RDE_CSV)
    parser.add_argument("--coformer-root", type=Path, default=DEFAULT_CF_ROOT)
    parser.add_argument("--coformer-run", default=DEFAULT_CF_RUN)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    run_dir = args.coformer_root / args.coformer_run
    if not run_dir.is_dir():
        raise SystemExit(f"CoFormer run not found: {run_dir}")
    if not args.rde_csv.is_file():
        raise SystemExit(f"RDE results not found: {args.rde_csv}")
    if not args.skempi_csv.is_file():
        raise SystemExit(f"SKEMPI csv not found: {args.skempi_csv}")

    df = build_oof_table(args.skempi_csv, args.rde_csv, run_dir, args.num_folds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "coformer_run": args.coformer_run,
        "n_samples": int(len(df)),
        "baselines": {
            "RDE-Net (copra subset)": _metrics(df, "ddG_pred_rde"),
            "CoFormer OOF": _metrics(df, "ddG_pred_cf"),
            "Mean (0.5/0.5)": _metrics(
                df.assign(ddG_pred=0.5 * df["ddG_pred_rde"] + 0.5 * df["ddG_pred_cf"]),
                "ddG_pred",
            ),
        },
        "stacking_in_sample": {},
        "stacking_meta_cv": {},
    }

    alpha, blend_m = grid_alpha_blend(df)
    summary["stacking_in_sample"]["alpha_blend"] = blend_m

    summary["stacking_meta_cv"]["ridge_1fold"] = meta_cv_ridge(df, args.num_folds)[1]
    summary["stacking_meta_cv"]["linear_2feat"] = meta_cv_linear(df, args.num_folds)
    summary["stacking_in_sample"]["residual_on_rde"] = residual_stack(df)

    out_csv = args.out_dir / f"oof_merged_{args.coformer_run}.csv"
    df[
        [
            "PDB",
            "MUTATION",
            "ddG",
            "ddG_pred_rde",
            "ddG_pred_cf",
            "oof_fold",
            "protein_level_group",
        ]
    ].to_csv(out_csv, index=False)

    summary_path = args.out_dir / f"stacking_summary_{args.coformer_run}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Aligned OOF rows: {len(df)}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_path}\n")
    for name, m in summary["baselines"].items():
        print(
            f"{name:28s} all={m['all_pearson']:.4f}  "
            f"pc(n>=10)={m['pc_pearson_rde_n10']:.4f}  "
            f"pc(n>3)={m['pc_pearson_copra_n3']:.4f}"
        )
    print()
    b = summary["stacking_in_sample"]["alpha_blend"]
    print(
        f"Best alpha blend (in-sample)  all={b['all_pearson']:.4f}  "
        f"pc(n>=10)={b['pc_pearson_rde_n10']:.4f}  "
        f"alpha_rde={b['alpha_rde']:.2f}"
    )
    r = summary["stacking_meta_cv"]["ridge_1fold"]
    print(
        f"Ridge meta-CV (honest)        all={r['all_pearson']:.4f}  "
        f"pc(n>=10)={r['pc_pearson_rde_n10']:.4f}"
    )
    lin = summary["stacking_meta_cv"]["linear_2feat"]
    print(
        f"Linear meta-CV (honest)       all={lin['all_pearson']:.4f}  "
        f"pc(n>=10)={lin['pc_pearson_rde_n10']:.4f}  "
        f"coef=({lin['coef_rde']:.3f},{lin['coef_cf']:.3f})"
    )
    res = summary["stacking_in_sample"]["residual_on_rde"]
    print(
        f"Residual RDE+CF (in-sample)   all={res['all_pearson']:.4f}  "
        f"pc(n>=10)={res['pc_pearson_rde_n10']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
