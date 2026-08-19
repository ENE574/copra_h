#!/usr/bin/env python3
"""Add train/val/test split columns for PEMPNI-style MPD training.

Modes:
  random (default): random complex holdout from MPD276.
  match_mpd48: pick val complexes whose features are closest to MPD48 PDBs.
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_seq(field: str) -> str:
    if pd.isna(field):
        return ""
    parts = []
    for part in str(field).split(","):
        part = part.strip()
        if not part:
            continue
        parts.append(part.split(":", 1)[1] if ":" in part else part)
    return "".join(parts)


def _seq_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _complex_table(df: pd.DataFrame, *, train: bool) -> pd.DataFrame:
    rows = []
    if train:
        groups = df.groupby("complex_group", sort=True)
    else:
        groups = df.groupby("PDB", sort=True)

    for key, sub in groups:
        prot = _parse_seq(sub["Protein sequences"].iloc[0])
        dna = _parse_seq(sub["DNA sequences"].iloc[0])
        rows.append(
            {
                "complex_group": int(sub["complex_group"].iloc[0]),
                "pdb": sub["PDB"].iloc[0],
                "n": len(sub),
                "ddg_mean": float(sub["DDG"].mean()),
                "ddg_std": float(sub["DDG"].std()) if len(sub) > 1 else 0.0,
                "prot_len": len(prot),
                "dna_len": len(dna),
                "prot_seq": prot,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_distance(train_tbl: pd.DataFrame, test_tbl: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    """Return distance matrix [n_test, n_train] and train complex_group ids."""
    feat_cols = ["n", "prot_len", "dna_len", "ddg_mean", "ddg_std"]
    all_tbl = pd.concat([train_tbl[feat_cols], test_tbl[feat_cols]], ignore_index=True)
    mu = all_tbl.mean().values
    sigma = all_tbl.std().values
    sigma[sigma < 1e-6] = 1.0

    z_train = (train_tbl[feat_cols].values - mu) / sigma
    z_test = (test_tbl[feat_cols].values - mu) / sigma
    feat_dist = np.linalg.norm(z_test[:, None, :] - z_train[None, :, :], axis=2)

    seq_dist = np.zeros_like(feat_dist)
    for i, trow in test_tbl.reset_index(drop=True).iterrows():
        for j, rrow in train_tbl.reset_index(drop=True).iterrows():
            seq_dist[i, j] = 1.0 - _seq_ratio(trow["prot_seq"], rrow["prot_seq"])

    dist = feat_dist + 0.75 * seq_dist
    return dist, train_tbl["complex_group"].astype(int).tolist()


def select_val_match_mpd48(
    train_tbl: pd.DataFrame,
    test_tbl: pd.DataFrame,
    *,
    n_val_complexes: int,
) -> list[int]:
    """Greedy k-center: cover MPD48 complexes with similar MPD276 val complexes."""
    dist, train_groups = _pairwise_distance(train_tbl, test_tbl)
    selected: list[int] = []
    remaining = set(train_groups)

    for _ in range(n_val_complexes):
        best_group = None
        best_score = float("inf")
        for group in remaining:
            cand = selected + [group]
            col_idx = [train_groups.index(g) for g in cand]
            score = float(np.sum(np.min(dist[:, col_idx], axis=1)))
            if score < best_score:
                best_score = score
                best_group = group
        if best_group is None:
            break
        selected.append(best_group)
        remaining.remove(best_group)
    return selected


def select_val_random(
    train_groups: np.ndarray,
    *,
    seed: int,
    n_val_complexes: int,
) -> list[int]:
    rng = np.random.RandomState(seed)
    picked = rng.choice(train_groups, size=n_val_complexes, replace=False)
    return [int(x) for x in picked]


def assign_fold_pempni_tv(
    df: pd.DataFrame,
    *,
    mode: str = "random",
    seed: int = 2024,
    n_val_complexes: int = 11,
    col_out: str = "fold_pempni_tv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" not in df.columns:
        raise ValueError("CSV must have a 'split' column (train/test).")
    if "complex_group" not in df.columns:
        raise ValueError("CSV must have a 'complex_group' column.")

    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"
    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    train_groups = train_df["complex_group"].unique()
    if n_val_complexes >= len(train_groups):
        raise ValueError(
            f"n_val_complexes={n_val_complexes} must be < num train complexes ({len(train_groups)})."
        )

    train_tbl = _complex_table(train_df, train=True)
    test_tbl = _complex_table(test_df, train=False)

    if mode == "random":
        val_groups = select_val_random(train_groups, seed=seed, n_val_complexes=n_val_complexes)
        match_report = pd.DataFrame()
    elif mode == "match_mpd48":
        val_groups = select_val_match_mpd48(
            train_tbl, test_tbl, n_val_complexes=n_val_complexes
        )
        dist, train_group_ids = _pairwise_distance(train_tbl, test_tbl)
        val_idx = [train_group_ids.index(g) for g in val_groups]
        nearest = []
        for ti in range(len(test_tbl)):
            j_local = int(np.argmin(dist[ti, val_idx]))
            j_global = val_idx[j_local]
            nearest.append(
                {
                    "mpd48_pdb": test_tbl.iloc[ti]["pdb"],
                    "nearest_val_pdb": train_tbl.iloc[j_global]["pdb"],
                    "distance": float(dist[ti, j_global]),
                }
            )
        match_report = pd.DataFrame(nearest)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    val_groups_set = set(val_groups)
    df = df.copy()
    df[col_out] = "test"
    df.loc[train_mask & df["complex_group"].isin(val_groups_set), col_out] = "val"
    df.loc[train_mask & ~df["complex_group"].isin(val_groups_set), col_out] = "train"

    manifest = (
        df.loc[train_mask, ["complex_group", "PDB", col_out]]
        .drop_duplicates(subset=["complex_group"])
        .sort_values("complex_group")
        .rename(columns={col_out: "role", "PDB": "pdb"})
    )
    return df, manifest, match_report


def _summary_stats(tbl: pd.DataFrame) -> dict[str, float]:
    return {
        "n_complexes": len(tbl),
        "n_mutations": float(tbl["n"].sum()),
        "mut_per_complex": float(tbl["n"].mean()),
        "prot_len_mean": float(tbl["prot_len"].mean()),
        "dna_len_mean": float(tbl["dna_len"].mean()),
        "ddg_mean": float(tbl["ddg_mean"].mean()),
        "ddg_std_mean": float(tbl["ddg_std"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/media/SSD0/csd/lrg/copra_h/datasets/MPD_merged/splits/MPD_merged_copra.csv"),
    )
    parser.add_argument("--mode", choices=["random", "match_mpd48"], default="random")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--n-val-complexes", type=int, default=11)
    parser.add_argument("--col-out", type=str, default="fold_pempni_tv")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--match-report",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.manifest is None:
        suffix = "match" if args.mode == "match_mpd48" else "random"
        args.manifest = args.csv.parent / f"MPD_pempni_train_val_{suffix}_manifest.csv"
    if args.match_report is None and args.mode == "match_mpd48":
        args.match_report = args.csv.parent / "MPD_pempni_val_to_mpd48_nearest.csv"

    df = pd.read_csv(args.csv)
    df, manifest, match_report = assign_fold_pempni_tv(
        df,
        mode=args.mode,
        seed=args.seed,
        n_val_complexes=args.n_val_complexes,
        col_out=args.col_out,
    )
    df.to_csv(args.csv, index=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    if len(match_report):
        match_report.to_csv(args.match_report, index=False)

    counts = df[args.col_out].value_counts().to_dict()
    print(f"Wrote {args.col_out} to {args.csv} (mode={args.mode})")
    print(f"Counts: {counts}")
    print(f"Manifest: {args.manifest}")
    print(manifest.groupby("role").size().to_dict(), "complexes")

    if args.mode == "match_mpd48":
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]
        train_tbl = _complex_table(train_df[train_df[args.col_out] == "train"], train=True)
        val_tbl = _complex_table(train_df[train_df[args.col_out] == "val"], train=True)
        test_tbl = _complex_table(test_df, train=False)
        print("\nDistribution summary:")
        for name, tbl in [("MPD48", test_tbl), ("val", val_tbl), ("train", train_tbl)]:
            print(name, _summary_stats(tbl))
        print(f"Match report: {args.match_report}")
        print("Val PDBs:", sorted(manifest[manifest.role == "val"].pdb.tolist()))


if __name__ == "__main__":
    main()
