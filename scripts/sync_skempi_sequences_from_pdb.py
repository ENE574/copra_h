#!/usr/bin/env python3
"""Sync SKEMPI CSV WT/Mut sequences to match PDB structure sequences."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.protein.proteins import ProteinInput
from data.structure_dataset import (
    _filter_chains_in_pdb,
    _chain_seq_dict,
    _map_csv_seq_to_struct,
    _mutation_sites_from_csv,
    _parse_chain_seq_entries,
)


def _format_chain_field(chain_ids: list, seq_by_chain: dict) -> str:
    parts = []
    for chain_id in chain_ids:
        seq = seq_by_chain.get(chain_id)
        if seq:
            parts.append(f"{chain_id}:{seq}")
    return ",".join(parts)


def _sync_chain_pair(struct_seq: str, old_wt: str, old_mut: str) -> tuple[str, str, bool]:
    old_wt = old_wt or struct_seq
    old_mut = old_mut or old_wt
    new_wt = struct_seq
    mapping = _map_csv_seq_to_struct(old_wt, struct_seq)
    mut_chars = list(struct_seq)
    for csv_idx, mut_aa in _mutation_sites_from_csv(old_wt, old_mut):
        struct_idx = mapping.get(csv_idx)
        if struct_idx is not None:
            mut_chars[struct_idx] = mut_aa
    new_mut = "".join(mut_chars)
    changed = (old_wt != new_wt) or (old_mut != new_mut)
    return new_wt, new_mut, changed


def _pdb_chain_seqs(pdb_path: Path, chains: list[str], cache: dict) -> dict[str, str]:
    key = (str(pdb_path), tuple(chains))
    if key not in cache:
        parsed = ProteinInput.from_path(str(pdb_path), return_dict=True, valid_chains=chains)
        cache[key] = {c: parsed[c].seq for c in chains if c in parsed}
    return cache[key]


def sync_row(row: pd.Series, pdb_root: Path, pdb_cache: dict) -> tuple[dict, list[str]]:
    pdb = str(row["PDB"]).strip()
    pdb_path = pdb_root / f"{pdb}.pdb"
    if not pdb_path.exists():
        return {}, [f"missing_pdb:{pdb}"]

    prot_chains = [
        c.strip() for c in str(row["Protein chains"]).split(",") if c.strip()
    ]
    partner_chains = [
        c.strip() for c in str(row["Protein chains B"]).split(",") if c.strip()
    ]
    prot_chains = _filter_chains_in_pdb(str(pdb_path), prot_chains)
    partner_chains = _filter_chains_in_pdb(str(pdb_path), partner_chains)
    all_chains = prot_chains + partner_chains
    if not all_chains:
        return {}, [f"no_valid_chains:{pdb}"]

    pdb_seqs = _pdb_chain_seqs(pdb_path, all_chains, pdb_cache)
    old_wt_prot = _chain_seq_dict(_parse_chain_seq_entries(row.get("Protein sequences")))
    old_wt_partner = _chain_seq_dict(_parse_chain_seq_entries(row.get("Protein sequences B")))
    old_mut = _chain_seq_dict(_parse_chain_seq_entries(row.get("Mutation sequences")))

    new_wt_prot = dict(old_wt_prot)
    new_wt_partner = dict(old_wt_partner)
    new_mut = dict(old_mut)
    notes: list[str] = []
    sid = f"{pdb}_{row['MUTATION']}"

    for chain_id in prot_chains:
        struct_seq = pdb_seqs.get(chain_id)
        if struct_seq is None:
            notes.append(f"missing_chain:{sid}:{chain_id}")
            continue
        old_w = old_wt_prot.get(chain_id, struct_seq)
        old_m = old_mut.get(chain_id, old_w)
        new_w, new_m, changed = _sync_chain_pair(struct_seq, old_w, old_m)
        new_wt_prot[chain_id] = new_w
        new_mut[chain_id] = new_m
        if changed:
            notes.append(f"updated:{sid}:{chain_id}:wt={len(old_w)}->{len(new_w)}")

    for chain_id in partner_chains:
        struct_seq = pdb_seqs.get(chain_id)
        if struct_seq is None:
            notes.append(f"missing_chain:{sid}:{chain_id}")
            continue
        old_w = old_wt_partner.get(chain_id, old_mut.get(chain_id, struct_seq))
        old_m = old_mut.get(chain_id, old_w)
        new_w, new_m, changed = _sync_chain_pair(struct_seq, old_w, old_m)
        new_wt_partner[chain_id] = new_w
        new_mut[chain_id] = new_m
        if changed:
            notes.append(f"updated:{sid}:{chain_id}:wt={len(old_w)}->{len(new_w)}")

    updates = {
        "Protein sequences": _format_chain_field(prot_chains, new_wt_prot),
        "Protein sequences B": _format_chain_field(partner_chains, new_wt_partner),
        "Mutation sequences": _format_chain_field(all_chains, new_mut),
    }
    return updates, notes


def audit_against_pdb(df: pd.DataFrame, pdb_root: Path) -> dict:
    pdb_cache: dict = {}
    stats = {
        "wt_len_mismatch": 0,
        "mut_len_mismatch": 0,
    }

    for _, row in df.iterrows():
        pdb = str(row["PDB"]).strip()
        pdb_path = pdb_root / f"{pdb}.pdb"
        if not pdb_path.exists():
            continue
        prot = [
            c.strip() for c in str(row["Protein chains"]).split(",") if c.strip()
        ]
        partner = [
            c.strip() for c in str(row["Protein chains B"]).split(",") if c.strip()
        ]
        prot = _filter_chains_in_pdb(str(pdb_path), prot)
        partner = _filter_chains_in_pdb(str(pdb_path), partner)
        pdb_seqs = _pdb_chain_seqs(pdb_path, prot + partner, pdb_cache)
        wt_prot = _chain_seq_dict(_parse_chain_seq_entries(row.get("Protein sequences")))
        wt_partner = _chain_seq_dict(_parse_chain_seq_entries(row.get("Protein sequences B")))
        mut = _chain_seq_dict(_parse_chain_seq_entries(row.get("Mutation sequences")))
        for chain_id in prot:
            s = pdb_seqs.get(chain_id)
            w = wt_prot.get(chain_id, "")
            if s and len(w) != len(s):
                stats["wt_len_mismatch"] += 1
            m = mut.get(chain_id, w)
            if s and len(m) != len(s):
                stats["mut_len_mismatch"] += 1
        for chain_id in partner:
            s = pdb_seqs.get(chain_id)
            w = wt_partner.get(chain_id, mut.get(chain_id, ""))
            if s and len(w) != len(s):
                stats["wt_len_mismatch"] += 1
            m = mut.get(chain_id, w)
            if s and len(m) != len(s):
                stats["mut_len_mismatch"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/splits/skempi.csv",
    )
    parser.add_argument(
        "--pdb-root",
        default="/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/PDBs",
    )
    parser.add_argument(
        "--report",
        default="/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/splits/skempi_pdb_seq_sync_report.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    pdb_root = Path(args.pdb_root)
    report_path = Path(args.report)

    df = pd.read_csv(csv_path)
    pdb_cache: dict = {}
    report_rows = []
    changed_rows = 0

    for idx, row in df.iterrows():
        updates, notes = sync_row(row, pdb_root, pdb_cache)
        if updates:
            old_prot = str(row.get("Protein sequences", ""))
            old_partner = str(row.get("Protein sequences B", ""))
            old_mut = str(row.get("Mutation sequences", ""))
            changed = (
                updates["Protein sequences"] != old_prot
                or updates["Protein sequences B"] != old_partner
                or updates["Mutation sequences"] != old_mut
            )
            if changed:
                changed_rows += 1
                if not args.dry_run:
                    for col, val in updates.items():
                        df.at[idx, col] = val
            for note in notes:
                if note.startswith("updated:"):
                    report_rows.append({"row_index": idx, "note": note})

    print(f"rows scanned: {len(df)}")
    print(f"rows changed: {changed_rows}")
    print(f"chain updates logged: {len(report_rows)}")

    if args.dry_run:
        print("dry-run: CSV not written")
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = csv_path.with_suffix(csv_path.suffix + f".bak_pre_pdb_seq_sync_{ts}")
        shutil.copy2(csv_path, backup)
        df.to_csv(csv_path, index=False)
        print(f"backup: {backup}")
        print(f"updated: {csv_path}")

    if report_rows:
        pd.DataFrame(report_rows).to_csv(report_path, index=False)
        print(f"report: {report_path}")

    print("post-sync audit:")
    audit_df = df if not args.dry_run else df.copy()
    if args.dry_run:
        for idx, row in audit_df.iterrows():
            updates, _ = sync_row(row, pdb_root, pdb_cache)
            for col, val in updates.items():
                audit_df.at[idx, col] = val
    stats = audit_against_pdb(audit_df, pdb_root)
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
