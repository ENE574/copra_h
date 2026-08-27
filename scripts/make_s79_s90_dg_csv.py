#!/usr/bin/env python
"""Generate dG inference CSVs for S79/S90 from their PDB structures.

S79/S90 ship only with `PDB` + `Binding_affinity` (and various scoring
columns) but the dG model needs `Protein sequences` / `Protein chains` /
`Protein chains B` / `Mutation sequences` / `fold_0` columns. We recover the
chain sequences by parsing each PDB and write a ready-to-infer CSV.

Label conversion to dG (kcal/mol, negative = binding):
  S79: Binding_affinity is all positive -> dG = -Binding_affinity
  S90: Binding_affinity is all negative -> dG = Binding_affinity

NOTE: chain assignment for multi-chain complexes is heuristic
(prot = first chain, partner = remaining chains). Verify for multi-chain
S79 complexes if exact per-complex metrics matter.
"""
import csv
import os
import glob
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.protein.proteins import ProteinInput

BASE = "/media/SSD0/csd/lrg/copra_h/datasets"
SPLITS = {
    "S79": os.path.join(BASE, "S79", "splits", "S79.csv"),
    "S90": os.path.join(BASE, "S90", "splits", "S90.csv"),
}


def parse_chains(pdb_path):
    d = ProteinInput.from_path(pdb_path, return_dict=True)
    # d: chain_id -> ProteinInput object; the sequence lives in .seq (str)
    out = []
    for c, obj in d.items():
        seq = getattr(obj, "seq", None)
        if isinstance(seq, str) and seq:
            out.append((c, seq))
    return out


def make_dg_csv(name, invert_label):
    src = SPLITS[name]
    pdb_dir = os.path.join(BASE, name, "PDBs")
    out = os.path.join(BASE, name, "splits", f"{name}_dg.csv")
    rows = list(csv.DictReader(open(src)))
    out_rows = []
    skipped = []
    for r in rows:
        pdb_raw = r.get("PDB") or r.get("PDB_NAME")
        pdb_id = pdb_raw.replace(".pdb", "").strip()
        pdb_path = os.path.join(pdb_dir, pdb_id + ".pdb")
        if not os.path.exists(pdb_path):
            skipped.append(pdb_id)
            continue
        chains = parse_chains(pdb_path)
        if len(chains) < 2:
            skipped.append(pdb_id + f"(chains={len(chains)})")
            continue
        prot_chains = [chains[0][0]]
        partner_chains = [c for c, _ in chains[1:]]
        prot_seq_field = ",".join(f"{c}:{s}" for c, s in chains[:1])
        partner_seq_field = ",".join(f"{c}:{s}" for c, s in chains[1:])
        try:
            ba = float(r["Binding_affinity"])
        except Exception:
            skipped.append(pdb_id + "(no BA)")
            continue
        dG = -ba if invert_label else ba
        out_rows.append({
            "PDB": pdb_id,
            "Protein chains": ",".join(prot_chains),
            "Protein chains B": ",".join(partner_chains),
            "dG": f"{dG:.6f}",
            "Protein sequences": prot_seq_field,
            "Protein sequences B": partner_seq_field,
            "Mutation sequences": "",
            "RNA sequences": "",
            "fold_0": "test",
        })
    fields = ["PDB", "Protein chains", "Protein chains B", "dG",
              "Protein sequences", "Protein sequences B",
              "Mutation sequences", "RNA sequences", "fold_0"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[{name}] wrote {len(out_rows)} rows -> {out}")
    if skipped:
        print(f"[{name}] skipped {len(skipped)}: {skipped[:10]}")


if __name__ == "__main__":
    make_dg_csv("S79", invert_label=True)
    make_dg_csv("S90", invert_label=False)
