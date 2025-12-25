#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

import extract_features as fe


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _as_path(base: Path, maybe_path: str) -> str:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return str(p if p.is_absolute() else base / p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Online embedding extraction without editing source configs.")
    parser.add_argument("--config", default="config/feature_extract.yml", help="Base feature extract config.")
    parser.add_argument("--output_root", default=None, help="Output root dir (overrides config output_dir).")
    parser.add_argument("--protein_fasta", default=None, help="Protein fasta path (file or dir).")
    parser.add_argument("--rna_fasta", default=None, help="RNA fasta path (file or dir).")
    parser.add_argument("--pdb_dir", default=None, help="PDB directory for structure models.")
    parser.add_argument("--models", default="", help="Comma-separated model names to run.")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda or cpu).")
    args = parser.parse_args()

    base_dir = Path.cwd().resolve()
    cfg_path = Path(args.config).resolve()
    cfg = _load_config(str(cfg_path))

    output_root = Path(args.output_root or cfg.get("output_dir", "outputs/feature_extraction")).resolve()
    device = args.device or cfg.get("device", "cuda")

    if "protein_sequence" in cfg:
        if args.protein_fasta:
            cfg["protein_sequence"]["fasta"] = _as_path(base_dir, args.protein_fasta)
    if "rna_sequence" in cfg:
        if args.rna_fasta:
            cfg["rna_sequence"]["fasta"] = _as_path(base_dir, args.rna_fasta)
    if "rna_structure" in cfg:
        if args.rna_fasta:
            cfg["rna_structure"]["fasta"] = _as_path(base_dir, args.rna_fasta)
    if "protein_structure" in cfg:
        if args.pdb_dir:
            cfg["protein_structure"]["pdb_dir"] = _as_path(base_dir, args.pdb_dir)

    selected = set(m.strip() for m in args.models.split(",") if m.strip()) or None

    if "protein_sequence" in cfg:
        fe.run_protein_sequence(cfg["protein_sequence"], base_dir, output_root, device, allowed=selected)
    if "protein_structure" in cfg:
        fe.run_protein_structure(cfg["protein_structure"], base_dir, output_root, device, allowed=selected)
    if "rna_sequence" in cfg:
        fe.run_rna_sequence(cfg["rna_sequence"], base_dir, output_root, device, allowed=selected)
    if "rna_structure" in cfg:
        fe.run_rna_structure(cfg["rna_structure"], base_dir, output_root, device, allowed=selected)


if __name__ == "__main__":
    main()
