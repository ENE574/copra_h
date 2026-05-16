import argparse
from pathlib import Path

import yaml

from typing import Optional, Set

from feature_extraction import extractors
from feature_extraction.dataset_utils import build_dataset_fastas, build_dataset_pdb_list


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _as_path(base: Path, maybe_path: str) -> str:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return str(p if p.is_absolute() else base / p)


def _allowed(name: str, allowed: Optional[Set[str]]) -> bool:
    return allowed is None or name in allowed


def run_protein_sequence(cfg: dict, base_dir: Path, output_root: Path, device: str, allowed: Optional[Set[str]] = None) -> None:
    fasta = _as_path(base_dir, cfg.get("fasta"))
    if not fasta:
        return
    models = cfg.get("models", {})
    if models.get("esm2", {}).get("enabled", True) and _allowed("esm2", allowed):
        model_cfg = models.get("esm2", {})
        extractors.extract_esm2(
            fasta_path=fasta,
            output_dir=str(output_root / "protein_sequence" / "esm2"),
            device=device,
            model_location=_as_path(base_dir, model_cfg.get("model_location")),
            repr_layer=model_cfg.get("repr_layer"),
            batch_size=model_cfg.get("batch_size", 1),
        )
    if models.get("prott5", {}).get("enabled", True) and _allowed("prott5", allowed):
        model_cfg = models.get("prott5", {})
        extractors.extract_prott5(
            fasta_path=fasta,
            output_dir=str(output_root / "protein_sequence" / "prott5"),
            device=device,
            model_dir=_as_path(base_dir, model_cfg.get("model_dir", "weights/ProtT5_weights")),
            batch_size=model_cfg.get("batch_size", 1),
        )
    if models.get("saprot", {}).get("enabled", True) and _allowed("saprot", allowed):
        model_cfg = models.get("saprot", {})
        extractors.extract_saprot(
            fasta_path=fasta,
            output_dir=str(output_root / "protein_sequence" / "saprot"),
            device=device,
            model_dir=_as_path(base_dir, model_cfg.get("model_dir", "weights/SaProt_weights")),
            batch_size=model_cfg.get("batch_size", 1),
        )


def run_protein_structure(cfg: dict, base_dir: Path, output_root: Path, device: str, allowed: Optional[Set[str]] = None) -> None:
    pdb_dir = _as_path(base_dir, cfg.get("pdb_dir"))
    fasta = _as_path(base_dir, cfg.get("fasta"))
    pdb_list = cfg.get("pdb_list")
    models = cfg.get("models", {})
    protein_single = output_root / "inputs" / "protein_single"
    protein_single_arg = str(protein_single) if protein_single.is_dir() else None
    if pdb_dir and models.get("esm_if1", {}).get("enabled", True) and _allowed("esm_if1", allowed):
        model_cfg = models.get("esm_if1", {})
        extractors.extract_esm_if1(
            pdb_dir=pdb_dir,
            output_dir=str(output_root / "protein_structure" / "esm_if1"),
            device=device,
            model_location=_as_path(base_dir, model_cfg.get("model_location")),
            chain_id=model_cfg.get("chain_id"),
            pdb_list=pdb_list,
            protein_single_dir=model_cfg.get("protein_single_dir") or protein_single_arg,
        )
    # Optional: requires vendored `ProteinMPNN/` (not in all checkouts); default off.
    if pdb_dir and models.get("proteinmpnn", {}).get("enabled", False) and _allowed("proteinmpnn", allowed):
        model_cfg = models.get("proteinmpnn", {})
        extractors.extract_protein_mpnn(
            pdb_dir=pdb_dir,
            output_dir=str(output_root / "protein_structure" / "proteinmpnn"),
            device=device,
            model_weights=_as_path(base_dir, model_cfg.get("model_weights", "weights/ProteinMPNN_weights/v_48_020.pt")),
            model_weights_ca_only=_as_path(base_dir, model_cfg.get("model_weights_ca_only"))
            if model_cfg.get("model_weights_ca_only")
            else None,
            ca_only=bool(model_cfg.get("ca_only", False)),
            pdb_list=pdb_list,
        )
    if fasta and models.get("protbert", {}).get("enabled", True) and _allowed("protbert", allowed):
        model_cfg = models.get("protbert", {})
        extractors.extract_protbert(
            fasta_path=fasta,
            output_dir=str(output_root / "protein_structure" / "protbert"),
            device=device,
            model_dir=_as_path(base_dir, model_cfg.get("model_dir", "weights/ProtBert_weights")),
            batch_size=model_cfg.get("batch_size", 1),
        )
    if pdb_dir and models.get("protrek", {}).get("enabled", False) and _allowed("protrek", allowed):
        model_cfg = models.get("protrek", {})
        extractors.extract_protrek(
            pdb_dir=pdb_dir,
            output_dir=str(output_root / "protein_structure" / "protrek"),
            device=device,
            model_dir=_as_path(base_dir, model_cfg.get("model_dir")),
            from_checkpoint=_as_path(base_dir, model_cfg.get("from_checkpoint")),
            protein_config=_as_path(base_dir, model_cfg.get("protein_config")),
            text_config=_as_path(base_dir, model_cfg.get("text_config")),
            structure_config=_as_path(base_dir, model_cfg.get("structure_config")),
            foldseek_bin=_as_path(base_dir, model_cfg.get("foldseek_bin")),
            batch_size=int(model_cfg.get("batch_size", 32)),
            chain_id=model_cfg.get("chain_id"),
            pdb_list=pdb_list,
            protein_single_dir=model_cfg.get("protein_single_dir") or protein_single_arg,
        )
    if models.get("alphafold2", {}).get("enabled", False) and _allowed("alphafold2", allowed):
        model_cfg = models.get("alphafold2", {})
        fasta = _as_path(base_dir, model_cfg.get("fasta"))
        output_dir = _as_path(
            base_dir,
            model_cfg.get("output_dir") or str(output_root / "protein_structure" / "alphafold2"),
        )
        data_dir = _as_path(base_dir, model_cfg.get("data_dir"))
        if fasta and data_dir and output_dir:
            extractors.extract_alphafold2_outputs(
                fasta_path=fasta,
                output_dir=output_dir,
                data_dir=data_dir,
                model_preset=model_cfg.get("model_preset", "monomer"),
                db_preset=model_cfg.get("db_preset", "reduced_dbs"),
                max_template_date=model_cfg.get("max_template_date", "2020-05-14"),
                use_gpu_relax=bool(model_cfg.get("use_gpu_relax", False)),
            )
            extractors.parse_alphafold2_features(
                output_dir=output_dir,
                save_dir=str(output_root / "protein_structure" / "alphafold2_features"),
            )


def run_rna_sequence(cfg: dict, base_dir: Path, output_root: Path, device: str, allowed: Optional[Set[str]] = None) -> None:
    fasta = _as_path(base_dir, cfg.get("fasta"))
    models = cfg.get("models", {})
    if fasta and models.get("rinalmo", {}).get("enabled", True) and _allowed("rinalmo", allowed):
        model_cfg = models.get("rinalmo", {})
        extractors.extract_rinalmo(
            fasta_path=fasta,
            output_dir=str(output_root / "rna_sequence" / "rinalmo"),
            device=device,
            model_weights=_as_path(base_dir, model_cfg.get("model_weights", "weights/rinalmo_weights/rinalmo_giga_pretrained.pt")),
            rinalmo_type=model_cfg.get("rinalmo_type", "650M"),
            batch_size=model_cfg.get("batch_size", 1),
        )
    if fasta and models.get("rna_fm", {}).get("enabled", True) and _allowed("rna_fm", allowed):
        model_cfg = models.get("rna_fm", {})
        extractors.extract_rna_fm(
            fasta_path=fasta,
            output_dir=str(output_root / "rna_sequence" / "rna_fm"),
            device=device,
            model_path=_as_path(base_dir, model_cfg.get("model_path", "weights/RNA-FM_weights/RNA-FM_pretrained.pth")),
            repr_layer=model_cfg.get("repr_layer"),
            batch_size=model_cfg.get("batch_size", 1),
        )
    if models.get("rna_msm", {}).get("enabled", False) and _allowed("rna_msm", allowed):
        model_cfg = models.get("rna_msm", {})
        extractors.extract_rna_msm(
            root_path=_as_path(base_dir, model_cfg.get("root_path", str(base_dir))),
            msa_path=_as_path(base_dir, model_cfg.get("msa_path", "RNA-MSM/results")),
            msa_list=_as_path(base_dir, model_cfg.get("msa_list")),
            model_path=_as_path(base_dir, model_cfg.get("model_path", "weights/RNA-MSM_weights/RNA_MSM_pretrained.ckpt")),
            output_dir=str(output_root / "rna_sequence" / "rna_msm"),
            device=model_cfg.get("device", device),
            extra_overrides=model_cfg.get("extra_overrides"),
            rna_single_dir=_as_path(base_dir, model_cfg["rna_single_dir"])
            if model_cfg.get("rna_single_dir")
            else None,
            build_missing_msas=bool(model_cfg.get("build_missing_msas", True)),
        )


def run_rna_structure(cfg: dict, base_dir: Path, output_root: Path, device: str, allowed: Optional[Set[str]] = None) -> None:
    models = cfg.get("models", {})
    fasta = _as_path(base_dir, cfg.get("fasta"))
    if fasta and models.get("rna_ernie", {}).get("enabled", False) and _allowed("rna_ernie", allowed):
        model_cfg = models.get("rna_ernie", {})
        extractors.extract_rna_ernie(
            fasta_path=fasta,
            output_dir=str(output_root / "rna_structure" / "rna_ernie"),
            device=model_cfg.get("device", "cpu"),
            model_dir=_as_path(base_dir, model_cfg.get("model_dir", "weights/RNAErnie_weights")),
            vocab_path=model_cfg.get("vocab_path", "RNAErnie/data/vocab/vocab_1MER.txt"),
            batch_size=model_cfg.get("batch_size", 256),
            max_seq_len=model_cfg.get("max_seq_len", 512),
        )
    if fasta and models.get("rnabert", {}).get("enabled", False) and _allowed("rnabert", allowed):
        model_cfg = models.get("rnabert", {})
        extractors.extract_rnabert(
            fasta_path=fasta,
            output_dir=str(output_root / "rna_structure" / "rnabert"),
            model_weights=_as_path(base_dir, model_cfg.get("model_weights", "weights/RNABERT_weights/bert_mul_2.pth")),
            batch_size=model_cfg.get("batch_size", 40),
            device=str(model_cfg.get("device", "cpu")),
        )
    if fasta and models.get("rhofold", {}).get("enabled", False) and _allowed("rhofold", allowed):
        model_cfg = models.get("rhofold", {})
        rhofold_fasta = _as_path(base_dir, model_cfg.get("fasta")) or fasta
        extractors.extract_rhofold(
            fasta_path=rhofold_fasta,
            output_dir=str(output_root / "rna_structure" / "rhofold"),
            device=model_cfg.get("device", device),
            ckpt_path=_as_path(base_dir, model_cfg.get("ckpt_path", "weights/RhFold_weights/model_20221010_params.pt")),
            input_a3m=_as_path(base_dir, model_cfg.get("input_a3m")),
            single_seq_pred=bool(model_cfg.get("single_seq_pred", True)),
            max_rna_length=int(model_cfg.get("max_rna_length", 1000)),
            truncate_rna=bool(model_cfg.get("truncate_rna", False)),
            cuda_safe_max_rna_length=int(model_cfg.get("cuda_safe_max_rna_length", 512)),
            skip_existing=bool(model_cfg.get("skip_existing", True)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model feature extraction.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to run (e.g. rna_ernie,rna_msm). Empty means run all enabled.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    # Resolve relative paths from the project root (cwd), not the config directory.
    base_dir = Path.cwd().resolve()
    cfg = _load_config(str(cfg_path))
    output_root = Path(cfg.get("output_dir", "outputs/feature_extraction")).resolve()
    device = cfg.get("device", "cuda")

    extractors.write_run_metadata(str(output_root), {"config_path": str(cfg_path)})
    selected = set(m.strip() for m in args.models.split(",") if m.strip()) or None

    if "dataset" in cfg:
        ds_cfg = cfg["dataset"]
        csv_path = _as_path(base_dir, ds_cfg.get("csv_path"))
        if csv_path:
            inputs_dir = Path(output_root) / "inputs"
            pdb_list_path = ds_cfg.get("pdb_list_path")
            pdb_list_resolved = _as_path(base_dir, pdb_list_path) if pdb_list_path else None
            extra_csv = ds_cfg.get("extra_csv_paths") or []
            extra_csv_resolved = [_as_path(base_dir, p) for p in extra_csv] if extra_csv else None
            mut_col = ds_cfg.get("mutation_col")
            fasta_paths = build_dataset_fastas(
                csv_path=csv_path,
                output_dir=str(inputs_dir),
                id_col=ds_cfg.get("id_col", "PDB"),
                protein_seq_col=ds_cfg.get("protein_seq_col", "Protein sequences"),
                rna_seq_col=ds_cfg.get("rna_seq_col", "RNA sequences"),
                protein_chain_col=ds_cfg.get("protein_chain_col", "Protein chains"),
                rna_chain_col=ds_cfg.get("rna_chain_col", "RNA chains"),
                pdb_list_path=pdb_list_resolved,
                extra_csv_paths=extra_csv_resolved,
                mutation_col=mut_col,
            )
            pdb_dir = ds_cfg.get("pdb_dir")
            if pdb_dir:
                pdb_info = build_dataset_pdb_list(
                    csv_path=csv_path,
                    pdb_dir=_as_path(base_dir, pdb_dir),
                    id_col=ds_cfg.get("id_col", "PDB"),
                    protein_chain_col=ds_cfg.get("protein_chain_col", "Protein chains"),
                    rna_chain_col=ds_cfg.get("rna_chain_col", "RNA chains"),
                    pdb_list_path=pdb_list_resolved,
                    extra_csv_paths=extra_csv_resolved,
                    mutation_col=mut_col,
                )
            cfg.setdefault("protein_sequence", {})
            cfg.setdefault("rna_sequence", {})
            cfg.setdefault("rna_structure", {})
            cfg.setdefault("protein_structure", {})
            if not cfg["protein_sequence"].get("fasta"):
                cfg["protein_sequence"]["fasta"] = fasta_paths["protein_fasta"]
            if not cfg["rna_sequence"].get("fasta"):
                cfg["rna_sequence"]["fasta"] = fasta_paths["rna_fasta"]
            if not cfg["rna_structure"].get("fasta"):
                cfg["rna_structure"]["fasta"] = fasta_paths["rna_fasta"]
            if not cfg["protein_structure"].get("pdb_dir"):
                cfg["protein_structure"]["pdb_dir"] = _as_path(base_dir, ds_cfg.get("pdb_dir"))
            if not cfg["protein_structure"].get("fasta"):
                cfg["protein_structure"]["fasta"] = fasta_paths["protein_fasta"]
            if pdb_dir and not cfg["protein_structure"].get("pdb_list"):
                cfg["protein_structure"]["pdb_list"] = pdb_info["pdb_files"]
            models_cfg = cfg["protein_structure"].setdefault("models", {})
            if "alphafold2" in models_cfg:
                if not models_cfg["alphafold2"].get("fasta"):
                    models_cfg["alphafold2"]["fasta"] = fasta_paths["protein_single_dir"]
            rna_struct_models = cfg["rna_structure"].setdefault("models", {})
            if "rhofold" in rna_struct_models:
                if not rna_struct_models["rhofold"].get("fasta"):
                    rna_struct_models["rhofold"]["fasta"] = fasta_paths["rna_single_dir"]
            rna_models = cfg["rna_sequence"].setdefault("models", {})
            if "rna_msm" in rna_models:
                msa_list_path = rna_models["rna_msm"].get("msa_list")
                if not msa_list_path:
                    rna_models["rna_msm"]["msa_list"] = fasta_paths["rna_msm_ids_unique"]
                else:
                    msa_list_resolved = Path(_as_path(base_dir, msa_list_path))
                    if not msa_list_resolved.exists():
                        rna_models["rna_msm"]["msa_list"] = fasta_paths["rna_msm_ids_unique"]
                rna_models["rna_msm"].setdefault("rna_single_dir", fasta_paths["rna_single_dir"])

    if "protein_sequence" in cfg:
        run_protein_sequence(cfg["protein_sequence"], base_dir, output_root, device, allowed=selected)
    if "protein_structure" in cfg:
        run_protein_structure(cfg["protein_structure"], base_dir, output_root, device, allowed=selected)
    if "rna_sequence" in cfg:
        run_rna_sequence(cfg["rna_sequence"], base_dir, output_root, device, allowed=selected)
    if "rna_structure" in cfg:
        run_rna_structure(cfg["rna_structure"], base_dir, output_root, device, allowed=selected)


if __name__ == "__main__":
    main()
