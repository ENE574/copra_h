#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
from pathlib import Path

import yaml
import torch


def _as_path(base: Path, maybe_path: str) -> str | None:
    if maybe_path is None:
        return None
    p = Path(maybe_path)
    return str(p if p.is_absolute() else base / p)


def _print_result(name: str, ok: bool, detail: str = "") -> None:
    status = "OK" if ok else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)


def _guard(fn, name: str):
    try:
        fn()
        _print_result(name, True)
        return True
    except Exception as exc:  # noqa: BLE001
        _print_result(name, False, str(exc))
        traceback.print_exc()
        return False


def check_esm2(cfg, base: Path):
    import esm

    model_location = _as_path(base, cfg.get("model_location"))
    if not model_location or not Path(model_location).exists():
        raise FileNotFoundError(f"missing model_location: {model_location}")
    model_path = Path(model_location)
    model_data = torch.load(str(model_path), map_location="cpu")
    esm.pretrained.load_model_and_alphabet_core(model_path.stem, model_data, regression_data=None)


def check_prott5(cfg, base: Path):
    from transformers import T5EncoderModel, T5Tokenizer

    model_dir = _as_path(base, cfg.get("model_dir", "weights/ProtT5_weights"))
    if not model_dir or not Path(model_dir).exists():
        raise FileNotFoundError(f"missing model_dir: {model_dir}")
    T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    T5EncoderModel.from_pretrained(model_dir)


def check_saprot(cfg, base: Path):
    from transformers import EsmForMaskedLM, EsmTokenizer

    model_dir = _as_path(base, cfg.get("model_dir", "weights/SaProt_weights"))
    if not model_dir or not Path(model_dir).exists():
        raise FileNotFoundError(f"missing model_dir: {model_dir}")
    EsmTokenizer.from_pretrained(model_dir)
    EsmForMaskedLM.from_pretrained(model_dir)


def check_esm_if1(cfg, base: Path):
    import esm

    model_location = _as_path(base, cfg.get("model_location"))
    if not model_location or not Path(model_location).exists():
        raise FileNotFoundError(f"missing model_location: {model_location}")
    esm.pretrained.load_model_and_alphabet(model_location)


def check_proteinmpnn(cfg, base: Path):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "ProteinMPNN"))
    import protein_mpnn_utils as mpnn_utils

    model_weights = _as_path(base, cfg.get("model_weights", "weights/ProteinMPNN_weights/v_48_020.pt"))
    if not model_weights or not Path(model_weights).exists():
        raise FileNotFoundError(f"missing model_weights: {model_weights}")
    checkpoint = torch.load(model_weights, map_location="cpu")
    hidden_dim = checkpoint.get("hidden_dim", 128)
    num_layers = checkpoint.get("num_layers", 3)
    model = mpnn_utils.ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,
        k_neighbors=checkpoint["num_edges"],
        ca_only=bool(cfg.get("ca_only", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])


def check_protbert(cfg, base: Path):
    from transformers import BertModel, BertTokenizer

    model_dir = _as_path(base, cfg.get("model_dir", "weights/ProtBert_weights"))
    if not model_dir or not Path(model_dir).exists():
        raise FileNotFoundError(f"missing model_dir: {model_dir}")
    BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
    BertModel.from_pretrained(model_dir)


def check_alphafold2(cfg, base: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "alphafold" / "run_alphafold.py"
    data_dir = _as_path(base, cfg.get("data_dir"))
    if not script_path.exists():
        raise FileNotFoundError(f"missing script: {script_path}")
    if not data_dir or not Path(data_dir).exists():
        raise FileNotFoundError(f"missing data_dir: {data_dir}")


def check_protrek(cfg, base: Path):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.modules.pop("utils", None)
    from feature_extraction.extractors import _resolve_foldseek_executable
    from ProTrek.model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

    model_dir = _as_path(base, cfg.get("model_dir"))
    if not model_dir or not Path(model_dir).exists():
        raise FileNotFoundError(f"missing model_dir: {model_dir}")
    foldseek_hint = _as_path(base, cfg.get("foldseek_bin")) if cfg.get("foldseek_bin") else None
    foldseek_path = _resolve_foldseek_executable(repo_root, foldseek_hint)
    if not foldseek_path:
        raise FileNotFoundError(f"missing foldseek executable (foldseek_bin={cfg.get('foldseek_bin')!r})")

    protein_config = _as_path(base, cfg.get("protein_config")) or str(Path(model_dir) / "esm2_t33_650M_UR50D")
    text_config = _as_path(base, cfg.get("text_config")) or str(Path(model_dir) / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    structure_config = _as_path(base, cfg.get("structure_config")) or str(Path(model_dir) / "foldseek_t30_150M")
    from_checkpoint = _as_path(base, cfg.get("from_checkpoint")) or str(Path(model_dir) / "ProTrek_650M.pt")
    if not Path(from_checkpoint).exists():
        raise FileNotFoundError(f"missing from_checkpoint: {from_checkpoint}")

    ProTrekTrimodalModel(
        protein_config=protein_config,
        text_config=text_config,
        structure_config=structure_config,
        from_checkpoint=from_checkpoint,
        init_metrics=False,
    )

def check_rinalmo(cfg, base: Path):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "RiNALMo"))
    from rinalmo.model.model import RiNALMo
    from rinalmo.config import model_config

    model_weights = _as_path(base, cfg.get("model_weights", "weights/rinalmo_weights/rinalmo_giga_pretrained.pt"))
    if not model_weights or not Path(model_weights).exists():
        raise FileNotFoundError(f"missing model_weights: {model_weights}")
    rinalmo_type = cfg.get("rinalmo_type", "650M")
    size = {"650M": "giga", "150M": "mega", "35M": "micro", "8M": "nano"}.get(rinalmo_type)
    if not size:
        raise ValueError(f"unsupported rinalmo_type: {rinalmo_type}")
    config = model_config(size)
    model = RiNALMo(config)
    model.load_state_dict(torch.load(model_weights, map_location="cpu"))


def check_rna_fm(cfg, base: Path):
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "RNA-FM"))
    import fm

    model_path = _as_path(base, cfg.get("model_path", "weights/RNA-FM_weights/RNA-FM_pretrained.pth"))
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"missing model_path: {model_path}")
    fm.pretrained.rna_fm_t12(model_location=model_path)


def check_rna_msm(cfg, base: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "RNA-MSM" / "RNA_MSM_Inference.py"
    model_path = _as_path(base, cfg.get("model_path", "weights/RNA-MSM_weights/RNA_MSM_pretrained.ckpt"))
    if not script_path.exists():
        raise FileNotFoundError(f"missing script: {script_path}")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"missing model_path: {model_path}")


def check_rna_ernie(cfg, base: Path):
    try:
        import paddle
        from paddlenlp.transformers import ErnieModel
    except ImportError as exc:
        raise RuntimeError("RNAErnie requires paddlepaddle and paddlenlp") from exc

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = _as_path(base, cfg.get("model_dir", "weights/RNAErnie_weights"))
    if not model_dir or not Path(model_dir).exists():
        raise FileNotFoundError(f"missing model_dir: {model_dir}")
    ErnieModel.from_pretrained(str(repo_root / model_dir))


def check_rnabert(cfg, base: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "RNABERT" / "MLM_SFP.py"
    model_weights = _as_path(base, cfg.get("model_weights", "weights/RNABERT_weights/bert_mul_2.pth"))
    if not script_path.exists():
        raise FileNotFoundError(f"missing script: {script_path}")
    if not model_weights or not Path(model_weights).exists():
        raise FileNotFoundError(f"missing model_weights: {model_weights}")


def check_rhofold(cfg, base: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "RhoFold" / "inference.py"
    ckpt_path = _as_path(base, cfg.get("ckpt_path", "weights/RhFold_weights/model_20221010_params.pt"))
    if not script_path.exists():
        raise FileNotFoundError(f"missing script: {script_path}")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"missing ckpt_path: {ckpt_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that enabled models can load.")
    parser.add_argument("--config", required=True, help="Path to feature_extract.yml")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    base = Path.cwd().resolve()
    cfg = yaml.safe_load(Path(args.config).read_text())

    failures = 0

    protein_sequence = cfg.get("protein_sequence", {}).get("models", {})
    if protein_sequence.get("esm2", {}).get("enabled", True):
        failures += 0 if _guard(lambda: check_esm2(protein_sequence.get("esm2", {}), base), "esm2") else 1
    if protein_sequence.get("prott5", {}).get("enabled", True):
        failures += 0 if _guard(lambda: check_prott5(protein_sequence.get("prott5", {}), base), "prott5") else 1
    if protein_sequence.get("saprot", {}).get("enabled", True):
        failures += 0 if _guard(lambda: check_saprot(protein_sequence.get("saprot", {}), base), "saprot") else 1

    protein_structure = cfg.get("protein_structure", {}).get("models", {})
    if protein_structure.get("esm_if1", {}).get("enabled", True):
        failures += 0 if _guard(lambda: check_esm_if1(protein_structure.get("esm_if1", {}), base), "esm_if1") else 1
    if protein_structure.get("protbert", {}).get("enabled", True):
        failures += 0 if _guard(lambda: check_protbert(protein_structure.get("protbert", {}), base), "protbert") else 1
    if protein_structure.get("alphafold2", {}).get("enabled", False):
        failures += 0 if _guard(lambda: check_alphafold2(protein_structure.get("alphafold2", {}), base), "alphafold2") else 1
    if protein_structure.get("protrek", {}).get("enabled", False):
        failures += 0 if _guard(lambda: check_protrek(protein_structure.get("protrek", {}), base), "protrek") else 1

    rna_sequence = cfg.get("rna_sequence", {}).get("models", {})
    if rna_sequence.get("rinalmo", {}).get("enabled", True):
        failures += 0 if _guard(lambda: check_rinalmo(rna_sequence.get("rinalmo", {}), base), "rinalmo") else 1
    if rna_sequence.get("rna_fm", {}).get("enabled", True):
        failures += 0 if _guard(lambda: check_rna_fm(rna_sequence.get("rna_fm", {}), base), "rna_fm") else 1
    if rna_sequence.get("rna_msm", {}).get("enabled", False):
        failures += 0 if _guard(lambda: check_rna_msm(rna_sequence.get("rna_msm", {}), base), "rna_msm") else 1

    rna_structure = cfg.get("rna_structure", {}).get("models", {})
    if rna_structure.get("rna_ernie", {}).get("enabled", False):
        failures += 0 if _guard(lambda: check_rna_ernie(rna_structure.get("rna_ernie", {}), base), "rna_ernie") else 1
    if rna_structure.get("rnabert", {}).get("enabled", False):
        failures += 0 if _guard(lambda: check_rnabert(rna_structure.get("rnabert", {}), base), "rnabert") else 1
    if rna_structure.get("rhofold", {}).get("enabled", False):
        failures += 0 if _guard(lambda: check_rhofold(rna_structure.get("rhofold", {}), base), "rhofold") else 1

    print(f"\nFailures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
