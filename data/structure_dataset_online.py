import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data.register import DataRegister
from data.structure_dataset import StructureDataset
from feature_extraction import extractors

R = DataRegister()


def _write_fasta(path: Path, records: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(f">{name}\n{seq}\n")


def _embedding_path(root: Path, base_dir: str, model: str, name: str) -> Path:
    return root / base_dir / model / f"{name}.pt"


@R.register("structure_dataset_online")
class StructureDatasetOnline(StructureDataset):
    """
    Online version of StructureDataset. It generates missing embeddings on the fly
    but keeps the existing precomputed embedding loading logic intact.
    """

    def __init__(
        self,
        *args,
        online_device: str = "cuda",
        online_batch_size: int = 1,
        esm2_weights: Optional[str] = None,
        prott5_weights: Optional[str] = None,
        saprot_weights: Optional[str] = None,
        rinalmo_weights: Optional[str] = None,
        rna_fm_weights: Optional[str] = None,
        rna_ernie_weights: Optional[str] = None,
        rna_ernie_vocab: Optional[str] = None,
        rnabert_weights: Optional[str] = None,
        rhofold_weights: Optional[str] = None,
        protein_mpnn_weights: Optional[str] = None,
        protbert_weights: Optional[str] = None,
        protrek_weights_dir: Optional[str] = None,
        protrek_foldseek_bin: Optional[str] = None,
        rna_msm_root_path: Optional[str] = None,
        rna_msm_msa_path: Optional[str] = None,
        rna_msm_model_path: Optional[str] = None,
        rna_msm_extra_overrides: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        kwargs["use_precomputed_embeddings"] = True
        super().__init__(*args, **kwargs)
        self.online_device = online_device
        self.online_batch_size = online_batch_size
        self.esm2_weights = esm2_weights
        self.prott5_weights = prott5_weights
        self.saprot_weights = saprot_weights
        self.rinalmo_weights = rinalmo_weights
        self.rna_fm_weights = rna_fm_weights
        self.rna_ernie_weights = rna_ernie_weights
        self.rna_ernie_vocab = rna_ernie_vocab
        self.rnabert_weights = rnabert_weights
        self.rhofold_weights = rhofold_weights
        self.protein_mpnn_weights = protein_mpnn_weights
        self.protbert_weights = protbert_weights
        self.protrek_weights_dir = protrek_weights_dir
        self.protrek_foldseek_bin = protrek_foldseek_bin
        self.rna_msm_root_path = rna_msm_root_path
        self.rna_msm_msa_path = rna_msm_msa_path
        self.rna_msm_model_path = rna_msm_model_path
        self.rna_msm_extra_overrides = rna_msm_extra_overrides or []

    def _ensure_sequence_embeddings(
        self,
        records: List[Tuple[str, str]],
        base_dir: str,
        model_name: str,
        extractor_fn,
        extractor_kwargs: Dict,
    ) -> None:
        emb_root = Path(self.embedding_root)
        missing = [
            (name, seq)
            for name, seq in records
            if not _embedding_path(emb_root, base_dir, model_name, name).exists()
        ]
        if not missing:
            return
        tmp_fasta = emb_root / "inputs" / f"._tmp_{model_name}_{os.getpid()}.fasta"
        _write_fasta(tmp_fasta, missing)
        kwargs = {k: v for k, v in extractor_kwargs.items() if v is not None}
        extractor_fn(str(tmp_fasta), str(emb_root / base_dir / model_name), **kwargs)
        tmp_fasta.unlink(missing_ok=True)

    def _ensure_structure_embeddings(
        self,
        pdb_path: Path,
        base_dir: str,
        model_name: str,
        extractor_fn,
        extractor_kwargs: Dict,
    ) -> None:
        emb_root = Path(self.embedding_root)
        out_dir = emb_root / base_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = emb_root / "._online_tmp" / model_name / pdb_path.stem
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_pdb = tmp_dir / pdb_path.name
        if not tmp_pdb.exists():
            shutil.copy2(pdb_path, tmp_pdb)
        try:
            kwargs = {k: v for k, v in extractor_kwargs.items() if v is not None}
            extractor_fn(str(tmp_dir), str(out_dir), **kwargs)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _ensure_rna_msm_embeddings(self, names: List[str]) -> None:
        if not names:
            return
        if not (self.rna_msm_root_path and self.rna_msm_msa_path and self.rna_msm_model_path):
            raise RuntimeError("RNA-MSM requires rna_msm_root_path, rna_msm_msa_path, and rna_msm_model_path.")
        emb_root = Path(self.embedding_root)
        msa_list = emb_root / "inputs" / f"._tmp_rna_msm_ids_{os.getpid()}.txt"
        msa_list.parent.mkdir(parents=True, exist_ok=True)
        msa_list.write_text("\n".join(names) + "\n", encoding="utf-8")
        try:
            extractors.extract_rna_msm(
                root_path=self.rna_msm_root_path,
                msa_path=self.rna_msm_msa_path,
                msa_list=str(msa_list),
                model_path=self.rna_msm_model_path,
                output_dir=str(emb_root / "rna_sequence" / "rna_msm"),
                device=self.online_device,
                extra_overrides=list(self.rna_msm_extra_overrides),
            )
        finally:
            msa_list.unlink(missing_ok=True)

    def _ensure_embeddings_for_item(self, item: Dict) -> None:
        emb_root = Path(self.embedding_root)
        raw_complex_id = item.get("complex", item.get("id", "unknown"))
        complex_id = extractors.safe_name(raw_complex_id)
        prot_chains = item.get("prot_chain_ids", [])
        rna_chains = item.get("rna_chain_ids", [])
        prot_seqs = item.get("prot_seqs", [])
        rna_seqs = item.get("rna_seqs", [])

        protein_records = []
        for chain_id, seq in zip(prot_chains, prot_seqs):
            name = extractors.safe_name(f"{complex_id}_prot_{chain_id}")
            protein_records.append((name, seq))
            fasta_path = emb_root / "inputs" / "protein_single" / f"{name}.fasta"
            if not fasta_path.exists():
                _write_fasta(fasta_path, [(name, seq)])

        rna_records = []
        for chain_id, seq in zip(rna_chains, rna_seqs):
            name = extractors.safe_name(f"{complex_id}_rna_{chain_id}")
            rna_records.append((name, seq))
            fasta_path = emb_root / "inputs" / "rna_single" / f"{name}.fasta"
            if not fasta_path.exists():
                _write_fasta(fasta_path, [(name, seq)])

        seq_prot_extractors = {
            "esm2": (extractors.extract_esm2, {"device": self.online_device, "model_location": self.esm2_weights, "batch_size": self.online_batch_size}),
            "prott5": (extractors.extract_prott5, {"device": self.online_device, "model_dir": self.prott5_weights, "batch_size": self.online_batch_size}),
            "saprot": (extractors.extract_saprot, {"device": self.online_device, "model_dir": self.saprot_weights, "batch_size": self.online_batch_size}),
        }
        seq_rna_extractors = {
            "rinalmo": (extractors.extract_rinalmo, {"device": self.online_device, "model_weights": self.rinalmo_weights, "batch_size": self.online_batch_size}),
            "rna_fm": (extractors.extract_rna_fm, {"device": self.online_device, "model_path": self.rna_fm_weights, "batch_size": self.online_batch_size}),
        }
        str_prot_extractors = {
            "esm_if1": (extractors.extract_esm_if1, {"device": self.online_device}),
            "protbert": (
                extractors.extract_protbert,
                {"device": self.online_device, "model_dir": self.protbert_weights, "batch_size": self.online_batch_size},
            ),
            "protrek": (
                extractors.extract_protrek,
                {
                    "device": self.online_device,
                    "model_dir": self.protrek_weights_dir,
                    "foldseek_bin": self.protrek_foldseek_bin,
                },
            ),
        }
        str_rna_extractors = {
            "rna_ernie": (
                extractors.extract_rna_ernie,
                {
                    "device": self.online_device,
                    "model_dir": self.rna_ernie_weights,
                    "vocab_path": self.rna_ernie_vocab,
                },
            ),
            "rnabert": (
                extractors.extract_rnabert,
                {"model_weights": self.rnabert_weights, "batch_size": self.online_batch_size},
            ),
            "rhofold": (
                extractors.extract_rhofold,
                {"device": self.online_device, "ckpt_path": self.rhofold_weights},
            ),
        }

        for model_name in self.seq_prot_models:
            if model_name not in seq_prot_extractors:
                raise KeyError(f"Unknown protein sequence model: {model_name}")
            extractor_fn, extractor_kwargs = seq_prot_extractors[model_name]
            self._ensure_sequence_embeddings(
                protein_records, "protein_sequence", model_name, extractor_fn, extractor_kwargs
            )

        for model_name in self.seq_rna_models:
            if model_name == "rna_msm":
                missing = [
                    name
                    for name, _ in rna_records
                    if not _embedding_path(emb_root, "rna_sequence", "rna_msm", name).exists()
                ]
                self._ensure_rna_msm_embeddings(missing)
                continue
            if model_name not in seq_rna_extractors:
                raise KeyError(f"Unknown RNA sequence model: {model_name}")
            extractor_fn, extractor_kwargs = seq_rna_extractors[model_name]
            self._ensure_sequence_embeddings(
                rna_records, "rna_sequence", model_name, extractor_fn, extractor_kwargs
            )

        pdb_path = Path(self.data_root) / f"{raw_complex_id}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"Missing PDB file: {pdb_path}")
        for model_name in self.str_prot_models:
            if model_name not in str_prot_extractors:
                raise KeyError(f"Unknown protein structure model: {model_name}")
            extractor_fn, extractor_kwargs = str_prot_extractors[model_name]
            need_any = False
            for chain_id in prot_chains:
                name = extractors.safe_name(f"{complex_id}_prot_{chain_id}")
                if not _embedding_path(emb_root, "protein_structure", model_name, name).exists():
                    need_any = True
                    break
            if need_any:
                if model_name == "protbert":
                    self._ensure_sequence_embeddings(
                        protein_records, "protein_structure", model_name, extractor_fn, extractor_kwargs
                    )
                else:
                    self._ensure_structure_embeddings(
                        pdb_path, "protein_structure", model_name, extractor_fn, extractor_kwargs
                    )

        for model_name in self.str_rna_models:
            if model_name not in str_rna_extractors:
                raise KeyError(f"Unknown RNA structure model: {model_name}")
            extractor_fn, extractor_kwargs = str_rna_extractors[model_name]
            self._ensure_sequence_embeddings(
                rna_records, "rna_structure", model_name, extractor_fn, extractor_kwargs
            )

    def __getitem__(self, idx):
        data = self.data[idx]
        self._ensure_embeddings_for_item(data)
        if self.transform is not None:
            data = self.transform(data)
        return data
