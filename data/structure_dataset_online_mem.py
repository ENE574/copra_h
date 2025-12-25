import os
import shutil
import tempfile
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data._utils.collate import default_collate

from data.register import DataRegister
from data.structure_dataset import CustomStructCollate, StructureDataset, na_alphabet_config, safe_name
from feature_extraction import extractors

R = DataRegister()


def _safe_chain(chain_id: str) -> str:
    return extractors.safe_name(chain_id or "X")


@R.register("structure_dataset_online_mem")
class StructureDatasetOnlineMem(StructureDataset):
    """
    Online dataset that extracts embeddings on-the-fly and keeps them in memory
    (no persistent embedding cache on disk).
    """

    def __init__(
        self,
        *args,
        online_device: str = "cuda",
        online_batch_size: int = 1,
        online_cache: bool = True,
        esm2_weights: Optional[str] = None,
        prott5_weights: Optional[str] = None,
        saprot_weights: Optional[str] = None,
        rinalmo_weights: Optional[str] = None,
        rinalmo_type: str = "650M",
        rna_fm_weights: Optional[str] = None,
        rna_ernie_weights: Optional[str] = None,
        rna_ernie_vocab: Optional[str] = None,
        rnabert_weights: Optional[str] = None,
        rhofold_weights: Optional[str] = None,
        protein_mpnn_weights: Optional[str] = None,
        protrek_weights_dir: Optional[str] = None,
        protrek_foldseek_bin: Optional[str] = None,
        rna_msm_root_path: Optional[str] = None,
        rna_msm_msa_path: Optional[str] = None,
        rna_msm_model_path: Optional[str] = None,
        rna_msm_extra_overrides: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        kwargs["use_precomputed_embeddings"] = False
        super().__init__(*args, **kwargs)
        self.online_device = online_device
        if self.online_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("online_device=cuda but CUDA is not available.")
        self.online_batch_size = online_batch_size
        self.online_cache = online_cache
        self.esm2_weights = esm2_weights
        self.prott5_weights = prott5_weights
        self.saprot_weights = saprot_weights
        self.rinalmo_weights = rinalmo_weights
        self.rinalmo_type = rinalmo_type
        self.rna_fm_weights = rna_fm_weights
        self.rna_ernie_weights = rna_ernie_weights
        self.rna_ernie_vocab = rna_ernie_vocab
        self.rnabert_weights = rnabert_weights
        self.rhofold_weights = rhofold_weights
        self.protein_mpnn_weights = protein_mpnn_weights
        self.protrek_weights_dir = protrek_weights_dir
        self.protrek_foldseek_bin = protrek_foldseek_bin
        self.rna_msm_root_path = rna_msm_root_path
        self.rna_msm_msa_path = rna_msm_msa_path
        self.rna_msm_model_path = rna_msm_model_path
        self.rna_msm_extra_overrides = rna_msm_extra_overrides or []
        self._model_cache: Dict[str, object] = {}
        self._emb_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def _cache_get(self, model: str, name: str) -> Optional[torch.Tensor]:
        if not self.online_cache:
            return None
        return self._emb_cache.get(model, {}).get(name)

    def _cache_set(self, model: str, name: str, tensor: torch.Tensor) -> None:
        if not self.online_cache:
            return
        self._emb_cache.setdefault(model, {})[name] = tensor

    def _get_esm2(self):
        if "esm2" in self._model_cache:
            return self._model_cache["esm2"]
        import esm

        if self.esm2_weights:
            model_path = Path(self.esm2_weights)
            regression_path = model_path.with_suffix("")
            regression_path = regression_path.with_name(regression_path.name + "-contact-regression.pt")
            if regression_path.exists():
                model, alphabet = esm.pretrained.load_model_and_alphabet(self.esm2_weights)
            else:
                model_data = torch.load(str(model_path), map_location="cpu")
                model, alphabet = esm.pretrained.load_model_and_alphabet_core(
                    model_path.stem, model_data, regression_data=None
                )
        else:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval().to(self.online_device)
        self._model_cache["esm2"] = (model, alphabet)
        return model, alphabet

    def _get_prott5(self):
        if "prott5" in self._model_cache:
            return self._model_cache["prott5"]
        from transformers import T5EncoderModel, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained(self.prott5_weights, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(self.prott5_weights)
        model.eval().to(self.online_device)
        self._model_cache["prott5"] = (model, tokenizer)
        return model, tokenizer

    def _get_saprot(self):
        if "saprot" in self._model_cache:
            return self._model_cache["saprot"]
        from transformers import EsmForMaskedLM, EsmTokenizer

        tokenizer = EsmTokenizer.from_pretrained(self.saprot_weights)
        model = EsmForMaskedLM.from_pretrained(self.saprot_weights)
        model.eval().to(self.online_device)
        self._model_cache["saprot"] = (model, tokenizer)
        return model, tokenizer

    def _get_rinalmo(self):
        if "rinalmo" in self._model_cache:
            return self._model_cache["rinalmo"]
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        sys.path.append(str(repo_root / "RiNALMo"))
        from rinalmo.data.alphabet import Alphabet
        from rinalmo.data.constants import CLS_TKN, EOS_TKN, MASK_TKN, PAD_TKN, RNA_TOKENS, UNK_TKN
        from rinalmo.model.model import RiNALMo
        from rinalmo.config import model_config

        alphabet = Alphabet(
            standard_tkns=RNA_TOKENS,
            special_tkns=[CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
        )
        size = "giga" if self.rinalmo_type == "650M" else "mega"
        config = model_config(size)
        model = RiNALMo(config)
        model.load_state_dict(torch.load(self.rinalmo_weights, map_location="cpu"))
        model.eval().to(self.online_device)
        if self.online_device.startswith("cuda"):
            model = model.half()
        self._model_cache["rinalmo"] = (model, alphabet)
        return model, alphabet

    def _get_rna_fm(self):
        if "rna_fm" in self._model_cache:
            return self._model_cache["rna_fm"]
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        sys.path.append(str(repo_root / "RNA-FM"))
        import fm

        model, alphabet = fm.pretrained.rna_fm_t12(model_location=self.rna_fm_weights)
        model.eval().to(self.online_device)
        self._model_cache["rna_fm"] = (model, alphabet)
        return model, alphabet

    def _get_rna_ernie(self):
        if "rna_ernie" in self._model_cache:
            return self._model_cache["rna_ernie"]
        if self.online_device == "cpu":
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        else:
            self._ensure_paddle_cuda_env()
        import paddle
        from paddlenlp.transformers import ErnieModel

        repo_root = Path(__file__).resolve().parents[1]
        import sys

        sys.path.append(str(repo_root / "RNAErnie"))
        from rna_ernie import BatchConverter

        paddle_device = self.online_device.replace("cuda", "gpu", 1)
        paddle.set_device(paddle_device)
        batch_converter = BatchConverter(
            k_mer=1,
            vocab_path=str(repo_root / (self.rna_ernie_vocab or "RNAErnie/data/vocab/vocab_1MER.txt")),
            batch_size=256,
            max_seq_len=512,
        )
        model = ErnieModel.from_pretrained(str(repo_root / (self.rna_ernie_weights or "weights/RNAErnie_weights")))
        model.eval()
        self._model_cache["rna_ernie"] = (model, batch_converter)
        return model, batch_converter

    def _ensure_paddle_cuda_env(self) -> None:
        if not self.online_device.startswith("cuda"):
            return
        ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
        candidates = []
        env_root = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
        search_dirs = [
            env_root / "lib",
            env_root / "lib64",
            env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cublas" / "lib",
            env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cudnn" / "lib",
        ]
        for path in search_dirs:
            if not path.is_dir():
                continue
            if path in candidates:
                continue
            if any(path.glob("libcudnn.so*")) or any(path.glob("libcublas.so*")):
                candidates.append(str(path))
        new_paths = [p for p in candidates if p not in ld_paths]
        if new_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths + ld_paths)

    def _get_esm_if1(self):
        if "esm_if1" in self._model_cache:
            return self._model_cache["esm_if1"]
        import esm

        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model.eval().to(self.online_device)
        self._model_cache["esm_if1"] = (model, alphabet)
        return model, alphabet

    def _get_protein_mpnn(self):
        if "proteinmpnn" in self._model_cache:
            return self._model_cache["proteinmpnn"]
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        sys.path.append(str(repo_root / "ProteinMPNN"))
        import protein_mpnn_utils as mpnn_utils

        checkpoint = torch.load(self.protein_mpnn_weights, map_location="cpu")
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
            ca_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval().to(self.online_device)
        self._model_cache["proteinmpnn"] = (model, mpnn_utils)
        return model, mpnn_utils

    def _get_protrek(self):
        if "protrek" in self._model_cache:
            return self._model_cache["protrek"]
        repo_root = Path(__file__).resolve().parents[1]
        import sys

        sys.path.insert(0, str(repo_root))
        sys.modules.pop("utils", None)
        from ProTrek.model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

        model_dir = Path(self.protrek_weights_dir)
        protein_config = str(model_dir / "esm2_t33_650M_UR50D")
        text_config = str(model_dir / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        structure_config = str(model_dir / "foldseek_t30_150M")
        from_checkpoint = str(model_dir / "ProTrek_650M.pt")
        model = ProTrekTrimodalModel(
            protein_config=protein_config,
            text_config=text_config,
            structure_config=structure_config,
            from_checkpoint=from_checkpoint,
            init_metrics=False,
        ).eval()
        model = model.to(self.online_device)
        self._model_cache["protrek"] = model
        return model

    def _extract_seq_esm2(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        model, alphabet = self._get_esm2()
        batch_converter = alphabet.get_batch_converter()
        outputs = {}
        with torch.no_grad():
            for batch in extractors.chunked(records, self.online_batch_size):
                labels, seqs = zip(*batch)
                _, _, tokens = batch_converter(batch)
                tokens = tokens.to(self.online_device)
                results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
                token_reps = results["representations"][model.num_layers]
                for i, label in enumerate(labels):
                    seq_len = len(seqs[i])
                    residue_rep = token_reps[i, 1 : seq_len + 1].detach().cpu()
                    outputs[label] = residue_rep
        return outputs

    def _extract_seq_prott5(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        model, tokenizer = self._get_prott5()
        outputs = {}
        with torch.no_grad():
            for batch in extractors.chunked(records, self.online_batch_size):
                labels, seqs = zip(*batch)
                seqs = [extractors._sanitize_protein_sequence(s) for s in seqs]
                enc = tokenizer.batch_encode_plus(
                    seqs,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.online_device)
                attention_mask = enc["attention_mask"].to(self.online_device)
                embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                for i, label in enumerate(labels):
                    valid_len = int(attention_mask[i].sum().item())
                    residue_rep = embeddings[i, : valid_len - 1].detach().cpu()
                    outputs[label] = residue_rep
        return outputs

    def _extract_seq_saprot(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        model, tokenizer = self._get_saprot()
        vocab = tokenizer.get_vocab()
        cls_id = tokenizer.cls_token_id
        eos_id = tokenizer.eos_token_id
        unk_id = tokenizer.unk_token_id
        max_len = tokenizer.model_max_length if tokenizer.model_max_length is not None else 2048
        outputs = {}
        with torch.no_grad():
            for label, seq in records:
                seq_clean = seq.strip().upper()
                token_ids = [cls_id] + [vocab.get(ch, unk_id) for ch in seq_clean] + [eos_id]
                if len(token_ids) > max_len:
                    token_ids = token_ids[: max_len - 1] + [eos_id]
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.online_device)
                attention_mask = torch.ones_like(input_ids, device=self.online_device)
                outputs_model = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden = outputs_model.hidden_states[-1]
                avail = hidden.shape[1] - 2
                take = min(len(seq_clean), avail)
                residue_rep = hidden[0, 1 : 1 + take].detach().cpu()
                outputs[label] = residue_rep
        return outputs

    def _extract_seq_rinalmo(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        model, alphabet = self._get_rinalmo()
        outputs = {}
        with torch.no_grad():
            for batch in extractors.chunked(records, self.online_batch_size):
                labels, seqs = zip(*batch)
                tokens = torch.tensor(alphabet.batch_tokenize(list(seqs)), dtype=torch.long).to(self.online_device)
                if self.online_device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        rep = model(tokens)["representation"]
                else:
                    rep = model(tokens)["representation"]
                for i, label in enumerate(labels):
                    seq_len = len(seqs[i])
                    residue_rep = rep[i, 1 : seq_len + 1].detach().cpu()
                    outputs[label] = residue_rep
        return outputs

    def _extract_seq_rna_fm(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        model, alphabet = self._get_rna_fm()
        batch_converter = alphabet.get_batch_converter()
        outputs = {}
        with torch.no_grad():
            for batch in extractors.chunked(records, self.online_batch_size):
                labels, seqs = zip(*batch)
                _, _, tokens = batch_converter(batch)
                tokens = tokens.to(self.online_device)
                results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
                token_reps = results["representations"][model.num_layers]
                for i, label in enumerate(labels):
                    seq_len = len(seqs[i])
                    residue_rep = token_reps[i, 1 : seq_len + 1].detach().cpu()
                    outputs[label] = residue_rep
        return outputs

    def _extract_seq_rna_ernie(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        model, batch_converter = self._get_rna_ernie()
        outputs = {}
        import paddle

        with paddle.no_grad():
            for names, seqs, inputs_ids in batch_converter(records):
                embeddings = model(inputs_ids)[0].detach().numpy()
                for i, name in enumerate(names):
                    seq_len = len(seqs[i])
                    token_embeddings = torch.tensor(embeddings[i, 1 : 1 + seq_len])
                    outputs[name] = token_embeddings
        return outputs

    def _extract_struct_esm_if1(self, pdb_path: Path, chain_id: str) -> torch.Tensor:
        import esm
        from esm.inverse_folding import util as if_util
        from biotite.sequence.seqtypes import ProteinSequence

        model, alphabet = self._get_esm_if1()
        try:
            coords, _ = if_util.load_coords(str(pdb_path), chain_id)
        except KeyError:
            std_map = ProteinSequence._dict_3to1
            mapping = extractors.NONSTANDARD_RESIDUE_MAP
            with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.readlines()
            new_lines = []
            for line in lines:
                if line.startswith(("ATOM  ", "HETATM", "LINK  ")) and len(line) >= 20:
                    resname = line[17:20]
                    if resname not in std_map:
                        mapped = mapping.get(resname, "GLY")
                        line = line[:17] + mapped + line[20:]
                new_lines.append(line)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp_handle:
                tmp_handle.write("".join(new_lines))
                tmp_path = Path(tmp_handle.name)
            try:
                coords, _ = if_util.load_coords(str(tmp_path), chain_id)
            finally:
                tmp_path.unlink(missing_ok=True)
        batch_converter = if_util.CoordBatchConverter(alphabet)
        coords_tensor, confidence, _, _, padding_mask = batch_converter(
            [(coords, None, None)], device=self.online_device
        )
        encoder_out = model.encoder.forward(coords_tensor, padding_mask, confidence, return_all_hiddens=False)
        rep = encoder_out["encoder_out"][0][1:-1, 0].detach().cpu()
        return rep

    def _extract_struct_protein_mpnn(self, pdb_path: Path, chain_id: str) -> torch.Tensor:
        model, mpnn_utils = self._get_protein_mpnn()
        pdb_dict_list = mpnn_utils.parse_PDB(str(pdb_path), input_chain_list=[chain_id], ca_only=False)
        if not pdb_dict_list:
            return torch.empty((0, 1))
        batch = [pdb_dict_list[0]]
        X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = mpnn_utils.tied_featurize(
            batch,
            device=self.online_device,
            chain_dict=None,
            fixed_position_dict=None,
            omit_AA_dict=None,
            tied_positions_dict=None,
            pssm_dict=None,
            bias_by_res_dict=None,
            ca_only=False,
        )
        X = X.to(self.online_device)
        mask = mask.to(self.online_device)
        residue_idx = residue_idx.to(self.online_device)
        chain_encoding_all = chain_encoding_all.to(self.online_device)
        E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = model.W_e(E)
        mask_attend = mpnn_utils.gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        true_len = int(lengths[0].item())
        rep = h_V[0, :true_len].detach().cpu()
        return rep

    def _extract_struct_protrek(self, pdb_path: Path, chain_id: str) -> torch.Tensor:
        from ProTrek.utils.foldseek_util import get_struc_seq

        model = self._get_protrek()
        tmp_dir = Path(tempfile.mkdtemp(prefix="protrek_"))
        cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            seq_dict = get_struc_seq(str(self.protrek_foldseek_bin), str(pdb_path), chains=[chain_id])
        finally:
            os.chdir(cwd)
        if not seq_dict or chain_id not in seq_dict:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return torch.empty((0, 1))
        foldseek_seq = seq_dict[chain_id][1].lower()
        reps = model.get_structure_repr([foldseek_seq], batch_size=1)
        rep = reps.detach().cpu()[0][: len(foldseek_seq)]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return rep

    def _extract_struct_rna_msm(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        if not (self.rna_msm_root_path and self.rna_msm_msa_path and self.rna_msm_model_path):
            raise RuntimeError("RNA-MSM requires rna_msm_root_path, rna_msm_msa_path, and rna_msm_model_path.")
        if not records:
            return {}
        tmp_dir = Path(tempfile.mkdtemp(prefix="rna_msm_"))
        msa_list = tmp_dir / "ids.txt"
        ids = []
        id_map: Dict[str, str] = {}
        for idx, (name, seq) in enumerate(records):
            msa_id = f"rna_{idx}"
            ids.append(msa_id)
            id_map[msa_id] = name
            msa_path = tmp_dir / f"{msa_id}.a2m_msa2"
            with open(msa_path, "w", encoding="utf-8") as handle:
                handle.write(f">{msa_id}\n{seq}\n")
        if not ids:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {}
        file_stems = sorted({p.stem.split(".")[0] for p in tmp_dir.glob("*.a2m_msa2")})
        if not file_stems:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {}
        msa_list.write_text("\n".join(file_stems) + "\n", encoding="utf-8")
        extractors.extract_rna_msm(
            root_path=str(tmp_dir),
            msa_path=".",
            msa_list="ids.txt",
            model_path=self.rna_msm_model_path,
            output_dir=str(tmp_dir),
            device=self.online_device,
            extra_overrides=list(self.rna_msm_extra_overrides),
        )
        outputs = {}
        for path in tmp_dir.glob("*.pt"):
            payload = torch.load(path, map_location="cpu")
            orig = id_map.get(path.stem, path.stem)
            outputs[orig] = payload["token_embeddings"]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return outputs

    def _extract_struct_rnabert(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="rnabert_"))
        fasta_path = tmp_dir / "input.fasta"
        with open(fasta_path, "w", encoding="utf-8") as handle:
            for name, seq in records:
                seq_clean = seq.upper().replace("T", "U")
                allowed = set("ACGU")
                seq_clean = "".join(ch if ch in allowed else "A" for ch in seq_clean)
                handle.write(f">{name}\n{seq_clean}\n")
        extractors.extract_rnabert(
            fasta_path=str(fasta_path),
            output_dir=str(tmp_dir),
            model_weights=self.rnabert_weights,
            batch_size=self.online_batch_size,
        )
        outputs = {}
        for path in tmp_dir.glob("*.pt"):
            payload = torch.load(path, map_location="cpu")
            outputs[path.stem] = payload["token_embeddings"]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return outputs

    def _extract_struct_rhofold(self, records: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
        tmp_dir = Path(tempfile.mkdtemp(prefix="rhofold_"))
        outputs = {}
        for name, seq in records:
            fasta_path = tmp_dir / f"{name}.fasta"
            with open(fasta_path, "w", encoding="utf-8") as handle:
                handle.write(f">{name}\n{seq}\n")
            extractors.extract_rhofold(
                fasta_path=str(fasta_path),
                output_dir=str(tmp_dir),
                device=self.online_device,
                ckpt_path=self.rhofold_weights,
                input_a3m=None,
                single_seq_pred=True,
            )
            out_path = tmp_dir / f"{name}.pt"
            if out_path.exists():
                payload = torch.load(out_path, map_location="cpu")
                outputs[name] = payload["token_embeddings"]
                out_path.unlink(missing_ok=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return outputs

    def _extract_embeddings_for_item(self, item: Dict) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        complex_id = extractors.safe_name(item.get("complex", item.get("id", "unknown")))
        prot_chains = item.get("prot_chain_ids", [])
        rna_chains = item.get("rna_chain_ids", [])
        prot_seqs = item.get("prot_seqs", [])
        rna_seqs = item.get("rna_seqs", [])
        pdb_path = Path(self.data_root) / f"{item.get('complex', item.get('id'))}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"Missing PDB file: {pdb_path}")

        prot_records = [(extractors.safe_name(f"{complex_id}_prot_{cid}"), seq) for cid, seq in zip(prot_chains, prot_seqs)]
        rna_records = [(extractors.safe_name(f"{complex_id}_rna_{cid}"), seq) for cid, seq in zip(rna_chains, rna_seqs)]

        seq_prot = {name: [] for name in self.seq_prot_models}
        seq_rna = {name: [] for name in self.seq_rna_models}
        str_prot = {name: [] for name in self.str_prot_models}
        str_rna = {name: [] for name in self.str_rna_models}

        for model_name in self.seq_prot_models:
            if model_name == "esm2":
                outputs = self._extract_seq_esm2(prot_records)
            elif model_name == "prott5":
                outputs = self._extract_seq_prott5(prot_records)
            elif model_name == "saprot":
                outputs = self._extract_seq_saprot(prot_records)
            else:
                raise KeyError(f"Unknown protein sequence model: {model_name}")
            for name, _ in prot_records:
                emb = outputs[name]
                self._cache_set(model_name, name, emb)
                seq_prot[model_name].append(emb)

        for model_name in self.seq_rna_models:
            if model_name == "rinalmo":
                outputs = self._extract_seq_rinalmo(rna_records)
            elif model_name == "rna_fm":
                outputs = self._extract_seq_rna_fm(rna_records)
            elif model_name == "rna_msm":
                outputs = self._extract_struct_rna_msm(rna_records)
            else:
                raise KeyError(f"Unknown RNA sequence model: {model_name}")
            for name, _ in rna_records:
                emb = outputs[name]
                self._cache_set(model_name, name, emb)
                seq_rna[model_name].append(emb)

        for model_name in self.str_prot_models:
            for cid, seq in zip(prot_chains, prot_seqs):
                name = extractors.safe_name(f"{complex_id}_prot_{cid}")
                cached = self._cache_get(model_name, name)
                if cached is not None:
                    str_prot[model_name].append(cached)
                    continue
                if model_name == "esm_if1":
                    emb = self._extract_struct_esm_if1(pdb_path, cid)
                elif model_name == "proteinmpnn":
                    emb = self._extract_struct_protein_mpnn(pdb_path, cid)
                elif model_name == "protrek":
                    emb = self._extract_struct_protrek(pdb_path, cid)
                else:
                    raise KeyError(f"Unknown protein structure model: {model_name}")
                self._cache_set(model_name, name, emb)
                str_prot[model_name].append(emb)

        for model_name in self.str_rna_models:
            if model_name == "rna_ernie":
                outputs = self._extract_seq_rna_ernie(rna_records)
            elif model_name == "rnabert":
                outputs = self._extract_struct_rnabert(rna_records)
            elif model_name == "rhofold":
                outputs = self._extract_struct_rhofold(rna_records)
            else:
                raise KeyError(f"Unknown RNA structure model: {model_name}")
            for name, _ in rna_records:
                emb = outputs[name]
                self._cache_set(model_name, name, emb)
                str_rna[model_name].append(emb)

        return {
            "seq_prot_embeddings": seq_prot,
            "seq_rna_embeddings": seq_rna,
            "str_prot_embeddings": str_prot,
            "str_rna_embeddings": str_rna,
        }

    def __getitem__(self, idx):
        data = self.data[idx]
        emb = self._extract_embeddings_for_item(data)
        data["seq_prot_embeddings_mem"] = emb["seq_prot_embeddings"]
        data["seq_rna_embeddings_mem"] = emb["seq_rna_embeddings"]
        data["str_prot_embeddings_mem"] = emb["str_prot_embeddings"]
        data["str_rna_embeddings_mem"] = emb["str_rna_embeddings"]
        if self.transform is not None:
            data = self.transform(data)
        return data


class CustomStructCollateOnlineMem(CustomStructCollate):
    def pad_for_berts(self, strategy, batch):
        import esm
        from rinalmo.data.alphabet import Alphabet

        prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        na_alphabet = Alphabet(**na_alphabet_config)
        mut_flag = 0
        prot_chains = [len(item['prot_seqs']) for item in batch]
        na_chains = [len(item['rna_seqs']) for item in batch]

        seq_prot_models = list(batch[0]["seq_prot_embeddings_mem"].keys())
        seq_rna_models = list(batch[0]["seq_rna_embeddings_mem"].keys())
        str_prot_models = list(batch[0]["str_prot_embeddings_mem"].keys())
        str_rna_models = list(batch[0]["str_rna_embeddings_mem"].keys())

        seq_prot_embeds = {name: [] for name in seq_prot_models}
        seq_rna_embeds = {name: [] for name in seq_rna_models}
        str_prot_embeds = {name: [] for name in str_prot_models}
        str_rna_embeds = {name: [] for name in str_rna_models}

        max_item_prot_length = [item['max_prot_length'] for item in batch]
        max_item_na_length = [item['max_na_length'] for item in batch]
        max_prot_length = max(max_item_prot_length)
        max_na_length = max(max_item_na_length)
        total_prot_chains = sum(prot_chains)
        total_na_chains = sum(na_chains)
        if self.eight:
            max_prot_length = math.ceil((max_prot_length + 2) / 8) * 8
            max_na_length = math.ceil((max_na_length + 2) / 8) * 8
        else:
            max_prot_length = max_prot_length + 2
            max_na_length = max_na_length + 2
        prot_batch = torch.empty([total_prot_chains, max_prot_length])
        prot_batch.fill_(prot_alphabet.padding_idx)
        if 'mut_seqs' in batch[0]:
            mut_flag = 1
            mut_batch = torch.empty([total_prot_chains, max_prot_length])
            mut_batch.fill_(prot_alphabet.padding_idx)
        na_batch = torch.empty([total_na_chains, max_na_length])
        na_batch.fill_(na_alphabet.pad_idx)
        curr_prot_idx = 0
        curr_na_idx = 0
        for item in batch:
            prot_seqs = item['prot_seqs']
            if 'mut_seqs' in item:
                mut_seqs = item['mut_seqs']
            na_seqs = item['rna_seqs']
            for i, prot_seq in enumerate(prot_seqs):
                prot_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                prot_seq_encode = prot_alphabet.encode(prot_seq)
                seq = torch.tensor(prot_seq_encode, dtype=torch.int64)
                prot_batch[curr_prot_idx, 1: len(prot_seq_encode)+1] = seq
                prot_batch[curr_prot_idx, len(prot_seq_encode)+1] = prot_alphabet.eos_idx
                if 'mut_seqs' in item:
                    mut_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                    mut_seq_encode = prot_alphabet.encode(mut_seqs[i])
                    seq_m = torch.tensor(mut_seq_encode, dtype=torch.int64)
                    mut_batch[curr_prot_idx, 1: len(mut_seq_encode)+1] = seq_m
                    mut_batch[curr_prot_idx, len(mut_seq_encode)+1] = prot_alphabet.eos_idx
                for model_name in seq_prot_models:
                    emb = item["seq_prot_embeddings_mem"][model_name][i]
                    seq_prot_embeds[model_name].append(self._pad_embedding(emb, len(prot_seq), max_prot_length))
                for model_name in str_prot_models:
                    emb = item["str_prot_embeddings_mem"][model_name][i]
                    str_prot_embeds[model_name].append(self._pad_embedding(emb, len(prot_seq), max_prot_length))
                curr_prot_idx += 1
            for j, na_seq in enumerate(na_seqs):
                na_seq_encode = na_alphabet.encode(na_seq)
                seq = torch.tensor(na_seq_encode, dtype=torch.int64)
                na_batch[curr_na_idx, :len(seq)] = seq
                for model_name in seq_rna_models:
                    emb = item["seq_rna_embeddings_mem"][model_name][j]
                    seq_rna_embeds[model_name].append(self._pad_embedding(emb, len(na_seq), max_na_length))
                for model_name in str_rna_models:
                    emb = item["str_rna_embeddings_mem"][model_name][j]
                    str_rna_embeds[model_name].append(self._pad_embedding(emb, len(na_seq), max_na_length))
                curr_na_idx += 1

        prot_mask = torch.zeros_like(prot_batch)
        na_mask = torch.zeros_like(na_batch)
        prot_mask[(prot_batch!=prot_alphabet.padding_idx) & (prot_batch!=prot_alphabet.eos_idx) & (prot_batch!=prot_alphabet.cls_idx)] = 1
        na_mask[(na_batch!=na_alphabet.pad_idx) & (na_batch!=na_alphabet.eos_idx) & (na_batch!=na_alphabet.cls_idx)] = 1
        seq_prot_batch = {name: torch.stack(seq_prot_embeds[name], dim=0) for name in seq_prot_models}
        seq_rna_batch = {name: torch.stack(seq_rna_embeds[name], dim=0) for name in seq_rna_models}
        str_prot_batch = {name: torch.stack(str_prot_embeds[name], dim=0) for name in str_prot_models}
        str_rna_batch = {name: torch.stack(str_rna_embeds[name], dim=0) for name in str_rna_models}
        if mut_flag:
            return (
                prot_batch.long(),
                mut_batch.long(),
                prot_chains,
                prot_mask,
                na_batch.long(),
                na_chains,
                na_mask,
                seq_prot_batch,
                seq_rna_batch,
                str_prot_batch,
                str_rna_batch,
            )
        return (
            prot_batch.long(),
            prot_chains,
            prot_mask,
            na_batch.long(),
            na_chains,
            na_mask,
            seq_prot_batch,
            seq_rna_batch,
            str_prot_batch,
            str_rna_batch,
        )

    def __call__(self, data_list):
        data_list_padded = self.collate_complex(data_list)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        if 'mut_seqs' in data_list[0]:
            (
                prot_batch,
                mut_batch,
                prot_chains,
                prot_mask,
                na_batch,
                na_chains,
                na_mask,
                seq_prot_batch,
                seq_rna_batch,
                str_prot_batch,
                str_rna_batch,
            ) = self.pad_for_berts(self.strategy, data_list)
            batch['prot_mut'] = mut_batch
        else:
            (
                prot_batch,
                prot_chains,
                prot_mask,
                na_batch,
                na_chains,
                na_mask,
                seq_prot_batch,
                seq_rna_batch,
                str_prot_batch,
                str_rna_batch,
            ) = self.pad_for_berts(self.strategy, data_list)
        batch['prot'] = prot_batch
        batch['prot_chains'] = prot_chains
        batch['protein_mask'] = prot_mask
        batch['na'] = na_batch
        batch['na_chains'] = na_chains
        batch['na_mask'] = na_mask
        batch['seq_prot_embeddings'] = seq_prot_batch
        batch['seq_rna_embeddings'] = seq_rna_batch
        batch['str_prot_embeddings'] = str_prot_batch
        batch['str_rna_embeddings'] = str_rna_batch
        batch['use_precomputed_embeddings'] = True
        batch['strategy'] = self.strategy
        batch['labels'] = batch['labels'].float()
        return batch
