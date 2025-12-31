import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def read_fasta(path: str) -> List[Tuple[str, str]]:
    records = []
    name = None
    seq_chunks = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_chunks)))
                name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if name is not None:
        records.append((name, "".join(seq_chunks)))
    return records


def chunked(items: List, batch_size: int) -> Iterable[List]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_pdb_files(pdb_dir: str, pdb_list: Optional[Iterable[str]]) -> List[Path]:
    if pdb_list is not None:
        base_dir = Path(pdb_dir)
        files = []
        for item in pdb_list:
            p = Path(item)
            if not p.is_absolute():
                p = base_dir / item
            if p.exists() and p.suffix.lower() in {".pdb", ".cif"}:
                files.append(p)
        return sorted(set(files), key=lambda path: path.name)
    pdb_dir = Path(pdb_dir)
    return sorted([p for p in pdb_dir.iterdir() if p.suffix in {".pdb", ".cif"}])


def _pdb_base_id(pdb_path: Path) -> str:
    """Return the dataset-level PDB id (strip chain suffixes in filenames like 1ABC_A_B.cif)."""
    return pdb_path.stem.split("_", 1)[0]


def _split_filename_chains(pdb_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Parse {pdb}_{prot_chain}_{rna_chain}.(cif|pdb) style names."""
    parts = pdb_path.stem.split("_")
    if len(parts) < 3:
        return None, None
    prot_chain = "_".join(parts[1:-1]).strip()
    rna_chain = parts[-1].strip()
    if not prot_chain or not rna_chain:
        return None, None
    return prot_chain, rna_chain


def _preferred_protein_chain_ids(output_dir: Path, pdb_path: Path, chain_id: Optional[str]) -> List[str]:
    if chain_id:
        return [chain_id]
    prot_chain, _ = _split_filename_chains(pdb_path)
    if prot_chain:
        return [prot_chain]
    return _get_expected_chain_ids(output_dir, pdb_path) or get_protein_chain_ids(pdb_path)


def _find_chain_id_in_model(model, chain_id: str) -> Optional[str]:
    if chain_id in model:
        return chain_id
    target = chain_id.lower()
    matches = [chain.id for chain in model.get_chains() if chain.id.lower() == target]
    if len(matches) == 1:
        return matches[0]
    return None


def _chain_id_candidates(pdb_path: Path, chain_id: str) -> List[str]:
    if not chain_id:
        return []
    candidates = [chain_id]
    label_ids = get_protein_chain_ids(pdb_path)
    if chain_id not in label_ids:
        for cid in label_ids:
            if cid.lower() == chain_id.lower() and cid not in candidates:
                candidates.append(cid)
    if pdb_path.suffix.lower() == ".cif":
        auth_ids = get_protein_chain_ids(pdb_path, auth_chains=True)
        if chain_id not in auth_ids:
            for cid in auth_ids:
                if cid.lower() == chain_id.lower() and cid not in candidates:
                    candidates.append(cid)
    return candidates


def _get_model_chain(pdb_path: Path, chain_id: str):
    from Bio.PDB import MMCIFParser, PDBParser

    fallback = (None, None)
    if pdb_path.suffix.lower() == ".cif":
        for use_auth in (False, True):
            parser = MMCIFParser(QUIET=True, auth_chains=use_auth)
            structure = parser.get_structure(pdb_path.stem, str(pdb_path))
            model0 = next(structure.get_models(), None)
            if model0 is None:
                continue
            resolved = _find_chain_id_in_model(model0, chain_id)
            if resolved:
                chain = model0[resolved]
                if _chain_has_protein_residue(chain):
                    return model0, resolved
                if fallback == (None, None):
                    fallback = (model0, resolved)
    else:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        model0 = next(structure.get_models(), None)
        if model0 is None:
            return fallback
        resolved = _find_chain_id_in_model(model0, chain_id)
        if resolved:
            chain = model0[resolved]
            if _chain_has_protein_residue(chain):
                return model0, resolved
            if fallback == (None, None):
                fallback = (model0, resolved)
    return fallback


def _get_reference_length(output_dir: Path, pdb_path: Path, chain_id: str) -> Optional[int]:
    """Try to read reference fasta length for pdb+chain from inputs/protein_single."""
    inputs_dir = output_dir.parent.parent / "inputs" / "protein_single"
    base_id = _pdb_base_id(pdb_path)
    fasta_path = inputs_dir / f"{base_id}_prot_{chain_id}.fasta"
    if not fasta_path.exists():
        return None
    try:
        recs = read_fasta(str(fasta_path))
        if not recs:
            return None
        return len(recs[0][1])
    except Exception:
        return None


def safe_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)


OUTPUT_EMBED_DIM = 0


def set_output_embed_dim(dim: Optional[int]) -> None:
    global OUTPUT_EMBED_DIM
    if dim is None:
        OUTPUT_EMBED_DIM = 0
        return
    OUTPUT_EMBED_DIM = int(dim)


def _adjust_embedding_dim(emb: torch.Tensor) -> torch.Tensor:
    target = OUTPUT_EMBED_DIM
    if not target or target <= 0:
        return emb
    if emb.shape[-1] == target:
        return emb
    if emb.shape[-1] > target:
        return emb[..., :target]
    pad_shape = list(emb.shape[:-1]) + [target - emb.shape[-1]]
    pad = torch.zeros(pad_shape, dtype=emb.dtype, device=emb.device)
    return torch.cat([emb, pad], dim=-1)


def save_tensor_payload(path: Path, payload: Dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_EMBED_DIM and "token_embeddings" in payload:
        payload = dict(payload)
        token_embeddings = payload.get("token_embeddings")
        if isinstance(token_embeddings, torch.Tensor):
            payload["token_embeddings"] = _adjust_embedding_dim(token_embeddings)
        seq_embedding = payload.get("sequence_embedding")
        if isinstance(seq_embedding, torch.Tensor):
            payload["sequence_embedding"] = _adjust_embedding_dim(seq_embedding)
    torch.save(payload, path)


def _rewrite_embeddings_in_dir(output_dir: Path) -> None:
    for emb_path in output_dir.glob("*.pt"):
        payload = torch.load(emb_path, map_location="cpu")
        if not isinstance(payload, dict) or "token_embeddings" not in payload:
            continue
        save_tensor_payload(emb_path, payload)


def save_numpy_payload(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _device_from_string(device: str) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)




# Map common non-standard residues to nearest standard amino acids.
NONSTANDARD_RESIDUE_MAP = {
    "MSE": "MET",  # selenomethionine
    "SEC": "CYS",  # selenocysteine
    "PYL": "LYS",  # pyrrolysine
    "SEP": "SER",  # phosphoserine
    "TPO": "THR",  # phosphothreonine
    "PTR": "TYR",  # phosphotyrosine
    "CSD": "CYS",
    "CSO": "CYS",
    "CSX": "CYS",
    "CME": "CYS",
    "CMT": "CYS",
    "SAH": "MET",
    "HYP": "PRO",  # hydroxyproline
    "DPR": "PRO",  # D-proline
    "F2F": "PHE",
    "FME": "MET",
    "MLY": "LYS",
    "M3L": "LYS",
    "LLP": "LYS",
    "KCX": "LYS",
    "PCA": "GLU",  # pyroglutamate from Glu
    "5HP": "GLU",  # 5-hydroxyproline/glutamate-like
    "ASX": "ASP",  # Asp/Asn ambiguous
    "GLX": "GLU",  # Glu/Gln ambiguous
}


def _get_expected_chain_ids(output_dir: Path, pdb_path: Path) -> Optional[List[str]]:
    inputs_dir = output_dir.parent.parent / "inputs" / "protein_single"
    if not inputs_dir.exists():
        return None
    base_id = _pdb_base_id(pdb_path)
    chain_ids = []
    for fasta_path in inputs_dir.glob(f"{base_id}_prot_*.fasta"):
        stem = fasta_path.stem
        if not stem.startswith(f"{base_id}_prot_"):
            continue
        chain_ids.append(stem.split("_prot_", 1)[1])
    return sorted(set(chain_ids)) or None


def _residue_has_atoms(residue, atom_names: Iterable[str]) -> bool:
    return all(atom_name in residue for atom_name in atom_names)


def _chain_has_protein_residue(chain) -> bool:
    from Bio.PDB import Polypeptide

    for residue in chain:
        resname = residue.get_resname().strip().upper()
        if Polypeptide.is_aa(residue, standard=False):
            return True
        if resname in NONSTANDARD_RESIDUE_MAP:
            return True
        if _residue_has_atoms(residue, ["N", "CA", "C"]):
            return True
    return False


def _chain_backbone_stats(pdb_path: Path, chain_id: str) -> Tuple[int, int]:
    """Return (#res with N/CA/C, #res with CA) for the resolved chain."""
    model0, resolved_chain = _get_model_chain(pdb_path, chain_id)
    if model0 is None or not resolved_chain:
        return 0, 0
    with_ncac = 0
    with_ca = 0
    for residue in model0[resolved_chain]:
        if "CA" in residue:
            with_ca += 1
        if _residue_has_atoms(residue, ["N", "CA", "C"]):
            with_ncac += 1
    return with_ncac, with_ca


def _write_clean_chain_pdb(
    pdb_path: Path,
    chain_id: str,
    dest_path: Path,
    ca_only: bool = False,
) -> bool:
    from Bio.PDB import PDBIO, Polypeptide
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain

    model0, resolved_chain = _get_model_chain(pdb_path, chain_id)
    if model0 is None or not resolved_chain:
        return False
    required_atoms = ["CA"] if ca_only else ["N", "CA", "C"]
    new_struct = Structure("tmp")
    new_model = Model(0)
    new_chain = Chain("A")
    new_struct.add(new_model)
    new_model.add(new_chain)
    res_idx = 1
    for residue in model0[resolved_chain]:
        if not _residue_has_atoms(residue, required_atoms):
            continue
        resname = residue.get_resname().strip().upper()
        if Polypeptide.is_aa(residue, standard=True):
            mapped = resname
        elif resname in NONSTANDARD_RESIDUE_MAP:
            mapped = NONSTANDARD_RESIDUE_MAP[resname]
        else:
            mapped = "GLY"
        new_res = residue.copy()
        new_res.resname = mapped
        new_res.id = (" ", res_idx, " ")
        res_idx += 1
        new_chain.add(new_res)
    if res_idx == 1:
        return False
    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(dest_path))
    return True


def get_protein_chain_ids(pdb_path: Path, auth_chains: bool = False) -> List[str]:
    from Bio.PDB import MMCIFParser, PDBParser, Polypeptide

    parser = (
        MMCIFParser(QUIET=True, auth_chains=auth_chains)
        if pdb_path.suffix.lower() == ".cif"
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    chain_ids = []
    for model in structure:
        for chain in model:
            has_aa = False
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=False):
                    has_aa = True
                    break
                if residue.get_resname() in NONSTANDARD_RESIDUE_MAP:
                    has_aa = True
                    break
                if _residue_has_atoms(residue, ["N", "CA", "C"]):
                    has_aa = True
                    break
            if has_aa:
                chain_ids.append(chain.id)
    return sorted(set(chain_ids))


def extract_esm2(
    fasta_path: str,
    output_dir: str,
    device: str = "cuda",
    model_location: Optional[str] = None,
    repr_layer: Optional[int] = None,
    batch_size: int = 1,
) -> None:
    import esm

    if model_location:
        model_path = Path(model_location)
        regression_path = model_path.with_suffix("")
        regression_path = regression_path.with_name(regression_path.name + "-contact-regression.pt")
        if regression_path.exists():
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
        else:
            model_data = torch.load(str(model_path), map_location="cpu")
            model_name = model_path.stem
            model, alphabet = esm.pretrained.load_model_and_alphabet_core(
                model_name, model_data, regression_data=None
            )
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)
    if repr_layer is None:
        repr_layer = model.num_layers

    batch_converter = alphabet.get_batch_converter()
    records = read_fasta(fasta_path)
    output_dir = ensure_dir(output_dir)

    with torch.no_grad():
        for batch in chunked(records, batch_size):
            labels, seqs = zip(*batch)
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device_obj)
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            token_reps = results["representations"][repr_layer]
            for i, label in enumerate(labels):
                seq_len = len(seqs[i])
                residue_rep = token_reps[i, 1 : seq_len + 1].detach().cpu()
                seq_rep = residue_rep.mean(0)
                payload = {
                    "token_embeddings": residue_rep,
                    "sequence_embedding": seq_rep,
                }
                save_tensor_payload(output_dir / f"{safe_name(label)}.pt", payload)


def _sanitize_protein_sequence(seq: str) -> str:
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    return " ".join(list(seq))


def extract_prott5(
    fasta_path: str,
    output_dir: str,
    device: str = "cuda",
    model_dir: str = "weights/ProtT5_weights",
    batch_size: int = 1,
) -> None:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_dir)
    model.eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)

    records = read_fasta(fasta_path)
    output_dir = ensure_dir(output_dir)

    with torch.no_grad():
        for batch in chunked(records, batch_size):
            labels, seqs = zip(*batch)
            seqs = [_sanitize_protein_sequence(s) for s in seqs]
            enc = tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device_obj)
            attention_mask = enc["attention_mask"].to(device_obj)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            for i, label in enumerate(labels):
                valid_len = int(attention_mask[i].sum().item())
                residue_rep = embeddings[i, : valid_len - 1].detach().cpu()
                seq_rep = residue_rep.mean(0)
                payload = {
                    "token_embeddings": residue_rep,
                    "sequence_embedding": seq_rep,
                }
                save_tensor_payload(output_dir / f"{safe_name(label)}.pt", payload)


def extract_saprot(
    fasta_path: str,
    output_dir: str,
    device: str = "cuda",
    model_dir: str = "weights/SaProt_weights",
    batch_size: int = 1,
) -> None:
    from transformers import EsmForMaskedLM, EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(model_dir)
    model = EsmForMaskedLM.from_pretrained(model_dir)
    model.eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)
    vocab = tokenizer.get_vocab()
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    unk_id = tokenizer.unk_token_id
    pad_id = tokenizer.pad_token_id
    max_len = tokenizer.model_max_length if tokenizer.model_max_length is not None else 2048

    records = read_fasta(fasta_path)
    output_dir = ensure_dir(output_dir)

    with torch.no_grad():
        for label, seq in records:
            seq_clean = seq.strip().upper()
            token_ids = [cls_id]
            for ch in seq_clean:
                token_ids.append(vocab.get(ch, unk_id))
            token_ids.append(eos_id)
            # Truncate if needed
            if len(token_ids) > max_len:
                token_ids = token_ids[: max_len - 1] + [eos_id]
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device_obj)
            attention_mask = torch.ones_like(input_ids, device=device_obj)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # [1, L, D]
            avail = hidden.shape[1] - 2  # exclude CLS/EOS
            seq_len = min(len(seq_clean), avail)
            take = seq_len
            residue_rep = hidden[0, 1 : 1 + take].detach().cpu()
            seq_rep = residue_rep.mean(0)
            payload = {
                "token_embeddings": residue_rep,
                "sequence_embedding": seq_rep,
            }
            save_tensor_payload(output_dir / f"{safe_name(label)}.pt", payload)


def extract_rinalmo(
    fasta_path: str,
    output_dir: str,
    device: str = "cuda",
    model_weights: str = "weights/rinalmo_weights/rinalmo_giga_pretrained.pt",
    rinalmo_type: str = "650M",
    batch_size: int = 1,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "RiNALMo"))
    from rinalmo.data.alphabet import Alphabet
    from rinalmo.data.constants import CLS_TKN, EOS_TKN, MASK_TKN, PAD_TKN, RNA_TOKENS, UNK_TKN
    from rinalmo.model.model import RiNALMo
    from rinalmo.config import model_config

    alphabet = Alphabet(
        standard_tkns=RNA_TOKENS,
        special_tkns=[CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
    )
    if rinalmo_type == "650M":
        size = "giga"
    elif rinalmo_type == "150M":
        size = "mega"
    elif rinalmo_type == "35M":
        size = "micro"
    elif rinalmo_type == "8M":
        size = "nano"
    else:
        raise ValueError(f"Unsupported rinalmo_type: {rinalmo_type}")
    config = model_config(size)
    model = RiNALMo(config)
    model.load_state_dict(torch.load(model_weights, map_location="cpu"))
    model.eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)
    if device_obj.type == "cuda":
        model = model.half()

    records = read_fasta(fasta_path)
    output_dir = ensure_dir(output_dir)

    with torch.no_grad():
        for batch in chunked(records, batch_size):
            labels, seqs = zip(*batch)
            tokens = torch.tensor(alphabet.batch_tokenize(list(seqs)), dtype=torch.long).to(device_obj)
            if device_obj.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(tokens)
            else:
                outputs = model(tokens)
            embeddings = outputs["representation"]
            for i, label in enumerate(labels):
                seq_len = len(seqs[i])
                residue_rep = embeddings[i, 1 : seq_len + 1].detach().cpu()
                seq_rep = residue_rep.mean(0)
                payload = {
                    "token_embeddings": residue_rep,
                    "sequence_embedding": seq_rep,
                }
                save_tensor_payload(output_dir / f"{safe_name(label)}.pt", payload)


def extract_protbert(
    fasta_path: str,
    output_dir: str,
    device: str = "cuda",
    model_dir: str = "weights/ProtBert_weights",
    batch_size: int = 1,
) -> None:
    from transformers import BertModel, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=False)
    model = BertModel.from_pretrained(model_dir)
    model.eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)

    records = read_fasta(fasta_path)
    output_dir = ensure_dir(output_dir)

    with torch.no_grad():
        for batch in chunked(records, batch_size):
            labels, raw_seqs = zip(*batch)
            seqs = [_sanitize_protein_sequence(s) for s in raw_seqs]
            enc = tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device_obj)
            attention_mask = enc["attention_mask"].to(device_obj)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            for i, label in enumerate(labels):
                valid_len = int(attention_mask[i].sum().item())
                token_len = max(0, valid_len - 2)
                seq_len = min(len(raw_seqs[i]), token_len)
                residue_rep = embeddings[i, 1 : 1 + seq_len].detach().cpu()
                seq_rep = residue_rep.mean(0) if residue_rep.numel() else torch.zeros(
                    embeddings.shape[-1]
                )
                payload = {
                    "token_embeddings": residue_rep,
                    "sequence_embedding": seq_rep,
                }
                save_tensor_payload(output_dir / f"{safe_name(label)}.pt", payload)


def extract_rna_fm(
    fasta_path: str,
    output_dir: str,
    device: str = "cuda",
    model_path: str = "weights/RNA-FM_weights/RNA-FM_pretrained.pth",
    repr_layer: Optional[int] = None,
    batch_size: int = 1,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "RNA-FM"))
    import fm

    model, alphabet = fm.pretrained.rna_fm_t12(model_location=model_path)
    model.eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)
    if repr_layer is None:
        repr_layer = model.num_layers

    batch_converter = alphabet.get_batch_converter()
    records = read_fasta(fasta_path)
    output_dir = ensure_dir(output_dir)

    with torch.no_grad():
        for batch in chunked(records, batch_size):
            labels, seqs = zip(*batch)
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device_obj)
            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            token_reps = results["representations"][repr_layer]
            for i, label in enumerate(labels):
                seq_len = len(seqs[i])
                residue_rep = token_reps[i, 1 : seq_len + 1].detach().cpu()
                seq_rep = residue_rep.mean(0)
                payload = {
                    "token_embeddings": residue_rep,
                    "sequence_embedding": seq_rep,
                }
                save_tensor_payload(output_dir / f"{safe_name(label)}.pt", payload)


def extract_esm_if1(
    pdb_dir: str,
    output_dir: str,
    device: str = "cuda",
    model_location: Optional[str] = None,
    chain_id: Optional[str] = None,
    pdb_list: Optional[Iterable[str]] = None,
) -> None:
    import esm
    from esm.inverse_folding import util as if_util

    if model_location:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
    else:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)

    output_dir = ensure_dir(output_dir)
    pdb_files = _resolve_pdb_files(pdb_dir, pdb_list)
    tmp_dir = ensure_dir(str(output_dir / "_tmp_pdb"))

    with torch.no_grad():
        for pdb_path in pdb_files:
            base_id = _pdb_base_id(pdb_path)
            chain_ids = _preferred_protein_chain_ids(output_dir, pdb_path, chain_id)
            if not chain_ids:
                continue
            for cid in chain_ids:
                name = f"{safe_name(base_id)}_prot_{safe_name(cid)}"
                out_path = output_dir / f"{name}.pt"
                if out_path.exists():
                    continue
                tmp_path = None
                coord_loaded = False
                last_exc: Optional[Exception] = None
                for cid_try in _chain_id_candidates(pdb_path, cid):
                    try:
                        coords, _ = if_util.load_coords(str(pdb_path), cid_try)
                        coord_loaded = True
                        break
                    except Exception as exc:
                        last_exc = exc
                if not coord_loaded:
                    tmp_path = tmp_dir / f"{pdb_path.stem}_{safe_name(cid)}.pdb"
                    if not _write_clean_chain_pdb(pdb_path, cid, tmp_path):
                        reason = f"{last_exc}" if last_exc else "unknown"
                        print(f"[esm_if1] Skipping {pdb_path.name} chain {cid}: unable to sanitize chain ({reason}).")
                        continue
                    try:
                        coords, _ = if_util.load_coords(str(tmp_path), "A")
                    except Exception as exc2:
                        print(f"[esm_if1] Skipping {pdb_path.name} chain {cid}: {exc2}.")
                        continue
                batch_converter = if_util.CoordBatchConverter(alphabet)
                coords_tensor, confidence, _, _, padding_mask = batch_converter(
                    [(coords, None, None)], device=device_obj
                )
                encoder_out = model.encoder.forward(
                    coords_tensor, padding_mask, confidence, return_all_hiddens=False
                )
                rep = encoder_out["encoder_out"][0][1:-1, 0]
                ref_len = _get_reference_length(Path(output_dir), pdb_path, cid) or rep.shape[0]
                take = min(rep.shape[0], ref_len)
                rep = rep[:take]
                payload = {
                    "token_embeddings": rep.detach().cpu(),
                    "sequence_embedding": rep.detach().cpu().mean(0),
                    "chain_id": cid,
                }
                save_tensor_payload(out_path, payload)
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink()


def extract_protein_mpnn(
    pdb_dir: str,
    output_dir: str,
    device: str = "cuda",
    model_weights: str = "weights/ProteinMPNN_weights/v_48_020.pt",
    model_weights_ca_only: Optional[str] = None,
    ca_only: bool = False,
    pdb_list: Optional[Iterable[str]] = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "ProteinMPNN"))
    import protein_mpnn_utils as mpnn_utils

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
        ca_only=ca_only,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)
    model_ca_only = None
    ca_only_supported: Optional[bool] = None
    ca_only_reason: Optional[str] = None

    def _ensure_ca_only_model() -> Optional[mpnn_utils.ProteinMPNN]:
        nonlocal model_ca_only
        nonlocal ca_only_supported, ca_only_reason
        if ca_only_supported is False:
            return None
        if model_ca_only is None:
            if not model_weights_ca_only:
                ca_only_supported = False
                ca_only_reason = "CA-only checkpoint not provided"
                return None
            checkpoint_ca = torch.load(model_weights_ca_only, map_location="cpu")
            ckpt_edge = checkpoint_ca.get("model_state_dict", {}).get("features.edge_embedding.weight")
            model_ca_only = mpnn_utils.ProteinMPNN(
                num_letters=21,
                node_features=checkpoint_ca.get("hidden_dim", hidden_dim),
                edge_features=checkpoint_ca.get("hidden_dim", hidden_dim),
                hidden_dim=checkpoint_ca.get("hidden_dim", hidden_dim),
                num_encoder_layers=checkpoint_ca.get("num_layers", num_layers),
                num_decoder_layers=checkpoint_ca.get("num_layers", num_layers),
                augment_eps=0.0,
                k_neighbors=checkpoint_ca.get("num_edges", checkpoint["num_edges"]),
                ca_only=True,
            )
            if ckpt_edge is None or ckpt_edge.shape != model_ca_only.features.edge_embedding.weight.shape:
                ca_only_supported = False
                ca_only_reason = "CA-only weights incompatible with checkpoint"
                model_ca_only = None
                return None
            try:
                model_ca_only.load_state_dict(checkpoint_ca["model_state_dict"])
            except RuntimeError as exc:
                ca_only_supported = False
                ca_only_reason = f"CA-only weights load failed: {exc}"
                model_ca_only = None
                return None
            model_ca_only.eval()
            model_ca_only = model_ca_only.to(device_obj)
            ca_only_supported = True
        return model_ca_only

    output_dir = ensure_dir(output_dir)
    pdb_files = _resolve_pdb_files(pdb_dir, pdb_list)

    if ca_only:
        model_ca = _ensure_ca_only_model()
        if model_ca is None:
            reason = ca_only_reason or "CA-only checkpoint not provided or incompatible"
            raise RuntimeError(f"ProteinMPNN CA-only requested but unavailable: {reason}")
        model = model_ca

    with torch.no_grad():
        for pdb_path in pdb_files:
            base_id = _pdb_base_id(pdb_path)
            chain_ids = _preferred_protein_chain_ids(output_dir, pdb_path, None)
            if pdb_path.suffix.lower() == ".cif":
                tmp_dir = ensure_dir(str(output_dir / "_tmp_pdb"))

                for cid in chain_ids:
                    name = f"{safe_name(base_id)}_prot_{safe_name(cid)}"
                    out_path = output_dir / f"{name}.pt"
                    if out_path.exists():
                        continue
                    use_ca_only = ca_only
                    tmp_path = tmp_dir / f"{pdb_path.stem}_{safe_name(cid)}.pdb"
                    if not tmp_path.exists():
                        if not _write_clean_chain_pdb(pdb_path, cid, tmp_path, ca_only=ca_only):
                            if not ca_only:
                                with_ncac, with_ca = _chain_backbone_stats(pdb_path, cid)
                                if with_ncac == 0 and with_ca > 0:
                                    if not _write_clean_chain_pdb(pdb_path, cid, tmp_path, ca_only=True):
                                        continue
                                    model_ca = _ensure_ca_only_model()
                                    if model_ca is None:
                                        reason = ca_only_reason or "CA-only weights unavailable"
                                        print(f"[proteinmpnn] Skipping {pdb_path.name} chain {cid}: {reason}.")
                                        continue
                                    use_ca_only = True
                                else:
                                    continue
                            else:
                                continue
                    pdb_dict_list = mpnn_utils.parse_PDB(str(tmp_path), input_chain_list=["A"], ca_only=use_ca_only)
                    if not pdb_dict_list:
                        continue
                    if not pdb_dict_list[0].get("seq"):
                        print(f"[proteinmpnn] Skipping {pdb_path.name} chain {cid}: no parseable residues.")
                        continue
                    batch = [pdb_dict_list[0]]
                    model_used = _ensure_ca_only_model() if use_ca_only else model
                    X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = mpnn_utils.tied_featurize(
                        batch,
                        device=device_obj,
                        chain_dict=None,
                        fixed_position_dict=None,
                        omit_AA_dict=None,
                        tied_positions_dict=None,
                        pssm_dict=None,
                        bias_by_res_dict=None,
                        ca_only=use_ca_only,
                    )
                    X = X.to(device_obj)
                    mask = mask.to(device_obj)
                    residue_idx = residue_idx.to(device_obj)
                    chain_encoding_all = chain_encoding_all.to(device_obj)
                    E, E_idx = model_used.features(X, mask, residue_idx, chain_encoding_all)
                    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
                    h_E = model_used.W_e(E)
                    mask_attend = mpnn_utils.gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
                    mask_attend = mask.unsqueeze(-1) * mask_attend
                    for layer in model_used.encoder_layers:
                        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
                    true_len = int(lengths[0].item())
                    ref_len = _get_reference_length(Path(output_dir), pdb_path, cid) or true_len
                    take = min(true_len, ref_len)
                    rep = h_V[0, :take].detach().cpu()
                    payload = {
                        "token_embeddings": rep,
                        "sequence_embedding": rep.mean(0),
                        "chain_id": cid,
                    }
                    save_tensor_payload(out_path, payload)
                for tmp_path in tmp_dir.glob(f"{pdb_path.stem}_*.pdb"):
                    tmp_path.unlink()
                continue

            pdb_for_mpnn = pdb_path
            for cid in chain_ids:
                name = f"{safe_name(base_id)}_prot_{safe_name(cid)}"
                out_path = output_dir / f"{name}.pt"
                if out_path.exists():
                    continue
                use_ca_only = ca_only
                pdb_dict_list = mpnn_utils.parse_PDB(str(pdb_for_mpnn), input_chain_list=[cid], ca_only=use_ca_only)
                if not pdb_dict_list:
                    if not ca_only:
                        with_ncac, with_ca = _chain_backbone_stats(pdb_path, cid)
                        if with_ncac == 0 and with_ca > 0:
                            model_ca = _ensure_ca_only_model()
                            if model_ca is None:
                                reason = ca_only_reason or "CA-only weights unavailable"
                                print(f"[proteinmpnn] Skipping {pdb_path.name} chain {cid}: {reason}.")
                                continue
                            use_ca_only = True
                            pdb_dict_list = mpnn_utils.parse_PDB(str(pdb_for_mpnn), input_chain_list=[cid], ca_only=True)
                    if not pdb_dict_list:
                        continue
                if not pdb_dict_list[0].get("seq"):
                    print(f"[proteinmpnn] Skipping {pdb_path.name} chain {cid}: no parseable residues.")
                    continue
                batch = [pdb_dict_list[0]]
                model_used = _ensure_ca_only_model() if use_ca_only else model
                X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = mpnn_utils.tied_featurize(
                    batch,
                    device=device_obj,
                    chain_dict=None,
                    fixed_position_dict=None,
                    omit_AA_dict=None,
                    tied_positions_dict=None,
                    pssm_dict=None,
                    bias_by_res_dict=None,
                    ca_only=use_ca_only,
                )
                X = X.to(device_obj)
                mask = mask.to(device_obj)
                residue_idx = residue_idx.to(device_obj)
                chain_encoding_all = chain_encoding_all.to(device_obj)
                E, E_idx = model_used.features(X, mask, residue_idx, chain_encoding_all)
                h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
                h_E = model_used.W_e(E)
                mask_attend = mpnn_utils.gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
                mask_attend = mask.unsqueeze(-1) * mask_attend
                for layer in model_used.encoder_layers:
                    h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
                true_len = int(lengths[0].item())
                ref_len = _get_reference_length(Path(output_dir), pdb_path, cid) or true_len
                take = min(true_len, ref_len)
                rep = h_V[0, :take].detach().cpu()
                payload = {
                    "token_embeddings": rep,
                    "sequence_embedding": rep.mean(0),
                    "chain_id": cid,
                }
                save_tensor_payload(out_path, payload)
            # tmp_path is only used in the CIF branch; nothing to clean here.


def extract_protrek(
    pdb_dir: str,
    output_dir: str,
    device: str = "cuda",
    model_dir: Optional[str] = None,
    from_checkpoint: Optional[str] = None,
    protein_config: Optional[str] = None,
    text_config: Optional[str] = None,
    structure_config: Optional[str] = None,
    foldseek_bin: Optional[str] = None,
    batch_size: int = 32,
    chain_id: Optional[str] = None,
    pdb_list: Optional[Iterable[str]] = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.modules.pop("utils", None)
    from ProTrek.model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
    from ProTrek.utils.foldseek_util import get_struc_seq
    from torch.nn.functional import normalize

    if not foldseek_bin or not Path(foldseek_bin).exists():
        raise FileNotFoundError(f"Foldseek binary not found: {foldseek_bin}")

    if model_dir:
        model_dir_path = Path(model_dir)
        protein_config = protein_config or str(model_dir_path / "esm2_t33_650M_UR50D")
        text_config = text_config or str(model_dir_path / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        structure_config = structure_config or str(model_dir_path / "foldseek_t30_150M")
        from_checkpoint = from_checkpoint or str(model_dir_path / "ProTrek_650M.pt")

    if not (protein_config and text_config and structure_config and from_checkpoint):
        raise ValueError("ProTrek configs and checkpoint must be specified.")

    model = ProTrekTrimodalModel(
        protein_config=protein_config,
        text_config=text_config,
        structure_config=structure_config,
        from_checkpoint=from_checkpoint,
        init_metrics=False,
    ).eval()
    device_obj = _device_from_string(device)
    model = model.to(device_obj)

    output_dir = ensure_dir(output_dir)
    pdb_files = _resolve_pdb_files(pdb_dir, pdb_list)

    with torch.no_grad():
        for pdb_path in pdb_files:
            base_id = _pdb_base_id(pdb_path)
            chain_ids = _preferred_protein_chain_ids(output_dir, pdb_path, chain_id)
            for cid in chain_ids:
                name = f"{safe_name(base_id)}_prot_{safe_name(cid)}"
                out_path = output_dir / f"{name}.pt"
                if out_path.exists():
                    continue
                seq_dict = None
                resolved_chain = None
                for cid_try in _chain_id_candidates(pdb_path, cid):
                    seq_dict = get_struc_seq(str(foldseek_bin), str(pdb_path), chains=[cid_try])
                    if seq_dict and cid_try in seq_dict:
                        resolved_chain = cid_try
                        break
                if not seq_dict or not resolved_chain:
                    continue
                foldseek_seq = seq_dict[resolved_chain][1].lower()
                seq_len = len(foldseek_seq)
                tokenizer = model.structure_encoder.tokenizer
                encoder = model.structure_encoder.model.esm
                proj = model.structure_encoder.out
                inputs = tokenizer.batch_encode_plus([foldseek_seq], return_tensors="pt", padding=False)
                inputs = {k: v.to(device_obj) for k, v in inputs.items()}
                last_hidden = encoder(**inputs).last_hidden_state
                token_repr = normalize(proj(last_hidden), dim=-1)[0]
                token_len = token_repr.shape[0]
                if token_len == seq_len + 2:
                    token_repr = token_repr[1:-1]
                elif token_len == seq_len + 1:
                    token_repr = token_repr[1:]
                elif token_len > seq_len:
                    token_repr = token_repr[:seq_len]
                ref_len = _get_reference_length(Path(output_dir), pdb_path, cid) or seq_len
                take = min(seq_len, ref_len, token_repr.shape[0])
                rep = token_repr[:take].detach().cpu()
                payload = {
                    "token_embeddings": rep,
                    "sequence_embedding": rep.mean(0),
                    "chain_id": cid,
                }
                save_tensor_payload(out_path, payload)


def extract_rna_msm(
    root_path: str,
    msa_path: str,
    msa_list: str,
    model_path: str,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    extra_overrides: Optional[List[str]] = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "RNA-MSM" / "RNA_MSM_Inference.py"
    overrides = [
        f"data.root_path={root_path}",
        f"data.MSA_path={msa_path}",
        f"data.MSA_list={msa_list}",
        f"data.model_path={model_path}",
        f"data.output_dir={output_dir or msa_path}",
        f"data.device={device}",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra.job.chdir=false",
    ]
    if extra_overrides:
        overrides.extend(extra_overrides)
    cmd = [sys.executable, str(script_path)] + overrides
    subprocess.run(cmd, check=True)
    if output_dir:
        output_dir = ensure_dir(output_dir)
        for emb_path in Path(output_dir).glob("*_emb.npy"):
            name = emb_path.stem.replace("_emb", "")
            embedding = torch.tensor(np.load(emb_path))
            payload = {
                "token_embeddings": embedding,
                "sequence_embedding": embedding.mean(0),
            }
            save_tensor_payload(output_dir / f"{safe_name(name)}.pt", payload)
            emb_path.unlink()
        for atp_path in Path(output_dir).glob("*_atp.npy"):
            atp_path.unlink()


def extract_rna_ernie(
    fasta_path: str,
    output_dir: str,
    device: str = "cpu",
    model_dir: str = "weights/RNAErnie_weights",
    vocab_path: str = "RNAErnie/data/vocab/vocab_1MER.txt",
    batch_size: int = 256,
    max_seq_len: int = 512,
) -> None:
    # Avoid MKL load issues on some systems when running on CPU.
    if device.lower() == "cpu":
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    try:
        import paddle
        from paddlenlp.transformers import ErnieModel
    except ImportError as exc:
        raise RuntimeError("RNAErnie requires paddlepaddle and paddlenlp.") from exc

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "RNAErnie"))
    from rna_ernie import BatchConverter

    paddle.set_device(device)
    batch_converter = BatchConverter(
        k_mer=1,
        vocab_path=str(repo_root / vocab_path),
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    model = ErnieModel.from_pretrained(str(repo_root / model_dir))
    model.eval()

    records = read_fasta(fasta_path)
    output_dir = ensure_dir(output_dir)

    with paddle.no_grad():
        for names, seqs, inputs_ids in batch_converter(records):
            embeddings = model(inputs_ids)[0].detach().numpy()
            for i, name in enumerate(names):
                seq_len = len(seqs[i])
                # Assume embedding layout: [CLS] + tokens + [SEP]
                token_embeddings = torch.tensor(embeddings[i, 1 : 1 + seq_len])
                seq_rep = token_embeddings.mean(0)
                payload = {
                    "token_embeddings": token_embeddings,
                    "sequence_embedding": seq_rep,
                }
                save_tensor_payload(output_dir / f"{safe_name(name)}.pt", payload)


def extract_rnabert(
    fasta_path: str,
    output_dir: str,
    model_weights: str = "weights/RNABERT_weights/bert_mul_2.pth",
    batch_size: int = 40,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "RNABERT" / "MLM_SFP.py"
    output_dir = ensure_dir(output_dir)
    out_file = output_dir / "rnabert_embeddings.pt"

    def _clean_seq(seq: str) -> str:
        seq = seq.upper().replace("T", "U")
        allowed = set("ACGU")
        # RNABERT filters out any sequence with non-ACGU tokens, so coerce to A.
        return "".join(ch if ch in allowed else "A" for ch in seq)

    records = read_fasta(fasta_path)
    cleaned_path = output_dir / "._tmp_rnabert_input.fasta"
    with open(cleaned_path, "w", encoding="utf-8") as handle:
        for name, seq in records:
            cleaned = _clean_seq(seq)
            handle.write(f">{name}\n{cleaned}\n")

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(repo_root / "RNABERT" / "RNA_bert_config.json"),
        "--pretraining",
        str(repo_root / model_weights),
        "--data_embedding",
        str(cleaned_path),
        "--embedding_output",
        str(out_file),
        "--batch",
        str(batch_size),
    ]
    subprocess.run(cmd, check=True)
    cleaned_path.unlink(missing_ok=True)
    _rewrite_embeddings_in_dir(output_dir)


def extract_rhofold(
    fasta_path: str,
    output_dir: str,
    device: str = "cuda",
    ckpt_path: str = "weights/RhFold_weights/model_20221010_params.pt",
    input_a3m: Optional[str] = None,
    single_seq_pred: bool = True,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "RhoFold" / "inference.py"
    output_dir = ensure_dir(output_dir)
    fasta_path = Path(fasta_path)
    fasta_files = [fasta_path]
    if fasta_path.is_dir():
        fasta_files = sorted([p for p in fasta_path.iterdir() if p.suffix in {".fa", ".fasta"}])
    for fasta_file in fasta_files:
        run_output = output_dir / f"._tmp_{fasta_file.stem}"
        run_output.mkdir(parents=True, exist_ok=True)
        local_input_a3m = input_a3m
        local_single_seq = single_seq_pred
        if local_single_seq and not local_input_a3m:
            # Build a minimal aligned MSA with a single sequence.
            seq = read_fasta(str(fasta_file))[0][1]
            local_input_a3m = str(run_output / "single.a3m")
            Path(local_input_a3m).write_text(f">{fasta_file.stem}\n{seq}\n")
            local_single_seq = False
        cmd = [
            sys.executable,
            str(script_path),
            "--device",
            device,
            "--ckpt",
            str(repo_root / ckpt_path),
            "--input_fas",
            str(fasta_file),
            "--output_dir",
            str(run_output),
        ]
        if local_input_a3m:
            cmd.extend(["--input_a3m", str(local_input_a3m)])
        if local_single_seq:
            cmd.extend(["--single_seq_pred", "True"])
        cmd.extend(["--relax_steps", "0", "--skip_pdb", "--save_embedding"])
        subprocess.run(cmd, check=True)
        tmp_embed = run_output / "rhofold_embeddings.pt"
        stem = fasta_file.stem
        chain = None
        if "_rna_" in stem:
            pdb_stem, chain = stem.split("_rna_", 1)
            out_name = f"{safe_name(pdb_stem)}_rna_{safe_name(chain)}.pt"
        else:
            out_name = f"{safe_name(stem)}.pt"
        if tmp_embed.exists():
            target = output_dir / out_name
            shutil.move(str(tmp_embed), str(target))
            _rewrite_embeddings_in_dir(output_dir)
        shutil.rmtree(run_output, ignore_errors=True)


def extract_alphafold2_outputs(
    fasta_path: str,
    output_dir: str,
    data_dir: str,
    model_preset: str = "monomer",
    db_preset: str = "reduced_dbs",
    max_template_date: str = "2020-05-14",
    use_gpu_relax: bool = False,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "alphafold" / "run_alphafold.py"
    fasta_arg = fasta_path
    fasta_path_obj = Path(fasta_path)
    if fasta_path_obj.is_dir():
        fasta_files = sorted([p for p in fasta_path_obj.iterdir() if p.suffix in {".fa", ".fasta"}])
        fasta_arg = ",".join(str(p) for p in fasta_files)
    cmd = [
        sys.executable,
        str(script_path),
        "--fasta_paths",
        str(fasta_arg),
        "--output_dir",
        str(output_dir),
        "--data_dir",
        str(data_dir),
        "--model_preset",
        model_preset,
        "--db_preset",
        db_preset,
        "--max_template_date",
        max_template_date,
        "--models_to_relax",
        "none",
        "--use_gpu_relax",
        "true" if use_gpu_relax else "false",
    ]
    subprocess.run(cmd, check=True)


def parse_alphafold2_features(output_dir: str, save_dir: str) -> None:
    output_dir = Path(output_dir)
    save_dir = ensure_dir(save_dir)
    for result_path in output_dir.glob("**/result_*.pkl"):
        with open(result_path, "rb") as handle:
            result = pickle.load(handle)
        payload = {}
        if "plddt" in result:
            payload["plddt"] = result["plddt"]
        if "distogram" in result:
            distogram = result["distogram"]
            if isinstance(distogram, dict):
                if "logits" in distogram:
                    payload["distogram_logits"] = distogram["logits"]
                if "bin_edges" in distogram:
                    payload["distogram_bin_edges"] = distogram["bin_edges"]
        if "predicted_aligned_error" in result:
            payload["predicted_aligned_error"] = result["predicted_aligned_error"]
        if payload:
            out_path = save_dir / f"{result_path.parent.name}_{result_path.stem}.npz"
            save_numpy_payload(out_path, payload)


def write_run_metadata(output_dir: str, metadata: Dict[str, str]) -> None:
    output_dir = ensure_dir(output_dir)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
