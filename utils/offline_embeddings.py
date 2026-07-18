from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

from utils.torch_compat import torch_load_compat


@dataclass(frozen=True)
class OfflineEmbeddingSpec:
    """
    Offline embedding layout:
      {root}/{entity_group}/{model_name}/{pdb}_{entity}_{chain}.pt

    entity_group:
      - protein_sequence / protein_structure / rna_sequence|dna_sequence / rna_structure|dna_structure
    entity:
      - prot / rna  (embedding filenames still use {pdb}_rna_{chain}.pt for DNA)
    """

    root: str
    mutant_subdir: Optional[str] = "mutant"

    # model names are folder names under each entity_group
    protein_sequence_models: Tuple[str, ...] = ("esm2",)
    protein_structure_models: Tuple[str, ...] = ("esm_if1",)
    rna_sequence_models: Tuple[str, ...] = ("rna_fm",)
    rna_structure_models: Tuple[str, ...] = ("rhofold",)
    dna_sequence_models: Tuple[str, ...] = ()
    dna_structure_models: Tuple[str, ...] = ()
    na_sequence_group: str = "rna_sequence"
    na_structure_group: str = "rna_structure"
    partner_use_protein_embeddings: bool = False
    na_wt_on_mut: bool = True

    def embedding_root(self, variant: str = "wt") -> str:
        if variant == "mut" and self.mutant_subdir:
            return os.path.join(self.root, self.mutant_subdir)
        return self.root

    def file_path(
        self,
        group: str,
        model: str,
        pdb_id: str,
        entity: str,
        chain_id: str,
        variant: str = "wt",
    ) -> str:
        fname = "{}_{}_{}.pt".format(pdb_id, entity, chain_id)
        roots = [self.embedding_root(variant)]
        if variant == "mut" and self.mutant_subdir:
            roots.append(self.root)

        for root in roots:
            path = os.path.join(root, group, model, fname)
            if os.path.exists(path):
                return path
            if group == "rna_structure":
                if model == "rna_ernie":
                    legacy = os.path.join(root, group, "ernie_rna", fname)
                    if os.path.exists(legacy):
                        return legacy
                if model == "ernie_rna":
                    new = os.path.join(root, group, "rna_ernie", fname)
                    if os.path.exists(new):
                        return new
        return os.path.join(roots[0], group, model, fname)


def _resolve_sample_id_candidates(item: dict, variant: str) -> List[str]:
    """Return sample id candidates in priority order for offline .pt lookup."""
    full_id = str(item.get("id", "") or "").strip()
    complex_id = str(item.get("complex", "") or "").strip()
    if variant == "mut":
        return [full_id] if full_id else ([complex_id] if complex_id else [])
    candidates: List[str] = []
    if full_id:
        candidates.append(full_id)
    if complex_id and complex_id not in candidates:
        candidates.append(complex_id)
    return candidates


def load_offline_token_embeddings_resolved(
    spec: OfflineEmbeddingSpec,
    group: str,
    model: str,
    sample_ids: List[str],
    entity: str,
    chain_id: str,
    variant: str = "wt",
) -> torch.Tensor:
    last_path = ""
    for sample_id in sample_ids:
        if not sample_id:
            continue
        path = spec.file_path(group, model, sample_id, entity, chain_id, variant=variant)
        last_path = path
        if os.path.isfile(path):
            return load_offline_token_embeddings(path)
    raise FileNotFoundError(
        "No offline embedding found for variant='{}' group='{}' model='{}' "
        "entity='{}' chain='{}' tried sample_ids={} last_path={}".format(
            variant, group, model, entity, chain_id, sample_ids, last_path
        )
    )


def stack_offline_embeddings_for_batch(
    spec: OfflineEmbeddingSpec,
    data_list: List[dict],
    max_prot_len: int,
    max_rna_len: int,
    variant: str = "wt",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load offline token embeddings for one variant (wt or mut) across a collated batch."""

    def _pad_into_token_space(x_residue: torch.Tensor, max_len: int) -> torch.Tensor:
        L = int(x_residue.shape[0])
        out = x_residue.new_zeros((max_len, int(x_residue.shape[1])))
        end = min(L, max_len - 2)
        if end > 0:
            out[1 : 1 + end] = x_residue[:end]
        return out

    partner_is_prot = spec.partner_use_protein_embeddings
    offline = {
        "prot_seq": {m: [] for m in spec.protein_sequence_models},
        "prot_struct": {m: [] for m in spec.protein_structure_models},
    }
    has_dna = len(spec.dna_sequence_models) > 0 or len(spec.dna_structure_models) > 0
    if has_dna:
        offline["dna_seq"] = {m: [] for m in spec.dna_sequence_models}
        offline["dna_struct"] = {m: [] for m in spec.dna_structure_models}
    else:
        if partner_is_prot:
            # PPI: partner uses protein embeddings → init rna_seq/rna_struct with protein models
            offline["rna_seq"] = {m: [] for m in spec.protein_sequence_models}
            offline["rna_struct"] = {m: [] for m in spec.protein_structure_models}
        else:
            offline["rna_seq"] = {m: [] for m in spec.rna_sequence_models}
            offline["rna_struct"] = {m: [] for m in spec.rna_structure_models}

    dna_models = spec.dna_sequence_models if has_dna else ()
    dna_struct_models = spec.dna_structure_models if has_dna else ()
    if partner_is_prot and not has_dna:
        rna_models = spec.protein_sequence_models
        rna_struct_models = spec.protein_structure_models
    else:
        rna_models = spec.rna_sequence_models if not has_dna else ()
        rna_struct_models = spec.rna_structure_models if not has_dna else ()

    for item in data_list:
        sample_ids = _resolve_sample_id_candidates(item, variant=variant)
        prot_chain_ids = item.get("prot_chain_ids", [])
        rna_chain_ids = item.get("rna_chain_ids", [])

        for chain_id in prot_chain_ids:
            for m in spec.protein_sequence_models:
                x = load_offline_token_embeddings_resolved(
                    spec, "protein_sequence", m, sample_ids, "prot", chain_id, variant=variant
                )
                offline["prot_seq"][m].append(_pad_into_token_space(x, max_prot_len))
            for m in spec.protein_structure_models:
                x = load_offline_token_embeddings_resolved(
                    spec, "protein_structure", m, sample_ids, "prot", chain_id, variant=variant
                )
                offline["prot_struct"][m].append(_pad_into_token_space(x, max_prot_len))

        if variant == "mut" and spec.na_wt_on_mut:
            na_variant = "wt"
        else:
            na_variant = variant
        na_sample_ids = _resolve_sample_id_candidates(item, variant=na_variant)
        na_entity = "prot" if spec.partner_use_protein_embeddings else "rna"
        na_seq_group = (
            "protein_sequence" if spec.partner_use_protein_embeddings else spec.na_sequence_group
        )
        na_str_group = (
            "protein_structure" if spec.partner_use_protein_embeddings else spec.na_structure_group
        )
        for chain_id in rna_chain_ids:
            for m in rna_models:
                x = load_offline_token_embeddings_resolved(
                    spec,
                    na_seq_group,
                    m,
                    na_sample_ids,
                    na_entity,
                    chain_id,
                    variant=na_variant,
                )
                offline["rna_seq"][m].append(_pad_into_token_space(x, max_rna_len))
            for m in rna_struct_models:
                x = load_offline_token_embeddings_resolved(
                    spec,
                    na_str_group,
                    m,
                    na_sample_ids,
                    na_entity,
                    chain_id,
                    variant=na_variant,
                )
                offline["rna_struct"][m].append(_pad_into_token_space(x, max_rna_len))
            for m in dna_models:
                x = load_offline_token_embeddings_resolved(
                    spec, na_seq_group, m, na_sample_ids, na_entity, chain_id, variant=na_variant,
                )
                offline["dna_seq"][m].append(_pad_into_token_space(x, max_rna_len))
            for m in dna_struct_models:
                x = load_offline_token_embeddings_resolved(
                    spec, na_str_group, m, na_sample_ids, na_entity, chain_id, variant=na_variant,
                )
                offline["dna_struct"][m].append(_pad_into_token_space(x, max_rna_len))

    for group in offline:
        for m in offline[group]:
            if len(offline[group][m]) == 0:
                raise RuntimeError(
                    "Offline embedding list is empty for variant='{}' group='{}' model='{}'. "
                    "Check sample ids / offline_embedding_root / offline_mutant_subdir.".format(
                        variant, group, m
                    )
                )
            offline[group][m] = torch.stack(offline[group][m], dim=0)
    return offline


def load_offline_token_embeddings(path: str, *, map_location: str = "cpu") -> torch.Tensor:
    """
    Returns residue-level token embeddings of shape [L, D] from a saved .pt.
    Expected file format is a dict containing key 'token_embeddings'.
    """
    obj = torch_load_compat(path, map_location=map_location)
    if not isinstance(obj, dict) or "token_embeddings" not in obj:
        raise ValueError("Offline embedding file missing 'token_embeddings': {}".format(path))
    x = obj["token_embeddings"]
    if not torch.is_tensor(x) or x.ndim != 2:
        raise ValueError("Expected token_embeddings to be a 2D tensor in {}".format(path))
    # Some RNA structure embeddings are float64; standardize to float32.
    return x.float()

