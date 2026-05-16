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
      - protein_sequence / protein_structure / rna_sequence / rna_structure
    entity:
      - prot / rna
    """

    root: str

    # model names are folder names under each entity_group
    protein_sequence_models: Tuple[str, str, str] = ("esm2", "prott5", "saprot")
    protein_structure_models: Tuple[str, str, str] = ("esm_if1", "protbert", "protrek")
    rna_sequence_models: Tuple[str, str, str] = ("rinalmo", "rna_fm", "rna_msm")
    rna_structure_models: Tuple[str, str, str] = ("rna_ernie", "rhofold", "rnabert")

    def file_path(self, group: str, model: str, pdb_id: str, entity: str, chain_id: str) -> str:
        fname = "{}_{}_{}.pt".format(pdb_id, entity, chain_id)
        path = os.path.join(self.root, group, model, fname)

        # Backward compatibility for historical naming:
        # rna_structure/ernie_rna  -> rna_structure/rna_ernie
        if os.path.exists(path):
            return path
        if group == "rna_structure":
            if model == "rna_ernie":
                legacy = os.path.join(self.root, group, "ernie_rna", fname)
                if os.path.exists(legacy):
                    return legacy
            if model == "ernie_rna":
                new = os.path.join(self.root, group, "rna_ernie", fname)
                if os.path.exists(new):
                    return new
        return path


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

