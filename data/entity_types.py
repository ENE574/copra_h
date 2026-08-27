"""Unified entity-pair abstraction for protein / RNA / DNA complexes and PPI."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EntityType(str, Enum):
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"

    @classmethod
    def from_string(cls, value: str) -> "EntityType":
        v = str(value).strip().lower()
        if v in ("protein", "prot", "ppi_prot"):
            return cls.PROTEIN
        if v in ("rna", "na", "prot_na"):
            return cls.RNA
        if v in ("dna",):
            return cls.DNA
        raise ValueError(f"Unknown entity type: {value}")


@dataclass(frozen=True)
class EntityPairSpec:
    """Side A is always the primary (mutation) chain group; side B is the partner."""

    entity_a_type: EntityType
    entity_b_type: EntityType
    interaction: str  # prot_na | prot_dna | ppi
    mutation_task: str = "none"  # none | ddg_prot_na | ddg_ppi

    @property
    def entity_b_is_protein(self) -> bool:
        return self.entity_b_type == EntityType.PROTEIN

    @property
    def identifier_a_is_zero(self) -> bool:
        """Batch ``identifier`` uses 0 for entity A and 1 for entity B."""
        return True


def resolve_entity_pair_from_args(args: Any) -> EntityPairSpec:
    """Build EntityPairSpec from dataset or model EasyDict / dict."""
    if args is None:
        return EntityPairSpec(EntityType.PROTEIN, EntityType.RNA, "prot_na")

    # Support both dict-style and attribute-style access
    def _get(d, key, default=None):
        if hasattr(d, key):
            return getattr(d, key)
        if isinstance(d, dict) and key in d:
            return d[key]
        return default

    ep = _get(args, "entity_pair")
    if ep is not None:
        return EntityPairSpec(
            entity_a_type=EntityType.from_string(_get(ep, "entity_a_type")),
            entity_b_type=EntityType.from_string(_get(ep, "entity_b_type")),
            interaction=str(_get(ep, "interaction", "prot_na")),
            mutation_task=str(_get(ep, "mutation_task", "none")),
        )

    ea_type = _get(args, "entity_a_type")
    if ea_type is not None:
        return EntityPairSpec(
            entity_a_type=EntityType.from_string(ea_type),
            entity_b_type=EntityType.from_string(_get(args, "entity_b_type", "rna")),
            interaction=str(_get(args, "interaction", "prot_na")),
            mutation_task=str(_get(args, "mutation_task", "none")),
        )

    entity_type = str(_get(args, "entity_type", "prot_na"))
    mutation_task = str(_get(args, "mutation_task", "none"))
    mut = bool(_get(args, "mut", False))

    if entity_type == "ppi":
        task = mutation_task if mutation_task != "none" else ("ddg_ppi" if mut else "none")
        return EntityPairSpec(
            EntityType.PROTEIN,
            EntityType.PROTEIN,
            "ppi",
            mutation_task=task,
        )

    na_type = str(_get(args, "entity_b_type", _get(args, "na_entity_type", "rna"))).lower()
    b_type = EntityType.DNA if na_type == "dna" else EntityType.RNA
    interaction = "prot_dna" if b_type == EntityType.DNA else "prot_na"
    if mutation_task != "none":
        task = mutation_task
    else:
        task = ("ddg_prot_dna" if b_type == EntityType.DNA else "ddg_prot_na") if mut else "none"
    return EntityPairSpec(EntityType.PROTEIN, b_type, interaction, mutation_task=task)


def entity_pair_to_legacy_entity_type(spec: EntityPairSpec) -> str:
    if spec.interaction == "ppi":
        return "ppi"
    return "prot_na"


def entity_pair_summary(spec: EntityPairSpec) -> Dict[str, str]:
    return {
        "entity_a_type": spec.entity_a_type.value,
        "entity_b_type": spec.entity_b_type.value,
        "interaction": spec.interaction,
        "mutation_task": spec.mutation_task,
    }
