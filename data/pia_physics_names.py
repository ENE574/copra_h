"""
Canonical PIA branch names = FoldX ``Stability`` / ``*_ST.fxout`` energy fields (one head each).

These match the tab-separated decomposition written by ``foldx/foldx_20270131 --command=Stability``.
Rosetta/PyRosetta paths fill the **same keys** via an approximate bridge (see ``data/pyrosetta_physics``).

DNA-specific terms (``DNA_Electro``, ``DNA_Stacking``, ``DNA_VDW``) are computed from
protein–DNA complex structures (see ``data/foldx_physics.compute_dna_physics_terms``).
"""

# Order matches typical FoldX stability column layout (see ``data/foldx_physics``).
PIA_PHYSICS_NAMES = (
    "Electro",
    "Energy_SolvP",
    "Energy_SolvH",
    "Energy_VdW",
    "Energy_Hbond",
    "DNA_Electro",
    "DNA_Stacking",
    "DNA_VDW",
)
