"""
Canonical PIA branch names = FoldX 5 ``Stability`` / ``*_ST.fxout`` energy fields (one head each).

These match the tab-separated decomposition written by ``foldx/foldx_20270131 --command=Stability``.
Rosetta/PyRosetta paths fill the **same keys** via an approximate bridge (see ``data/pyrosetta_physics``).
"""

# Order matches typical FoldX stability column layout (see ``data/foldx_physics``).
PIA_PHYSICS_NAMES = ("Electro", "Energy_SolvP", "Energy_SolvH", "Energy_VdW")
