"""
PyRosetta PIA targets: Rosetta ``fa_*`` score terms are mapped onto **FoldX-aligned**
``PIA_PHYSICS_NAMES`` (``Electro``, ``Energy_SolvP``, ``Energy_SolvH``, ``Energy_VdW``, ``Energy_Hbond``).

The bridge is approximate (Rosetta solvation is not split into FoldX polar / hydrophobic).

Requires PyRosetta when ``use_pyrosetta_physics=True`` in ``StructureDataset``.
Prefer ``num_workers: 0`` in the DataLoader (forked workers + PyRosetta can be fragile).
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, Optional

import torch

from data.pia_physics_names import PIA_PHYSICS_NAMES

_PYROSETTA_AVAILABLE: Optional[bool] = None
_INIT_DONE = False
_SCOREFXN = None
_SCOREFXN_NAME: Optional[str] = None


def pyrosetta_available() -> bool:
    global _PYROSETTA_AVAILABLE
    if _PYROSETTA_AVAILABLE is None:
        try:
            import pyrosetta  # noqa: F401

            _PYROSETTA_AVAILABLE = True
        except ImportError:
            _PYROSETTA_AVAILABLE = False
    return bool(_PYROSETTA_AVAILABLE)


def init_pyrosetta(extra_options: str = "-mute all") -> None:
    """Idempotent PyRosetta init (call from main process before DataLoader workers)."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    if not pyrosetta_available():
        raise ImportError(
            "PyRosetta is not installed. Install it in your conda env, then retry "
            "(see https://www.pyrosetta.org/downloads)."
        )
    from pyrosetta import init

    opts = "-mute all"
    if extra_options:
        opts = f"{opts} {extra_options}".strip()
    init(extra_options=opts)
    _INIT_DONE = True


def _get_scorefxn(name: str):
    global _SCOREFXN, _SCOREFXN_NAME
    if _SCOREFXN is not None and _SCOREFXN_NAME == name:
        return _SCOREFXN
    from pyrosetta import create_score_function

    _SCOREFXN = create_score_function(name)
    _SCOREFXN_NAME = name
    return _SCOREFXN


_ROSETTA_SCORE_KEYS = ("fa_elec", "fa_sol", "fa_atr", "fa_rep", "fa_hbond")


def _score_type_enum(name: str):
    from pyrosetta.rosetta.core.scoring import ScoreType

    if not hasattr(ScoreType, name):
        return None
    return getattr(ScoreType, name)


def extract_rosetta_score_terms(pose, sfxn) -> Dict[str, float]:
    """Whole-complex weighted Rosetta ``fa_*`` totals (not FoldX field names)."""
    sfxn(pose)
    emap = pose.energies().total_energies()
    out: Dict[str, float] = {}
    for name in _ROSETTA_SCORE_KEYS:
        st = _score_type_enum(name)
        if st is None:
            out[name] = 0.0
            continue
        try:
            out[name] = float(emap[st])
        except Exception:
            out[name] = 0.0
    return out


def rosetta_terms_to_foldx_pia_labels(raw: Dict[str, float]) -> Dict[str, float]:
    """Map Rosetta decomposition into FoldX-aligned PIA head keys."""
    fe = float(raw.get("fa_elec", 0.0))
    fs = float(raw.get("fa_sol", 0.0))
    fa = float(raw.get("fa_atr", 0.0))
    fr = float(raw.get("fa_rep", 0.0))
    fh = float(raw.get("fa_hbond", 0.0))
    return {
        "Electro": fe,
        "Energy_SolvP": 0.5 * fs,
        "Energy_SolvH": 0.5 * fs,
        "Energy_VdW": fa + fr,
        "Energy_Hbond": fh,
    }


def zero_physics_targets() -> Dict[str, torch.Tensor]:
    """Placeholder targets (no Rosetta); use only with ``physics_aux_max_weight: 0`` or for ablations."""
    return {k: torch.tensor(0.0, dtype=torch.float32) for k in PIA_PHYSICS_NAMES}


def compute_physics_targets_tensor(
    pdb_path: str,
    *,
    scorefxn_name: str = "ref2015",
    init_extra_options: str = "",
) -> Dict[str, torch.Tensor]:
    """
    Load ``pdb_path``, score with ``scorefxn_name``, return tensors keyed like ``PIA_PHYSICS_NAMES``.

    On failure (missing file, parse error, scoring error), returns zeros for all keys.
    """
    init_pyrosetta(init_extra_options)
    from pyrosetta import pose_from_file

    zero = zero_physics_targets()
    if not pdb_path or not os.path.isfile(pdb_path):
        return zero

    try:
        pose = pose_from_file(pdb_path)
    except Exception:
        return zero

    try:
        sfxn = _get_scorefxn(scorefxn_name)
        raw = extract_rosetta_score_terms(pose, sfxn)
        pia = rosetta_terms_to_foldx_pia_labels(raw)
        return {k: torch.tensor(pia[k], dtype=torch.float32) for k in PIA_PHYSICS_NAMES}
    except Exception:
        return zero


def compute_physics_targets_tensor_cached(
    pdb_path: str,
    cache: Optional[dict],
    *,
    scorefxn_name: str = "ref2015",
    init_extra_options: str = "",
) -> Dict[str, torch.Tensor]:
    """In-process memo keyed by (path, mtime, sfxn, extra_opts)."""
    if cache is None:
        return compute_physics_targets_tensor(
            pdb_path,
            scorefxn_name=scorefxn_name,
            init_extra_options=init_extra_options,
        )
    try:
        mtime = os.path.getmtime(pdb_path)
    except OSError:
        mtime = -1.0
    key = (os.path.abspath(pdb_path), mtime, scorefxn_name, init_extra_options)
    if key not in cache:
        cache[key] = compute_physics_targets_tensor(
            pdb_path,
            scorefxn_name=scorefxn_name,
            init_extra_options=init_extra_options,
        )
    return cache[key]


_warned_non_pyrosetta_pia = False


def warn_zero_pia_targets_when_no_pyrosetta() -> None:
    """Once: PIA keys follow FoldX ``Stability`` names; without CSV or PyRosetta, labels are zeros."""
    global _warned_non_pyrosetta_pia
    if _warned_non_pyrosetta_pia:
        return
    _warned_non_pyrosetta_pia = True
    warnings.warn(
        "StructureDataset: use_pyrosetta_physics=False but PIA heads expect FoldX-aligned energy terms "
        f"{tuple(PIA_PHYSICS_NAMES)}. physics_targets are zeros. "
        "Set physics_targets_csv to a precomputed CSV (e.g. FoldX), use_pyrosetta_physics=True, "
        "or set physics_aux_max_weight to 0.",
        UserWarning,
        stacklevel=2,
    )
