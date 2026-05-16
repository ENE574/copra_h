"""
FoldX-derived PIA targets: keys are exactly ``PIA_PHYSICS_NAMES`` (FoldX ``Stability`` field names).

1. **FoldX 5.x** one-line ``*_ST.fxout`` (tab-separated), from ``foldx/foldx_20270131 --command=Stability``:

   - ``Electro`` (field index 5)
   - ``Energy_SolvP`` (6)
   - ``Energy_SolvH`` (7)
   - ``Energy_VdW`` (4)

   (``Energy_vdwclash`` / ``backbone_vdwclash`` are separate FoldX terms; not used as PIA heads here.)

2. **Older tabular** ``Average*.fxout`` with a header row — mapped into the same four names.
"""
from __future__ import annotations

import csv
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _load_pia_physics_names() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location(
        "_pia_physics_names",
        Path(__file__).resolve().parent / "pia_physics_names.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load pia_physics_names.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tuple(getattr(mod, "PIA_PHYSICS_NAMES"))


PIA_PHYSICS_NAMES = _load_pia_physics_names()


def _norm_header(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    return s


def _float(x: str) -> float:
    x = x.strip()
    if not x:
        return 0.0
    return float(x)


def parse_foldx_tabular_row(header: List[str], data: List[str]) -> Dict[str, float]:
    """Map header cells -> float values from one data row (same length)."""
    out: Dict[str, float] = {}
    for h, v in zip(header, data):
        key = _norm_header(h)
        if key in ("pdb", "pdb_id", "file", "name"):
            continue
        try:
            out[key] = _float(v)
        except ValueError:
            continue
    return out


def _try_parse_foldx5_stability_st_line(line: str) -> Optional[Dict[str, float]]:
    """
    FoldX 5 Stability writes e.g. ``6DCC_0_ST.fxout``: one tab-separated line, first field PDB path.

    Indices (0-based): 0=pdb, 1=total, 2=BackHbond, 3=SideHbond, 4=Energy_VdW, 5=Electro,
    6=Energy_SolvP, 7=Energy_SolvH, 8=Energy_vdwclash, 9=Entropy_sidec, 10=Entropy_mainc,
    ..., 15=backbone_vdwclash (verified against console breakdown for 6DCC).
    """
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    parts = raw.split("\t")
    if len(parts) < 8:
        return None
    pdb_field = parts[0].strip().lower()
    if ".pdb" not in pdb_field:
        return None
    try:
        for i in (1, 4, 5, 6, 7):
            float(parts[i])
    except (ValueError, IndexError):
        return None

    return {
        "Electro": float(parts[5]),
        "Energy_SolvP": float(parts[6]),
        "Energy_SolvH": float(parts[7]),
        "Energy_VdW": float(parts[4]),
    }


def foldx_terms_to_pia_targets(terms: Mapping[str, float]) -> Dict[str, float]:
    """
    Map FoldX / legacy table keys -> ``PIA_PHYSICS_NAMES`` (kcal/mol as in FoldX).
    """
    # Already FoldX Stability four-field dict (from ``*_ST.fxout``).
    if all(k in terms for k in PIA_PHYSICS_NAMES):
        return {k: float(terms[k]) for k in PIA_PHYSICS_NAMES}

    def get(*names: str) -> float:
        for n in names:
            k = _norm_header(n)
            if k in terms:
                return float(terms[k])
        return 0.0

    elec = get("electro", "electrostatics", "energy_electro")
    sol_p = get("energy_solvp", "solvation_polar", "solvationpolar")
    sol_h = get("energy_solvh", "solvation_hydrophobic", "solvationhydrophobic")
    vdw = get("energy_vdw", "van_der_waals", "vanderwaals")
    clash = get(
        "energy_vdwclash",
        "van_der_waals_clashes",
        "vanderwaalsclashes",
        "van_der_waals_clash",
    )

    if sol_p == 0.0 and sol_h == 0.0:
        sol_combined = get("solvation", "energy_solvation")
        if sol_combined != 0.0:
            sol_p = 0.5 * sol_combined
            sol_h = 0.5 * sol_combined

    return {
        "Electro": elec,
        "Energy_SolvP": sol_p,
        "Energy_SolvH": sol_h,
        "Energy_VdW": vdw + clash,
    }


def parse_foldx_fxout(path: Path) -> Dict[str, float]:
    """
    Parse a FoldX ``.fxout`` file: prefer FoldX 5 ``*_ST.fxout`` single line; else header+data table.
    """
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in text:
        parsed = _try_parse_foldx5_stability_st_line(line)
        if parsed is not None:
            return parsed

    header_idx = None
    header_cells: Optional[List[str]] = None

    for i, line in enumerate(text):
        raw = line.strip()
        if not raw or raw.startswith("#") or raw.startswith("---"):
            continue
        if "\t" not in line and "\t" not in raw:
            # try multiple spaces as separator (older outputs)
            parts = re.split(r"\s{2,}|\t", line.strip())
        else:
            parts = line.strip().split("\t")
        if len(parts) < 4:
            continue
        norms = [_norm_header(p) for p in parts]
        if "pdb" in norms[0] or norms[0] == "pdb":
            header_cells = [p.strip() for p in parts]
            header_idx = i
            break
        # header sometimes starts with empty first cell
        if any("electrostatic" in n for n in norms) and any("vdw" in n or "waals" in n for n in norms):
            header_cells = [p.strip() for p in parts]
            header_idx = i
            break

    if header_cells is None or header_idx is None:
        raise ValueError(f"No tabular header found in {path}")

    for j in range(header_idx + 1, len(text)):
        line = text[j].strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        if "\t" in line:
            data_cells = line.split("\t")
        else:
            data_cells = re.split(r"\s{2,}", line.strip())
        if len(data_cells) < min(5, len(header_cells) // 2):
            continue
        # pad / trim to header length
        if len(data_cells) < len(header_cells):
            data_cells = data_cells + [""] * (len(header_cells) - len(data_cells))
        else:
            data_cells = data_cells[: len(header_cells)]
        terms = parse_foldx_tabular_row(header_cells, data_cells)
        if terms:
            return terms

    raise ValueError(f"No data row after header in {path}")


def foldx_fxout_to_physics_tensors(path: Path) -> Dict[str, Any]:
    import torch

    raw = parse_foldx_fxout(path)
    pia = foldx_terms_to_pia_targets(raw)
    return {k: torch.tensor(float(pia[k]), dtype=torch.float32) for k in PIA_PHYSICS_NAMES}


def load_physics_targets_csv(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a CSV keyed by ``sample_id`` with columns matching ``PIA_PHYSICS_NAMES``.

    Legacy CSVs with ``fa_elec``, ``fa_sol``, ``fa_atr``, ``fa_rep`` are still accepted
    (mapped into FoldX-aligned keys).
    """
    import torch

    path = Path(path)
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("sample_id") or row.get("PDB") or "").strip()
            if not sid:
                continue
            try:
                if all(c in row and row[c].strip() != "" for c in PIA_PHYSICS_NAMES):
                    out[sid] = {
                        k: torch.tensor(float(row[k]), dtype=torch.float32) for k in PIA_PHYSICS_NAMES
                    }
                elif all(c in row for c in ("fa_elec", "fa_sol", "fa_atr", "fa_rep")):
                    out[sid] = {
                        "Electro": torch.tensor(float(row["fa_elec"]), dtype=torch.float32),
                        "Energy_SolvP": torch.tensor(0.5 * float(row["fa_sol"]), dtype=torch.float32),
                        "Energy_SolvH": torch.tensor(0.5 * float(row["fa_sol"]), dtype=torch.float32),
                        "Energy_VdW": torch.tensor(
                            float(row["fa_atr"]) + float(row["fa_rep"]), dtype=torch.float32
                        ),
                    }
                else:
                    continue
            except (KeyError, ValueError):
                continue
    return out


def compute_physics_target_stats_from_csv(path: str | Path) -> tuple[dict[str, float], dict[str, float]]:
    """
    Population mean and std per ``PIA_PHYSICS_NAMES`` over all valid rows in ``physics_targets_csv``.

    Used to z-score physics auxiliary targets (and predictions) so MSE is not dominated by
    large-magnitude terms like solvation energies.
    """
    import torch

    rows = load_physics_targets_csv(path)
    if not rows:
        return {k: 0.0 for k in PIA_PHYSICS_NAMES}, {k: 1.0 for k in PIA_PHYSICS_NAMES}
    mu: dict[str, float] = {}
    std: dict[str, float] = {}
    for k in PIA_PHYSICS_NAMES:
        vals = torch.stack([rows[sid][k] for sid in rows])
        mu[k] = float(vals.mean().item())
        std[k] = float(vals.std(unbiased=False).clamp_min(1e-6).item())
    return mu, std
