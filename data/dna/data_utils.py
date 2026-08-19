import os
import numpy as np
import pandas as pd
from typing import Any, List, Literal
import torch
import cpdb
from collections import defaultdict
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import pandas as pd

from .base_constants import (
    DNA_ATOMS, 
    DNA_NUCLEOTIDES, 
    PURINES,
    PYRIMIDINES,
    FILL_VALUE
)


_DNA_RESIDUE_TO_NT = {
    "DA": "A",
    "DC": "C",
    "DG": "G",
    "DT": "T",
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
}


def normalize_dna_residue(name: str) -> str:
    """Map DNA residue names (DA/DC/DG/DT) to DNA alphabet (A/C/G/T)."""
    name = str(name).strip().upper()
    if name in _DNA_RESIDUE_TO_NT:
        return _DNA_RESIDUE_TO_NT[name]
    return name


def pdb_to_array_dna(
        filepath: str, 
        valid_chains=None,
        return_sec_struct: bool = True,
        return_sasa: bool = True,
        keep_insertions: bool = False, 
        keep_pseudoknots: bool = False
    ):
    df = cpdb.parse(filepath, df=True)

    if not keep_insertions:
        df = remove_insertions(df)
    dnas = defaultdict(dict)
    if valid_chains is None:
        valid_chains = list(df['chain_id'].unique())
    for chain in valid_chains:
        sub_df = df[df['chain_id']==chain]
        sequence, res_nb, coords, coord_mask, seqtype, seq_mask = chain_to_array_dna(sub_df, keep_insertions)
        dnas[chain]['res_nb'] = res_nb
        dnas[chain]['seq'] = sequence
        dnas[chain]['atom_positions'] = coords
        dnas[chain]['atom_mask'] = coord_mask
        dnas[chain]['basetype'] = seqtype
        dnas[chain]['mask'] = seq_mask
    return dnas


def chain_to_array_dna(
    df: pd.DataFrame,
    keep_insertions: bool = True,
):
    df["residue_id"] = (
        df["chain_id"]
        + ":"
        + df["residue_name"]
        + ":"
        + df["residue_number"].astype(str)
    )
    if keep_insertions:
        df["residue_id"] = df.residue_id + ":" + df.insertion
    df = df[df['residue_name'] != 'HOH']
    df = df[df['record_name'] != 'HETATM']
    
    nt_list = [normalize_dna_residue(res.split(":")[1]) for res in df.residue_id.unique()]
    nt_list = [nt if nt in DNA_NUCLEOTIDES else "_" for nt in nt_list]
    res_nb = np.array([int(res.split(":")[2]) for res in df.residue_id.unique()])

    seq_type = []
    seq_mask = []
    for nt in nt_list:
        if nt in DNA_NUCLEOTIDES:
            seq_type.append(DNA_NUCLEOTIDES.index(nt))
            seq_mask.append(1)
        else:
            seq_type.append(len(DNA_NUCLEOTIDES))
            seq_mask.append(0)
    sequence = "".join(nt_list)
    seq_type = np.array(seq_type, dtype=np.int32)
    seq_mask = np.array(seq_mask, dtype=np.bool_)

    coords, coord_mask = df_to_array_dna(df, center=False)
    assert coords.shape[0] == len(sequence), "Sequence and coordinates must be the same length"

    return sequence, res_nb, coords, coord_mask, seq_type, seq_mask


def df_to_array_dna(
    df: pd.DataFrame,
    atoms_to_keep: List[str] = DNA_ATOMS,
    fill_value: float = FILL_VALUE,
    center: bool = True
):
    if center:
        df.x_coord -= df.x_coord.mean()
        df.y_coord -= df.y_coord.mean()
        df.z_coord -= df.z_coord.mean()

    num_residues = len([res.split(":")[1] for res in df.residue_id.unique()])
    df = df.loc[df["atom_name"].isin(atoms_to_keep)]
    residue_indices = pd.factorize(np.array(df.residue_id))[0]
    atom_indices = df["atom_name"].map(lambda x: atoms_to_keep.index(x)).values.astype(np.int32)
    positions = (
        np.zeros((num_residues, len(atoms_to_keep), 3), dtype=np.float32) + fill_value
    )
    mask = np.zeros((num_residues, len(atoms_to_keep)), dtype=np.bool_)
    positions[residue_indices, atom_indices] = np.array(
        df[["x_coord", "y_coord", "z_coord"]].values, dtype=np.float32)
    mask[residue_indices, atom_indices] = 1
    return positions, mask


def cif_to_array_dna(
        filepath: str, 
        valid_chains=None,
        return_sec_struct: bool = True,
        return_sasa: bool = True,
        keep_insertions: bool = False, 
        keep_pseudoknots: bool = False
    ):
    dico = MMCIF2Dict(filepath)
    df = pd.DataFrame.from_dict(dico, orient='index')
    df = df.transpose()

    dnas = defaultdict(dict)
    new_df_dict = defaultdict(list)
    if valid_chains is None:
        valid_chains = list(df['_atom_site.auth_asym_id'].unique())
    
    for chain in valid_chains:
        chain_mask = df["_atom_site.auth_asym_id"] == chain
        chain_df = df[chain_mask].copy()
        
        new_df_dict = {
            'chain_id': chain_df["_atom_site.auth_asym_id"].values,
            'residue_name': chain_df["_atom_site.label_comp_id"].values,
            'residue_number': chain_df["_atom_site.auth_seq_id"].values,
            'atom_name': chain_df["_atom_site.label_atom_id"].values,
            'x_coord': np.array([float(x) for x in chain_df["_atom_site.Cartn_x"].values]),
            'y_coord': np.array([float(x) for x in chain_df["_atom_site.Cartn_y"].values]),
            'z_coord': np.array([float(x) for x in chain_df["_atom_site.Cartn_z"].values]),
            'insertion': [' '] * len(chain_df),
        }
        
        new_df = pd.DataFrame(new_df_dict)
        new_df["residue_id"] = (
            new_df["chain_id"]
            + ":"
            + new_df["residue_name"]
            + ":"
            + new_df["residue_number"].astype(str)
        )
        
        nt_list = [normalize_dna_residue(res.split(":")[1]) for res in new_df.residue_id.unique()]
        nt_list = [nt if nt in DNA_NUCLEOTIDES else "_" for nt in nt_list]
        res_nb = np.array([int(res.split(":")[2]) for res in new_df.residue_id.unique()])

        seq_type = []
        seq_mask = []
        for nt in nt_list:
            if nt in DNA_NUCLEOTIDES:
                seq_type.append(DNA_NUCLEOTIDES.index(nt))
                seq_mask.append(1)
            else:
                seq_type.append(len(DNA_NUCLEOTIDES))
                seq_mask.append(0)
        sequence = "".join(nt_list)
        seq_type = np.array(seq_type, dtype=np.int32)
        seq_mask = np.array(seq_mask, dtype=np.bool_)

        coords, coord_mask = df_to_array_dna(new_df, center=False)
        
        dnas[chain]['res_nb'] = res_nb
        dnas[chain]['seq'] = sequence
        dnas[chain]['atom_positions'] = coords
        dnas[chain]['atom_mask'] = coord_mask
        dnas[chain]['basetype'] = seq_type
        dnas[chain]['mask'] = seq_mask
    
    return dnas


def remove_insertions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["insertion"] == " "]