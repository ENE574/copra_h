import numpy as np
import dataclasses
import sys
from pathlib import Path
import io
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass
sys.path.append('/home/CoPRA/')
import data.protein.proteins as proteins
from data.protein.atom_convert import atom37_to_atom14
from data.protein.proteins import chains_from_cif_string, chains_from_pdb_string
import data.rna.rnas as rnas
import data.dna.dnas as dnas

rna_residues = ['A', 'G', 'C', 'U']
dna_residues = ['DA', 'DC', 'DG', 'DT', 'DU', 'A', 'C', 'G', 'T']
na_residues = rna_residues + dna_residues
protein_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
SUPER_PROT_IDX = 27
SUPER_RNA_IDX = 28
SUPER_CPLX_IDX = 29
SUPER_CHAIN_IDX = 4
PADDING_NODE_IDX = 26

@dataclass
class ComplexInput:
    seq: str # L
    mask: np.ndarray # (L, )
    restype: np.ndarray # (L, ) # In total 21 + 5 = 26 types, including '_' and 'X'
    res_nb: np.ndarray # (L, )
    prot_seqs: list
    na_seqs: list
    atom_mask: np.ndarray # (L, 37 + 27)
    atom_positions: np.ndarray #(L, 37 + 27, 3)
    
    atom41_mask: np.ndarray # (L, 14 + 27)
    atom41_positions: np.ndarray #(L, 14 + 27, 3)
    
    identifier: np.ndarray #(L, ), to identify rna or protein
    chainid: np.ndarray # (L, ), to identify chain of the complex
    
    @classmethod
    def from_path(
        self,
        path,
        valid_rna_chains=None,
        valid_prot_chains=None,
        valid_partner_chains=None,
        entity_type="prot_na",
        na_entity_type="rna",
    ):
        if isinstance(path, io.IOBase):
            file_string = path.read()
        else:
            path = Path(path)
            file_string = path.read_text()
        if valid_prot_chains is None:
            valid_prot_chains = []
        if valid_rna_chains is None:
            valid_rna_chains = []
        if valid_partner_chains is None:
            valid_partner_chains = []
        if len(valid_prot_chains) == 0 and len(valid_rna_chains) == 0 and len(valid_partner_chains) == 0:
            if '.pdb' in str(path):
                chains = chains_from_pdb_string(file_string)
            elif '.cif' in str(path):
                chains = chains_from_cif_string(file_string)
                
            for chain in chains:
                for residue in chain:
                    resname = residue.get_resname()
                    if resname in protein_residues: 
                        valid_prot_chains.append(chain.get_full_id()[2])
                        break
                    if resname in na_residues:
                        valid_rna_chains.append(chain.get_full_id()[2])
                        break
        protein = proteins.ProteinInput.from_path(path, with_angles=False, return_dict=True, valid_chains=valid_prot_chains)
        if entity_type == "ppi":
            partner = proteins.ProteinInput.from_path(
                path, with_angles=False, return_dict=True, valid_chains=valid_partner_chains
            )
            prot_chains = [c for c in valid_prot_chains if c in protein]
            partner_chains = [c for c in valid_partner_chains if c in partner]
            if len(prot_chains) == 0 or len(partner_chains) == 0:
                return None
            complex_dict = complex_merge_ppi(
                [protein[chain] for chain in prot_chains],
                [partner[chain] for chain in partner_chains],
            )
        else:
            is_dna = na_entity_type.lower() == "dna"
            if is_dna:
                na = dnas.DNAInput.from_path(path, valid_rna_chains)
                na_chains = [c for c in valid_rna_chains if c in na]
            else:
                na = rnas.RNAInput.from_path(path, valid_rna_chains)
                na_chains = [c for c in valid_rna_chains if c in na]
            
            prot_chains = [c for c in valid_prot_chains if c in protein]
            if len(prot_chains) == 0 or len(na_chains) == 0:
                return None
            
            if is_dna:
                complex_dict = complex_merge_dna(
                    [protein[chain] for chain in prot_chains],
                    [na[chain] for chain in na_chains],
                )
            else:
                complex_dict = complex_merge(
                    [protein[chain] for chain in prot_chains],
                    [na[chain] for chain in na_chains],
                )
        return self(**complex_dict)
    
    @property
    def length(self):
        return len(self.seq)
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        texts = []
        texts += [f'seq: {self.seq}']
        texts += [f'length: {len(self.seq)}']
        texts += [f"mask: {''.join(self.mask.astype('int').astype('str'))}"]
        if self.chainid is not None:
            texts += [f"chainid: {''.join(self.chainid.astype('int').astype('str'))}"]
            texts += [f"identifier: {''.join(self.identifier.astype('int').astype('str'))}"]
        names = [
            'restype',
            'atom_mask',
            'atom_positions',
        ]
        for name in names:
            value = getattr(self, name)
            if value is None:
                text = f'{name}: None'
            else:
                text = f'{name}: {value.shape}'
            texts += [text]
        text = ', \n  '.join(texts)
        text = f'Protein-RNA Complex(\n  {text}\n)'
        return text
    
    
def complex_merge(protein, rna):
    assert len(protein) > 0
    assert len(rna) > 0

    p_lengths = [p.length for i, p in enumerate(protein)]
    r_lengths = [r.length for i, r in enumerate(rna)]
    lengths = p_lengths + r_lengths
    prot_list = []
    na_list = []
    seq = "".join([item.seq for item in protein + rna])        
    # identifier = np.array([0] * len(protein) + [1] * len(rna), dtype=np.int32)
    mask = np.concatenate([item.mask for item in protein + rna] )
    chain_arr = np.concatenate([[i] * p for i, p in enumerate(lengths)]).astype('int')
    res_nb = np.zeros([len(seq)])
    restype = np.zeros([len(seq)])
    atom_positions = np.zeros([len(seq), 37 + 27, 3])
    atom41_positions = np.zeros([len(seq), 14 + 27, 3])
    atom41_masks = np.zeros([len(seq), 14 + 27])
    atom_masks = np.zeros([len(seq), 37 + 27])
    identifier = np.zeros([len(seq)])
    curr_idx = 0
    for item in protein:
        prot_list.append(item.seq)
        res_nb[curr_idx: curr_idx+item.length] = item.res_nb
        restype[curr_idx: curr_idx+item.length] = item.aatype
        identifier[curr_idx: curr_idx+item.length] = 0
        atom_positions[curr_idx: curr_idx+item.length, :37, :] = item.atom_positions
        atom14, mask_14, arrs = atom37_to_atom14(item.aatype, item.atom_positions, [item.atom_mask])
        mask_14 = arrs[0] * mask_14
        atom41_positions[curr_idx: curr_idx+item.length, :14, :] = atom14
        atom41_masks[curr_idx: curr_idx+item.length, :14] = mask_14
        atom_masks[curr_idx: curr_idx+item.length, :37] = item.atom_mask
        curr_idx += item.length
    for item in rna:
        na_list.append(item.seq)
        res_nb[curr_idx: curr_idx+item.length] = item.res_nb
        restype[curr_idx: curr_idx+item.length] = item.basetype + 21
        identifier[curr_idx: curr_idx+item.length] = 1
        atom_positions[curr_idx: curr_idx+item.length, 37:, :] = item.atom_positions
        atom41_positions[curr_idx: curr_idx+item.length, 14:, :] = item.atom_positions
        atom41_masks[curr_idx: curr_idx+item.length, 14:] = item.atom_mask
        atom_masks[curr_idx: curr_idx+item.length, 37:] = item.atom_mask
        curr_idx += item.length

    complex_dict = {
        'seq': seq,
        'mask': mask,
        'restype': restype,
        'res_nb': res_nb,
        
        'prot_seqs': prot_list,
        'na_seqs': na_list,

        'atom_mask': atom_masks,
        'atom_positions': atom_positions,
        
        'atom41_mask': atom41_masks,
        'atom41_positions': atom41_positions,

        'identifier': identifier,
        'chainid': chain_arr
    }
    
    return complex_dict


def complex_merge_ppi(protein_a, protein_b):
    """Merge two protein entities; partner (B) uses identifier=1 like the NA branch."""
    assert len(protein_a) > 0
    assert len(protein_b) > 0

    a_lengths = [p.length for p in protein_a]
    b_lengths = [p.length for p in protein_b]
    lengths = a_lengths + b_lengths
    prot_list = []
    na_list = []
    seq = "".join([item.seq for item in protein_a + protein_b])
    mask = np.concatenate([item.mask for item in protein_a + protein_b])
    chain_arr = np.concatenate([[i] * p for i, p in enumerate(lengths)]).astype('int')
    res_nb = np.zeros([len(seq)])
    restype = np.zeros([len(seq)])
    atom_positions = np.zeros([len(seq), 37 + 27, 3])
    atom41_positions = np.zeros([len(seq), 14 + 27, 3])
    atom41_masks = np.zeros([len(seq), 14 + 27])
    atom_masks = np.zeros([len(seq), 37 + 27])
    identifier = np.zeros([len(seq)])
    curr_idx = 0

    def _append_protein_chain(item, entity_id, store_in_prot_list):
        nonlocal curr_idx
        if store_in_prot_list:
            prot_list.append(item.seq)
        else:
            na_list.append(item.seq)
        res_nb[curr_idx: curr_idx + item.length] = item.res_nb
        restype[curr_idx: curr_idx + item.length] = item.aatype
        identifier[curr_idx: curr_idx + item.length] = entity_id
        atom_positions[curr_idx: curr_idx + item.length, :37, :] = item.atom_positions
        atom14, mask_14, arrs = atom37_to_atom14(item.aatype, item.atom_positions, [item.atom_mask])
        mask_14 = arrs[0] * mask_14
        atom41_positions[curr_idx: curr_idx + item.length, :14, :] = atom14
        atom41_masks[curr_idx: curr_idx + item.length, :14] = mask_14
        atom_masks[curr_idx: curr_idx + item.length, :37] = item.atom_mask
        curr_idx += item.length

    for item in protein_a:
        _append_protein_chain(item, 0, store_in_prot_list=True)
    for item in protein_b:
        _append_protein_chain(item, 1, store_in_prot_list=False)

    return {
        'seq': seq,
        'mask': mask,
        'restype': restype,
        'res_nb': res_nb,
        'prot_seqs': prot_list,
        'na_seqs': na_list,
        'atom_mask': atom_masks,
        'atom_positions': atom_positions,
        'atom41_mask': atom41_masks,
        'atom41_positions': atom41_positions,
        'identifier': identifier,
        'chainid': chain_arr,
    }


def complex_merge_dna(protein, dna):
    """Merge protein and DNA entities."""
    assert len(protein) > 0
    assert len(dna) > 0

    p_lengths = [p.length for i, p in enumerate(protein)]
    d_lengths = [d.length for i, d in enumerate(dna)]
    lengths = p_lengths + d_lengths
    prot_list = []
    na_list = []
    seq = "".join([item.seq for item in protein + dna])        
    mask = np.concatenate([item.mask for item in protein + dna] )
    chain_arr = np.concatenate([[i] * p for i, p in enumerate(lengths)]).astype('int')
    res_nb = np.zeros([len(seq)])
    restype = np.zeros([len(seq)])
    atom_positions = np.zeros([len(seq), 37 + 27, 3])
    atom41_positions = np.zeros([len(seq), 14 + 27, 3])
    atom41_masks = np.zeros([len(seq), 14 + 27])
    atom_masks = np.zeros([len(seq), 37 + 27])
    identifier = np.zeros([len(seq)])
    curr_idx = 0
    for item in protein:
        prot_list.append(item.seq)
        res_nb[curr_idx: curr_idx+item.length] = item.res_nb
        restype[curr_idx: curr_idx+item.length] = item.aatype
        identifier[curr_idx: curr_idx+item.length] = 0
        atom_positions[curr_idx: curr_idx+item.length, :37, :] = item.atom_positions
        atom14, mask_14, arrs = atom37_to_atom14(item.aatype, item.atom_positions, [item.atom_mask])
        mask_14 = arrs[0] * mask_14
        atom41_positions[curr_idx: curr_idx+item.length, :14, :] = atom14
        atom41_masks[curr_idx: curr_idx+item.length, :14] = mask_14
        atom_masks[curr_idx: curr_idx+item.length, :37] = item.atom_mask
        curr_idx += item.length
    for item in dna:
        na_list.append(item.seq)
        res_nb[curr_idx: curr_idx+item.length] = item.res_nb
        restype[curr_idx: curr_idx+item.length] = item.basetype + 21
        identifier[curr_idx: curr_idx+item.length] = 1
        atom_positions[curr_idx: curr_idx+item.length, 37:, :] = item.atom_positions
        atom41_positions[curr_idx: curr_idx+item.length, 14:, :] = item.atom_positions
        atom41_masks[curr_idx: curr_idx+item.length, 14:] = item.atom_mask
        atom_masks[curr_idx: curr_idx+item.length, 37:] = item.atom_mask
        curr_idx += item.length

    return {
        'seq': seq,
        'mask': mask,
        'restype': restype,
        'res_nb': res_nb,
        'prot_seqs': prot_list,
        'na_seqs': na_list,
        'atom_mask': atom_masks,
        'atom_positions': atom_positions,
        'atom41_mask': atom41_masks,
        'atom41_positions': atom41_positions,
        'identifier': identifier,
        'chainid': chain_arr,
    }


if __name__ == '__main__':
    comp = ComplexInput.from_path('./datasets/PRA310/PDBs/1RPU.pdb')
    print("Complex:", comp)
    print(comp.atom_positions[1])