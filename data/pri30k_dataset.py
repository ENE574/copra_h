from data.register import DataRegister
from torch.utils.data import Dataset
import pandas as pd
import esm
import torch
from rinalmo.data.constants import *
from rinalmo.data.alphabet import Alphabet
from tqdm import tqdm
import diskcache
import os
import math
import time
from data.transforms import get_transform
from torch.utils.data._utils.collate import default_collate
from typing import Optional, Dict
from easydict import EasyDict
from data.structure_dataset import _process_structure

na_alphabet_config = {
    "standard_tkns": RNA_TOKENS,
    "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
}

R = DataRegister()
# ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4
# ATOM_P, ATOM_C4, ATOM_NB = 37, 38, 
@R.register('pri30k_dataset')
class PRI30kDataset(Dataset):
    ''' 
    The implementation of Protein-RNA structure Dataset
    '''
    def __init__(self, 
                 dataframe, 
                 data_root, 
                 col_prot_name='PDB',
                 col_prot_chain='Protein chains',
                 col_na_chain='RNA chains',
                 col_binding_site='Binding site renumbered merged',
                 col_ligand='Binding ligands',
                 diskcache=None,
                 transform=None,
                 **kwargs
                 ):
        self.data_root = data_root
        self.df: pd.DataFrame = dataframe.copy()
        self.df.reset_index(drop=True, inplace=True)
        self.col_prot_name = col_prot_name
        self.col_prot_chain = col_prot_chain
        self.col_na_chain = col_na_chain
        self.col_binding_site = col_binding_site
        self.col_ligand = col_ligand
        self.diskcache = diskcache
        self.prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.na_alphabet = Alphabet(**na_alphabet_config)
        
        self.transform = get_transform(transform)
        
        # self.load_data()
        
    def load_data(self, idx):
        row = self.df.loc[idx]
        structure_id = row[self.col_prot_name]
        prot_chains = [row[self.col_prot_chain]]
        na_chains = [row[self.col_na_chain]]
        structure_id = structure_id + '_' + prot_chains[0] + '_' + na_chains[0]
        if self.diskcache is None or structure_id not in self.diskcache:
            structure_name = structure_id + '.cif'
            pdb_path = os.path.join(self.data_root, structure_name)
            
            cplx = _process_structure(pdb_path, structure_id, prot_chains, na_chains, gpu='cuda:0')
            
            ligand_id = row[self.col_prot_name] + '_' + row[self.col_na_chain]
            L = len(cplx['seq'])
            gpu_atoms = cplx['pos_heavyatom']
            gpu_masks = cplx['mask_heavyatom']
            distance_map = torch.linalg.norm(gpu_atoms[:, None, :, None, :]- gpu_atoms[None, :, None, :, :], dim=-1, ord=2).reshape(L, L, -1)
            mask = (gpu_masks[:, None, :, None] * gpu_masks[None, :, None, :]).reshape(L, L, -1)
            distance_map[~mask] = torch.inf
            atom_min_dist = torch.min(distance_map, dim=-1)[0]
            
            max_prot_length = 0
            max_na_length = 0
            for prot_seq in cplx.prot_seqs:
                if len(prot_seq) > max_prot_length:
                    max_prot_length = len(prot_seq)
            for na_seq in cplx.rna_seqs:
                if len(na_seq) > max_na_length:
                    max_na_length = len(na_seq)
    
            item = {
                'ligand_id': ligand_id, # no need to pad
                'atom_min_dist': atom_min_dist, # needs 2D padding
                'max_prot_length': max_prot_length, # will be ignored in batching
                'max_na_length': max_na_length, # will be ignored in batching
                'can_bind': row[self.col_ligand] # no need to pad
            }
            
            cplx.update(item)
            cplx["prot_chain_ids"] = prot_chains
            cplx["rna_chain_ids"] = na_chains
            if self.diskcache is not None:
                for key in cplx:
                    if isinstance(cplx[key], torch.Tensor):
                        cplx[key] = cplx[key].detach().cpu()
                self.diskcache[structure_id] = cplx
            return cplx
        else:
            data = self.diskcache[structure_id]
            # diskcache may contain old entries without these keys; reconstruct from csv row
            if "prot_chain_ids" not in data:
                data["prot_chain_ids"] = prot_chains
            if "rna_chain_ids" not in data:
                data["rna_chain_ids"] = na_chains
            return data

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.load_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

EXCLUDE_KEYS = []
DEFAULT_PAD_VALUES = {
    'restype': 26,
    'mask_atoms': 0,
    'chain_nb': -1,
    'identifier': -1
}

class PRI30kStructCollate(object):
    def __init__(self, strategy='separate', dataset_args=None, length_ref_key='restype', pad_values=DEFAULT_PAD_VALUES, exclude_keys=EXCLUDE_KEYS, eight=True):
        super().__init__()
        self.strategy = strategy
        self.dataset_args = dataset_args
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.exclude_keys = exclude_keys
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n - l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def pad_2d(self, x, n, value=10000):
        assert isinstance(x, torch.Tensor)
        assert x.shape[0] == x.shape[1]
        if x.size(0) == n and x.size(1) == n:
            return x
        pad_size_1 = [n - x.size(0)] + list(x.shape[1:])
        pad = torch.full(pad_size_1, fill_value=value).to(x)
        x_padded_1 = torch.cat([x, pad], dim=0)
        pad_size_2 = [x_padded_1.shape[0]] + [n - x_padded_1.size(1)] + list(x_padded_1.shape[2:])
        pad = torch.full(pad_size_2, fill_value=value).to(x)
        x_padded = torch.cat([x_padded_1, pad], dim=1)
        return x_padded
            
        

    def collate_complex(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys_inter = self._get_common_keys(data_list)
        keys = []
        keys_not_pad = []
        keys_ignore = [
            'prot_seqs', 'rna_seqs',
            'prot_chain_ids', 'rna_chain_ids',
            'max_prot_length', 'max_na_length',
            'can_bind', 'ligand_id', 'atom_min_dist'
        ]
        pad_2d = ['atom_min_dist']
        for key in keys_inter:
            if key in keys_ignore:
                continue
            elif key not in self.exclude_keys:
                keys.append(key)
            else:
                keys_not_pad.append(key)
    
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }

            for k in pad_2d:
                data_padded[k] = self.pad_2d(data[k], max_length)
                
            for k in keys_not_pad:
                data_padded[k] = data[k]
                
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return data_list_padded

    def pad_for_berts(self, batch):
        prot_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        na_alphabet = Alphabet(**na_alphabet_config)
        prot_chains = [len(item['prot_seqs']) for item in batch]
        na_chains = [len(item['rna_seqs']) for item in batch]
        max_item_prot_length = [item['max_prot_length'] for item in batch]
        max_item_na_length = [item['max_na_length'] for item in batch]
        max_prot_length = max(max_item_prot_length)
        max_na_length = max(max_item_na_length)
        total_prot_chains = sum(prot_chains)
        total_na_chains = sum(na_chains)
        if self.eight:
            max_prot_length = math.ceil((max_prot_length + 2) / 8) * 8
            max_na_length =  math.ceil((max_na_length + 2) / 8) * 8
        else:
            max_prot_length = max_prot_length + 2
            max_na_length = max_na_length + 2
        prot_batch = torch.empty([total_prot_chains, max_prot_length])
        prot_batch.fill_(prot_alphabet.padding_idx)
        na_batch = torch.empty([total_na_chains, max_na_length])
        na_batch.fill_(na_alphabet.pad_idx)
        curr_prot_idx = 0
        curr_na_idx = 0
        for item in batch:
            prot_seqs = item['prot_seqs']
            na_seqs = item['rna_seqs']
            for prot_seq in prot_seqs:
                prot_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                prot_seq_encode = prot_alphabet.encode(prot_seq)
                seq = torch.tensor(prot_seq_encode, dtype=torch.int64)
                prot_batch[curr_prot_idx, 1: len(prot_seq_encode)+1] = seq
                prot_batch[curr_prot_idx, len(prot_seq_encode)+1] = prot_alphabet.eos_idx
                curr_prot_idx += 1
            for na_seq in na_seqs:
                na_seq_encode = na_alphabet.encode(na_seq)
                seq = torch.tensor(na_seq_encode, dtype=torch.int64)
                na_batch[curr_na_idx, :len(seq)] = seq
                curr_na_idx += 1

        prot_mask = torch.zeros_like(prot_batch)
        na_mask = torch.zeros_like(na_batch)
        prot_mask[(prot_batch!=prot_alphabet.padding_idx) & (prot_batch!=prot_alphabet.eos_idx) & (prot_batch!=prot_alphabet.cls_idx)] = 1
        na_mask[(na_batch!=na_alphabet.pad_idx) & (na_batch!=na_alphabet.eos_idx) & (na_batch!=na_alphabet.cls_idx)] = 1
        return prot_batch.long(), prot_chains, prot_mask, na_batch.long(), na_chains, na_mask

    def gen_clip_label(self, ligand_ids, can_bind_info):
        clip_label = torch.eye(len(ligand_ids))
        for i, can_bind in enumerate(can_bind_info):
            candidates = can_bind.split(',')
            for j, ligand_id in enumerate(ligand_ids):
                if ligand_id in candidates:
                    clip_label[i, j] = 1
        return clip_label.long()
                    

    def __call__(self, data_list):
        data_list_padded = self.collate_complex(data_list)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        prot_batch, prot_chains, prot_mask, na_batch, na_chains, na_mask = self.pad_for_berts(data_list)
        batch['prot'] = prot_batch
        batch['prot_chains'] = prot_chains
        batch['protein_mask'] = prot_mask
        batch['na'] = na_batch
        batch['na_chains'] = na_chains
        batch['na_mask'] = na_mask
        batch['strategy'] = self.strategy
        ligand_ids =  [item['ligand_id'] for item in data_list]
        can_bind_info = [item['can_bind'] for item in data_list]
        batch['clip_label'] = self.gen_clip_label(ligand_ids, can_bind_info)

        use_offline = False
        offline_root = None
        if self.dataset_args is not None:
            use_offline = bool(getattr(self.dataset_args, "use_offline_embeddings", False))
            offline_root = getattr(self.dataset_args, "offline_embedding_root", None)

        if use_offline:
            if offline_root is None:
                raise ValueError("use_offline_embeddings=True but dataset_args.offline_embedding_root is not set")
            from utils.offline_embeddings import OfflineEmbeddingSpec, load_offline_token_embeddings

            spec = OfflineEmbeddingSpec(root=offline_root)
            max_prot_len = int(prot_batch.shape[1])
            max_rna_len = int(na_batch.shape[1])

            def _pad_into_token_space(x_residue, max_len):
                L = int(x_residue.shape[0])
                out = x_residue.new_zeros((max_len, int(x_residue.shape[1])))
                end = min(L, max_len - 2)
                if end > 0:
                    out[1:1 + end] = x_residue[:end]
                return out

            offline = {
                "prot_seq": {m: [] for m in spec.protein_sequence_models},
                "prot_struct": {m: [] for m in spec.protein_structure_models},
                "rna_seq": {m: [] for m in spec.rna_sequence_models},
                "rna_struct": {m: [] for m in spec.rna_structure_models},
            }

            for item in data_list:
                pdb_id = item["ligand_id"].split("_")[0]
                prot_chain_ids = item.get("prot_chain_ids", [])
                rna_chain_ids = item.get("rna_chain_ids", [])

                for chain_id in prot_chain_ids:
                    for m in spec.protein_sequence_models:
                        p = spec.file_path("protein_sequence", m, pdb_id, "prot", chain_id)
                        offline["prot_seq"][m].append(_pad_into_token_space(load_offline_token_embeddings(p), max_prot_len))
                    for m in spec.protein_structure_models:
                        p = spec.file_path("protein_structure", m, pdb_id, "prot", chain_id)
                        offline["prot_struct"][m].append(_pad_into_token_space(load_offline_token_embeddings(p), max_prot_len))

                for chain_id in rna_chain_ids:
                    for m in spec.rna_sequence_models:
                        p = spec.file_path("rna_sequence", m, pdb_id, "rna", chain_id)
                        offline["rna_seq"][m].append(_pad_into_token_space(load_offline_token_embeddings(p), max_rna_len))
                    for m in spec.rna_structure_models:
                        p = spec.file_path("rna_structure", m, pdb_id, "rna", chain_id)
                        offline["rna_struct"][m].append(_pad_into_token_space(load_offline_token_embeddings(p), max_rna_len))

            for group in offline:
                for m in offline[group]:
                    if len(offline[group][m]) == 0:
                        raise RuntimeError(
                            "Offline embedding list is empty for group='{}' model='{}'. "
                            "Likely missing prot_chain_ids/rna_chain_ids in cached samples, or wrong offline_embedding_root."
                            .format(group, m)
                        )
                    offline[group][m] = torch.stack(offline[group][m], dim=0)

            batch["use_offline_embeddings"] = True
            batch["offline_embeddings"] = offline

        return batch