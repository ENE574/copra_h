from data.complex import ComplexInput
from data.register import DataRegister
from torch.utils.data import Dataset
import pandas as pd
from esm.data import Alphabet as ESMAlphabet
import torch
from utils.torch_compat import torch_load_compat
from tqdm import tqdm
from rinalmo.data.constants import *
from rinalmo.data.alphabet import Alphabet
from tqdm import tqdm
import os
import math
from data.transforms import get_transform
from torch.utils.data._utils.collate import default_collate
from typing import Optional, Dict
from easydict import EasyDict
from data.protein.residue_constants import restype_order, restype_num
from data.rna.base_constants import RNA_NUCLEOTIDES

na_alphabet_config = {
    "standard_tkns": RNA_TOKENS,
    "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
}

R = DataRegister()
# ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4
# ATOM_P, ATOM_C4, ATOM_NB = 37, 38, 

def _process_structure(structure_path, structure_id, valid_prot_chains=None, valid_rna_chains=None, gpu=None) -> Optional[Dict]:
    cplx = ComplexInput.from_path(structure_path, valid_prot_chains=valid_prot_chains, valid_rna_chains=valid_rna_chains)
    if cplx is None:
        print(f'[INFO] Failed to parse structure. Too few valid residues: {structure_path}')
        return None

    data = EasyDict({
        'seq': cplx.seq, 'prot_seqs': cplx.prot_seqs, 'rna_seqs': cplx.na_seqs, 'res_nb': torch.LongTensor(cplx.res_nb),
        'chain_nb': torch.LongTensor(cplx.chainid), 'identifier': torch.LongTensor(cplx.identifier),
        'restype': torch.LongTensor(cplx.restype), 'seq_mask': torch.BoolTensor(cplx.mask),
        'pos_heavyatom': torch.FloatTensor(cplx.atom41_positions), 'mask_heavyatom': torch.BoolTensor(cplx.atom41_mask),
        'atom64_positions': torch.FloatTensor(cplx.atom_positions), 'atom64_mask': torch.BoolTensor(cplx.atom_mask), 
    })
    if gpu is not None:
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(gpu)
    data['id'] = structure_id
    return data


def safe_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in name)


@R.register('structure_dataset')
class StructureDataset(Dataset):
    ''' 
    The implementation of Protein-RNA structure Dataset
    '''
    def __init__(self, 
                 dataframe, 
                 data_root, 
                 col_prot_name='PDB',
                 col_prot_chain='Protein chains',
                 col_na_chain='RNA chains',
                 col_prot='Protein sequences',
                 col_na='RNA sequences',
                 col_label='△G(kcal/mol)',
                 diskcache=None,
                 transform=None,
                 mut=False,
                 col_mut='Mutation sequences',
                 use_precomputed_embeddings: bool = False,
                 embedding_root: str = "outputs/feature_extraction",
                 seq_prot_models: Optional[list] = None,
                 seq_rna_models: Optional[list] = None,
                 str_prot_models: Optional[list] = None,
                 str_rna_models: Optional[list] = None,
                 protein_embedding_model: str = "esm2",
                 rna_embedding_model: str = "rinalmo",
                 embedding_strict: bool = True,
                 use_pyrosetta_physics: bool = False,
                 pyrosetta_scorefxn: str = "ref2015",
                 pyrosetta_init_extra: str = "",
                 physics_targets_csv: Optional[str] = None,
                 **kwargs
                 ):
        self.data_root = data_root
        self.df: pd.DataFrame = dataframe.copy()
        self.col_prot_name = col_prot_name
        self.col_prot_chain = col_prot_chain
        self.col_na_chain = col_na_chain
        self.col_label = col_label
        self.col_prot = col_prot
        self.col_na = col_na
        self.type = 'reg'
        self.diskcache = diskcache
        self.prot_alphabet = ESMAlphabet.from_architecture("ESM-1b")
        self.na_alphabet = Alphabet(**na_alphabet_config)
        self.mut = mut
        self.col_mut = col_mut
        def _ensure_list(value, fallback):
            if value is None:
                return list(fallback)
            if isinstance(value, str):
                return [value]
            return list(value)

        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.embedding_root = embedding_root
        self.protein_embedding_model = protein_embedding_model
        self.rna_embedding_model = rna_embedding_model
        self.seq_prot_models = _ensure_list(seq_prot_models, [protein_embedding_model])
        self.seq_rna_models = _ensure_list(seq_rna_models, [rna_embedding_model])
        self.str_prot_models = _ensure_list(str_prot_models, ["esm_if1", "protrek", "protbert"])
        self.str_rna_models = _ensure_list(str_rna_models, ["rna_ernie", "rnabert", "rhofold"])
        self.embedding_strict = embedding_strict
        self.use_pyrosetta_physics = bool(use_pyrosetta_physics)
        self.pyrosetta_scorefxn = str(pyrosetta_scorefxn)
        self.pyrosetta_init_extra = str(pyrosetta_init_extra)
        self._pyrosetta_memo: dict = {}
        self.physics_targets_csv = physics_targets_csv
        self._csv_physics_by_id: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if self.physics_targets_csv:
            if not os.path.isfile(self.physics_targets_csv):
                raise FileNotFoundError(f"physics_targets_csv not found: {self.physics_targets_csv}")
            from data.foldx_physics import load_physics_targets_csv

            self._csv_physics_by_id = load_physics_targets_csv(self.physics_targets_csv)
            if self.use_pyrosetta_physics:
                print(
                    "[INFO] physics_targets_csv is set; using CSV labels and skipping PyRosetta scoring."
                )
            self.use_pyrosetta_physics = False

        if self.use_pyrosetta_physics:
            from data.pyrosetta_physics import init_pyrosetta, pyrosetta_available

            if not pyrosetta_available():
                raise ImportError(
                    "use_pyrosetta_physics=True but PyRosetta is not importable. "
                    "Install PyRosetta in this environment (https://www.pyrosetta.org/downloads)."
                )
            init_pyrosetta(self.pyrosetta_init_extra)
        elif self._csv_physics_by_id is None:
            from data.pyrosetta_physics import warn_zero_pia_targets_when_no_pyrosetta

            warn_zero_pia_targets_when_no_pyrosetta()

        self.transform = get_transform(transform)

        self.load_data()

    def _resolve_pdb_path(self, structure_id: str) -> str:
        if self.mut:
            return os.path.join(self.data_root, structure_id.split("_")[0] + ".pdb")
        return os.path.join(self.data_root, f"{structure_id}.pdb")

    def load_data(self):
        self.data = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            structure_id = row[self.col_prot_name]
            complex = row[self.col_prot_name]
            if self.mut:
                structure_id += '_' + row['MUTATION']
            if self.diskcache is None or structure_id not in self.diskcache:
                prot_chains = [c.strip() for c in row[self.col_prot_chain].split(',')]
                na_chains = [c.strip() for c in row[self.col_na_chain].split(',')]
                pdb_path = self._resolve_pdb_path(structure_id)

                label = float(row[self.col_label])
                
                cplx = _process_structure(pdb_path, structure_id, prot_chains, na_chains)

                if self.mut:
                    prot_mut = row[self.col_mut]
                    mut_list = prot_mut.split(',')
                    mut_list_to_type = []
                    for mut_seq in mut_list:
                        mut_seq = mut_seq[2:]
                        for res in mut_seq:
                            restype_idx = restype_order.get(res, restype_num)
                            mut_list_to_type.append(restype_idx)
                    na = row[self.col_na]
                    na_list = na.split(',')
                    for na_seq in na_list:
                        na_seq = na_seq[2:]
                        for na in na_seq:
                            if na in RNA_NUCLEOTIDES:
                                na_idx = RNA_NUCLEOTIDES.index(na) + 21
                            else:
                                na_idx = len(RNA_NUCLEOTIDES) + 21
                            mut_list_to_type.append(na_idx)
                    mut_seqs = [i[2:] for i in mut_list]
                    mut_restype = torch.tensor(mut_list_to_type, device=cplx['restype'].device)
                    mut_identifier = mut_restype != cplx['restype']
                    assert len(mut_restype) == len(cplx['restype'])
                    if (mut_restype != cplx['restype']).sum().item() != 1:
                        print('Name:', structure_id)
                        print("Mut:", mut_restype)
                        print("Wild:", cplx['restype'])
                        print("Diff:", (mut_restype != cplx['restype']).sum())
                L = len(cplx['seq'])
                gpu_atoms = cplx['pos_heavyatom']
                gpu_masks = cplx['mask_heavyatom']
                distance_map = torch.linalg.norm(gpu_atoms[:, None, :, None, :]- gpu_atoms[None, :, None, :, :], dim=-1, ord=2).reshape(L, L, -1)
                mask = (gpu_masks[:, None, :, None] * gpu_masks[None, :, None, :]).reshape(L, L, -1)
                distance_map[~mask] = torch.inf
                atom_min_dist = torch.min(distance_map, dim=-1)[0]

                if self._csv_physics_by_id is not None:
                    row = self._csv_physics_by_id.get(structure_id)
                    if row is None:
                        from data.pyrosetta_physics import zero_physics_targets

                        physics_targets = zero_physics_targets()
                    else:
                        physics_targets = {k: v.clone() for k, v in row.items()}
                elif self.use_pyrosetta_physics:
                    from data.pyrosetta_physics import compute_physics_targets_tensor_cached

                    physics_targets = compute_physics_targets_tensor_cached(
                        pdb_path,
                        self._pyrosetta_memo,
                        scorefxn_name=self.pyrosetta_scorefxn,
                        init_extra_options=self.pyrosetta_init_extra,
                    )
                else:
                    from data.pyrosetta_physics import zero_physics_targets

                    physics_targets = zero_physics_targets()

                max_prot_length = 0
                max_na_length = 0
                for prot_seq in cplx.prot_seqs:
                    if len(prot_seq) > max_prot_length:
                        max_prot_length = len(prot_seq)
                for na_seq in cplx.rna_seqs:
                    if len(na_seq) > max_na_length:
                        max_na_length = len(na_seq)
                if self.mut:
                    item = {
                        'complex': complex,
                        'labels': label,
                        'atom_min_dist': atom_min_dist, # needs 2D padding
                        'max_prot_length': max_prot_length,
                        'max_na_length': max_na_length,
                        'mut_seqs': mut_seqs,
                        'mut_restype': mut_restype,
                        'mut_identifier': mut_identifier,
                        'physics_targets': physics_targets
                }
                else:
                    item = {
                        'complex': complex,
                        'labels': label,
                        'atom_min_dist': atom_min_dist, # needs 2D padding
                        'max_prot_length': max_prot_length,
                        'max_na_length': max_na_length,
                        'physics_targets': physics_targets
                    }
                item['prot_chain_ids'] = prot_chains
                item['rna_chain_ids'] = na_chains
                if self.use_precomputed_embeddings:
                    item['use_precomputed_embeddings'] = True
                    item['embedding_root'] = str(self.embedding_root)
                    item['seq_prot_models'] = list(self.seq_prot_models)
                    item['seq_rna_models'] = list(self.seq_rna_models)
                    item['str_prot_models'] = list(self.str_prot_models)
                    item['str_rna_models'] = list(self.str_rna_models)
                    item['embedding_strict'] = self.embedding_strict
                    
                cplx.update(item)
                # Keep chain ids so collate/model can map to offline embeddings:
                # offline files are named like {PDB}_prot_{chain}.pt and {PDB}_rna_{chain}.pt
                cplx["prot_chain_ids"] = prot_chains
                cplx["rna_chain_ids"] = na_chains
                # print("Complex {} is:".format(i), cplx)
                self.data.append(cplx)
                if self.diskcache is not None:
                    self.diskcache[structure_id] = cplx
            
            else:
                data = self.diskcache[structure_id]
                data['complex'] = complex
                # diskcache may contain old entries without these keys; reconstruct from csv row
                try:
                    data["prot_chain_ids"] = row[self.col_prot_chain].split(',')
                    data["rna_chain_ids"] = row[self.col_na_chain].split(',')
                except Exception:
                    # keep backward compatibility if columns are missing
                    if "prot_chain_ids" not in data:
                        data["prot_chain_ids"] = []
                    if "rna_chain_ids" not in data:
                        data["rna_chain_ids"] = []
                # Always refresh label from the current CSV row (diskcache may hold old dtypes).
                data["labels"] = float(row[self.col_label])
                if self._csv_physics_by_id is not None:
                    row = self._csv_physics_by_id.get(structure_id)
                    if row is None:
                        from data.pyrosetta_physics import zero_physics_targets

                        data["physics_targets"] = zero_physics_targets()
                    else:
                        data["physics_targets"] = {k: v.clone() for k, v in row.items()}
                elif self.use_pyrosetta_physics:
                    from data.pyrosetta_physics import compute_physics_targets_tensor_cached

                    pdb_path = self._resolve_pdb_path(structure_id)
                    data["physics_targets"] = compute_physics_targets_tensor_cached(
                        pdb_path,
                        self._pyrosetta_memo,
                        scorefxn_name=self.pyrosetta_scorefxn,
                        init_extra_options=self.pyrosetta_init_extra,
                    )
                else:
                    from data.pyrosetta_physics import zero_physics_targets

                    data["physics_targets"] = zero_physics_targets()
                self.data.append(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        # print("Before Transform:", data)
        if self.transform is not None:
            data = self.transform(data)
        return data

EXCLUDE_KEYS = ['labels', 'complex']
DEFAULT_PAD_VALUES = {
    'restype': 26,
    'mask_atoms': 0,
    'chain_nb': -1,
}

class CustomStructCollate(object):
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
    def _pad_embedding(token_embeddings, seq_len, max_len):
        dim = token_embeddings.shape[-1]
        padded = torch.zeros([max_len, dim], dtype=token_embeddings.dtype)
        take = min(seq_len, token_embeddings.shape[0])
        padded[1:1 + take] = token_embeddings[:take]
        return padded

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

    def collate_complex(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys_inter = self._get_common_keys(data_list)
        keys = []
        keys_not_pad = []
        keys_ignore = [
            'prot_seqs', 'rna_seqs', 'mut_seqs',
            'prot_chain_ids', 'rna_chain_ids',
            'max_prot_length', 'max_na_length', 'atom_min_dist'
        ]
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
            for k in keys_not_pad:
                data_padded[k] = data[k]
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return data_list_padded

    def pad_for_berts(self, strategy, batch):
        prot_alphabet = ESMAlphabet.from_architecture("ESM-1b")
        na_alphabet = Alphabet(**na_alphabet_config)
        mut_flag = 0
        prot_chains = [len(item['prot_seqs']) for item in batch]
        na_chains = [len(item['rna_seqs']) for item in batch]
        use_precomputed = bool(batch[0].get("use_precomputed_embeddings", False))
        if use_precomputed:
            emb_root = Path(batch[0].get("embedding_root", "outputs/feature_extraction"))
            seq_prot_models = list(batch[0].get("seq_prot_models", ["esm2"]))
            seq_rna_models = list(batch[0].get("seq_rna_models", ["rinalmo"]))
            str_prot_models = list(batch[0].get("str_prot_models", ["esm_if1", "protrek", "protbert"]))
            str_rna_models = list(batch[0].get("str_rna_models", ["rna_ernie", "rnabert", "rhofold"]))
            strict = bool(batch[0].get("embedding_strict", True))
            seq_prot_embeds = {name: [] for name in seq_prot_models}
            seq_rna_embeds = {name: [] for name in seq_rna_models}
            str_prot_embeds = {name: [] for name in str_prot_models}
            str_rna_embeds = {name: [] for name in str_rna_models}

            def _load_embedding(base_dir, name, seq_len, max_len, model_name):
                path = emb_root / base_dir / model_name / f"{name}.pt"
                if not path.exists():
                    if strict:
                        raise FileNotFoundError(f"Missing embedding: {path}")
                    return None
                payload = torch_load_compat(path, map_location="cpu")
                token_embeddings = payload["token_embeddings"]
                if token_embeddings.shape[0] != seq_len and strict:
                    raise ValueError(
                        f"Embedding length mismatch for {path}: "
                        f"expected {seq_len}, got {token_embeddings.shape[0]}"
                    )
                return self._pad_embedding(token_embeddings, seq_len, max_len)
        
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
        if 'mut_seqs' in batch[0]:
            mut_flag = 1
            mut_batch = torch.empty([total_prot_chains, max_prot_length])
            mut_batch.fill_(prot_alphabet.padding_idx)
        na_batch = torch.empty([total_na_chains, max_na_length])
        na_batch.fill_(na_alphabet.pad_idx)
        curr_prot_idx = 0
        curr_na_idx = 0
        for item in batch:
            prot_seqs = item['prot_seqs']
            if 'mut_seqs' in item:
                mut_seqs = item['mut_seqs']
            na_seqs = item['rna_seqs']
            prot_chain_ids = item.get('prot_chain_ids', [])
            rna_chain_ids = item.get('rna_chain_ids', [])
            complex_id = item.get('complex', '')
            for i, prot_seq in enumerate(prot_seqs):
                prot_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                prot_seq_encode = prot_alphabet.encode(prot_seq)
                seq = torch.tensor(prot_seq_encode, dtype=torch.int64)
                prot_batch[curr_prot_idx, 1: len(prot_seq_encode)+1] = seq
                prot_batch[curr_prot_idx, len(prot_seq_encode)+1] = prot_alphabet.eos_idx
                if 'mut_seqs' in item:
                    mut_batch[curr_prot_idx, 0] = prot_alphabet.cls_idx
                    mut_seq_encode = prot_alphabet.encode(mut_seqs[i])
                    seq_m = torch.tensor(mut_seq_encode, dtype=torch.int64)
                    mut_batch[curr_prot_idx, 1: len(mut_seq_encode)+1] = seq_m
                    mut_batch[curr_prot_idx, len(mut_seq_encode)+1] = prot_alphabet.eos_idx
                if use_precomputed:
                    chain_id = prot_chain_ids[i] if i < len(prot_chain_ids) else 'X'
                    name = safe_name(f"{complex_id}_prot_{chain_id or 'X'}")
                    for model_name in seq_prot_models:
                        emb = _load_embedding("protein_sequence", name, len(prot_seq), max_prot_length, model_name)
                        if emb is not None:
                            seq_prot_embeds[model_name].append(emb)
                    for model_name in str_prot_models:
                        emb = _load_embedding("protein_structure", name, len(prot_seq), max_prot_length, model_name)
                        if emb is not None:
                            str_prot_embeds[model_name].append(emb)
                curr_prot_idx += 1
            for j, na_seq in enumerate(na_seqs):
                # na_batch[curr_na_idx, 0] = na_alphabet.cls_idx
                # NA encoder adds CLS and EOS by default
                na_seq_encode = na_alphabet.encode(na_seq)
                seq = torch.tensor(na_seq_encode, dtype=torch.int64)
                na_batch[curr_na_idx, :len(seq)] = seq
                # na_batch[curr_na_idx, len(na_seq_encode)+1] = na_alphabet.eos_idx
                if use_precomputed:
                    chain_id = rna_chain_ids[j] if j < len(rna_chain_ids) else 'X'
                    name = safe_name(f"{complex_id}_rna_{chain_id or 'X'}")
                    for model_name in seq_rna_models:
                        emb = _load_embedding("rna_sequence", name, len(na_seq), max_na_length, model_name)
                        if emb is not None:
                            seq_rna_embeds[model_name].append(emb)
                    for model_name in str_rna_models:
                        emb = _load_embedding("rna_structure", name, len(na_seq), max_na_length, model_name)
                        if emb is not None:
                            str_rna_embeds[model_name].append(emb)
                curr_na_idx += 1
        prot_mask = torch.zeros_like(prot_batch)
        na_mask = torch.zeros_like(na_batch)
        prot_mask[(prot_batch!=prot_alphabet.padding_idx) & (prot_batch!=prot_alphabet.eos_idx) & (prot_batch!=prot_alphabet.cls_idx)] = 1
        na_mask[(na_batch!=na_alphabet.pad_idx) & (na_batch!=na_alphabet.eos_idx) & (na_batch!=na_alphabet.cls_idx)] = 1
        if use_precomputed:
            seq_prot_batch = {name: torch.stack(seq_prot_embeds[name], dim=0) for name in seq_prot_models}
            seq_rna_batch = {name: torch.stack(seq_rna_embeds[name], dim=0) for name in seq_rna_models}
            str_prot_batch = {name: torch.stack(str_prot_embeds[name], dim=0) for name in str_prot_models}
            str_rna_batch = {name: torch.stack(str_rna_embeds[name], dim=0) for name in str_rna_models}
        else:
            seq_prot_batch = None
            seq_rna_batch = None
            str_prot_batch = None
            str_rna_batch = None
        if mut_flag:
            return (
                prot_batch.long(),
                mut_batch.long(),
                prot_chains,
                prot_mask,
                na_batch.long(),
                na_chains,
                na_mask,
                seq_prot_batch,
                seq_rna_batch,
                str_prot_batch,
                str_rna_batch,
            )
        else:
            return (
                prot_batch.long(),
                prot_chains,
                prot_mask,
                na_batch.long(),
                na_chains,
                na_mask,
                seq_prot_batch,
                seq_rna_batch,
                str_prot_batch,
                str_rna_batch,
            )

    def __call__(self, data_list):
        def _to_plain(obj):
            # Avoid returning EasyDict or other custom Mapping types in the batch.
            # Lightning/DDP moves batches to device by reconstructing Mapping types,
            # and EasyDict cannot be reconstructed from tuple pairs.
            if isinstance(obj, dict):
                return {k: _to_plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_to_plain(v) for v in obj)
            if hasattr(obj, "items") and obj.__class__.__name__ == "EasyDict":
                return {k: _to_plain(v) for k, v in obj.items()}
            return obj

        data_list = [_to_plain(x) for x in data_list]
        data_list_padded = self.collate_complex(data_list)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)
        if 'mut_seqs' in data_list[0]:
            (
                prot_batch,
                mut_batch,
                prot_chains,
                prot_mask,
                na_batch,
                na_chains,
                na_mask,
                seq_prot_batch,
                seq_rna_batch,
                str_prot_batch,
                str_rna_batch,
            ) = self.pad_for_berts(self.strategy, data_list)
            batch['prot_mut'] = mut_batch
        else:
            (
                prot_batch,
                prot_chains,
                prot_mask,
                na_batch,
                na_chains,
                na_mask,
                seq_prot_batch,
                seq_rna_batch,
                str_prot_batch,
                str_rna_batch,
            ) = self.pad_for_berts(self.strategy, data_list)
        batch['prot'] = prot_batch
        batch['prot_chains'] = prot_chains
        batch['protein_mask'] = prot_mask
        batch['na'] = na_batch
        batch['na_chains'] = na_chains
        batch['na_mask'] = na_mask
        if seq_prot_batch is not None:
            batch['seq_prot_embeddings'] = seq_prot_batch
            batch['seq_rna_embeddings'] = seq_rna_batch
            batch['str_prot_embeddings'] = str_prot_batch
            batch['str_rna_embeddings'] = str_rna_batch
            batch['use_precomputed_embeddings'] = True
        batch['strategy'] = self.strategy
        labs = batch["labels"]
        if isinstance(labs, torch.Tensor):
            batch["labels"] = labs.float().reshape(-1)
        elif isinstance(labs, (list, tuple)):
            # default_collate leaves a list when per-sample label dtypes differ (e.g. str vs float in CSV).
            batch["labels"] = torch.tensor([float(x) for x in labs], dtype=torch.float32)
        else:
            batch["labels"] = torch.as_tensor(labs, dtype=torch.float32).reshape(-1)

        # Optional: attach offline embeddings (12 models, 4 groups) to the batch so the model
        # can bypass online ESM/RiNALMo forward.
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
                # current token space: [CLS] + residues + [EOS] + pad
                # offline token_embeddings are residue-level [L, D]
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

            # Important: keep the same iteration order as pad_for_berts (item -> chains)
            for item in data_list:
                # Prefer structure id (e.g. PDB_MUTATION for mut datasets) so offline .pt names match extract_features.
                pdb_id = str(item.get("id", item.get("complex", "")))
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

            # Stack into [total_chains, T, D] tensors
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
    
    
