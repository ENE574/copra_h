from data.complex import ComplexInput
from data.register import DataRegister
from data.entity_types import (
    entity_pair_summary,
    entity_pair_to_legacy_entity_type,
    resolve_entity_pair_from_args,
)
from torch.utils.data import Dataset
import pandas as pd
from esm.data import Alphabet as ESMAlphabet
import torch
from utils.torch_compat import torch_load_compat
from tqdm import tqdm
import os
import math
from data.transforms import get_transform
from torch.utils.data._utils.collate import default_collate
from typing import Optional, Dict
from easydict import EasyDict
from data.protein.residue_constants import restype_order, restype_num
from data.rna.base_constants import RNA_NUCLEOTIDES

R = DataRegister()


def build_na_alphabet_config():
    """Backward-compatible RNA alphabet config for online extraction modules."""
    from rinalmo.data.constants import (
        CLS_TKN,
        EOS_TKN,
        MASK_TKN,
        PAD_TKN,
        RNA_TOKENS,
        UNK_TKN,
    )
    return {
        "standard_tkns": RNA_TOKENS,
        "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
    }


# ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4
# ATOM_P, ATOM_C4, ATOM_NB = 37, 38, 

def _parse_chain_seq_entries(field) -> list:
    """Parse 'A:SEQ,B:SEQ' into [('A', 'SEQ'), ...] preserving order."""
    entries = []
    if field is None or (isinstance(field, float) and pd.isna(field)):
        return entries
    for part in str(field).split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        chain, seq = part.split(":", 1)
        entries.append((chain.strip(), seq.strip()))
    return entries


def _chain_seq_dict(entries) -> dict:
    return {chain: seq for chain, seq in entries}


def _mutation_sites_from_csv(wt_seq: str, mut_seq: str) -> list:
    sites = []
    for i in range(min(len(wt_seq), len(mut_seq))):
        if wt_seq[i] != mut_seq[i]:
            sites.append((i, mut_seq[i]))
    return sites


def _map_csv_seq_to_struct(csv_seq: str, struct_seq: str) -> dict:
    """Map indices in csv_seq to indices in struct_seq via sequence alignment."""
    import difflib

    mapping = {}
    matcher = difflib.SequenceMatcher(None, csv_seq, struct_seq, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
        elif tag == "replace":
            block_len = min(i2 - i1, j2 - j1)
            for k in range(block_len):
                mapping[i1 + k] = j1 + k
    return mapping


def _aa_to_restype_idx(aa: str, is_rna: bool = False) -> int:
    if is_rna:
        if aa in RNA_NUCLEOTIDES:
            return RNA_NUCLEOTIDES.index(aa) + 21
        return len(RNA_NUCLEOTIDES) + 21
    return restype_order.get(aa, restype_num)


def _apply_chain_mutations(
    struct_seq: str,
    wt_csv: str,
    mut_csv: str,
    base_offset: int,
    mut_restype: torch.Tensor,
    is_rna: bool,
) -> str:
    wt_csv = wt_csv or struct_seq
    mut_csv = mut_csv or wt_csv
    wt_to_struct = _map_csv_seq_to_struct(wt_csv, struct_seq)
    mut_chars = list(struct_seq)
    for csv_idx, mut_aa in _mutation_sites_from_csv(wt_csv, mut_csv):
        struct_idx = wt_to_struct.get(csv_idx)
        if struct_idx is None:
            continue
        mut_chars[struct_idx] = mut_aa
        mut_restype[base_offset + struct_idx] = _aa_to_restype_idx(mut_aa, is_rna=is_rna)
    return "".join(mut_chars)


def _build_mut_restype_and_seqs(cplx, prot_chains, na_chains, row, entity_type, col_prot, col_na, col_mut):
    mut_entries = _chain_seq_dict(_parse_chain_seq_entries(row[col_mut]))
    wt_prot_entries = _chain_seq_dict(_parse_chain_seq_entries(row[col_prot]))
    wt_na_entries = _chain_seq_dict(_parse_chain_seq_entries(row[col_na])) if col_na in row else {}

    mut_restype = cplx["restype"].clone()
    mut_seqs = []
    struct_seqs = list(cplx["prot_seqs"]) + list(cplx["rna_seqs"])
    chain_ids = prot_chains + na_chains
    offset = 0

    for chain_idx, (chain_id, struct_seq) in enumerate(zip(chain_ids, struct_seqs)):
        is_prot_chain = chain_idx < len(prot_chains)
        mut_csv = mut_entries.get(chain_id)
        if is_prot_chain:
            wt_csv = wt_prot_entries.get(chain_id, struct_seq)
            if mut_csv is None:
                mut_csv = wt_csv
            mut_seq = _apply_chain_mutations(
                struct_seq, wt_csv, mut_csv, offset, mut_restype, is_rna=False
            )
            mut_seqs.append(mut_seq)
        elif entity_type == "ppi":
            wt_csv = wt_prot_entries.get(chain_id, struct_seq)
            if mut_csv is None:
                mut_csv = wt_csv
            _apply_chain_mutations(struct_seq, wt_csv, mut_csv, offset, mut_restype, is_rna=False)
        else:
            wt_csv = wt_na_entries.get(chain_id, struct_seq)
            if mut_csv is None:
                mut_csv = wt_na_entries.get(chain_id, struct_seq)
            _apply_chain_mutations(struct_seq, wt_csv, mut_csv, offset, mut_restype, is_rna=True)
        offset += len(struct_seq)

    return mut_restype, mut_seqs


def _filter_chains_in_pdb(pdb_path: str, chain_ids: list) -> list:
    """Keep chain ids that are present as protein chains in the PDB."""
    if not chain_ids:
        return []
    from data.protein.proteins import ProteinInput

    parsed = ProteinInput.from_path(pdb_path, return_dict=True, valid_chains=chain_ids)
    return [c for c in chain_ids if c in parsed]


def _process_structure(
    structure_path,
    structure_id,
    valid_prot_chains=None,
    valid_rna_chains=None,
    valid_partner_chains=None,
    entity_type="prot_na",
    gpu=None,
) -> Optional[Dict]:
    cplx = ComplexInput.from_path(
        structure_path,
        valid_prot_chains=valid_prot_chains,
        valid_rna_chains=valid_rna_chains,
        valid_partner_chains=valid_partner_chains,
        entity_type=entity_type,
    )
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
                 col_partner_chain='Protein chains B',
                 col_prot='Protein sequences',
                 col_na='RNA sequences',
                 col_label='△G(kcal/mol)',
                 entity_type='prot_na',
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
                 rna_embedding_model: str = "rna_fm",
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
        self.col_partner_chain = col_partner_chain
        self.col_label = col_label
        self.col_prot = col_prot
        self.col_na = col_na
        self.entity_b_type = str(kwargs.get("entity_b_type", "rna"))
        self.mutation_task = str(kwargs.get("mutation_task", "none"))
        self.entity_type = str(entity_type)
        self.mut = mut
        self.col_mut = col_mut
        self.entity_pair = resolve_entity_pair_from_args(self)
        if entity_type == "prot_na" and self.entity_pair.interaction == "prot_dna":
            self.entity_type = "prot_na"
        elif self.entity_pair.interaction == "ppi":
            self.entity_type = "ppi"
        else:
            self.entity_type = entity_pair_to_legacy_entity_type(self.entity_pair)
        self._entity_pair_meta = entity_pair_summary(self.entity_pair)
        self.type = 'reg'
        self.diskcache = diskcache
        self.prot_alphabet = ESMAlphabet.from_architecture("ESM-1b")
        def _ensure_list(value, fallback):
            if value is None:
                return list(fallback)
            if isinstance(value, str):
                return [value]
            return list(value)

        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.embedding_root = embedding_root
        # NA sequence/structure offline embedding 子目录由 offline_na_sequence_group /
        # offline_na_structure_group 决定（DNA 复合 -> dna_sequence/dna_structure，
        # RNA 复合 -> rna_sequence/rna_structure），默认保持 RNA 行为以兼容旧配置。
        self.offline_na_sequence_group = str(kwargs.get("offline_na_sequence_group", "rna_sequence"))
        self.offline_na_structure_group = str(kwargs.get("offline_na_structure_group", "rna_structure"))
        self.protein_embedding_model = protein_embedding_model
        self.rna_embedding_model = rna_embedding_model
        self.seq_prot_models = _ensure_list(seq_prot_models, [protein_embedding_model])
        self.seq_rna_models = _ensure_list(seq_rna_models, [rna_embedding_model])
        self.str_prot_models = _ensure_list(str_prot_models, ["esm_if1"])
        self.str_rna_models = _ensure_list(str_rna_models, ["rhofold"])
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
                prot_chains = [c.strip() for c in str(row[self.col_prot_chain]).split(',') if c.strip()]
                if self.entity_type == 'ppi':
                    na_chains = [c.strip() for c in str(row[self.col_partner_chain]).split(',') if c.strip()]
                    partner_chains = na_chains
                    rna_chains = []
                else:
                    partner_chains = []
                    rna_chains = [c.strip() for c in str(row[self.col_na_chain]).split(',') if c.strip()]
                    na_chains = rna_chains
                pdb_path = self._resolve_pdb_path(structure_id)

                label = float(row[self.col_label])

                prot_chains = _filter_chains_in_pdb(pdb_path, prot_chains)
                if self.entity_type == "ppi":
                    partner_chains = _filter_chains_in_pdb(pdb_path, partner_chains)
                    na_chains = partner_chains

                cplx = _process_structure(
                    pdb_path,
                    structure_id,
                    prot_chains,
                    rna_chains,
                    valid_partner_chains=partner_chains,
                    entity_type=self.entity_type,
                )
                if cplx is None:
                    print(f"[WARN] Skip {structure_id}: no valid chains in {pdb_path}")
                    continue

                if self.mut:
                    mut_restype, mut_seqs = _build_mut_restype_and_seqs(
                        cplx,
                        prot_chains,
                        na_chains,
                        row,
                        self.entity_type,
                        self.col_prot,
                        self.col_na,
                        self.col_mut,
                    )
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
                complex_group = complex.split('.mut.')[0]  # strip .mut.* suffix for within-PDB grouping
                if self.mut:
                    item = {
                        'complex': complex,
                        'complex_group': complex_group,
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
                        'complex_group': complex_group,
                        'labels': label,
                        'atom_min_dist': atom_min_dist, # needs 2D padding
                        'max_prot_length': max_prot_length,
                        'max_na_length': max_na_length,
                        'physics_targets': physics_targets
                    }
                item['prot_chain_ids'] = prot_chains
                item['rna_chain_ids'] = na_chains
                item['entity_pair'] = dict(self._entity_pair_meta)
                import sys
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
                data['complex_group'] = complex.split('.mut.')[0]
                # diskcache may contain old entries without these keys; reconstruct from csv row
                try:
                    pdb_path = self._resolve_pdb_path(structure_id)
                    prot_ids = [c.strip() for c in str(row[self.col_prot_chain]).split(',') if c.strip()]
                    data["prot_chain_ids"] = _filter_chains_in_pdb(pdb_path, prot_ids)
                    if self.entity_type == 'ppi':
                        partner_ids = [
                            c.strip() for c in str(row[self.col_partner_chain]).split(',') if c.strip()
                        ]
                        data["rna_chain_ids"] = _filter_chains_in_pdb(pdb_path, partner_ids)
                    else:
                        data["rna_chain_ids"] = [
                            c.strip() for c in str(row[self.col_na_chain]).split(',') if c.strip()
                        ]
                except Exception:
                    # keep backward compatibility if columns are missing
                    if "prot_chain_ids" not in data:
                        data["prot_chain_ids"] = []
                    if "rna_chain_ids" not in data:
                        data["rna_chain_ids"] = []
                # diskcache may hold old entries without entity_pair; reconstruct from dataset metadata
                if "entity_pair" not in data:
                    data["entity_pair"] = dict(self._entity_pair_meta)
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

EXCLUDE_KEYS = ['labels', 'complex', 'complex_group']
DEFAULT_PAD_VALUES = {
    'restype': 26,
    'mut_restype': 26,
    'mut_identifier': 0,
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

    def _na_encoding_mode(self):
        if self.dataset_args is None:
            return "rinalmo"
        if getattr(self.dataset_args, "entity_type", "") == "ppi":
            return "protein"
        if getattr(self.dataset_args, "use_offline_embeddings", False):
            return "length"
        return "rinalmo"

    def _get_na_alphabet(self, prot_alphabet):
        mode = self._na_encoding_mode()
        if mode == "protein":
            return prot_alphabet, mode
        if mode == "length":
            return None, mode
        from rinalmo.data.alphabet import Alphabet
        from rinalmo.data.constants import (
            CLS_TKN,
            EOS_TKN,
            MASK_TKN,
            PAD_TKN,
            RNA_TOKENS,
            UNK_TKN,
        )
        na_alphabet_config = {
            "standard_tkns": RNA_TOKENS,
            "special_tkns": [CLS_TKN, PAD_TKN, EOS_TKN, UNK_TKN, MASK_TKN],
        }
        return Alphabet(**na_alphabet_config), mode

    def pad_for_berts(self, strategy, batch):
        prot_alphabet = ESMAlphabet.from_architecture("ESM-1b")
        na_alphabet, na_mode = self._get_na_alphabet(prot_alphabet)
        mut_flag = 0
        prot_chains = [len(item['prot_seqs']) for item in batch]
        na_chains = [len(item['rna_seqs']) for item in batch]
        use_precomputed = bool(batch[0].get("use_precomputed_embeddings", False))
        if use_precomputed:
            emb_root = Path(batch[0].get("embedding_root", "outputs/feature_extraction"))
            seq_prot_models = list(batch[0].get("seq_prot_models", ["esm2"]))
            seq_rna_models = list(batch[0].get("seq_rna_models", ["rna_fm"]))
            str_prot_models = list(batch[0].get("str_prot_models", ["esm_if1"]))
            str_rna_models = list(batch[0].get("str_rna_models", ["rhofold"]))
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
        if na_mode == "length":
            na_batch.zero_()
        elif na_mode == "protein":
            na_batch.fill_(prot_alphabet.padding_idx)
        else:
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
                if na_mode == "length":
                    seq_len = min(len(na_seq), max(0, max_na_length - 2))
                    if seq_len > 0:
                        na_batch[curr_na_idx, 1:1 + seq_len] = 1
                elif na_mode == "protein":
                    na_batch[curr_na_idx, 0] = prot_alphabet.cls_idx
                    na_seq_encode = prot_alphabet.encode(na_seq)
                    seq = torch.tensor(na_seq_encode, dtype=torch.int64)
                    na_batch[curr_na_idx, 1: len(na_seq_encode) + 1] = seq
                    na_batch[curr_na_idx, len(na_seq_encode) + 1] = prot_alphabet.eos_idx
                else:
                    na_seq_encode = na_alphabet.encode(na_seq)
                    seq = torch.tensor(na_seq_encode, dtype=torch.int64)
                    na_batch[curr_na_idx, :len(seq)] = seq
                if use_precomputed:
                    chain_id = rna_chain_ids[j] if j < len(rna_chain_ids) else 'X'
                    name = safe_name(f"{complex_id}_rna_{chain_id or 'X'}")
                    for model_name in seq_rna_models:
                        emb = _load_embedding(self.offline_na_sequence_group, name, len(na_seq), max_na_length, model_name)
                        if emb is not None:
                            seq_rna_embeds[model_name].append(emb)
                    for model_name in str_rna_models:
                        emb = _load_embedding(self.offline_na_structure_group, name, len(na_seq), max_na_length, model_name)
                        if emb is not None:
                            str_rna_embeds[model_name].append(emb)
                curr_na_idx += 1
        prot_mask = torch.zeros_like(prot_batch)
        na_mask = torch.zeros_like(na_batch)
        prot_mask[(prot_batch!=prot_alphabet.padding_idx) & (prot_batch!=prot_alphabet.eos_idx) & (prot_batch!=prot_alphabet.cls_idx)] = 1
        if na_mode == "length":
            na_mask[(na_batch > 0)] = 1
        elif na_mode == "protein":
            na_mask[
                (na_batch != prot_alphabet.padding_idx)
                & (na_batch != prot_alphabet.eos_idx)
                & (na_batch != prot_alphabet.cls_idx)
            ] = 1
        else:
            na_mask[
                (na_batch != na_alphabet.pad_idx)
                & (na_batch != na_alphabet.eos_idx)
                & (na_batch != na_alphabet.cls_idx)
            ] = 1
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

        # Optional: attach offline embeddings (4 models, 4 groups) to the batch so the model
        # can bypass online ESM/RiNALMo forward.
        use_offline = False
        offline_root = None
        if self.dataset_args is not None:
            use_offline = bool(getattr(self.dataset_args, "use_offline_embeddings", False))
            offline_root = getattr(self.dataset_args, "offline_embedding_root", None)

        if use_offline:
            if offline_root is None:
                raise ValueError("use_offline_embeddings=True but dataset_args.offline_embedding_root is not set")
            from utils.offline_embeddings import OfflineEmbeddingSpec, stack_offline_embeddings_for_batch

            def _model_tuple(key, default):
                val = getattr(self.dataset_args, key, None) if self.dataset_args is not None else None
                if val is None:
                    return default
                if isinstance(val, str):
                    return () if val == "none" else (val,)
                items = [x for x in list(val) if str(x) != "none"]
                return tuple(items) if items else default

            mut_subdir = getattr(self.dataset_args, "offline_mutant_subdir", "mutant")
            if mut_subdir in (None, "", "none", "None"):
                mut_subdir = None

            na_seq_group = getattr(self.dataset_args, "offline_na_sequence_group", "rna_sequence")
            na_str_group = getattr(self.dataset_args, "offline_na_structure_group", "rna_structure")
            partner_use_prot = bool(getattr(self.dataset_args, "offline_partner_use_protein_embeddings", False))
            na_wt_on_mut = bool(getattr(self.dataset_args, "offline_na_wt_on_mut", True))
            spec = OfflineEmbeddingSpec(
                root=offline_root,
                mutant_subdir=mut_subdir,
                protein_sequence_models=_model_tuple("seq_prot_models", ("esm2",)),
                protein_structure_models=_model_tuple("str_prot_models", ("esm_if1",)),
                rna_sequence_models=_model_tuple("seq_rna_models", ("rna_fm",)),
                rna_structure_models=_model_tuple("str_rna_models", ("rhofold",)),
                dna_sequence_models=_model_tuple("seq_dna_models", ()),
                dna_structure_models=_model_tuple("str_dna_models", ()),
                na_sequence_group=str(na_seq_group),
                na_structure_group=str(na_str_group),
                partner_use_protein_embeddings=partner_use_prot,
                na_wt_on_mut=na_wt_on_mut,
            )
            max_prot_len = int(prot_batch.shape[1])
            max_rna_len = int(na_batch.shape[1])
            is_mut_batch = "mut_seqs" in data_list[0]

            if is_mut_batch:
                batch["offline_embeddings"] = stack_offline_embeddings_for_batch(
                    spec, data_list, max_prot_len, max_rna_len, variant="wt"
                )
                batch["offline_embeddings_mut"] = stack_offline_embeddings_for_batch(
                    spec, data_list, max_prot_len, max_rna_len, variant="mut"
                )
            else:
                batch["offline_embeddings"] = stack_offline_embeddings_for_batch(
                    spec, data_list, max_prot_len, max_rna_len, variant="wt"
                )

            batch["use_offline_embeddings"] = True

        return batch
    
    
