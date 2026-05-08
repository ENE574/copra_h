import torch.nn as nn
import torch
import esm
from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from models.encoders.pair import ResiduePairEncoder
from models.register import ModelRegister
from models.components.coformer import CoFormer
import torch.nn.functional as F
from models.lora_tune import LoRAESM, LoRARiNALMo, ESMConfig, RiNALMoConfig
import random
from data.complex import SUPER_PROT_IDX, SUPER_RNA_IDX, SUPER_CPLX_IDX, SUPER_CHAIN_IDX

from peft import (
    LoraConfig,
    get_peft_model,
)
R = ModelRegister()

def load_esm(esm_type):
    if esm_type == '650M':
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    elif esm_type == '3B':
        model, _ = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm_type == '15B':
        model, _ = esm.pretrained.esm2_t48_15B_UR50D()
    elif esm_type == '150M':
        model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    elif esm_type == '35M':
        model, _ = esm.pretrained.esm2_t12_35M_UR50D()
    elif esm_type == '8M':
        model, _ = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise NotImplementedError
    feat_size = model.embed_dim
    return model, feat_size
    
def load_rinalmo(rinalmo_weights, rinalmo_type):
    if rinalmo_type == '650M':
        size = 'giga'
    elif rinalmo_type == '150M':
        size = 'mega'
    elif rinalmo_type == '35M':
        size = 'micro'
    elif rinalmo_type == '8M':
        size = 'nano'
    config = model_config(size)
    model = RiNALMo(config)
    # alphabet = Alphabet(**config['alphabet'])
    model.load_state_dict(torch.load(rinalmo_weights))
    feat_size = config.globals.embed_dim
    return model, feat_size

def cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx):
    # print("Input shape:", prot_embedding.shape, na_embedding.shape)
    # result = prot_embedding.new_full([len(prot_embedding), seq_len, prot_embedding.shape[-1]], 0) # (N, L, E)
    new_complexes = []
    masks = []
    for i in range(len(prot_embedding)):
        item_prot_embed = prot_embedding[i]
        item_prot_mask = prot_mask[i]
        item_na_embed = na_embedding[i]
        item_na_mask = na_mask[i]
        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)
        if patch_idx is not None:
            selected = torch.index_select(selected, 0, patch_idx[i])
        p1d = (0, 0, 0, max_len-len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)
        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad)
    result = torch.stack(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    return result, masks

def segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx=None):
    cum_prot = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(prot_chains), dim=0)]).int()
    cum_na = torch.cat([torch.tensor([0]), torch.cumsum(torch.Tensor(na_chains), dim=0)]).int()
    new_complexes = []
    masks = []
    for i, (s_prot, e_prot, s_na, e_na) in enumerate(zip(cum_prot[:-1], cum_prot[1:], cum_na[:-1], cum_na[1:])):
        item_prot_embed = prot_embedding[s_prot:e_prot].reshape((-1, prot_embedding.shape[-1]))
        item_prot_mask = prot_mask[s_prot:e_prot].reshape(-1)
        item_na_embed = na_embedding[s_na: e_na].reshape((-1, na_embedding.shape[-1]))
        item_na_mask = na_mask[s_na: e_na].reshape(-1)
        item_embed = torch.cat([item_prot_embed, item_na_embed], dim=0)
        indices = torch.nonzero(torch.cat([item_prot_mask, item_na_mask])).flatten()
        selected = torch.index_select(item_embed, 0, indices)
        if patch_idx is not None:
            selected = torch.index_select(selected, 0, patch_idx[i])
        p1d = (0, 0, 0, max_len-len(selected))
        selected_pad = F.pad(selected, p1d, 'constant', 0)
        mask = torch.zeros((selected_pad.shape[0]), device=selected.device)
        mask[:len(selected)] = 1
        # # selected_pad = torch.cat([selected, torch.zeros((seq_len-len(selected), prot_embedding.shape[-1]), device=selected.device)], dim=0)
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad.unsqueeze(0))
    result = torch.cat(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    # print("Result shape:", result)
    return result, masks

@R.register('copra')
class ESM2RiNALMo(nn.Module):
    def __init__(self, 
                 rinalmo_weights='./weights/rinalmo_giga_pretrained.pt',
                 esm_type='650M',
                 rinalmo_type='650M',
                 pooling='mean',
                 output_dim=1,
                 pair_dim=320,
                 fix_lms=True,
                 lora_tune=True,
                 lora_rank=16,
                 lora_alpha=32,
                 representation_layer=33,
                 dist_dim=40,
                 use_offline_embeddings=False,
                 offline_dim=1280,
                 main_refinement_scale=0.1,
                 **kwargs
                 ):
        super(ESM2RiNALMo, self).__init__()
        self.use_offline_embeddings = use_offline_embeddings
        if not self.use_offline_embeddings:
            self.esm, esm_feat_size = load_esm(esm_type)
            self.rinalmo, rinalmo_feat_size = load_rinalmo(rinalmo_weights, rinalmo_type)
        else:
            self.esm = None
            self.rinalmo = None
            esm_feat_size = None
            rinalmo_feat_size = None
        self.pair_encoder = ResiduePairEncoder(pair_dim, max_num_atoms=4)  # N, CA, C, O,
        self.c_former = CoFormer(**kwargs['coformer'])
        self.representation_layer = representation_layer
        self.proj = 0
        if not self.use_offline_embeddings and esm_feat_size != rinalmo_feat_size:
            self.proj = 1
            self.project_feat = nn.Linear(esm_feat_size, rinalmo_feat_size)
        self.complex_dim = kwargs['coformer']['embed_dim']
        self.feat_size = offline_dim if self.use_offline_embeddings else rinalmo_feat_size
        self.proj_cplx= nn.Linear(self.feat_size, self.complex_dim)
        if (not self.use_offline_embeddings) and lora_tune:
            import re
            pattern = r'\((\w+)\): Linear'
            rinalmo_linear_layers = re.findall(pattern, str(self.rinalmo.modules))
            rinalmo_linear_modules = list(set(rinalmo_linear_layers))
            print("In rinalmo:", rinalmo_linear_modules)
            rinalmo_linear_modules = ['Wqkv']
            esm_linear_layers = re.findall(pattern, str(self.esm.modules))
            esm_linear_modules = list(set(esm_linear_layers))
            print("In esm:", esm_linear_modules)
            print("Getting Lora Models...")
            # copied from LongLoRA
            rinalmo_lora_config = LoraConfig(
                r=lora_rank,
                bias="none",
                target_modules=rinalmo_linear_modules,
                lora_alpha=lora_alpha
            )
            esm_lora_config = LoraConfig(
                r=lora_rank,
                bias="none",
                target_modules=esm_linear_modules,
                lora_alpha=lora_alpha
            )
            rinalmo_config = RiNALMoConfig()
            esm_config = ESMConfig()
            self.rinalmo = LoRARiNALMo(self.rinalmo, rinalmo_config)
            self.esm = LoRAESM(self.esm, esm_config)
            self.rinalmo = get_peft_model(self.rinalmo, rinalmo_lora_config)
            print("Get RINALMO DONE!!!!!")
            self.rinalmo.print_trainable_parameters()
            self.esm = get_peft_model(self.esm, esm_lora_config)
            print("Get ESM DONE!!!!!")
            self.esm.print_trainable_parameters()
        elif (not self.use_offline_embeddings) and fix_lms:
            for p in self.rinalmo.parameters():
                p.requires_grad_(False)
            for p in self.esm.parameters():
                p.requires_grad_(False)

        # Offline embedding fusion modules (12 models, 4 groups) + interactions
        if self.use_offline_embeddings:
            def _as_list(x):
                if x is None:
                    return []
                if isinstance(x, str):
                    return [x]
                return list(x)

            # Per-model input dims
            self._offline_dims = {
                "prot_seq": {"esm2": 1280, "prott5": 1024, "saprot": 1280},
                "prot_struct": {"esm_if1": 512, "protbert": 1024, "protrek": 1024},
                "rna_seq": {"rinalmo": 1280, "rna_fm": 640, "rna_msm": 768},
                "rna_struct": {"ernie_rna": 768, "rhofold": 384, "rnabert": 120},
            }

            self._all_offline_models = {
                "prot_seq": ["esm2", "prott5", "saprot"],
                "prot_struct": ["esm_if1", "protbert", "protrek"],
                "rna_seq": ["rinalmo", "rna_fm", "rna_msm"],
                "rna_struct": ["ernie_rna", "rhofold", "rnabert"],
            }

            enabled = {
                "prot_seq": set(_as_list(kwargs.get("seq_prot_models"))),
                "rna_seq": set(_as_list(kwargs.get("seq_rna_models"))),
                "prot_struct": set(_as_list(kwargs.get("str_prot_models"))),
                "rna_struct": set(_as_list(kwargs.get("str_rna_models"))),
            }

            disable_all_groups = set()
            for group in list(enabled.keys()):
                # allow explicit disable via: ['none'] or 'none'
                if enabled[group] == {"none"}:
                    enabled[group] = set()
                    disable_all_groups.add(group)

            if "rna_ernie" in enabled["rna_struct"]:
                enabled["rna_struct"].remove("rna_ernie")
                enabled["rna_struct"].add("ernie_rna")
            for group, all_ms in self._all_offline_models.items():
                if len(enabled[group]) == 0 and group not in disable_all_groups:
                    enabled[group] = set(all_ms)
            self._enabled_offline_models = enabled

            def _make_proj(d_in):
                return nn.Linear(d_in, self.feat_size)

            self.offline_proj = nn.ModuleDict()
            for group, m2d in self._offline_dims.items():
                for m, d_in in m2d.items():
                    self.offline_proj["{}/{}".format(group, m)] = _make_proj(d_in)

            self.offline_q_proj = nn.ModuleDict({
                "prot_seq": nn.Linear(self.feat_size * 3, self.feat_size),
                "prot_struct": nn.Linear(self.feat_size * 3, self.feat_size),
                "rna_seq": nn.Linear(self.feat_size * 3, self.feat_size),
                "rna_struct": nn.Linear(self.feat_size * 3, self.feat_size),
            })
            self.offline_k_proj = nn.ModuleDict({
                "prot_seq": nn.Linear(self.feat_size, self.feat_size),
                "prot_struct": nn.Linear(self.feat_size, self.feat_size),
                "rna_seq": nn.Linear(self.feat_size, self.feat_size),
                "rna_struct": nn.Linear(self.feat_size, self.feat_size),
            })
            self.offline_v_proj = nn.ModuleDict({
                "prot_seq": nn.Linear(self.feat_size, self.feat_size),
                "prot_struct": nn.Linear(self.feat_size, self.feat_size),
                "rna_seq": nn.Linear(self.feat_size, self.feat_size),
                "rna_struct": nn.Linear(self.feat_size, self.feat_size),
            })
            # Use the same head count as CoFormer when possible
            self.offline_num_heads = int(kwargs["coformer"]["num_heads"])
            self._head_dim = self.feat_size // self.offline_num_heads
            assert self.feat_size % self.offline_num_heads == 0, "offline_dim must be divisible by coformer.num_heads"

            # Entity-internal seq<->struct bidirectional cross-attention
            self.prot_seq_to_struct = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
            self.prot_struct_to_seq = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
            self.rna_seq_to_struct = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
            self.rna_struct_to_seq = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)

            # Cross-entity bidirectional attention: protein <-> RNA
            self.prot_to_rna = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
            self.rna_to_prot = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
                
        self.pooling = pooling
        print("Pooling Strategy:", self.pooling)

        self.main_refinement_scale = float(main_refinement_scale)
        if self.pooling == 'token':
            self.prot_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.rna_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            self.complex_embedding = nn.Parameter(torch.zeros((1, self.complex_dim), dtype=torch.float32))
            nn.init.normal_(self.prot_embedding)
            nn.init.normal_(self.rna_embedding)
            nn.init.normal_(self.complex_embedding)
        if pair_dim != self.complex_dim:
            self.z_proj = nn.Linear(pair_dim, self.complex_dim)
        
        # Physical Decomposition Heads
        self.physics_names = ['contact', 'electro', 'hydrophobic', 'stacking']
        self.physics_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
                nn.Linear(self.feat_size, 1)
            ) for name in self.physics_names
        })
        # Learnable weights for each physical component (initialized to 1.0)
        self.physics_weights = nn.Parameter(torch.ones(len(self.physics_names)))
        
        self.pred_head = nn.Sequential(
            nn.Linear(self.complex_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, output_dim)
        )
        # For mask distance pretraining
        self.mask_token = nn.Parameter(torch.randn(size=(1, pair_dim)))
        self.dist_head = nn.Sequential(
            nn.Linear(pair_dim, self.feat_size), nn.ReLU(),
            nn.Linear(self.feat_size, dist_dim)
        )

    def _offline_group_fusion(self, group_name, x_list):
        """
        Group-wise fusion (3 models per group).
        Q: joint from three embeddings (concat -> linear)
        K/V: from three model representations.
        Attention is over the 3 sources (model dimension) for each token.
        Args:
          x_list: list of 3 tensors, each [B, T, D_in] already in self.feat_size
        Returns:
          fused: [B, T, self.feat_size]
        """
        x_cat = torch.cat(x_list, dim=-1)  # [B,T,3D]
        q = self.offline_q_proj[group_name](x_cat)  # [B,T,D]
        # stack sources
        k = torch.stack([self.offline_k_proj[group_name](x) for x in x_list], dim=2)  # [B,T,3,D]
        v = torch.stack([self.offline_v_proj[group_name](x) for x in x_list], dim=2)  # [B,T,3,D]

        B, T, S, D = k.shape
        H = self.offline_num_heads
        Dh = self._head_dim
        qh = q.view(B, T, H, Dh)  # [B,T,H,Dh]
        kh = k.view(B, T, S, H, Dh)  # [B,T,S,H,Dh]
        vh = v.view(B, T, S, H, Dh)  # [B,T,S,H,Dh]

        # scores: [B,T,H,S]
        scores = (qh.unsqueeze(2) * kh).sum(-1) / (Dh ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        # fused per head: [B,T,H,Dh]
        fused_h = (attn.unsqueeze(-1) * vh).sum(dim=2)
        fused = fused_h.reshape(B, T, D)
        return fused
    
    def _forward(self, input, strategy='separate', need_mask=False):
        prot_input = input['prot']
        prot_chains = input['prot_chains']
        prot_mask = input['protein_mask']
        na_input = input['na']
        na_chains = input['na_chains']
        na_mask = input['na_mask']

        if self.use_offline_embeddings or bool(input.get("use_offline_embeddings", False)):
            offline = input["offline_embeddings"]

            # Convert token-space offline embeddings -> residue-space per complex
            # (strip CLS/EOS/PAD using masks, then concatenate chains inside each complex).
            def _to_residue_per_complex(x_token, chain_counts, chain_mask):
                out = []
                lens = []
                idx = 0
                for n_chains in chain_counts:
                    parts = []
                    for _ in range(int(n_chains)):
                        m = chain_mask[idx].bool()
                        parts.append(x_token[idx][m])
                        idx += 1
                    if len(parts) == 1:
                        x = parts[0]
                    else:
                        x = torch.cat(parts, dim=0)
                    out.append(x)
                    lens.append(int(x.shape[0]))
                return out, lens

            # residue-space lists (len = N_complex), each item [L_i, D_in]
            prot_seq_list = {k: None for k in offline["prot_seq"].keys()}
            prot_struct_list = {k: None for k in offline["prot_struct"].keys()}
            rna_seq_list = {k: None for k in offline["rna_seq"].keys()}
            rna_struct_list = {k: None for k in offline["rna_struct"].keys()}

            for m in prot_seq_list:
                prot_seq_list[m], prot_lens = _to_residue_per_complex(offline["prot_seq"][m], prot_chains, prot_mask)
            for m in prot_struct_list:
                prot_struct_list[m], _ = _to_residue_per_complex(offline["prot_struct"][m], prot_chains, prot_mask)
            for m in rna_seq_list:
                rna_seq_list[m], rna_lens = _to_residue_per_complex(offline["rna_seq"][m], na_chains, na_mask)
            for m in rna_struct_list:
                rna_struct_list[m], _ = _to_residue_per_complex(offline["rna_struct"][m], na_chains, na_mask)

            # Pad residue-space to batch tensors for attention blocks
            def _pad_list(xs, max_len):
                B = len(xs)
                D = int(xs[0].shape[-1]) if B > 0 else 0
                out = xs[0].new_zeros((B, max_len, D))
                pad_mask = torch.ones((B, max_len), device=out.device, dtype=torch.bool)  # True=ignore
                for i, x in enumerate(xs):
                    L = int(x.shape[0])
                    if L > 0:
                        out[i, :L] = x
                        pad_mask[i, :L] = False
                return out, pad_mask

            max_prot = max(prot_lens) if len(prot_lens) > 0 else 0
            max_rna = max(rna_lens) if len(rna_lens) > 0 else 0

            def _proj(group, model, x):
                return self.offline_proj["{}/{}".format(group, model)](x)

            def _maybe_disable(group, model, x):
                enabled = getattr(self, "_enabled_offline_models", None)
                if enabled is None:
                    return x
                if model not in enabled.get(group, set()):
                    return x * 0.0
                return x

            # Prepare 3 projected tensors per group for fusion
            prot_seq_3 = []
            for m in ["esm2", "prott5", "saprot"]:
                t, _ = _pad_list(prot_seq_list[m], max_prot)
                prot_seq_3.append(_maybe_disable("prot_seq", m, _proj("prot_seq", m, t)))
            prot_struct_3 = []
            for m in ["esm_if1", "protbert", "protrek"]:
                t, _ = _pad_list(prot_struct_list[m], max_prot)
                prot_struct_3.append(_maybe_disable("prot_struct", m, _proj("prot_struct", m, t)))
            rna_seq_3 = []
            for m in ["rinalmo", "rna_fm", "rna_msm"]:
                t, _ = _pad_list(rna_seq_list[m], max_rna)
                rna_seq_3.append(_maybe_disable("rna_seq", m, _proj("rna_seq", m, t)))
            rna_struct_3 = []
            for m in ["ernie_rna", "rhofold", "rnabert"]:
                t, _ = _pad_list(rna_struct_list[m], max_rna)
                rna_struct_3.append(_maybe_disable("rna_struct", m, _proj("rna_struct", m, t)))

            # Padding masks (same across models within an entity)
            _, prot_pad_mask = _pad_list(prot_seq_list["esm2"], max_prot)
            _, rna_pad_mask = _pad_list(rna_seq_list["rinalmo"], max_rna)

            # 1) within-group fusion
            prot_seq = self._offline_group_fusion("prot_seq", prot_seq_3)
            prot_struct = self._offline_group_fusion("prot_struct", prot_struct_3)
            rna_seq = self._offline_group_fusion("rna_seq", rna_seq_3)
            rna_struct = self._offline_group_fusion("rna_struct", rna_struct_3)

            # 2) within-entity bidirectional interaction
            prot_struct2, _ = self.prot_seq_to_struct(query=prot_struct, key=prot_seq, value=prot_seq, key_padding_mask=prot_pad_mask)
            prot_seq2, _ = self.prot_struct_to_seq(query=prot_seq, key=prot_struct, value=prot_struct, key_padding_mask=prot_pad_mask)
            rna_struct2, _ = self.rna_seq_to_struct(query=rna_struct, key=rna_seq, value=rna_seq, key_padding_mask=rna_pad_mask)
            rna_seq2, _ = self.rna_struct_to_seq(query=rna_seq, key=rna_struct, value=rna_struct, key_padding_mask=rna_pad_mask)
            prot_entity = prot_seq2 + prot_struct2
            rna_entity = rna_seq2 + rna_struct2

            # 3) cross-entity bidirectional interaction (per complex; batch size is N_complex)
            prot_final, _ = self.prot_to_rna(query=prot_entity, key=rna_entity, value=rna_entity, key_padding_mask=rna_pad_mask)
            rna_final, _ = self.rna_to_prot(query=rna_entity, key=prot_entity, value=prot_entity, key_padding_mask=prot_pad_mask)

            # Build residue-space complex embedding: [prot residues][rna residues], padded to max_len from structure batch
            max_len = input['pos_atoms'].shape[1]
            out_embedding = prot_final.new_zeros((len(prot_lens), max_len, self.feat_size))
            masks = torch.zeros((len(prot_lens), max_len), device=out_embedding.device, dtype=torch.bool)
            for i, (Lp, Lr) in enumerate(zip(prot_lens, rna_lens)):
                combined = torch.cat([prot_final[i, :Lp], rna_final[i, :Lr]], dim=0)  # [Lp+Lr, D]

                # If structure transform selected a patch, apply the same indices to offline embeddings.
                if "patch_idx" in input:
                    # patch_idx is per-complex indices into the full complex sequence (protein first, RNA second)
                    idx = input["patch_idx"][i]
                    idx = idx[idx >= 0].long()
                    combined = torch.index_select(combined, 0, idx)

                # Finally fit into max_len (patch_size) with truncation if needed
                Lc = int(min(combined.shape[0], max_len))
                if Lc > 0:
                    out_embedding[i, :Lc] = combined[:Lc]
                    masks[i, :Lc] = True

            out_embedding = self.proj_cplx(out_embedding)
            key_padding_mask = ~masks

            aa=input['restype']
            res_nb=input['res_nb']
            chain_nb=input['chain_nb']
            pos_atoms=input['pos_atoms']
            mask_atoms=input['mask_atoms']

            z = self.pair_encoder(
                aa=aa,
                res_nb=res_nb,
                chain_nb=chain_nb,
                pos_atoms=pos_atoms,
                mask_atoms=mask_atoms,
            )
            if need_mask:
                for i in range(z.shape[0]):
                    to_mask = torch.rand(1).item() > 0.5
                    if not to_mask:
                        continue
                    valid = list(range(3, z.shape[1]))
                    mask_indices = random.sample(valid, int(len(valid) * 0.15))
                    z[i, mask_indices, :, :] = self.mask_token.repeat(len(mask_indices), z.shape[2], 1)
                    z[i, :, mask_indices, :] = self.mask_token.repeat(z.shape[1], len(mask_indices), 1)

            return out_embedding, z, key_padding_mask

        else:
            with torch.cuda.amp.autocast():
                prot_embedding = self.esm(
                    prot_input,
                    repr_layers=[self.representation_layer],
                    return_contacts=False,
                )['representations'][self.representation_layer]
                na_embedding = self.rinalmo(na_input)['representation']
                if self.proj:
                    prot_embedding = self.project_feat(prot_embedding)

        prot_embedding = prot_embedding.float()
        na_embedding = na_embedding.float()
        max_len = input['pos_atoms'].shape[1]
        # Adjust the embeddings from LMs for CoFormer
        if 'patch_idx' in input:
            patch_idx = input['patch_idx']
        else:
            patch_idx = None
        if strategy == 'separate':
            # input shape [N', L], where N' is flexible in every batch
            out_embedding, masks = segment_cat_pad(prot_embedding, prot_chains, prot_mask, na_embedding, na_chains, na_mask, max_len, patch_idx)
            assert out_embedding.shape[0] == input['size']
        else:
            out_embedding, masks = cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx)
            assert out_embedding.shape[0] == input['size']

        out_embedding = self.proj_cplx(out_embedding)
        key_padding_mask = ~masks
        
        aa=input['restype']
        res_nb=input['res_nb']
        chain_nb=input['chain_nb']
        pos_atoms=input['pos_atoms']
        mask_atoms=input['mask_atoms']
        
        if self.pooling == 'token':
            mask_special = torch.zeros((len(out_embedding), 1), device=out_embedding.device, dtype=key_padding_mask.dtype)
            cplx_embed = self.complex_embedding.repeat(len(out_embedding), 1, 1)
            prot_embed = self.prot_embedding.repeat(len(out_embedding), 1, 1)
            rna_embed = self.rna_embedding.repeat(len(out_embedding), 1, 1)
            
            out_embedding = torch.cat([cplx_embed, prot_embed, rna_embed, out_embedding], dim=1)
            key_padding_mask = torch.cat([mask_special, mask_special, mask_special, key_padding_mask], dim=1)
            
            cplx_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_CPLX_IDX
            prot_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_PROT_IDX
            rna_type = torch.ones_like(mask_special, device=out_embedding.device, dtype=aa.dtype) * SUPER_RNA_IDX
            aa = torch.cat([cplx_type, prot_type, rna_type, aa], dim=1)
            
            res_nb_cplx = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 0
            res_nb_prot = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 1
            res_nb_rna = torch.ones_like(mask_special, device=out_embedding.device, dtype=res_nb.dtype) * 2
            
            res_nb = torch.cat([res_nb_cplx, res_nb_prot, res_nb_rna, res_nb], dim=1)
            super_chain_id = torch.ones_like(mask_special, device=out_embedding.device, dtype=chain_nb.dtype) * SUPER_CHAIN_IDX
            chain_nb = torch.cat([super_chain_id, super_chain_id, super_chain_id, chain_nb], dim=1)
            
            center_cplx = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2], 3), device=out_embedding.device, dtype=pos_atoms.dtype)
            center_prot = ((pos_atoms * (1-input['identifier'])[:, :, None, None] * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((1-input['identifier'][:, :, None]) * mask_atoms + 1e-10).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, 4, 1)
            center_rna = ((pos_atoms * (input['identifier'][:, :, None, None]) * mask_atoms.unsqueeze(-1)).reshape([len(out_embedding), -1, 3]).sum(dim=1) / ((input['identifier'][:, :, None]) * mask_atoms + 1e-10).reshape([len(out_embedding), -1]).sum(dim=-1).unsqueeze(-1))[:, None, None, :].repeat(1, 1, 4, 1)
            pos_atoms = torch.cat([center_cplx, center_prot, center_rna, pos_atoms], dim=1)
            # noise = torch.randn_like(pos_atoms, dtype=torch.float32, device=pos_atoms.device)
            # pos_atoms += noise
            mask_atom = torch.zeros((len(out_embedding), 1, pos_atoms.shape[2]), device=out_embedding.device, dtype=mask_atoms.dtype)
            mask_atom[:,:,0] = 1
            mask_atoms = torch.cat([mask_atom, mask_atom, mask_atom, mask_atoms], dim=1)


        z = self.pair_encoder(
            aa=aa,
            res_nb=res_nb,
            chain_nb=chain_nb,
            pos_atoms=pos_atoms,
            mask_atoms=mask_atoms,
        )
        if need_mask:
            # Random mask rows/columns, 50% probability to mask 15% positions, 50% probability to keep the same
            for i in range(z.shape[0]):
                to_mask = torch.rand(1).item() > 0.5
                if not to_mask:
                    continue
                valid = list(range(3, z.shape[1]))
                mask_indices = random.sample(valid, int(len(valid) * 0.15))
                z[i, mask_indices, :, :] = self.mask_token.repeat(len(mask_indices), z.shape[2], 1)
                z[i, :, mask_indices, :] = self.mask_token.repeat(z.shape[1], len(mask_indices), 1)
            
        return out_embedding, z, key_padding_mask
        
    def forward(self, input, strategy='separate', stage='finetune', need_mask=False):
        out_embedding, z, key_padding_mask = self._forward(input, strategy, need_mask=need_mask)
        if stage == 'finetune':
                
            output, z, attn = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)

            complex_embedding = output + self.z_proj(z).sum(-2) * 0.001
            if self.pooling == 'token':
                pooled = output[:, 0, :]
            else:
                pooled = (output * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
                if self.pooling == 'mean':
                    seq_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
                    pooled = pooled / (seq_mask_sum + 1e-10)

            # Predict physical components (auxiliary, for regularization / interpretability)
            physics_outputs = {}
            for name in self.physics_names:
                physics_outputs[name] = self.physics_heads[name](pooled).squeeze(-1)

            # Main prediction from pred_head (this is the only term used for final prediction)
            main_pred = self.pred_head(pooled).squeeze(-1)

            return {
                'pred': main_pred,
                'physics': physics_outputs,
                'main_refinement': main_pred,
                'pooled': pooled
            }
            
        elif stage == 'pretune':
            # -------------------------------------------CLIP feature generation ----------------------------------------------
            res_identifier = input['identifier']
            attn_mask = torch.ones((out_embedding.shape[0], out_embedding.shape[1], out_embedding.shape[1]), device=out_embedding.device).bool()
            if self.pooling == 'token':
                prot_token_identifier = torch.zeros(len(out_embedding), 1, dtype=res_identifier.dtype, device=res_identifier.device)
                rna_token_identifier = torch.ones(len(out_embedding), 1, dtype=res_identifier.dtype, device=res_identifier.device)
                res_identifier = torch.cat([prot_token_identifier, rna_token_identifier, res_identifier], dim=1)
                attn_mask[:, 1:, 1:] = (res_identifier[:, :, None] == res_identifier[:, None, :])
            attn_mask = ~attn_mask
            # all the ones in transformer mask means ignoring, which is different from the meaning of pos_mask !!!!
            if torch.isnan(z).any():
                print("Found Nan in z!")
            output, z, _ = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False, attn_mask=attn_mask)
            
            # Output Embedding: [N, E]
            if self.pooling == 'token':
                complex_embedding = output[:, 0, :].squeeze(1)
                prot_embedding = output[:, 1, :].squeeze(1)
                rna_embedding = output[:, 2, :].squeeze(1)
            else:
                complex_embedding = (output * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
                prot_embedding = (output * (~key_padding_mask).unsqueeze(-1) * (1-input['identifier']).unsqueeze(-1)).sum(dim=1)
                rna_embedding = (output * (~key_padding_mask).unsqueeze(-1) * (input['identifier'].unsqueeze(-1))).sum(dim=1)
                if self.pooling == 'mean':
                    cplx_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
                    prot_mask_sum = ((~key_padding_mask) * (1-input['identifier'])).sum(dim=1, keepdim=True)
                    rna_mask_sum = ((~key_padding_mask) * (input['identifier'])).sum(dim=1, keepdim=True)
                    complex_embedding = complex_embedding / (cplx_mask_sum + 1e-10)
                    prot_embedding = prot_embedding / (prot_mask_sum + 1e-10)
                    rna_embedding = rna_embedding / (rna_mask_sum + 1e-10)
                    
            similarity = F.cosine_similarity(prot_embedding[:, None, :], rna_embedding[None, :, :], dim=2)

            if torch.isnan(z).any():
                print("Found Nan in z!")
            # ------------------------------------- Atom-level distance precdiction -------------------------------------------
            
            output, z, _ = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False, attn_mask=None)

            if torch.isnan(z).any():
                print("Found Nan in z!")

            dist_logits = self.dist_head(z)
            dist_logits = dist_logits[:, 3:, 3:, :]
            # dist_prob = F.softmax(dist_logits, dim=-1)
            return dist_logits, similarity 
        
        elif stage == 'mutation':
            input['prot'] = input['prot_mut']
            input['restype'] = input['mut_restype']
            out_mut, z_mut, _ = self._forward(input, strategy)
            deep = False
            if deep:
                out_forward = out_embedding - out_mut
                z_forward = z - z_mut
                
                out_inv = out_mut - out_embedding
                z_inv = z_mut - z
                
                
                output_forward, z_forward, attn = self.c_former(out_forward, z_forward, key_padding_mask=key_padding_mask, need_attn_weights=False)
                complex_embedding = output_forward + self.z_proj(z_forward).sum(-2) * 0.001
                # Default to be token embeding
                complex_embedding = complex_embedding[:, 0, :].squeeze(1)
                
                output_forward = self.pred_head(complex_embedding)
                output_forward = output_forward.squeeze(1)
                
                output_inv, z_inv, attn = self.c_former(out_inv, z_inv, key_padding_mask=key_padding_mask, need_attn_weights=False)
                complex_embedding_inv = output_inv + self.z_proj(z_inv).sum(-2) * 0.001
                # Default to be token embeding
                complex_embedding_inv = complex_embedding_inv[:, 0, :].squeeze(1)
                
                output_inv = self.pred_head(complex_embedding_inv)
                output_inv = output_inv.squeeze(1)
                
                return output_forward, output_inv
            else:
                output_wild, z_wild, attn = self.c_former(out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False)
                output_mut, z_mut, attn = self.c_former(out_mut, z_mut, key_padding_mask=key_padding_mask, need_attn_weights=False)
                wild_embedding = output_wild + self.z_proj(z_wild).sum(-2) * 0.001
                # Default to be token embeding
                wild_embedding = wild_embedding[:, 0, :].squeeze(1)
                mut_embedding = output_mut + self.z_proj(z_mut).sum(-2) * 0.001
                mut_embedding = mut_embedding[:, 0, :].squeeze(1)
                
                
                forward_embedding = wild_embedding - mut_embedding
                inv_embedding = mut_embedding - wild_embedding
                
                output_forward = self.pred_head(forward_embedding).squeeze(1)
                output_inv = self.pred_head(inv_embedding).squeeze(1)
                
                return output_forward, output_inv

        else:
            raise NotImplementedError
            

        
        
            
            
