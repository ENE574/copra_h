import torch
import torch.nn as nn
import torch.nn.functional as F
from models.register import ModelRegister

R = ModelRegister()

class MultiSourceAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, q, kv_list, mask=None):
        # q: [B, L, D]
        # kv_list: list of 3 [B, L, D]
        out_list = []
        for kv in kv_list:
            # mask: [B, L] -> key_padding_mask: [B, L] (True where padding)
            key_padding_mask = ~mask if mask is not None else None
            attn_out, _ = self.mha(q, kv, kv, key_padding_mask=key_padding_mask)
            out_list.append(attn_out)

        # Aggregate multi-source attention results (baseline: mean)
        return torch.stack(out_list, dim=0).mean(dim=0)

@R.register('copra_offline12')
class CopraOffline12(nn.Module):
    def __init__(self, 
                 d_model=512, 
                 num_heads=8, 
                 output_dim=1,
                 ablate=None,
                 **kwargs):
        super().__init__()
        self.ablate = ablate or []
        
        # Dimensions of 12 models
        self.dims = {
            'prot_seq': [1280, 1024, 1280],      # esm2, prott5, saprot
            'prot_struct': [512, 1024, 1024],    # esm_if1, protbert, protrek
            'rna_seq': [1280, 640, 768],         # rinalmo, rna_fm, rna_msm
            'rna_struct': [768, 384, 120]       # ernie_rna, rhfold(rhofold), rnabert
        }
        
        # 0. Pre-normalization for 12 models
        self.pre_lns = nn.ModuleDict()
        for group, dims in self.dims.items():
            for i, d in enumerate(dims):
                self.pre_lns[f'{group}_{i}'] = nn.LayerNorm(d)
        
        # 1. Projections to unified d_model
        self.projs = nn.ModuleDict()
        for group, dims in self.dims.items():
            for i, d in enumerate(dims):
                self.projs[f'{group}_{i}'] = nn.Linear(d, d_model)
        
        # 2. Intra-Group Fusion (3-way)
        # Q is combined from 3 models
        self.fusion_q = nn.ModuleDict({
            name: nn.Linear(d_model * 3, d_model) for name in self.dims.keys()
        })
        self.group_attn = nn.ModuleDict({
            name: MultiSourceAttention(d_model, num_heads) for name in self.dims.keys()
        })
        
        # 3. Entity-Level Cross Interaction (Structure <-> Sequence)
        self.intra_entity_attn = nn.ModuleDict({
            'prot_str_to_seq': MultiSourceAttention(d_model, num_heads),
            'prot_seq_to_str': MultiSourceAttention(d_model, num_heads),
            'rna_str_to_seq': MultiSourceAttention(d_model, num_heads),
            'rna_seq_to_str': MultiSourceAttention(d_model, num_heads),
        })
        
        # 4. Cross-Entity Interaction (Protein <-> RNA)
        self.cross_entity_attn = nn.ModuleDict({
            'prot_to_rna': MultiSourceAttention(d_model, num_heads),
            'rna_to_prot': MultiSourceAttention(d_model, num_heads),
        })
        
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, batch, strategy=None, stage=None, **kwargs):
        # Extract offline embeddings from batch
        # offline_prot_seq_embeds: list of 3 [B, Lp, D]
        p_seq_raw = batch['offline_prot_seq_embeds']
        p_str_raw = batch['offline_prot_struct_embeds']
        r_seq_raw = batch['offline_rna_seq_embeds']
        r_str_raw = batch['offline_rna_struct_embeds']
        
        p_mask = batch['offline_prot_mask']
        r_mask = batch['offline_rna_mask']
        
        # 0. Pre-normalize and project
        p_seq = [self.projs[f'prot_seq_{i}'](self.pre_lns[f'prot_seq_{i}'](t)) for i, t in enumerate(p_seq_raw)]
        p_str = [self.projs[f'prot_struct_{i}'](self.pre_lns[f'prot_struct_{i}'](t)) for i, t in enumerate(p_str_raw)]
        r_seq = [self.projs[f'rna_seq_{i}'](self.pre_lns[f'rna_seq_{i}'](t)) for i, t in enumerate(r_seq_raw)]
        r_str = [self.projs[f'rna_struct_{i}'](self.pre_lns[f'rna_struct_{i}'](t)) for i, t in enumerate(r_str_raw)]
        
        # 2. Group Fusion
        def fuse_group(tensors, mask, name):
            q_combined = self.fusion_q[name](torch.cat(tensors, dim=-1))
            return self.group_attn[name](q_combined, tensors, mask)
        
        p_seq_f = fuse_group(p_seq, p_mask, 'prot_seq')
        p_str_f = fuse_group(p_str, p_mask, 'prot_struct')
        r_seq_f = fuse_group(r_seq, r_mask, 'rna_seq')
        r_str_f = fuse_group(r_str, r_mask, 'rna_struct')
        
        # 3. Intra-Entity Cross Interaction (Sequence <-> Structure)
        # Prot
        if 'prot_seq' in self.ablate:
            p_entity = p_str_f
        elif 'prot_struct' in self.ablate:
            p_entity = p_seq_f
        else:
            p_seq_inter = self.intra_entity_attn['prot_seq_to_str'](p_seq_f, [p_str_f], p_mask)
            p_str_inter = self.intra_entity_attn['prot_str_to_seq'](p_str_f, [p_seq_f], p_mask)
            p_entity = p_seq_inter + p_str_inter
            
        # RNA
        if 'rna_seq' in self.ablate:
            r_entity = r_str_f
        elif 'rna_struct' in self.ablate:
            r_entity = r_seq_f
        else:
            r_seq_inter = self.intra_entity_attn['rna_seq_to_str'](r_seq_f, [r_str_f], r_mask)
            r_str_inter = self.intra_entity_attn['rna_str_to_seq'](r_str_f, [r_seq_f], r_mask)
            r_entity = r_seq_inter + r_str_inter
        
        # 4. Cross-Entity Interaction (Prot <-> RNA)
        if 'prot' in self.ablate:
            p_final = p_entity
            r_final = r_entity
        elif 'rna' in self.ablate:
            p_final = p_entity
            r_final = r_entity
        else:
            p_final = self.cross_entity_attn['prot_to_rna'](p_entity, [r_entity], r_mask)
            r_final = self.cross_entity_attn['rna_to_prot'](r_entity, [p_entity], p_mask)
        
        # 5. Global Pooling & Prediction
        # Simple mean pooling over tokens
        def global_pool(x, mask):
            # x: [B, L, D], mask: [B, L]
            mask_expand = mask.unsqueeze(-1).expand_as(x)
            return (x * mask_expand).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-6)
            
        p_vec = global_pool(p_final, p_mask)
        r_vec = global_pool(r_final, r_mask)
        
        # Combine (add) and predict
        combined = p_vec + r_vec
        pred = self.pred_head(combined).squeeze(-1)
        
        return pred
