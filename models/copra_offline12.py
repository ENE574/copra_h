import torch
import torch.nn as nn
import torch.nn.functional as F
from models.register import ModelRegister
from data.pia_physics_names import PIA_PHYSICS_NAMES

R = ModelRegister()


@R.register('copra_offline12')
class CopraOffline12(nn.Module):
    """Simplified unified model for multi-entity ddG prediction."""
    def __init__(self, 
                 d_model=512, 
                 num_heads=8, 
                 output_dim=1,
                 ablate=None,
                 seq_prot_models=None,
                 str_prot_models=None,
                 seq_rna_models=None,
                 str_rna_models=None,
                 seq_dna_models=None,
                 str_dna_models=None,
                 **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ablate = ablate or []
        self.output_dim = output_dim
        
        self.seq_prot_models = seq_prot_models or ['esm2']
        self.str_prot_models = str_prot_models or ['esm_if1']
        self.seq_rna_models = seq_rna_models or ['rna_fm']
        self.str_rna_models = str_rna_models or ['rhofold']
        self.seq_dna_models = seq_dna_models or ['dnabert2']
        self.str_dna_models = str_dna_models or ['rf2na']
        
        self.projs = nn.ModuleDict()
        self.pre_lns = nn.ModuleDict()
        self.fusion_q = nn.ModuleDict()
        
        self._build_network()
        
        self.entity_type = None
        self.entity_groups = None
        
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
        self.pred_head_residual = nn.Linear(d_model, output_dim)
        
        self.physics_names = list(PIA_PHYSICS_NAMES)
        
        self.use_diff_pred = bool(kwargs.get('use_diff_pred', False))
    
    def _build_network(self):
        group_config = {
            'prot_seq': {
                'models': self.seq_prot_models,
                'dims': {'esm2': 1280, 'esm1v': 1280, 'esm1b': 1280}
            },
            'prot_struct': {
                'models': self.str_prot_models,
                'dims': {'esm_if1': 512, 'esm_if2': 512}
            },
            'rna_seq': {
                'models': self.seq_rna_models,
                'dims': {'rna_fm': 640}
            },
            'rna_struct': {
                'models': self.str_rna_models,
                'dims': {'rhofold': 384}
            },
            'dna_seq': {
                'models': self.seq_dna_models,
                'dims': {'dnabert2': 768}
            },
            'dna_struct': {
                'models': self.str_dna_models,
                'dims': {'rf2na': 640}
            }
        }
        
        for group_name, config in group_config.items():
            model_names = config['models']
            num_models = len(model_names)
            
            for i, model_name in enumerate(model_names):
                input_dim = config['dims'].get(model_name, 1280)
                self.pre_lns[f'{group_name}_{i}'] = nn.LayerNorm(input_dim)
                self.projs[f'{group_name}_{i}'] = nn.Linear(input_dim, self.d_model)
            
            self.fusion_q[group_name] = nn.Linear(self.d_model * num_models, self.d_model)
        
        self.cross_attn_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.cross_attn_residual = nn.Linear(self.d_model * 2, self.d_model)
        self.cross_attn_norm = nn.LayerNorm(self.d_model)

    def _infer_entity_type(self, batch):
        """Infer entity type from offline_embeddings keys."""
        emb_dict = batch['offline_embeddings']
        groups = sorted(emb_dict.keys())
        
        first_tensor = list(emb_dict.values())[0]
        first_tensor = list(first_tensor.values())[0] if isinstance(first_tensor, dict) else first_tensor
        batch_size = first_tensor.size(0)
        
        entity_a_groups = []
        entity_b_groups = []
        
        for group in groups:
            if group not in emb_dict:
                continue
            model_dict = emb_dict[group]
            model_tensor = list(model_dict.values())[0] if isinstance(model_dict, dict) else model_dict
            
            if model_tensor.size(0) != batch_size:
                continue
                
            if group.startswith('prot_'):
                entity_a_groups.append(group)
            elif group.startswith('rna_'):
                entity_b_groups.append(group)
                self.entity_type = 'prot_rna'
            elif group.startswith('dna_'):
                entity_b_groups.append(group)
                self.entity_type = 'prot_dna'
        
        if not self.entity_type and len(entity_a_groups) > 0:
            all_prot_groups = entity_a_groups.copy()
            mid = len(all_prot_groups) // 2
            entity_a_groups = all_prot_groups[:mid]
            entity_b_groups = all_prot_groups[mid:]
            self.entity_type = 'prot_prot'
        
        return entity_a_groups, entity_b_groups

    def _get_entity_features(self, emb_dict, groups):
        """Get concatenated features for an entity (before pooling)."""
        features_list = []
        expected_bs = None
        
        for group_name in groups:
            if group_name not in emb_dict:
                continue
                
            group_models = list(emb_dict[group_name].values())
            if not group_models:
                continue
            
            if expected_bs is None:
                expected_bs = group_models[0].size(0)
            
            projected = []
            for i, t in enumerate(group_models):
                if t.size(0) != expected_bs:
                    continue
                key = f'{group_name}_{i}'
                if key in self.projs:
                    p = self.projs[key](self.pre_lns[key](t))
                    projected.append(p)
            
            if projected:
                fused = torch.cat(projected, dim=-1)
                fused = self.fusion_q[group_name](fused)
                features_list.append(fused)
        
        if features_list:
            return torch.stack(features_list, dim=-1).mean(dim=-1)
        else:
            return torch.zeros(expected_bs if expected_bs else 1, 1, self.d_model, 
                             device=next(self.parameters()).device)

    def _cross_attention(self, q, k, v, mask=None):
        """Cross attention between two sequences."""
        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        
        q = q.view(batch_size, q_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, k_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, k_len, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model // self.num_heads, dtype=torch.float32))
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return output, attn_weights

    def _pool_entity(self, emb_dict, groups):
        """Pool all embeddings for an entity into a single vector."""
        features = self._get_entity_features(emb_dict, groups)
        return features.mean(dim=1)

    def forward(self, batch, strategy=None, stage=None, **kwargs):
        if self.entity_groups is None:
            entity_a_groups, entity_b_groups = self._infer_entity_type(batch)
            self.entity_groups = {
                'entity_a': entity_a_groups,
                'entity_b': entity_b_groups
            }
        
        use_diff_pred = bool(getattr(self, 'use_diff_pred', False))
        
        if use_diff_pred and 'offline_embeddings_mut' in batch:
            return self._forward_diff_pred(batch, strategy, stage, **kwargs)
        return self._forward_single(batch, strategy, stage, **kwargs)
    
    def _forward_single(self, batch, strategy=None, stage=None, **kwargs):
        emb_dict = batch['offline_embeddings']
        
        entity_a_feat = self._get_entity_features(emb_dict, self.entity_groups['entity_a'])
        entity_b_feat = self._get_entity_features(emb_dict, self.entity_groups['entity_b'])
        
        batch_size = entity_a_feat.size(0)
        
        entity_a_mask = batch.get('protein_mask', None)
        entity_b_mask = batch.get('na_mask', None)
        
        if entity_a_mask is not None and entity_a_mask.size(0) > batch_size:
            entity_a_mask = entity_a_mask[:batch_size]
        if entity_b_mask is not None and entity_b_mask.size(0) > batch_size:
            entity_b_mask = entity_b_mask[:batch_size]
        
        if entity_a_feat.size(1) == entity_b_feat.size(1):
            a_attended, _ = self._cross_attention(entity_a_feat, entity_b_feat, entity_b_feat, entity_b_mask)
            b_attended, _ = self._cross_attention(entity_b_feat, entity_a_feat, entity_a_feat, entity_a_mask)
            
            combined_feat = torch.cat([entity_a_feat + a_attended, entity_b_feat + b_attended], dim=-1)
            
            combined_feat_residual = self.cross_attn_residual(combined_feat.mean(dim=1))
            
            combined_feat = self.cross_attn_proj(combined_feat)
            combined_feat = self.cross_attn_norm(combined_feat)
            
            pooled = combined_feat.mean(dim=1) + combined_feat_residual
        else:
            pooled_a = entity_a_feat.mean(dim=1)
            pooled_b = entity_b_feat.mean(dim=1)
            pooled = pooled_a + pooled_b
        
        pred = self.pred_head(pooled).squeeze(-1) + self.pred_head_residual(pooled).squeeze(-1)
        
        return {'ddg_pred': pred, 'ddg_pred_inv': -pred, 'pooled_mut': pooled}

    def _forward_diff_pred(self, batch, strategy=None, stage=None, **kwargs):
        wt_emb = batch['offline_embeddings']
        mut_emb = batch['offline_embeddings_mut']
        
        target_bs = None
        if 'labels' in batch:
            target_bs = batch['labels'].size(0)
        
        if target_bs is None:
            def get_batch_size(emb_dict, groups):
                for g in groups:
                    if g in emb_dict:
                        t = list(emb_dict[g].values())[0]
                        return t.size(0)
                return None
            
            bs_wt_a = get_batch_size(wt_emb, self.entity_groups['entity_a'])
            bs_wt_b = get_batch_size(wt_emb, self.entity_groups['entity_b'])
            target_bs = min(bs_wt_a or 999, bs_wt_b or 999) if (bs_wt_a and bs_wt_b) else (bs_wt_a or bs_wt_b or 1)
        
        entity_a_wt = self._get_entity_features(wt_emb, self.entity_groups['entity_a'])
        entity_b_wt = self._get_entity_features(wt_emb, self.entity_groups['entity_b'])
        
        if entity_a_wt.size(0) > target_bs:
            entity_a_wt = entity_a_wt[:target_bs]
        if entity_b_wt.size(0) > target_bs:
            entity_b_wt = entity_b_wt[:target_bs]
        
        entity_a_mask = batch.get('protein_mask', None)
        entity_b_mask = batch.get('na_mask', None)
        
        if entity_a_mask is not None and entity_a_mask.size(0) > target_bs:
            entity_a_mask = entity_a_mask[:target_bs]
        if entity_b_mask is not None and entity_b_mask.size(0) > target_bs:
            entity_b_mask = entity_b_mask[:target_bs]
        
        if entity_a_wt.size(1) == entity_b_wt.size(1):
            a_attended_wt, _ = self._cross_attention(entity_a_wt, entity_b_wt, entity_b_wt, entity_b_mask)
            b_attended_wt, _ = self._cross_attention(entity_b_wt, entity_a_wt, entity_a_wt, entity_a_mask)
            
            combined_wt = torch.cat([entity_a_wt + a_attended_wt, entity_b_wt + b_attended_wt], dim=-1)
            
            combined_wt_residual = self.cross_attn_residual(combined_wt.mean(dim=1))
            
            combined_wt = self.cross_attn_proj(combined_wt)
            combined_wt = self.cross_attn_norm(combined_wt)
            pooled_wt = combined_wt.mean(dim=1) + combined_wt_residual
        else:
            pooled_wt = entity_a_wt.mean(dim=1) + entity_b_wt.mean(dim=1)
        
        entity_a_mut = self._get_entity_features(mut_emb, self.entity_groups['entity_a'])
        entity_b_mut = self._get_entity_features(mut_emb, self.entity_groups['entity_b'])
        
        if entity_a_mut.size(0) > target_bs:
            entity_a_mut = entity_a_mut[:target_bs]
        if entity_b_mut.size(0) > target_bs:
            entity_b_mut = entity_b_mut[:target_bs]
        
        if entity_a_mut.size(1) == entity_b_mut.size(1):
            a_attended_mut, _ = self._cross_attention(entity_a_mut, entity_b_mut, entity_b_mut, entity_b_mask)
            b_attended_mut, _ = self._cross_attention(entity_b_mut, entity_a_mut, entity_a_mut, entity_a_mask)
            
            combined_mut = torch.cat([entity_a_mut + a_attended_mut, entity_b_mut + b_attended_mut], dim=-1)
            
            combined_mut_residual = self.cross_attn_residual(combined_mut.mean(dim=1))
            
            combined_mut = self.cross_attn_proj(combined_mut)
            combined_mut = self.cross_attn_norm(combined_mut)
            pooled_mut = combined_mut.mean(dim=1) + combined_mut_residual
        else:
            pooled_mut = entity_a_mut.mean(dim=1) + entity_b_mut.mean(dim=1)
        
        dG_wt = self.pred_head(pooled_wt).squeeze(-1) + self.pred_head_residual(pooled_wt).squeeze(-1)
        dG_mut = self.pred_head(pooled_mut).squeeze(-1) + self.pred_head_residual(pooled_mut).squeeze(-1)
        
        ddG_pred = dG_mut - dG_wt
        ddG_pred_inv = -ddG_pred
        
        return {'ddg_pred': ddG_pred, 'ddg_pred_inv': ddG_pred_inv, 'pooled_mut': pooled_mut}
    
    def _pool_entity_fixed_bs(self, emb_dict, groups, target_bs):
        """Pool all embeddings for an entity into a single vector with fixed batch size."""
        pooled_list = []
        
        for group_name in groups:
            if group_name not in emb_dict:
                continue
                
            group_models = list(emb_dict[group_name].values())
            if not group_models:
                continue
            
            # Project and concatenate all models in this group
            projected = []
            for i, t in enumerate(group_models):
                # Only use tensors with target batch size
                if t.size(0) != target_bs:
                    continue
                key = f'{group_name}_{i}'
                if key in self.projs:
                    p = self.projs[key](self.pre_lns[key](t))
                    projected.append(p)
            
            if projected:
                fused = torch.cat(projected, dim=-1)
                fused = self.fusion_q[group_name](fused)
                pooled = fused.mean(dim=1)
                pooled_list.append(pooled)
        
        if pooled_list:
            return torch.stack(pooled_list, dim=0).mean(dim=0)
        else:
            return torch.zeros(target_bs, self.d_model, device=next(self.parameters()).device)
