import torch.nn as nn
import torch
from models.encoders.pair import ResiduePairEncoder
from models.register import ModelRegister
from models.components.coformer import CoFormer
from models.components.mut_local_gnn import (
    MutLocalGNN,
    _get_cb_positions_batch,
    extract_local_mutation_window,
)
import torch.nn.functional as F
from data.complex import SUPER_PROT_IDX, SUPER_RNA_IDX, SUPER_CPLX_IDX, SUPER_CHAIN_IDX
from data.entity_types import EntityPairSpec, EntityType, resolve_entity_pair_from_args
from data.pia_physics_names import PIA_PHYSICS_NAMES

R = ModelRegister()

# ---------------------------------------------------------------------------
# Shared utility functions (used by other model files: esm_rinalmo_seq, ipa)
# ---------------------------------------------------------------------------

def load_esm(esm_type):
    import esm
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


def load_nucleotide_transformer(model_path="weights/NucleotideTransformer_weights"):
    from transformers import AutoModelForMaskedLM
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True, output_hidden_states=True
    )
    model.eval()
    feat_size = model.config.hidden_size  # 1280 for 500M model
    return model, feat_size


def load_rinalmo(rinalmo_weights, rinalmo_type):
    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo
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
    model.load_state_dict(torch.load(rinalmo_weights))
    feat_size = config.globals.embed_dim
    return model, feat_size

def cat_pad(prot_embedding, prot_mask, na_embedding, na_mask, max_len, patch_idx):
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
        masks.append(mask.unsqueeze(0))
        new_complexes.append(selected_pad.unsqueeze(0))
    result = torch.cat(new_complexes, dim=0)
    masks = torch.cat(masks, dim=0).bool()
    return result, masks

# ---------------------------------------------------------------------------

@R.register('copra')
class ESM2RiNALMo(nn.Module):
    def __init__(self, 
                 pooling='mean',
                 output_dim=1,
                 pair_dim=320,
                 dist_dim=40,
                 offline_dim=1280,
                 main_refinement_scale=0.1,
                 ablate_interactions=None,
                 ablate_fusion_all=False,
                 **kwargs
                 ):
        super(ESM2RiNALMo, self).__init__()
        self.ablate_interactions = ablate_interactions or []
        self.ablate_fusion_all = ablate_fusion_all
        # Always use offline embeddings mode.
        self.use_offline_embeddings = True

        self.pair_encoder = ResiduePairEncoder(pair_dim, max_num_atoms=4)  # N, CA, C, O,
        self.c_former = CoFormer(**kwargs['coformer'])
        self.complex_dim = kwargs['coformer']['embed_dim']
        self.feat_size = offline_dim
        self.proj_cplx = nn.Linear(self.feat_size, self.complex_dim)

        # =========================== Offline embedding modules ===========================
        # Each group has exactly ONE model.
        def _as_list(x):
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            return list(x)

        # Known model input dims (single model per group).
        self._offline_dims = {
            "prot_seq": {"esm2": 1280},
            "prot_struct": {"esm_if1": 512},
            "rna_seq": {"rna_fm": 640},
            "rna_struct": {"rhofold": 384},
            "dna_seq": {"dnabert2": 768},
            "dna_struct": {"rf2na": 384},
        }

        # Default: exactly one model per group.
        self._all_offline_models = {
            "prot_seq": ["esm2"],
            "prot_struct": ["esm_if1"],
            "rna_seq": ["rna_fm"],
            "rna_struct": ["rhofold"],
            "dna_seq": ["dnabert2"],
            "dna_struct": ["rf2na"],
        }

        enabled = {
            "prot_seq": set(_as_list(kwargs.get("seq_prot_models"))),
            "rna_seq": set(_as_list(kwargs.get("seq_rna_models"))),
            "dna_seq": set(_as_list(kwargs.get("seq_dna_models"))),
            "prot_struct": set(_as_list(kwargs.get("str_prot_models"))),
            "rna_struct": set(_as_list(kwargs.get("str_rna_models"))),
            "dna_struct": set(_as_list(kwargs.get("str_dna_models"))),
        }

        disable_all_groups = set()
        for group in list(enabled.keys()):
            if enabled[group] == {"none"}:
                enabled[group] = set()
                disable_all_groups.add(group)

        # Backward compat: if seq_rna_models / str_rna_models contain DNA models
        # (e.g. dnabert2, rf2na) and seq_dna_models / str_dna_models are not set,
        # auto-route them to dna_seq / dna_struct groups.
        for na_type in ("dna",):
            seq_key = f"{na_type}_seq"
            str_key = f"{na_type}_struct"
            rna_seq_key = "rna_seq"
            rna_str_key = "rna_struct"
            if not enabled[seq_key] and seq_key in self._offline_dims:
                rna_models = enabled.get(rna_seq_key, set())
                dna_specific = {m for m in rna_models if m in self._offline_dims[seq_key]}
                if dna_specific:
                    enabled[seq_key] = dna_specific
                    enabled[rna_seq_key] = rna_models - dna_specific
            if not enabled[str_key] and str_key in self._offline_dims:
                rna_models = enabled.get(rna_str_key, set())
                dna_specific = {m for m in rna_models if m in self._offline_dims[str_key]}
                if dna_specific:
                    enabled[str_key] = dna_specific
                    enabled[rna_str_key] = rna_models - dna_specific

        for group, default_ms in self._all_offline_models.items():
            if len(enabled[group]) == 0 and group not in disable_all_groups:
                enabled[group] = set(default_ms)
        self._enabled_offline_models = enabled
        self._enabled_offline_model_lists = {
            group: [m for m in self._offline_dims[group] if m in enabled[group]]
            for group in enabled
        }

        def _make_proj(d_in):
            return nn.Linear(d_in, self.feat_size)

        self.offline_proj = nn.ModuleDict()
        for group, m2d in self._offline_dims.items():
            for m, d_in in m2d.items():
                self.offline_proj["{}/{}".format(group, m)] = _make_proj(d_in)

        # Use the same head count as CoFormer for cross-attention.
        self.offline_num_heads = int(kwargs["coformer"]["num_heads"])
        self._head_dim = self.feat_size // self.offline_num_heads
        assert self.feat_size % self.offline_num_heads == 0, "offline_dim must be divisible by coformer.num_heads"

        # Entity-internal seq<->struct bidirectional cross-attention
        self.prot_seq_to_struct = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
        self.prot_struct_to_seq = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
        self.rna_seq_to_struct = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
        self.rna_struct_to_seq = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
        self.dna_seq_to_struct = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
        self.dna_struct_to_seq = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)

        # Cross-entity bidirectional attention
        self.prot_to_rna = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
        self.rna_to_prot = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
        self.prot_to_dna = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
        self.dna_to_prot = nn.MultiheadAttention(self.feat_size, self.offline_num_heads, batch_first=True)
                
        self.pooling = pooling
        self.mutation_diff_mode = str(kwargs.get("mutation_diff_mode", "full")).lower()
        self.mutation_local_window = int(kwargs.get("mutation_local_window", 16))
        default_local_pool = "center" if self.mutation_diff_mode == "local" else "mean"
        self.mutation_local_pool = str(kwargs.get("mutation_local_pool", default_local_pool)).lower()
        self.mutation_ddg_use_struct = bool(kwargs.get("mutation_ddg_use_struct", self.mutation_diff_mode == "local"))
        self.mutation_ddg_residual = bool(kwargs.get("mutation_ddg_residual", False))
        self.mutation_local_gnn = bool(kwargs.get("mutation_local_gnn", False))
        self.mutation_local_gnn_k = int(kwargs.get("mutation_local_gnn_k", 8))
        self.mutation_local_gnn_layers = int(kwargs.get("mutation_local_gnn_layers", 2))
        self.mutation_ddg_wt_context_fuse = bool(
            kwargs.get("mutation_ddg_wt_context_fuse", self.mutation_diff_mode == "local")
        )
        self.entity_pair: EntityPairSpec = resolve_entity_pair_from_args(kwargs)
        self.entity_b_use_protein_fusion = bool(
            kwargs.get(
                "entity_b_use_protein_fusion",
                self.entity_pair.entity_b_is_protein,
            )
        )
        self.mutation_task = str(
            kwargs.get("mutation_task", self.entity_pair.mutation_task)
        )
        print(
            "Entity pair:",
            self.entity_pair.interaction,
            "A=%s B=%s mutation_task=%s"
            % (
                self.entity_pair.entity_a_type.value,
                self.entity_pair.entity_b_type.value,
                self.mutation_task,
            ),
        )
        print("Pooling Strategy:", self.pooling)
        if self.mutation_diff_mode != "full":
            print(
                "Mutation diff mode:", self.mutation_diff_mode,
                "local window:", self.mutation_local_window,
                "local pool:", self.mutation_local_pool,
                "ddg struct branch:", self.mutation_ddg_use_struct,
                "ddg residual head:", self.mutation_ddg_residual,
                "local gnn:", self.mutation_local_gnn,
                "wt context fuse:", self.mutation_ddg_wt_context_fuse,
            )

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
        
        # PIA heads = FoldX ``Stability`` field names (see ``data/pia_physics_names``).
        self.physics_names = list(PIA_PHYSICS_NAMES)
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
        # Learnable affine output calibration (scale + bias) to correct systematic
        # under-scaling of predicted ddG magnitude (e.g. pred = 0.56*true + 0.32).
        # Initialized to identity; controlled by config flag ddg_output_calibrate (default off).
        self.ddg_output_calibrate = bool(kwargs.get("ddg_output_calibrate", False))
        if self.ddg_output_calibrate:
            self.ddg_calib_scale = nn.Parameter(torch.ones(1))
            self.ddg_calib_bias = nn.Parameter(torch.zeros(1))
        else:
            self.ddg_calib_scale = None
            self.ddg_calib_bias = None
        if self.mutation_diff_mode == "local" and self.mutation_ddg_use_struct:
            self.ddg_local_fuse = nn.Linear(self.complex_dim * 2, self.complex_dim)
        else:
            self.ddg_local_fuse = None
        if self.mutation_diff_mode == "local" and self.mutation_local_gnn:
            self.mut_local_gnn = MutLocalGNN(
                self.complex_dim,
                k=self.mutation_local_gnn_k,
                num_layers=self.mutation_local_gnn_layers,
            )
            # Start near raw center (sigmoid(-1.1) ~ 0.25) to avoid GNN over-smoothing.
            self.mutation_local_gnn_gate = nn.Parameter(torch.tensor(-1.1, dtype=torch.float32))
        else:
            self.mut_local_gnn = None
            self.mutation_local_gnn_gate = None
        if self.mutation_diff_mode == "local" and self.mutation_ddg_wt_context_fuse:
            self.ddg_wt_delta = nn.Linear(self.complex_dim * 2, self.complex_dim)
            # Small gated residual on wt context; starts as pure local diff.
            self.mutation_ddg_wt_context_gate = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32))
            with torch.no_grad():
                self.ddg_wt_delta.weight.zero_()
                self.ddg_wt_delta.bias.zero_()
        else:
            self.ddg_wt_delta = None
            self.mutation_ddg_wt_context_gate = None
        # dist_dim kept for backward config compat (no-op)

    def _offline_fuse_group(self, group_name, x_list):
        """Each group has exactly 1 model; simply return it."""
        if len(x_list) == 0:
            raise ValueError("No offline models enabled for group '{}'".format(group_name))
        return x_list[0]

    def _offline_to_residue_per_complex(self, x_token, chain_counts, chain_mask):
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

    def _offline_pad_list(self, xs, max_len):
        B = len(xs)
        D = int(xs[0].shape[-1]) if B > 0 else 0
        out = xs[0].new_zeros((B, max_len, D))
        pad_mask = torch.ones((B, max_len), device=out.device, dtype=torch.bool)
        for i, x in enumerate(xs):
            L = int(x.shape[0])
            if L > 0:
                out[i, :L] = x
                pad_mask[i, :L] = False
        return out, pad_mask

    def _offline_cross_entity(self, offline, prot_chains, prot_mask, na_chains, na_mask, input=None, branch="fused"):
        """
        Full offline dG embedding path up to (and including) protein<->RNA/DNA cross-attention.
        branch: fused (seq+struct), seq (sequence only), struct (structure only).
        Returns prot_final, na_final [B, T, D], prot_lens, na_lens (per complex).
        """
        branch = str(branch).lower()
        if branch not in {"fused", "seq", "struct"}:
            raise ValueError("branch must be one of fused, seq, struct")
        prot_seq_list = {k: None for k in offline["prot_seq"].keys()}
        prot_struct_list = {k: None for k in offline["prot_struct"].keys()}

        # Dynamic entity_b_type: per-batch from input dict (multi-task support), fallback to model default.
        # Note: default_collate transforms entity_pair values into lists (e.g. ["protein"]); unwrap here.
        if input is not None and "entity_pair" in input:
            entity_b_type_val = input["entity_pair"].get("entity_b_type", "rna")
            if isinstance(entity_b_type_val, list):
                entity_b_type_val = entity_b_type_val[0]
            entity_b_type_str = str(entity_b_type_val).lower()
            is_dna = entity_b_type_str == "dna"
            is_prot = entity_b_type_str == "protein"
        else:
            is_dna = self.entity_pair.entity_b_type == EntityType.DNA
            is_prot = self.entity_pair.entity_b_type == EntityType.PROTEIN
        if is_dna:
            na_seq_key = "dna_seq"
            na_struct_key = "dna_struct"
        else:
            na_seq_key = "rna_seq"
            na_struct_key = "rna_struct"
        # Projection / model-lookup group for partner entity (PPI: reuse prot_seq/prot_struct)
        na_seq_proj = "prot_seq" if is_prot else na_seq_key
        na_struct_proj = "prot_struct" if is_prot else na_struct_key

        # Handle mismatch between data storage keys and model keys
        # Data may be stored under "rna_seq"/"rna_struct" even for DNA when dna_models are empty
        if na_seq_key not in offline and "rna_seq" in offline:
            na_seq_key = "rna_seq"
        if na_struct_key not in offline and "rna_struct" in offline:
            na_struct_key = "rna_struct"

        na_seq_list = {k: None for k in offline.get(na_seq_key, {}).keys()}
        na_struct_list = {k: None for k in offline.get(na_struct_key, {}).keys()}

        na_lens = []

        for m in prot_seq_list:
            prot_seq_list[m], prot_lens = self._offline_to_residue_per_complex(
                offline["prot_seq"][m], prot_chains, prot_mask
            )
        for m in prot_struct_list:
            prot_struct_list[m], _ = self._offline_to_residue_per_complex(
                offline["prot_struct"][m], prot_chains, prot_mask
            )
        for m in na_seq_list:
            na_seq_list[m], na_lens = self._offline_to_residue_per_complex(
                offline[na_seq_key][m], na_chains, na_mask
            )
        for m in na_struct_list:
            na_struct_list[m], _ = self._offline_to_residue_per_complex(
                offline[na_struct_key][m], na_chains, na_mask
            )

        max_prot = max(prot_lens) if len(prot_lens) > 0 else 0
        max_na = max(na_lens) if len(na_lens) > 0 else 0

        def _proj(group, model, x):
            return self.offline_proj["{}/{}".format(group, model)](x)

        def _maybe_disable(group, model, x):
            enabled = getattr(self, "_enabled_offline_models", None)
            if enabled is None:
                return x
            if model not in enabled.get(group, set()):
                return x * 0.0
            return x

        def _prepare_group_tensors(group, list_dict, max_len):
            tensors = []
            for m in self._enabled_offline_model_lists.get(group, []):
                t, _ = self._offline_pad_list(list_dict[m], max_len)
                tensors.append(_maybe_disable(group, m, _proj(group, m, t)))
            return tensors

        def _primary_model(group, fallback_group=None):
            models = self._enabled_offline_model_lists.get(group, [])
            if models:
                return models[0]
            if fallback_group is not None:
                fb = self._enabled_offline_model_lists.get(fallback_group, [])
                if fb:
                    return fb[0]
            raise ValueError("No offline models enabled for group '{}'".format(group))

        prot_seq_tensors = _prepare_group_tensors("prot_seq", prot_seq_list, max_prot)
        prot_struct_tensors = _prepare_group_tensors("prot_struct", prot_struct_list, max_prot)
        na_seq_tensors = _prepare_group_tensors(na_seq_proj, na_seq_list, max_na)
        na_struct_tensors = _prepare_group_tensors(na_struct_proj, na_struct_list, max_na)

        enabled = getattr(self, "_enabled_offline_models", {})
        prot_has_seq = len(enabled.get("prot_seq", set())) > 0
        prot_has_struct = len(enabled.get("prot_struct", set())) > 0
        na_has_seq = len(enabled.get(na_seq_proj, set())) > 0
        na_has_struct = len(enabled.get(na_struct_proj, set())) > 0

        _prim_prot_seq = _primary_model("prot_seq", "prot_struct") if prot_has_seq or prot_has_struct else None
        _prim_na_seq = _primary_model(na_seq_proj, na_struct_proj) if na_has_seq or na_has_struct else None

        if _prim_prot_seq is not None and _prim_prot_seq in prot_seq_list:
            _, prot_pad_mask = self._offline_pad_list(
                prot_seq_list[_prim_prot_seq], max_prot
            )
        elif _prim_prot_seq is not None and _prim_prot_seq in prot_struct_list:
            _, prot_pad_mask = self._offline_pad_list(
                prot_struct_list[_prim_prot_seq], max_prot
            )
        else:
            prot_pad_mask = torch.zeros(1, max_prot, device=next(self.parameters()).device, dtype=torch.bool)

        if _prim_na_seq is not None and _prim_na_seq in na_seq_list:
            _, na_pad_mask = self._offline_pad_list(
                na_seq_list[_prim_na_seq], max_na
            )
        elif _prim_na_seq is not None and _prim_na_seq in na_struct_list:
            _, na_pad_mask = self._offline_pad_list(
                na_struct_list[_prim_na_seq], max_na
            )
        else:
            na_pad_mask = torch.zeros(1, max_na, device=next(self.parameters()).device, dtype=torch.bool)

        prot_seq = self._offline_fuse_group("prot_seq", prot_seq_tensors) if prot_seq_tensors else torch.zeros(1, max_prot, self.feat_size, device=next(self.parameters()).device)
        prot_struct = self._offline_fuse_group("prot_struct", prot_struct_tensors) if prot_struct_tensors else torch.zeros(1, max_prot, self.feat_size, device=next(self.parameters()).device)
        na_seq = self._offline_fuse_group(na_seq_proj, na_seq_tensors) if na_seq_tensors else torch.zeros(1, max_na, self.feat_size, device=next(self.parameters()).device)
        na_struct = self._offline_fuse_group(na_struct_proj, na_struct_tensors) if na_struct_tensors else torch.zeros(1, max_na, self.feat_size, device=next(self.parameters()).device)

        if branch == "seq":
            if not prot_has_seq:
                raise ValueError("branch='seq' requires enabled prot_seq offline models")
            if not na_has_seq:
                raise ValueError(f"branch='seq' requires enabled {na_seq_key} offline models")
            prot_entity = prot_seq
            na_entity = na_seq
        elif branch == "struct":
            if not prot_has_struct:
                raise ValueError("branch='struct' requires enabled prot_struct offline models")
            if not na_has_struct:
                raise ValueError(f"branch='struct' requires enabled {na_struct_key} offline models")
            prot_entity = prot_struct
            na_entity = na_struct
        else:
            if self.ablate_fusion_all:
                prot_entity = prot_seq + prot_struct
            elif not prot_has_struct:
                prot_entity = prot_seq
            elif not prot_has_seq:
                prot_entity = prot_struct
            else:
                prot_struct2, _ = self.prot_seq_to_struct(
                    query=prot_struct, key=prot_seq, value=prot_seq, key_padding_mask=prot_pad_mask
                )
                prot_seq2, _ = self.prot_struct_to_seq(
                    query=prot_seq, key=prot_struct, value=prot_struct, key_padding_mask=prot_pad_mask
                )
                prot_entity = prot_seq2 + prot_struct2

            if self.entity_b_use_protein_fusion:
                # PPI: entity B is protein — reuse protein seq/struct fusion on partner branch.
                if self.ablate_fusion_all:
                    na_entity = na_seq + na_struct
                elif not na_has_struct:
                    na_entity = na_seq
                elif not na_has_seq:
                    na_entity = na_struct
                else:
                    na_struct2, _ = self.prot_seq_to_struct(
                        query=na_struct, key=na_seq, value=na_seq, key_padding_mask=na_pad_mask
                    )
                    na_seq2, _ = self.prot_struct_to_seq(
                        query=na_seq, key=na_struct, value=na_struct, key_padding_mask=na_pad_mask
                    )
                    na_entity = na_seq2 + na_struct2
            elif is_dna:
                # DNA-specific fusion
                if self.ablate_fusion_all:
                    na_entity = na_seq + na_struct
                elif not na_has_struct:
                    na_entity = na_seq
                elif not na_has_seq:
                    na_entity = na_struct
                else:
                    na_struct2, _ = self.dna_seq_to_struct(
                        query=na_struct, key=na_seq, value=na_seq, key_padding_mask=na_pad_mask
                    )
                    na_seq2, _ = self.dna_struct_to_seq(
                        query=na_seq, key=na_struct, value=na_struct, key_padding_mask=na_pad_mask
                    )
                    na_entity = na_seq2 + na_struct2
            elif self.ablate_fusion_all:
                na_entity = na_seq + na_struct
            elif not na_has_struct:
                na_entity = na_seq
            elif not na_has_seq:
                na_entity = na_struct
            else:
                na_struct2, _ = self.rna_seq_to_struct(
                    query=na_struct, key=na_seq, value=na_seq, key_padding_mask=na_pad_mask
                )
                na_seq2, _ = self.rna_struct_to_seq(
                    query=na_seq, key=na_struct, value=na_struct, key_padding_mask=na_pad_mask
                )
                na_entity = na_seq2 + na_struct2

        if self.ablate_fusion_all or "cross_entity" in self.ablate_interactions:
            prot_final = prot_entity
            na_final = na_entity
        elif is_dna:
            prot_final, _ = self.prot_to_dna(
                query=prot_entity, key=na_entity, value=na_entity, key_padding_mask=na_pad_mask
            )
            na_final, _ = self.dna_to_prot(
                query=na_entity, key=prot_entity, value=prot_entity, key_padding_mask=prot_pad_mask
            )
        else:
            prot_final, _ = self.prot_to_rna(
                query=prot_entity, key=na_entity, value=na_entity, key_padding_mask=na_pad_mask
            )
            na_final, _ = self.rna_to_prot(
                query=na_entity, key=prot_entity, value=prot_entity, key_padding_mask=prot_pad_mask
            )

        return prot_final, na_final, prot_lens, na_lens

    def _offline_assemble_complex(self, prot_final, na_final, prot_lens, na_lens, input):
        """Concatenate [prot|na] per complex into a padded batch (before proj_cplx)."""
        max_len = input["pos_atoms"].shape[1]
        out_embedding = prot_final.new_zeros((len(prot_lens), max_len, self.feat_size))
        masks = torch.zeros((len(prot_lens), max_len), device=out_embedding.device, dtype=torch.bool)
        for i, (Lp, Ln) in enumerate(zip(prot_lens, na_lens)):
            combined = torch.cat([prot_final[i, :Lp], na_final[i, :Ln]], dim=0)
            if "patch_idx" in input:
                idx = input["patch_idx"][i]
                idx = idx[idx >= 0].long()
                combined = torch.index_select(combined, 0, idx)
            Lc = int(min(combined.shape[0], max_len))
            if Lc > 0:
                out_embedding[i, :Lc] = combined[:Lc]
                masks[i, :Lc] = True
        return out_embedding, masks

    def _offline_branch_complex(self, offline, input, branch="fused"):
        prot_final, rna_final, prot_lens, rna_lens = self._offline_cross_entity(
            offline,
            input["prot_chains"],
            input["protein_mask"],
            input["na_chains"],
            input["na_mask"],
            input=input,
            branch=branch,
        )
        emb, masks = self._offline_assemble_complex(
            prot_final, rna_final, prot_lens, rna_lens, input
        )
        return emb, masks

    def _encode_structure_pair(self, input):
        aa = input["restype"]
        z = self.pair_encoder(
            aa=aa,
            res_nb=input["res_nb"],
            chain_nb=input["chain_nb"],
            pos_atoms=input["pos_atoms"],
            mask_atoms=input["mask_atoms"],
        )
        return z

    def _forward_offline(self, input, offline):
        prot_final, rna_final, prot_lens, rna_lens = self._offline_cross_entity(
            offline,
            input["prot_chains"],
            input["protein_mask"],
            input["na_chains"],
            input["na_mask"],
            input=input,
        )
        out_embedding, masks = self._offline_assemble_complex(
            prot_final, rna_final, prot_lens, rna_lens, input
        )
        out_embedding = self.proj_cplx(out_embedding)
        key_padding_mask = ~masks
        z = self._encode_structure_pair(input)
        return out_embedding, z, key_padding_mask

    def _pool_local_mutation_diff(self, wt_emb, mt_emb, input, seq_masks):
        """Local wt-mt diff around the mutation site; pool to a single vector for ddG."""
        padding_mask = ~seq_masks
        mut_id = input["mut_identifier"]
        window = self.mutation_local_window
        pool_mode = self.mutation_local_pool
        diff = wt_emb - mt_emb

        pos_cb = None
        if self.mut_local_gnn is not None:
            if "pos_atoms" not in input or "mask_atoms" not in input:
                raise ValueError("mutation_local_gnn requires batch['pos_atoms'] and batch['mask_atoms']")
            pos_cb = _get_cb_positions_batch(input["pos_atoms"], input["mask_atoms"])

        local, valid_local, local_pos = extract_local_mutation_window(
            diff, padding_mask, mut_id, window, pos_cb=pos_cb
        )

        if self.mut_local_gnn is not None:
            local_proj = self.proj_cplx(local)
            raw_center = local_proj[:, window, :]
            gnn_out = self.mut_local_gnn(local_proj, local_pos, valid_local)
            gnn_center = gnn_out[:, window, :]
            gnn_alpha = torch.sigmoid(self.mutation_local_gnn_gate)
            if pool_mode == "center":
                pooled = raw_center + gnn_alpha * (gnn_center - raw_center)
            elif pool_mode == "sum":
                pooled = (gnn_out * valid_local.unsqueeze(-1).float()).sum(dim=1)
            elif pool_mode == "mean":
                denom = valid_local.sum(dim=1, keepdim=True).clamp(min=1).float()
                pooled = (gnn_out * valid_local.unsqueeze(-1).float()).sum(dim=1) / denom
            else:
                raise ValueError("mutation_local_pool must be one of center, mean, sum")
            return pooled

        if pool_mode == "center":
            pooled = local[:, window, :]
        elif pool_mode == "sum":
            pooled = (local * valid_local.unsqueeze(-1).float()).sum(dim=1)
        elif pool_mode == "mean":
            denom = valid_local.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled = (local * valid_local.unsqueeze(-1).float()).sum(dim=1) / denom
        else:
            raise ValueError("mutation_local_pool must be one of center, mean, sum")
        return self.proj_cplx(pooled)

    def _pool_wt_complex_context(self, wt_emb, wt_masks):
        """Mutation-invariant complex context from wt embeddings only."""
        emb = self.proj_cplx(wt_emb)
        mask = wt_masks.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (emb * mask).sum(dim=1) / denom

    def _pool_cformer_output(self, output, key_padding_mask, identifier=None):
        complex_embedding = output
        if self.pooling == "token":
            return complex_embedding[:, 0, :].squeeze(1)
        pooled = (complex_embedding * (~key_padding_mask).unsqueeze(-1)).sum(dim=1)
        if self.pooling == "mean":
            if identifier is None:
                seq_mask_sum = (~key_padding_mask).sum(dim=1, keepdim=True)
            else:
                seq_mask_sum = ((~key_padding_mask) * identifier).sum(dim=1, keepdim=True)
            pooled = pooled / (seq_mask_sum + 1e-10)
        return pooled

    def _coformer_pool(self, out_embedding, z, key_padding_mask, identifier=None):
        output, z, _ = self.c_former(
            out_embedding, z, key_padding_mask=key_padding_mask, need_attn_weights=False
        )
        output = output + self.z_proj(z).sum(-2) * 0.001
        return self._pool_cformer_output(output, key_padding_mask, identifier=identifier)

    def _apply_calib(self, raw, inverse=False):
        """Apply learnable affine calibration to a raw scalar prediction.

        For the inverse (negated-mutation) branch we flip the bias sign so that
        ``_apply_calib(raw_inv, inverse=True) == -_apply_calib(raw)`` holds when
        ``raw_inv == -raw`` (odd prediction head), preserving inversion consistency.
        """
        if self.ddg_calib_scale is None:
            return raw
        b = self.ddg_calib_bias if not inverse else -self.ddg_calib_bias
        return self.ddg_calib_scale * raw + b

    def _forward_mutation_offline(self, input, strategy="separate"):
        """
        ddG offline path: parallel wt/mt cross-entity flows, subtract, then one CoFormer on the diff.
        MT branch is also run for PIA auxiliary loss (pooled_mut).
        """
        if "offline_embeddings_mut" not in input:
            raise ValueError(
                "ddG offline mode requires batch['offline_embeddings_mut']. "
                "Use a mut dataset (mut: true) with offline_mutant_subdir configured."
            )
        wt_prot, wt_rna, prot_lens, rna_lens = self._offline_cross_entity(
            input["offline_embeddings"],
            input["prot_chains"],
            input["protein_mask"],
            input["na_chains"],
            input["na_mask"],
            input=input,
        )
        mt_prot, mt_rna, _, _ = self._offline_cross_entity(
            input["offline_embeddings_mut"],
            input["prot_chains"],
            input["protein_mask"],
            input["na_chains"],
            input["na_mask"],
            input=input,
        )

        wt_emb, wt_masks = self._offline_assemble_complex(
            wt_prot, wt_rna, prot_lens, rna_lens, input
        )
        mt_emb, mt_masks = self._offline_assemble_complex(
            mt_prot, mt_rna, prot_lens, rna_lens, input
        )

        pair_input = input
        if "mut_restype" in input:
            pair_input = {**input, "restype": input["mut_restype"]}
        z = self._encode_structure_pair(pair_input)

        if self.mutation_diff_mode == "local":
            if "mut_identifier" not in input:
                raise ValueError("mutation_diff_mode=local requires batch['mut_identifier']")
            if self.mutation_ddg_use_struct:
                wt_seq_emb, wt_masks = self._offline_branch_complex(
                    input["offline_embeddings"], input, branch="seq"
                )
                mt_seq_emb, _ = self._offline_branch_complex(
                    input["offline_embeddings_mut"], input, branch="seq"
                )
                wt_struct_emb, _ = self._offline_branch_complex(
                    input["offline_embeddings"], input, branch="struct"
                )
                mt_struct_emb, _ = self._offline_branch_complex(
                    input["offline_embeddings_mut"], input, branch="struct"
                )
                pooled_seq = self._pool_local_mutation_diff(wt_seq_emb, mt_seq_emb, input, wt_masks)
                pooled_struct = self._pool_local_mutation_diff(wt_struct_emb, mt_struct_emb, input, wt_masks)
                pooled_diff = self.ddg_local_fuse(torch.cat([pooled_seq, pooled_struct], dim=-1))
            else:
                pooled_diff = self._pool_local_mutation_diff(wt_emb, mt_emb, input, wt_masks)
            mt_emb = self.proj_cplx(mt_emb)
            pooled_mut = self._coformer_pool(mt_emb, z, ~mt_masks)
            if self.mutation_ddg_residual:
                return {
                    "ddg_mut_feat": pooled_diff,
                    "ddg_wt_context": self._pool_wt_complex_context(wt_emb, wt_masks),
                    "pooled_mut": pooled_mut,
                }
            pooled_for_pred = pooled_diff
            if self.ddg_wt_delta is not None:
                wt_ctx = self._pool_wt_complex_context(wt_emb, wt_masks)
                wt_alpha = torch.sigmoid(self.mutation_ddg_wt_context_gate)
                pooled_for_pred = pooled_diff + wt_alpha * self.ddg_wt_delta(
                    torch.cat([pooled_diff, wt_ctx], dim=-1)
                )
            ddg_pred = self._apply_calib(self.pred_head(pooled_for_pred).squeeze(-1))
            ddg_pred_inv = self._apply_calib(self.pred_head(-pooled_for_pred).squeeze(-1), inverse=True)
            return {
                "ddg_pred": ddg_pred,
                "ddg_pred_inv": ddg_pred_inv,
                "pooled_mut": pooled_mut,
            }

        diff_emb, diff_masks = self._offline_assemble_complex(
            wt_prot - mt_prot, wt_rna - mt_rna, prot_lens, rna_lens, input
        )
        diff_emb = self.proj_cplx(diff_emb)
        diff_key_padding_mask = ~diff_masks

        pooled_diff = self._coformer_pool(diff_emb, z, diff_key_padding_mask)
        ddg_pred = self._apply_calib(self.pred_head(pooled_diff).squeeze(-1))
        ddg_pred_inv = self._apply_calib(self.pred_head(-pooled_diff).squeeze(-1), inverse=True)

        mt_emb = self.proj_cplx(mt_emb)
        pooled_mut = self._coformer_pool(mt_emb, z, ~mt_masks)

        # ---- 能量分解项进入主预测（可解释加性融合）----
        # 用突变态(pooled_mut)与野生态(pooled_wt)分别过 physics_heads 得到两态绝对能量项，
        # Δterm = term(mut) - term(wt) 与 FoldX ddG = ΔE 语义对齐，
        # ddG ≈ data_driven + Σ wᵢ·Δtermᵢ 构成显式可解释分解。
        wt_emb = self.proj_cplx(wt_emb)  # project wt_emb to coformer dim before pooling (symmetric to mt_emb)
        pooled_wt = self._coformer_pool(wt_emb, z, ~wt_masks)
        phys_mt = {n: self.physics_heads[n](pooled_mut).squeeze(-1) for n in self.physics_names}
        phys_wt = {n: self.physics_heads[n](pooled_wt).squeeze(-1) for n in self.physics_names}
        phys_delta = {n: phys_mt[n] - phys_wt[n] for n in self.physics_names}
        phys_stack = torch.stack([phys_delta[n] for n in self.physics_names], dim=-1)
        phys_sum = (phys_stack * self.physics_weights).sum(dim=-1)
        ddg_pred = ddg_pred + self.main_refinement_scale * phys_sum
        ddg_pred_inv = ddg_pred_inv - self.main_refinement_scale * phys_sum

        return {
            "ddg_pred": ddg_pred,
            "ddg_pred_inv": ddg_pred_inv,
            "pooled_mut": pooled_mut,
            "physics": phys_delta,           # 各项 Δterm（与 FoldX ddG 语义一致）
            "physics_sum": phys_sum,         # Σ wᵢ·Δtermᵢ 物理分解对 ddG 的贡献
            "physics_weights": self.physics_weights,  # 各项权重（归因用）
        }

    def _forward(self, input, strategy='separate'):
        """Offline-only forward: assemble complex embeddings from pre-computed features."""
        if "offline_embeddings" not in input:
            raise ValueError(
                "offline_embeddings missing from batch. "
                "Set use_offline_embeddings: true and offline_embedding_root in the dataset YAML "
                "(see config/datasets/PRA310.yml)."
            )
        return self._forward_offline(input, input["offline_embeddings"])
        
    def forward(self, input, strategy='separate', stage='finetune'):
        if stage == 'mutation':
            return self._forward_mutation_offline(input, strategy=strategy)

        out_embedding, z, key_padding_mask = self._forward(input, strategy)
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

            # Main prediction from pred_head (data-driven term)
            data_pred = self._apply_calib(self.pred_head(pooled).squeeze(-1))

            # ---- 能量分解项进入主预测（可解释加性融合，dG 单态分支）----
            # dG 为单态（无 wt/mt 配对），故用绝对能量项 termᵢ 而非 Δtermᵢ。
            # dG ≈ data_driven + scale · Σ wᵢ·termᵢ 构成显式可解释分解，
            # 每项贡献 scale·wᵢ·termᵢ 可直接解析归因。
            phys_stack = torch.stack(
                [physics_outputs[n] for n in self.physics_names], dim=-1
            )
            physics_sum = (phys_stack * self.physics_weights).sum(dim=-1)
            main_pred = data_pred + self.main_refinement_scale * physics_sum

            return {
                'pred': main_pred,
                'pred_data': data_pred,                 # 纯数据驱动分量（归因用）
                'physics': physics_outputs,             # 各项绝对能量 termᵢ
                'physics_sum': physics_sum,             # Σ wᵢ·termᵢ 物理分解对 dG 的贡献
                'physics_weights': self.physics_weights,
                'main_refinement': main_pred,
                'pooled': pooled
            }
            
        else:
            raise NotImplementedError
            
        
        
            
            
