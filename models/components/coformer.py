import math

import torch
from torch import nn
from torch.nn import functional as F

from models.components.rope import RotaryPositionEmbedding

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    In the cited paper beta is set to 1 and is not learnable;
    but by the Swish definition it is learnable parameter otherwise
    it is SiLU activation function (https://paperswithcode.com/method/swish)
    """
    def __init__(self, size_in, size_out, beta_is_learnable=True, bias=True):
        """
        Args:
            size_in: input embedding dimension
            size_out: output embedding dimension
            beta_is_learnable: whether beta is learnable or set to 1, learnable by default
            bias: whether use bias term, enabled by default
        """
        super().__init__()
        self.linear = nn.Linear(size_in, size_out, bias=bias)
        self.linear_gate = nn.Linear(size_in, size_out, bias=bias)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=beta_is_learnable)  

    def forward(self, x):
        linear_out = self.linear(x)
        swish_out = linear_out * torch.sigmoid(self.beta * linear_out)
        return swish_out * self.linear_gate(x)


class CoFormer(nn.Module):
    def __init__(self, embed_dim, pair_dim, num_blocks, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4, use_flash_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.use_rot_emb = use_rot_emb
        self.use_flash_attn = use_flash_attn
        self.struct_proj = nn.Linear(pair_dim * 2, embed_dim)

        self.blocks = nn.ModuleList(
            [
                SequenceStructureFusionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    use_rot_emb=use_rot_emb,
                    attn_qkv_bias=attn_qkv_bias,
                    transition_dropout=transition_dropout,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    transition_factor=transition_factor,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, struct_embed, key_padding_mask=None, need_attn_weights=False, attn_mask=None):
        # Structure pooling: reduce 2D pairwise features to a per-token 1D representation.
        struct_pool = self._pool_structure(struct_embed, key_padding_mask)
        seq = x
        for block in self.blocks:
            seq = block(seq, struct_pool, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        F_out = self.final_layer_norm(seq)
        return F_out

    def _pool_structure(self, struct_embed, key_padding_mask=None):
        if key_padding_mask is None:
            row_pool = struct_embed.mean(dim=2)
            col_pool = struct_embed.mean(dim=1)
        else:
            valid = (~key_padding_mask).float()
            col_mask = valid.unsqueeze(1).unsqueeze(-1)
            row_mask = valid.unsqueeze(2).unsqueeze(-1)
            row_sum = (struct_embed * col_mask).sum(dim=2)
            col_sum = (struct_embed * row_mask).sum(dim=1)
            col_count = col_mask.sum(dim=2).clamp_min(1.0)
            row_count = row_mask.sum(dim=1).clamp_min(1.0)
            row_pool = row_sum / col_count
            col_pool = col_sum / row_count
            row_pool = row_pool * valid.unsqueeze(-1)
            col_pool = col_pool * valid.unsqueeze(-1)

        struct_pool = torch.cat([row_pool, col_pool], dim=-1)
        struct_pool = self.struct_proj(struct_pool)
        return struct_pool


class SequenceStructureFusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, use_rot_emb=True, attn_qkv_bias=False, transition_dropout=0.0, attention_dropout=0.0, residual_dropout=0.0, transition_factor=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimensionality must be divisible by the number of attention heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_rot_emb = use_rot_emb

        if use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.head_dim)

        self.seq_norm = nn.LayerNorm(embed_dim)
        self.struct_norm = nn.LayerNorm(embed_dim)

        self.seq_q_proj = nn.Linear(embed_dim, embed_dim, bias=attn_qkv_bias)
        self.struct_k_proj = nn.Linear(embed_dim, embed_dim, bias=attn_qkv_bias)
        self.struct_v_proj = nn.Linear(embed_dim, embed_dim, bias=attn_qkv_bias)

        self.struct_q_proj = nn.Linear(embed_dim, embed_dim, bias=attn_qkv_bias)
        self.seq_k_proj = nn.Linear(embed_dim, embed_dim, bias=attn_qkv_bias)
        self.seq_v_proj = nn.Linear(embed_dim, embed_dim, bias=attn_qkv_bias)

        self.attn_dropout = nn.Dropout(p=attention_dropout)
        self.residual_dropout = nn.Dropout(p=residual_dropout)

        self.transition = nn.Sequential(
            SwiGLU(embed_dim, int(2 / 3 * transition_factor * embed_dim), beta_is_learnable=True, bias=True),
            nn.Dropout(p=transition_dropout),
            nn.Linear(int(2 / 3 * transition_factor * embed_dim), embed_dim, bias=True),
        )

        self.fuse_layer_norm = nn.LayerNorm(embed_dim)
        self.output_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, seq_embed, struct_pool, key_padding_mask=None, attn_mask=None):
        seq_norm = self.seq_norm(seq_embed)
        struct_norm = self.struct_norm(struct_pool)

        # Bidirectional attention: sequence queries structure features and vice versa.
        seq_queries = self.seq_q_proj(seq_norm)
        struct_keys = self.struct_k_proj(struct_norm)
        struct_values = self.struct_v_proj(struct_norm)
        seq_to_struct_attn = self._scaled_attention(
            seq_queries,
            struct_keys,
            struct_values,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        struct_queries = self.struct_q_proj(struct_norm)
        seq_keys = self.seq_k_proj(seq_norm)
        seq_values = self.seq_v_proj(seq_norm)
        struct_to_seq_attn = self._scaled_attention(
            struct_queries,
            seq_keys,
            seq_values,
            attn_mask=self._transpose_mask(attn_mask),
            key_padding_mask=key_padding_mask,
        )

        # Feature fusion: combine original sequence with both attentional updates.
        fused = seq_embed + self.residual_dropout(seq_to_struct_attn) + self.residual_dropout(struct_to_seq_attn)
        fused_norm = self.fuse_layer_norm(fused)
        fused = fused + self.residual_dropout(self.transition(fused_norm))
        fused = self.output_layer_norm(fused)

        if key_padding_mask is not None:
            fused = fused.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return fused

    def _scaled_attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        bsz, q_len, _ = q.size()
        q = self._prepare_heads(q)
        k = self._prepare_heads(k)
        v = self._prepare_heads(v)

        if self.use_rot_emb:
            q, k = self.rotary_emb(q, k)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                expanded_mask = attn_mask.unsqueeze(1)
            else:
                expanded_mask = attn_mask
            attn = attn.masked_fill(expanded_mask, float("-inf"))

        if key_padding_mask is not None:
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)

        if key_padding_mask is not None:
            output = output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return output

    def _prepare_heads(self, tensor):
        bsz, seq_len, _ = tensor.size()
        tensor = tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return tensor

    def _transpose_mask(self, attn_mask):
        if attn_mask is None:
            return None
        return attn_mask.transpose(-1, -2)
