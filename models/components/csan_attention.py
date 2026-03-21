import torch
import torch.nn as nn

from models.components.rope import RotaryPositionEmbedding


class CSANMultiHeadSelfAttention(nn.Module):
    """
    CoFormer-compatible CSAN multi-head self-attention.

    Interface (compatible with `MultiHeadSelfAttention` in `models/components/attention.py`):
      - x: [B, L, C]
      - struct_embed: [B, L, L, pair_dim]
      - attn_mask: (optional) [B, L, L] or [B, H, L, L], True means ignore
      - key_pad_mask: (optional) [B, L], True means ignore
    """

    def __init__(
        self,
        embed_dim,
        pair_dim,
        num_heads,
        attention_dropout=0.0,
        use_rot_emb=True,
        bias=False,
        c_squeeze=8,
        csan_eta=0.25,
        attn_qkv_bias=False,
        head_dim=None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim ({}) must be divisible by num_heads ({}).".format(embed_dim, num_heads))
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim ({}) must be divisible by 4 for CSAN gating.".format(embed_dim))

        self.embed_dim = embed_dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.c_squeeze = c_squeeze
        self.csan_eta = csan_eta

        self.head_dim = head_dim if head_dim is not None else (embed_dim // num_heads)
        if self.head_dim * self.num_heads != embed_dim:
            raise ValueError(
                "head_dim ({}) * num_heads ({}) must equal embed_dim ({}).".format(
                    self.head_dim, self.num_heads, embed_dim
                )
            )
        self.scale = self.head_dim ** -0.5

        self.use_rot_emb = use_rot_emb
        if self.use_rot_emb:
            self.rotary_emb = RotaryPositionEmbedding(self.head_dim)

        # struct_attr: [B, H, L, L]
        self.struct_bias = nn.Linear(pair_dim, num_heads, bias=bias)

        # CSAN gating networks (adapted from swin_csan's window attention)
        ratio = 8
        threshold = 24
        hidden_dim = embed_dim // ratio if (embed_dim // ratio) > threshold else threshold

        self.u = nn.Linear(embed_dim, embed_dim // 4)
        self.reduce = nn.Linear(embed_dim // 4, hidden_dim)
        self.act = nn.GELU()

        self.ka = nn.Linear(hidden_dim, embed_dim)
        self.kv = nn.Linear(hidden_dim, embed_dim)
        self.magk = nn.Linear(hidden_dim, num_heads)
        self.temp = nn.Linear(hidden_dim, num_heads)

        # qkv projections: [B, H, L, Dh]
        attn_dim = self.head_dim * self.num_heads  # == embed_dim
        self.qkv = nn.Linear(embed_dim, attn_dim * 3, bias=attn_qkv_bias)

        # dropout/softmax
        self.attn_drop = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

        # output projections + struct update (same as CoFormer's MultiHeadAttention)
        self.outer_squeeze = nn.Linear(embed_dim, c_squeeze)
        self.outer_linear = nn.Linear(c_squeeze * c_squeeze, pair_dim)
        self.struct_out_proj = nn.Linear(pair_dim, pair_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    @staticmethod
    def quasi_linear(x, eta):
        # A numerically-stable piecewise approximation used by CSAN.
        return torch.where(x > eta, x, eta * torch.exp(x.clamp(max=eta) / eta - 1))

    def forward(self, x, struct_embed, attn_mask=None, key_pad_mask=None):
        bs, seqlen, c_in = x.shape
        if c_in != self.embed_dim:
            raise ValueError("Unexpected x last dim {}, expected {}.".format(c_in, self.embed_dim))
        if struct_embed.shape[:3] != (bs, seqlen, seqlen):
            raise ValueError("Unexpected struct_embed shape {}, expected (B, L, L, pair_dim).".format(struct_embed.shape))

        qkv = self.qkv(x).reshape(bs, seqlen, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: [B, H, L, Dh]

        if self.use_rot_emb:
            q, k = self.rotary_emb(q, k)

        q = q * self.scale
        attn_logits = torch.matmul(q, k.transpose(-1, -2))  # [B, H, L, L]

        # Add structure into attention logits: struct_attr [B, H, L, L]
        struct_attr = self.struct_bias(struct_embed).permute(0, 3, 1, 2)
        attn_logits = attn_logits + struct_attr

        # Masking: True means ignore
        if attn_mask is not None:
            if attn_mask.dim() < attn_logits.dim():
                attn_mask = attn_mask[:, None, ...]  # [B, 1, L, L]
            if attn_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(attn_mask, float("-inf"))
            else:
                attn_logits = attn_logits + attn_mask

        if key_pad_mask is not None:
            attn_logits = attn_logits.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # CSAN gating (non-fused path)
        qscore = self.quasi_linear(attn_logits, self.csan_eta).mean(dim=-1, keepdim=True)  # [B, H, L, 1]
        u = self.u(x).view(bs, seqlen, self.num_heads, -1).transpose(1, 2)  # [B, H, L, Dh/4]
        ind = qscore * u.sigmoid()
        ind = ind.transpose(1, 2).reshape(bs, seqlen, -1)  # [B, L, C/4]

        indm = self.reduce(ind)
        indm = self.act(indm)

        ka = torch.tanh(self.ka(indm)).view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2) + 1.0
        kv = torch.tanh(self.kv(indm)).view(bs, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        temp = torch.tanh(self.temp(indm)).transpose(1, 2).unsqueeze(-1) + 1.0  # [B, H, L, 1]
        magk = torch.tanh(self.magk(indm)).transpose(1, 2).unsqueeze(-2) + 1.0  # [B, H, 1, L]

        attn_probs = self.softmax(attn_logits * temp) * magk  # [B, H, L, L]
        attn_probs = self.attn_drop(attn_probs)

        out = torch.matmul(attn_probs, v) * ka + v * kv  # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(bs, seqlen, self.embed_dim)  # [B, L, C]

        # Update struct_embed using the same CoFormer rule as MultiHeadAttention
        a = self.outer_squeeze(out)  # [B, L, c_squeeze]
        outer_product = torch.einsum("...bc,...de->...bdce", a, a)  # [B, L, L, c_squeeze, c_squeeze]
        struct_output = struct_embed + self.outer_linear(outer_product.reshape(list(outer_product.shape[:-2]) + [-1]))

        out = self.out_proj(out)
        struct_output = self.struct_out_proj(struct_output)
        return out, struct_output, attn_probs

