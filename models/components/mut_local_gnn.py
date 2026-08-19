import torch
import torch.nn as nn


def _get_cb_positions_batch(pos_atoms, mask_atoms):
    """Cβ per residue; fall back to Cα when Cβ is missing."""
    pos_ca = pos_atoms[:, :, 1]
    if pos_atoms.shape[2] < 5:
        return pos_ca
    pos_cb = pos_atoms[:, :, 4]
    mask_cb = mask_atoms[:, :, 4].unsqueeze(-1)
    return torch.where(mask_cb, pos_cb, pos_ca)


def _build_knn_indices(pos, valid, k):
    """k-NN within each graph; invalid nodes are excluded from neighbor sets."""
    b, n, _ = pos.shape
    dist = torch.cdist(pos, pos)
    invalid = ~valid
    dist = dist.masked_fill(invalid.unsqueeze(1), 1e6)
    dist = dist.masked_fill(invalid.unsqueeze(2), 1e6)
    k_eff = min(k, n)
    _, knn_idx = dist.topk(k_eff, dim=-1, largest=False)
    return knn_idx


class MutLocalGNNLayer(nn.Module):
    def __init__(self, dim, edge_dim=32):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(4, edge_dim),
            nn.ReLU(),
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim + edge_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.update_mlp[-1].weight)
        nn.init.zeros_(self.update_mlp[-1].bias)

    def forward(self, h, pos, knn_idx, valid):
        b, n, d = h.shape
        batch_idx = torch.arange(b, device=h.device)[:, None, None]
        h_j = h[batch_idx, knn_idx]
        pos_j = pos[batch_idx, knn_idx]
        rel = pos_j - pos.unsqueeze(2)
        dist = rel.norm(dim=-1, keepdim=True)
        edge_feat = self.edge_mlp(torch.cat([rel, dist], dim=-1))
        msg = self.msg_mlp(torch.cat([h_j, edge_feat], dim=-1))

        neighbor_valid = torch.gather(valid, 1, knn_idx.reshape(b, -1)).reshape_as(knn_idx)
        denom = neighbor_valid.sum(dim=2, keepdim=True).clamp(min=1).float()
        agg = (msg * neighbor_valid.unsqueeze(-1).float()).sum(dim=2) / denom
        delta = self.update_mlp(torch.cat([h, agg], dim=-1))
        return h + delta


class MutLocalGNN(nn.Module):
    """Lightweight k-NN message passing on a fixed mutation-centered residue window."""

    def __init__(self, dim, k=8, num_layers=2, edge_dim=32):
        super().__init__()
        self.k = int(k)
        self.layers = nn.ModuleList([
            MutLocalGNNLayer(dim, edge_dim=edge_dim) for _ in range(int(num_layers))
        ])

    def forward(self, h, pos, valid):
        """
        Args:
            h: [B, N, D] node features (projected wt-mt diff)
            pos: [B, N, 3] Cβ/Cα coordinates
            valid: [B, N] bool, True for real residues in the window
        """
        knn_idx = _build_knn_indices(pos, valid, self.k)
        for layer in self.layers:
            h = layer(h, pos, knn_idx, valid)
        return h


def extract_local_mutation_window(
    diff,
    padding_mask,
    mut_id,
    window,
    pos_cb=None,
):
    """Extract a fixed-length window around the mutation site."""
    b, l, d = diff.shape
    win_len = 2 * window + 1
    local = diff.new_zeros(b, win_len, d)
    local_valid = torch.zeros(b, win_len, device=diff.device, dtype=torch.bool)
    local_pos = None
    if pos_cb is not None:
        local_pos = pos_cb.new_zeros(b, win_len, 3)

    for batch_idx in range(b):
        valid = ~padding_mask[batch_idx]
        sites = torch.nonzero(mut_id[batch_idx].bool() & valid, as_tuple=False).flatten()
        if len(sites) == 0:
            center = int(diff[batch_idx].abs().sum(dim=-1).argmax())
        else:
            center = int(sites[0])
        for out_j, src_j in enumerate(range(center - window, center + window + 1)):
            if 0 <= src_j < l and valid[src_j]:
                local[batch_idx, out_j] = diff[batch_idx, src_j]
                local_valid[batch_idx, out_j] = True
                if local_pos is not None:
                    local_pos[batch_idx, out_j] = pos_cb[batch_idx, src_j]
    return local, local_valid, local_pos
