"""
AH-GNN: Adaptive Heterogeneous Graph Neural Network
====================================================
Kiến trúc:
  1. AdaptiveAdjacency  — học A_tilde động (Eq.6 paper)
  2. NodeSpecificGCN    — weight riêng từng nút (Eq.7,8 paper)
  3. TemporalGRU        — capture chuỗi thời gian
  4. ETADecoder         — predict travel time cho từng edge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# 1. ADAPTIVE ADJACENCY (Eq.6 trong paper)
#    A_tilde = Softmax(ReLU(E · E^T))
# ─────────────────────────────────────────────
class AdaptiveAdjacency(nn.Module):
    def __init__(self, n_nodes, embed_dim=16):
        super().__init__()
        # Node embedding E ∈ R^{N × d}
        self.node_embed = nn.Parameter(torch.randn(n_nodes, embed_dim))

    def forward(self):
        # A_tilde = Softmax(ReLU(E · E^T))
        A = torch.mm(self.node_embed, self.node_embed.T)
        A = F.relu(A)
        A = F.softmax(A, dim=1)
        return A  # [N, N]


# ─────────────────────────────────────────────
# 2. NODE-SPECIFIC GCN (Eq.7,8 trong paper)
#    W_v = Θ · e_v  (weight sinh từ node embedding)
#    h_v = σ(Σ A_vu · h_u · W_v)
# ─────────────────────────────────────────────
class NodeSpecificGCN(nn.Module):
    def __init__(self, n_nodes, in_dim, out_dim, embed_dim=16):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Θ: ánh xạ từ embedding → weight matrix (Eq.7)
        self.weight_gen = nn.Linear(embed_dim, in_dim * out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_nodes, out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H, A, node_embed):
        """
        H          : [B, N, in_dim]
        A          : [N, N]  adaptive adjacency
        node_embed : [N, embed_dim]
        """
        B, N, _ = H.shape

        # Sinh weight riêng cho từng nút (Eq.7): W_v = Θ · e_v
        # W_v shape: [N, in_dim, out_dim]
        W = self.weight_gen(node_embed)  # [N, in_dim*out_dim]
        W = W.view(N, self.in_dim, self.out_dim)  # [N, in_dim, out_dim]

        # Aggregate láng giềng: H_agg = A · H  (Eq.8)
        # A: [N,N], H: [B,N,in_dim] → H_agg: [B,N,in_dim]
        H_agg = torch.einsum("nm,bmd->bnd", A, H)

        # Apply node-specific weight: h_v = H_agg_v · W_v
        # H_agg: [B,N,in_dim], W: [N,in_dim,out_dim]
        out = torch.einsum("bni,nio->bno", H_agg, W)  # [B,N,out_dim]
        out = out + self.bias.unsqueeze(0)  # bias
        out = self.norm(F.relu(out))
        return out  # [B, N, out_dim]


# ─────────────────────────────────────────────
# 3. SPATIO-TEMPORAL BLOCK
#    NodeSpecificGCN → GRU → NodeSpecificGCN
# ─────────────────────────────────────────────
class SpatioTemporalBlock(nn.Module):
    def __init__(self, n_nodes, in_dim, hidden_dim, embed_dim=16):
        super().__init__()
        self.gcn1 = NodeSpecificGCN(n_nodes, in_dim, hidden_dim, embed_dim)
        self.gcn2 = NodeSpecificGCN(n_nodes, hidden_dim, hidden_dim, embed_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

    def forward(self, X_seq, A, node_embed):
        """
        X_seq : [B, T, N, F]
        A     : [N, N]
        → out : [B, T, N, hidden_dim]
        """
        B, T, N, F = X_seq.shape
        outs = []

        for t in range(T):
            h = self.gcn1(X_seq[:, t, :, :], A, node_embed)  # [B, N, hid]
            h = self.gcn2(h, A, node_embed)  # [B, N, hid]
            outs.append(h)

        # Stack → [B, T, N, hid]
        H = torch.stack(outs, dim=1)

        # GRU theo chiều thời gian cho từng nút
        # Reshape: [B*N, T, hid]
        H_reshaped = H.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        H_gru, _ = self.gru(H_reshaped)  # [B*N, T, hid]
        H_out = H_gru.reshape(B, N, T, self.hidden_dim)
        H_out = H_out.permute(0, 2, 1, 3)  # [B, T, N, hid]
        return H_out


# ─────────────────────────────────────────────
# 4. ETA DECODER
#    Aggregate 2 nút (src, dst) → predict travel time
# ─────────────────────────────────────────────
class ETADecoder(nn.Module):
    def __init__(self, hidden_dim, n_edges, edge_feat_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_edges)
        self.act = nn.ReLU()

    def forward(self, H_last, edge_index, edge_feats):
        """
        H_last    : [B, N, hidden_dim]  - node embeddings tại T cuối
        edge_index: [E, 2]              - (src_idx, dst_idx)
        edge_feats: [B, E, 3]           - congestion, delay, length
        → out     : [B, E]
        """
        src_idx = edge_index[:, 0]  # [E]
        dst_idx = edge_index[:, 1]  # [E]

        # Lấy embedding của src và dst cho từng edge
        h_src = H_last[:, src_idx, :]  # [B, E, hid]
        h_dst = H_last[:, dst_idx, :]  # [B, E, hid]

        # Concat: [B, E, 2*hid + edge_feat]
        x = torch.cat([h_src, h_dst, edge_feats], dim=-1)
        x = self.act(self.fc1(x))  # [B, E, hid]

        # Predict travel time cho từng edge
        out = self.fc2(x).squeeze(-1)  # [B, E]
        # Nếu fc2 out dim = 1
        if out.dim() == 3:
            out = out.squeeze(-1)
        return F.relu(out)  # travel time >= 0


# ─────────────────────────────────────────────
# 5. AH-GNN FULL MODEL
# ─────────────────────────────────────────────
class AHGNN(nn.Module):
    def __init__(
        self,
        n_nodes,
        n_edges,
        in_dim=6,
        hidden_dim=32,
        embed_dim=16,
        edge_feat_dim=3,
        n_layers=2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

        # Adaptive adjacency module
        self.adaptive_adj = AdaptiveAdjacency(n_nodes, embed_dim)

        # Shared node embedding (dùng cho cả adj lẫn weight gen)
        self.node_embed = self.adaptive_adj.node_embed

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # ST blocks
        self.st_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(n_nodes, hidden_dim, hidden_dim, embed_dim)
                for _ in range(n_layers)
            ]
        )

        # ETA decoder
        self.decoder = ETADecoder(hidden_dim, 1, edge_feat_dim)

    def forward(self, X_seq, edge_index, edge_feats):
        """
        X_seq     : [B, T, N, F]
        edge_index: [E, 2]
        edge_feats: [B, E, 3]
        → Y_hat   : [B, E]
        """
        B, T, N, F = X_seq.shape

        # Học adaptive adjacency
        A = self.adaptive_adj()  # [N, N]

        # Project input
        X = self.input_proj(X_seq.reshape(B * T, N, F))
        X = X.reshape(B, T, N, self.hidden_dim)

        # Qua các ST blocks
        H = X
        for block in self.st_blocks:
            H = block(H, A, self.node_embed)  # [B, T, N, hid]

        # Lấy timestep cuối cùng
        H_last = H[:, -1, :, :]  # [B, N, hid]

        # Decode → ETA
        Y_hat = self.decoder(H_last, edge_index, edge_feats)  # [B, E]
        return Y_hat

    def get_adaptive_adj(self):
        """Lấy learned adjacency để visualize trong paper"""
        return self.adaptive_adj().detach().cpu().numpy()
