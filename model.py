"""
Zone-Aware Heterogeneous GNN (ZAH-GNN)
=======================================
Theo spec tuần 1 — Thành viên B

Kiến trúc:
  1. GPS-initialized Node Embedding     (Section 3.3)
  2. Sinusoidal Time Encoding + MLP     (Section 3.2)
  3. Time-Conditioned Zone Embedding    (Section 3.1)
  4. Adaptive Adjacency Matrix          (Eq.6)
  5. Node-Specific GCN                  (Eq.7,8)
  6. Temporal GRU
  7. Node-level Congestion Decoder → [B, N, T_out]

Interface chuẩn theo spec:
    forward(X, Z=None, time_idx=None, A_static=None)
    X        : [B, N, in_channels]   in_channels = T_in * F
    Z        : [N, K]                zone labels (binary, K=6 poi types)
    time_idx : [B]                   time label index (0-3)
    A_static : [N, N]                normalized OSRM/physical adjacency
    → output : [B, N, T_out]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# 1. GPS-INITIALIZED NODE EMBEDDING (Section 3.3)
#    Khởi tạo từ tọa độ GPS thay vì random
# ─────────────────────────────────────────────
class GPSNodeEmbedding(nn.Module):
    def __init__(self, n_nodes, embed_dim, gps_coords=None):
        """
        gps_coords: [N, 2] tensor (lat, lon) — nếu None thì random
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projection từ GPS (2D) → embed_dim
        self.gps_proj = nn.Linear(2, embed_dim, bias=False)

        if gps_coords is not None:
            # Normalize GPS coords về [-1, 1]
            coords = torch.FloatTensor(gps_coords)
            coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-8)
            # Khởi tạo embedding = GPS projection (prior vật lý)
            with torch.no_grad():
                init_embed = self.gps_proj(coords)
            self.node_embed = nn.Parameter(init_embed)
        else:
            self.node_embed = nn.Parameter(torch.randn(n_nodes, embed_dim))

    def forward(self):
        return self.node_embed  # [N, embed_dim]


# ─────────────────────────────────────────────
# 2. SINUSOIDAL TIME ENCODING (Section 3.2, Eq.3)
#    Mã hoá thời gian liên tục thay vì 4 nhãn rời rạc
# ─────────────────────────────────────────────
class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, d_time=16, out_dim=32):
        """
        d_time : chiều của sinusoidal encoding
        out_dim: chiều output sau MLP (= embed_dim)
        """
        super().__init__()
        self.d_time = d_time
        # MLP nhỏ chiếu time encoding → weight matrix
        self.mlp = nn.Sequential(
            nn.Linear(d_time * 2, out_dim),  # *2 vì sin + cos
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def encode(self, time_idx):
        """
        time_idx: [B] — index 0-3 (night/normal/rush_m/rush_e)
        Trả về: [B, d_time*2] sinusoidal encoding
        """
        device = time_idx.device
        B = time_idx.shape[0]
        # Map 4 label → giờ đại diện: night=2, normal=11, rush_m=8, rush_e=17
        hour_map = torch.tensor([2.0, 11.0, 8.0, 17.0], device=device)
        hour = hour_map[time_idx.long()]  # [B]

        # Sinusoidal encoding (Eq.3)
        i = torch.arange(self.d_time, device=device).float()
        div = torch.pow(10000, 2 * i / self.d_time)  # [d_time]
        sin_enc = torch.sin(hour.unsqueeze(1) / div)  # [B, d_time]
        cos_enc = torch.cos(hour.unsqueeze(1) / div)  # [B, d_time]
        return torch.cat([sin_enc, cos_enc], dim=-1)  # [B, d_time*2]

    def forward(self, time_idx):
        enc = self.encode(time_idx)  # [B, d_time*2]
        return self.mlp(enc)  # [B, out_dim]


# ─────────────────────────────────────────────
# 3. TIME-CONDITIONED ZONE EMBEDDING (Section 3.1, Eq.2)
#    z_tilde(v,t) = g_t ⊙ MLP(z(v))
# ─────────────────────────────────────────────
class TimeConditionedZoneEmbedding(nn.Module):
    def __init__(self, n_zone_types, embed_dim, time_dim=32):
        """
        n_zone_types: K = số loại POI (6 trong data hiện tại)
        embed_dim   : chiều embedding output
        time_dim    : chiều time encoding
        """
        super().__init__()
        # MLP(z(v)): zone one-hot → embedding
        self.zone_mlp = nn.Sequential(
            nn.Linear(n_zone_types, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Gating: time → gate vector g_t ∈ R^{embed_dim}
        self.gate_proj = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.Sigmoid(),  # gate ∈ [0,1]
        )

    def forward(self, Z, time_enc):
        """
        Z        : [N, K]   zone one-hot labels
        time_enc : [B, time_dim]  sinusoidal time encoding
        → [B, N, embed_dim]
        """
        z_embed = self.zone_mlp(Z)  # [N, embed_dim]
        gate = self.gate_proj(time_enc)  # [B, embed_dim]

        # Broadcast: z_embed [N, D] × gate [B, D] → [B, N, D]
        z_tilde = gate.unsqueeze(1) * z_embed.unsqueeze(0)  # [B, N, D]
        return z_tilde


# ─────────────────────────────────────────────
# 4. ADAPTIVE ADJACENCY (Eq.6)
#    A_tilde = Softmax(ReLU(E · E^T))
# ─────────────────────────────────────────────
class AdaptiveAdjacency(nn.Module):
    def __init__(self, node_embedding_module):
        super().__init__()
        self.node_embed_module = node_embedding_module

    def forward(self, A_static=None):
        E = self.node_embed_module()  # [N, embed_dim]
        A_adaptive = F.softmax(F.relu(torch.mm(E, E.T)), dim=1)
        if A_static is not None:
            # Kết hợp adaptive + static
            A = 0.5 * A_adaptive + 0.5 * A_static
        else:
            A = A_adaptive
        return A  # [N, N]


# ─────────────────────────────────────────────
# 5. NODE-SPECIFIC GCN (Eq.7,8)
# ─────────────────────────────────────────────
class NodeSpecificGCN(nn.Module):
    def __init__(self, n_nodes, in_dim, out_dim, embed_dim):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_gen = nn.Linear(embed_dim, in_dim * out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_nodes, out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H, A, node_embed):
        """
        H         : [B, N, in_dim]
        A         : [N, N]
        node_embed: [N, embed_dim]
        """
        N = H.shape[1]
        W = self.weight_gen(node_embed)  # [N, in_dim*out_dim]
        W = W.view(N, self.in_dim, self.out_dim)  # [N, in_dim, out_dim]
        H_agg = torch.einsum("nm,bmd->bnd", A, H)  # [B, N, in_dim]
        out = torch.einsum("bni,nio->bno", H_agg, W)  # [B, N, out_dim]
        out = out + self.bias.unsqueeze(0)
        return self.norm(F.relu(out))


# ─────────────────────────────────────────────
# 6. SPATIO-TEMPORAL BLOCK
# ─────────────────────────────────────────────
class SpatioTemporalBlock(nn.Module):
    def __init__(self, n_nodes, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.gcn1 = NodeSpecificGCN(n_nodes, in_dim, hidden_dim, embed_dim)
        self.gcn2 = NodeSpecificGCN(n_nodes, hidden_dim, hidden_dim, embed_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes

    def forward(self, X_seq, A, node_embed):
        """
        X_seq : [B, T, N, in_dim]
        → out : [B, T, N, hidden_dim]
        """
        B, T, N, _ = X_seq.shape
        outs = [
            self.gcn2(self.gcn1(X_seq[:, t], A, node_embed), A, node_embed)
            for t in range(T)
        ]
        H = torch.stack(outs, dim=1)  # [B, T, N, hid]
        H_r = H.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        H_g, _ = self.gru(H_r)  # [B*N, T, hid]
        return H_g.reshape(B, N, T, self.hidden_dim).permute(0, 2, 1, 3)


# ─────────────────────────────────────────────
# 7. ZONE-AWARE HETEROGENEOUS GNN (Full Model)
# ─────────────────────────────────────────────
class ZAHGNNModel(nn.Module):
    """
    Interface chuẩn theo spec:
        forward(X, Z=None, time_idx=None, A_static=None)
        → [B, N, T_out]
    """

    def __init__(
        self,
        num_nodes,
        in_channels,  # T_in * F
        out_channels,  # T_out = 3
        hidden_dim=32,
        embed_dim=16,
        n_zone_types=6,
        time_d=16,
        n_layers=2,
        gps_coords=None,  # [N, 2] numpy array
        T_in=12,
        F_feat=4,
        **kwargs
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.T_in = T_in
        self.F_feat = F_feat

        # 1. GPS Node Embedding
        self.node_embed_module = GPSNodeEmbedding(num_nodes, embed_dim, gps_coords)

        # 2. Sinusoidal Time Encoding
        self.time_enc = SinusoidalTimeEncoding(d_time=time_d, out_dim=embed_dim)

        # 3. Time-Conditioned Zone Embedding
        self.zone_embed = TimeConditionedZoneEmbedding(
            n_zone_types, embed_dim, embed_dim
        )

        # 4. Adaptive Adjacency
        self.adaptive_adj = AdaptiveAdjacency(self.node_embed_module)

        # 5. Input projection: in_channels → hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # 6. Fusion: node_embed + zone_embed → hidden_dim
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

        # 7. ST Blocks
        self.st_blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(num_nodes, hidden_dim, hidden_dim, embed_dim)
                for _ in range(n_layers)
            ]
        )

        # 8. Node-level decoder → [B, N, T_out]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, X, Z=None, time_idx=None, A_static=None):
        """
        X        : [B, N, in_channels]   (in_channels = T_in * F)
        Z        : [N, K]                zone one-hot
        time_idx : [B]                   time label 0-3
        A_static : [N, N]
        → output : [B, N, T_out]
        """
        B, N, C = X.shape
        device = X.device

        # -- Node embedding (GPS-initialized)
        node_embed = self.node_embed_module()  # [N, embed_dim]

        # -- Time encoding
        if time_idx is None:
            time_idx = torch.ones(B, dtype=torch.long, device=device)
        t_enc = self.time_enc(time_idx)  # [B, embed_dim]

        # -- Zone embedding (time-conditioned)
        if Z is None:
            Z = torch.zeros(N, 6, device=device)
        z_tilde = self.zone_embed(Z, t_enc)  # [B, N, embed_dim]

        # -- Fuse node_embed + zone_embed
        node_exp = node_embed.unsqueeze(0).expand(B, -1, -1)  # [B, N, embed_dim]
        fused = self.fusion(torch.cat([node_exp, z_tilde], dim=-1))  # [B, N, embed_dim]

        # -- Adaptive adjacency
        A = self.adaptive_adj(A_static)  # [N, N]

        # -- Project input + reshape to [B, T_in, N, F]
        H = self.input_proj(X)  # [B, N, hidden_dim]
        # Expand thành sequence giả [B, 1, N, hid] nếu không có T
        H = H.unsqueeze(1)  # [B, 1, N, hid]

        # -- ST Blocks (dùng fused embedding thay vì raw node_embed)
        # Truyền fused[0] (share across batch là OK vì embed không phụ thuộc sample)
        node_embed_fused = fused.mean(0)  # [N, embed_dim] — mean across batch
        for block in self.st_blocks:
            H = block(H, A, node_embed_fused)  # [B, 1, N, hid]

        # -- Decode: [B, N, T_out]
        H_last = H[:, -1, :, :]  # [B, N, hid]
        out = self.decoder(H_last)  # [B, N, T_out]
        return out

    def get_adaptive_adj(self, A_static=None):
        return self.adaptive_adj(A_static).detach().cpu().numpy()


# ─────────────────────────────────────────────
# BACKWARD COMPAT: giữ tên AHGNN cho train_model.py cũ
# ─────────────────────────────────────────────
AHGNN = ZAHGNNModel
