"""
LEEN v2: ESM-Guided Local Equivariant Energy Network.

Key innovation over v1: instead of concatenating ESM features to nodes,
ESM embeddings GATE the message passing in the EGNN.

    conserved residue pair → strong message (rigid structural constraint)
    variable residue pair  → weak message (flexible, less constrained)

This means:
  - Geometry branch processes structure (EGNN)
  - Evolution branch controls information flow (gating)
  - They interact at every layer, not just at the output

Architecture:
    ┌──────────────────────┐    ┌──────────────────────┐
    │  Geometry (13-dim)   │    │  ESM2 (1280-dim)     │
    │  + AA embedding      │    │  → project to 128    │
    │        ↓             │    │        ↓              │
    │  EGNN node features  │◄───│  Gate on messages    │
    │  (updated each layer)│    │  (frozen ESM signal) │
    └──────────┬───────────┘    └──────────────────────┘
               ↓
    Cross-attention: structure queries evolution
               ↓
    Energy head → E(aa, env) for all 20 AAs
               ↓
    ΔΔG = E(mut) − E(wt)    [exact antisymmetry]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Data container (same as v1, with optional ESM features)
# ──────────────────────────────────────────────────────────────

@dataclass
class LocalGraphV2:
    """Local graph with separate geometry and ESM feature fields."""

    # Geometry
    node_features: torch.Tensor   # [N, geo_dim]  (13-dim structural features)
    coords: torch.Tensor          # [N, 3]
    edge_index: torch.Tensor      # [2, E]
    edge_attr: torch.Tensor | None

    # Amino acid identity
    aa_indices: torch.Tensor      # [N]  (0–19, or 20 for unknown)

    # ESM (optional — model works without it, just no gating)
    esm_features: torch.Tensor | None  # [N, 1280] frozen ESM2 per-residue

    # Mutation info
    mut_pos: torch.Tensor         # [B]
    wt_aa: torch.Tensor           # [B]
    mut_aa: torch.Tensor          # [B]
    ddg: torch.Tensor | None      # [B]
    batch: torch.Tensor           # [N]

    def to(self, device):
        fields = {}
        for k, v in self.__dict__.items():
            fields[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return LocalGraphV2(**fields)


# ──────────────────────────────────────────────────────────────
# ESM-Gated EGNN Layer
# ──────────────────────────────────────────────────────────────

class ESMGatedEGNNLayer(nn.Module):
    """EGNN layer where ESM embeddings gate the structural messages.

    Standard EGNN computes messages from pairs of node features + distance.
    This layer additionally modulates each message by a gate derived from
    the ESM representations of the source and target residues.

    Biological intuition:
        conserved pair → gate ≈ 1 → strong structural message
        variable pair  → gate ≈ 0 → weak structural message
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 256,
        esm_proj_dim: int = 128,
    ):
        super().__init__()

        # ── Structural message MLP ──
        edge_input_dim = 2 * node_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # ── ESM-guided gate ──
        # Input: projected ESM of source + target → gate per message dim
        self.esm_gate = nn.Sequential(
            nn.Linear(2 * esm_proj_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # ── Node update ──
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # ── Coordinate update ──
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

    def forward(
        self,
        h: torch.Tensor,         # [N, node_dim]
        x: torch.Tensor,         # [N, 3]
        edge_index: torch.Tensor, # [2, E]
        edge_attr: torch.Tensor | None,
        esm_h: torch.Tensor | None,  # [N, esm_proj_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        src, dst = edge_index

        # ── Structural messages ──
        rel_pos = x[src] - x[dst]
        dist_sq = (rel_pos ** 2).sum(-1, keepdim=True)

        edge_in = [h[src], h[dst], dist_sq]
        if edge_attr is not None:
            edge_in.append(edge_attr)
        messages = self.edge_mlp(torch.cat(edge_in, dim=-1))

        # ── ESM gating ──
        if esm_h is not None:
            esm_pair = torch.cat([esm_h[src], esm_h[dst]], dim=-1)
            gate = self.esm_gate(esm_pair)
            messages = messages * gate

        # ── Aggregate at target nodes ──
        agg = torch.zeros(h.size(0), messages.size(-1), device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # ── Node update (residual) ──
        h_out = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # ── Coordinate update ──
        coord_w = self.coord_mlp(messages)
        weighted = rel_pos * coord_w
        shift = torch.zeros_like(x)
        shift.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)
        x_out = x + shift

        return h_out, x_out


# ──────────────────────────────────────────────────────────────
# Cross-Attention: structure queries evolution
# ──────────────────────────────────────────────────────────────

class StructureEvolutionCrossAttention(nn.Module):
    """Cross-attention where the structural representation at the mutation
    site queries ESM representations of ALL local residues.

    This captures: "which evolutionary signals in my neighborhood are
    most relevant to predicting energy at this position?"
    """

    def __init__(self, struct_dim: int, esm_proj_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = struct_dim // num_heads
        assert struct_dim % num_heads == 0

        self.q_proj = nn.Linear(struct_dim, struct_dim)
        self.k_proj = nn.Linear(esm_proj_dim, struct_dim)
        self.v_proj = nn.Linear(esm_proj_dim, struct_dim)
        self.out_proj = nn.Linear(struct_dim, struct_dim)
        self.norm = nn.LayerNorm(struct_dim)

    def forward(
        self,
        h_mut: torch.Tensor,     # [B, struct_dim]  mutation site repr
        esm_h: torch.Tensor,     # [N_total, esm_proj_dim]
        batch: torch.Tensor,     # [N_total]  maps nodes to samples
        mut_pos: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        B = h_mut.size(0)
        device = h_mut.device

        # Build per-sample padded ESM context
        # Find max local graph size
        counts = torch.bincount(batch, minlength=B)
        max_nodes = counts.max().item()

        # Pad ESM features into [B, max_nodes, esm_proj_dim]
        esm_padded = torch.zeros(B, max_nodes, esm_h.size(-1), device=device)
        mask = torch.ones(B, max_nodes, dtype=torch.bool, device=device)

        for i in range(B):
            node_mask = (batch == i)
            nodes = esm_h[node_mask]
            n = nodes.size(0)
            esm_padded[i, :n] = nodes
            mask[i, :n] = False  # False = attend, True = ignore

        # Cross-attention
        Q = self.q_proj(h_mut).unsqueeze(1)        # [B, 1, d]
        K = self.k_proj(esm_padded)                 # [B, N, d]
        V = self.v_proj(esm_padded)                 # [B, N, d]

        # Multi-head reshape
        d = self.head_dim
        nh = self.num_heads
        Q = Q.view(B, 1, nh, d).transpose(1, 2)    # [B, nh, 1, d]
        K = K.view(B, max_nodes, nh, d).transpose(1, 2)  # [B, nh, N, d]
        V = V.view(B, max_nodes, nh, d).transpose(1, 2)

        # Scaled dot product with mask
        attn = (Q @ K.transpose(-2, -1)) / (d ** 0.5)  # [B, nh, 1, N]
        attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # handle all-masked rows

        out = (attn @ V).transpose(1, 2).reshape(B, -1)  # [B, struct_dim]
        out = self.out_proj(out)

        return self.norm(h_mut + out)


# ──────────────────────────────────────────────────────────────
# Full LEEN v2 Model
# ──────────────────────────────────────────────────────────────

class LEENv2(nn.Module):
    """LEEN v2: ESM-gated equivariant energy network.

    Two branches:
        1. Geometry → EGNN (structural message passing)
        2. ESM → projection (evolutionary gating signal)

    ESM modulates EGNN messages at every layer.
    After encoding, cross-attention lets the mutation site attend
    to evolutionary context of all neighbors.
    Energy-difference readout gives exact antisymmetry.
    """

    NUM_AA = 20

    def __init__(
        self,
        geo_dim: int = 13,
        esm_dim: int = 1280,
        esm_proj_dim: int = 128,
        aa_embed_dim: int = 64,
        node_dim: int = 128,
        edge_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        energy_hidden_dim: int = 256,
        use_cross_attention: bool = True,
        symmetrize_site: bool = False,
        head_type: str = "energy",
        cross_attn_heads: int = 4,
        encoder_type: str = "egnn",  # "egnn" | "gvp"
    ):
        super().__init__()
        self.geo_dim = geo_dim
        self.esm_dim = esm_dim
        self.use_cross_attention = use_cross_attention
        self.symmetrize_site = symmetrize_site
        self.head_type = head_type
        self.encoder_type = encoder_type

        # ── Amino acid embedding ──
        self.aa_embedding = nn.Embedding(self.NUM_AA + 1, aa_embed_dim)

        # ── Geometry branch: project to encoder working dim ──
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim + aa_embed_dim, node_dim),
            nn.SiLU(),
        )

        # ── ESM branch: project 1280 → compact ──
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, esm_proj_dim),
            nn.SiLU(),
            nn.LayerNorm(esm_proj_dim),
        )

        # ── Encoder (swappable) ──
        if encoder_type == "egnn":
            self.layers = nn.ModuleList([
                ESMGatedEGNNLayer(
                    node_dim=node_dim, edge_dim=edge_dim,
                    hidden_dim=hidden_dim, esm_proj_dim=esm_proj_dim,
                )
                for _ in range(num_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
            self.drop = nn.Dropout(dropout)
            self.encoder = None  # use layer-by-layer
        elif encoder_type == "gvp":
            from leen.models.gvp_encoder import GVPEncoder
            self.encoder = GVPEncoder(
                input_dim=node_dim, node_dim=node_dim, edge_dim=edge_dim,
                hidden_dim=hidden_dim, num_layers=num_layers,
                dropout=dropout, esm_proj_dim=esm_proj_dim,
            )
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")
        self.norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # ── Cross-attention: structure queries evolution ──
        if use_cross_attention:
            self.cross_attn = StructureEvolutionCrossAttention(
                struct_dim=node_dim,
                esm_proj_dim=esm_proj_dim,
                num_heads=cross_attn_heads,
            )

        # ── Energy head ──
        self.energy_head = nn.Sequential(
            nn.Linear(node_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(energy_hidden_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(energy_hidden_dim, self.NUM_AA),
        )
        self._init_energy_head()
        self.scalar_head = nn.Sequential(
            nn.Linear(node_dim + 2 * aa_embed_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(energy_hidden_dim, energy_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(energy_hidden_dim, 1),
        )

    def _init_energy_head(self):
        for m in self.energy_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, graph: LocalGraphV2) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the local graph, return (h_mut, esm_h).

        h_mut: [B, node_dim] — structural repr at mutation site.
        esm_h: [N, esm_proj_dim] — projected ESM for cross-attn.
        """
        # Geometry branch
        aa_emb = self.aa_embedding(graph.aa_indices)
        
        if getattr(self, 'symmetrize_site', False):
            # exact anti-symmetry WITH site identity: site AA emb = ½(emb[wt]+emb[mut])
            wt_emb = self.aa_embedding(graph.wt_aa)
            mut_emb = self.aa_embedding(graph.mut_aa)
            aa_emb = aa_emb.clone()
            aa_emb[graph.mut_pos] = 0.5 * (wt_emb + mut_emb)
            
        h = self.geo_proj(torch.cat([graph.node_features, aa_emb], dim=-1))
        x = graph.coords

        # ESM branch
        if graph.esm_features is not None:
            esm_h = self.esm_proj(graph.esm_features)
        else:
            esm_h = None

        # ESM-gated encoder
        if self.encoder is not None:
            # GVP: full encoder stack
            h, x = self.encoder(h, x, graph.edge_index, graph.edge_attr, esm_h)
        else:
            # EGNN: layer-by-layer
            for layer, norm in zip(self.layers, self.norms):
                h, x = layer(h, x, graph.edge_index, graph.edge_attr, esm_h)
                h = norm(h)
                h = self.drop(h)

        # Gather mutation-site representation
        h_mut = h[graph.mut_pos]  # [B, node_dim]

        # Cross-attention: structure queries evolution
        if self.use_cross_attention and esm_h is not None:
            h_mut = self.cross_attn(h_mut, esm_h, graph.batch, graph.mut_pos)

        return h_mut, esm_h

    def forward(self, graph: LocalGraphV2) -> dict[str, torch.Tensor]:
        h_mut, esm_h = self.encode(graph)

        if self.head_type == "scalar":
            wt_e = self.aa_embedding(graph.wt_aa)
            mut_e = self.aa_embedding(graph.mut_aa)
            ddg_pred = self.scalar_head(torch.cat([h_mut, wt_e, mut_e], dim=-1)).squeeze(-1)
            return {"ddg_pred": ddg_pred, "energy": None, "h_mut": h_mut}

        energy = self.energy_head(h_mut)
        e_wt = energy.gather(1, graph.wt_aa.unsqueeze(1)).squeeze(1)
        e_mut = energy.gather(1, graph.mut_aa.unsqueeze(1)).squeeze(1)
        ddg_pred = e_mut - e_wt

        return {"ddg_pred": ddg_pred, "energy": energy, "h_mut": h_mut}

    def predict_with_perturbation(
        self, graph: LocalGraphV2, sigma: float = 0.2
    ) -> dict[str, torch.Tensor]:
        h_mut, esm_h = self.encode(graph)

        if self.head_type == "scalar":
            wt_e = self.aa_embedding(graph.wt_aa); mut_e = self.aa_embedding(graph.mut_aa)
            xc = torch.cat([h_mut, wt_e, mut_e], dim=-1)
            xp = torch.cat([h_mut + torch.randn_like(h_mut) * sigma, wt_e, mut_e], dim=-1)
            return {"ddg_pred": self.scalar_head(xc).squeeze(-1),
                    "ddg_perturbed": self.scalar_head(xp).squeeze(-1),
                    "energy": None, "h_mut": h_mut}

        energy_clean = self.energy_head(h_mut)
        energy_pert = self.energy_head(h_mut + torch.randn_like(h_mut) * sigma)

        e_wt = energy_clean.gather(1, graph.wt_aa.unsqueeze(1)).squeeze(1)
        e_mut = energy_clean.gather(1, graph.mut_aa.unsqueeze(1)).squeeze(1)

        e_wt_p = energy_pert.gather(1, graph.wt_aa.unsqueeze(1)).squeeze(1)
        e_mut_p = energy_pert.gather(1, graph.mut_aa.unsqueeze(1)).squeeze(1)

        return {
            "ddg_pred": e_mut - e_wt,
            "ddg_perturbed": e_mut_p - e_wt_p,
            "energy": energy_clean,
            "h_mut": h_mut,
        }
