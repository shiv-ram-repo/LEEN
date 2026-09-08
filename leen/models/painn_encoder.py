"""
PaiNN encoder for LEEN.

Polarizable Atom Interaction Neural Network (Schütt et al., ICML 2021).
Maintains scalar features s and equivariant vector features v, updating
both through message passing with directional information.

Differences from GVP:
  - PaiNN uses radial basis functions for distance encoding
  - Separate message and update blocks (vs combined in GVP)
  - Different gating mechanism

Same interface as EGNN/GVP — drop-in replacement in LEENv2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _rbf_embed(dist: torch.Tensor, num_rbf: int = 20, cutoff: float = 10.0) -> torch.Tensor:
    """Sinc-based radial basis function embedding."""
    n = torch.arange(1, num_rbf + 1, device=dist.device, dtype=dist.dtype)
    # dist: [E], n: [num_rbf] → need [E, num_rbf]
    return torch.sin(n.unsqueeze(0) * torch.pi * dist.unsqueeze(-1) / cutoff) / dist.unsqueeze(-1).clamp(min=1e-6)


def _cosine_cutoff(dist: torch.Tensor, cutoff: float = 10.0) -> torch.Tensor:
    """Smooth cutoff function that goes to zero at the boundary."""
    return 0.5 * (torch.cos(torch.pi * dist / cutoff) + 1.0) * (dist <= cutoff).float()


class PaiNNMessage(nn.Module):
    """PaiNN message block: computes scalar and vector messages."""

    def __init__(self, node_dim: int, num_rbf: int = 20, esm_proj_dim: int = 0):
        super().__init__()

        self.node_dim = node_dim

        # Scalar message
        self.scalar_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )

        # RBF → filter
        self.rbf_proj = nn.Linear(num_rbf, 3 * node_dim)
        self.num_rbf = num_rbf

        # ESM gating (optional)
        self.has_esm = esm_proj_dim > 0
        if self.has_esm:
            self.esm_gate = nn.Sequential(
                nn.Linear(2 * esm_proj_dim, node_dim),
                nn.SiLU(),
                nn.Linear(node_dim, 3 * node_dim),
                nn.Sigmoid(),
            )

    def forward(self, s, v, edge_index, rel_pos, dist, esm_h=None):
        """
        s: [N, F] scalar features
        v: [N, F, 3] vector features
        edge_index: [2, E]
        rel_pos: [E, 3] relative positions (src - dst)
        dist: [E, 1] distances
        esm_h: [N, esm_proj_dim] or None
        """
        src, dst = edge_index
        F = self.node_dim

        # Source scalar features → 3F channels
        phi = self.scalar_mlp(s[src])  # [E, 3F]

        # Distance filter
        rbf = _rbf_embed(dist.squeeze(-1), self.num_rbf)  # [E, num_rbf]
        W = self.rbf_proj(rbf)  # [E, 3F]
        cutoff = _cosine_cutoff(dist.squeeze(-1)).unsqueeze(-1)  # [E, 1]

        # Element-wise product
        filt = phi * W * cutoff  # [E, 3F]

        # ESM gating
        if self.has_esm and esm_h is not None:
            gate = self.esm_gate(torch.cat([esm_h[src], esm_h[dst]], dim=-1))
            filt = filt * gate

        # Split into 3 channels
        filt_s, filt_v1, filt_v2 = filt.split(F, dim=-1)

        # Scalar messages
        ds = torch.zeros_like(s)
        ds.scatter_add_(0, dst.unsqueeze(-1).expand_as(filt_s), filt_s)

        # Vector messages
        # Component 1: scale existing source vectors
        v_src = v[src]  # [E, F, 3]
        msg_v1 = v_src * filt_v1.unsqueeze(-1)  # [E, F, 3]

        # Component 2: project along relative direction
        rel_dir = rel_pos / dist.clamp(min=1e-6)  # [E, 3]
        msg_v2 = filt_v2.unsqueeze(-1) * rel_dir.unsqueeze(-2)  # [E, F, 3]

        msg_v = msg_v1 + msg_v2

        dv = torch.zeros_like(v)
        dst_exp = dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg_v)
        dv.scatter_add_(0, dst_exp, msg_v)

        return ds, dv


class PaiNNUpdate(nn.Module):
    """PaiNN update block: updates scalar and vector features using
    invariant information extracted from the vector channel."""

    def __init__(self, node_dim: int):
        super().__init__()
        self.U = nn.Linear(node_dim, node_dim, bias=False)
        self.V = nn.Linear(node_dim, node_dim, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )

    def forward(self, s, v):
        """
        s: [N, F]
        v: [N, F, 3]
        """
        Uv = self.U(v.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]
        Vv = self.V(v.transpose(-1, -2)).transpose(-1, -2)  # [N, F, 3]

        # Invariant: inner product of transformed vectors
        Vv_norm = torch.sum(Vv ** 2, dim=-1).clamp(min=1e-8).sqrt()  # [N, F]

        mlp_in = torch.cat([s, Vv_norm], dim=-1)  # [N, 2F]
        mlp_out = self.mlp(mlp_in)  # [N, 3F]

        F = s.size(-1)
        a_ss, a_sv, a_vv = mlp_out.split(F, dim=-1)

        ds = a_ss + a_sv * torch.sum(Uv * Vv, dim=-1)  # [N, F]
        dv = a_vv.unsqueeze(-1) * Uv  # [N, F, 3]

        return ds, dv


class PaiNNEncoder(nn.Module):
    """PaiNN encoder stack for LEEN.

    Maintains scalar (invariant) and vector (equivariant) features.
    Same interface as EGNN/GVP — drop-in replacement.
    """

    def __init__(
        self,
        input_dim: int,
        node_dim: int = 128,
        edge_dim: int = 16,
        num_layers: int = 6,
        num_rbf: int = 20,
        dropout: float = 0.1,
        esm_proj_dim: int = 0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, node_dim)

        self.message_blocks = nn.ModuleList()
        self.update_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.message_blocks.append(
                PaiNNMessage(node_dim, num_rbf, esm_proj_dim)
            )
            self.update_blocks.append(PaiNNUpdate(node_dim))
            self.norms.append(nn.LayerNorm(node_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        esm_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        rel_pos = coords[src] - coords[dst]
        dist = rel_pos.norm(dim=-1, keepdim=True)

        s = self.input_proj(node_features)
        v = torch.zeros(s.size(0), s.size(1), 3, device=s.device)

        for msg, upd, norm in zip(self.message_blocks, self.update_blocks, self.norms):
            ds, dv = msg(s, v, edge_index, rel_pos, dist, esm_h)
            s = s + ds
            v = v + dv

            ds2, dv2 = upd(s, v)
            s = norm(s + ds2)
            v = v + dv2

            s = self.dropout(s)

        return s, coords
