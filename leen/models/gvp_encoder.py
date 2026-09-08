"""
GVP-GNN encoder for LEEN.

Geometric Vector Perceptrons (Jing et al., ICLR 2021) process
(scalar, vector) feature tuples. The vector features are SE(3)-equivariant
and capture directional information: hydrogen bond geometry, sidechain
orientation, steric clash directionality.

EGNN only passes scalar messages + coordinate updates.
GVP passes scalar AND vector messages, giving it strictly more
representational capacity for structural reasoning.

Same interface as the EGNN encoder — drop-in replacement in LEENv2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_no_nan(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """L2 norm with gradient-safe zero handling."""
    return torch.clamp(torch.sum(x ** 2, dim=dim, keepdim=keepdim), min=1e-8).sqrt()


class GVPLinear(nn.Module):
    """Core GVP operation: maps (scalar, vector) → (scalar, vector).

    Scalar path: s_out = activation(W_s * [s_in, ||v_in||])
    Vector path: v_out = W_v * v_in * gate(s_in, ||v_in||)

    The vector norms provide invariant geometric information to the scalar
    path, while the scalar path gates the vector outputs.
    """

    def __init__(
        self,
        s_in: int, s_out: int,
        v_in: int, v_out: int,
        activations: bool = True,
    ):
        super().__init__()
        self.s_out = s_out
        self.v_out = v_out

        # Scalar: takes s_in + v_in (vector norms) → s_out
        self.W_s = nn.Linear(s_in + v_in, s_out)

        if v_out > 0:
            # Vector: v_in → v_out (rotation-equivariant linear)
            self.W_v = nn.Linear(v_in, v_out, bias=False)
            # Gate: scalar signal controls vector magnitude
            self.W_gate = nn.Linear(s_in + v_in, v_out)

        self.act = nn.SiLU() if activations else nn.Identity()

    def forward(
        self, s: torch.Tensor, v: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        s: [*, s_in] scalar features
        v: [*, v_in, 3] vector features (or None)

        Returns: (s_out, v_out)
        """
        if v is not None:
            v_norm = _norm_no_nan(v, dim=-1)  # [*, v_in]
            s_cat = torch.cat([s, v_norm], dim=-1)
        else:
            s_cat = s

        s_out = self.act(self.W_s(s_cat))

        if self.v_out > 0 and v is not None:
            v_out = self.W_v(v.transpose(-1, -2)).transpose(-1, -2)  # [*, v_out, 3]
            gate = torch.sigmoid(self.W_gate(s_cat)).unsqueeze(-1)   # [*, v_out, 1]
            v_out = v_out * gate
        else:
            v_out = None

        return s_out, v_out


class GVPMessageLayer(nn.Module):
    """GVP message passing layer with optional ESM gating.

    Messages carry both scalar and vector information between residues.
    ESM gating modulates scalar message strength based on evolutionary
    conservation of the source-target pair.
    """

    def __init__(
        self,
        s_dim: int, v_dim: int,
        edge_s_dim: int,
        s_hidden: int = 256,
        esm_proj_dim: int = 0,
    ):
        super().__init__()

        # Edge message: (s_src, s_dst, edge_feat, ||rel_pos||) → message
        edge_in = 2 * s_dim + edge_s_dim + 1  # +1 for distance
        self.edge_mlp = GVPLinear(edge_in, s_hidden, 0, v_dim)

        # Vector message from relative positions
        self.vec_proj = nn.Linear(1, v_dim, bias=False)

        # Node update
        self.node_update = GVPLinear(s_dim + s_hidden, s_dim, v_dim + v_dim, v_dim)
        self.norm_s = nn.LayerNorm(s_dim)

        # ESM gating (optional)
        self.has_esm_gate = esm_proj_dim > 0
        if self.has_esm_gate:
            self.esm_gate = nn.Sequential(
                nn.Linear(2 * esm_proj_dim, s_hidden),
                nn.SiLU(),
                nn.Linear(s_hidden, s_hidden),
                nn.Sigmoid(),
            )

    def forward(
        self,
        s: torch.Tensor,       # [N, s_dim]
        v: torch.Tensor,       # [N, v_dim, 3]
        x: torch.Tensor,       # [N, 3]
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        esm_h: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        src, dst = edge_index
        rel_pos = x[src] - x[dst]                         # [E, 3]
        dist = _norm_no_nan(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        rel_dir = rel_pos / dist.clamp(min=1e-6)            # [E, 3] unit vector

        # Scalar edge features
        edge_in = [s[src], s[dst], dist]
        if edge_attr is not None:
            edge_in.append(edge_attr)
        edge_s = torch.cat(edge_in, dim=-1)

        # Scalar messages
        msg_s, _ = self.edge_mlp(edge_s, None)  # [E, s_hidden]

        # ESM gating on scalar messages
        if self.has_esm_gate and esm_h is not None:
            gate = self.esm_gate(torch.cat([esm_h[src], esm_h[dst]], dim=-1))
            msg_s = msg_s * gate

        # Vector messages from relative directions
        msg_v = self.vec_proj(dist).unsqueeze(-1) * rel_dir.unsqueeze(-2)  # [E, v_dim, 3]

        # Aggregate
        agg_s = torch.zeros(s.size(0), msg_s.size(-1), device=s.device)
        agg_s.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg_s), msg_s)

        agg_v = torch.zeros(v.size(0), msg_v.size(1), 3, device=v.device)
        dst_exp = dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg_v)
        agg_v.scatter_add_(0, dst_exp, msg_v)

        # Update
        s_in = torch.cat([s, agg_s], dim=-1)
        v_in = torch.cat([v, agg_v], dim=-2)
        s_new, v_new = self.node_update(s_in, v_in)

        # Residual
        s_out = self.norm_s(s + s_new)
        v_out = v + v_new

        return s_out, v_out


class GVPEncoder(nn.Module):
    """GVP-GNN encoder stack for LEEN.

    Replaces EGNN with a more expressive encoder that processes
    both scalar and vector (directional) features.

    Interface matches EGNN encoder — drop-in replacement.
    """

    def __init__(
        self,
        input_dim: int,
        node_dim: int = 128,
        edge_dim: int = 16,
        v_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        esm_proj_dim: int = 0,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, node_dim)
        self.v_init = nn.Linear(node_dim, v_dim * 3)
        self.v_dim = v_dim

        self.layers = nn.ModuleList([
            GVPMessageLayer(
                s_dim=node_dim,
                v_dim=v_dim,
                edge_s_dim=edge_dim,
                s_hidden=hidden_dim,
                esm_proj_dim=esm_proj_dim,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        esm_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (h [N, node_dim], x [N, 3])
        """
        s = self.input_proj(node_features)  # [N, node_dim]
        v = self.v_init(s).view(s.size(0), self.v_dim, 3)  # [N, v_dim, 3]

        for layer in self.layers:
            s, v = layer(s, v, coords, edge_index, edge_attr, esm_h)
            s = self.dropout(s)

        return s, coords  # coords unchanged (invariant encoder)
