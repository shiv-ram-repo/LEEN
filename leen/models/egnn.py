"""
E(n) Equivariant Graph Neural Network (EGNN).

Based on: Satorras et al., "E(n) Equivariant Graph Neural Networks" (ICML 2021).

Pure PyTorch implementation — no torch-geometric dependency.
Operates on local protein structure graphs where:
  - Nodes = residues (features + Cα coordinates)
  - Edges = spatial contacts within a distance cutoff

Key property: node features are SE(3)-invariant, coordinate updates are
SE(3)-equivariant. This means the energy prediction does not depend on
how the protein is oriented in space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    """Single E(n)-equivariant message-passing layer.

    Updates node features h and (optionally) coordinates x such that:
      - h transforms are invariant to rotations/translations
      - x transforms are equivariant to rotations/translations

    Messages are computed from pairs of node features plus the squared
    inter-node distance (an invariant scalar), ensuring equivariance.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 256,
        update_coords: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.update_coords = update_coords
        self.residual = residual

        # ---------- edge / message MLP ----------
        # Input: [h_i, h_j, ||x_i - x_j||^2, edge_attr]
        edge_input_dim = 2 * node_dim + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # ---------- node update MLP ----------
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # ---------- coordinate update MLP ----------
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            # Small init so early coord updates don't explode
            nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)

        # ---------- optional attention gate on messages ----------
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node features, shape [N, node_dim].
            x: Node coordinates, shape [N, 3].
            edge_index: [2, E] with edge_index[0] = source, edge_index[1] = target.
            edge_attr: Optional edge features, shape [E, edge_dim].

        Returns:
            h_out: Updated node features [N, node_dim].
            x_out: Updated coordinates [N, 3].
        """
        src, dst = edge_index  # each [E]

        # --- Compute invariant distance features ---
        rel_pos = x[src] - x[dst]  # [E, 3]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # --- Build edge inputs and compute messages ---
        edge_input = [h[src], h[dst], dist_sq]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)

        messages = self.edge_mlp(edge_input)  # [E, hidden_dim]

        # Attention gating
        att = self.att_mlp(messages)  # [E, 1]
        messages = messages * att

        # --- Aggregate messages at destination nodes ---
        agg = torch.zeros(h.size(0), messages.size(-1), device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # --- Update node features ---
        h_out = self.node_mlp(torch.cat([h, agg], dim=-1))
        if self.residual:
            h_out = h + h_out

        # --- Update coordinates (equivariant) ---
        if self.update_coords:
            coord_weights = self.coord_mlp(messages)  # [E, 1]
            weighted_pos = rel_pos * coord_weights  # [E, 3]
            coord_shift = torch.zeros_like(x)
            coord_shift.scatter_add_(
                0, dst.unsqueeze(-1).expand_as(weighted_pos), weighted_pos
            )
            x_out = x + coord_shift
        else:
            x_out = x

        return h_out, x_out


class EGNN(nn.Module):
    """Stack of EGNN layers with optional layer norm.

    This is the encoder backbone: it takes a local protein graph and
    produces SE(3)-invariant node features that encode the structural
    context of each residue.
    """

    def __init__(
        self,
        input_dim: int,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 6,
        update_coords: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project raw node features to working dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, node_dim),
            nn.SiLU(),
        )

        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    hidden_dim=hidden_dim,
                    update_coords=update_coords,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(node_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: Raw residue features [N, input_dim].
            coords: Cα coordinates [N, 3].
            edge_index: [2, E].
            edge_attr: [E, edge_dim] or None.

        Returns:
            h: Encoded node features [N, node_dim].
            x: (Possibly updated) coordinates [N, 3].
        """
        h = self.input_proj(node_features)
        x = coords

        for layer, norm in zip(self.layers, self.norms):
            h, x = layer(h, x, edge_index, edge_attr)
            h = norm(h)
            h = self.dropout(h)

        return h, x
